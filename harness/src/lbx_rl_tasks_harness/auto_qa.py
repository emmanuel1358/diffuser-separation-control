from __future__ import annotations

import asyncio
import json
import os
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from lbx_rl_tasks_harness.message_text import message_text as _message_text
from lbx_rl_tasks_harness.models import HarnessProblem
from lbx_rl_tasks_harness.models_config import (
    has_key_for_model,
    provider_for_model,
    required_env_var,
    strip_provider_prefix,
)
from lbx_rl_tasks_harness.llm import build_model as _build_model
from lbx_rl_tasks_harness.review_utils import (
    normalize_review_checks,
    read_limited as _read_limited,
    repo_root as _repo_root,
)
from lbx_rl_tasks_harness.rubric_quality import _parse_json_object

AUTOQA_MODEL_ENV = "LBX_RL_AUTOQA_MODEL"
AUTOQA_REASONING_EFFORT_ENV = "LBX_RL_AUTOQA_REASONING_EFFORT"
DEFAULT_AUTOQA_MODEL = "openai:gpt-5.5"
DEFAULT_AUTOQA_REASONING_EFFORT = "xhigh"

# Reasoning models occasionally return a blank/non-JSON completion (a transient
# provider hiccup, truncation, or a soft safety refusal). Retry a few times with
# a short backoff before giving up — most empties clear on a retry.
AUTOQA_MAX_EMPTY_RETRIES = 2
AUTOQA_RETRY_BACKOFF_SEC = 2.0
# Keep a bounded slice of the raw model output on failure so the PR feedback is
# diagnosable ("unexplained CI failure") instead of a bare traceback.
AUTOQA_RAW_EXCERPT_CHARS = 2000

MAX_DOC_CHARS = 20000
MAX_TASK_FILE_CHARS = 30000
MAX_DATA_FILE_CHARS = 8000
MAX_BUNDLE_CHARS = 140000
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {".alignerr", ".git", "__pycache__"}

AUTOQA_CHECKS = [
    "Prompt clarity",
    "Output contract",
    "MuJoCo physics validity",
    "Reference solution",
    "Baseline and difficulty calibration",
    "Scorer determinism",
    "Rubric coverage",
    "Reward-hacking resistance",
    "Hidden data and fixtures",
    "Harness proof consistency",
]


async def run_auto_qa_check(
    problem: HarnessProblem,
    *,
    proof: dict[str, Any],
    grade_payload: dict[str, Any],
    rubric_quality: dict[str, Any],
    model_name: str | None = None,
    reasoning_effort: str | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Review a submitted task package without rerunning scoring or rubric checks."""
    checked_at = datetime.now(UTC).isoformat()
    effective_model = model_name or default_autoqa_model_from_env()
    effective_effort = reasoning_effort or default_autoqa_reasoning_effort_from_env()

    if not has_key_for_model(effective_model):
        return {
            "status": "skipped",
            "checked_at": checked_at,
            "model": effective_model,
            "reasoning_effort": effective_effort,
            "reason": f"{required_env_var(effective_model)} is not set",
            "checks": [],
        }

    if problem.source_problem_dir is None:
        return {
            "status": "skipped",
            "checked_at": checked_at,
            "model": effective_model,
            "reasoning_effort": effective_effort,
            "reason": "source_problem_dir is not available",
            "checks": [],
        }

    raw_text = ""
    try:
        model = _build_autoqa_model(effective_model, reasoning_effort=effective_effort)
        task_type = _task_type(problem.source_problem_dir)
        expected_checks = _autoqa_checks(task_type)
        prompt = _autoqa_prompt(problem, proof, grade_payload, rubric_quality)
        raw_text = await _invoke_model_with_retry(
            model, prompt, stream_callback=stream_callback
        )
    except Exception as exc:  # noqa: BLE001 - persist diagnostics for PR feedback.
        # Model build / invoke failure (import, network, provider 5xx). Advisory.
        return _autoqa_error_result(
            checked_at,
            effective_model,
            effective_effort,
            reason=f"{type(exc).__name__}: {exc}",
            raw_text=raw_text,
        )

    if not raw_text.strip():
        # Blank completion even after retries: a transient hiccup, truncation, or
        # a safety refusal. Surfaced as an explainable, non-blocking (advisory)
        # infra error rather than a raw JSONDecodeError.
        return _autoqa_error_result(
            checked_at,
            effective_model,
            effective_effort,
            reason=(
                f"Auto QA model returned empty output after "
                f"{AUTOQA_MAX_EMPTY_RETRIES + 1} attempts "
                f"(model={effective_model}, effort={effective_effort}); "
                f"treated as advisory (non-blocking)."
            ),
            raw_text=raw_text,
        )

    try:
        parsed = _parse_json_object(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        return _autoqa_error_result(
            checked_at,
            effective_model,
            effective_effort,
            reason=(
                f"Auto QA model returned non-parseable output "
                f"(model={effective_model}, effort={effective_effort}): "
                f"{type(exc).__name__}: {exc}"
            ),
            raw_text=raw_text,
        )

    parsed["status"] = "completed"
    parsed["checked_at"] = checked_at
    parsed["model"] = effective_model
    parsed["reasoning_effort"] = effective_effort
    return _normalize_autoqa(parsed, expected_checks)


def _autoqa_error_result(
    checked_at: str,
    model: str,
    reasoning_effort: str,
    *,
    reason: str,
    raw_text: str,
) -> dict[str, Any]:
    """Build a status="error" result, attaching a truncated raw-output excerpt.

    status="error" is an INFRA failure (model/parse), which the CLI treats as
    advisory (non-blocking) — see runner.run_auto_qa_only. The excerpt makes the
    failure diagnosable in PR feedback instead of an "unexplained" traceback.
    """
    result: dict[str, Any] = {
        "status": "error",
        "checked_at": checked_at,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "reason": reason,
        "checks": [],
    }
    excerpt = raw_text.strip()
    if excerpt:
        result["raw_output_excerpt"] = excerpt[:AUTOQA_RAW_EXCERPT_CHARS]
    return result


def default_autoqa_model_from_env() -> str:
    return os.environ.get(AUTOQA_MODEL_ENV, "").strip() or DEFAULT_AUTOQA_MODEL


def default_autoqa_reasoning_effort_from_env() -> str:
    return (
        os.environ.get(AUTOQA_REASONING_EFFORT_ENV, "").strip()
        or DEFAULT_AUTOQA_REASONING_EFFORT
    )


def _build_autoqa_model(model_name: str, *, reasoning_effort: str):
    provider = provider_for_model(model_name)
    concrete_name = strip_provider_prefix(model_name)
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Auto QA requires OpenAI model dependencies. Install with `uv sync`."
            ) from exc
        return ChatOpenAI(
            model=concrete_name,
            reasoning={"effort": reasoning_effort},
            max_tokens=None,
        )
    return _build_model(model_name)


async def _invoke_model_with_retry(
    model: Any,
    prompt: str,
    *,
    stream_callback: Callable[[str], None] | None,
    max_empty_retries: int = AUTOQA_MAX_EMPTY_RETRIES,
    backoff_sec: float = AUTOQA_RETRY_BACKOFF_SEC,
) -> str:
    """Invoke the model, retrying only when the completion is blank.

    Empty completions from reasoning models are usually transient (provider
    hiccup, truncation, soft refusal), so retry a few times with a short linear
    backoff. Returns the last completion (possibly still blank) so the caller can
    craft an explainable, non-blocking error instead of a raw JSONDecodeError.
    A non-empty completion returns immediately (no wasted model calls).
    """
    text = ""
    for attempt in range(max_empty_retries + 1):
        text = await _invoke_model(model, prompt, stream_callback=stream_callback)
        if text.strip():
            return text
        if attempt < max_empty_retries:
            await asyncio.sleep(backoff_sec * (attempt + 1))
    return text


async def _invoke_model(
    model: Any, prompt: str, *, stream_callback: Callable[[str], None] | None
) -> str:
    if stream_callback is not None and hasattr(model, "astream"):
        parts: list[str] = []
        async for chunk in model.astream(prompt):
            text = _message_text(chunk)
            if text:
                parts.append(text)
                stream_callback(text)
        return "".join(parts)

    response = await model.ainvoke(prompt)
    return _message_text(response)


def _autoqa_prompt(
    problem: HarnessProblem,
    proof: dict[str, Any],
    grade_payload: dict[str, Any],
    rubric_quality: dict[str, Any],
) -> str:
    problem_dir = problem.source_problem_dir
    assert problem_dir is not None
    repo_root = _repo_root(problem_dir)
    task_type = _task_type(problem_dir)
    guidance_bundle = {
        "RUBRIC_GUIDANCE.md": _read_limited(
            repo_root / "docs" / "RUBRIC_GUIDANCE.md", MAX_DOC_CHARS
        ),
        "GRADING.md": _read_limited(repo_root / "docs" / "GRADING.md", MAX_DOC_CHARS),
    }
    guidance_bundle.update(_task_type_guidance(repo_root, task_type))
    task_bundle = _task_bundle(problem_dir)
    standards = _task_type_standards(task_type)
    expected_checks = _autoqa_checks(task_type)
    reviewer_role = _reviewer_role(task_type)
    blocking_concerns = _blocking_concerns(task_type)
    solver_leak_guidance = _solver_leak_guidance(task_type)

    return f"""Formatting re-enabled
You are {reviewer_role}.

Goal: decide whether this task package is solvable, scientifically correct, and deterministic. Review the task prompt, output contract, reference solution, grader/scorer, rubric, hidden/public data manifest, baseline/reference calibration, submitted build proof, and latest harness/rubric-quality context.

Do not grade the latest model attempt. Review the task package itself. Ground every warning or failure in concrete evidence from the provided files. Do not require a single implementation path unless the prompt or physics requires it.

Do not critique Dockerfiles, base image tags, package pinning, dependency installation style, `task.toml` internet access, or container packaging/reproducibility concerns. Mention runtime setup only if the submitted proof shows the task cannot be evaluated at all.

Return ONLY valid JSON with this shape:
{{
  "overall_assessment": "pass|needs_changes|fail",
  "summary": "one short paragraph",
  "is_solvable": {{"value": true, "evidence": "specific evidence"}},
  "scientifically_correct": {{"value": true, "evidence": "specific evidence"}},
  "confidence": "low|medium|high",
  "blocking_issues": ["specific issue with file evidence"],
  "non_blocking_feedback": ["specific feedback with file evidence"],
  "checks": [
    {{
      "check": "one of: {", ".join(expected_checks)}",
      "status": "pass|warning|fail",
      "reason": "specific evidence from provided files",
      "recommendation": "actionable fix, or empty string if pass"
    }}
  ]
}}

Include exactly one object for each named check. Treat these as blocking concerns when severe: {blocking_concerns}.
{solver_leak_guidance}

Task type: {task_type or "unknown"}

Task standards to enforce:
{standards}

STANDARD GUIDANCE:
{json.dumps(guidance_bundle, indent=2, default=str)[:MAX_BUNDLE_CHARS]}

TASK PACKAGE:
{task_bundle}

SUBMITTED BUILD PROOF:
{json.dumps(proof, indent=2, default=str)[:30000]}

SUBMITTED HARNESS RESULT:
{json.dumps(grade_payload, indent=2, default=str)[:30000]}

SUBMITTED RUBRIC QUALITY REVIEW:
{json.dumps(rubric_quality, indent=2, default=str)[:30000]}
"""


def _task_type(problem_dir: Path) -> str:
    task_toml = problem_dir / "task.toml"
    try:
        data = tomllib.loads(task_toml.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    difficulty = data.get("difficulty")
    if isinstance(difficulty, dict):
        return str(difficulty.get("task_type") or "").strip().lower()
    return ""


def _autoqa_checks(task_type: str) -> list[str]:
    if task_type == "cfd":
        return _checks_with_physics_and_solver_leak("CFD physics validity")
    if task_type == "structures":
        return _checks_with_physics_and_solver_leak("Structural physics validity")
    return list(AUTOQA_CHECKS)


def _checks_with_physics_and_solver_leak(physics_check: str) -> list[str]:
    checks = [
        physics_check if check == "MuJoCo physics validity" else check
        for check in AUTOQA_CHECKS
    ]
    checks.append("Solver-agnostic surface leakage")
    return checks


def _reviewer_role(task_type: str) -> str:
    if task_type == "cfd":
        return "an expert CFD task QA reviewer for verifiable LLM evaluation tasks"
    if task_type == "structures":
        return (
            "an expert structural engineering task QA reviewer for verifiable "
            "LLM evaluation tasks"
        )
    return "an expert task QA reviewer for verifiable LLM evaluation tasks"


def _blocking_concerns(task_type: str) -> str:
    if task_type == "cfd":
        return (
            "impossible or contradictory prompt, wrong /tmp/output contract, "
            "invalid CFD physics, non-deterministic scorer, reward-hacking "
            "vulnerabilities, missing reference/baseline calibration, "
            "solver/tool/provenance leaks in instruction.md, or proof/rubric "
            "evidence that the task teaches the wrong behavior"
        )
    if task_type == "structures":
        return (
            "impossible or contradictory prompt, wrong /tmp/output contract, "
            "invalid structural physics, non-deterministic scorer, "
            "reward-hacking vulnerabilities, missing reference/baseline "
            "calibration, solver/tool/provenance leaks in instruction.md, or "
            "proof/rubric evidence that the task teaches the wrong behavior"
        )
    return (
        "impossible or contradictory prompt, wrong /tmp/output contract, "
        "invalid MuJoCo physics, non-deterministic scorer, reward-hacking "
        "vulnerabilities, missing reference/baseline calibration, or "
        "proof/rubric evidence that the task teaches the wrong behavior"
    )


def _solver_leak_guidance(task_type: str) -> str:
    if task_type not in {"cfd", "structures"}:
        return ""
    return (
        '\nFor the "Solver-agnostic surface leakage" check, inspect ONLY '
        "instruction.md. Fail or warn when instruction.md mentions a solver, "
        "names a solver/tool, mandates commands or APIs, asks for solver "
        "evidence/provenance artifacts, reveals solver/tool implementation details, or implies a "
        "public surrogate/probe. Physics-grounded scoring details are allowed "
        "when they avoid solver/tool names and implementation/provenance hints. "
        "Do not use public/preloaded data, task.toml output declarations, task "
        "hints, README text, hidden grader/oracle/private files, or proof "
        "artifacts as evidence for this specific check."
    )


def _task_type_guidance(repo_root: Path, task_type: str) -> dict[str, str]:
    if task_type == "cfd":
        return {
            "cfd_environments.md": _read_limited(
                repo_root / "project_guidelines" / "cfd" / "cfd_environments.md",
                MAX_DOC_CHARS,
            )
        }
    if task_type == "structures":
        return {
            "STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md": _read_limited(
                repo_root
                / "project_guidelines"
                / "strctural_engineering"
                / "STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md",
                MAX_DOC_CHARS,
            )
        }
    return {
        "mujoco_environments.md": _read_limited(
            repo_root / "project_guidelines" / "mujoco_environments.md",
            MAX_DOC_CHARS,
        )
    }


def _task_type_standards(task_type: str) -> str:
    common = [
        "- final artifacts must be under /tmp/output;",
        "- scoring must be deterministic and must not use an LLM judge;",
        "- reference/baseline calibration should leave headroom for meaningful task difficulty;",
    ]
    if task_type == "cfd":
        return "\n".join(
            common
            + [
                "- instruction.md must be solver-agnostic: it must not mention solvers, name a solver/tool, mandate specific commands, reveal solver/tool implementation details, or imply a surrogate/probe; physics-grounded scoring details are allowed;",
                "- solver implementation details may remain outside instruction.md: public data may include probe/model files, and solution/scorer/hidden oracle code may run the real solver; do NOT require instruction.md or the agent transcript to mention solver use, and do NOT require a solver-evidence artifact;",
                "- difficulty must come from the hidden physics and tolerances (objective, feasibility shell, cost cap, worst-case over a hidden sweep) so a guessed or closed-form design fails;",
            ]
        )
    if task_type == "structures":
        return "\n".join(
            common
            + [
                "- instruction.md must be solver-agnostic: it must not mention solvers, name a solver/tool, mandate specific commands, reveal solver/tool implementation details, or imply a surrogate/probe; physics-grounded scoring details are allowed;",
                "- solver implementation details may remain outside instruction.md: public data may include probe/model files, and solution/scorer/hidden oracle code may run the real structural analysis; do NOT require instruction.md or the agent transcript to mention solver use, and do NOT require a solver-evidence artifact;",
                "- difficulty must come from the hidden physics and tolerances (targets, feasibility limits, worst-case over hidden ground motions) so an equivalent-linear or guessed estimate fails;",
            ]
        )
    return "\n".join(
        common
        + [
            "- good MuJoCo rubrics use compilation, MjModel inspection, fixed-seed rollouts, static checks, and robustness perturbations;",
            "- strong tasks aim for dense criteria across structural, static, rollout, and robustness strata;",
            "- graders should reject NaNs, exploding energy, contact pathologies, vacuous structures, and degenerate solutions;",
        ]
    )


def _task_bundle(problem_dir: Path) -> str:
    sections: list[str] = []
    for rel in [
        "instruction.md",
        "task.toml",
        "metadata.json",
        "scorer/compute_score.py",
        "solution/solve.sh",
        "baselines/naive.sh",
        "README.md",
    ]:
        path = problem_dir / rel
        sections.append(f"## {rel}\n{_read_limited(path, MAX_TASK_FILE_CHARS)}")

    for rel in ["data", "scorer/data", "solution", "baselines"]:
        path = problem_dir / rel
        sections.append(f"## {rel} manifest\n{_directory_manifest(path)}")
        sections.extend(_small_text_files(path, rel))

    text = "\n\n".join(sections)
    if len(text) <= MAX_BUNDLE_CHARS:
        return text
    return text[:MAX_BUNDLE_CHARS] + "\n...[task bundle truncated]..."


def _directory_manifest(path: Path) -> str:
    if not path.exists():
        return f"[missing: {path.name}]"
    rows: list[str] = []
    for child in sorted(path.rglob("*")):
        rel = child.relative_to(path)
        if _skip_path(rel):
            continue
        if child.is_dir():
            continue
        rows.append(f"- {rel.as_posix()} ({child.stat().st_size} bytes)")
    return "\n".join(rows) if rows else "[empty]"


def _small_text_files(path: Path, label: str) -> list[str]:
    if not path.exists():
        return []
    sections: list[str] = []
    for child in sorted(path.rglob("*")):
        rel = child.relative_to(path)
        if (
            _skip_path(rel)
            or child.is_dir()
            or child.suffix.lower() not in TEXT_SUFFIXES
        ):
            continue
        if child.stat().st_size > MAX_DATA_FILE_CHARS:
            continue
        sections.append(
            f"## {label}/{rel.as_posix()}\n{_read_limited(child, MAX_DATA_FILE_CHARS)}"
        )
    return sections


def _skip_path(rel: Path) -> bool:
    return any(part in SKIP_PARTS for part in rel.parts) or _looks_secret(rel)


def _looks_secret(rel: Path) -> bool:
    name = rel.name.lower()
    parts = {part.lower() for part in rel.parts}
    if name == ".env.secrets.gen":
        return True
    if "config" in parts and name.endswith(".json"):
        return "sa-" in name or "lb-" in name or "secret" in name
    return False


def _normalize_autoqa(parsed: dict[str, Any], expected_checks: list[str]) -> dict[str, Any]:
    parsed["overall_assessment"] = _normalize_choice(
        parsed.get("overall_assessment"),
        {"pass", "needs_changes", "fail"},
        "needs_changes",
    )
    parsed["summary"] = str(parsed.get("summary") or parsed.get("reason") or "")
    parsed["confidence"] = _normalize_choice(
        parsed.get("confidence"), {"low", "medium", "high"}, "medium"
    )
    parsed["is_solvable"] = _normalize_bool_evidence(parsed.get("is_solvable"))
    parsed["scientifically_correct"] = _normalize_bool_evidence(
        parsed.get("scientifically_correct")
    )
    parsed["blocking_issues"] = _normalize_string_list(parsed.get("blocking_issues"))
    parsed["non_blocking_feedback"] = _normalize_string_list(
        parsed.get("non_blocking_feedback")
    )
    parsed["checks"] = _normalize_checks(parsed.get("checks"), expected_checks)
    return parsed


def _normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default


def _normalize_bool_evidence(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        raw = value.get("value")
        evidence = str(value.get("evidence") or "")
    else:
        raw = value
        evidence = ""
    if isinstance(raw, bool):
        normalized = raw
    else:
        normalized = str(raw).strip().lower() in {"true", "yes", "pass"}
    return {"value": normalized, "evidence": evidence}


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _normalize_checks(raw_checks: Any, expected_checks: list[str]) -> list[dict[str, str]]:
    return normalize_review_checks(
        raw_checks,
        expected_checks,
        missing_recommendation="Rerun Auto QA.",
    )
