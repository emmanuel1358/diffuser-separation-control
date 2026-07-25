from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lbx_rl_tasks_harness.message_text import message_text as _message_text
from lbx_rl_tasks_harness.models import HarnessProblem
from lbx_rl_tasks_harness.models_config import (
    default_model_from_env,
    has_key_for_model,
    required_env_var,
)
from lbx_rl_tasks_harness.llm import build_model as _build_model
from lbx_rl_tasks_harness.review_utils import (
    normalize_review_checks,
    read_limited as _read_limited,
    repo_root as _repo_root,
)

MAX_GUIDANCE_CHARS = 20000
MAX_SCORER_CHARS = 30000
MAX_PROMPT_CHARS = 12000

RUBRIC_QUALITY_CHECKS = [
    "Prompt alignment",
    "Measurability",
    "Logical independence",
    "No inessential complexity",
    "No unjustified single solution path",
    "Ambiguity and reference handling",
    "Weight reasonableness",
    "Expert task coverage",
    "Formatting not overweighted",
]


async def run_rubric_quality_check(
    problem: HarnessProblem,
    *,
    grade_payload: dict[str, Any],
    model_name: str | None = None,
) -> dict[str, Any]:
    """Ask a model to review scorer rubric quality against RUBRIC_GUIDANCE.md."""
    checked_at = datetime.now(UTC).isoformat()
    task_model = problem.metadata.get("runner", {}).get("api_model_name")
    effective_model = model_name or default_model_from_env(
        task_model if isinstance(task_model, str) else None
    )

    if not has_key_for_model(effective_model):
        return {
            "status": "skipped",
            "checked_at": checked_at,
            "model": effective_model,
            "reason": f"{required_env_var(effective_model)} is not set",
            "checks": [],
        }

    if problem.source_problem_dir is None:
        return {
            "status": "skipped",
            "checked_at": checked_at,
            "model": effective_model,
            "reason": "source_problem_dir is not available",
            "checks": [],
        }

    try:
        model = _build_model(effective_model)
        prompt = _quality_prompt(problem, grade_payload)
        response = await model.ainvoke(prompt)
        raw_text = _message_text(response)
        parsed = _parse_json_object(raw_text)
        parsed["status"] = "completed"
        parsed["checked_at"] = checked_at
        parsed["model"] = effective_model
        parsed["checks"] = _normalize_checks(parsed.get("checks"))
        parsed["summary"] = str(parsed.get("summary") or "")
        return parsed
    except Exception as exc:  # noqa: BLE001 - store diagnostics without hiding score.
        return {
            "status": "error",
            "checked_at": checked_at,
            "model": effective_model,
            "reason": f"{type(exc).__name__}: {exc}",
            "checks": [],
        }


def _quality_prompt(problem: HarnessProblem, grade_payload: dict[str, Any]) -> str:
    problem_dir = problem.source_problem_dir
    assert problem_dir is not None
    repo_root = _repo_root(problem_dir)
    guidance = _read_limited(
        repo_root / "docs" / "RUBRIC_GUIDANCE.md", MAX_GUIDANCE_CHARS
    )
    instruction = _read_limited(problem_dir / "instruction.md", MAX_PROMPT_CHARS)
    task_toml = _read_limited(problem_dir / "task.toml", MAX_PROMPT_CHARS)
    scorer = _read_limited(
        problem_dir / "scorer" / "compute_score.py", MAX_SCORER_CHARS
    )

    return f"""You are reviewing rubric criteria quality for one task.

Your job: given the TASK PROMPT/INSTRUCTIONS, the SCORER RUBRIC CRITERIA implemented in compute_score.py, and the STANDARD RUBRIC GUIDANCE, decide whether the rubric criteria adhere to the guidance.

Focus on the rubric criteria themselves. Do not grade the agent submission, reference solution, or latest score. Use the latest harness grade only as context for what criteria exist and how they are weighted.

Return ONLY valid JSON with this shape:
{{
  "summary": "one short paragraph",
  "checks": [
    {{
      "check": "one of: {", ".join(RUBRIC_QUALITY_CHECKS)}",
      "status": "pass|warning|fail",
      "reason": "specific evidence from prompt/scorer",
      "recommendation": "actionable fix, or empty string if pass"
    }}
  ]
}}

Include exactly one object for each named check. Each reason must cite concrete evidence from the task prompt/instructions or scorer rubric criteria.

STANDARD RUBRIC GUIDANCE:
{guidance}

TASK PROMPT / INSTRUCTIONS:
{instruction}

TASK TOML:
{task_toml}

SCORER RUBRIC CRITERIA / compute_score.py:
{scorer}

LATEST HARNESS GRADE PAYLOAD (context only; do not grade the agent submission):
{json.dumps(grade_payload, indent=2, default=str)[:20000]}
"""


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("rubric quality response must be a JSON object")
    return data


def _normalize_checks(raw_checks: Any) -> list[dict[str, str]]:
    return normalize_review_checks(
        raw_checks,
        RUBRIC_QUALITY_CHECKS,
        missing_recommendation="Rerun rubric quality review.",
    )
