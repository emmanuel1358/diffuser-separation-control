"""Export a task directory to a Boreal submission payload.

Two entry points:

  * :func:`build_job_payload`        single-problem job
  * :func:`build_batch_job_payload`  multi-problem job (one per attempt)

Both produce the dict that goes to Boreal's ``/jobs/submit`` endpoint, plus
:func:`export_taiga` writes the legacy ``problems-metadata.json`` shape that
the existing CLI consumes.

Design notes (vs worldsim / ML_Envs)::

  * **No rubric in the payload.** ``rubric: []`` and a single
    ``grading_strategy: [{type: mcp, weight: 1.0}]`` always. The grader is
    image-baked; per-criterion detail flows back via the in-image grader's
    ``Grade.metadata.structured_subscores`` field that the rubric MCP server
    packs into its reply (see ``runtime/rubric/server.py``). Worldsim's
    split-strategy machinery (split rubric + agentic_grader promotion +
    penalty-avoidance conversion) only existed because of the dead Anthropic
    proxy on long CUA episodes — irrelevant for TU-only RL tasks.

  * **No CU/CUA fields.** No ``interaction_mode``, no ``cua_*``, no dual
    TU/CU image setting. RL tasks are tool-use only.

  * **All submission knobs come from `task.toml [runner]`** (attempts,
    turn_limit, max_ctx, context_mode, timeouts, model, ...). Defaults are
    sensible — old task.toml files without ``[runner]`` keep working.

  * **Tiny ``test_file`` shim only.** Boreal's current rubric MCP runtime
    executes ``extra_fields.test_file`` in a subprocess, so we ship a stable
    shim that imports the image-baked scorer from ``/mcp_server/grader`` and
    leaves task-specific scoring logic in the image.

  * **Redacted ground-truth evidence.** If ``.alignerr/build_proof.json``
    already contains ``ground_truth_result``, the exporter surfaces a compact
    proof summary under ``extra_fields.ground_truth_evidence``. It never ships
    raw solution files, hidden fixtures, trajectories, or scorer metadata.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alignerr_plugin.base_image import resolve_base_flavor_for_resource
from alignerr_plugin.ground_truth import expected_ground_truth_score, sha256_file
from alignerr_plugin.runtime_notices import GPU_NOTICE, TPU_NOTICE
from alignerr_plugin.schemas import (
    ML_GRADING_TIMEOUT_SEC,
    ML_MAX_EPISODE_SEC,
    ML_SETUP_TIMEOUT_SEC,
    ML_TOOL_TIMEOUT_SEC,
    RunnerConfig,
    TaskToml,
)
from alignerr_plugin.solver_hints import task_type_solver_hint
from alignerr_plugin.taiga_resources import (
    RESOURCE_RANK,
    is_accelerator_resource,
    validate_required_resources,
)
from alignerr_plugin.utils import load_task_toml, read_prompt, task_id, write_json

# ── Image + resource selection ────────────────────────────────────

# Exec the prebuilt venv console script directly (matching the base image's
# Dockerfile CMD). `uv run` re-resolves/builds the editable `rubric` project on
# every cold container start — under production load that repeatedly blew the
# 120s MCP init timeout, losing whole rollouts before the agent started. The
# venv is already synced at image-build time, so a plain exec starts instantly.
STARTUP_COMMAND = "/opt/lbx-runtime/.venv/bin/rubric mcp"

# ML_Envs bases install the rubric MCP server --system (no venv), so `rubric` is a
# plain console script on PATH.
MLENVS_STARTUP_COMMAND = "rubric mcp"

# ML timeout pins (ML_SETUP/GRADING/TOOL_TIMEOUT_SEC, ML_MAX_EPISODE_SEC) are the
# hour-scale, non-author-controlled timeouts imported from schemas. For
# ``task_type == "ml"`` the exporter force-pins every timeout field to them
# regardless of what task.toml carries; grading in particular runs at Taiga's
# maximum (ML_GRADING_TIMEOUT_SEC). Non-ml task types keep their author-set /
# default RunnerTimeouts values.


def _effective_timeouts(task_toml: TaskToml) -> tuple[int, int, int, int | None]:
    """Return ``(setup, grading, tool, max_episode)`` seconds after the ml pin.

    Timeouts are Taiga/Boreal-only (Harbor ignores them). ``task_type == "ml"``
    ignores author timeouts entirely and runs at the hour-scale ``ML_*`` pins --
    a minutes-scale cap silently fails real ml setup/tool/grading, and ml grading
    must run at Taiga's maximum. Every other task type keeps its author-set /
    default ``RunnerTimeouts`` values, so the pin is strictly ml-only.
    """
    timeouts = task_toml.runner.timeouts
    if str(task_toml.difficulty.task_type) == "ml":
        return (
            ML_SETUP_TIMEOUT_SEC,
            ML_GRADING_TIMEOUT_SEC,
            ML_TOOL_TIMEOUT_SEC,
            ML_MAX_EPISODE_SEC,
        )
    return (
        timeouts.setup_sec,
        timeouts.grading_sec,
        timeouts.tool_sec,
        timeouts.max_episode_sec,
    )

# Default base tag. `base/build_and_push.sh` computes the drift-hashed tag and
# the deploy/export sets LBX_RL_TASKS_BASE_IMAGE_TAG to it; this committed
# literal is the fallback. Keep this a plain ``os.environ.get(..., "literal")``
# assignment: the mothership sync workflow parses and re-pins it per repo, so the
# format must not change.
BASE_IMAGE_TAG = os.environ.get("LBX_RL_TASKS_BASE_IMAGE_TAG", "runtime-ml-core-py313-1b75cb075439")
# The cpu flavor can be pinned independently of the shared gpu/cpu tag; the
# mothership sync re-pins this assignment per repo (same format contract as
# BASE_IMAGE_TAG above).
CPU_BASE_IMAGE_TAG = os.environ.get("LBX_RL_TASKS_CPU_BASE_IMAGE_TAG", BASE_IMAGE_TAG)

CPU_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base"
GPU_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-gpu"
GPU_OPENROAD_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-gpu-openroad"
GPU_BLACKWELL_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-gpu-blackwell"
CUDA_GRAPHICS_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-cuda-graphics"
TPU_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-tpu"
# mlenvs-specific registry images (not shared with the native flavors).
MLENVS_SLIM_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-mlenvs-slim"
MLENVS_GPU_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-mlenvs-gpu"
MLENVS_CUDA_GRAPHICS_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-mlenvs-cuda-graphics"
MLENVS_TPU_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-mlenvs-tpu"
# Blackwell overlays (local-only; not deployable on Taiga).
MLENVS_GPU_BLACKWELL_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-mlenvs-gpu-blackwell"
MLENVS_CUDA_GRAPHICS_BLACKWELL_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-mlenvs-cuda-graphics-blackwell"


def _tpu_base_image_tag() -> str:
    # tpu uses a distinct tag prefix (py3.12). Set via env at deploy
    # (build_and_push.sh publishes the drift-hashed tpu tag); the committed
    # fallback mirrors the cpu/gpu registry-style pin (not a "-local" dev tag).
    return os.environ.get("LBX_RL_TASKS_TPU_BASE_IMAGE_TAG", "runtime-ml-tpu-py312-1b75cb075439")


def _mlenvs_slim_base_image_tag() -> str:
    return os.environ.get("LBX_RL_TASKS_MLENVS_SLIM_BASE_IMAGE_TAG", "runtime-mlenvs-slim-py312-1b75cb075439")


def _mlenvs_base_image_tag() -> str:
    # mlenvs bases share a py3.12 tag prefix; the image suffix disambiguates.
    return os.environ.get("LBX_RL_TASKS_MLENVS_BASE_IMAGE_TAG", "runtime-mlenvs-py312-1b75cb075439")


def _mlenvs_blackwell_base_image_tag() -> str:
    # cu128 / sm_120 overlays (local-only).
    return os.environ.get(
        "LBX_RL_TASKS_MLENVS_BLACKWELL_BASE_IMAGE_TAG",
        "runtime-mlenvs-blackwell-py312-1b75cb075439",
    )


def _blackwell_base_image_tag() -> str:
    return os.environ.get(
        "LBX_RL_TASKS_BLACKWELL_BASE_IMAGE_TAG",
        "runtime-ml-blackwell-py313-1b75cb075439",
    )


def _base_image_and_tag(flavor: str) -> tuple[str, str]:
    """Map a resolved base flavor to its (registry image, drift-hashed tag)."""
    if flavor == "gpu":
        return GPU_BASE_IMAGE, BASE_IMAGE_TAG
    if flavor == "gpu-openroad":
        return GPU_OPENROAD_BASE_IMAGE, BASE_IMAGE_TAG
    if flavor == "gpu-blackwell":
        return GPU_BLACKWELL_BASE_IMAGE, _blackwell_base_image_tag()
    if flavor == "cuda-graphics":
        return CUDA_GRAPHICS_BASE_IMAGE, BASE_IMAGE_TAG
    if flavor == "tpu":
        return TPU_BASE_IMAGE, _tpu_base_image_tag()
    if flavor == "mlenvs-slim":
        return MLENVS_SLIM_BASE_IMAGE, _mlenvs_slim_base_image_tag()
    if flavor == "mlenvs-gpu":
        return MLENVS_GPU_BASE_IMAGE, _mlenvs_base_image_tag()
    if flavor == "mlenvs-cuda-graphics":
        return MLENVS_CUDA_GRAPHICS_BASE_IMAGE, _mlenvs_base_image_tag()
    if flavor == "mlenvs-tpu":
        return MLENVS_TPU_BASE_IMAGE, _mlenvs_base_image_tag()
    if flavor == "mlenvs-gpu-blackwell":
        return MLENVS_GPU_BLACKWELL_BASE_IMAGE, _mlenvs_blackwell_base_image_tag()
    if flavor == "mlenvs-cuda-graphics-blackwell":
        return (
            MLENVS_CUDA_GRAPHICS_BLACKWELL_BASE_IMAGE,
            _mlenvs_blackwell_base_image_tag(),
        )
    return CPU_BASE_IMAGE, CPU_BASE_IMAGE_TAG


# Rough "size" rank used by batch submission to pick the most demanding
# resource tier across selected problems (so each container has at least
# what its own task asked for). Higher = more demanding.
_RESOURCE_RANK: dict[str, int] = RESOURCE_RANK

TMUX_NOTICE = (
    "For long-running training, you may use the dedicated tmux tool, not tmux "
    "inside the bash tool, or an equivalent persistent session to avoid losing work."
)


def derive_taiga_resources(problem_dir: Path) -> dict[str, str]:
    """Derive Boreal base image, tag, model, and required_resources from
    ``task.toml`` ``[environment]``."""
    task_toml = load_task_toml(problem_dir)
    return _derive_resources_from_toml(task_toml)


def _derive_resources_from_toml(task_toml: TaskToml) -> dict[str, str]:
    env = task_toml.environment
    runner = task_toml.runner
    required = validate_required_resources(env.required_resources)
    flavor = resolve_base_flavor_for_resource(getattr(env, "base_flavor", "auto"), required)
    base_image, base_tag = _base_image_and_tag(flavor)

    return {
        "required_resources": required,
        "api_model_name": runner.api_model_name,
        "base_flavor": flavor,
        "base_image": base_image,
        "base_tag": base_tag,
        "base_image_ref": f"{base_image}:{base_tag}",
    }


def _resource_rank(resource: str) -> int:
    if resource in _RESOURCE_RANK:
        return _RESOURCE_RANK[resource]
    return -1


def _max_required_resources(resources: list[str]) -> str:
    if not resources:
        return "16vcpu+64gib"
    return max(resources, key=_resource_rank)


def _prompt_mentions_tmux(prompt: str) -> bool:
    return "tmux" in prompt.lower()


def _accelerator_notice(required_resources: str) -> str:
    lower = (required_resources or "").lower()
    if "tpu" in lower:
        return TPU_NOTICE
    if "/" not in lower:
        return ""
    return GPU_NOTICE


def _append_runtime_notices(prompt: str, required_resources: str) -> str:
    trailing: list[str] = []
    accelerator_notice = _accelerator_notice(required_resources)
    if accelerator_notice:
        trailing.append(accelerator_notice)
    if not _prompt_mentions_tmux(prompt):
        trailing.append(TMUX_NOTICE)
    if not trailing:
        return prompt
    return prompt.rstrip("\n") + "\n\n" + " ".join(trailing)


# ── Per-problem entry ─────────────────────────────────────────────


def _task_type_hint(task_type: str) -> str:
    return task_type_solver_hint(task_type)


def _read_prompt(problem_dir: Path) -> str:
    return read_prompt(problem_dir)


def _task_classification_metadata(task_toml: TaskToml) -> dict[str, Any]:
    d = task_toml.difficulty
    out = {
        "task_type": d.task_type,
        "domain": d.domain,
        "reward_type": d.reward_type,
        "license": d.license,
        "license_source": d.license_source,
        "description": task_toml.task.description or "",
        "is_impossible": d.is_impossible,
    }
    return out


def _taiga_hints(task_toml: TaskToml) -> list[dict[str, Any]]:
    solver_availability_hint = _task_type_hint(str(task_toml.difficulty.task_type)).strip()
    hints = [{"message": solver_availability_hint, "enabled": True}] if solver_availability_hint else []
    hints.extend(
        {
            "message": hint.text,
            "enabled": hint.enabled,
            "spoiler_level": hint.spoiler_level,
        }
        for hint in task_toml.hint
        if hint.text.strip()
    )
    return hints


def _taiga_outputs(task_toml: TaskToml) -> list[dict[str, Any]]:
    """Serialize authored output declarations for Taiga and QA consumers."""
    return [
        {
            "path": output.path,
            "required": output.required,
            "description": output.description,
        }
        for output in task_toml.outputs
    ]


def _problem_set_name(task_toml: TaskToml) -> str:
    task_type = (task_toml.difficulty.task_type or "task").strip().lower()
    task_type = re.sub(r"[^a-z0-9]+", "_", task_type).strip("_") or "task"
    return f"lbx_rl_tasks_{task_type}"


_GROUND_TRUTH_PROOF_PATH = Path(".alignerr") / "build_proof.json"
_GROUND_TRUTH_ARTIFACT_FIELDS = (
    "path",
    "logical_path",
    "sha256",
    "bytes",
    "width",
    "height",
)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _ground_truth_passed(score: float | None, task_toml: TaskToml) -> bool | None:
    if score is None:
        return False
    expectation = expected_ground_truth_score(
        task_toml.difficulty.reward_type,
        deterministic_epsilon=task_toml.ground_truth.score_epsilon,
        continuous_epsilon=task_toml.ground_truth.continuous_score_epsilon,
    )
    return expectation.passed(score)


def _redacted_review_artifacts(raw_artifacts: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_artifacts, list):
        return []

    artifacts: list[dict[str, Any]] = []
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            continue
        artifact = {
            field: raw_artifact[field]
            for field in _GROUND_TRUTH_ARTIFACT_FIELDS
            if field in raw_artifact and raw_artifact[field] is not None
        }
        if artifact:
            artifacts.append(artifact)
    return artifacts


def _ground_truth_evidence(problem_dir: Path, task_toml: TaskToml) -> dict[str, Any] | None:
    """Return a compact, non-secret ground-truth proof summary for Taiga."""
    proof_path = problem_dir / _GROUND_TRUTH_PROOF_PATH
    if not proof_path.exists():
        return None

    try:
        proof = json.loads(proof_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(proof, dict):
        return None

    result = proof.get("ground_truth_result")
    if not isinstance(result, dict):
        return None

    score = _finite_float(result.get("score"))
    expectation = expected_ground_truth_score(
        task_toml.difficulty.reward_type,
        deterministic_epsilon=task_toml.ground_truth.score_epsilon,
        continuous_epsilon=task_toml.ground_truth.continuous_score_epsilon,
    )
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "source": _GROUND_TRUTH_PROOF_PATH.as_posix(),
        "build_proof_sha256": sha256_file(proof_path),
        "task_type": task_toml.difficulty.task_type,
        "domain": task_toml.difficulty.domain,
        "reward_type": task_toml.difficulty.reward_type,
        "expected_score": expectation.target,
        "score_epsilon": expectation.epsilon,
    }
    passed = _ground_truth_passed(score, task_toml)
    if passed is not None:
        evidence["passed"] = passed
    if score is not None:
        evidence["score"] = score

    for key in ("runtime", "graded_at"):
        value = result.get(key)
        if isinstance(value, str) and value:
            evidence[key] = value

    review_artifacts = _redacted_review_artifacts(result.get("review_artifacts"))
    if review_artifacts:
        evidence["review_artifacts"] = review_artifacts

    return evidence


def test_file_shim() -> str:
    """Return the Boreal shim that invokes the image-baked task scorer."""
    return """import importlib.util
import inspect
import pathlib
import sys

for _path in ("/runtime/grading/src", "/mcp_server/grader"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_grader_path = pathlib.Path("/mcp_server/grader/compute_score.py")
_spec = importlib.util.spec_from_file_location("task_compute_score", _grader_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot import {_grader_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_compute_score = _module.compute_score
_takes_args = bool(inspect.signature(_compute_score).parameters)


def compute_score():
    # ML_Envs-mode graders define a no-arg compute_score() that reads the baked
    # /tmp/output and /mcp_server/data paths directly; call it with no args.
    if not _takes_args:
        return _compute_score()
    # Native graders take (workspace, trajectory, private). The rubric runtime
    # injects the agent transcript as a ``TRANSCRIPT`` global (empty when none);
    # forward it as ``trajectory`` for transcript-based anti-cheat checks.
    return _compute_score(
        workspace=pathlib.Path("/tmp/output"),
        trajectory=globals().get("TRANSCRIPT") or [],
        private=pathlib.Path("/mcp_server/data"),
    )
"""


def _taiga_container_runtime(required_resources: str) -> str:
    """Taiga isolation runtime for a resolved resource tier.

    Accelerator tiers carry either a ``+<accelerator>/<count>`` share (GPU) or a
    TPU topology suffix such as ``+tpuv5e1x1``. CPU-only tiers are
    ``<n>vcpu+<m>gib``. This mirrors the submit-time safety net in
    ``grade-fork-pr.yml`` so the exporter emits the runtime Taiga needs directly,
    instead of leaking the local-harness ``runner.container_runtime`` literal
    (firecracker/docker).
    """
    return "gvisor" if is_accelerator_resource(required_resources) else "firecracker"


def _build_problem_entry(
    problem_dir: Path,
    *,
    image_ref: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-problem entry dict that lives under
    ``problems_metadata.problem_set.problems[]``."""
    from alignerr_plugin import mlenvs
    from alignerr_plugin.preloaded import load_preloaded_manifest

    # mlenvs bases install the rubric MCP server --system (no venv); point their
    # startup command at the --system console script.
    startup_command = (
        MLENVS_STARTUP_COMMAND
        if mlenvs.is_mlenvs_task(problem_dir)
        else STARTUP_COMMAND
    )

    task_toml = load_task_toml(problem_dir)
    runner: RunnerConfig = task_toml.runner
    # ml tasks are force-pinned to the hour-scale ML_* timeouts; other task types
    # keep their author-set / default values (see _effective_timeouts). The
    # job-level max_episode is applied separately in _job_level_fields.
    setup_timeout_seconds, grading_timeout_seconds, tool_timeout_seconds, _ = (
        _effective_timeouts(task_toml)
    )
    resources = _derive_resources_from_toml(task_toml)
    prompt = _read_prompt(problem_dir)
    shim = test_file_shim()

    required_resources = validate_required_resources(
        (overrides or {}).get("required_resources", resources["required_resources"])
    )
    prompt = _append_runtime_notices(prompt, required_resources)
    preloaded_manifest = load_preloaded_manifest(problem_dir)
    required_tools = (overrides or {}).get("required_tools", runner.required_tools)
    # Derive the Taiga isolation runtime from the resolved resource tier so
    # accelerator tasks deploy under gVisor and CPU tasks under firecracker. The
    # task.toml runner.container_runtime literal (firecracker/docker) is a
    # LOCAL-harness knob and is intentionally NOT forwarded to Taiga -- pinning a
    # GPU task to firecracker would ship it under the wrong sandbox. An explicit
    # override still wins.
    container_runtime = (overrides or {}).get("container_runtime") or _taiga_container_runtime(required_resources)
    enable_anthropic_api = (overrides or {}).get("enable_anthropic_api", runner.enable_anthropic_api)
    outputs = _taiga_outputs(task_toml)
    task_metadata = dict(task_toml.metadata) if task_toml.metadata else {}
    if outputs:
        # Keep task.toml [[outputs]] available to local Taiga-format harness
        # readers and QA tools that inspect extra_fields.task_metadata.
        task_metadata["outputs"] = outputs
    extra_fields: dict[str, Any] = {
        "test_file": shim,
        # The in-image rubric runtime uses this as its internal subprocess
        # timeout, matching Boreal's outer grading timeout for the task.
        "grading_timeout_seconds": grading_timeout_seconds,
        # Free-form metadata authors can drop in via task.toml
        "task_metadata": task_metadata,
    }
    ground_truth_evidence = _ground_truth_evidence(problem_dir, task_toml)
    if ground_truth_evidence is not None:
        extra_fields["ground_truth_evidence"] = ground_truth_evidence

    metadata = _task_classification_metadata(task_toml)
    metadata["base_image"] = {
        "flavor": resources["base_flavor"],
        "image": resources["base_image"],
        "tag": resources["base_tag"],
        "ref": resources["base_image_ref"],
    }

    entry: dict[str, Any] = {
        "id": task_id(problem_dir),
        "image": image_ref,
        "startup_command": startup_command,
        "task_prompt": prompt,
        "required_tools": required_tools,
        "required_resources": required_resources,
        "container_runtime": container_runtime,
        "enable_anthropic_api": enable_anthropic_api,
        "output_directory": "/tmp/output",
        # Schema-required per-problem field (Taiga job_runner). ML_Envs pins
        # "allowed" on every problem; omit it and the problem entry fails the
        # per-problem schema's `required` check at submit.
        "scratchpad": "allowed",
        "setup_timeout_seconds": setup_timeout_seconds,
        "grading_timeout_seconds": grading_timeout_seconds,
        "tool_timeout_seconds": tool_timeout_seconds,
        "rubric": [],
        "grading_strategy": [{"type": "mcp", "weight": 1.0}],
        "metadata": metadata,
        "extra_fields": extra_fields,
    }
    hints = _taiga_hints(task_toml)
    if hints:
        entry["hints"] = hints
    if outputs:
        # Top-level visibility mirrors task.toml for Taiga-side QA and triage.
        entry["outputs"] = outputs
    if preloaded_manifest:
        # Deploy-time read-only mounts (large datasets / HF weights) instead of
        # image-baked data. Empty -> field omitted (image-baked fallback).
        entry["preloaded_files"] = preloaded_manifest
    if task_toml.environment.hidden_env:
        # Surface the hidden-environment RPC mode (env/hybrid) for platform
        # visibility. The runtime supervisor reads it from the baked
        # /task/task.toml, so this field is informational for the deploy side.
        entry["hidden_env"] = task_toml.environment.hidden_env
    return entry


# ── Job-level payload ─────────────────────────────────────────────


def _job_level_fields(
    runner: RunnerConfig,
    *,
    n_attempts: int | None,
    turn_limit: int | None,
    max_ctx: int | None,
    model: str | None,
    environment_id: str | None,
    max_episode_sec: int | None,
) -> dict[str, Any]:
    """Compose the job-level (non-per-problem) payload fields.

    Caller-supplied args win over ``[runner]`` defaults so the CLI can
    override per submission.
    """
    effective_attempts = n_attempts if n_attempts is not None else runner.attempts
    effective_max_ctx = max_ctx if max_ctx is not None else runner.max_ctx
    effective_model = model if model else runner.api_model_name
    context_mode = runner.context_mode

    payload: dict[str, Any] = {
        "api_model_name": effective_model,
        "n_attempts_per_problem": effective_attempts,
        "max_ctx": effective_max_ctx,
        "enable_autocompact": context_mode == "autocompact",
        "enable_memory": context_mode == "memory",
        "priority": runner.priority,
        "iteration_order": runner.iteration_order,
    }

    # Turn limit: caller arg wins (None means "use runner default"). Explicit
    # null/0 in [runner] means "unlimited" — emit no turn_limit field.
    effective_turn_limit = turn_limit if turn_limit is not None else runner.turn_limit
    if effective_turn_limit and effective_turn_limit > 0:
        payload["turn_limit"] = effective_turn_limit

    if max_episode_sec and max_episode_sec > 0:
        payload["max_timeout_seconds"] = max_episode_sec

    if runner.serialize_restore_test_interval:
        payload["serialize_restore_test_interval"] = runner.serialize_restore_test_interval
    if runner.checkpoint_ttl:
        payload["checkpoint_ttl"] = runner.checkpoint_ttl
    if environment_id:
        payload["environment_id"] = environment_id

    return payload


def build_job_payload(
    problem_dir: Path,
    *,
    image_ref: str,
    n_attempts: int | None = None,
    turn_limit: int | None = None,
    max_ctx: int | None = None,
    model: str | None = None,
    environment_id: str | None = None,
) -> dict[str, Any]:
    """Build a single-problem Boreal job payload."""
    task_toml = load_task_toml(problem_dir)
    runner = task_toml.runner

    problem_entry = _build_problem_entry(problem_dir, image_ref=image_ref)
    job_fields = _job_level_fields(
        runner,
        n_attempts=n_attempts,
        turn_limit=turn_limit,
        max_ctx=max_ctx,
        model=model,
        environment_id=environment_id,
        max_episode_sec=_effective_timeouts(task_toml)[3],
    )

    payload: dict[str, Any] = {
        "name": f"rl_{problem_entry['id']}_{int(time.time())}",
        **job_fields,
        "problems_metadata": {
            "problem_set": {
                "owner": "labelbox",
                "name": _problem_set_name(task_toml),
                "description": task_toml.task.description or "",
                "version": "1.0.0",
                "created_at": datetime.now(UTC).isoformat(),
                "metadata": {"source_repo": "lbx-rl-tasks-mothership"},
                "required_resources": problem_entry["required_resources"],
                "container_runtime": problem_entry["container_runtime"],
                "problems": [problem_entry],
            }
        },
    }
    return payload


def build_batch_job_payload(
    problem_dirs: list[Path],
    *,
    image_refs: dict[str, str],
    n_attempts: int | None = None,
    turn_limit: int | None = None,
    max_ctx: int | None = None,
    model: str | None = None,
    environment_id: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Boreal job payload that runs N problems in one submission.

    ``image_refs`` maps each problem id to its digest-pinned per-task image.
    The job-level fields are taken from the FIRST problem's ``[runner]``;
    per-problem fields come from each problem's own ``[runner]`` with
    ``overrides`` applied last.
    """
    if not problem_dirs:
        raise ValueError("build_batch_job_payload requires at least one problem")

    entries: list[dict[str, Any]] = []
    for pd in problem_dirs:
        pid = task_id(pd)
        if pid not in image_refs:
            raise ValueError(
                f"image_refs missing entry for problem {pid!r}; supply " "the digest-pinned per-task image."
            )
        entries.append(
            _build_problem_entry(
                pd,
                image_ref=image_refs[pid],
                overrides=overrides,
            )
        )

    first_toml = load_task_toml(problem_dirs[0])
    problem_set_name = _problem_set_name(first_toml)
    job_fields = _job_level_fields(
        first_toml.runner,
        n_attempts=n_attempts,
        turn_limit=turn_limit,
        max_ctx=max_ctx,
        model=model,
        environment_id=environment_id,
        max_episode_sec=_effective_timeouts(first_toml)[3],
    )

    shared_resources = validate_required_resources(
        (overrides or {}).get(
            "required_resources",
            _max_required_resources([e["required_resources"] for e in entries]),
        )
    )
    shared_runtime = (overrides or {}).get("container_runtime") or _taiga_container_runtime(shared_resources)

    return {
        "name": f"rl_batch_{int(time.time())}",
        **job_fields,
        "problems_metadata": {
            "problem_set": {
                "owner": "labelbox",
                "name": f"{problem_set_name}_batch",
                "description": f"Batch of {len(entries)} RL tasks",
                "version": "1.0.0",
                "created_at": datetime.now(UTC).isoformat(),
                "metadata": {"source_repo": "lbx-rl-tasks-mothership"},
                "required_resources": shared_resources,
                "container_runtime": shared_runtime,
                "problems": entries,
            }
        },
    }


# ── Legacy `problems-metadata.json` writer ────────────────────────


def export_taiga(
    problem_dir: Path,
    output_path: Path,
    *,
    image_ref: str = "PLACEHOLDER",
) -> dict[str, Any]:
    """Write the legacy ``problems-metadata.json`` shape and the
    ``.taiga_submit.json`` sidecar.

    The shape matches what ML_Envs ships so downstream scripts (Boreal submit
    workflows, polling, dashboards) keep working. The full job-level fields
    (n_attempts, turn_limit, max_ctx, ...) are NOT included here — those go
    on the live submit payload built by :func:`build_job_payload`.
    """
    task_toml = load_task_toml(problem_dir)
    resources = _derive_resources_from_toml(task_toml)
    problem_entry = _build_problem_entry(problem_dir, image_ref=image_ref)

    metadata = {
        "problem_set": {
            "owner": "labelbox",
            "name": _problem_set_name(task_toml),
            "description": task_toml.task.description,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "metadata": {"source_repo": "lbx-rl-tasks-mothership"},
            "required_resources": resources["required_resources"],
            "container_runtime": problem_entry["container_runtime"],
            "problems": [problem_entry],
        }
    }
    output_path.write_text(json.dumps(metadata, indent=2) + "\n")

    sidecar = {
        "task_id": problem_entry["id"],
        "image": image_ref,
        **resources,
    }
    write_json(problem_dir / ".taiga_submit.json", sidecar)
    return sidecar
