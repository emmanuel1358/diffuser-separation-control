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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alignerr_plugin.base_image import resolve_base_flavor_for_resource
from alignerr_plugin.capabilities import (
    CapabilityConfig,
    ServiceSpec,
    implicit_agent_service,
    is_capability_task,
    model_dump,
    resolve_capabilities,
)
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
    CPU_PERF_RESOURCE_OPTIONS,
    CPU_RESOURCE_OPTIONS,
    RESOURCE_RANK,
    is_accelerator_resource,
    is_cpu_resource,
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
BASE_IMAGE_TAG = os.environ.get(
    "LBX_RL_TASKS_BASE_IMAGE_TAG", "runtime-ml-core-py313-1b75cb075439"
)
# The cpu flavor can be pinned independently of the shared gpu/cpu tag; the
# mothership sync re-pins this assignment per repo (same format contract as
# BASE_IMAGE_TAG above).
CPU_BASE_IMAGE_TAG = os.environ.get("LBX_RL_TASKS_CPU_BASE_IMAGE_TAG", BASE_IMAGE_TAG)

CPU_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base"
GPU_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-gpu"
GPU_OPENROAD_BASE_IMAGE = (
    "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-gpu-openroad"
)
GPU_BLACKWELL_BASE_IMAGE = (
    "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-gpu-blackwell"
)
CUDA_GRAPHICS_BASE_IMAGE = (
    "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-cuda-graphics"
)
TPU_BASE_IMAGE = "us-east1-docker.pkg.dev/gcp-taiga/labelbox/lbx-tasks-base-tpu"

# Set by the automatic CPU QA lane to a CPU required_resources tier. When present,
# the export forces that tier and swaps the base flavor for its CPU counterpart, so
# the image, tag, container_runtime AND the agent-facing accelerator notice all
# describe the machine the job will actually land on. Unset everywhere else, which
# leaves the declared tier untouched.
QA_CPU_RESOURCE_ENV = "LBX_TAIGA_QA_CPU_RESOURCE"

_CAPABILITY_SUMMARY_SCHEMA = "alignerr.taiga.capability-summary.v1"
_OUTER_CAPSULE_IMAGE_RE = re.compile(
    r"^[^\s@]+@sha256:[0-9a-f]{64}\Z",
    re.IGNORECASE,
)
_RESOURCE_DIMENSIONS = ("cpus", "memory_mb", "storage_mb", "gpus")
_CPU_MEMORY_DIMENSIONS = ("cpus", "memory_mb")
_CPU_RESOURCE_RE = re.compile(
    r"^(?P<cpus>\d+)vcpu\+(?P<memory_gib>\d+)gib(?:\+perf)?\Z"
)


def _tpu_base_image_tag() -> str:
    # tpu uses a distinct tag prefix (py3.12). Set via env at deploy
    # (build_and_push.sh publishes the drift-hashed tpu tag); the committed
    # fallback mirrors the cpu/gpu registry-style pin (not a "-local" dev tag).
    return os.environ.get(
        "LBX_RL_TASKS_TPU_BASE_IMAGE_TAG", "runtime-ml-tpu-py312-1b75cb075439"
    )


def _graphics_base_image_tag() -> str:
    # cuda-graphics runs py3.13 like cpu/gpu but keeps its own tag prefix and
    # env var, so its CUDA-native rasterization layers are pinned independently.
    return os.environ.get(
        "LBX_RL_TASKS_GRAPHICS_BASE_IMAGE_TAG",
        "runtime-ml-graphics-py313-1b75cb075439",
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
        return CUDA_GRAPHICS_BASE_IMAGE, _graphics_base_image_tag()
    if flavor == "tpu":
        return TPU_BASE_IMAGE, _tpu_base_image_tag()
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


def qa_cpu_flavor_for(flavor: str) -> str:
    """Map a resolved base flavor to its CPU-lane counterpart.

    Every flavor shares one runtime contract, so the CPU QA lane always runs on
    the ``cpu`` base.
    """
    _ = flavor
    return "cpu"


def _qa_cpu_resource_override() -> str:
    return (os.environ.get(QA_CPU_RESOURCE_ENV) or "").strip()


def _derive_resources_from_toml(task_toml: TaskToml) -> dict[str, str]:
    env = task_toml.environment
    runner = task_toml.runner
    required = validate_required_resources(env.required_resources)
    # Resolve against the DECLARED tier first so a bad author flavor still fails
    # loudly, before the CPU QA lane below can swap it out.
    flavor = resolve_base_flavor_for_resource(
        getattr(env, "base_flavor", "auto"), required
    )

    override = _qa_cpu_resource_override()
    if override:
        required = validate_required_resources(override)
        if not is_cpu_resource(required):
            raise ValueError(
                f"{QA_CPU_RESOURCE_ENV} must be a CPU required_resources tier; got "
                f"{required!r}. The CPU QA lane submits without the deploy-to-taiga "
                "reviewer gate, so it must never request an accelerator."
            )
        flavor = qa_cpu_flavor_for(flavor)

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
    solver_availability_hint = _task_type_hint(
        str(task_toml.difficulty.task_type)
    ).strip()
    hints = (
        [{"message": solver_availability_hint, "enabled": True}]
        if solver_availability_hint
        else []
    )
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


def _require_outer_capsule_image(
    *,
    image_ref: str,
    image_is_outer_capsule: bool,
) -> str:
    """Validate and return the assertion source for a capsule image reference."""
    if not image_is_outer_capsule:
        raise ValueError(
            "capability-aware Taiga export requires the supplied image_ref to be "
            "the built outer capsule; trusted export code must pass "
            "image_is_outer_capsule=True"
        )

    if image_ref != "LOCAL_IMAGE" and not _OUTER_CAPSULE_IMAGE_RE.fullmatch(image_ref):
        raise ValueError(
            "capability-aware Taiga export requires a digest-pinned outer capsule "
            "image_ref (<repository>@sha256:<64 hex>), or LOCAL_IMAGE for an "
            f"explicit local capsule; got {image_ref!r}"
        )
    return "exporter_flag"


def _resolve_capabilities_for_problem(
    problem_dir: Path,
    task_toml: TaskToml,
) -> CapabilityConfig:
    """Apply the same implicit main-service convention as capsule packaging."""
    capabilities = resolve_capabilities(task_toml)
    if capabilities.agent_service is not None:
        return capabilities
    default_context = next(
        (
            context
            for context in ("environment", "environment/main")
            if (problem_dir / context / "Dockerfile").is_file()
        ),
        None,
    )
    if default_context is None:
        raise ValueError(
            "capability task has no agent service and neither "
            "environment/Dockerfile nor environment/main/Dockerfile is available"
        )
    platform = (
        task_toml.agent.resources.platform
        if task_toml.agent.resources is not None
        else None
    )
    default_agent = implicit_agent_service(
        context=default_context,
        resources=capabilities.agent_resources,
        network=(
            task_toml.agent.resources.network
            if task_toml.agent.resources is not None
            else None
        ),
        platform=platform,
    )
    return CapabilityConfig(
        services=(default_agent, *capabilities.services),
        artifacts=capabilities.artifacts,
        captures=capabilities.captures,
        mcp_servers=capabilities.mcp_servers,
        verifier_mcp_servers=capabilities.verifier_mcp_servers,
        volumes=capabilities.volumes,
        agent_resources=capabilities.agent_resources,
        verifier_resources=capabilities.verifier_resources,
    )


def _validate_capability_mcp(task_toml: TaskToml) -> None:
    """Allow only the audited SSE proxy implemented by the capsule runtime."""
    unsupported = sorted(
        f"{server.name}:{server.transport}"
        for server in task_toml.mcp_servers
        if server.transport != "sse"
    )
    if unsupported:
        raise ValueError(
            "Taiga task capsules support only declared SSE MCP endpoints through "
            "the audited service-DNS proxy; unsupported task-local MCP transports: "
            + ", ".join(unsupported)
        )


def _canonical_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _resource_vector(value: Any, *, label: str) -> dict[str, Any]:
    """Project a resource declaration to non-secret numeric capacity fields."""
    raw = dict(value) if isinstance(value, Mapping) else model_dump(value)
    vector: dict[str, Any] = {}
    for field in _RESOURCE_DIMENSIONS:
        raw_value = raw.get(field)
        if raw_value is None:
            continue
        if isinstance(raw_value, bool):
            raise TypeError(f"{label} {field} must be numeric, not boolean")
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} {field} must be numeric") from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{label} {field} must be finite and non-negative")
        if field != "cpus" and not number.is_integer():
            raise ValueError(f"{label} {field} must be an integer")
        vector[field] = _canonical_number(number)

    gpu_types = raw.get("gpu_types")
    if isinstance(gpu_types, Sequence) and not isinstance(
        gpu_types, (str, bytes, bytearray)
    ):
        normalized_types = sorted({str(gpu_type) for gpu_type in gpu_types if gpu_type})
        if normalized_types:
            vector["gpu_types"] = normalized_types
    if raw.get("tpu"):
        vector["tpu"] = True
    return vector


def _merge_resource_vectors(
    vectors: Sequence[Mapping[str, Any]],
    *,
    operation: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for field in _RESOURCE_DIMENSIONS:
        values = [vector[field] for vector in vectors if field in vector]
        if values:
            merged[field] = sum(values) if operation == "sum" else max(values)
    gpu_types = sorted(
        {
            str(gpu_type)
            for vector in vectors
            for gpu_type in vector.get("gpu_types", [])
        }
    )
    if gpu_types:
        merged["gpu_types"] = gpu_types
    if any(vector.get("tpu") is True for vector in vectors):
        merged["tpu"] = True
    return merged


def _sum_resource_vectors(*vectors: Mapping[str, Any]) -> dict[str, Any]:
    return _merge_resource_vectors(vectors, operation="sum")


def _max_resource_vectors(*vectors: Mapping[str, Any]) -> dict[str, Any]:
    return _merge_resource_vectors(vectors, operation="max")


def _resource_capacity(required_resources: str) -> tuple[int, int]:
    match = _CPU_RESOURCE_RE.fullmatch(required_resources)
    if match is None:
        raise ValueError(
            "nested-Docker capability tasks require a CPU Taiga resource enum; "
            f"got {required_resources!r}"
        )
    return int(match.group("cpus")), int(match.group("memory_gib")) * 1024


def _minimum_cpu_capacity_fit(
    cpus: float,
    memory_mb: int,
    *,
    perf: bool,
) -> str | None:
    options = CPU_PERF_RESOURCE_OPTIONS if perf else CPU_RESOURCE_OPTIONS
    fits = [
        option
        for option in options
        if (
            _resource_capacity(option)[0] >= cpus
            and _resource_capacity(option)[1] >= memory_mb
        )
    ]
    if not fits:
        return None
    return min(
        fits,
        key=lambda option: (
            _resource_capacity(option)[0],
            _resource_capacity(option)[1],
        ),
    )


def _service_resource_vector(service: ServiceSpec) -> dict[str, Any]:
    return _resource_vector(
        service.resources,
        label=f"service {service.name!r} resources",
    )


def _phase_resource_intent(
    task_toml: TaskToml,
    capabilities: CapabilityConfig,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Compute conservative, non-overlapping agent and verifier phase peaks."""
    missing: list[str] = []
    agent_resources = _resource_vector(
        capabilities.agent_resources,
        label="agent resources",
    )
    main_resources = (
        _service_resource_vector(capabilities.agent_service)
        if capabilities.agent_service is not None
        else {}
    )
    main_peak = _max_resource_vectors(agent_resources, main_resources)
    for field in _CPU_MEMORY_DIMENSIONS:
        if field not in main_peak:
            missing.append(f"agent-main.{field}")

    agent_components: list[Mapping[str, Any]] = [main_peak]
    for service in capabilities.services:
        if service.role in {"agent", "verifier"}:
            continue
        service_resources = _service_resource_vector(service)
        for field in _CPU_MEMORY_DIMENSIONS:
            if field not in service_resources:
                missing.append(f"service:{service.name}.{field}")
        agent_components.append(service_resources)
    agent_phase = _sum_resource_vectors(*agent_components)
    # The outer disk must also accommodate the task-level storage request. Child
    # service storage is additive, while environment.storage_mb is an outer floor.
    agent_phase = _max_resource_vectors(
        agent_phase,
        {"storage_mb": task_toml.environment.storage_mb},
    )

    verifier_resources = _resource_vector(
        capabilities.verifier_resources,
        label="verifier resources",
    )
    verifier_service_resources = (
        _service_resource_vector(capabilities.verifier_service)
        if capabilities.verifier_service is not None
        else {}
    )
    verifier_phase = _max_resource_vectors(
        verifier_resources,
        verifier_service_resources,
    )
    for field in _CPU_MEMORY_DIMENSIONS:
        if field not in verifier_phase:
            missing.append(f"verifier.{field}")
    return agent_phase, verifier_phase, sorted(set(missing))


def _outer_resource_summary(
    task_toml: TaskToml,
    capabilities: CapabilityConfig,
    *,
    selected_required_resources: str,
) -> dict[str, Any]:
    """Validate known phase peaks while preserving the authored Taiga enum."""
    authored = validate_required_resources(task_toml.environment.required_resources)
    selected = validate_required_resources(selected_required_resources)
    if not is_cpu_resource(selected):
        raise ValueError(
            "Taiga nested-Docker capsules require the Firecracker CPU runtime, "
            f"but selected required_resources {selected!r} is an accelerator tier"
        )

    agent_phase, verifier_phase, missing = _phase_resource_intent(
        task_toml,
        capabilities,
    )
    outer_peak = _max_resource_vectors(agent_phase, verifier_phase)
    if outer_peak.get("gpus", 0) or outer_peak.get("tpu"):
        raise ValueError(
            "Taiga nested-Docker capsules currently run under Firecracker and "
            "cannot satisfy child GPU/TPU resource declarations"
        )

    selected_cpus, selected_memory_mb = _resource_capacity(selected)
    complete_cpu_memory = all(field in outer_peak for field in _CPU_MEMORY_DIMENSIONS)
    minimum_fit: str | None = None
    exact_fit = False
    if complete_cpu_memory:
        outer_cpus = float(outer_peak["cpus"])
        outer_memory_mb = int(outer_peak["memory_mb"])
        perf = selected.endswith("+perf")
        minimum_fit = _minimum_cpu_capacity_fit(
            outer_cpus,
            outer_memory_mb,
            perf=perf,
        )
        if minimum_fit is None:
            raise ValueError(
                "computed outer capsule resource intent cannot map to any Taiga "
                f"CPU enum: cpus={outer_peak['cpus']}, "
                f"memory_mb={outer_peak['memory_mb']}"
            )
        if outer_cpus > selected_cpus or outer_memory_mb > selected_memory_mb:
            raise ValueError(
                "computed outer capsule resource intent exceeds selected "
                f"required_resources {selected!r}: peak cpus={outer_peak['cpus']}, "
                f"memory_mb={outer_peak['memory_mb']}; minimum capacity fit is "
                f"{minimum_fit!r}"
            )
        candidate_options = CPU_PERF_RESOURCE_OPTIONS if perf else CPU_RESOURCE_OPTIONS
        exact_fit = any(
            _resource_capacity(option) == (outer_cpus, outer_memory_mb)
            for option in candidate_options
        )
    else:
        known_shortfalls = []
        if float(outer_peak.get("cpus", 0)) > selected_cpus:
            known_shortfalls.append(
                f"cpus={outer_peak['cpus']} exceeds {selected_cpus}"
            )
        if int(outer_peak.get("memory_mb", 0)) > selected_memory_mb:
            known_shortfalls.append(
                f"memory_mb={outer_peak['memory_mb']} exceeds {selected_memory_mb}"
            )
        if known_shortfalls:
            raise ValueError(
                "known outer capsule resource intent exceeds selected "
                f"required_resources {selected!r} even though some child limits "
                f"are undeclared: {', '.join(known_shortfalls)}"
            )

    reasons = ["capsule_image_and_daemon_overhead_not_encoded"]
    if missing:
        reasons.append("incomplete_child_cpu_or_memory_declarations")
    if outer_peak.get("storage_mb", 0):
        reasons.append("storage_and_named_volume_capacity_not_encoded")
    if complete_cpu_memory and not exact_fit:
        reasons.append("aggregate_peak_is_not_an_exact_taiga_enum")

    mapping: dict[str, Any] = {
        "strategy": "preserve_selected_required_resources",
        "capacity_validated": complete_cpu_memory and not missing,
        "preflight_required": True,
        "preflight_requirement": (
            "Verify the built outer capsule peak, including child images, dockerd, "
            "named volumes, and undeclared child limits, fits selected "
            "required_resources before Taiga submission."
        ),
        "reasons": sorted(reasons),
    }
    if minimum_fit is not None:
        mapping["minimum_capacity_fit"] = minimum_fit
    if missing:
        mapping["unknown_fields"] = missing

    return {
        "authored_required_resources": authored,
        "selected_required_resources": selected,
        "selection": "authored" if selected == authored else "explicit_override",
        "agent_service_peak": agent_phase,
        "verifier_phase": verifier_phase,
        "outer_peak": outer_peak,
        "mapping": mapping,
    }


def _timeout_value(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return math.ceil(number)


def _capability_effective_timeouts(
    task_toml: TaskToml,
    capabilities: CapabilityConfig,
) -> tuple[int, int, int, int | None]:
    """Surface long native agent, build, and verifier phase timeouts."""
    setup, grading, tool, max_episode = _effective_timeouts(task_toml)
    agent_resource = model_dump(task_toml.agent.resources)
    verifier_resource = model_dump(task_toml.verifier.resources)

    build_timeouts = [
        _timeout_value(agent_resource.get("build_timeout_sec")),
        _timeout_value(verifier_resource.get("build_timeout_sec")),
    ]
    agent_runtime_timeouts = [
        _timeout_value(task_toml.agent.timeout_sec),
        _timeout_value(agent_resource.get("runtime_timeout_sec")),
    ]
    verifier_runtime_timeouts = [
        _timeout_value(task_toml.verifier.timeout_sec),
        _timeout_value(verifier_resource.get("runtime_timeout_sec")),
    ]
    for service in capabilities.services:
        raw_resources = model_dump(service.raw.get("resources"))
        build_timeouts.append(_timeout_value(raw_resources.get("build_timeout_sec")))
        runtime_timeout = _timeout_value(raw_resources.get("runtime_timeout_sec"))
        if service.role == "verifier":
            verifier_runtime_timeouts.append(runtime_timeout)
        else:
            agent_runtime_timeouts.append(runtime_timeout)

    setup = max([setup, *(value for value in build_timeouts if value is not None)])
    tool = max(
        [tool, *(value for value in agent_runtime_timeouts if value is not None)]
    )
    grading = max(
        [grading, *(value for value in verifier_runtime_timeouts if value is not None)]
    )
    if max_episode is not None:
        max_episode = max(
            [
                max_episode,
                *(value for value in agent_runtime_timeouts if value is not None),
            ]
        )
    return setup, grading, tool, max_episode


def _submission_timeouts(
    task_toml: TaskToml,
) -> tuple[int, int, int, int | None]:
    if not is_capability_task(task_toml):
        return _effective_timeouts(task_toml)
    return _capability_effective_timeouts(
        task_toml,
        resolve_capabilities(task_toml),
    )


def _canonical_service_role(role: str) -> str:
    return "main" if role == "agent" else role


def _service_identity(service: ServiceSpec) -> dict[str, Any]:
    raw = service.raw
    identity: dict[str, Any] = {
        "name": service.name,
        "role": _canonical_service_role(service.role),
        "source": (
            {"kind": "bundled_build"}
            if service.build is not None
            else {
                "kind": "digest_pinned_image",
                "digest": str(service.image).rsplit("@", 1)[-1].lower(),
            }
        ),
    }
    platform = (
        service.build.platform if service.build is not None else None
    ) or model_dump(raw.get("resources")).get("platform")
    if platform:
        identity["platform"] = str(platform)

    dependencies = []
    for dependency in raw.get("depends_on", []):
        row = model_dump(dependency)
        if row.get("service"):
            dependencies.append(
                {
                    "service": str(row["service"]),
                    "condition": str(row.get("condition") or "started"),
                }
            )
    if dependencies:
        identity["depends_on"] = sorted(
            dependencies,
            key=lambda row: (row["service"], row["condition"]),
        )

    mounts = []
    for mount in raw.get("volumes", []):
        row = model_dump(mount)
        if row.get("volume") and row.get("target"):
            mounts.append(
                {
                    "volume": str(row["volume"]),
                    "target": str(row["target"]),
                    "mode": str(row.get("mode") or "rw"),
                }
            )
    if mounts:
        identity["volumes"] = sorted(
            mounts,
            key=lambda row: (row["volume"], row["target"], row["mode"]),
        )
    resources = _service_resource_vector(service)
    if resources:
        identity["resources"] = resources
    return identity


def _capability_summary(
    task_toml: TaskToml,
    capabilities: CapabilityConfig,
    *,
    contract_assertion: str,
    selected_required_resources: str,
) -> dict[str, Any]:
    """Build a deterministic allowlisted summary without commands or secrets."""
    workspace = model_dump(task_toml.workspace)
    workspace_root = str(workspace.get("root") or "/workdir")
    workspace_identity: dict[str, Any] = {
        "root": workspace_root,
        "agent_cwd": str(workspace.get("agent_cwd") or workspace_root),
        "output_root": (
            task_toml.result.output_root if task_toml.result else "/tmp/output"
        ),
        "init_policy": str(workspace.get("init_policy") or "empty"),
    }
    if workspace.get("seed"):
        workspace_identity["seed"] = str(workspace["seed"])
    if "git_baseline" in workspace:
        workspace_identity["git_baseline"] = workspace["git_baseline"]

    artifacts = []
    for artifact in task_toml.artifacts:
        raw = artifact.model_dump(mode="python", exclude_none=True)
        artifacts.append(
            {
                "name": artifact.name,
                "kind": str(raw["kind"]),
                "service": artifact.service,
                "destination": artifact.destination,
                "required": artifact.required,
            }
        )

    captures = []
    for index, capture in enumerate(task_toml.captures):
        capture_identity: dict[str, Any] = {
            "order": index,
            "name": capture.name,
            "service": capture.service,
            "failure_policy": capture.failure_policy,
        }
        if capture.atomic_destination is not None:
            capture_identity["atomic_destination"] = capture.atomic_destination
        captures.append(capture_identity)

    tools = [
        {"name": name, "kind": "builtin"}
        for name in sorted(set(task_toml.runner.required_tools))
    ]
    for server in sorted(task_toml.mcp_servers, key=lambda item: item.name):
        tool_identity: dict[str, Any] = {
            "name": server.name,
            "kind": "mcp",
            "transport": server.transport,
            "access": server.access,
        }
        if server.service is not None:
            tool_identity["service"] = server.service
        if server.depends_on:
            tool_identity["depends_on"] = sorted(server.depends_on)
        tools.append(tool_identity)

    gates = [
        {
            "order": index,
            "name": gate.name,
            "kind": gate.kind,
            "required": gate.required,
            **({"report": gate.report} if gate.report is not None else {}),
        }
        for index, gate in enumerate(task_toml.gates)
    ]
    reports = [
        {
            "name": report.name,
            "format": report.format,
            "path": report.path,
            "required": report.required,
        }
        for report in sorted(task_toml.reports, key=lambda item: item.name)
    ]

    return {
        "schema_version": _CAPABILITY_SUMMARY_SCHEMA,
        "runtime": {
            "outer_image_contract": "outer_capsule",
            "contract_assertion": contract_assertion,
            "isolation": "firecracker",
            "orchestration": "nested_docker",
        },
        "workspace": workspace_identity,
        "artifacts": sorted(artifacts, key=lambda row: row["name"]),
        "services": [
            _service_identity(service)
            for service in sorted(capabilities.services, key=lambda item: item.name)
        ],
        "volumes": [
            {"name": volume.name}
            for volume in sorted(task_toml.volumes, key=lambda item: item.name)
        ],
        "captures": captures,
        "tools": tools,
        "gates": gates,
        "reports": reports,
        "resources": _outer_resource_summary(
            task_toml,
            capabilities,
            selected_required_resources=selected_required_resources,
        ),
    }


def _problem_set_name(task_toml: TaskToml) -> str:
    task_type = (task_toml.difficulty.task_type or "task").strip().lower()
    task_type = re.sub(r"[^a-z0-9]+", "_", task_type).strip("_") or "task"
    return f"lbx_rl_tasks_{task_type}"


_GROUND_TRUTH_PROOF_PATH = Path(".alignerr") / "build_proof.json"
_CALIBRATION_EVIDENCE_PATH = Path(".alignerr") / "calibration.evidence.json"
_TRUSTED_CALIBRATION_DIR_ENV = "LBX_TRUSTED_CALIBRATION_DIR"
_REQUIRE_TRUSTED_CONTINUOUS_ENV = "LBX_REQUIRE_TRUSTED_CONTINUOUS_EVALUATION"
_CALIBRATION_EVIDENCE_SCHEMA = "continuous-calibration-evidence.v1"
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


def _ground_truth_evidence(
    problem_dir: Path, task_toml: TaskToml
) -> dict[str, Any] | None:
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


def _calibration_evidence(problem_dir: Path) -> dict[str, Any] | None:
    """Return a non-secret calibration identity for runtime verification."""
    trusted_root = os.environ.get(_TRUSTED_CALIBRATION_DIR_ENV)
    if trusted_root:
        calibration_root = Path(trusted_root)
        lock_path = calibration_root / "calibration.lock.json"
        evidence_path = calibration_root / "calibration.evidence.json"
        trusted = True
    else:
        lock_path = problem_dir / "calibration.lock.json"
        evidence_path = problem_dir / _CALIBRATION_EVIDENCE_PATH
        trusted = False
    if not lock_path.is_file() or not evidence_path.is_file():
        return None
    try:
        lock = json.loads(lock_path.read_text())
        calibration = json.loads(evidence_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    lock_sha = sha256_file(lock_path)
    if (
        not isinstance(calibration, dict)
        or calibration.get("schema_version") != _CALIBRATION_EVIDENCE_SCHEMA
        or calibration.get("lock_sha256") != lock_sha
        or calibration.get("task_spec_sha256") != lock.get("task_spec_sha256")
        or calibration.get("evaluation_plan_sha256")
        != lock.get("evaluation_plan_sha256")
        or calibration.get("security_tier")
        != (lock.get("evaluation_plan") or {}).get("security_tier")
        or calibration.get("inputs") != lock.get("inputs")
        or calibration.get("qualification") != lock.get("qualification")
    ):
        return None
    return {
        "schema_version": 1,
        "lock_sha256": lock_sha,
        "task_spec_sha256": lock.get("task_spec_sha256"),
        "evaluation_plan_sha256": lock.get("evaluation_plan_sha256"),
        "security_tier": (lock.get("evaluation_plan") or {}).get("security_tier"),
        "policy": lock.get("policy"),
        "_trusted": trusted,
    }


def _evaluation_plan_evidence(problem_dir: Path) -> dict[str, Any] | None:
    """Return identity for native policy/custom sealed evaluation plans."""
    from grading.evaluation.plan import validate_serialized_plan

    path = problem_dir / "scorer" / "evaluation.plan.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        plan_sha = validate_serialized_plan(payload)
    except ValueError:
        return None
    return {
        "schema_version": 1,
        "path": "evaluation.plan.json",
        "sha256": sha256_file(path),
        "plan_sha256": plan_sha,
        "security_tier": payload.get("security_tier"),
    }


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
_task = getattr(_module, "TASK", None)
_compute_score = getattr(_module, "compute_score", None)
_takes_args = bool(inspect.signature(_compute_score).parameters) if callable(_compute_score) else False


def compute_score():
    # Declarative rubric tasks are invoked directly. Authors do not own a
    # top-level exception/normalization boundary.
    from grading.evaluation import RubricTask
    if isinstance(_task, RubricTask):
        return _task.grade(
            workspace=pathlib.Path("/tmp/output"),
            trajectory=globals().get("TRANSCRIPT") or [],
            private=pathlib.Path("/mcp_server/data"),
        )
    if not callable(_compute_score):
        raise RuntimeError("grader defines neither TASK=RubricTask(...) nor compute_score()")
    # A no-arg compute_score() reads the baked /tmp/output and /mcp_server/data
    # paths directly; call it with no args.
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
    image_is_outer_capsule: bool = False,
) -> dict[str, Any]:
    """Build the per-problem entry dict that lives under
    ``problems_metadata.problem_set.problems[]``."""
    from alignerr_plugin.preloaded import load_preloaded_manifest

    task_toml = load_task_toml(problem_dir)
    runner: RunnerConfig = task_toml.runner
    capabilities: CapabilityConfig | None = None
    contract_assertion: str | None = None
    if is_capability_task(task_toml):
        capabilities = _resolve_capabilities_for_problem(problem_dir, task_toml)
        contract_assertion = _require_outer_capsule_image(
            image_ref=image_ref,
            image_is_outer_capsule=image_is_outer_capsule,
        )
        _validate_capability_mcp(task_toml)
    # ml tasks are force-pinned to the hour-scale ML_* timeouts; other task types
    # keep their author-set / default values. Capability tasks additionally
    # surface their long build/agent/verifier phase limits. The job-level
    # max_episode is applied separately in _job_level_fields.
    setup_timeout_seconds, grading_timeout_seconds, tool_timeout_seconds, _ = (
        _submission_timeouts(task_toml)
    )
    resources = _derive_resources_from_toml(task_toml)

    startup_command = STARTUP_COMMAND
    prompt = _read_prompt(problem_dir)
    shim = test_file_shim()

    required_resources = validate_required_resources(
        (overrides or {}).get("required_resources", resources["required_resources"])
    )
    prompt = _append_runtime_notices(prompt, required_resources)
    preloaded_manifest = load_preloaded_manifest(problem_dir)
    required_tools = (overrides or {}).get("required_tools", runner.required_tools)
    runtime_override = (overrides or {}).get("container_runtime")
    if capabilities is not None:
        if runtime_override not in (None, "firecracker"):
            raise ValueError(
                "capability-aware Taiga export requires container_runtime="
                f"'firecracker' for nested Docker; got {runtime_override!r}"
            )
        container_runtime = "firecracker"
    else:
        # Derive the Taiga isolation runtime from the resolved resource tier so
        # accelerator tasks deploy under gVisor and CPU tasks under firecracker.
        # The task.toml runner.container_runtime literal is a LOCAL-harness knob.
        container_runtime = runtime_override or _taiga_container_runtime(
            required_resources
        )
    enable_anthropic_api = (overrides or {}).get(
        "enable_anthropic_api", runner.enable_anthropic_api
    )
    outputs = _taiga_outputs(task_toml)
    task_metadata = dict(task_toml.metadata) if task_toml.metadata else {}
    if outputs:
        # Keep task.toml [[outputs]] available to local Taiga-format harness
        # readers and QA tools that inspect extra_fields.task_metadata.
        task_metadata["outputs"] = outputs
    if capabilities is not None:
        assert contract_assertion is not None
        task_metadata["capability_summary"] = _capability_summary(
            task_toml,
            capabilities,
            contract_assertion=contract_assertion,
            selected_required_resources=required_resources,
        )
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
    is_continuous = (
        str(task_toml.difficulty.reward_type) == "continuous_scoring_function"
    )
    is_rubric = str(task_toml.difficulty.reward_type) == "multi_deterministic_rubrics"
    calibration_evidence = _calibration_evidence(problem_dir) if is_continuous else None
    calibration_is_trusted = bool(
        calibration_evidence and calibration_evidence.pop("_trusted", False)
    )
    if (
        is_continuous
        and image_ref != "LOCAL_IMAGE"
        and (
            (
                (problem_dir / "calibration.lock.json").is_file()
                and calibration_evidence is None
            )
            or (calibration_evidence is not None and not calibration_is_trusted)
            or (
                os.environ.get(_TRUSTED_CALIBRATION_DIR_ENV)
                and calibration_evidence is None
            )
        )
    ):
        raise ValueError(
            "trusted continuous calibration evidence is missing or stale; trusted "
            "CI must regenerate the lock/evidence bundle before export"
        )
    if calibration_evidence is not None:
        # Local harness containers may use the baked author lock for faithful
        # iteration. Submitted Taiga versions must overlay it with the
        # trusted-CI promoted mount, which hides the image fallback marker.
        calibration_evidence["requires_trusted_mount"] = image_ref != "LOCAL_IMAGE"
        extra_fields["calibration"] = calibration_evidence
    evaluation_plan_evidence = (
        _evaluation_plan_evidence(problem_dir) if (is_continuous or is_rubric) else None
    )
    if (
        is_continuous
        and image_ref != "LOCAL_IMAGE"
        and os.environ.get(_REQUIRE_TRUSTED_CONTINUOUS_ENV) == "1"
        and calibration_evidence is None
        and evaluation_plan_evidence is None
    ):
        raise ValueError(
            "continuous Taiga export requires trusted calibration or evaluation "
            "plan evidence"
        )
    if evaluation_plan_evidence is not None:
        evaluation_plan_evidence["requires_trusted_mount"] = image_ref != "LOCAL_IMAGE"
        extra_fields["evaluation_plan"] = evaluation_plan_evidence
    if is_rubric and image_ref != "LOCAL_IMAGE" and evaluation_plan_evidence is None:
        raise ValueError(
            "declarative rubric Taiga export requires scorer/evaluation.plan.json"
        )
    if is_continuous:
        security_tier = (calibration_evidence or {}).get("security_tier") or (
            evaluation_plan_evidence or {}
        ).get("security_tier")
        extra_fields["continuous_evaluation"] = {
            "required": True,
            "security_tier": security_tier,
            "attestation_required": image_ref != "LOCAL_IMAGE",
            "trace_required": image_ref != "LOCAL_IMAGE"
            and security_tier in {"sealed_challenge", "sealed_rescore"},
        }
    elif is_rubric:
        security_tier = (evaluation_plan_evidence or {}).get("security_tier")
        extra_fields["rubric_evaluation"] = {
            "required": True,
            "security_tier": security_tier,
            "attestation_required": image_ref != "LOCAL_IMAGE",
            "trace_required": image_ref != "LOCAL_IMAGE"
            and security_tier in {"sealed_challenge", "sealed_rescore"},
        }

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
        "output_directory": (
            task_toml.result.output_root
            if capabilities is not None and task_toml.result is not None
            else "/tmp/output"
        ),
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
    if capabilities is not None:
        workspace = model_dump(task_toml.workspace)
        entry["code_root"] = str(
            workspace.get("agent_cwd") or workspace.get("root") or "/workdir"
        )
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
        payload["serialize_restore_test_interval"] = (
            runner.serialize_restore_test_interval
        )
    if runner.checkpoint_ttl:
        payload["checkpoint_ttl"] = runner.checkpoint_ttl
    if environment_id:
        payload["environment_id"] = environment_id

    return payload


def build_job_payload(
    problem_dir: Path,
    *,
    image_ref: str,
    image_is_outer_capsule: bool = False,
    n_attempts: int | None = None,
    turn_limit: int | None = None,
    max_ctx: int | None = None,
    model: str | None = None,
    environment_id: str | None = None,
) -> dict[str, Any]:
    """Build a single-problem Boreal job payload.

    Trusted export code must explicitly assert that a capability task's
    ``image_ref`` is its built outer capsule. Author metadata is not trust evidence.
    """
    task_toml = load_task_toml(problem_dir)
    runner = task_toml.runner

    problem_entry = _build_problem_entry(
        problem_dir,
        image_ref=image_ref,
        image_is_outer_capsule=image_is_outer_capsule,
    )
    job_fields = _job_level_fields(
        runner,
        n_attempts=n_attempts,
        turn_limit=turn_limit,
        max_ctx=max_ctx,
        model=model,
        environment_id=environment_id,
        max_episode_sec=_submission_timeouts(task_toml)[3],
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
    image_is_outer_capsule: bool = False,
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
                f"image_refs missing entry for problem {pid!r}; supply "
                "the digest-pinned per-task image."
            )
        entries.append(
            _build_problem_entry(
                pd,
                image_ref=image_refs[pid],
                overrides=overrides,
                image_is_outer_capsule=image_is_outer_capsule,
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
        max_episode_sec=_submission_timeouts(first_toml)[3],
    )

    shared_resources = validate_required_resources(
        (overrides or {}).get(
            "required_resources",
            _max_required_resources([e["required_resources"] for e in entries]),
        )
    )
    shared_runtime = (overrides or {}).get(
        "container_runtime"
    ) or _taiga_container_runtime(shared_resources)

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
    image_is_outer_capsule: bool = False,
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
    problem_entry = _build_problem_entry(
        problem_dir,
        image_ref=image_ref,
        image_is_outer_capsule=image_is_outer_capsule,
    )

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
