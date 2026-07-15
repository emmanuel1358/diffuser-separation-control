"""Synthesize a minimal ``metadata.json`` into the internal ``TaskToml`` the rest
of the pipeline consumes. Opt-in per task by ``metadata.json`` shape; everything
not authored is pinned below. See docs/ for the full metadata.json schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from alignerr_plugin.schemas import (
    ML_GRADING_TIMEOUT_SEC,
    ML_MAX_EPISODE_SEC,
    ML_SETUP_TIMEOUT_SEC,
    ML_TOOL_TIMEOUT_SEC,
    ProblemMetadata,
    TaskToml,
)
from alignerr_plugin.task_metadata import metadata_validation_issues
from alignerr_plugin.taiga_resources import (
    is_graphics_resource,
    is_tpu_resource,
    validate_required_resources,
)

REQUIRED_METADATA_KEYS: frozenset[str] = frozenset(
    {"ml_task_type", "required_resources", "domain", "license", "license_source"}
)
OPTIONAL_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "docker-base",
        "dependencies",
        "apt_extras",
        "description",
        "hf_resources",
        "env_dependencies",
        "grading_dependencies",
    }
)
ALLOWED_METADATA_KEYS: frozenset[str] = REQUIRED_METADATA_KEYS | OPTIONAL_METADATA_KEYS

# hf_resources: read-only Hugging Face weight/dataset mounts. Each entry is a
# repo string ("org/name") or an object with these keys.
HF_RESOURCE_KEYS: frozenset[str] = frozenset(
    {"repo_id", "revision", "repo_type", "allow_patterns", "ignore_patterns"}
)
HF_REPO_TYPES: frozenset[str] = frozenset({"model", "dataset"})
# Hub cache lives under <HF_HOME>/hub/<repo_type>s--<org>--<name>.
HF_HUB_CACHE = "/tmp/.cache/huggingface/hub"

# Grading paradigms. dataset = static held-out data; env/hybrid = hidden env
# server; sim_policy = held-out-seed rollout of a submitted policy.
MLENVS_TASK_TYPES: frozenset[str] = frozenset(
    {"dataset", "env", "hybrid", "sim_policy"}
)

# docker-base selector. "default" auto-selects (mlenvs-gpu, or
# mlenvs-cuda-graphics on a "+graphics" tier); "cuda-graphics"/"tpu" force those.
VALID_DOCKER_BASES: frozenset[str] = frozenset({"default", "cuda-graphics", "tpu"})
DEFAULT_DOCKER_BASE = "default"

# --- Pinned constants ------------------------------------------------------

TASK_NAME_PREFIX = "labelbox"

# api_model_name is left to the schema default (the submit pipeline obscures the
# model via TAIGA_MODEL_NAME); never hardcode a concrete model id here.
PINNED_RUNNER: dict = {
    "attempts": 3,
    "turn_limit": 1500,
    "max_ctx": 1_000_000,
    "context_mode": "none",
    "required_tools": ["bash", "str_replace_editor", "tmux"],
}

# ML_Envs pins the same hour-scale ML timeouts the Taiga exporter force-pins for
# native ml tasks. Reference the shared schemas.ML_*_TIMEOUT_SEC constants (a
# single source of truth) so ML_Envs and native ml tasks can never drift apart;
# grading_sec is Taiga's maximum and the Taiga exporter independently pins ml
# grading to the same value.
PINNED_TIMEOUTS: dict = {
    "setup_sec": ML_SETUP_TIMEOUT_SEC,
    "grading_sec": ML_GRADING_TIMEOUT_SEC,
    "tool_sec": ML_TOOL_TIMEOUT_SEC,
    "max_episode_sec": ML_MAX_EPISODE_SEC,
}

PINNED_CONTINUOUS_SCORE_EPSILON = 0.05


class MlEnvsMetadataError(ValueError):
    """Raised when a ``metadata.json`` is malformed."""


def is_mlenvs_task(problem_dir: Path) -> bool:
    """Return whether ``problem_dir`` opts into ML_Envs mode.

    True iff it ships no ``task.toml`` and carries a marker: a ``test_file.py``
    grader, or a ``metadata.json`` declaring ``ml_task_type``. Any parse problem
    in ``metadata.json`` falls back to the marker-file checks.
    """
    if (problem_dir / "task.toml").is_file():
        return False
    if (problem_dir / "test_file.py").is_file():
        return True
    path = problem_dir / "metadata.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and "ml_task_type" in data


def _read_raw_metadata(problem_dir: Path) -> dict:
    path = problem_dir / "metadata.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise MlEnvsMetadataError(f"metadata.json missing at {path}") from exc
    except json.JSONDecodeError as exc:
        raise MlEnvsMetadataError(f"metadata.json at {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise MlEnvsMetadataError(f"metadata.json at {path} must be a JSON object")
    return data


def load_mlenvs_metadata(problem_dir: Path) -> dict:
    """Parse + fully validate a ``metadata.json``, collecting all issues at once."""
    data = _read_raw_metadata(problem_dir)
    issues: list[str] = []

    unknown = sorted(set(data) - ALLOWED_METADATA_KEYS)
    if unknown:
        issues.append(
            f"unknown keys {unknown}; allowed: {sorted(ALLOWED_METADATA_KEYS)}"
        )
    missing = sorted(REQUIRED_METADATA_KEYS - set(data))
    if missing:
        issues.append(f"missing required keys {missing}")

    task_type = data.get("ml_task_type")
    if "ml_task_type" in data and task_type not in MLENVS_TASK_TYPES:
        issues.append(
            f"ml_task_type must be one of {sorted(MLENVS_TASK_TYPES)}; got {task_type!r}"
        )

    required_resources = data.get("required_resources")
    if "required_resources" in data:
        try:
            required_resources = validate_required_resources(required_resources)
        except ValueError as exc:
            issues.append(str(exc))
            required_resources = None

    docker_base = data.get("docker-base", DEFAULT_DOCKER_BASE)
    if docker_base not in VALID_DOCKER_BASES:
        issues.append(
            f"docker-base must be one of {sorted(VALID_DOCKER_BASES)}; got {docker_base!r}"
        )

    for key in ("dependencies", "apt_extras", "env_dependencies", "grading_dependencies"):
        if key in data:
            value = data[key]
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                issues.append(f"{key} must be a list of non-empty strings")

    issues.extend(_env_dependencies_issues(data, task_type))
    issues.extend(_grading_dependencies_issues(data))

    if "description" in data and not isinstance(data["description"], str):
        issues.append("description must be a string")

    issues.extend(_hf_resources_issues(data, docker_base, required_resources))

    # Cross-field base/resource consistency, only when the enum itself validated.
    if isinstance(required_resources, str):
        if docker_base == "tpu" and not is_tpu_resource(required_resources):
            issues.append(
                "docker-base='tpu' requires a TPU required_resources tier "
                f"(e.g. 13vcpu+32gib+tpuv5e1x1); got {required_resources!r}"
            )
        if docker_base == "cuda-graphics" and not is_graphics_resource(required_resources):
            issues.append(
                "docker-base='cuda-graphics' requires a '+graphics' required_resources "
                f"tier in this template; got {required_resources!r} (lbx reaches the "
                "cuda-graphics base via a +graphics enum)"
            )
        if is_tpu_resource(required_resources) and task_type in {"env", "hybrid"}:
            issues.append(
                f"ml_task_type={task_type!r} (hidden env server) cannot run on a TPU "
                "tier; use an H100 tier for env/hybrid tasks"
            )

    # Classification fields via the shared validator, re-homed to metadata.json.
    classification_issues = metadata_validation_issues(
        task_type="ml",
        domain=data.get("domain", ""),
        reward_type="continuous_scoring_function",
        license_id=data.get("license", ""),
        license_source=data.get("license_source", ""),
    )
    for issue in classification_issues:
        issues.append(issue.replace("task.toml [difficulty].", "").strip())

    if issues:
        rel = problem_dir.name
        raise MlEnvsMetadataError(
            f"metadata.json ({rel}) is invalid:\n  - " + "\n  - ".join(issues)
        )

    return data


def mlenvs_task_id(problem_dir: Path) -> str:
    """Problem id from the directory name (one trailing ``_taiga`` stripped)."""
    name = problem_dir.resolve().name
    if name.endswith("_taiga"):
        name = name[: -len("_taiga")]
    if not name:
        raise MlEnvsMetadataError(
            f"cannot derive a problem id from directory name {problem_dir.name!r}"
        )
    return name


def base_flavor_for(docker_base: str, required_resources: str) -> str:
    """Map ``docker-base`` + the resource tier to an mlenvs-* base flavor."""
    docker_base = docker_base or DEFAULT_DOCKER_BASE
    # A TPU or "+graphics" tier routes to its base regardless of docker-base;
    # docker-base still lets an author force cuda-graphics/tpu explicitly.
    if docker_base == "tpu" or is_tpu_resource(required_resources):
        return "mlenvs-tpu"
    if docker_base == "cuda-graphics" or is_graphics_resource(required_resources):
        return "mlenvs-cuda-graphics"
    return "mlenvs-gpu"


def _pkg_name(spec: str) -> str:
    """Normalized distribution name from a pip requirement spec (``foo[all]==1`` -> ``foo``)."""
    name = spec.strip().lower()
    for sep in ("[", "<", ">", "=", "!", "~", " ", ";", "@"):
        name = name.split(sep, 1)[0]
    return name.strip().replace("_", "-")


# A normalized distribution name (post-_pkg_name). A VCS URL (`git+https://...`)
# or local path (`./pkg`, `/abs/pkg`) leaves URL/path artifacts (`:`, `/`, `+`)
# that this rejects -- so such specs are "un-analyzable" for the overlap check.
_VALID_PKG_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _dep_name_analyzable(spec: str) -> bool:
    """True if ``spec``'s distribution name can be resolved for the overlap check.

    A PEP 508 direct reference (``name @ git+https://...``) IS analyzable --
    ``_pkg_name`` yields the ``name`` before ``@``. A BARE VCS URL / local path
    (no ``name @`` prefix) is NOT: it has no resolvable name, so the same package
    spelled as a URL in one list and a plain name in the other would slip past the
    env/agent isolation overlap check.
    """
    return bool(_VALID_PKG_NAME_RE.match(_pkg_name(spec)))


def _env_dependencies_issues(data: dict, task_type: object) -> list[str]:
    """Cross-field checks for ``env_dependencies`` (env-server-only pip deps)."""
    if "env_dependencies" not in data:
        return []
    env_deps = data["env_dependencies"]
    if not isinstance(env_deps, list):
        return []  # shape error already reported by the list-of-strings check
    issues: list[str] = []
    if task_type not in {"env", "hybrid"}:
        issues.append(
            "env_dependencies is only used by env/hybrid tasks (the hidden env "
            f"server); ml_task_type={task_type!r} has no env server. Put "
            "agent-visible packages in 'dependencies' instead."
        )
    agent_specs = [s for s in data.get("dependencies", []) or [] if isinstance(s, str)]
    env_specs = [s for s in env_deps if isinstance(s, str)]
    agent_pkgs = {_pkg_name(s) for s in agent_specs}
    overlap = sorted({_pkg_name(s) for s in env_specs} & agent_pkgs)
    if overlap:
        issues.append(
            f"packages {overlap} appear in both 'dependencies' (agent-visible) and "
            "'env_dependencies' (env-only); a 'dependencies' entry is importable by "
            "the agent, defeating the env_dependencies isolation. Put each package in "
            "exactly one list."
        )
    # A bare VCS URL / local path has no resolvable name, so the overlap check
    # above cannot see it -- the same package as a URL in one list and a plain
    # name in the other would silently reach the agent. Require an analyzable
    # spelling (a named PyPI spec, or the PEP 508 `name @ url` form).
    unanalyzable = sorted(
        {s.strip() for s in (env_specs + agent_specs) if s.strip() and not _dep_name_analyzable(s)}
    )
    if unanalyzable:
        issues.append(
            f"dependency spec(s) {unanalyzable} are a VCS URL or local path with no "
            "'name @ url' prefix, so the env/agent isolation overlap check cannot "
            "resolve their package name. Use a named PyPI spec (or the PEP 508 "
            "'name @ url' form) so the same package cannot appear in both lists under "
            "different spellings and silently reach the agent."
        )
    return issues


def _grading_dependencies_issues(data: dict) -> list[str]:
    """Cross-field checks for ``grading_dependencies`` (grader-only pip deps).

    Installed root-only under ``/mcp_server/grading_deps`` and prepended to the
    grader worker's ``sys.path`` before ``compute_score`` loads, so the (root)
    grader can import them but the uid-1000 agent cannot (0700 ``/mcp_server``
    blocks it, exactly as for the held-out truth). Unlike ``env_dependencies``,
    allowed for EVERY task type -- a dataset grader has no other hidden-dep
    channel. Use it for a scoring/reference library that would leak the intended
    approach if the agent could see it in ``dependencies``.
    """
    if "grading_dependencies" not in data:
        return []
    grading_deps = data["grading_dependencies"]
    if not isinstance(grading_deps, list):
        return []  # shape error already reported by the list-of-strings check
    issues: list[str] = []
    grading_specs = [s for s in grading_deps if isinstance(s, str)]
    agent_specs = [s for s in data.get("dependencies", []) or [] if isinstance(s, str)]
    agent_pkgs = {_pkg_name(s) for s in agent_specs}
    overlap = sorted({_pkg_name(s) for s in grading_specs} & agent_pkgs)
    if overlap:
        issues.append(
            f"packages {overlap} appear in both 'dependencies' (agent-visible) and "
            "'grading_dependencies' (grader-only); a 'dependencies' entry is importable "
            "by the agent, defeating the grading_dependencies isolation. Put each "
            "package in exactly one list."
        )
    unanalyzable = sorted(
        {s.strip() for s in grading_specs if s.strip() and not _dep_name_analyzable(s)}
    )
    if unanalyzable:
        issues.append(
            f"grading_dependencies spec(s) {unanalyzable} are a VCS URL or local path "
            "with no 'name @ url' prefix, so the isolation overlap check cannot resolve "
            "their package name. Use a named PyPI spec (or the PEP 508 'name @ url' form)."
        )
    return issues


def hidden_env_for_task_type(task_type: str) -> str:
    """Map ``ml_task_type`` to ``[environment].hidden_env`` (env/hybrid on, else off)."""
    return task_type if task_type in {"env", "hybrid"} else ""


# A Hugging Face repo_id is `namespace/name`: exactly one slash, each component
# starting alphanumeric and otherwise `[A-Za-z0-9._-]`. Stronger than a bare
# "/"-in-string check, matching the packer's deploy-time repo_id validation so a
# malformed id fails at authoring time instead of only fail-closed at deploy.
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _valid_hf_repo_id(rid: object) -> bool:
    return isinstance(rid, str) and bool(_HF_REPO_ID_RE.match(rid.strip()))


def _hf_resources_issues(data: dict, docker_base: str, required_resources: object) -> list[str]:
    """Validate ``metadata.json:hf_resources``."""
    if "hf_resources" not in data:
        return []
    resources = data["hf_resources"]
    if not isinstance(resources, list):
        return ["hf_resources must be a list"]
    issues: list[str] = []
    # HF mounts are not supported on the TPU base.
    if docker_base == "tpu" or (
        isinstance(required_resources, str) and is_tpu_resource(required_resources)
    ):
        issues.append(
            "hf_resources is not supported with a TPU base (docker-base='tpu' / TPU tier)"
        )
    for i, entry in enumerate(resources):
        where = f"hf_resources[{i}]"
        if isinstance(entry, str):
            if not _valid_hf_repo_id(entry):
                issues.append(f"{where}: repo string must look like 'org/name'; got {entry!r}")
            continue
        if not isinstance(entry, dict):
            issues.append(f"{where}: must be a repo string or an object with repo_id")
            continue
        unknown = sorted(set(entry) - HF_RESOURCE_KEYS)
        if unknown:
            issues.append(f"{where}: unknown keys {unknown}; allowed {sorted(HF_RESOURCE_KEYS)}")
        rid = entry.get("repo_id")
        if not _valid_hf_repo_id(rid):
            issues.append(f"{where}: repo_id must be a string like 'org/name'; got {rid!r}")
        if entry.get("repo_type", "model") not in HF_REPO_TYPES:
            issues.append(
                f"{where}: repo_type must be one of {sorted(HF_REPO_TYPES)}; "
                f"got {entry.get('repo_type')!r}"
            )
        for pat_key in ("allow_patterns", "ignore_patterns"):
            if pat_key in entry and not (
                isinstance(entry[pat_key], list)
                and all(isinstance(p, str) for p in entry[pat_key])
            ):
                issues.append(f"{where}: {pat_key} must be a list of strings")
    return issues


def hf_resources_to_preloaded(hf_resources: list | None) -> list[dict]:
    """Map ``metadata.json:hf_resources`` -> ``[[preloaded_files]]`` read-only HF mounts."""
    entries: list[dict] = []
    for entry in hf_resources or []:
        if isinstance(entry, str):
            repo_id, revision, repo_type = entry, "", "model"
        else:
            repo_id = entry.get("repo_id", "")
            revision = entry.get("revision", "") or ""
            repo_type = entry.get("repo_type", "model")
        folder = f"{repo_type}s--" + repo_id.replace("/", "--")
        entries.append(
            {
                "hf_repo": repo_id,
                "hf_revision": revision,
                "mount_path": f"{HF_HUB_CACHE}/{folder}",
                "read_only": True,
            }
        )
    return entries


def synthesize_task_toml(problem_dir: Path) -> TaskToml:
    """Expand a ``metadata.json`` into the internal ``TaskToml``."""
    meta = load_mlenvs_metadata(problem_dir)
    task_id_value = mlenvs_task_id(problem_dir)
    docker_base = meta.get("docker-base", DEFAULT_DOCKER_BASE)

    toml_dict: dict = {
        "schema_version": "1.1",
        "task": {
            "name": f"{TASK_NAME_PREFIX}/{task_id_value}",
            "description": meta.get("description", ""),
        },
        "environment": {
            "required_resources": meta["required_resources"],
            "allow_internet": False,
            "base_flavor": base_flavor_for(docker_base, meta["required_resources"]),
            "hidden_env": hidden_env_for_task_type(meta["ml_task_type"]),
        },
        "agent": {"timeout_sec": PINNED_TIMEOUTS["max_episode_sec"]},
        "verifier": {"timeout_sec": PINNED_TIMEOUTS["grading_sec"], "env": []},
        "ground_truth": {
            "continuous_score_epsilon": PINNED_CONTINUOUS_SCORE_EPSILON,
        },
        "runner": {
            **PINNED_RUNNER,
            "timeouts": dict(PINNED_TIMEOUTS),
        },
        "difficulty": {
            "task_type": "ml",
            "domain": meta["domain"],
            "reward_type": "continuous_scoring_function",
            "license": meta["license"],
            "license_source": meta["license_source"],
        },
        # Build-only selectors (docker-base, dependencies, apt_extras) are
        # intentionally NOT placed here -- they must not leak to the agent-facing payload.
        "metadata": {"ml_task_type": meta["ml_task_type"]},
        "preloaded_files": hf_resources_to_preloaded(meta.get("hf_resources", [])),
    }
    return TaskToml.model_validate(toml_dict)


def synthesize_problem_metadata(problem_dir: Path) -> ProblemMetadata:
    """Build the Alignerr ``metadata.json`` envelope (benchmark + instance_id)."""
    task_id_value = mlenvs_task_id(problem_dir)
    return ProblemMetadata(
        benchmark="taiga_task",
        problem_data={"instance_id": task_id_value, "description": ""},
    )
