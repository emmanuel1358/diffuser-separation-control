"""Dependency-free parsing for optional nested-service task capabilities."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from grading.faults import InfrastructureFault
from rubric.capsule_runtime import validate_service_name

_MAX_TASK_TOML_BYTES = 2 * 1024 * 1024
_DEFAULT_CAPSULE_DIR = Path("/task/capsule")
_DEFAULT_STATE_ROOT = Path("/run/lbx-task-service-runtime")
_DEFAULT_SEALED_ROOT = Path("/mcp_server/service-artifacts")
_PROTECTED_HOST_PATHS = (
    Path("/"),
    Path("/dev"),
    Path("/proc"),
    Path("/sys"),
    Path("/var/run"),
    Path("/run/docker.sock"),
    Path("/mcp_server/data"),
    Path("/mcp_server/grader"),
    Path("/mcp_server/calibration"),
    Path("/runtime/grading"),
    Path("/tmp/output"),
    Path("/workdir"),
)
_TASK_SELECTED_RUNTIME_PATH_KEYS = {
    "capsule_dir",
    "capsule_directory",
    "state_dir",
    "runtime_dir",
    "sealed_dir",
    "artifact_dir",
    "capture_dir",
}
_DEFAULT_RESULT_PATHS = (
    "/logs/verifier/reward.json",
    "/logs/verifier/reward.txt",
)
_ROLE_ALIASES = {
    "agent": "main",
    "main": "main",
    "sidecar": "sidecar",
    "service": "sidecar",
    "init": "init",
    "one-shot": "init",
    "oneshot": "init",
    "verifier": "verifier",
    "grader": "verifier",
}
_PROJECT_RE = re.compile(r"[^a-z0-9_-]+")
_GIT_REF_RE = re.compile(r"^[0-9a-f]{40}$")
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


class ServiceConfigurationError(InfrastructureFault):
    """The trusted service capability configuration is invalid."""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _first_string(mapping: Mapping[str, Any], *names: str) -> str | None:
    value = _first(mapping, *names)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _rows(value: Any) -> list[tuple[str | None, Mapping[str, Any]]]:
    if isinstance(value, list):
        return [(None, row) for row in value if isinstance(row, Mapping)]
    if isinstance(value, Mapping):
        if any(key in value for key in ("name", "service", "command", "source")):
            return [(None, value)]
        return [
            (str(name), row)
            for name, row in value.items()
            if isinstance(name, str) and isinstance(row, Mapping)
        ]
    return []


def _positive_float(
    value: Any,
    *,
    default: float,
    label: str,
    maximum: float = 24 * 60 * 60,
) -> float:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ServiceConfigurationError(f"{label} must be a number") from exc
    if not 0 < result <= maximum:
        raise ServiceConfigurationError(
            f"{label} must be greater than zero and at most {maximum:g}"
        )
    return result


def _positive_int(value: Any, *, default: int, label: str, maximum: int) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise ServiceConfigurationError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ServiceConfigurationError(f"{label} must be an integer") from exc
    if not 0 < result <= maximum:
        raise ServiceConfigurationError(
            f"{label} must be greater than zero and at most {maximum}"
        )
    return result


def _absolute_host_path(value: Any, *, default: Path, label: str) -> Path:
    raw = str(value).strip() if value not in (None, "") else str(default)
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise ServiceConfigurationError(f"{label} must be an absolute normalized path")
    return path


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _reject_symlink_ancestors(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                raise ServiceConfigurationError(
                    f"{label} has a symlink ancestor: {current}"
                )
            if not current.exists():
                break
        except OSError as exc:
            raise ServiceConfigurationError(
                f"could not inspect {label} ancestor {current}: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class RuntimeOperatorRoots:
    """Trusted roots selected by the outer runtime, never by ``task.toml``."""

    capsule: Path = _DEFAULT_CAPSULE_DIR
    state: Path = _DEFAULT_STATE_ROOT
    sealed: Path = _DEFAULT_SEALED_ROOT

    def __post_init__(self) -> None:
        roots = {
            "capsule root": self.capsule,
            "state root": self.state,
            "sealed root": self.sealed,
        }
        for label, path in roots.items():
            if not path.is_absolute() or ".." in path.parts or path == Path("/"):
                raise ServiceConfigurationError(
                    f"{label} must be an absolute non-root normalized path"
                )
            _reject_symlink_ancestors(path, label=label)
            for protected in _PROTECTED_HOST_PATHS[1:]:
                if _paths_overlap(path, protected):
                    raise ServiceConfigurationError(
                        f"{label} overlaps protected host path {protected}"
                    )
        if len(set(roots.values())) != len(roots):
            raise ServiceConfigurationError(
                "capsule, state, and sealed operator roots must be distinct"
            )


DEFAULT_OPERATOR_ROOTS = RuntimeOperatorRoots()


def _absolute_container_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceConfigurationError(f"{label} must be a non-empty absolute path")
    path = PurePosixPath(value.strip())
    if not path.is_absolute() or ".." in path.parts:
        raise ServiceConfigurationError(f"{label} must be an absolute normalized path")
    return str(path)


def _safe_verifier_result_path(value: Any, *, label: str) -> str:
    result = _absolute_container_path(value, label=label)
    path = PurePosixPath(result)
    protected = tuple(
        PurePosixPath(item)
        for item in (
            "/bin",
            "/boot",
            "/dev",
            "/etc",
            "/mcp_server",
            "/proc",
            "/run",
            "/sbin",
            "/sys",
            "/task",
            "/usr",
            "/var/run",
        )
    )
    if path == PurePosixPath("/") or any(
        path == root or root in path.parents for root in protected
    ):
        raise ServiceConfigurationError(f"{label} uses a protected container path")
    return result


def _relative_destination(value: Any, *, default: str, label: str) -> str:
    raw = str(value).strip() if value not in (None, "") else default
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or raw in ("", "."):
        raise ServiceConfigurationError(
            f"{label} must be a non-empty relative path without '..'"
        )
    if path.parts[0] == "manifest.json":
        raise ServiceConfigurationError(f"{label} shadows reserved manifest.json")
    return str(path)


def _relative_task_path(value: Any, *, label: str, allow_dot: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceConfigurationError(f"{label} must be a relative task path")
    raw = value.strip()
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or (not allow_dot and raw in {"", "."}):
        raise ServiceConfigurationError(
            f"{label} must be a normalized relative task path"
        )
    return str(path)


def _normalize_user(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ServiceConfigurationError("service user must be a username, UID, or UID:GID")


def _command_text(value: Any, *, label: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item for item in value)
    ):
        return shlex.join(value)
    raise ServiceConfigurationError(f"{label} must be a command string or argv list")


def _validated_http_url(
    value: str | None,
    *,
    label: str,
    allowed_hosts: set[str],
    require_port: bool,
) -> str:
    if value is None:
        raise ServiceConfigurationError(f"{label} is required")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ServiceConfigurationError(f"{label} is invalid: {exc}") from exc
    if parsed.scheme != "http":
        raise ServiceConfigurationError(f"{label} must use http")
    if (
        parsed.hostname not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.query
    ):
        raise ServiceConfigurationError(
            f"{label} must target declared service DNS without credentials, "
            "query parameters, or fragments"
        )
    if require_port and port is None:
        raise ServiceConfigurationError(f"{label} must declare an explicit port")
    if port is not None and not 1 <= port <= 65535:
        raise ServiceConfigurationError(f"{label} port is out of range")
    if not parsed.path.startswith("/"):
        raise ServiceConfigurationError(f"{label} must contain an absolute path")
    return value


def _require_unprivileged_user(value: str) -> str:
    username = value.strip().lower()
    uid = username.split(":", 1)[0]
    numeric_root = uid.isdecimal() and int(uid, 10) == 0
    if username == "root" or numeric_root:
        raise ServiceConfigurationError(
            "main service command proxy requires a non-root configured user"
        )
    return value


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """Runtime-relevant fields for one Compose service."""

    name: str
    role: str
    user: str | None = None
    workdir: str | None = None
    shell: str = "/bin/sh"


@dataclass(frozen=True, slots=True)
class CaptureHook:
    """One ordered pre-verification command."""

    service: str
    command: str
    timeout_s: float
    user: str | None
    accepted_exit_codes: tuple[int, ...]
    atomic_destination: str | None
    failure_policy: str


@dataclass(frozen=True, slots=True)
class WorkspaceRuntimeSpec:
    """Workspace initialization semantics enforced inside the main service."""

    seed: str | None
    root: str
    agent_cwd: str
    init_policy: str
    git_baseline: bool | str
    clean_paths: tuple[str, ...]
    checkpoint_restore: bool


@dataclass(frozen=True, slots=True)
class ServiceArtifact:
    """A declared path copied out of a live service."""

    kind: str
    source: str
    service: str
    destination: str
    required: bool
    exclude: tuple[str, ...]
    max_bytes: int
    max_files: int
    max_depth: int
    preserve_mode: bool


@dataclass(frozen=True, slots=True)
class ToolEndpoint:
    """A declared task-local tool endpoint and its readiness contract."""

    name: str
    transport: str
    service: str
    url: str | None
    command: tuple[str, ...]
    readiness_kind: str | None
    readiness_service: str
    readiness_command: str | None
    readiness_url: str | None
    readiness_host: str | None
    readiness_port: int | None
    readiness_timeout_s: float
    readiness_interval_s: float


@dataclass(frozen=True, slots=True)
class TaskServiceConfig:
    """Complete runtime configuration derived from ``/task/task.toml``."""

    task_toml_path: Path
    operator_roots: RuntimeOperatorRoots
    capsule_dir: Path
    capsule_manifest: str
    compose_file: str | None
    state_dir: Path
    sealed_dir: Path
    project_name: str
    services: tuple[ServiceSpec, ...]
    main_service: str
    verifier_service: str | None
    agent_user: str
    agent_workdir: str
    agent_shell: str
    startup_timeout_s: float
    daemon_timeout_s: float
    command_timeout_s: float
    verifier_timeout_s: float
    max_output_bytes: int
    editor_max_bytes: int
    workspace: WorkspaceRuntimeSpec | None
    captures: tuple[CaptureHook, ...]
    artifacts: tuple[ServiceArtifact, ...]
    tools: tuple[ToolEndpoint, ...]
    verifier_result_paths: tuple[str, ...]
    verifier_reward_path: str
    primary_reward: str
    subscores_key: str | None

    def __post_init__(self) -> None:
        reward_key = (
            self.primary_reward.strip() if isinstance(self.primary_reward, str) else ""
        )
        object.__setattr__(self, "primary_reward", reward_key or "score")

    @property
    def agent_services(self) -> tuple[str, ...]:
        return tuple(
            service.name for service in self.services if service.role != "verifier"
        )

    @property
    def sidecar_services(self) -> tuple[str, ...]:
        return tuple(
            service.name
            for service in self.services
            if service.role in {"sidecar", "init"}
        )

    def service(self, name: str) -> ServiceSpec:
        for service in self.services:
            if service.name == name:
                return service
        raise ServiceConfigurationError(f"unknown configured service {name!r}")


def _runtime_table(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = _as_mapping(payload.get("service_runtime"))
    if direct:
        return direct
    runtime = _as_mapping(payload.get("runtime"))
    for key in ("service", "services", "nested_docker", "task_capsule"):
        nested = _as_mapping(runtime.get(key))
        if nested:
            return nested
    capabilities = _as_mapping(payload.get("capabilities"))
    for key in ("service_runtime", "services", "nested_docker", "task_capsule"):
        nested = _as_mapping(capabilities.get(key))
        if nested:
            return nested
    return {}


def _capability_declared(
    payload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    service_rows: list[tuple[str | None, Mapping[str, Any]]],
) -> bool:
    if runtime:
        return runtime.get("enabled", True) is not False
    if service_rows:
        return True
    if any(
        name in payload
        for name in (
            "workspace",
            "artifacts",
            "captures",
            "mcp_servers",
            "evaluation",
            "gates",
            "reports",
            "result",
        )
    ):
        return True
    tools = payload.get("tools")
    if isinstance(tools, Mapping) and "mcp_servers" in tools:
        return True
    capabilities = payload.get("capabilities")
    if isinstance(capabilities, list):
        names = {str(value).strip().lower().replace("-", "_") for value in capabilities}
        return bool(
            names
            & {
                "artifacts",
                "captures",
                "evaluation",
                "gates",
                "mcp_servers",
                "nested_docker",
                "reports",
                "result",
                "services",
                "service_runtime",
                "task_capsule",
                "workspace",
            }
        )
    if isinstance(capabilities, Mapping):
        for name in (
            "artifacts",
            "captures",
            "evaluation",
            "gates",
            "mcp_servers",
            "nested_docker",
            "reports",
            "result",
            "services",
            "service_runtime",
            "task_capsule",
            "workspace",
        ):
            if capabilities.get(name) is True:
                return True
    return False


def _capsule_service_rows(
    capsule_dir: Path,
    manifest_name: str,
) -> list[tuple[str | None, Mapping[str, Any]]]:
    relative = PurePosixPath(manifest_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ServiceConfigurationError(
            "capsule manifest must be a relative normalized path"
        )
    manifest_path = capsule_dir.joinpath(*relative.parts)
    if not manifest_path.exists():
        return []
    _reject_symlink_ancestors(manifest_path, label="capsule manifest")
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > _MAX_TASK_TOML_BYTES
    ):
        raise ServiceConfigurationError(
            f"capsule manifest is not a bounded regular file: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceConfigurationError(
            f"could not parse capsule manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ServiceConfigurationError("capsule manifest must contain an object")
    raw_rows = payload.get("images")
    if raw_rows is None:
        raw_rows = payload.get("services")
    rows = _rows(raw_rows)
    projected: list[tuple[str | None, Mapping[str, Any]]] = []
    for keyed_name, row in rows:
        name = keyed_name or _first_string(row, "service", "name")
        if name is None:
            raise ServiceConfigurationError(
                "capsule manifest service is missing a name"
            )
        role = _first_string(row, "role") or "sidecar"
        projected.append((name, {"name": name, "role": role}))
    return projected


def _parse_services(
    service_rows: list[tuple[str | None, Mapping[str, Any]]],
    runtime: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[tuple[ServiceSpec, ...], str, str | None]:
    services: list[ServiceSpec] = []
    seen: set[str] = set()
    configured_main = _first_string(runtime, "main_service", "main")
    verifier_table = _as_mapping(payload.get("verifier"))
    configured_verifier = _first_string(runtime, "verifier_service") or _first_string(
        verifier_table, "service", "service_name"
    )

    for keyed_name, row in service_rows:
        name = keyed_name or _first_string(row, "name", "service")
        if name is None:
            raise ServiceConfigurationError("each service requires a name")
        name = validate_service_name(name)
        if name in seen:
            raise ServiceConfigurationError(f"duplicate service {name!r}")
        seen.add(name)
        raw_role = (_first_string(row, "role", "kind", "type") or "sidecar").lower()
        try:
            role = _ROLE_ALIASES[raw_role]
        except KeyError as exc:
            raise ServiceConfigurationError(
                f"service {name!r} has unsupported role {raw_role!r}"
            ) from exc
        if configured_main == name:
            role = "main"
        if configured_verifier == name:
            role = "verifier"
        workdir_raw = _first(row, "workdir", "working_dir", "agent_cwd")
        workdir = (
            _absolute_container_path(workdir_raw, label=f"workdir for service {name!r}")
            if workdir_raw not in (None, "")
            else None
        )
        shell = _first_string(row, "shell") or (
            "/bin/bash" if role == "main" else "/bin/sh"
        )
        if not PurePosixPath(shell).is_absolute() or ".." in PurePosixPath(shell).parts:
            raise ServiceConfigurationError(
                f"shell for service {name!r} must be an absolute path"
            )
        services.append(
            ServiceSpec(
                name=name,
                role=role,
                user=_normalize_user(_first(row, "user", "run_as")),
                workdir=workdir,
                shell=shell,
            )
        )

    main_services = [service.name for service in services if service.role == "main"]
    if configured_main and configured_main not in seen:
        raise ServiceConfigurationError(
            f"configured main service {configured_main!r} is not declared"
        )
    if not main_services and "main" in seen:
        main_services = ["main"]
        services = [
            ServiceSpec(
                name=service.name,
                role="main" if service.name == "main" else service.role,
                user=service.user,
                workdir=service.workdir,
                shell=service.shell,
            )
            for service in services
        ]
    if len(main_services) != 1:
        raise ServiceConfigurationError(
            "nested service runtime requires exactly one main service"
        )
    verifier_services = [
        service.name for service in services if service.role == "verifier"
    ]
    if configured_verifier and configured_verifier not in seen:
        raise ServiceConfigurationError(
            f"configured verifier service {configured_verifier!r} is not declared"
        )
    if len(verifier_services) > 1:
        raise ServiceConfigurationError(
            "nested service runtime supports at most one verifier service"
        )
    return (
        tuple(services),
        main_services[0],
        verifier_services[0] if verifier_services else None,
    )


def _parse_captures(
    payload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    main_service: str,
    known_services: set[str],
) -> tuple[CaptureHook, ...]:
    raw = _first(payload, "captures", "capture_hooks")
    if raw is None:
        raw = runtime.get("captures")
    if raw is None:
        raw = _as_mapping(payload.get("verifier")).get("collect")
    hooks: list[CaptureHook] = []
    for _key, row in _rows(raw):
        service = validate_service_name(_first_string(row, "service") or main_service)
        if service not in known_services:
            raise ServiceConfigurationError(
                f"capture hook targets undeclared service {service!r}"
            )
        command = _command_text(_first(row, "command"), label="capture hook command")
        raw_codes = _first(row, "accepted_exit_codes", "success_exit_codes")
        if raw_codes is None:
            codes = (0,)
        elif isinstance(raw_codes, list) and raw_codes:
            try:
                codes = tuple(int(code) for code in raw_codes)
            except (TypeError, ValueError) as exc:
                raise ServiceConfigurationError(
                    "capture accepted_exit_codes must contain integers"
                ) from exc
        else:
            raise ServiceConfigurationError(
                "capture accepted_exit_codes must be a non-empty list"
            )
        policy = (
            _first_string(row, "failure_policy", "on_failure") or "infrastructure"
        ).lower()
        if policy not in {"infrastructure", "agent"}:
            raise ServiceConfigurationError(
                "capture failure_policy must be 'infrastructure' or 'agent'"
            )
        atomic_raw = _first(row, "atomic_destination", "atomic_path")
        atomic_destination = (
            _absolute_container_path(atomic_raw, label="capture atomic_destination")
            if atomic_raw not in (None, "")
            else None
        )
        hooks.append(
            CaptureHook(
                service=service,
                command=command,
                timeout_s=_positive_float(
                    _first(row, "timeout_s", "timeout_sec"),
                    default=60.0,
                    label="capture timeout",
                    maximum=6 * 60 * 60,
                ),
                user=_normalize_user(_first(row, "user")),
                accepted_exit_codes=codes,
                atomic_destination=atomic_destination,
                failure_policy=policy,
            )
        )
    return tuple(hooks)


def _parse_workspace(payload: Mapping[str, Any]) -> WorkspaceRuntimeSpec | None:
    raw_workspace = payload.get("workspace")
    if raw_workspace is None:
        return None
    if not isinstance(raw_workspace, Mapping):
        raise ServiceConfigurationError("[workspace] must be a TOML table")
    root = _absolute_container_path(raw_workspace.get("root"), label="workspace root")
    agent_cwd = _absolute_container_path(
        raw_workspace.get("agent_cwd", root), label="workspace agent_cwd"
    )
    root_path = PurePosixPath(root)
    cwd_path = PurePosixPath(agent_cwd)
    if cwd_path != root_path and root_path not in cwd_path.parents:
        raise ServiceConfigurationError("workspace agent_cwd must be inside root")
    init_policy = str(raw_workspace.get("init_policy", "copy")).strip().lower()
    if init_policy not in {"copy", "overlay", "reuse", "empty"}:
        raise ServiceConfigurationError(
            "workspace init_policy must be copy, overlay, reuse, or empty"
        )
    seed_raw = raw_workspace.get("seed")
    seed = (
        _relative_task_path(seed_raw, label="workspace seed", allow_dot=True)
        if seed_raw not in (None, "")
        else None
    )
    if seed is None and init_policy != "empty":
        raise ServiceConfigurationError(
            "workspace seed is required unless init_policy is empty"
        )
    raw_git_baseline = raw_workspace.get("git_baseline", True)
    if isinstance(raw_git_baseline, bool):
        git_baseline: bool | str = raw_git_baseline
    elif isinstance(raw_git_baseline, str) and _GIT_REF_RE.fullmatch(
        raw_git_baseline.strip()
    ):
        git_baseline = raw_git_baseline.strip()
    else:
        raise ServiceConfigurationError(
            "workspace git_baseline must be a boolean or 40-character commit ID"
        )
    raw_clean_paths = raw_workspace.get("clean_paths", [])
    if not isinstance(raw_clean_paths, list):
        raise ServiceConfigurationError("workspace clean_paths must be a list")
    clean_paths = tuple(
        _relative_task_path(value, label="workspace clean_paths entry")
        for value in raw_clean_paths
    )
    if len(clean_paths) != len(set(clean_paths)):
        raise ServiceConfigurationError("workspace clean_paths entries must be unique")
    checkpoint_restore = raw_workspace.get("checkpoint_restore", True)
    if not isinstance(checkpoint_restore, bool):
        raise ServiceConfigurationError("workspace checkpoint_restore must be boolean")
    return WorkspaceRuntimeSpec(
        seed=seed,
        root=root,
        agent_cwd=agent_cwd,
        init_policy=init_policy,
        git_baseline=git_baseline,
        clean_paths=clean_paths,
        checkpoint_restore=checkpoint_restore,
    )


def _artifact_rows(raw: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(raw, list):
        return
    for entry in raw:
        if isinstance(entry, str):
            yield {"source": entry}
            continue
        if not isinstance(entry, Mapping):
            continue
        sources = entry.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, str):
                    yield {
                        **entry,
                        "source": source,
                        "_path_set": True,
                    }
        else:
            yield entry


def _parse_artifacts(
    payload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    main_service: str,
    known_services: set[str],
) -> tuple[ServiceArtifact, ...]:
    raw = payload.get("artifacts")
    if raw is None:
        raw = runtime.get("artifacts")
    artifacts: list[ServiceArtifact] = []
    destinations: set[str] = set()
    for row in _artifact_rows(raw):
        kind = str(row.get("kind", "tree")).strip().lower()
        if kind not in {"file", "tree", "path_set", "binary", "service"}:
            raise ServiceConfigurationError(f"artifact has unsupported kind {kind!r}")
        source = _absolute_container_path(
            _first(row, "source", "path"), label="artifact source"
        )
        service = validate_service_name(_first_string(row, "service") or main_service)
        if service not in known_services:
            raise ServiceConfigurationError(
                f"artifact targets undeclared service {service!r}"
            )
        default_destination = f"{service}/{source.lstrip('/')}"
        destination_value = _first(row, "destination", "target")
        if row.get("_path_set") and destination_value not in (None, ""):
            destination_value = (
                f"{str(destination_value).rstrip('/')}/"
                f"{PurePosixPath(source).name or 'path'}"
            )
        destination = _relative_destination(
            destination_value,
            default=default_destination,
            label=f"artifact destination for {source!r}",
        )
        if destination in destinations:
            raise ServiceConfigurationError(
                f"multiple artifacts use sealed destination {destination!r}"
            )
        destinations.add(destination)
        excludes = row.get("exclude", row.get("excludes", []))
        if not isinstance(excludes, list) or not all(
            isinstance(value, str) and value for value in excludes
        ):
            raise ServiceConfigurationError(
                "artifact exclude must be a list of strings"
            )
        preserve_mode_raw = row.get("preserve_mode", kind == "binary")
        if not isinstance(preserve_mode_raw, bool):
            raise ServiceConfigurationError("artifact preserve_mode must be boolean")
        if kind != "binary" and "preserve_mode" in row:
            raise ServiceConfigurationError(
                "artifact preserve_mode is only valid for binary artifacts"
            )
        artifacts.append(
            ServiceArtifact(
                kind=kind,
                source=source,
                service=service,
                destination=destination,
                required=bool(row.get("required", True)),
                exclude=tuple(excludes),
                max_bytes=_positive_int(
                    _first(row, "max_bytes", "byte_limit"),
                    default=512 * 1024 * 1024,
                    label=f"max_bytes for artifact {source!r}",
                    maximum=4 * 1024**4,
                ),
                max_files=_positive_int(
                    _first(row, "max_files", "file_limit"),
                    default=10_000,
                    label=f"max_files for artifact {source!r}",
                    maximum=1_000_000,
                ),
                max_depth=_positive_int(
                    _first(row, "max_depth"),
                    default=64,
                    label=f"max_depth for artifact {source!r}",
                    maximum=1024,
                ),
                preserve_mode=preserve_mode_raw,
            )
        )
    return tuple(artifacts)


def _tool_rows(
    payload: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[tuple[str | None, Mapping[str, Any]]]:
    tools = _as_mapping(payload.get("tools"))
    rows = _rows(tools.get("mcp_servers"))
    if not rows:
        rows = _rows(payload.get("mcp_servers"))
    if not rows:
        runtime_tools = _as_mapping(runtime.get("tools"))
        rows = _rows(runtime_tools.get("mcp_servers") or runtime.get("tools"))
    environment = _as_mapping(payload.get("environment"))
    mcp_rows = _rows(environment.get("mcp_servers"))
    return [*rows, *mcp_rows]


def _parse_tools(
    payload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    main_service: str,
    known_services: set[str],
) -> tuple[ToolEndpoint, ...]:
    tools: list[ToolEndpoint] = []
    seen: set[str] = set()
    for keyed_name, row in _tool_rows(payload, runtime):
        name = keyed_name or _first_string(row, "name")
        if name is None:
            # Built-in tool lists may be represented as non-table values; they
            # are intentionally outside this task-local endpoint registry.
            continue
        if not _TOOL_NAME_RE.fullmatch(name):
            raise ServiceConfigurationError(
                f"task-local MCP server name is invalid: {name!r}"
            )
        if name in seen:
            raise ServiceConfigurationError(f"duplicate task tool {name!r}")
        seen.add(name)
        transport = (_first_string(row, "transport") or "stdio").lower()
        if transport != "sse":
            raise ServiceConfigurationError(
                f"tool {name!r} uses unsupported transport {transport!r}; "
                "only audited SSE proxying is available"
            )
        service = validate_service_name(
            _first_string(row, "service", "depends_on") or main_service
        )
        if service not in known_services:
            raise ServiceConfigurationError(
                f"tool {name!r} targets undeclared service {service!r}"
            )
        forbidden_connection_fields = sorted(
            key
            for key in ("auth", "headers", "token")
            if row.get(key) not in (None, {}, "")
        )
        if forbidden_connection_fields:
            raise ServiceConfigurationError(
                f"tool {name!r} cannot declare connection credentials or headers: "
                + ", ".join(forbidden_connection_fields)
            )
        command_raw = row.get("command", [])
        if isinstance(command_raw, str):
            command = (command_raw,)
        elif isinstance(command_raw, list) and all(
            isinstance(value, str) and value for value in command_raw
        ):
            command = tuple(command_raw)
        else:
            raise ServiceConfigurationError(
                f"tool {name!r} command must be a string or argv list"
            )
        if command:
            raise ServiceConfigurationError(
                f"tool {name!r} cannot declare a command for SSE transport"
            )
        url = _validated_http_url(
            _first_string(row, "url"),
            label=f"tool {name!r} SSE URL",
            allowed_hosts={service},
            require_port=True,
        )
        readiness = _as_mapping(row.get("readiness"))
        readiness_kind = _first_string(readiness, "kind", "type")
        if readiness_kind not in {None, "command", "http", "tcp"}:
            raise ServiceConfigurationError(
                f"tool {name!r} has unsupported readiness kind " f"{readiness_kind!r}"
            )
        readiness_service = validate_service_name(
            _first_string(readiness, "service") or service
        )
        if readiness_service not in known_services:
            raise ServiceConfigurationError(
                f"tool {name!r} readiness targets undeclared service "
                f"{readiness_service!r}"
            )
        readiness_command_raw = _first(readiness, "command") or _first(
            row, "readiness_command"
        )
        readiness_command = (
            _command_text(
                readiness_command_raw,
                label=f"readiness command for tool {name!r}",
            )
            if readiness_command_raw not in (None, "")
            else None
        )
        readiness_url = _first_string(readiness, "url")
        if readiness_kind == "http":
            readiness_url = _validated_http_url(
                readiness_url,
                label=f"readiness URL for tool {name!r}",
                allowed_hosts={
                    readiness_service,
                    "localhost",
                    "127.0.0.1",
                },
                require_port=True,
            )
        elif readiness_url is not None:
            raise ServiceConfigurationError(
                f"tool {name!r} readiness URL requires kind='http'"
            )
        readiness_host = _first_string(readiness, "host")
        if readiness_kind == "tcp":
            if readiness_host not in {
                readiness_service,
                "localhost",
                "127.0.0.1",
            }:
                raise ServiceConfigurationError(
                    f"tool {name!r} TCP readiness host must use service-local DNS"
                )
        elif readiness_host is not None:
            raise ServiceConfigurationError(
                f"tool {name!r} readiness host requires kind='tcp'"
            )
        readiness_port_raw = _first(readiness, "port")
        readiness_port = (
            _positive_int(
                readiness_port_raw,
                default=1,
                label=f"readiness port for tool {name!r}",
                maximum=65535,
            )
            if readiness_port_raw not in (None, "")
            else None
        )
        if readiness_kind == "tcp" and readiness_port is None:
            raise ServiceConfigurationError(
                f"tool {name!r} TCP readiness requires a port"
            )
        tools.append(
            ToolEndpoint(
                name=name,
                transport=transport,
                service=service,
                url=url,
                command=command,
                readiness_kind=readiness_kind,
                readiness_service=readiness_service,
                readiness_command=readiness_command,
                readiness_url=readiness_url,
                readiness_host=readiness_host,
                readiness_port=readiness_port,
                readiness_timeout_s=_positive_float(
                    _first(readiness, "timeout_s", "timeout_sec")
                    or _first(row, "readiness_timeout_s", "readiness_timeout_sec"),
                    default=60.0,
                    label=f"readiness timeout for tool {name!r}",
                    maximum=60 * 60,
                ),
                readiness_interval_s=_positive_float(
                    _first(readiness, "interval_s", "interval_sec"),
                    default=1.0,
                    label=f"readiness interval for tool {name!r}",
                    maximum=60.0,
                ),
            )
        )
    return tuple(tools)


def _result_paths(payload: Mapping[str, Any]) -> tuple[str, ...]:
    result = _as_mapping(payload.get("result"))
    if result:
        output_root = _absolute_container_path(
            result.get("output_root", "/tmp/output"),
            label="result output_root",
        )
        reward_file = result.get("reward_file", "grade.json")
        if not isinstance(reward_file, str) or not reward_file.strip():
            raise ServiceConfigurationError("result reward_file must be non-empty")
        reward_path = PurePosixPath(output_root) / PurePosixPath(reward_file)
        if ".." in reward_path.parts:
            raise ServiceConfigurationError("result reward_file must not contain '..'")
        paths = [
            _safe_verifier_result_path(
                str(reward_path),
                label="result reward path",
            )
        ]
        trace_file = result.get("trace_file")
        if isinstance(trace_file, str) and trace_file.strip():
            trace_path = PurePosixPath(trace_file)
            trace_path = (
                trace_path
                if trace_path.is_absolute()
                else PurePosixPath(output_root) / trace_path
            )
            if ".." in trace_path.parts:
                raise ServiceConfigurationError(
                    "result trace_file must not contain '..'"
                )
            paths.append(
                _safe_verifier_result_path(
                    str(trace_path),
                    label="result trace path",
                )
            )
        selected_reports = result.get("reports", [])
        if isinstance(selected_reports, list):
            selected_names = {
                str(name) for name in selected_reports if isinstance(name, str)
            }
            for _key, report in _rows(payload.get("reports")):
                name = _first_string(report, "name")
                report_path_raw = _first_string(report, "path")
                if name not in selected_names or report_path_raw is None:
                    continue
                report_path = PurePosixPath(report_path_raw)
                report_path = (
                    report_path
                    if report_path.is_absolute()
                    else PurePosixPath(output_root) / report_path
                )
                if ".." in report_path.parts:
                    raise ServiceConfigurationError(
                        f"report {name!r} path must not contain '..'"
                    )
                paths.append(
                    _safe_verifier_result_path(
                        str(report_path),
                        label=f"report {name!r} path",
                    )
                )
        paths = list(dict.fromkeys(paths))
        return tuple(paths)
    verifier = _as_mapping(payload.get("verifier"))
    raw = _first(verifier, "result_files", "result_paths", "results")
    if raw is None:
        return _DEFAULT_RESULT_PATHS
    if not isinstance(raw, list) or not raw:
        raise ServiceConfigurationError(
            "verifier result_files must be a non-empty list"
        )
    return tuple(
        _safe_verifier_result_path(value, label="verifier result path") for value in raw
    )


def load_task_service_config(
    task_toml_path: Path = Path("/task/task.toml"),
    *,
    operator_roots: RuntimeOperatorRoots = DEFAULT_OPERATOR_ROOTS,
) -> TaskServiceConfig | None:
    """Load optional service capabilities, returning ``None`` for legacy tasks."""
    try:
        stat = task_toml_path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ServiceConfigurationError(
            f"could not inspect task configuration {task_toml_path}: {exc}"
        ) from exc
    if not task_toml_path.is_file() or task_toml_path.is_symlink():
        raise ServiceConfigurationError(
            f"task configuration is not a regular file: {task_toml_path}"
        )
    if stat.st_size > _MAX_TASK_TOML_BYTES:
        raise ServiceConfigurationError(
            f"task configuration exceeds {_MAX_TASK_TOML_BYTES} bytes"
        )
    try:
        with task_toml_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ServiceConfigurationError(
            f"could not parse task configuration {task_toml_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ServiceConfigurationError("task.toml must contain a TOML table")

    runtime = _runtime_table(payload)
    raw_services = payload.get("services")
    if raw_services is None:
        raw_services = runtime.get("services")
    service_rows = _rows(raw_services)
    if not _capability_declared(payload, runtime, service_rows):
        return None
    if runtime.get("enabled") is False:
        return None
    selected_runtime_paths = sorted(
        key for key in _TASK_SELECTED_RUNTIME_PATH_KEYS if key in runtime
    )
    capsule_table = _as_mapping(payload.get("capsule"))
    selected_capsule_paths = sorted(
        key for key in ("directory", "path") if key in capsule_table
    )
    if selected_runtime_paths or selected_capsule_paths:
        selected = selected_runtime_paths + [
            f"capsule.{key}" for key in selected_capsule_paths
        ]
        raise ServiceConfigurationError(
            "task.toml cannot select operator-owned host paths: " + ", ".join(selected)
        )
    capsule_manifest = (
        _first_string(runtime, "manifest", "manifest_file")
        or _first_string(capsule_table, "manifest")
        or "manifest.json"
    )
    if not service_rows:
        service_rows = _capsule_service_rows(
            operator_roots.capsule,
            capsule_manifest,
        )
    if not service_rows:
        service_rows = [("main", {"name": "main", "role": "main"})]

    services, main_service, verifier_service = _parse_services(
        service_rows, runtime, payload
    )
    known_services = {service.name for service in services}
    main_spec = next(service for service in services if service.name == main_service)
    workspace = _as_mapping(payload.get("workspace"))
    agent = _as_mapping(payload.get("agent"))
    agent_user = (
        _first_string(runtime, "agent_user")
        or _normalize_user(_first(agent, "user", "run_as"))
        or main_spec.user
        or "1000:1000"
    )
    agent_user = _require_unprivileged_user(agent_user)
    agent_workdir = (
        _first_string(runtime, "agent_workdir")
        or _first_string(workspace, "agent_cwd", "writable_root", "root")
        or main_spec.workdir
        or "/workdir"
    )
    agent_workdir = _absolute_container_path(
        agent_workdir, label="main service agent workdir"
    )

    task = _as_mapping(payload.get("task"))
    task_identity = _first_string(task, "name", "id") or task_toml_path.parent.name
    project_base = _PROJECT_RE.sub("-", task_identity.lower()).strip("-_")
    project_hash = hashlib.sha256(task_identity.encode()).hexdigest()[:10]
    project_name = f"lbx-{project_base[:32] or 'task'}-{project_hash}"
    capsule_dir = operator_roots.capsule
    state_dir = operator_roots.state / project_name
    sealed_dir = operator_roots.sealed / project_name
    verifier = _as_mapping(payload.get("verifier"))
    workspace_spec = _parse_workspace(payload)
    tool_endpoints = _parse_tools(
        payload,
        runtime,
        main_service=main_service,
        known_services=known_services,
    )
    service_roles = {service.name: service.role for service in services}
    for endpoint in tool_endpoints:
        if service_roles[endpoint.service] in {"init", "verifier"}:
            raise ServiceConfigurationError(
                f"tool {endpoint.name!r} must target a long-running main or "
                "sidecar service"
            )
        if service_roles[endpoint.readiness_service] in {"init", "verifier"}:
            raise ServiceConfigurationError(
                f"tool {endpoint.name!r} readiness must target a long-running "
                "main or sidecar service"
            )

    verifier_result_paths = _result_paths(payload)
    return TaskServiceConfig(
        task_toml_path=task_toml_path,
        operator_roots=operator_roots,
        capsule_dir=capsule_dir,
        capsule_manifest=capsule_manifest,
        compose_file=_first_string(runtime, "compose_file", "compose_path"),
        state_dir=state_dir,
        sealed_dir=sealed_dir,
        project_name=project_name,
        services=services,
        main_service=main_service,
        verifier_service=verifier_service,
        agent_user=agent_user,
        agent_workdir=agent_workdir,
        agent_shell=_first_string(runtime, "agent_shell") or main_spec.shell,
        startup_timeout_s=_positive_float(
            _first(runtime, "startup_timeout_s", "startup_timeout_sec"),
            default=180.0,
            label="service startup timeout",
        ),
        daemon_timeout_s=_positive_float(
            _first(runtime, "daemon_timeout_s", "daemon_timeout_sec"),
            default=60.0,
            label="dockerd startup timeout",
        ),
        command_timeout_s=_positive_float(
            _first(runtime, "command_timeout_s", "command_timeout_sec"),
            default=6 * 60 * 60,
            label="agent command timeout",
        ),
        verifier_timeout_s=_positive_float(
            _first(verifier, "timeout_s", "timeout_sec"),
            default=600.0,
            label="verifier timeout",
        ),
        max_output_bytes=_positive_int(
            runtime.get("max_output_bytes"),
            default=4 * 1024 * 1024,
            label="runtime max_output_bytes",
            maximum=64 * 1024 * 1024,
        ),
        editor_max_bytes=_positive_int(
            runtime.get("editor_max_bytes"),
            default=8 * 1024 * 1024,
            label="runtime editor_max_bytes",
            maximum=64 * 1024 * 1024,
        ),
        workspace=workspace_spec,
        captures=_parse_captures(
            payload,
            runtime,
            main_service=main_service,
            known_services=known_services,
        ),
        artifacts=_parse_artifacts(
            payload,
            runtime,
            main_service=main_service,
            known_services=known_services,
        ),
        tools=tool_endpoints,
        verifier_result_paths=verifier_result_paths,
        verifier_reward_path=verifier_result_paths[0],
        primary_reward=_first_string(_as_mapping(payload.get("result")), "reward_key")
        or _first_string(verifier, "primary_reward", "reward_key")
        or "score",
        subscores_key=_first_string(_as_mapping(payload.get("result")), "subscores_key")
        or "subscores",
    )
