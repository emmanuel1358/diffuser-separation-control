"""Shared schemas for the universal task plugin."""

import math
import re
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from alignerr_plugin.delivery import normalize_delivery_platform
from alignerr_plugin.taiga_resources import (
    TaigaRequiredResources,
    is_tpu_resource,
    validate_required_resources,
)
from alignerr_plugin.task_metadata import (
    metadata_validation_issues,
    normalize_enum_value,
)

# Accepted values for [environment].hidden_env (the env_server activation gate).
# "" disables it; "env"/"hybrid" turn the hidden-environment RPC server on.
HIDDEN_ENV_MODES: tuple[str, ...] = ("", "env", "hybrid")
# Hugging Face hub cache root inside the flagship base images. Every flagship
# Dockerfile sets HF_HOME=/tmp/hf-cache, and huggingface_hub resolves repos from
# <HF_HOME>/hub/<repo_type>s--<org>--<name>, so mounting there makes
# from_pretrained("org/name") work offline with no author-side path juggling.
HF_HOME = "/tmp/hf-cache"
HF_HUB_CACHE = f"{HF_HOME}/hub"
HF_REPO_TYPES: tuple[str, ...] = ("model", "dataset")

# A Hugging Face repo_id is `namespace/name`: exactly one slash, each component
# starting alphanumeric and otherwise [A-Za-z0-9._-]. Matches the packer's
# deploy-time validation so a malformed id fails at authoring time instead of
# only fail-closed at deploy.
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_CAPABILITY_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_OCI_PLATFORM_RE = re.compile(
    r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)?$"
)
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PINNED_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$", re.IGNORECASE)
_ALLOWED_SERVICE_CAPABILITIES = frozenset({"SYS_PTRACE"})
_SOCKET_MARKERS = (
    "docker.sock",
    "containerd.sock",
    "podman.sock",
)
_FORBIDDEN_WRITABLE_ROOTS: tuple[PurePosixPath, ...] = tuple(
    PurePosixPath(path)
    for path in (
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/lib",
        "/lib64",
        "/mcp_server",
        "/proc",
        "/run",
        "/run/lbx-task-service-runtime",
        "/sbin",
        "/sys",
        "/task/capsule",
        "/usr",
        "/Users",
        "/var/run",
        "/Volumes",
        "/private",
    )
)
_PROTECTED_SERVICE_ROOTS: tuple[PurePosixPath, ...] = tuple(
    PurePosixPath(path)
    for path in (
        "/dev",
        "/mcp_server",
        "/proc",
        "/run/lbx-task-service-runtime",
        "/sys",
        "/task/capsule",
    )
)


def _validate_capability_name(value: str, label: str) -> str:
    """Validate a stable TOML identifier used for cross-section references."""
    value = value.strip()
    if not _CAPABILITY_NAME_RE.fullmatch(value):
        raise ValueError(
            f"{label} must start with a lowercase letter and contain only "
            "lowercase letters, numbers, '.', '_' or '-'"
        )
    return value


def _validate_relative_task_path(
    value: str, label: str, *, allow_dot: bool = False
) -> str:
    """Reject host paths and traversal for task-repository inputs."""
    value = value.strip()
    if not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ValueError(
            f"{label} must be task-relative (absolute/host paths are forbidden)"
        )
    if path == PurePosixPath(".") and not allow_dot:
        raise ValueError(f"{label} must name a path below the task root")
    return str(path)


def _validate_container_path(value: str, label: str) -> str:
    """Validate an absolute path inside a task container."""
    value = value.strip()
    if not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute container path without '..'")
    if path == PurePosixPath("/"):
        raise ValueError(f"{label} cannot be the container root")
    return str(path)


def _reject_socket_path(value: str, label: str) -> str:
    lowered = value.lower()
    if any(marker in lowered for marker in _SOCKET_MARKERS):
        raise ValueError(f"{label} must not reference a container runtime socket")
    return value


def _reject_protected_service_path(value: str, label: str) -> str:
    value = _validate_container_path(value, label)
    _reject_socket_path(value, label)
    path = PurePosixPath(value)
    for protected in _PROTECTED_SERVICE_ROOTS:
        if path == protected or protected in path.parents:
            raise ValueError(f"{label} cannot be inside protected path {protected}")
    return value


def _validate_writable_root(value: str, label: str) -> str:
    """Keep framework-managed writable roots away from privileged runtime trees."""
    value = _validate_container_path(value, label)
    path = PurePosixPath(value)
    for forbidden in _FORBIDDEN_WRITABLE_ROOTS:
        if path == forbidden or forbidden in path.parents:
            raise ValueError(f"{label} cannot be inside reserved path {forbidden}")
    return value


def _validate_output_root(value: str, label: str) -> str:
    """Keep result roots in the platform-owned writable output tree."""
    value = _validate_container_path(value, label)
    path = PurePosixPath(value)
    output = PurePosixPath("/tmp/output")
    if path != output and output not in path.parents:
        raise ValueError(f"{label} must be /tmp/output or one of its descendants")
    return value


def _validate_destination_path(value: str, label: str) -> str:
    """Validate a container-absolute or transfer-root-relative destination."""
    path = PurePosixPath(value)
    if path.is_absolute():
        return _validate_container_path(value, label)
    return _validate_relative_task_path(value, label)


def _validate_artifact_destination(value: str, label: str) -> str:
    """Keep sealed artifact destinations relative to the transfer root."""
    value = _validate_relative_task_path(value, label)
    if PurePosixPath(value).parts[0] == "manifest.json":
        raise ValueError(f"{label} cannot shadow reserved manifest.json")
    return value


def _validate_capture_destination(value: str, label: str) -> str:
    """Capture commands atomically replace absolute service-local /tmp paths."""
    value = value.strip()
    path = PurePosixPath(value)
    temporary_root = PurePosixPath("/tmp")
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute path below /tmp without '..'")
    value = _reject_protected_service_path(value, label)
    if temporary_root not in path.parents:
        raise ValueError(f"{label} must be an absolute path below /tmp")
    return value


def _normalize_target_platform(
    platform: str | None, architecture: str | None, label: str
) -> tuple[str | None, str | None]:
    """Canonicalize the Linux target consumed by build/export backends."""
    if platform is None:
        return (
            (f"linux/{architecture}" if architecture is not None else None),
            architecture,
        )
    platform = platform.strip().lower()
    if not _OCI_PLATFORM_RE.fullmatch(platform):
        raise ValueError(f"{label} must use OCI os/architecture[/variant] syntax")
    parts = PurePosixPath(platform).parts
    if parts[0] != "linux":
        raise ValueError(f"{label} must target linux")
    platform_architecture = parts[1]
    if platform_architecture not in {"amd64", "arm64"}:
        raise ValueError(f"{label} architecture must be 'amd64' or 'arm64'")
    if architecture is not None and architecture != platform_architecture:
        raise ValueError(
            f"{label} architecture {platform_architecture!r} conflicts with "
            f"architecture {architecture!r}"
        )
    return platform, platform_architecture


def _validate_http_url(value: str, label: str) -> str:
    """Validate an HTTP(S) endpoint without credentials or fragments."""
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid URL: {exc}") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be an absolute http:// or https:// URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{label} must not embed credentials")
    if parsed.fragment:
        raise ValueError(f"{label} must not contain a fragment")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{label} port must be between 1 and 65535")
    return value


def _is_path_within(path_value: str, root_value: str) -> bool:
    path = PurePosixPath(path_value)
    root = PurePosixPath(root_value)
    return path == root or root in path.parents


def hf_hub_mount_path(repo_id: str, repo_type: str = "model") -> str:
    """Canonical hub-cache mount path for an HF repo inside the task container."""
    folder = f"{repo_type}s--" + repo_id.replace("/", "--")
    return f"{HF_HUB_CACHE}/{folder}"


class OutputSpec(BaseModel):
    """Expected agent output path."""

    path: str
    required: bool = True
    description: str = ""

    @field_validator("path")
    @classmethod
    def path_must_be_output_scoped(cls, value: str) -> str:
        """Keep authored outputs in Boreal's writable output directory."""
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or path.parts[:3] != ("/", "tmp", "output")
        ):
            raise ValueError("output paths must be absolute and under /tmp/output")
        return value


class TaskSection(BaseModel):
    """Human-facing task metadata."""

    name: str
    description: str = ""
    authors: list[dict[str, str]] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class CapabilityModel(BaseModel):
    """Strict base for opt-in native capability declarations."""

    model_config = ConfigDict(extra="forbid")


class WorkspaceSection(CapabilityModel):
    """Runtime-enforced workspace initialization and checkpoint policy."""

    seed: str | None = None
    root: str
    agent_cwd: str | None = None
    init_policy: Literal["copy", "overlay", "reuse", "empty"] = "copy"
    git_baseline: StrictBool | str = True
    clean_paths: list[str] = Field(default_factory=list)
    checkpoint_restore: StrictBool = True

    @field_validator("init_policy", mode="before")
    @classmethod
    def normalize_init_policy(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_workspace(self) -> "WorkspaceSection":
        self.root = _validate_writable_root(self.root, "[workspace].root")
        if self.seed is not None:
            if not self.seed.strip():
                self.seed = None
            else:
                self.seed = _validate_relative_task_path(
                    self.seed, "[workspace].seed", allow_dot=True
                )
        if self.seed is None and self.init_policy != "empty":
            raise ValueError(
                "[workspace].seed is required unless init_policy is 'empty'"
            )

        if self.agent_cwd is None:
            self.agent_cwd = self.root
        else:
            self.agent_cwd = _validate_writable_root(
                self.agent_cwd, "[workspace].agent_cwd"
            )
            if not _is_path_within(self.agent_cwd, self.root):
                raise ValueError("[workspace].agent_cwd must be inside workspace.root")

        if isinstance(self.git_baseline, str):
            self.git_baseline = self.git_baseline.strip()
            if not _GIT_COMMIT_RE.fullmatch(self.git_baseline):
                raise ValueError(
                    "[workspace].git_baseline must be a boolean or "
                    "40-character lowercase commit ID"
                )

        self.clean_paths = [
            _validate_relative_task_path(clean_path, "[workspace].clean_paths entry")
            for clean_path in self.clean_paths
        ]
        if len(self.clean_paths) != len(set(self.clean_paths)):
            raise ValueError("[workspace].clean_paths entries must be unique")
        return self


class ArtifactLimits(CapabilityModel):
    """Bound artifact traversal and transfer work."""

    max_bytes: int | None = Field(default=None, gt=0)
    max_files: int | None = Field(default=None, gt=0)
    max_depth: int | None = Field(default=None, gt=0, le=1024)

    @field_validator("max_depth", mode="before")
    @classmethod
    def max_depth_must_be_an_integer(cls, value: object) -> object:
        if isinstance(value, bool):
            raise PydanticCustomError(
                "artifact_max_depth_type",
                "artifact max_depth must be an integer",
            )
        return value


class _ArtifactBase(CapabilityModel):
    """Fields shared by every native artifact shape."""

    name: str
    destination: str | None = None
    service: str = "main"
    exclude: list[str] = Field(default_factory=list)
    required: bool = True
    max_bytes: int | None = Field(default=None, gt=0)
    max_files: int | None = Field(default=None, gt=0)
    max_depth: int | None = Field(default=None, gt=0, le=1024)

    @model_validator(mode="before")
    @classmethod
    def flatten_limits(cls, value: object) -> object:
        """Accept a grouped TOML limits table and dump canonical flat limits."""
        if not isinstance(value, dict) or "limits" not in value:
            if isinstance(value, dict) and isinstance(value.get("max_depth"), bool):
                raise PydanticCustomError(
                    "artifact_max_depth_type",
                    "artifact max_depth must be an integer",
                )
            return value
        normalized = dict(value)
        raw_limits = normalized.pop("limits")
        limits = ArtifactLimits.model_validate(raw_limits)
        for field, limit in limits.model_dump(exclude_none=True).items():
            if field in normalized and normalized[field] != limit:
                raise ValueError(f"artifact {field} conflicts with the value in limits")
            normalized[field] = limit
        return normalized

    @model_validator(mode="after")
    def validate_common_artifact_fields(self) -> "_ArtifactBase":
        self.name = _validate_capability_name(self.name, "artifact name")
        self.service = _validate_capability_name(
            self.service, f"artifact {self.name!r} service"
        )
        destination = self.destination or self.name
        self.destination = _validate_artifact_destination(
            destination, f"artifact {self.name!r} destination"
        )
        for pattern in self.exclude:
            _validate_relative_task_path(
                pattern, f"artifact {self.name!r} exclude pattern"
            )
        if len(self.exclude) != len(set(self.exclude)):
            raise ValueError(f"artifact {self.name!r} exclude patterns must be unique")
        return self


class FileArtifact(_ArtifactBase):
    """A single regular-file submission."""

    kind: Literal["file"]
    source: str

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _reject_protected_service_path(value, "file artifact source")


class TreeArtifact(_ArtifactBase):
    """A recursively collected directory tree."""

    kind: Literal["tree"]
    source: str

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _reject_protected_service_path(value, "tree artifact source")


class PathSetArtifact(_ArtifactBase):
    """A stable set of files and trees collected into one artifact."""

    kind: Literal["path_set"]
    sources: list[str] = Field(min_length=1)

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, values: list[str]) -> list[str]:
        for value in values:
            _reject_protected_service_path(value, "path-set artifact source")
        if len(values) != len(set(values)):
            raise ValueError("path-set artifact sources must be unique")
        return values


class BinaryArtifact(_ArtifactBase):
    """A binary file whose bytes and mode must be preserved exactly."""

    kind: Literal["binary"]
    source: str
    preserve_mode: StrictBool = True

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _reject_protected_service_path(value, "binary artifact source")


class ServiceArtifact(_ArtifactBase):
    """An artifact collected from a declared sidecar or init service."""

    kind: Literal["service"]
    source: str
    service: str

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _reject_protected_service_path(value, "service artifact source")


type ArtifactSpec = Annotated[
    FileArtifact | TreeArtifact | PathSetArtifact | BinaryArtifact | ServiceArtifact,
    Field(discriminator="kind"),
]


class ResourceSpec(CapabilityModel):
    """Independent resource and network policy for one execution phase."""

    cpus: float | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    storage_mb: int | None = Field(default=None, gt=0)
    gpus: int = Field(default=0, ge=0)
    gpu_types: list[str] = Field(default_factory=list)
    platform: str | None = None
    architecture: Literal["amd64", "arm64"] | None = None
    build_timeout_sec: float | None = Field(default=None, gt=0)
    runtime_timeout_sec: float | None = Field(default=None, gt=0)
    network: Literal["none", "isolated", "internet"] = "isolated"

    @model_validator(mode="after")
    def validate_resources(self) -> "ResourceSpec":
        self.gpu_types = [gpu_type.strip() for gpu_type in self.gpu_types]
        if any(not gpu_type for gpu_type in self.gpu_types):
            raise ValueError("gpu_types cannot contain empty names")
        if len(self.gpu_types) != len(set(self.gpu_types)):
            raise ValueError("gpu_types must be unique")
        if self.gpus and not self.gpu_types:
            raise ValueError("gpu_types is required when gpus is greater than zero")
        if not self.gpus and self.gpu_types:
            raise ValueError("gpu_types requires gpus greater than zero")
        self.platform, self.architecture = _normalize_target_platform(
            self.platform, self.architecture, "resource platform"
        )
        return self


class ServiceBuild(CapabilityModel):
    """Task-relative Docker build inputs for a service image."""

    context: str
    dockerfile: str = "Dockerfile"
    target: str | None = None
    args: dict[str, str] = Field(default_factory=dict)
    platform: str | None = None
    architecture: Literal["amd64", "arm64"] | None = None
    pull: bool = False
    no_cache: bool = False

    @model_validator(mode="after")
    def validate_build(self) -> "ServiceBuild":
        self.context = _validate_relative_task_path(
            self.context, "service build context", allow_dot=True
        )
        self.dockerfile = _validate_relative_task_path(
            self.dockerfile, "service build dockerfile"
        )
        if any(not key.strip() for key in self.args):
            raise ValueError("service build argument names cannot be empty")
        self.platform, self.architecture = _normalize_target_platform(
            self.platform, self.architecture, "service build platform"
        )
        return self


class ServiceDependency(CapabilityModel):
    """Start a service after another service reaches a declared state."""

    service: str
    condition: Literal["started", "healthy", "completed"] = "started"

    @field_validator("service")
    @classmethod
    def validate_service(cls, value: str) -> str:
        return _validate_capability_name(value, "service dependency")

    @field_validator("condition", mode="before")
    @classmethod
    def normalize_condition(cls, value: object) -> object:
        aliases = {
            "service_started": "started",
            "service_healthy": "healthy",
            "service_completed_successfully": "completed",
        }
        return aliases.get(value, value) if isinstance(value, str) else value


type Command = str | list[str]


class ServiceHealthcheck(CapabilityModel):
    """Bounded service liveness/readiness command."""

    command: Command
    interval_sec: float = Field(default=5.0, gt=0)
    timeout_sec: float = Field(default=5.0, gt=0)
    retries: int = Field(default=3, gt=0)
    start_period_sec: float = Field(default=0.0, ge=0)
    start_interval_sec: float | None = Field(default=None, gt=0)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: Command) -> Command:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("healthcheck command cannot be empty")
        elif not value or not value[0]:
            raise ValueError("healthcheck command argv cannot be empty")
        return value


class PortSpec(CapabilityModel):
    """A service port, optionally published by the backend."""

    container_port: int = Field(ge=1, le=65535)
    host_port: int | None = Field(default=None, ge=1, le=65535)
    protocol: Literal["tcp", "udp"] = "tcp"
    name: str | None = None


class VolumeMount(CapabilityModel):
    """Mount a declared named volume into one service."""

    volume: str
    target: str
    mode: Literal["ro", "rw"] = "rw"

    @model_validator(mode="after")
    def validate_mount(self) -> "VolumeMount":
        self.volume = _validate_capability_name(self.volume, "volume mount reference")
        _reject_socket_path(self.volume, "volume mount reference")
        self.target = _reject_protected_service_path(self.target, "volume mount target")
        return self


class NamedVolume(CapabilityModel):
    """A backend-managed volume; host bind sources are intentionally absent."""

    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = _validate_capability_name(value, "volume name")
        return _reject_socket_path(value, "volume name")


class ServiceSpec(CapabilityModel):
    """A main, sidecar, one-shot init, or isolated verifier container."""

    name: str
    role: Literal["main", "sidecar", "init", "verifier"]
    image: str | None = None
    build: ServiceBuild | None = None
    command: Command | None = None
    user: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    ports: list[PortSpec] = Field(default_factory=list)
    network_mode: Literal["bridge", "none", "share"] = "bridge"
    network_share_target: str | None = None
    network_aliases: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    shm_mb: int | None = Field(default=None, gt=0)
    resources: ResourceSpec | None = None
    depends_on: list[ServiceDependency] = Field(default_factory=list)
    restart: Literal["no", "on-failure", "always", "unless-stopped"] = "no"
    healthcheck: ServiceHealthcheck | None = None
    volumes: list[VolumeMount] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_service(self) -> "ServiceSpec":
        self.name = _validate_capability_name(self.name, "service name")
        has_image = bool(self.image and self.image.strip())
        if has_image == (self.build is not None):
            raise ValueError(
                f"service {self.name!r} requires exactly one of image or build"
            )
        if self.image is not None:
            self.image = self.image.strip()
            if not _PINNED_IMAGE_RE.fullmatch(self.image):
                raise ValueError(
                    f"service {self.name!r} image must be pinned as "
                    "<repository>@sha256:<64 hex>"
                )

        if self.role == "main" and self.user is not None:
            uid = self.user.strip().lower().split(":", 1)[0]
            if uid in {"0", "root"}:
                raise ValueError("main service user must be unprivileged")

        orchestration_variables = {
            "CONTAINERD_ADDRESS",
            "CONTAINER_HOST",
            "DOCKER_HOST",
        }
        for key, value in self.env.items():
            if key.upper() in orchestration_variables and value:
                raise ValueError(
                    f"service {self.name!r} must not receive orchestration "
                    f"variable {key!r}"
                )
            _reject_socket_path(
                value, f"service {self.name!r} environment variable {key!r}"
            )

        if self.command is not None:
            if isinstance(self.command, str):
                if not self.command.strip():
                    raise ValueError(f"service {self.name!r} command cannot be empty")
            elif not self.command or not self.command[0]:
                raise ValueError(f"service {self.name!r} command argv cannot be empty")

        if self.network_mode == "share":
            if self.network_share_target is None:
                raise ValueError(
                    f"service {self.name!r} share network mode requires "
                    "network_share_target"
                )
            self.network_share_target = _validate_capability_name(
                self.network_share_target, "network share target"
            )
        elif self.network_share_target is not None:
            raise ValueError(
                f"service {self.name!r} network_share_target requires "
                "network_mode = 'share'"
            )

        for alias in self.network_aliases:
            _validate_capability_name(alias, f"service {self.name!r} network alias")
        if len(self.network_aliases) != len(set(self.network_aliases)):
            raise ValueError(f"service {self.name!r} network aliases must be unique")
        if self.network_aliases and self.network_mode != "bridge":
            raise ValueError(
                f"service {self.name!r} network aliases require bridge network mode"
            )

        self.capabilities = [
            capability.strip().upper() for capability in self.capabilities
        ]
        if any(not capability for capability in self.capabilities):
            raise ValueError(f"service {self.name!r} capabilities cannot be empty")
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError(f"service {self.name!r} capabilities must be unique")
        unsupported_capabilities = sorted(
            set(self.capabilities) - _ALLOWED_SERVICE_CAPABILITIES
        )
        if unsupported_capabilities:
            raise ValueError(
                f"service {self.name!r} requests forbidden capabilities "
                f"{unsupported_capabilities}; only SYS_PTRACE is allowed"
            )

        dependency_names = [dependency.service for dependency in self.depends_on]
        if len(dependency_names) != len(set(dependency_names)):
            raise ValueError(f"service {self.name!r} dependencies must be unique")
        mount_targets = [mount.target for mount in self.volumes]
        if len(mount_targets) != len(set(mount_targets)):
            raise ValueError(f"service {self.name!r} volume targets must be unique")
        if self.role == "init" and self.restart != "no":
            raise ValueError("init services cannot use a restart policy")
        return self


class CaptureSpec(CapabilityModel):
    """An ordered, failure-atomic pre-verification service hook."""

    name: str
    service: str
    command: Command
    timeout_sec: float = Field(gt=0)
    atomic_destination: str | None = Field(
        default=None,
        validation_alias=AliasChoices("atomic_destination", "destination"),
    )
    accepted_exit_codes: list[int] = Field(default_factory=lambda: [0])
    failure_policy: Literal["agent", "infrastructure"] = Field(
        default="infrastructure",
        validation_alias=AliasChoices("failure_policy", "fault_policy"),
    )

    @model_validator(mode="after")
    def validate_capture(self) -> "CaptureSpec":
        self.name = _validate_capability_name(self.name, "capture name")
        self.service = _validate_capability_name(
            self.service, f"capture {self.name!r} service"
        )
        if self.atomic_destination is not None:
            self.atomic_destination = _validate_capture_destination(
                self.atomic_destination,
                f"capture {self.name!r} atomic_destination",
            )
        if isinstance(self.command, str):
            if not self.command.strip():
                raise ValueError(f"capture {self.name!r} command cannot be empty")
        elif not self.command or not self.command[0]:
            raise ValueError(f"capture {self.name!r} command argv cannot be empty")
        if not self.accepted_exit_codes:
            raise ValueError(
                f"capture {self.name!r} accepted_exit_codes cannot be empty"
            )
        if any(code < 0 or code > 255 for code in self.accepted_exit_codes):
            raise ValueError(
                f"capture {self.name!r} accepted_exit_codes must be in 0..255"
            )
        if len(self.accepted_exit_codes) != len(set(self.accepted_exit_codes)):
            raise ValueError(
                f"capture {self.name!r} accepted_exit_codes must be unique"
            )
        return self


class _ReadinessBase(CapabilityModel):
    timeout_sec: float = Field(default=30.0, gt=0)
    interval_sec: float = Field(default=1.0, gt=0)


class HTTPReadiness(_ReadinessBase):
    kind: Literal["http"]
    url: str
    accepted_statuses: list[int] = Field(default_factory=lambda: [200])

    @model_validator(mode="after")
    def validate_http_readiness(self) -> "HTTPReadiness":
        self.url = _validate_http_url(self.url, "HTTP readiness URL")
        if not self.accepted_statuses:
            raise ValueError("HTTP readiness accepted_statuses cannot be empty")
        if any(status < 100 or status > 599 for status in self.accepted_statuses):
            raise ValueError("HTTP readiness statuses must be in 100..599")
        if len(self.accepted_statuses) != len(set(self.accepted_statuses)):
            raise ValueError("HTTP readiness statuses must be unique")
        return self


class TCPReadiness(_ReadinessBase):
    kind: Literal["tcp"]
    host: str
    port: int = Field(ge=1, le=65535)

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        if not value.strip() or any(character.isspace() for character in value):
            raise ValueError("TCP readiness host must be non-empty")
        return value


class CommandReadiness(_ReadinessBase):
    kind: Literal["command"]
    command: Command
    service: str | None = None

    @model_validator(mode="after")
    def validate_command_readiness(self) -> "CommandReadiness":
        if isinstance(self.command, str):
            if not self.command.strip():
                raise ValueError("command readiness command cannot be empty")
        elif not self.command or not self.command[0]:
            raise ValueError("command readiness argv cannot be empty")
        if self.service is not None:
            self.service = _validate_capability_name(
                self.service, "command readiness service"
            )
        return self


type ReadinessSpec = Annotated[
    HTTPReadiness | TCPReadiness | CommandReadiness,
    Field(discriminator="kind"),
]


class _MCPServerBase(CapabilityModel):
    name: str
    service: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    readiness: ReadinessSpec | None = None
    access: Literal["agent", "verifier", "both"] = "agent"

    @model_validator(mode="after")
    def validate_common_mcp_fields(self) -> "_MCPServerBase":
        self.name = _validate_capability_name(self.name, "MCP server name")
        if self.service is not None:
            self.service = _validate_capability_name(
                self.service, f"MCP server {self.name!r} service"
            )
        self.depends_on = [
            _validate_capability_name(
                dependency, f"MCP server {self.name!r} dependency"
            )
            for dependency in self.depends_on
        ]
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError(f"MCP server {self.name!r} dependencies must be unique")
        return self


class StdioMCPServer(_MCPServerBase):
    transport: Literal["stdio"]
    command: Command
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_stdio_server(self) -> "StdioMCPServer":
        if isinstance(self.command, str):
            if not self.command.strip():
                raise ValueError("stdio MCP command cannot be empty")
        elif not self.command or not self.command[0]:
            raise ValueError("stdio MCP command argv cannot be empty")
        if self.cwd is not None:
            self.cwd = _validate_writable_root(self.cwd, "stdio MCP cwd")
        return self


class SSEMCPServer(_MCPServerBase):
    transport: Literal["sse"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_http_url(value, "SSE MCP URL")


type MCPServerSpec = Annotated[
    StdioMCPServer | SSEMCPServer,
    Field(discriminator="transport"),
]


class EvaluationSection(CapabilityModel):
    """Metadata describing the existing grader entrypoint and trust envelope."""

    engine: Literal["rubric_task", "continuous_task", "legacy_runner"]
    entrypoint: str | None = None
    hidden_fixtures: list[str] = Field(default_factory=list)
    repetitions: int = Field(default=1, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evaluation(self) -> "EvaluationSection":
        if self.entrypoint is not None:
            self.entrypoint = _validate_relative_task_path(
                self.entrypoint, "evaluation entrypoint"
            )
        for fixture in self.hidden_fixtures:
            _validate_relative_task_path(fixture, "evaluation hidden fixture")
        if len(self.hidden_fixtures) != len(set(self.hidden_fixtures)):
            raise ValueError("evaluation hidden fixtures must be unique")
        return self


class GateSpec(CapabilityModel):
    """One ordered structural, behavioral, or quality gate."""

    name: str
    kind: Literal[
        "structural",
        "behavioral",
        "performance",
        "determinism",
        "static",
        "custom",
    ]
    command: Command | None = None
    required: bool = True
    weight: float = Field(default=1.0, gt=0)
    timeout_sec: float | None = Field(default=None, gt=0)
    repetitions: int = Field(default=1, gt=0)
    report: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_gate(self) -> "GateSpec":
        self.name = _validate_capability_name(self.name, "gate name")
        if self.command is not None:
            if isinstance(self.command, str):
                if not self.command.strip():
                    raise ValueError(f"gate {self.name!r} command cannot be empty")
            elif not self.command or not self.command[0]:
                raise ValueError(f"gate {self.name!r} command argv cannot be empty")
        if self.report is not None:
            self.report = _validate_capability_name(
                self.report, f"gate {self.name!r} report"
            )
        return self


class ReportSpec(CapabilityModel):
    """A bounded verifier diagnostic report."""

    name: str
    format: Literal["json", "ctrf", "junit", "text", "custom"]
    path: str
    required: bool = False
    max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)

    @model_validator(mode="after")
    def validate_report(self) -> "ReportSpec":
        self.name = _validate_capability_name(self.name, "report name")
        self.path = _validate_destination_path(self.path, f"report {self.name!r} path")
        return self


class ResultSection(CapabilityModel):
    """Canonical reward and structured result locations."""

    output_root: str = "/tmp/output"
    reward_file: str = "grade.json"
    reward_key: str = "score"
    subscores_key: str | None = "subscores"
    reports: list[str] = Field(default_factory=list)
    trace_file: str | None = None
    agent_fault_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    infrastructure_fault: Literal["discard"] = "discard"

    @field_validator("agent_fault_reward", mode="before")
    @classmethod
    def agent_fault_reward_must_be_finite(cls, value: object) -> object:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value
        if not math.isfinite(numeric):
            raise ValueError("[result].agent_fault_reward must be finite")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> "ResultSection":
        object.__setattr__(
            self,
            "output_root",
            _validate_output_root(self.output_root, "[result].output_root"),
        )
        object.__setattr__(
            self,
            "reward_file",
            _validate_relative_task_path(
                self.reward_file,
                "[result].reward_file",
            ),
        )
        if self.trace_file is not None:
            object.__setattr__(
                self,
                "trace_file",
                _validate_relative_task_path(
                    self.trace_file,
                    "[result].trace_file",
                ),
            )
        reward_key = self.reward_key.strip()
        if not reward_key:
            raise ValueError("[result].reward_key cannot be empty")
        object.__setattr__(self, "reward_key", reward_key)
        if self.subscores_key is not None:
            subscores_key = self.subscores_key.strip()
            if not subscores_key:
                raise ValueError("[result].subscores_key cannot be empty")
            object.__setattr__(self, "subscores_key", subscores_key)
        reports = [
            _validate_capability_name(report, "[result].reports entry")
            for report in self.reports
        ]
        if len(reports) != len(set(reports)):
            raise ValueError("[result].reports entries must be unique")
        object.__setattr__(self, "reports", reports)
        return self


class EnvironmentSection(BaseModel):
    """Runtime requirements shared by Harbor and Boreal export."""

    model_config = ConfigDict(extra="forbid")

    required_resources: TaigaRequiredResources
    storage_mb: int = 50000
    allow_internet: bool = True
    # Base image flavor: "auto" (infer from required_resources), or an explicit
    # compatible flavor. See alignerr_plugin.base_image.
    base_flavor: str = "auto"
    # Opt-in hidden-environment RPC server (env_server) for simulation-style
    # tasks where the agent interacts with a black-box env over /tmp/env.sock:
    #   ""       -> off (default; classic static task)
    #   "env"    -> env server on; agent interacts ONLY through the socket
    #   "hybrid" -> env server on; task also ships static data/ files
    # Requires scorer/data/env.py (make_env) and a public data/env_client.py.
    # See docs/HIDDEN_ENV.md.
    hidden_env: str = ""

    @field_validator("required_resources", mode="before")
    @classmethod
    def required_resources_must_be_taiga_enum(cls, value: object) -> str:
        """Force authors to pick a Taiga-supported resource tier verbatim."""
        return validate_required_resources(value)

    @model_validator(mode="after")
    def validate_environment(self) -> "EnvironmentSection":
        """Validate base flavor compatibility and hidden-env mode."""
        from alignerr_plugin.base_image import resolve_base_flavor_for_resource

        self.required_resources = validate_required_resources(self.required_resources)
        self.base_flavor = (
            (self.base_flavor or "auto").strip().lower().replace("_", "-")
        )
        resolve_base_flavor_for_resource(self.base_flavor, self.required_resources)
        self.hidden_env = (self.hidden_env or "").strip().lower()
        if self.hidden_env not in HIDDEN_ENV_MODES:
            raise ValueError(
                f"[environment].hidden_env must be one of {list(HIDDEN_ENV_MODES)} "
                f"(got {self.hidden_env!r}); use 'env' or 'hybrid' to enable the "
                "hidden-environment RPC server, or '' to disable it."
            )
        return self


class AgentSection(BaseModel):
    """Agent runtime configuration."""

    timeout_sec: int | None = 1800
    # Harbor/Prometheus: non-root uid the agent harness runs as (typically ``agent``).
    user: str | None = None
    # Opt-in native capability resource envelope. Legacy tasks continue using
    # [environment].required_resources when this is omitted.
    resources: ResourceSpec | None = None


class VerifierSection(BaseModel):
    """Verifier runtime configuration.

    ``env`` is the list of environment variables the verifier (the
    in-image grader) needs at runtime — Harbor's `harbor run` requests
    approval for each. For LLM-judged criteria this should include
    ``ANTHROPIC_API_KEY``.
    """

    timeout_sec: int = 600
    env: list[str] = Field(default_factory=list)
    sandbox: bool = False  # opt-in: grade in a subprocess (e.g. for nvproxy cleanup)
    # Harbor/Prometheus: root verifier runs grading with access to /mcp_server/data.
    user: str | None = None
    resources: ResourceSpec | None = None
    capabilities: list[str] = Field(default_factory=list)

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, value: list[str]) -> list[str]:
        normalized = [capability.strip().upper() for capability in value]
        if any(not capability for capability in normalized):
            raise ValueError("verifier capabilities cannot be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("verifier capabilities must be unique")
        unsupported = sorted(set(normalized) - _ALLOWED_SERVICE_CAPABILITIES)
        if unsupported:
            raise ValueError(
                "verifier requests forbidden capabilities "
                f"{unsupported}; only SYS_PTRACE is allowed"
            )
        return normalized


class GroundTruthSection(BaseModel):
    """Oracle solution and reviewer artifact requirements.

    ``solution/solve.sh`` is the executable ground-truth submission. The render
    command runs after that solution has produced its normal outputs and must
    create the declared reviewer video artifacts.
    """

    render_command: str = ""
    render_outputs: list[OutputSpec] = Field(default_factory=list)
    score_epsilon: float = 1e-9
    continuous_score_epsilon: float = 0.05
    # Opt-in: run the oracle solve + grade + render INSIDE the built task image
    # instead of on the host. Required for tasks whose grader/renderer depend on
    # engines that live only in the base image (OpenFOAM, SU2, Meep, OpenROAD).
    in_container: bool = False
    # Highest score a trivial / no-op / prompt-example submission is allowed to
    # earn. The validator grades an empty submission (and, when extractable, the
    # prompt's example) through the real scorer and fails the task if either
    # lands above this ceiling -- the signal that "do nothing" or "copy the
    # example" out-scores genuine work (a dominant reward-hacking failure mode).
    max_trivial_score: float = 0.5
    # Highest score an empty/no-op submission may earn on a
    # continuous_scoring_function task. Continuous graders must anchor
    # "no attempt" to 0; this tolerance only absorbs float/curve noise.
    zero_anchor_epsilon: float = 0.01


class ReferenceSection(BaseModel):
    """Local reference-solution runner configuration (author iteration loop).

    In-container execution, persistent cache, and optional train/grade
    separation.
    """

    execution: Literal["auto", "host", "container"] = "auto"
    cache_dir: str = ".alignerr/reference_cache/output"
    entrypoint: str = "solve.sh"
    proof_mode: Literal["auto", "artifact", "execute"] = "auto"


# Hour-scale per-stage runner timeouts (seconds) that **ML tasks** are pinned to.
# Single source of truth for the ML "fair-chance" timeouts: the Taiga exporter
# force-pins ``task_type == "ml"`` to these. Minutes-scale caps make Boreal/Taiga
# finish before an agent (or a human building a reference solution) can plausibly
# solve an ML task. Only ML tasks are pinned; other task types keep the
# (author-overridable) RunnerTimeouts defaults below.
ML_SETUP_TIMEOUT_SEC = 7200  # 2h — environment/setup phase
ML_GRADING_TIMEOUT_SEC = 10800  # 3h — Taiga's maximum grading timeout
ML_TOOL_TIMEOUT_SEC = 21600  # 6h — per tool-call budget
ML_MAX_EPISODE_SEC = 21600  # 6h — job-level episode wall-clock


class RunnerTimeouts(BaseModel):
    """Per-stage Boreal/Taiga runner timeouts (seconds).

    These are **Taiga/Boreal-only** — Harbor ignores them (it exposes its own
    resource controls via ``[environment]``). Each field drives a distinct field
    in the exported Taiga/Boreal payload (see ``exporters/taiga.py``):

    ==================  ==========================  ==================================
    RunnerTimeouts      Taiga payload field         Scope
    ==================  ==========================  ==================================
    ``setup_sec``       ``setup_timeout_seconds``   per-problem entry
    ``grading_sec``     ``grading_timeout_seconds`` per-problem entry **and** the
                                                    in-image rubric subprocess
                                                    (``extra_fields``)
    ``tool_sec``        ``tool_timeout_seconds``    per-problem entry
    ``max_episode_sec`` ``max_timeout_seconds``     job-level (``None`` = unlimited)
    ==================  ==========================  ==================================

    The defaults below apply to non-ML task types and stay author-overridable.
    ``task_type == "ml"`` ignores these entirely: the Taiga exporter force-pins
    ml tasks to the hour-scale ``ML_*_TIMEOUT_SEC`` values above regardless of
    author input, because minutes-scale caps silently fail real ML runs.
    """

    setup_sec: int = 600
    grading_sec: int = 600
    tool_sec: int = 120
    max_episode_sec: int | None = 3600  # None = unlimited


class RunnerConfig(BaseModel):
    """Boreal job + episode-level runtime configuration.

    These knobs become per-problem and job-level fields in the Boreal payload
    built by the exporter. They have no effect on Harbor (which exposes its
    own resource controls via `[environment]`).
    """

    model_config = ConfigDict(extra="allow")

    attempts: int = 3
    turn_limit: int | None = 1430  # None = unlimited
    max_ctx: int = 1_000_000
    context_mode: Literal["none", "autocompact", "memory"] = "autocompact"
    priority: Literal["low", "high"] = "high"
    iteration_order: Literal["problems_first", "attempts_first"] = "problems_first"
    checkpoint_ttl: str | None = None  # e.g. "30d"
    serialize_restore_test_interval: int | None = None
    api_model_name: str = "claude-fable-5"
    required_tools: list[str] = Field(
        default_factory=lambda: ["bash", "str_replace_editor", "tmux"]
    )
    # In-container Anthropic API access is a reward-hack surface; opt in only
    # for tasks that explicitly require agent-side model calls.
    enable_anthropic_api: bool = False
    container_runtime: Literal["firecracker", "docker"] = "firecracker"

    timeouts: RunnerTimeouts = Field(default_factory=RunnerTimeouts)


class Difficulty(BaseModel):
    """Task classification metadata.

    These fields drive validation, routing, dashboards, and Taiga metadata.
    None of these affect grading directly.
    """

    # Required enum: ml / mujoco / cfd / structures.
    task_type: str = ""
    # Required enum scoped by task_type, e.g. model_environment_construction
    # for mujoco or scientific_discovery_computational_science for ml.
    domain: str = ""
    # Required enum: continuous_scoring_function / multi_deterministic_rubrics.
    reward_type: str = ""
    # Permissive dataset license (SPDX id). Required when task_type == "ml";
    # optional otherwise. See alignerr_plugin.task_metadata.LICENSES for the
    # allowed set. Preserved verbatim (not normalized) for downstream display.
    license: str = ""
    # Provenance pointer the licensing reviewer verifies. Required for ml tasks:
    # the upstream http(s) URL where the license was confirmed, or -- when
    # license == "self_generated" -- a short reason the task generates its own data.
    license_source: str = ""
    is_impossible: bool = False  # task is intentionally unachievable (red-teaming)

    @model_validator(mode="after")
    def validate_metadata_enums(self) -> "Difficulty":
        self.task_type = normalize_enum_value(self.task_type)
        self.domain = normalize_enum_value(self.domain)
        self.reward_type = normalize_enum_value(self.reward_type)
        issues = metadata_validation_issues(
            task_type=self.task_type,
            domain=self.domain,
            reward_type=self.reward_type,
            license_id=self.license,
            license_source=self.license_source,
        )
        if issues:
            raise ValueError("; ".join(issues))
        return self


class PreloadedFile(BaseModel):
    """A read-only artifact mounted into the container at deploy time instead of
    baked into the image.

    Declare exactly one source:

    * ``source`` -- a directory tree under the task dir, packed into a
      content-addressed squashfs and uploaded once (shared/deduped across tasks);
    * ``hf_repo`` -- a Hugging Face repo (optionally pinned to ``hf_revision``,
      narrowed by ``allow_patterns`` / ``ignore_patterns``), fetched and packed
      in hub-cache layout so ``from_pretrained`` resolves it offline.

    ``mount_path`` is optional for ``hf_repo`` mounts: it defaults to the
    canonical hub-cache location derived from the repo id and ``repo_type``.

    The deploy pipeline (``scripts/sync_mount.sh``) packs/uploads and stamps the
    concrete ``remote_path`` into ``.alignerr/preloaded_files.json``; the
    exporter emits that manifest. This is how large datasets and model weights
    avoid image bloat (one shared object, not one copy per task image).
    """

    model_config = ConfigDict(extra="forbid")

    source: str = ""
    hf_repo: str = ""
    hf_revision: str = ""
    # Selects the hub-cache folder prefix ("models--" / "datasets--"), which is
    # what makes an offline from_pretrained / load_dataset resolve the mount.
    repo_type: str = "model"
    # Narrow the download; omitted allow_patterns means the whole repo. Packed
    # objects are keyed by pattern too, so two tasks taking different slices of
    # one repo do not collide in the shared cache.
    allow_patterns: list[str] = Field(default_factory=list)
    ignore_patterns: list[str] = Field(default_factory=list)
    mount_path: str = ""
    read_only: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "PreloadedFile":
        if bool(self.source) == bool(self.hf_repo):
            raise ValueError(
                "[[preloaded_files]] requires exactly one of `source` or `hf_repo`"
            )
        if self.hf_repo:
            self.hf_repo = self.hf_repo.strip()
            if not _HF_REPO_ID_RE.match(self.hf_repo):
                raise ValueError(
                    "[[preloaded_files]].hf_repo must look like 'org/name'; "
                    f"got {self.hf_repo!r}"
                )
            if self.repo_type not in HF_REPO_TYPES:
                raise ValueError(
                    f"[[preloaded_files]].repo_type must be one of "
                    f"{list(HF_REPO_TYPES)}; got {self.repo_type!r}"
                )
            if not self.mount_path:
                self.mount_path = hf_hub_mount_path(self.hf_repo, self.repo_type)
        else:
            if self.allow_patterns or self.ignore_patterns:
                raise ValueError(
                    "[[preloaded_files]].allow_patterns/ignore_patterns only apply "
                    "to `hf_repo` mounts"
                )
            if not self.mount_path:
                raise ValueError(
                    "[[preloaded_files]].mount_path is required for `source` mounts"
                )
        if not PurePosixPath(self.mount_path).is_absolute():
            raise ValueError("[[preloaded_files]].mount_path must be an absolute path")
        return self


class Hint(BaseModel):
    """A single Taiga hint shipped with the task."""

    text: str
    enabled: bool = True
    spoiler_level: float = Field(default=0.5, ge=0.0, le=1.0)


class DeliverySection(BaseModel):
    """Post-CI delivery destination.

    Mothership reads this to select delivery. Omit the section to keep the
    historical default of Taiga delivery. Prometheus CFD/structures retain
    ``platform = "prometheus"`` while mothership also derives a parallel Taiga
    mirror for those numerical-solver task types.
    """

    platform: Literal["taiga", "prometheus"] = "taiga"
    eval: bool = False

    @field_validator("platform", mode="before")
    @classmethod
    def platform_must_be_canonical(cls, value: object) -> str:
        return normalize_delivery_platform(value)


class TaskToml(BaseModel):
    """Parsed task.toml.

    Sections beyond `[task]` mostly have sensible defaults. `[difficulty]` is
    required in practice because task_type, domain, and reward_type are enum
    metadata used by validation and dashboards.

    Unknown top-level sections are rejected, so a misspelled section fails loudly
    instead of being silently dropped along with the checks it drives.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.1"
    task: TaskSection
    environment: EnvironmentSection = Field(default_factory=EnvironmentSection)
    agent: AgentSection = Field(default_factory=AgentSection)
    verifier: VerifierSection = Field(default_factory=VerifierSection)
    ground_truth: GroundTruthSection = Field(default_factory=GroundTruthSection)
    reference: ReferenceSection = Field(default_factory=ReferenceSection)
    outputs: list[OutputSpec] = Field(default_factory=list)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    difficulty: Difficulty = Field(default_factory=Difficulty)
    hint: list[Hint] = Field(default_factory=list)
    preloaded_files: list[PreloadedFile] = Field(default_factory=list)
    delivery: DeliverySection = Field(default_factory=DeliverySection)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Native long-horizon/software capabilities are opt-in. Their absence keeps
    # the historical one-image task contract and v1.1 parsing behavior.
    workspace: WorkspaceSection | None = None
    artifacts: list[ArtifactSpec] = Field(default_factory=list)
    services: list[ServiceSpec] = Field(default_factory=list)
    volumes: list[NamedVolume] = Field(default_factory=list)
    captures: list[CaptureSpec] = Field(default_factory=list)
    mcp_servers: list[MCPServerSpec] = Field(default_factory=list)
    evaluation: EvaluationSection | None = None
    gates: list[GateSpec] = Field(default_factory=list)
    reports: list[ReportSpec] = Field(default_factory=list)
    result: ResultSection | None = None

    @model_validator(mode="after")
    def _validate_cross_section(self) -> "TaskToml":
        issues: list[str] = []
        issues.extend(self._hidden_env_issues())
        issues.extend(self._preloaded_files_issues())
        issues.extend(self._capability_issues())
        if issues:
            raise ValueError("; ".join(issues))
        return self

    def _capability_issues(self) -> list[str]:
        """Validate references and uniqueness across native capability sections."""
        issues: list[str] = []
        service_names = [service.name for service in self.services]
        service_by_name = {service.name: service for service in self.services}
        duplicate_services = sorted(
            name for name in set(service_names) if service_names.count(name) > 1
        )
        for name in duplicate_services:
            issues.append(f"duplicate service name {name!r}")

        if self.services:
            main_services = [
                service.name for service in self.services if service.role == "main"
            ]
            if len(main_services) != 1:
                issues.append(
                    "services must declare exactly one role = 'main' "
                    f"(found {len(main_services)})"
                )
            verifier_services = [
                service.name for service in self.services if service.role == "verifier"
            ]
            if len(verifier_services) > 1:
                issues.append(
                    "services may declare at most one role = 'verifier' "
                    f"(found {len(verifier_services)})"
                )
            if verifier_services:
                if self.result is None:
                    issues.append(
                        f"verifier service {verifier_services[0]!r} requires a "
                        "[result] section with explicit reward_file and reward_key"
                    )
                else:
                    missing_reward_fields = sorted(
                        {"reward_file", "reward_key"} - self.result.model_fields_set
                    )
                    if missing_reward_fields:
                        issues.append(
                            f"verifier service {verifier_services[0]!r} requires "
                            "explicit [result] fields: "
                            + ", ".join(missing_reward_fields)
                        )
            valid_services = set(service_names)
        else:
            # The legacy single-image runtime has an implicit main service.
            valid_services = {"main"}

        volume_names = [volume.name for volume in self.volumes]
        duplicate_volumes = sorted(
            name for name in set(volume_names) if volume_names.count(name) > 1
        )
        for name in duplicate_volumes:
            issues.append(f"duplicate volume name {name!r}")
        valid_volumes = set(volume_names)

        artifact_names = [artifact.name for artifact in self.artifacts]
        for name in sorted(
            name for name in set(artifact_names) if artifact_names.count(name) > 1
        ):
            issues.append(f"duplicate artifact name {name!r}")
        for artifact in self.artifacts:
            if artifact.service not in valid_services:
                issues.append(
                    f"artifact {artifact.name!r} references unknown service "
                    f"{artifact.service!r}"
                )

        capture_names = [capture.name for capture in self.captures]
        for name in sorted(
            name for name in set(capture_names) if capture_names.count(name) > 1
        ):
            issues.append(f"duplicate capture name {name!r}")
        for capture in self.captures:
            if capture.service not in valid_services:
                issues.append(
                    f"capture {capture.name!r} references unknown service "
                    f"{capture.service!r}"
                )

        for service in self.services:
            for dependency in service.depends_on:
                if dependency.service not in valid_services:
                    issues.append(
                        f"service {service.name!r} depends on unknown service "
                        f"{dependency.service!r}"
                    )
                    continue
                if dependency.service == service.name:
                    issues.append(f"service {service.name!r} cannot depend on itself")
                    continue
                target = service_by_name.get(dependency.service)
                if (
                    dependency.condition == "healthy"
                    and target is not None
                    and target.healthcheck is None
                ):
                    issues.append(
                        f"service {service.name!r} requires {dependency.service!r} "
                        "to be healthy, but that service has no healthcheck"
                    )
                if (
                    dependency.condition == "completed"
                    and target is not None
                    and target.role != "init"
                ):
                    issues.append(
                        f"service {service.name!r} uses completed dependency "
                        f"{dependency.service!r}, but it is not an init service"
                    )

            if (
                service.network_share_target is not None
                and service.network_share_target not in valid_services
            ):
                issues.append(
                    f"service {service.name!r} shares the network of unknown service "
                    f"{service.network_share_target!r}"
                )
            if service.network_share_target == service.name:
                issues.append(
                    f"service {service.name!r} cannot share its own network namespace"
                )

            for mount in service.volumes:
                if mount.volume not in valid_volumes:
                    issues.append(
                        f"service {service.name!r} references unknown volume "
                        f"{mount.volume!r}"
                    )

        mcp_servers = list(self.mcp_servers)
        mcp_names = [server.name for server in mcp_servers]
        for name in sorted(
            name for name in set(mcp_names) if mcp_names.count(name) > 1
        ):
            issues.append(f"duplicate MCP server name {name!r}")
        builtin_names = list(self.runner.required_tools)
        for name in sorted(
            name for name in set(builtin_names) if builtin_names.count(name) > 1
        ):
            issues.append(f"duplicate built-in tool name {name!r}")
        for name in sorted(set(builtin_names) & set(mcp_names)):
            issues.append(f"tool name {name!r} is declared as both built-in and MCP")
        for server in mcp_servers:
            if server.service is not None and server.service not in valid_services:
                issues.append(
                    f"MCP server {server.name!r} references unknown service "
                    f"{server.service!r}"
                )
            for dependency in server.depends_on:
                if dependency not in valid_services:
                    issues.append(
                        f"MCP server {server.name!r} depends on unknown service "
                        f"{dependency!r}"
                    )
            readiness = server.readiness
            if (
                isinstance(readiness, CommandReadiness)
                and readiness.service is not None
                and readiness.service not in valid_services
            ):
                issues.append(
                    f"MCP server {server.name!r} readiness references unknown "
                    f"service {readiness.service!r}"
                )

        report_names = [report.name for report in self.reports]
        for name in sorted(
            name for name in set(report_names) if report_names.count(name) > 1
        ):
            issues.append(f"duplicate report name {name!r}")
        valid_reports = set(report_names)

        gate_names = [gate.name for gate in self.gates]
        for name in sorted(
            name for name in set(gate_names) if gate_names.count(name) > 1
        ):
            issues.append(f"duplicate gate name {name!r}")
        for gate in self.gates:
            if gate.report is not None and gate.report not in valid_reports:
                issues.append(
                    f"gate {gate.name!r} references unknown report {gate.report!r}"
                )
        if self.result is not None:
            for report_name in self.result.reports:
                if report_name not in valid_reports:
                    issues.append(f"result references unknown report {report_name!r}")

        if self.workspace is not None:
            output_root = self.result.output_root if self.result else "/tmp/output"
            if _is_path_within(self.workspace.root, output_root) or _is_path_within(
                output_root, self.workspace.root
            ):
                issues.append(
                    "[workspace].root and the result output root must not overlap"
                )
        return issues

    def _hidden_env_issues(self) -> list[str]:
        """Cross-field checks for the hidden-environment RPC gate.

        Applies to every task type, not just ml: the env server is a property of
        ``[environment]``, so a non-ml task that switches it on is subject to the
        same tier constraint.
        """
        if not self.environment.hidden_env:
            return []
        if is_tpu_resource(self.environment.required_resources):
            return [
                (
                    "[environment].hidden_env (the RPC env server) cannot run on a "
                    f"TPU tier; got {self.environment.required_resources!r}"
                )
            ]
        return []

    def _preloaded_files_issues(self) -> list[str]:
        """Reject HF mounts on TPU tiers and duplicate repo/mount declarations."""
        issues: list[str] = []
        hf_entries = [entry for entry in self.preloaded_files if entry.hf_repo]
        if hf_entries and is_tpu_resource(self.environment.required_resources):
            issues.append(
                "[[preloaded_files]].hf_repo mounts are not supported on a TPU tier "
                f"(got {self.environment.required_resources!r})"
            )
        seen_repos: set[tuple[str, str]] = set()
        for entry in hf_entries:
            key = (entry.hf_repo, entry.repo_type)
            if key in seen_repos:
                issues.append(
                    f"duplicate [[preloaded_files]] hf_repo {entry.hf_repo!r} "
                    f"(repo_type {entry.repo_type!r}); declare it once"
                )
            seen_repos.add(key)
        seen_mounts: set[str] = set()
        for entry in self.preloaded_files:
            if entry.mount_path in seen_mounts:
                issues.append(
                    f"duplicate [[preloaded_files]].mount_path {entry.mount_path!r}"
                )
            seen_mounts.add(entry.mount_path)
        return issues


class ProblemMetadata(BaseModel):
    """Alignerr metadata.json."""

    benchmark: str = "taiga_task"
    problem_data: dict[str, Any]


class StageResult(BaseModel):
    """Result for one validation stage."""

    passed: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int = 0


class ValidationResult(BaseModel):
    """Full validation result emitted by the task validator."""

    problem_id: str
    benchmark: str
    status: Literal["valid", "invalid"]
    stages: dict[str, StageResult]
    metadata: dict[str, Any] = Field(default_factory=dict)
