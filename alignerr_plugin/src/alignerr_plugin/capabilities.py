"""Defensive accessors for optional multi-service task capabilities.

The capability schema is intentionally consumed through ``getattr`` and
``model_dump`` rather than concrete schema classes.  This keeps packaging and
format exporters usable while the authoring schema evolves independently.
"""

from __future__ import annotations

import copy
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

_PINNED_IMAGE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}\Z", re.IGNORECASE)
_SERVICE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RESOURCE_FIELDS = (
    "build_timeout_sec",
    "cpus",
    "memory_mb",
    "network_mode",
    "storage_mb",
    "gpus",
    "gpu_types",
    "tpu",
)
_CAPABILITY_SECTION_NAMES = (
    "capabilities",
    "capability",
    "task_capsule",
    "capsule",
)
_MISSING = object()
_ALLOWED_CAPABILITIES = frozenset({"SYS_PTRACE"})
_FORBIDDEN_COMPOSE_FIELDS = frozenset(
    {
        "devices",
        "privileged",
        "volumes_from",
    }
)


def model_dump(value: Any) -> dict[str, Any]:
    """Return a plain mapping without depending on a particular model class."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            result = dump(mode="python", exclude_none=True)
        except TypeError:
            result = dump()
        return dict(result) if isinstance(result, Mapping) else {}
    attributes = getattr(value, "__dict__", None)
    return dict(attributes) if isinstance(attributes, Mapping) else {}


def _value(value: Any, *names: str, default: Any = None) -> Any:
    mapping = model_dump(value)
    for name in names:
        if name in mapping:
            return mapping[name]
        attribute = getattr(value, name, _MISSING)
        if attribute is not _MISSING:
            return attribute
    return default


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _first_non_none(values: Sequence[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _capability_section(task: Any) -> Any:
    for name in _CAPABILITY_SECTION_NAMES:
        section = _value(task, name)
        if section is not None:
            return section
    environment = _value(task, "environment")
    for name in _CAPABILITY_SECTION_NAMES:
        section = _value(environment, name)
        if section is not None:
            return section
    runtime = _value(task, "runtime")
    if _value(runtime, "services") is not None:
        return runtime
    return None


def _service_container(task: Any) -> Any:
    section = _capability_section(task)
    return _first_non_none(
        [
            _value(section, "services"),
            _value(task, "services"),
            _value(_value(task, "environment"), "services"),
            _value(_value(task, "runtime"), "services"),
        ]
    )


def _normalized_role(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        value = next((item for item in value if item), "sidecar")
    role = str(value or "sidecar").strip().lower().replace("_", "-")
    if role in {"agent", "main", "workspace", "primary"}:
        return "agent"
    if role in {"verifier", "grader", "test", "tests"}:
        return "verifier"
    return role or "sidecar"


@dataclass(frozen=True)
class ServiceBuild:
    """A source build that trusted packaging may materialize."""

    context: str
    dockerfile: str = "Dockerfile"
    args: tuple[tuple[str, str], ...] = ()
    target: str | None = None
    platform: str | None = None
    pull: bool = False
    no_cache: bool = False


@dataclass(frozen=True)
class ServiceSpec:
    """Schema-independent service definition."""

    name: str
    role: str
    image: str | None = None
    build: ServiceBuild | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    compose: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def harbor_name(self) -> str:
        return "main" if self.role == "agent" else self.name


@dataclass(frozen=True)
class CapabilityConfig:
    """Normalized capability fields shared by the exporters."""

    services: tuple[ServiceSpec, ...]
    artifacts: tuple[dict[str, Any], ...] = ()
    captures: tuple[dict[str, Any], ...] = ()
    mcp_servers: tuple[dict[str, Any], ...] = ()
    verifier_mcp_servers: tuple[dict[str, Any], ...] = ()
    volumes: tuple[dict[str, Any], ...] = ()
    agent_resources: dict[str, Any] = field(default_factory=dict)
    verifier_resources: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_service(self) -> ServiceSpec | None:
        return next(
            (service for service in self.services if service.role == "agent"), None
        )

    @property
    def verifier_service(self) -> ServiceSpec | None:
        return next(
            (service for service in self.services if service.role == "verifier"),
            None,
        )


def _build_spec(service: Any) -> ServiceBuild | None:
    raw_build = _value(service, "build")
    if raw_build in (None, False):
        return None
    if raw_build is True:
        raw_build = {}
    if isinstance(raw_build, str):
        return ServiceBuild(context=raw_build)

    context = str(
        _first_non_none(
            [
                _value(raw_build, "context", "path"),
                _value(service, "context", "build_context"),
                ".",
            ]
        )
    )
    dockerfile = str(
        _first_non_none(
            [
                _value(raw_build, "dockerfile", "file"),
                _value(service, "dockerfile"),
                "Dockerfile",
            ]
        )
    )
    raw_args = _value(raw_build, "args", "build_args", default={})
    args = tuple(
        sorted((str(key), str(value)) for key, value in model_dump(raw_args).items())
    )
    target = _value(raw_build, "target")
    platform = _value(raw_build, "platform")
    return ServiceBuild(
        context=context,
        dockerfile=dockerfile,
        args=args,
        target=str(target) if target else None,
        platform=str(platform) if platform else None,
        pull=bool(_value(raw_build, "pull", default=False)),
        no_cache=bool(_value(raw_build, "no_cache", default=False)),
    )


def _seconds(value: Any) -> str:
    return f"{float(value):g}s"


def _compose_dependencies(value: Any) -> dict[str, dict[str, str]]:
    condition_aliases = {
        "started": "service_started",
        "service_started": "service_started",
        "healthy": "service_healthy",
        "service_healthy": "service_healthy",
        "completed": "service_completed_successfully",
        "service_completed_successfully": "service_completed_successfully",
    }
    dependencies: dict[str, dict[str, str]] = {}
    if isinstance(value, Mapping):
        rows = value.items()
    else:
        rows = ((None, row) for row in _items(value))
    for keyed_name, row in rows:
        if isinstance(row, str):
            name = str(keyed_name or row)
            condition = "service_started"
        else:
            raw = model_dump(row)
            name = str(
                _first_non_none([raw.get("service"), raw.get("name"), keyed_name, ""])
            )
            raw_condition = raw.get("condition") or "started"
            condition = condition_aliases.get(str(raw_condition), str(raw_condition))
        if name:
            dependencies[name] = {"condition": condition}
    return dependencies


def _compose_projection(service: Any) -> dict[str, Any]:
    raw = model_dump(service)
    nested = model_dump(_value(service, "compose"))
    for field_name in sorted(_FORBIDDEN_COMPOSE_FIELDS):
        if field_name in raw or field_name in nested:
            raise ValueError(
                f"service {raw.get('name')!r} declares forbidden compose field "
                f"{field_name!r}"
            )
    raw_capabilities = _first_non_none(
        [
            nested.get("cap_add"),
            raw.get("cap_add"),
            raw.get("capabilities"),
        ]
    )
    capabilities = [
        str(item).strip().upper() for item in _items(raw_capabilities) if str(item)
    ]
    unsupported_capabilities = sorted(set(capabilities) - _ALLOWED_CAPABILITIES)
    if unsupported_capabilities:
        raise ValueError(
            f"service {raw.get('name')!r} requests forbidden capabilities "
            f"{unsupported_capabilities}; only SYS_PTRACE is allowed"
        )
    for value in _walk_values(raw):
        if isinstance(value, str) and "docker.sock" in value.lower():
            raise ValueError(
                f"service {raw.get('name')!r} must not reference docker.sock"
            )
    projected: dict[str, Any] = {}

    command = _first_non_none([nested.get("command"), raw.get("command")])
    if command is not None:
        projected["command"] = copy.deepcopy(command)
    entrypoint = _first_non_none([nested.get("entrypoint"), raw.get("entrypoint")])
    if entrypoint is not None:
        projected["entrypoint"] = copy.deepcopy(entrypoint)
    environment = _first_non_none(
        [
            nested.get("environment"),
            nested.get("env"),
            raw.get("environment"),
            raw.get("env"),
        ]
    )
    if environment is not None:
        projected["environment"] = copy.deepcopy(model_dump(environment))
    user = _first_non_none([nested.get("user"), raw.get("user")])
    if user is not None:
        projected["user"] = user
    workdir = _first_non_none(
        [
            nested.get("working_dir"),
            nested.get("workdir"),
            raw.get("working_dir"),
            raw.get("workdir"),
        ]
    )
    if workdir:
        projected["working_dir"] = workdir

    published: list[str] = []
    exposed: list[str] = []
    for port in _items(_first_non_none([nested.get("ports"), raw.get("ports")])):
        if isinstance(port, (str, int)):
            published.append(str(port))
            continue
        port_data = model_dump(port)
        container_port = port_data.get("container_port")
        if container_port is None:
            continue
        protocol = str(port_data.get("protocol") or "tcp")
        host_port = port_data.get("host_port")
        suffix = "" if protocol == "tcp" else f"/{protocol}"
        if host_port is None:
            exposed.append(f"{container_port}{suffix}")
        else:
            published.append(f"{host_port}:{container_port}{suffix}")
    if published:
        projected["ports"] = published
    if exposed:
        projected["expose"] = exposed

    dependencies = _compose_dependencies(
        _first_non_none([nested.get("depends_on"), raw.get("depends_on")])
    )
    if dependencies:
        projected["depends_on"] = dependencies

    healthcheck = model_dump(
        _first_non_none([nested.get("healthcheck"), raw.get("healthcheck")])
    )
    if healthcheck:
        health_command = healthcheck.get("test") or healthcheck.get("command")
        if isinstance(health_command, str):
            test = ["CMD-SHELL", health_command]
        else:
            test = [str(item) for item in _items(health_command)]
            if test and test[0] not in {"CMD", "CMD-SHELL", "NONE"}:
                test.insert(0, "CMD")
        projected["healthcheck"] = {
            "test": test,
            "interval": _seconds(healthcheck.get("interval_sec", 5)),
            "timeout": _seconds(healthcheck.get("timeout_sec", 5)),
            "retries": int(healthcheck.get("retries", 3)),
        }
        if healthcheck.get("start_period_sec") is not None:
            projected["healthcheck"]["start_period"] = _seconds(
                healthcheck["start_period_sec"]
            )
        if healthcheck.get("start_interval_sec") is not None:
            projected["healthcheck"]["start_interval"] = _seconds(
                healthcheck["start_interval_sec"]
            )

    volume_values = _first_non_none([nested.get("volumes"), raw.get("volumes")])
    volumes: list[Any] = []
    for volume in _items(volume_values):
        if isinstance(volume, str):
            raise TypeError(
                f"service {raw.get('name')!r} volume mounts must use declared "
                "named-volume objects; host binds are forbidden"
            )
        volume_data = model_dump(volume)
        source = volume_data.get("volume") or volume_data.get("source")
        target = volume_data.get("target")
        if source and target:
            if "/" in str(source) or str(source).startswith((".", "~")):
                raise ValueError(
                    f"service {raw.get('name')!r} volume source {source!r} "
                    "looks like a host bind"
                )
            mode = volume_data.get("mode") or "rw"
            volumes.append(f"{source}:{target}:{mode}")
    if volumes:
        projected["volumes"] = volumes

    network_mode = str(raw.get("network_mode") or "bridge")
    if network_mode == "none":
        projected["network_mode"] = "none"
    elif network_mode == "share" and raw.get("network_share_target"):
        projected["network_mode"] = f"service:{raw['network_share_target']}"
    aliases = raw.get("network_aliases")
    if aliases and "network_mode" not in projected:
        projected["networks"] = {
            "default": {"aliases": [str(alias) for alias in _items(aliases)]}
        }
    if capabilities:
        projected["cap_add"] = capabilities
    if raw.get("shm_mb") is not None:
        projected["shm_size"] = f"{int(raw['shm_mb'])}m"
    restart = raw.get("restart")
    if restart and restart != "no":
        projected["restart"] = restart
    service_resources = project_resource_fields(raw.get("resources"))
    limits: dict[str, Any] = {}
    if service_resources.get("cpus") is not None:
        limits["cpus"] = service_resources["cpus"]
    if service_resources.get("memory_mb") is not None:
        limits["memory"] = f"{service_resources['memory_mb']}M"
    if limits:
        projected["deploy"] = {"resources": {"limits": limits}}
    if service_resources.get("gpus"):
        projected["gpus"] = service_resources["gpus"]
    resource_data = model_dump(raw.get("resources"))
    if resource_data.get("platform"):
        projected["platform"] = str(resource_data["platform"])

    resource_network = str(resource_data.get("network") or "isolated").lower()
    if resource_network == "none" and "network_mode" not in projected:
        projected.pop("networks", None)
        projected["network_mode"] = "none"
    elif resource_network == "isolated" and "network_mode" not in projected:
        aliases = [str(alias) for alias in _items(raw.get("network_aliases"))]
        projected["networks"] = {
            "alignerr-isolated": {"aliases": aliases} if aliases else {}
        }

    for key in (
        "read_only",
        "tmpfs",
        "stop_grace_period",
        "init",
        "cap_drop",
        "sysctls",
        "extra_hosts",
        "hostname",
        "labels",
    ):
        value = _first_non_none([nested.get(key), raw.get(key)])
        if value is not None:
            projected[key] = copy.deepcopy(value)
    return projected


def implicit_agent_service(
    *,
    context: str,
    resources: Mapping[str, Any],
    network: str | None = None,
    platform: str | None = None,
) -> ServiceSpec:
    """Build the implicit main service through the normal Compose projection."""
    resource_data = dict(resources)
    resource_data.pop("network_mode", None)
    if network:
        resource_data["network"] = network
    if platform:
        resource_data["platform"] = platform
    raw = {
        "name": "main",
        "role": "main",
        "build": {"context": context, "platform": platform},
        "resources": resource_data,
    }
    return ServiceSpec(
        name="main",
        role="agent",
        build=ServiceBuild(context=context, platform=platform),
        resources=dict(resources),
        compose=_compose_projection(raw),
        raw=raw,
    )


def _walk_values(value: Any) -> list[Any]:
    values: list[Any] = [value]
    if isinstance(value, Mapping):
        for item in value.values():
            values.extend(_walk_values(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            values.extend(_walk_values(item))
    return values


def _service_entries(container: Any) -> list[tuple[str | None, Any]]:
    if container is None:
        return []
    mapping = model_dump(container)
    if mapping and not any(
        key in mapping for key in ("name", "role", "kind", "image", "build")
    ):
        return [(str(name), service) for name, service in mapping.items()]
    return [(None, service) for service in _items(container)]


def resolve_services(task: Any) -> list[ServiceSpec]:
    """Resolve optional service declarations without importing schema classes."""
    services: list[ServiceSpec] = []
    seen: set[str] = set()
    for keyed_name, raw_service in _service_entries(_service_container(task)):
        raw = model_dump(raw_service)
        name = str(
            _first_non_none(
                [
                    _value(raw_service, "name", "id", "service"),
                    keyed_name,
                    "",
                ]
            )
        ).strip()
        if not name:
            raise ValueError("capability service is missing a name")
        if not _SERVICE_NAME_RE.fullmatch(name):
            raise ValueError(
                f"invalid capability service name {name!r}; use letters, digits, "
                "'.', '_' or '-'"
            )
        if name in seen:
            raise ValueError(f"duplicate capability service name {name!r}")
        seen.add(name)

        raw_role = _value(raw_service, "role", "kind", "roles")
        if raw_role is None and name.lower() in {"main", "agent", "workspace"}:
            raw_role = "agent"
        elif raw_role is None and name.lower() in {"verifier", "grader", "tests"}:
            raw_role = "verifier"
        role = _normalized_role(raw_role)
        image = _value(raw_service, "image", "image_ref", "docker_image")
        image = str(image).strip() if image else None
        build = _build_spec(raw_service)
        if bool(image) == (build is not None):
            raise ValueError(
                f"service {name!r} requires exactly one of a pinned image or build"
            )
        resources = project_resource_fields(_value(raw_service, "resources"))
        compose = _compose_projection(raw_service)
        target_platform = (
            build.platform
            if build is not None and build.platform
            else resources.get("platform")
        )
        if target_platform:
            compose["platform"] = str(target_platform)
        services.append(
            ServiceSpec(
                name=name,
                role=role,
                image=image,
                build=build,
                resources=resources,
                compose=compose,
                raw=raw,
            )
        )

    agent_services = [service for service in services if service.role == "agent"]
    verifier_services = [service for service in services if service.role == "verifier"]
    if len(agent_services) > 1:
        raise ValueError("capability tasks may declare only one agent service")
    if len(verifier_services) > 1:
        raise ValueError("capability tasks may declare only one verifier service")
    compose_names: dict[str, str] = {}
    for service in services:
        if service.role != "agent" and service.name == "main":
            raise ValueError("service name 'main' is reserved for the agent service")
        if service.role != "verifier" and service.name == "verifier":
            raise ValueError(
                "service name 'verifier' is reserved for the verifier service"
            )
        previous = compose_names.setdefault(service.harbor_name, service.name)
        if previous != service.name:
            raise ValueError(
                f"services {previous!r} and {service.name!r} map to the same "
                f"compose name {service.harbor_name!r}"
            )
    _validate_trust_boundaries(task, services)
    return services


def is_capability_task(task: Any) -> bool:
    """Return whether a task opts into any native capability contract section."""
    if _service_entries(_service_container(task)):
        return True
    return any(
        bool(_value(task, name))
        for name in (
            "workspace",
            "artifacts",
            "captures",
            "mcp_servers",
            "evaluation",
        )
    )


def _trusted_post_agent_verifier(task: Any) -> bool:
    evaluation = _value(task, "evaluation")
    metadata = model_dump(_value(evaluation, "metadata"))
    return metadata.get("trusted_post_agent_verifier") is True


def _validate_trust_boundaries(task: Any, services: Sequence[ServiceSpec]) -> None:
    """Reject dependency and volume paths that cross the verifier trust boundary."""
    roles = {service.name: service.role for service in services}
    trusted_post_agent = _trusted_post_agent_verifier(task)
    for service in services:
        if str(service.raw.get("network_mode") or "") == "share":
            target = str(service.raw.get("network_share_target") or "")
            target_role = roles.get(target)
            if target_role is not None and (
                (service.role == "verifier") != (target_role == "verifier")
            ):
                raise ValueError(
                    f"service {service.name!r} must not share a network namespace "
                    f"across the verifier trust boundary with service {target!r}"
                )

        dependencies = _compose_dependencies(service.raw.get("depends_on"))
        for dependency in dependencies:
            dependency_role = roles.get(dependency)
            if service.role != "verifier" and dependency_role == "verifier":
                raise ValueError(
                    f"service {service.name!r} must not depend on verifier "
                    f"service {dependency!r}"
                )
            if (
                service.role == "verifier"
                and dependency_role != "verifier"
                and not trusted_post_agent
            ):
                raise ValueError(
                    f"verifier service {service.name!r} must not depend on "
                    f"agent-reachable service {dependency!r} unless "
                    "evaluation.metadata.trusted_post_agent_verifier = true"
                )

    verifier_volumes: set[str] = set()
    agent_reachable_volumes: set[str] = set()
    for service in services:
        destination = (
            verifier_volumes if service.role == "verifier" else agent_reachable_volumes
        )
        for mount in _items(service.raw.get("volumes")):
            name = _value(mount, "volume", "source")
            if name:
                destination.add(str(name))
    shared = sorted(verifier_volumes & agent_reachable_volumes)
    if shared:
        raise ValueError(
            "named volumes cannot be shared between agent-reachable services "
            f"and the verifier: {shared}"
        )


def is_digest_pinned_image(image_ref: str) -> bool:
    """Return whether an external image is pinned to a full sha256 digest."""
    return bool(_PINNED_IMAGE_RE.fullmatch(image_ref.strip()))


def reject_unpinned_external_images(services: Sequence[ServiceSpec]) -> None:
    """Fail before build/pull when an external image can drift."""
    for service in services:
        if (
            service.image
            and service.build is None
            and not is_digest_pinned_image(service.image)
        ):
            raise ValueError(
                f"service {service.name!r} uses unpinned external image "
                f"{service.image!r}; use <repository>@sha256:<64 hex>"
            )


def _size_mb(value: Any, *, numeric_unit_mb: int = 1) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value * numeric_unit_mb)
    text = str(value).strip().lower().replace(" ", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(kib|kb|mib|mb|gib|gb|ti?b)?", text)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2) or "mb"
    multiplier = {
        "kib": 1 / 1024,
        "kb": 1 / 1000,
        "mib": 1,
        "mb": 1,
        "gib": 1024,
        "gb": 1000,
        "tib": 1024 * 1024,
        "tb": 1000 * 1000,
    }[unit]
    return int(amount * multiplier)


def _resource_string_fields(value: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    cpu = re.search(r"(?P<count>\d+)vcpu", value, re.IGNORECASE)
    memory = re.search(r"(?P<count>\d+)gib", value, re.IGNORECASE)
    gpu = re.search(
        r"\+(?P<kind>h100|a100|t4|l4)(?:/(?P<count>\d+))?",
        value,
        re.IGNORECASE,
    )
    tpu = re.search(
        r"\+tpu(?P<kind>v[0-9]+[a-z]*)(?P<topology>[0-9]+x[0-9]+(?:x[0-9]+)*)",
        value,
        re.IGNORECASE,
    )
    if cpu:
        fields["cpus"] = int(cpu.group("count"))
    if memory:
        fields["memory_mb"] = int(memory.group("count")) * 1024
    if gpu:
        # Taiga resource suffixes describe shares (h100/8 is one-eighth of a
        # device), while Harbor accepts only a whole-device count.
        fields["gpus"] = 1
        fields["gpu_types"] = [gpu.group("kind").upper()]
    if tpu:
        fields["tpu"] = {
            "type": tpu.group("kind").lower(),
            "topology": tpu.group("topology").lower(),
        }
    return fields


def project_resource_fields(value: Any) -> dict[str, Any]:
    """Project flexible resource declarations to Harbor's typed fields."""
    if isinstance(value, str):
        return _resource_string_fields(value)
    raw = model_dump(value)
    if not raw:
        return {}

    fields: dict[str, Any] = {}
    cpus = _first_non_none(
        [raw.get("cpus"), raw.get("cpu"), raw.get("vcpu"), raw.get("cpu_count")]
    )
    if cpus is not None:
        cpu_value = float(cpus)
        fields["cpus"] = int(cpu_value) if cpu_value.is_integer() else cpu_value

    memory_mb = _first_non_none(
        [
            raw.get("memory_mb"),
            _size_mb(raw.get("memory")),
            _size_mb(raw.get("memory_gb"), numeric_unit_mb=1024),
        ]
    )
    if memory_mb is not None:
        fields["memory_mb"] = int(memory_mb)

    storage_mb = _first_non_none(
        [
            raw.get("storage_mb"),
            _size_mb(raw.get("storage")),
            _size_mb(raw.get("storage_gb"), numeric_unit_mb=1024),
        ]
    )
    if storage_mb is not None:
        fields["storage_mb"] = int(storage_mb)

    gpus = _first_non_none(
        [raw.get("gpus"), raw.get("gpu_count"), raw.get("accelerator_count")]
    )
    if gpus is not None:
        fields["gpus"] = int(gpus)
    gpu_types = _first_non_none(
        [raw.get("gpu_types"), raw.get("gpu_type"), raw.get("accelerator_types")]
    )
    if gpu_types:
        fields["gpu_types"] = [str(item) for item in _items(gpu_types)]
    if raw.get("tpu") is not None:
        fields["tpu"] = copy.deepcopy(model_dump(raw["tpu"]) or raw["tpu"])
    if raw.get("build_timeout_sec") is not None:
        fields["build_timeout_sec"] = float(raw["build_timeout_sec"])
    network = raw.get("network")
    if network:
        fields["network_mode"] = (
            "public" if str(network) == "internet" else "no-network"
        )
    elif raw.get("allow_internet") is not None:
        fields["network_mode"] = (
            "public" if bool(raw["allow_internet"]) else "no-network"
        )
    elif any(
        key in raw
        for key in (
            "cpus",
            "cpu",
            "memory_mb",
            "storage_mb",
            "gpus",
            "gpu_types",
            "platform",
            "runtime_timeout_sec",
        )
    ):
        fields["network_mode"] = "no-network"

    required = raw.get("required_resources")
    if isinstance(required, str):
        for key, item in _resource_string_fields(required).items():
            fields.setdefault(key, item)
    return fields


def _service_name_map(services: Sequence[ServiceSpec]) -> dict[str, str]:
    return {service.name: service.harbor_name for service in services}


def _mapped_service(value: Any, names: Mapping[str, str]) -> str | None:
    if value is None or value == "":
        return None
    name = str(value)
    return names.get(name, name)


def _artifact_projections(value: Any, names: Mapping[str, str]) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"source": value}]
    raw = model_dump(value)
    sources = _first_non_none([raw.get("sources"), raw.get("source"), raw.get("path")])
    source_values = [str(source) for source in _items(sources) if source]
    if not source_values:
        raise ValueError("capability artifact is missing source/path")
    destination = _first_non_none([raw.get("destination"), raw.get("name")])
    excludes = _first_non_none([raw.get("exclude"), raw.get("excludes")])
    service = _mapped_service(raw.get("service"), names)
    artifacts: list[dict[str, Any]] = []
    for index, source in enumerate(source_values):
        artifact: dict[str, Any] = {"source": source}
        if destination:
            projected_destination = str(destination)
            if len(source_values) > 1:
                basename = PurePosixPath(source).name or f"path-{index + 1}"
                projected_destination = (
                    f"{projected_destination.rstrip('/')}/{basename}"
                )
            artifact["destination"] = projected_destination
        if excludes:
            artifact["exclude"] = [str(item) for item in _items(excludes)]
        if service:
            artifact["service"] = service
        artifacts.append(artifact)
    return artifacts


def _capture_projection(value: Any, names: Mapping[str, str]) -> dict[str, Any]:
    raw = model_dump(value)
    command = _first_non_none(
        [raw.get("command"), raw.get("exec"), raw.get("snapshot_command")]
    )
    if not command:
        raise ValueError("capability capture is missing command")
    command_text = (
        command
        if isinstance(command, str)
        else shlex.join(str(part) for part in _items(command))
    )
    accepted_exit_codes = [
        int(code) for code in _items(raw.get("accepted_exit_codes") or [0])
    ]
    if accepted_exit_codes != [0]:
        accepted = "|".join(str(code) for code in accepted_exit_codes)
        command_text = (
            f"set +e; {command_text}; _lbx_capture_status=$?; "
            f'case "$_lbx_capture_status" in {accepted}) ;; '
            '*) exit "$_lbx_capture_status" ;; esac'
        )
    destination = _first_non_none(
        [
            raw.get("atomic_destination"),
            raw.get("destination"),
            raw.get("atomic_path"),
        ]
    )
    temporary = f"{destination}.tmp" if destination else None
    if temporary and temporary in command_text and " mv " not in f" {command_text} ":
        command_text += (
            f" && test -e {shlex.quote(temporary)}"
            f" && mv {shlex.quote(temporary)} {shlex.quote(str(destination))}"
        )
    capture: dict[str, Any] = {
        "service": _mapped_service(raw.get("service"), names) or "main",
        "command": command_text,
    }
    timeout = _first_non_none([raw.get("timeout_sec"), raw.get("timeout")])
    if timeout is not None:
        capture["timeout_sec"] = float(timeout)
    if raw.get("user") is not None:
        capture["user"] = raw["user"]
    return capture


def _replace_url_service(url: str, names: Mapping[str, str]) -> str:
    for source, destination in names.items():
        url = re.sub(
            rf"(?<=://){re.escape(source)}(?=[:/])",
            destination,
            url,
        )
    return url


def _mcp_projection(value: Any, names: Mapping[str, str]) -> dict[str, Any]:
    raw = model_dump(value)
    name = _first_non_none([raw.get("name"), raw.get("id"), raw.get("service")])
    if not name:
        raise ValueError("capability MCP declaration is missing a name")
    transport = str(raw.get("transport") or "sse").lower().replace("_", "-")
    if transport == "http":
        transport = "streamable-http"
    projected: dict[str, Any] = {"name": str(name), "transport": transport}
    url = raw.get("url")
    if not url and raw.get("service") and raw.get("port"):
        path = str(raw.get("path") or "/mcp")
        if not path.startswith("/"):
            path = f"/{path}"
        host = _mapped_service(raw["service"], names)
        url = f"http://{host}:{int(raw['port'])}{path}"
    if url:
        projected["url"] = _replace_url_service(str(url), names)
    command = raw.get("command")
    command_args: list[str] = []
    if command:
        if isinstance(command, str):
            projected["command"] = command
        else:
            argv = [str(item) for item in _items(command)]
            if argv:
                projected["command"] = argv[0]
                command_args.extend(argv[1:])
    command_args.extend(str(item) for item in _items(raw.get("args")))
    if command_args:
        projected["args"] = command_args
    projected["_access"] = str(raw.get("access") or "agent")
    return projected


def _field_container(task: Any, section: Any, *names: str) -> Any:
    values = [_value(section, *names), _value(task, *names)]
    environment = _value(task, "environment")
    values.append(_value(environment, *names))
    return _first_non_none(values)


def _resource_pair(
    task: Any, section: Any, services: Sequence[ServiceSpec]
) -> tuple[dict[str, Any], dict[str, Any]]:
    resource_container = _value(section, "resources")
    resource_mapping = model_dump(resource_container)
    agent = project_resource_fields(
        _first_non_none(
            [
                _value(section, "agent_resources", "main_resources"),
                resource_mapping.get("agent"),
                resource_mapping.get("main"),
            ]
        )
    )
    verifier = project_resource_fields(
        _first_non_none(
            [
                _value(section, "verifier_resources", "grader_resources"),
                resource_mapping.get("verifier"),
                resource_mapping.get("grader"),
            ]
        )
    )
    if (
        not agent
        and resource_mapping
        and not any(
            key in resource_mapping for key in ("agent", "main", "verifier", "grader")
        )
    ):
        agent = project_resource_fields(resource_mapping)

    agent_service = next(
        (service for service in services if service.role == "agent"), None
    )
    verifier_service = next(
        (service for service in services if service.role == "verifier"), None
    )
    if not agent:
        agent = project_resource_fields(_value(_value(task, "agent"), "resources"))
    if not verifier:
        verifier = project_resource_fields(
            _value(_value(task, "verifier"), "resources")
        )
    if not agent and agent_service:
        agent = dict(agent_service.resources)
    if not verifier and verifier_service:
        verifier = dict(verifier_service.resources)

    environment = _value(task, "environment")
    if not agent:
        agent = project_resource_fields(environment)
    verifier_section = _value(task, "verifier")
    verifier_environment = _value(verifier_section, "environment")
    if not verifier:
        verifier = project_resource_fields(verifier_environment)
    return agent, verifier


def resolve_capabilities(task: Any) -> CapabilityConfig:
    """Resolve services plus typed capture/artifact/MCP/resource projections."""
    services = tuple(resolve_services(task))
    reject_unpinned_external_images(services)
    section = _capability_section(task)
    names = _service_name_map(services) or {"main": "main"}

    artifacts = tuple(
        artifact
        for item in _items(
            _field_container(task, section, "artifacts", "artifact_exports")
        )
        for artifact in _artifact_projections(item, names)
    )
    captures = tuple(
        _capture_projection(item, names)
        for item in _items(
            _field_container(
                task,
                section,
                "captures",
                "capture",
                "collect",
                "verifier_captures",
            )
        )
    )
    tools = _value(task, "tools")
    mcp_values = _items(
        _first_non_none(
            [
                _value(tools, "mcp_servers"),
                _field_container(task, section, "mcp_servers", "mcp"),
            ]
        )
    )
    for service in services:
        service_mcp = _first_non_none(
            [service.raw.get("mcp_server"), service.raw.get("mcp")]
        )
        for item in _items(service_mcp):
            item_mapping = model_dump(item)
            if item_mapping and "service" not in item_mapping:
                item_mapping["service"] = service.name
            mcp_values.append(item_mapping or item)
    projected_mcp = tuple(_mcp_projection(item, names) for item in mcp_values)
    mcp_servers = tuple(
        {key: value for key, value in server.items() if key != "_access"}
        for server in projected_mcp
        if server["_access"] in {"agent", "both"}
    )
    verifier_mcp_servers = tuple(
        {key: value for key, value in server.items() if key != "_access"}
        for server in projected_mcp
        if server["_access"] in {"verifier", "both"}
    )
    volumes = tuple(
        {
            "name": str(_value(volume, "name", "id")),
            "scope": str(_value(volume, "scope", default="shared")),
        }
        for volume in _items(_value(task, "volumes"))
        if _value(volume, "name", "id")
    )
    agent_resources, verifier_resources = _resource_pair(task, section, services)
    return CapabilityConfig(
        services=services,
        artifacts=artifacts,
        captures=captures,
        mcp_servers=mcp_servers,
        verifier_mcp_servers=verifier_mcp_servers,
        volumes=volumes,
        agent_resources=agent_resources,
        verifier_resources=verifier_resources,
    )


def _environment_variables(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items()}
    variables: dict[str, str] = {}
    for item in _items(value):
        name = str(item)
        if name:
            variables[name] = f"${{{name}}}"
    return variables


def project_harbor_task_data(
    task_data: Mapping[str, Any], capabilities: CapabilityConfig
) -> dict[str, Any]:
    """Project capability fields into native Harbor ``task.toml`` fields."""
    projected = copy.deepcopy(dict(task_data))
    projected["schema_version"] = "1.4"
    for name in (
        *_CAPABILITY_SECTION_NAMES,
        "services",
        "captures",
        "mcp_servers",
        "volumes",
        "tools",
    ):
        projected.pop(name, None)

    environment = projected.setdefault("environment", {})
    if not isinstance(environment, dict):
        environment = {}
        projected["environment"] = environment
    authored_storage_mb = environment.get("storage_mb")
    for name in (*_CAPABILITY_SECTION_NAMES, "services"):
        environment.pop(name, None)
    for key in _RESOURCE_FIELDS:
        environment.pop(key, None)
    environment.update(capabilities.agent_resources)
    if isinstance(authored_storage_mb, (int, float)) and not isinstance(
        authored_storage_mb, bool
    ):
        projected_storage_mb = environment.get("storage_mb")
        environment["storage_mb"] = max(
            int(authored_storage_mb),
            int(projected_storage_mb or 0),
        )
    if capabilities.mcp_servers:
        environment["mcp_servers"] = [
            copy.deepcopy(server) for server in capabilities.mcp_servers
        ]
    else:
        environment.setdefault("mcp_servers", [])

    verifier = projected.setdefault("verifier", {})
    if not isinstance(verifier, dict):
        verifier = {}
        projected["verifier"] = verifier
    verifier["env"] = _environment_variables(verifier.get("env"))
    verifier.pop("resources", None)
    verifier["environment_mode"] = "separate"
    verifier_environment = verifier.setdefault("environment", {})
    if not isinstance(verifier_environment, dict):
        verifier_environment = {}
        verifier["environment"] = verifier_environment
    for key in _RESOURCE_FIELDS:
        verifier_environment.pop(key, None)
    verifier_environment.update(capabilities.verifier_resources)
    verifier_environment["mcp_servers"] = [
        copy.deepcopy(server) for server in capabilities.verifier_mcp_servers
    ]
    if capabilities.captures:
        verifier["collect"] = [
            copy.deepcopy(capture) for capture in capabilities.captures
        ]
    else:
        verifier.pop("collect", None)

    projected["artifacts"] = [
        copy.deepcopy(artifact) for artifact in capabilities.artifacts
    ]
    agent = projected.setdefault("agent", {})
    if isinstance(agent, dict):
        agent.pop("resources", None)
    return projected


def remap_depends_on(
    value: Any, services: Sequence[ServiceSpec], *, omit_main: bool = False
) -> Any:
    """Remap authored service names to Harbor compose names."""
    names = _service_name_map(services)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for name, condition in value.items():
            mapped = names.get(str(name), str(name))
            if omit_main and mapped == "main":
                continue
            result[mapped] = copy.deepcopy(condition)
        return result
    result = []
    for name in _items(value):
        mapped = names.get(str(name), str(name))
        if not (omit_main and mapped == "main"):
            result.append(mapped)
    return result


__all__ = [
    "CapabilityConfig",
    "ServiceBuild",
    "ServiceSpec",
    "implicit_agent_service",
    "is_capability_task",
    "is_digest_pinned_image",
    "model_dump",
    "project_harbor_task_data",
    "project_resource_fields",
    "reject_unpinned_external_images",
    "remap_depends_on",
    "resolve_capabilities",
    "resolve_services",
]
