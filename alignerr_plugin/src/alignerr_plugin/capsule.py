"""Trusted task-capsule image packaging and outer Taiga context generation."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w
import yaml

from alignerr_plugin.capabilities import (
    CapabilityConfig,
    ServiceSpec,
    implicit_agent_service,
    is_capability_task,
    reject_unpinned_external_images,
    resolve_capabilities,
)
from alignerr_plugin.exporters.taiga import STARTUP_COMMAND
from alignerr_plugin.materialization import (
    copy_directory_contents_safe,
    materialize_task_inputs,
    materialize_workspace_seed,
    require_local_dockerfile_target,
    resolve_workspace,
    staged_output_directory,
    workspace_dockerfile_overlay,
)
from alignerr_plugin.utils import load_task_toml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GRADER_DIR = _REPO_ROOT / "grader"
_RUBRIC_DIR = _REPO_ROOT / "taiga_runtime" / "rubric"
_BASE_DIR = _REPO_ROOT / "base"

IMAGE_ARCHIVE_NAME = "images.docker.tar"
IMAGE_MANIFEST_NAME = "images.manifest.json"
IMAGE_MANIFEST_SCHEMA = "alignerr.task-capsule.images.v1"
CAPSULE_MANIFEST_NAME = "manifest.json"
CAPSULE_MANIFEST_SCHEMA = "alignerr.task-capsule.v1"
_CAPSULE_WORKSPACE_SEED = ".alignerr-workspace-seed"
_CAPSULE_BASE_IMAGE = (
    "python:3.13-slim@sha256:"
    "d49c1ff87eb98eac346fc250f52925f726eb913c43a92854246dd03c9692ad67"
)
_UV_IMAGE = (
    "ghcr.io/astral-sh/uv:0.8.17@sha256:"
    "e4644cb5bd56fdc2c5ea3ee0525d9d21eed1603bccd6a21f887a938be7e85be1"
)

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class MaterializedServiceImage:
    """A child image resolved to immutable identity and a bundle-local tag."""

    name: str
    role: str
    digest: str
    archive_ref: str
    source: str
    source_ref: str | None = None
    platform: str = ""
    os: str = ""
    architecture: str = ""


@dataclass(frozen=True)
class ImageBundle:
    """Paths and identities for a deterministic child-image bundle."""

    archive_path: Path
    manifest_path: Path
    images: tuple[MaterializedServiceImage, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class CapsuleExport:
    """Generated outer Taiga capsule context."""

    context_dir: Path
    dockerfile: Path
    compose_file: Path
    capsule_manifest: Path
    image_bundle: ImageBundle


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-._")
    return slug or "service"


def _run(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return runner(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
    )


def _resolve_inside(root: Path, value: str, *, label: str) -> Path:
    root = root.resolve()
    candidate = Path(value)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside {root}: {value!r}") from exc
    return resolved


def _build_tag(service: ServiceSpec) -> str:
    assert service.build is not None
    identity = {
        "name": service.name,
        "context": service.build.context,
        "dockerfile": service.build.dockerfile,
        "args": service.build.args,
        "target": service.build.target,
        "platform": service.build.platform,
        "service_platform": _service_platform(service),
        "pull": service.build.pull,
        "no_cache": service.build.no_cache,
    }
    suffix = hashlib.sha256(_stable_json(identity).encode()).hexdigest()[:16]
    return f"alignerr-capsule-build/{_slug(service.name)}:{suffix}"


def _service_platform(service: ServiceSpec) -> str | None:
    if service.build is not None and service.build.platform:
        return service.build.platform
    if service.resources.get("platform"):
        return str(service.resources["platform"])
    resources = service.raw.get("resources")
    if isinstance(resources, Mapping) and resources.get("platform"):
        return str(resources["platform"])
    if service.compose.get("platform"):
        return str(service.compose["platform"])
    return None


def _target_platform(service: ServiceSpec) -> tuple[str, str, str]:
    platform = _service_platform(service)
    if not platform:
        raise ValueError(
            f"service {service.name!r} requires an explicit target platform "
            "(for example 'linux/amd64')"
        )
    parts = platform.lower().split("/")
    if len(parts) not in {2, 3} or not all(parts[:2]):
        raise ValueError(
            f"service {service.name!r} has invalid target platform {platform!r}; "
            "expected os/architecture[/variant]"
        )
    return platform, parts[0], parts[1]


def _build_service_image(
    problem_dir: Path,
    service: ServiceSpec,
    *,
    docker_command: str,
    runner: CommandRunner,
) -> str:
    build = service.build
    assert build is not None
    context = _resolve_inside(
        problem_dir, build.context, label=f"service {service.name!r} build context"
    )
    if not context.is_dir():
        raise ValueError(
            f"service {service.name!r} build context does not exist: {context}"
        )
    dockerfile = _resolve_inside(
        context,
        build.dockerfile,
        label=f"service {service.name!r} Dockerfile",
    )
    if not dockerfile.is_file():
        raise ValueError(
            f"service {service.name!r} Dockerfile does not exist: {dockerfile}"
        )

    tag = _build_tag(service)
    build_context = context
    build_dockerfile = dockerfile
    temporary_context: tempfile.TemporaryDirectory[str] | None = None
    if service.role in {"agent", "verifier"}:
        workspace = resolve_workspace(load_task_toml(problem_dir))
        temporary_context = tempfile.TemporaryDirectory(
            prefix=f"alignerr-{_slug(service.name)}-"
        )
        build_context = Path(temporary_context.name)
        copy_directory_contents_safe(context, build_context, task_root=problem_dir)
        materialize_workspace_seed(
            problem_dir,
            build_context / ".alignerr-workspace-seed",
            workspace,
        )
        original = (build_context / dockerfile.relative_to(context)).read_text()
        if build.target:
            target = require_local_dockerfile_target(
                original,
                build.target,
                label=f"service {service.name!r} build",
            )
            original = original.rstrip() + f"\n\nFROM {target}\n"
        overlay = workspace_dockerfile_overlay(
            workspace,
            user="agent" if service.role == "agent" else "root",
            verifier=service.role == "verifier",
        )
        build_dockerfile = build_context / "Dockerfile.alignerr"
        build_dockerfile.write_text(original.rstrip() + "\n" + overlay)
    command = [
        docker_command,
        "build",
        "--tag",
        tag,
        "--file",
        str(build_dockerfile),
    ]
    if build.target and temporary_context is None:
        command.extend(["--target", build.target])
    platform, _, _ = _target_platform(service)
    command.extend(["--platform", platform])
    if build.pull:
        command.append("--pull")
    if build.no_cache:
        command.append("--no-cache")
    for key, value in build.args:
        command.extend(["--build-arg", f"{key}={value}"])
    command.append(str(build_context))
    try:
        _run(runner, command, cwd=problem_dir)
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()
    return tag


def _inspect_image(
    image_ref: str, *, docker_command: str, runner: CommandRunner
) -> dict[str, Any]:
    completed = _run(
        runner,
        [docker_command, "image", "inspect", image_ref],
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"docker image inspect returned invalid JSON for {image_ref!r}"
        ) from exc
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"docker image inspect returned no record for {image_ref!r}")


def _image_digest(service: ServiceSpec, inspected: Mapping[str, Any]) -> str:
    image_id = str(inspected.get("Id") or "").lower()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        return image_id
    repo_digests = inspected.get("RepoDigests")
    if isinstance(repo_digests, list):
        for value in sorted(str(item) for item in repo_digests):
            if "@sha256:" in value:
                digest = value.rsplit("@", 1)[1].lower()
                if re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                    return digest
    if service.build is None and service.image:
        return service.image.rsplit("@", 1)[1].lower()
    raise RuntimeError(
        f"service {service.name!r} image has no full sha256 identity after build"
    )


def materialize_service_images(
    problem_dir: Path,
    services: Sequence[ServiceSpec],
    *,
    trusted_build: bool,
    runner: CommandRunner = subprocess.run,
    docker_command: str = "docker",
) -> tuple[MaterializedServiceImage, ...]:
    """Build/pull child images and resolve their immutable identities.

    No Docker command is allowed unless the caller explicitly identifies the
    execution as a trusted build phase.
    """
    reject_unpinned_external_images(services)
    if not trusted_build:
        raise PermissionError(
            "child images may only be built or pulled in a trusted build phase; "
            "pass trusted_build=True from trusted CI"
        )
    if not services:
        raise ValueError("task capsule requires at least one service image")

    materialized: list[MaterializedServiceImage] = []
    for service in sorted(services, key=lambda item: (item.role, item.name)):
        platform, target_os, target_architecture = _target_platform(service)
        if service.build is not None:
            local_ref = _build_service_image(
                problem_dir,
                service,
                docker_command=docker_command,
                runner=runner,
            )
            source = "build"
            source_ref = None
        elif service.image:
            local_ref = service.image
            pull_command = [docker_command, "pull"]
            pull_command.extend(["--platform", platform])
            pull_command.append(local_ref)
            _run(runner, pull_command)
            source = "external"
            source_ref = service.image
        else:
            raise ValueError(
                f"service {service.name!r} has neither a build nor a pinned image"
            )

        inspected = _inspect_image(
            local_ref, docker_command=docker_command, runner=runner
        )
        inspected_os = str(inspected.get("Os") or target_os).lower()
        inspected_architecture = str(
            inspected.get("Architecture") or target_architecture
        ).lower()
        if (inspected_os, inspected_architecture) != (
            target_os,
            target_architecture,
        ):
            raise RuntimeError(
                f"service {service.name!r} resolved to "
                f"{inspected_os}/{inspected_architecture}, expected "
                f"{target_os}/{target_architecture}"
            )
        digest = _image_digest(service, inspected)
        archive_ref = (
            f"alignerr-capsule/{_slug(service.name)}:{digest.replace(':', '-')}"
        )
        _run(
            runner,
            [docker_command, "image", "tag", local_ref, archive_ref],
        )
        materialized.append(
            MaterializedServiceImage(
                name=service.name,
                role=service.role,
                digest=digest,
                archive_ref=archive_ref,
                source=source,
                source_ref=source_ref,
                platform=platform,
                os=inspected_os,
                architecture=inspected_architecture,
            )
        )
    return tuple(materialized)


def canonicalize_docker_archive(source: Path, destination: Path) -> None:
    """Rewrite a Docker archive with stable order and metadata.

    This promises deterministic archive bytes only. Docker builds and upstream
    image resolution remain inputs controlled by the trusted build phase.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, mode="r:*") as input_tar:
        members = sorted(input_tar.getmembers(), key=lambda member: member.name)
        with tarfile.open(
            destination, mode="w", format=tarfile.PAX_FORMAT
        ) as output_tar:
            for member in members:
                normalized = copy.copy(member)
                normalized.uid = 0
                normalized.gid = 0
                normalized.uname = ""
                normalized.gname = ""
                normalized.mtime = 0
                normalized.pax_headers = {}
                if normalized.isdir():
                    normalized.mode = 0o755
                elif normalized.isfile():
                    normalized.mode = 0o644
                fileobj = input_tar.extractfile(member) if member.isfile() else None
                output_tar.addfile(normalized, fileobj)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_manifest(
    archive_path: Path, images: Sequence[MaterializedServiceImage]
) -> dict[str, Any]:
    archive_sha256 = _sha256(archive_path)
    image_rows = [
        {
            "name": image.name,
            "service": image.name,
            "role": "main" if image.role == "agent" else image.role,
            "archive": archive_path.name,
            "archive_sha256": archive_sha256,
            "archive_ref": image.archive_ref,
            "image_ref": image.archive_ref,
            "digest": image.digest,
            "image_digest": image.digest,
            "platform": image.platform,
            "os": image.os,
            "architecture": image.architecture,
            "source": image.source,
            **({"source_ref": image.source_ref} if image.source_ref else {}),
            **(
                {"source_digest": image.source_ref.rsplit("@", 1)[1]}
                if image.source_ref and "@" in image.source_ref
                else {}
            ),
        }
        for image in sorted(images, key=lambda item: (item.role, item.name))
    ]
    return {
        "schema": IMAGE_MANIFEST_SCHEMA,
        "determinism": {
            "scope": "archive-bytes",
            "excludes": ["child-image-builds", "upstream-image-resolution"],
        },
        "archive": {
            "format": "docker-archive",
            "path": archive_path.name,
            "sha256": f"sha256:{archive_sha256}",
        },
        "images": image_rows,
        "services": image_rows,
    }


def export_image_bundle(
    problem_dir: Path,
    services: Sequence[ServiceSpec],
    output_dir: Path,
    *,
    trusted_build: bool,
    runner: CommandRunner = subprocess.run,
    docker_command: str = "docker",
) -> ImageBundle:
    """Materialize services and write an archive-deterministic bundle."""
    validated_services = load_capabilities(problem_dir).services
    if {service.name for service in services} != {
        service.name for service in validated_services
    }:
        raise ValueError(
            "image bundle service names must match the validated task.toml "
            "capability projection"
        )
    services = validated_services
    images = materialize_service_images(
        problem_dir,
        services,
        trusted_build=trusted_build,
        runner=runner,
        docker_command=docker_command,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / IMAGE_ARCHIVE_NAME
    manifest_path = output_dir / IMAGE_MANIFEST_NAME

    with tempfile.NamedTemporaryFile(
        prefix="capsule-images-", suffix=".tar", dir=output_dir, delete=False
    ) as handle:
        raw_archive = Path(handle.name)
    try:
        _run(
            runner,
            [
                docker_command,
                "image",
                "save",
                "--output",
                str(raw_archive),
                *sorted(image.archive_ref for image in images),
            ],
        )
        canonicalize_docker_archive(raw_archive, archive_path)
    finally:
        raw_archive.unlink(missing_ok=True)

    manifest = _image_manifest(archive_path, images)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return ImageBundle(
        archive_path=archive_path,
        manifest_path=manifest_path,
        images=images,
        manifest=manifest,
    )


def _service_image_map(
    images: Sequence[MaterializedServiceImage],
) -> dict[str, MaterializedServiceImage]:
    return {image.name: image for image in images}


def capsule_compose_data(
    capabilities: CapabilityConfig,
    images: Sequence[MaterializedServiceImage],
) -> dict[str, Any]:
    """Render the digest-locked nested service graph."""
    by_name = _service_image_map(images)
    services: dict[str, Any] = {}
    for service in sorted(capabilities.services, key=lambda item: item.name):
        image = by_name.get(service.name)
        if image is None:
            raise ValueError(
                f"image bundle is missing capability service {service.name!r}"
            )
        definition = copy.deepcopy(service.compose)
        network_mode = definition.get("network_mode")
        if isinstance(network_mode, str) and network_mode.startswith("service:"):
            target = network_mode.split(":", 1)[1]
            mapped = next(
                (item.name for item in capabilities.services if item.name == target),
                target,
            )
            definition["network_mode"] = f"service:{mapped}"
        definition["image"] = image.archive_ref
        definition["pull_policy"] = "never"
        definition["cap_drop"] = ["ALL"]
        security_options = {
            str(option).strip()
            for option in definition.get("security_opt", [])
            if str(option).strip()
        }
        security_options.add("no-new-privileges:true")
        definition["security_opt"] = sorted(security_options)
        if service.role == "verifier":
            definition.pop("networks", None)
            definition["network_mode"] = "none"
        services[service.name] = definition
    compose: dict[str, Any] = {"services": services}
    volumes = {str(volume["name"]): {} for volume in capabilities.volumes}
    if volumes:
        compose["volumes"] = volumes
    if any(
        "alignerr-isolated" in service.get("networks", {})
        for service in services.values()
    ):
        compose["networks"] = {"alignerr-isolated": {"internal": True}}
    return compose


def render_capsule_compose(
    capabilities: CapabilityConfig,
    images: Sequence[MaterializedServiceImage],
) -> str:
    """Return deterministic YAML for the outer capsule's nested services."""
    return yaml.safe_dump(
        capsule_compose_data(capabilities, images),
        sort_keys=True,
        default_flow_style=False,
    )


_INSTALL_NESTED_DOCKER = """\
#!/bin/sh
set -eu

if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends docker.io ca-certificates curl
    rm -rf /var/lib/apt/lists/*
fi

command -v docker >/dev/null 2>&1
command -v curl >/dev/null 2>&1

case "$(uname -m)" in
    x86_64|amd64)
        compose_arch=x86_64
        compose_sha=383ce6698cd5d5bbf958d2c8489ed75094e34a77d340404d9f32c4ae9e12baf0
        ;;
    aarch64|arm64)
        compose_arch=aarch64
        compose_sha=6e9fbd5daa20dca5d7d89145081ae8155d68ef2928b497d9f85b54fe0f9dbb2c
        ;;
    *)
        echo "unsupported architecture for pinned Compose v2: $(uname -m)" >&2
        exit 1
        ;;
esac
plugin=/usr/local/lib/docker/cli-plugins/docker-compose
mkdir -p "$(dirname "$plugin")"
curl -fsSL \
    "https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-${compose_arch}" \
    -o "$plugin"
echo "${compose_sha}  ${plugin}" | sha256sum -c -
chmod 0755 "$plugin"
test "$(docker compose version --short)" = "2.29.7"
"""

_CAPSULE_ENTRYPOINT = """\
#!/bin/sh
set -eu

command -v docker >/dev/null 2>&1
command -v dockerd >/dev/null 2>&1
docker compose version >/dev/null 2>&1

# The rubric's TaskServiceRuntime owns dockerd, verifies/loads the embedded
# archives, starts Compose, and tears it down. Keep one orchestration authority.
exec "$@"
"""


def _capsule_dockerfile() -> str:
    command = json.dumps(STARTUP_COMMAND.split())
    return f"""\
FROM {_CAPSULE_BASE_IMAGE}

USER root
COPY --from={_UV_IMAGE} /uv /usr/local/bin/uv
COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-cpu.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh base/install-task-deps.sh /tmp/base/
RUN if [ ! -x /opt/lbx-runtime/.venv/bin/rubric ]; then \\
        command -v apt-get >/dev/null 2>&1 \\
        || (echo "capsule runtime base needs the lbx runtime or apt-get" >&2; exit 1); \\
        SKIP_COMMON_REQUIREMENTS=1 \\
        /tmp/base/install-common.sh; \\
    fi

COPY task/environment/ /tmp/task-deps/environment/
COPY --chown=root:root task/scorer/ /tmp/task-deps/scorer/
RUN /tmp/base/install-task-deps.sh /tmp/task-deps && rm -rf /tmp/task-deps
COPY --chown=root:root task/scorer/data/ /mcp_server/data/
COPY --chown=root:root task/scorer/ /mcp_server/grader/
COPY task/data/ /workspace/data/
COPY --chown=root:root task/ /task/source/
COPY task/task.toml task/instruction.md /task/
COPY --chown=root:root task/{_CAPSULE_WORKSPACE_SEED}/ /task/{_CAPSULE_WORKSPACE_SEED}/
RUN rm -rf /mcp_server/grader/data \\
    && rm -rf /data \\
    && ln -s /workspace/data /data \\
    && chown -R root:root /mcp_server \\
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {{}} + \\
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {{}} + \\
    && if [ -d /task/source/solution ]; then chmod -R go-rwx /task/source/solution; fi \\
    && chmod 0700 /mcp_server

COPY images/{IMAGE_ARCHIVE_NAME} /task/capsule/images/{IMAGE_ARCHIVE_NAME}
COPY images/{IMAGE_MANIFEST_NAME} /task/capsule/images/{IMAGE_MANIFEST_NAME}
COPY docker-compose.yaml {CAPSULE_MANIFEST_NAME} /task/capsule/
COPY --chmod=0755 install-nested-docker.sh capsule-entrypoint.sh /usr/local/bin/
RUN /usr/local/bin/install-nested-docker.sh

WORKDIR /workdir
ENTRYPOINT ["/usr/local/bin/capsule-entrypoint.sh"]
CMD {command}
"""


def _copytree_clean(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".DS_Store",
        ),
    )


def _copy_task_capsule_files(
    problem_dir: Path,
    destination: Path,
) -> str | None:
    materialize_task_inputs(problem_dir, destination)
    workspace = resolve_workspace(load_task_toml(problem_dir))
    runtime_seed: str | None = None
    if workspace.seed is not None:
        materialize_workspace_seed(
            problem_dir,
            destination / _CAPSULE_WORKSPACE_SEED,
            workspace,
        )
        task_toml_path = destination / "task.toml"
        with task_toml_path.open("rb") as handle:
            materialized_task = tomllib.load(handle)
        workspace_table = materialized_task.get("workspace")
        if not isinstance(workspace_table, dict):
            raise TypeError("capsule workspace must be a TOML table")
        workspace_table["seed"] = _CAPSULE_WORKSPACE_SEED
        task_toml_path.write_text(tomli_w.dumps(materialized_task))
        runtime_seed = _CAPSULE_WORKSPACE_SEED
    else:
        # Keep the generic outer Dockerfile valid for intentionally empty
        # workspaces, which do not consume a seed at runtime.
        (destination / _CAPSULE_WORKSPACE_SEED).mkdir(exist_ok=True)
    for name in ("data", "scorer", "environment"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    (destination / "scorer" / "data").mkdir(parents=True, exist_ok=True)
    return runtime_seed


def _capsule_manifest(
    capabilities: CapabilityConfig,
    bundle: ImageBundle,
    *,
    workspace_seed: str | None,
) -> dict[str, Any]:
    archive_sha256 = _sha256(bundle.archive_path)
    return {
        "schema": CAPSULE_MANIFEST_SCHEMA,
        "determinism": {
            "scope": "embedded-archive-bytes",
            "excludes": ["child-image-builds", "outer-image-build"],
        },
        "compose_file": "docker-compose.yaml",
        "image_manifest": f"images/{bundle.manifest_path.name}",
        "image_archive": f"images/{bundle.archive_path.name}",
        "images": [
            {
                "service": image.name,
                "role": "main" if image.role == "agent" else image.role,
                "archive": f"images/{bundle.archive_path.name}",
                "archive_sha256": archive_sha256,
                "image_ref": image.archive_ref,
                "image_digest": image.digest,
                "platform": image.platform,
                "os": image.os,
                "architecture": image.architecture,
            }
            for image in sorted(bundle.images, key=lambda item: (item.role, item.name))
        ],
        "startup_command": STARTUP_COMMAND,
        "workspace": {"seed": workspace_seed} if workspace_seed else None,
        "agent_service": (
            capabilities.agent_service.name if capabilities.agent_service else None
        ),
        "nested_services": [
            service.name
            for service in sorted(capabilities.services, key=lambda item: item.name)
            if service.role not in {"agent", "verifier"}
        ],
        "verifier_service": (
            capabilities.verifier_service.name
            if capabilities.verifier_service
            else None
        ),
    }


def write_capsule_context(
    problem_dir: Path,
    output_dir: Path,
    capabilities: CapabilityConfig,
    bundle: ImageBundle,
) -> CapsuleExport:
    """Generate the outer Dockerfile/context around a prepared image bundle."""
    validated_capabilities = load_capabilities(problem_dir)
    if capabilities != validated_capabilities:
        raise ValueError(
            "capsule context capabilities must come from validated task.toml"
        )
    agent_service = capabilities.agent_service
    if agent_service is None:
        raise ValueError("task capsule requires exactly one service with role='agent'")
    by_name = _service_image_map(bundle.images)
    agent_image = by_name.get(agent_service.name)
    if agent_image is None:
        raise ValueError("task capsule image bundle does not contain the agent service")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for source in (bundle.archive_path, bundle.manifest_path):
        destination = images_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    workspace_seed = _copy_task_capsule_files(problem_dir, output_dir / "task")
    _copytree_clean(_GRADER_DIR, output_dir / "grader")
    _copytree_clean(_RUBRIC_DIR, output_dir / "taiga_runtime" / "rubric")
    _copytree_clean(_BASE_DIR, output_dir / "base")

    compose_file = output_dir / "docker-compose.yaml"
    compose_file.write_text(render_capsule_compose(capabilities, bundle.images))
    (output_dir / "install-nested-docker.sh").write_text(_INSTALL_NESTED_DOCKER)
    (output_dir / "install-nested-docker.sh").chmod(0o755)
    (output_dir / "capsule-entrypoint.sh").write_text(_CAPSULE_ENTRYPOINT)
    (output_dir / "capsule-entrypoint.sh").chmod(0o755)

    capsule_manifest = output_dir / CAPSULE_MANIFEST_NAME
    capsule_manifest.write_text(
        json.dumps(
            _capsule_manifest(
                capabilities,
                bundle,
                workspace_seed=workspace_seed,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    dockerfile = output_dir / "Dockerfile"
    dockerfile.write_text(_capsule_dockerfile())
    return CapsuleExport(
        context_dir=output_dir,
        dockerfile=dockerfile,
        compose_file=compose_file,
        capsule_manifest=capsule_manifest,
        image_bundle=ImageBundle(
            archive_path=images_dir / bundle.archive_path.name,
            manifest_path=images_dir / bundle.manifest_path.name,
            images=bundle.images,
            manifest=bundle.manifest,
        ),
    )


def load_capabilities(problem_dir: Path) -> CapabilityConfig:
    """Validate TaskToml before resolving any capability projection."""
    task_path = problem_dir / "task.toml"
    task = load_task_toml(problem_dir)
    if not is_capability_task(task):
        raise ValueError(f"{task_path} does not declare capability sections")
    capabilities = resolve_capabilities(task)
    if capabilities.agent_service is None:
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
                "environment/Dockerfile nor environment/main/Dockerfile is "
                "available as a default"
            )
        platform = (
            task.agent.resources.platform if task.agent.resources is not None else None
        )
        default_agent = implicit_agent_service(
            context=default_context,
            resources=capabilities.agent_resources,
            network=(
                task.agent.resources.network
                if task.agent.resources is not None
                else None
            ),
            platform=platform,
        )
        capabilities = CapabilityConfig(
            services=(default_agent, *capabilities.services),
            artifacts=capabilities.artifacts,
            captures=capabilities.captures,
            mcp_servers=capabilities.mcp_servers,
            verifier_mcp_servers=capabilities.verifier_mcp_servers,
            volumes=capabilities.volumes,
            agent_resources=capabilities.agent_resources,
            verifier_resources=capabilities.verifier_resources,
        )
    return capabilities


def export_task_capsule(
    problem_dir: Path,
    output_dir: Path,
    *,
    trusted_build: bool = False,
    runner: CommandRunner = subprocess.run,
    docker_command: str = "docker",
    force: bool = False,
) -> CapsuleExport:
    """Build/pull children, bundle them, and generate an outer Taiga context."""
    problem_dir = problem_dir.resolve()
    capabilities = load_capabilities(problem_dir)
    reject_unpinned_external_images(capabilities.services)
    if not trusted_build:
        raise PermissionError(
            "child images may only be built or pulled in a trusted build phase; "
            "pass trusted_build=True from trusted CI"
        )
    with staged_output_directory(
        output_dir,
        force=force,
        source_tree=problem_dir,
    ) as stage:
        staging_bundle_dir = stage / ".bundle"
        bundle = export_image_bundle(
            problem_dir,
            capabilities.services,
            staging_bundle_dir,
            trusted_build=trusted_build,
            runner=runner,
            docker_command=docker_command,
        )
        write_capsule_context(
            problem_dir,
            stage,
            capabilities,
            bundle,
        )
        shutil.rmtree(staging_bundle_dir, ignore_errors=True)
    return CapsuleExport(
        context_dir=output_dir,
        dockerfile=output_dir / "Dockerfile",
        compose_file=output_dir / "docker-compose.yaml",
        capsule_manifest=output_dir / CAPSULE_MANIFEST_NAME,
        image_bundle=ImageBundle(
            archive_path=output_dir / "images" / IMAGE_ARCHIVE_NAME,
            manifest_path=output_dir / "images" / IMAGE_MANIFEST_NAME,
            images=bundle.images,
            manifest=bundle.manifest,
        ),
    )


__all__ = [
    "CAPSULE_MANIFEST_NAME",
    "IMAGE_ARCHIVE_NAME",
    "IMAGE_MANIFEST_NAME",
    "CapsuleExport",
    "ImageBundle",
    "MaterializedServiceImage",
    "canonicalize_docker_archive",
    "capsule_compose_data",
    "export_image_bundle",
    "export_task_capsule",
    "load_capabilities",
    "materialize_service_images",
    "render_capsule_compose",
    "write_capsule_context",
]
