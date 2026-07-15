"""Base-image flavors and content drift hashing.

Base images are tagged with a 12-char hash over everything that determines their
contents (Dockerfiles, install scripts, requirements, baked grader/taiga_runtime
sources), so a stale base can never be silently reused.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

BASE_TAG_PREFIX = "runtime-ml-core-py313"
BLACKWELL_BASE_TAG_PREFIX = "runtime-ml-blackwell-py313"
TPU_BASE_TAG_PREFIX = "runtime-ml-tpu-py312"
# mlenvs-specific bases (py3.12), not shared with the native flavors.
MLENVS_BASE_TAG_PREFIX = "runtime-mlenvs-py312"
# Minimal py3.12 base (no heavy ML stack) for local harness runs on weak hosts.
MLENVS_SLIM_BASE_TAG_PREFIX = "runtime-mlenvs-slim-py312"
# cu128 / sm_120 overlays, local-dev only, NOT deployable on Taiga.
MLENVS_BLACKWELL_BASE_TAG_PREFIX = "runtime-mlenvs-blackwell-py312"
BLACKWELL_GPU_TYPES = frozenset({"b100", "b200", "gb200"})


@dataclass(frozen=True)
class BaseFlavor:
    """A buildable base-image flavor."""

    name: str
    image_suffix: str  # appended to the base image repo (e.g. "-gpu")
    dockerfile: str  # path relative to the repo root
    tag_prefix: str = BASE_TAG_PREFIX
    parent: str | None = None


BASE_FLAVORS: dict[str, BaseFlavor] = {
    "cpu": BaseFlavor("cpu", "", "base/cpu/Dockerfile"),
    "gpu": BaseFlavor("gpu", "-gpu", "base/gpu/Dockerfile"),
    "gpu-openroad": BaseFlavor(
        "gpu-openroad",
        "-gpu-openroad",
        "base/gpu-openroad/Dockerfile",
        parent="gpu",
    ),
    "gpu-blackwell": BaseFlavor(
        "gpu-blackwell",
        "-gpu-blackwell",
        "base/gpu-blackwell/Dockerfile",
        tag_prefix=BLACKWELL_BASE_TAG_PREFIX,
    ),
    "cuda-graphics": BaseFlavor(
        "cuda-graphics", "-cuda-graphics", "base/cuda-graphics/Dockerfile"
    ),
    "tpu": BaseFlavor(
        "tpu", "-tpu", "base/tpu/Dockerfile", tag_prefix=TPU_BASE_TAG_PREFIX
    ),
    # --- mlenvs-specific flavors (not shared with native verticals) ----------
    "mlenvs-slim": BaseFlavor(
        "mlenvs-slim", "-mlenvs-slim", "base/mlenvs-slim/Dockerfile",
        tag_prefix=MLENVS_SLIM_BASE_TAG_PREFIX,
    ),
    "mlenvs-gpu": BaseFlavor(
        "mlenvs-gpu", "-mlenvs-gpu", "base/mlenvs-gpu/Dockerfile",
        tag_prefix=MLENVS_BASE_TAG_PREFIX,
    ),
    "mlenvs-cuda-graphics": BaseFlavor(
        "mlenvs-cuda-graphics", "-mlenvs-cuda-graphics",
        "base/mlenvs-cuda-graphics/Dockerfile", tag_prefix=MLENVS_BASE_TAG_PREFIX,
        parent="mlenvs-gpu",
    ),
    "mlenvs-tpu": BaseFlavor(
        "mlenvs-tpu", "-mlenvs-tpu", "base/mlenvs-tpu/Dockerfile",
        tag_prefix=MLENVS_BASE_TAG_PREFIX,
    ),
    # Blackwell overlays: local-dev only (cu128 / sm_120); never auto-selected.
    "mlenvs-gpu-blackwell": BaseFlavor(
        "mlenvs-gpu-blackwell", "-mlenvs-gpu-blackwell",
        "base/mlenvs-gpu-blackwell/Dockerfile",
        tag_prefix=MLENVS_BLACKWELL_BASE_TAG_PREFIX, parent="mlenvs-gpu",
    ),
    "mlenvs-cuda-graphics-blackwell": BaseFlavor(
        "mlenvs-cuda-graphics-blackwell", "-mlenvs-cuda-graphics-blackwell",
        "base/mlenvs-cuda-graphics-blackwell/Dockerfile",
        tag_prefix=MLENVS_BLACKWELL_BASE_TAG_PREFIX, parent="mlenvs-cuda-graphics",
    ),
}

# The mlenvs-mode flavors, in one place for the resolver / build tooling.
MLENVS_FLAVORS: frozenset[str] = frozenset(
    {
        "mlenvs-slim",
        "mlenvs-gpu",
        "mlenvs-cuda-graphics",
        "mlenvs-tpu",
        "mlenvs-gpu-blackwell",
        "mlenvs-cuda-graphics-blackwell",
    }
)

# Globs (relative to repo root) whose file contents determine a base image.
_HASH_GLOBS = (
    "base/**/Dockerfile",
    "base/*.sh",
    "base/requirements-*.txt",
    "base/clear_execstack.py",
    "grader/src/**/*.py",
    "grader/pyproject.toml",
    "taiga_runtime/**/*.py",
    "taiga_runtime/**/*.toml",
)
_HASH_EXCLUDE_PARTS = {
    "__pycache__",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "tests",
}


def _hash_input_files(repo_root: Path) -> list[Path]:
    seen: set[Path] = set()
    for pattern in _HASH_GLOBS:
        for path in repo_root.glob(pattern):
            if not path.is_file():
                continue
            if any(part in _HASH_EXCLUDE_PARTS for part in path.parts):
                continue
            seen.add(path)
    return sorted(seen, key=lambda p: p.relative_to(repo_root).as_posix())


def base_drift_hash(repo_root: Path) -> str:
    """12-char content hash over all base-image build inputs."""
    digest = hashlib.sha256()
    for path in _hash_input_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def base_image_tag(repo_root: Path, flavor: str = "cpu") -> str:
    """Drift-hashed tag for ``flavor`` (e.g. ``runtime-ml-core-py313-ab12cd34ef56``)."""
    base_flavor = BASE_FLAVORS.get(flavor, BASE_FLAVORS["cpu"])
    return f"{base_flavor.tag_prefix}-{base_drift_hash(repo_root)}"


BASE_FLAVOR_CHOICES: tuple[str, ...] = ("auto", *BASE_FLAVORS.keys())


def _normalize_gpu_type(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def uses_blackwell_gpu(gpu_types: list[str] | tuple[str, ...] | None) -> bool:
    """Return whether any requested GPU type needs the Blackwell base."""
    return any(
        _normalize_gpu_type(gpu_type) in BLACKWELL_GPU_TYPES
        for gpu_type in (gpu_types or ())
    )


def resolve_base_flavor(
    declared: str | None,
    gpus: int,
    gpu_types: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Resolve the effective base flavor.

    ``"auto"`` (the default) maps to ``gpu`` when the task requests a GPU, else
    ``cpu``. Blackwell GPU types auto-select ``gpu-blackwell``. An explicit
    flavor must be one of :data:`BASE_FLAVORS`.
    """
    value = (declared or "auto").strip().lower().replace("_", "-")
    if value == "auto":
        if gpus > 0 and uses_blackwell_gpu(gpu_types):
            return "gpu-blackwell"
        return "gpu" if gpus > 0 else "cpu"
    if value not in BASE_FLAVORS:
        raise ValueError(
            f"unknown base_flavor {declared!r}; must be one of "
            f"{sorted(BASE_FLAVOR_CHOICES)}"
        )
    return value


def resolve_base_flavor_for_resource(
    declared: str | None,
    required_resources: str,
) -> str:
    """Resolve the base flavor from a Taiga resource enum.

    ``"auto"`` maps CPU tiers to ``cpu``, H100 tiers to ``gpu``, graphics H100
    tiers to ``cuda-graphics``, and TPU tiers to ``tpu``. Explicit flavors must
    be compatible with the selected Taiga resource tier.
    """
    from alignerr_plugin.taiga_resources import (
        is_cpu_resource,
        is_graphics_resource,
        is_h100_resource,
        is_tpu_resource,
    )

    value = (declared or "auto").strip().lower().replace("_", "-")
    if value not in BASE_FLAVOR_CHOICES:
        raise ValueError(
            f"unknown base_flavor {declared!r}; must be one of "
            f"{sorted(BASE_FLAVOR_CHOICES)}"
        )

    if value == "auto":
        if is_tpu_resource(required_resources):
            return "tpu"
        if is_graphics_resource(required_resources):
            return "cuda-graphics"
        if is_h100_resource(required_resources):
            return "gpu"
        return "cpu"

    if value == "cpu" and not is_cpu_resource(required_resources):
        raise ValueError(
            f"[environment].base_flavor = {declared!r} requires a CPU "
            f"required_resources tier (got {required_resources!r})"
        )
    if value == "tpu" and not is_tpu_resource(required_resources):
        raise ValueError(
            f"[environment].base_flavor = {declared!r} requires a TPU "
            f"required_resources tier (got {required_resources!r})"
        )
    if value == "cuda-graphics" and not is_graphics_resource(required_resources):
        raise ValueError(
            f"[environment].base_flavor = {declared!r} requires a graphics "
            f"required_resources tier (got {required_resources!r})"
        )
    if value in {"gpu", "gpu-openroad"} and (
        not is_h100_resource(required_resources)
        or is_graphics_resource(required_resources)
    ):
        raise ValueError(
            f"[environment].base_flavor = {declared!r} requires a non-graphics "
            f"H100 required_resources tier (got {required_resources!r})"
        )
    if value == "gpu-blackwell":
        raise ValueError(
            "[environment].base_flavor = 'gpu-blackwell' is not supported by "
            "the current Taiga required_resources enum"
        )
    # Blackwell overlays are local-dev only; not valid for any Taiga tier.
    if value in {"mlenvs-gpu-blackwell", "mlenvs-cuda-graphics-blackwell"}:
        raise ValueError(
            f"[environment].base_flavor = {declared!r} is a local-only Blackwell "
            "overlay and is not supported by the current Taiga required_resources "
            "enum"
        )
    # mlenvs-gpu on non-graphics/non-TPU tiers; mlenvs-cuda-graphics on graphics;
    # mlenvs-tpu on TPU. Mirrors the native gpu/cuda-graphics/tpu tier rules.
    if value == "mlenvs-gpu" and (
        is_graphics_resource(required_resources) or is_tpu_resource(required_resources)
    ):
        raise ValueError(
            f"[environment].base_flavor = {declared!r} requires a non-graphics, "
            f"non-TPU required_resources tier (got {required_resources!r})"
        )
    if value == "mlenvs-cuda-graphics" and not is_graphics_resource(required_resources):
        raise ValueError(
            f"[environment].base_flavor = {declared!r} requires a graphics "
            f"required_resources tier (got {required_resources!r})"
        )
    if value == "mlenvs-tpu" and not is_tpu_resource(required_resources):
        raise ValueError(
            f"[environment].base_flavor = {declared!r} requires a TPU "
            f"required_resources tier (got {required_resources!r})"
        )
    return value


def expand_base_flavors(flavors: list[str] | tuple[str, ...]) -> list[str]:
    """Return flavors with parent dependencies inserted before their children."""
    expanded: list[str] = []
    visiting: set[str] = set()

    def add(flavor: str) -> None:
        value = flavor.strip().lower().replace("_", "-")
        if value not in BASE_FLAVORS:
            raise ValueError(
                f"unknown base flavor {flavor!r}; must be one of {sorted(BASE_FLAVORS)}"
            )
        if value in expanded:
            return
        if value in visiting:
            raise ValueError(f"cycle in base flavor parents at {value!r}")
        visiting.add(value)
        parent = BASE_FLAVORS[value].parent
        if parent:
            add(parent)
        visiting.remove(value)
        if value not in expanded:
            expanded.append(value)

    for flavor in flavors:
        if flavor.strip():
            add(flavor)
    return expanded
