"""Local Docker runtime image helpers for template self-contained builds."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from alignerr_plugin.utils import load_task_toml

LOCAL_CPU_BASE_IMAGE = "lbx-tasks-base"
LOCAL_GPU_BASE_IMAGE = "lbx-tasks-base-gpu"
LOCAL_GPU_OPENROAD_BASE_IMAGE = "lbx-tasks-base-gpu-openroad"
LOCAL_GPU_BLACKWELL_BASE_IMAGE = "lbx-tasks-base-gpu-blackwell"
LOCAL_CUDA_GRAPHICS_BASE_IMAGE = "lbx-tasks-base-cuda-graphics"
LOCAL_TPU_BASE_IMAGE = "lbx-tasks-base-tpu"
LOCAL_BASE_TAG = "runtime-ml-core-py313-local"
LOCAL_PLATFORM = "linux/amd64"

# The local tag is a fixed name, so freshness rides on the same labels
# `base/build_and_push.sh` stamps onto the pushed bases: the drift hash is a
# content hash over every base build input, so a cached image whose label does
# not match the working tree was built from a different base/ and cannot be
# reused. Without this an author whose cache predates a base/ change gets the
# failure from inside the task's Docker build instead (e.g. a missing
# /opt/lbx-runtime/install-task-deps.sh), which names nothing useful.
LOCAL_BASE_DRIFT_LABEL = "lbx.base.drift_hash"
LOCAL_BASE_FLAVOR_LABEL = "lbx.base.flavor"

# Set to reuse a cached base that failed the drift check, for the case where an
# author knows their task does not depend on the base/ change in flight.
ALLOW_STALE_BASE_ENV = "LBX_RL_TASKS_ALLOW_STALE_BASE"

# Quoted to the author before a rebuild starts; a full base is large and slow
# enough (much slower under emulation) that triggering one unannounced is hostile.
_BASE_REBUILD_COST = "~16.5 GB of image layers and tens of minutes"


class BuildOOMError(RuntimeError):
    """A base-image build was killed by an out-of-memory signature.

    Subclasses RuntimeError so callers catching RuntimeError still handle it;
    raised separately from a generic build failure so the message can point at
    host memory rather than at the failing install phase.
    """


# Build-time OOM signatures: buildkit's "exit code: 137" for a SIGKILLed RUN step,
# plus compiler-OOM lines a from-source wheel emits when cc1/cc1plus is killed.
# Deliberately NOT "Cannot allocate memory": glibc emits that ENOMEM string for
# transient non-build allocation failures too (e.g. a package fetch/unpack under
# memory pressure), which would misclassify a recoverable error as a build OOM and
# trigger a spurious slim downgrade. A real step OOM is already caught by the
# buildkit "exit code: 137" line and the compiler-specific signatures above.
_OOM_PATTERNS = (
    "exit code: 137",
    "internal compiler error: Killed",
    "cc1plus: out of memory",
    "cc1: out of memory",
    "virtual memory exhausted",
    "Killed (program cc1",
)


def _is_build_oom(output: str, returncode: int) -> bool:
    """Whether a failed build looks OOM-killed, by OUTPUT SIGNATURE not exit code.

    A bare outer exit 137 is deliberately NOT treated as OOM (docker kill /
    timeout / Ctrl-C also exit 137), so a non-OOM interruption is never
    misreported as an out-of-memory host.
    """
    _ = returncode  # kept for signature/back-compat; classification is by output
    return any(pattern in output for pattern in _OOM_PATTERNS)


# Per-flavor (local image name, Dockerfile path).
_LOCAL_BASE_BY_FLAVOR: dict[str, tuple[str, str]] = {
    "cpu": (LOCAL_CPU_BASE_IMAGE, "base/cpu/Dockerfile"),
    "gpu": (LOCAL_GPU_BASE_IMAGE, "base/gpu/Dockerfile"),
    "gpu-openroad": (LOCAL_GPU_OPENROAD_BASE_IMAGE, "base/gpu-openroad/Dockerfile"),
    "gpu-blackwell": (
        LOCAL_GPU_BLACKWELL_BASE_IMAGE,
        "base/gpu-blackwell/Dockerfile",
    ),
    "cuda-graphics": (
        LOCAL_CUDA_GRAPHICS_BASE_IMAGE,
        "base/cuda-graphics/Dockerfile",
    ),
    "tpu": (LOCAL_TPU_BASE_IMAGE, "base/tpu/Dockerfile"),
}


# LOCAL-ONLY: both bases are cu130 now, but base/gpu is the CUDA -runtime image
# and exports no sm_120 arch flags, so on a Blackwell dev GPU (compute cap
# >= 10.0) any task CUDA extension builds for the wrong architecture (or not at
# all, with no nvcc). Swap in the blackwell overlay, which is -devel and pins
# sm_120. Never touches the Taiga export.
_LOCAL_BLACKWELL_OVERLAY: dict[str, str] = {
    "gpu": "gpu-blackwell",
}

_BLACKWELL_DETECTED: bool | None = None  # cached nvidia-smi probe result


def local_gpu_is_blackwell() -> bool:
    """Whether the local GPU needs the sm_120 Blackwell base (compute cap >= 10.0).

    Cached; ``LBX_RL_TASKS_LOCAL_BLACKWELL=1``/``0`` forces it on/off. Any probe
    failure defaults to False.
    """
    global _BLACKWELL_DETECTED
    override = os.environ.get("LBX_RL_TASKS_LOCAL_BLACKWELL")
    if override is not None:
        return override.strip().lower() not in ("", "0", "false", "no")
    if _BLACKWELL_DETECTED is not None:
        return _BLACKWELL_DETECTED
    _BLACKWELL_DETECTED = _probe_blackwell()
    return _BLACKWELL_DETECTED


def _probe_blackwell() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    for line in out.stdout.splitlines():
        try:
            if float(line.strip()) >= 10.0:
                return True
        except ValueError:
            continue
    return False


@dataclass(frozen=True)
class LocalBaseImage:
    image: str
    tag: str
    dockerfile: Path

    @property
    def ref(self) -> str:
        return f"{self.image}:{self.tag}"


def local_base_image_for_problem(problem_dir: Path) -> LocalBaseImage:
    """Return the repo-local flagship base image required by a task."""
    from alignerr_plugin.base_image import resolve_base_flavor_for_resource

    task_toml = load_task_toml(problem_dir)
    env = task_toml.environment
    resolved = resolve_base_flavor_for_resource(
        getattr(env, "base_flavor", "auto"), env.required_resources
    )
    # Local Blackwell swap (dev machines only; Taiga export is unaffected).
    if resolved in _LOCAL_BLACKWELL_OVERLAY and local_gpu_is_blackwell():
        overlay = _LOCAL_BLACKWELL_OVERLAY[resolved]
        print(
            f"Local Blackwell GPU detected: using {overlay} instead of {resolved} "
            "for the local build (cu130 / sm_120).",
            flush=True,
        )
        resolved = overlay
    image, dockerfile = _LOCAL_BASE_BY_FLAVOR[resolved]
    return LocalBaseImage(image=image, tag=LOCAL_BASE_TAG, dockerfile=Path(dockerfile))


def ensure_local_base_image(repo_root: Path, problem_dir: Path) -> LocalBaseImage:
    """Build the task's local base image (and its parent chain) if absent or stale."""
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required for local harness runs")
    from alignerr_plugin.base_image import base_drift_hash

    base = local_base_image_for_problem(problem_dir)
    drift = base_drift_hash(repo_root)
    if _can_reuse_cached_base(base, drift):
        return base
    inherited = _ensure_parent_base_image(repo_root, base, drift)
    _build_base_image(repo_root, base, inherited)
    return base


def _allow_stale_base() -> bool:
    value = os.environ.get(ALLOW_STALE_BASE_ENV, "")
    return value.strip().lower() not in ("", "0", "false", "no")


def _can_reuse_cached_base(base: LocalBaseImage, drift: str) -> bool:
    """Whether the cached local base matches the working tree's base/ inputs.

    Announces the reason and the cost before returning False for a stale cache,
    so a rebuild is never a surprise.
    """
    cached = _cached_base_drift_hash(base.ref)
    if cached is None:
        return False  # never built here; the ordinary first-run build path
    if cached == drift:
        return True
    reason = (
        f"it was built from a different base/ revision "
        f"(image {cached}, working tree {drift})"
        if cached
        else "it predates base-image drift labelling, so its contents "
        "cannot be checked against the current base/"
    )
    if _allow_stale_base():
        print(
            f"WARNING: reusing stale local base image {base.ref} because "
            f"{ALLOW_STALE_BASE_ENV} is set: {reason}. The task build will fail "
            "if it reaches for anything the current base/ provides that this "
            "image predates, such as /opt/lbx-runtime/install-task-deps.sh.",
            flush=True,
        )
        return True
    print(
        f"Local base image {base.ref} is stale: {reason}.\n"
        f"Rebuilding it so the task builds against the current base/ "
        f"(costs {_BASE_REBUILD_COST}).\n"
        f"Set {ALLOW_STALE_BASE_ENV}=1 to reuse the cached image instead, only "
        "if your task does not depend on the base/ change.",
        flush=True,
    )
    return False


def _build_base_image(repo_root: Path, base: LocalBaseImage, drift: str) -> None:
    dockerfile = repo_root / base.dockerfile
    if not dockerfile.exists():
        raise FileNotFoundError(f"local base Dockerfile not found: {dockerfile}")
    print(f"Building local base image {base.ref} from {dockerfile}...", flush=True)
    # Tee the build output: stream live while capturing it to classify OOM failures.
    proc = subprocess.Popen(
        [
            "docker",
            "build",
            "--progress",
            "plain",
            "--platform",
            LOCAL_PLATFORM,
            "--file",
            str(dockerfile),
            "--tag",
            base.ref,
            "--label",
            f"{LOCAL_BASE_FLAVOR_LABEL}={_flavor_for_base(base) or ''}",
            "--label",
            f"{LOCAL_BASE_DRIFT_LABEL}={drift}",
            str(repo_root),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Pump output in a background thread so the 3600s deadline is enforced by
    # proc.wait(timeout=...) even when the build wedges producing no output.
    captured: list[str] = []
    assert proc.stdout is not None

    def _pump(stream) -> None:
        for line in stream:
            sys.stdout.write(line)
            captured.append(line)

    reader = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
    reader.start()
    try:
        returncode = proc.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(
            f"local base image build for {base.ref} timed out after 3600s"
        )
    reader.join(timeout=5)
    if returncode != 0:
        output = "".join(captured)
        if _is_build_oom(output, returncode):
            raise BuildOOMError(
                f"local base image build for {base.ref} was OOM-killed "
                f"(exit {returncode}); output matched an out-of-memory signature. "
                "Raise the Docker VM memory limit or build on a larger host."
            )
        raise RuntimeError(
            f"local base image build failed for {base.ref}. "
            "See the Docker output above for the failing install-common phase. "
            "If the failure happened while downloading or exporting packages, "
            "check Docker disk usage with `docker system df`."
        )


def _flavor_for_base(base: LocalBaseImage) -> str | None:
    return next(
        (
            name
            for name, (image, dockerfile) in _LOCAL_BASE_BY_FLAVOR.items()
            if image == base.image and Path(dockerfile) == base.dockerfile
        ),
        None,
    )


def _ensure_parent_base_image(repo_root: Path, base: LocalBaseImage, drift: str) -> str:
    """Build the local parent-chain (bottom-up) before an overlay flavor that FROMs
    it, stopping at the first ancestor image that is present and current.

    Returns the drift hash the resulting chain actually represents, which is
    ``drift`` unless an ancestor was stale and reused under
    ``ALLOW_STALE_BASE_ENV``. Labelling the child with the current hash in that
    case would launder the ancestor's staleness: the overlay would look current
    forever after, and one override would permanently disarm the drift check for
    it. Inheriting the stale hash instead keeps the next run honest.
    """
    from alignerr_plugin.base_image import BASE_FLAVORS

    flavor = _flavor_for_base(base)
    if flavor is None:
        return drift
    parent = BASE_FLAVORS[flavor].parent
    if not parent:
        return drift
    parent_image, parent_dockerfile = _LOCAL_BASE_BY_FLAVOR[parent]
    parent_base = LocalBaseImage(
        image=parent_image, tag=base.tag, dockerfile=Path(parent_dockerfile)
    )
    if _can_reuse_cached_base(parent_base, drift):
        cached = _cached_base_drift_hash(parent_base.ref)
        return drift if cached == drift else (cached or "")
    inherited = _ensure_parent_base_image(repo_root, parent_base, drift)
    _build_base_image(repo_root, parent_base, inherited)
    return inherited


def _cached_base_drift_hash(image_ref: str) -> str | None:
    """Drift hash labelled on a locally cached base, or None when it isn't cached.

    Returns an empty string for an image that exists but carries no drift label,
    i.e. one built before this repo labelled local bases.
    """
    completed = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            f'{{{{index .Config.Labels "{LOCAL_BASE_DRIFT_LABEL}"}}}}',
            image_ref,
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0:
        return None
    label = completed.stdout.strip()
    # Go templates render a missing map key as "<no value>".
    return "" if label == "<no value>" else label
