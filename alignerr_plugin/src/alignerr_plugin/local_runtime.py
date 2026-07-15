"""Local Docker runtime image helpers for template self-contained builds."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path

from alignerr_plugin.utils import load_task_toml

LOCAL_CPU_BASE_IMAGE = "lbx-tasks-base"
LOCAL_GPU_BASE_IMAGE = "lbx-tasks-base-gpu"
LOCAL_GPU_OPENROAD_BASE_IMAGE = "lbx-tasks-base-gpu-openroad"
LOCAL_GPU_BLACKWELL_BASE_IMAGE = "lbx-tasks-base-gpu-blackwell"
LOCAL_CUDA_GRAPHICS_BASE_IMAGE = "lbx-tasks-base-cuda-graphics"
LOCAL_TPU_BASE_IMAGE = "lbx-tasks-base-tpu"
LOCAL_MLENVS_SLIM_BASE_IMAGE = "lbx-tasks-base-mlenvs-slim"
# mlenvs-specific bases (not shared with native flavors).
LOCAL_MLENVS_GPU_BASE_IMAGE = "lbx-tasks-base-mlenvs-gpu"
LOCAL_MLENVS_CUDA_GRAPHICS_BASE_IMAGE = "lbx-tasks-base-mlenvs-cuda-graphics"
LOCAL_MLENVS_TPU_BASE_IMAGE = "lbx-tasks-base-mlenvs-tpu"
# Blackwell overlays (local-only cu128 / sm_120 dev builds).
LOCAL_MLENVS_GPU_BLACKWELL_BASE_IMAGE = "lbx-tasks-base-mlenvs-gpu-blackwell"
LOCAL_MLENVS_CUDA_GRAPHICS_BLACKWELL_BASE_IMAGE = (
    "lbx-tasks-base-mlenvs-cuda-graphics-blackwell"
)
LOCAL_BASE_TAG = "runtime-ml-core-py313-local"
LOCAL_PLATFORM = "linux/amd64"

# Compute base "flavor". "heavy" is the production-equivalent base; "slim" is the
# local-only stripped mlenvs base for low-RAM dev hosts where the heavy image OOMs.
# "auto" tries heavy and falls back to slim on a build-time OOM. Slim is only
# meaningful for compute tasks (no GPU/graphics/TPU stack; the agent pip-installs
# at runtime).
VALID_FLAVORS = ("auto", "heavy", "slim")
# Extra agent turns on a slim fallback, to offset the runtime pip-install overhead.
SLIM_FALLBACK_TURN_BONUS = 50


class BuildOOMError(RuntimeError):
    """A base-image build was killed by an out-of-memory signature.

    Subclasses RuntimeError so callers catching RuntimeError still handle it; the
    flavor orchestrator catches it specifically to fall back to the slim base.
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

    A bare outer exit 137 is deliberately NOT treated as OOM (docker kill / timeout
    / Ctrl-C also exit 137), so auto-mode never silently downgrades to slim on a
    non-OOM interruption.
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
    "mlenvs-slim": (LOCAL_MLENVS_SLIM_BASE_IMAGE, "base/mlenvs-slim/Dockerfile"),
    "mlenvs-gpu": (LOCAL_MLENVS_GPU_BASE_IMAGE, "base/mlenvs-gpu/Dockerfile"),
    "mlenvs-cuda-graphics": (
        LOCAL_MLENVS_CUDA_GRAPHICS_BASE_IMAGE,
        "base/mlenvs-cuda-graphics/Dockerfile",
    ),
    "mlenvs-tpu": (LOCAL_MLENVS_TPU_BASE_IMAGE, "base/mlenvs-tpu/Dockerfile"),
    "mlenvs-gpu-blackwell": (
        LOCAL_MLENVS_GPU_BLACKWELL_BASE_IMAGE,
        "base/mlenvs-gpu-blackwell/Dockerfile",
    ),
    "mlenvs-cuda-graphics-blackwell": (
        LOCAL_MLENVS_CUDA_GRAPHICS_BLACKWELL_BASE_IMAGE,
        "base/mlenvs-cuda-graphics-blackwell/Dockerfile",
    ),
}


# LOCAL-ONLY: on a Blackwell dev GPU (compute cap >= 10.0) the cu121 bases can't
# run their kernels, so swap in the cu128 overlay. Never touches the Taiga export.
_LOCAL_BLACKWELL_OVERLAY: dict[str, str] = {
    "gpu": "gpu-blackwell",
    "mlenvs-gpu": "mlenvs-gpu-blackwell",
    "mlenvs-cuda-graphics": "mlenvs-cuda-graphics-blackwell",
}

_BLACKWELL_DETECTED: bool | None = None  # cached nvidia-smi probe result


def local_gpu_is_blackwell() -> bool:
    """Whether the local GPU needs a cu128 Blackwell base (compute cap >= 10.0).

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
    # Resolved tier ("heavy" | "slim") and whether slim was reached via a heavy OOM.
    flavor: str = "heavy"
    flavor_fallback: bool = False

    @property
    def ref(self) -> str:
        return f"{self.image}:{self.tag}"


def local_base_image_for_problem(
    problem_dir: Path, *, flavor: str = "heavy"
) -> LocalBaseImage:
    """Return the repo-local base image required by a task.

    ``flavor="heavy"`` resolves the task's compute base (with the local Blackwell
    overlay swap); ``flavor="slim"`` short-circuits to the stripped ``mlenvs-slim``
    base. ``flavor`` here is the resolved TIER; ``auto`` is decided one level up in
    :func:`ensure_local_base_image`.
    """
    if flavor == "slim":
        image, dockerfile = _LOCAL_BASE_BY_FLAVOR["mlenvs-slim"]
        return LocalBaseImage(
            image=image, tag=LOCAL_BASE_TAG, dockerfile=Path(dockerfile), flavor="slim"
        )

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
            "for the local build (cu128 / sm_120).",
            flush=True,
        )
        resolved = overlay
    image, dockerfile = _LOCAL_BASE_BY_FLAVOR[resolved]
    return LocalBaseImage(
        image=image, tag=LOCAL_BASE_TAG, dockerfile=Path(dockerfile), flavor="heavy"
    )


def _slim_available(problem_dir: Path) -> bool:
    """Whether slim can substitute: an ML_Envs compute task (heavy base is
    ``mlenvs-gpu`` / its Blackwell overlay). Graphics/TPU tasks need their heavy base."""
    from alignerr_plugin import mlenvs

    if not mlenvs.is_mlenvs_task(problem_dir):
        return False
    heavy = local_base_image_for_problem(problem_dir, flavor="heavy")
    return heavy.image in {
        LOCAL_MLENVS_GPU_BASE_IMAGE,
        LOCAL_MLENVS_GPU_BLACKWELL_BASE_IMAGE,
    }


def _ensure_built(repo_root: Path, base: LocalBaseImage) -> None:
    """Build ``base`` (and its parent chain) if it is not already present."""
    if _docker_image_exists(base.ref):
        return
    _ensure_parent_base_image(repo_root, base)
    _build_base_image(repo_root, base)


def ensure_local_base_image(
    repo_root: Path, problem_dir: Path, *, flavor: str = "auto"
) -> LocalBaseImage:
    """Build the task's local base image if absent, honoring the compute flavor.

    ``auto`` builds heavy and, on a build-time OOM, falls back to slim (when
    viable). ``heavy`` forces the production base; ``slim`` forces the stripped
    base (rejected for GPU/graphics/TPU tasks).
    """
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required for local harness runs")
    if flavor not in VALID_FLAVORS:
        raise ValueError(f"--flavor must be one of {list(VALID_FLAVORS)}; got {flavor!r}")

    slim_ok = _slim_available(problem_dir)

    if flavor == "slim":
        if not slim_ok:
            raise RuntimeError(
                "--flavor slim is only available for ML_Envs compute tasks: the "
                "mlenvs-slim base carries no GPU/graphics/TPU stack, so a "
                "graphics / TPU (or non-ML_Envs) task must build its heavy base."
            )
        slim = local_base_image_for_problem(problem_dir, flavor="slim")
        _ensure_built(repo_root, slim)
        return slim

    if flavor == "heavy":
        heavy = local_base_image_for_problem(problem_dir, flavor="heavy")
        _ensure_built(repo_root, heavy)
        return heavy

    # auto: heavy first; on a build-time OOM fall back to slim when it is viable.
    heavy = local_base_image_for_problem(problem_dir, flavor="heavy")
    try:
        _ensure_built(repo_root, heavy)
        return heavy
    except BuildOOMError:
        if not slim_ok:
            raise
        bar = "=" * 72
        print(
            f"\n{bar}\n"
            f"Heavy base {heavy.ref} OOMed while building. Falling back to the\n"
            "local-only slim base; the agent installs ML wheels at runtime and\n"
            f"gets +{SLIM_FALLBACK_TURN_BONUS} turns. Use --flavor heavy to force\n"
            f"the production base on a larger host.\n{bar}",
            flush=True,
        )
        slim = replace(
            local_base_image_for_problem(problem_dir, flavor="slim"),
            flavor_fallback=True,
        )
        _ensure_built(repo_root, slim)
        return slim


def _build_base_image(repo_root: Path, base: LocalBaseImage) -> None:
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
                f"(exit {returncode}); output matched an out-of-memory signature."
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


def _ensure_parent_base_image(repo_root: Path, base: LocalBaseImage) -> None:
    """Build the local parent-chain (bottom-up) before an overlay flavor that FROMs
    it, stopping at the first ancestor image that already exists."""
    from alignerr_plugin.base_image import BASE_FLAVORS

    flavor = _flavor_for_base(base)
    if flavor is None:
        return
    parent = BASE_FLAVORS[flavor].parent
    if not parent:
        return
    parent_image, parent_dockerfile = _LOCAL_BASE_BY_FLAVOR[parent]
    parent_base = LocalBaseImage(
        image=parent_image, tag=base.tag, dockerfile=Path(parent_dockerfile)
    )
    if _docker_image_exists(parent_base.ref):
        return
    _ensure_parent_base_image(repo_root, parent_base)
    _build_base_image(repo_root, parent_base)


def _docker_image_exists(image_ref: str) -> bool:
    completed = subprocess.run(
        ["docker", "image", "inspect", image_ref],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    return completed.returncode == 0
