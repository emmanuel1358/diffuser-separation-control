from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from alignerr_plugin.base_image import (
    BASE_FLAVORS,
    BASE_TAG_PREFIX,
    base_drift_hash,
    base_image_tag,
    expand_base_flavors,
    resolve_base_flavor,
    resolve_base_flavor_for_resource,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_script_handles_empty_cpu_suffix_and_parent_args() -> None:
    script = REPO_ROOT / "base" / "build_and_push.sh"
    completed = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    text = script.read_text()
    assert "IFS='|' read -r suffix dockerfile tag_prefix" in text
    assert 'if [[ "${#parent_build_args[@]}" -gt 0 ]]' in text


def test_build_script_defaults_to_every_flavor() -> None:
    """A partial default is never correct.

    ``base_drift_hash`` hashes all base inputs globally, not per flavor, so any
    base edit moves every flavor's tag. ``expand_base_flavors`` only pulls in
    parents, so a flavor left out of the default list simply never gets built at
    the new tag.
    """
    script = (REPO_ROOT / "base" / "build_and_push.sh").read_text()
    default = next(
        line.split("=", 1)[1].strip('"')
        for line in script.splitlines()
        if line.startswith("FLAVORS=")
    )
    assert set(default.split(",")) == set(BASE_FLAVORS)


def test_install_common_restores_torchs_nccl_after_the_extras() -> None:
    """xgboost's nvidia-nccl-cu12 overwrites torch's nvidia-nccl-cu13.

    Both distributions install the same ``nvidia/nccl/lib/libnccl.so.2``, so the
    last one installed wins and torch loads whatever is left there. The repair
    only works if it is ordered after every install that can pull in the cu12
    build, and it has to be verified against the file on disk -- torch's own
    ``torch.cuda.nccl.version()`` reads a compile-time header constant and
    reports the pinned version either way.
    """
    text = (REPO_ROOT / "base" / "install-common.sh").read_text()

    extras = 'uv pip install --python "${RUNTIME_VENV_DIR}/bin/python" --no-cache "${runtime_constraints[@]}" "${torch_constraint_args[@]}" -r "${BASE_EXTRA_REQUIREMENTS}"'
    reinstall = "--reinstall-package nvidia-nccl-cu13"
    assert extras in text
    assert reinstall in text
    assert text.index(extras) < text.index(reinstall)

    # Version comes from torch's metadata, not a literal, so the two cannot drift.
    assert 'md.requires("torch")' in text
    assert '"nvidia-nccl-cu13==${nccl_pin}"' in text

    # The check has to dlopen the resolved library rather than trust metadata,
    # and it has to run again after the reinstall so a repair that did not take
    # fails the build.
    assert "ncclGetVersion" in text
    assert text.rindex('nccl_verify "${nccl_pin}"') > text.index(reinstall)


def test_nccl_check_skips_flavors_whose_torch_bundles_no_cuda() -> None:
    """The skip has to key off torch's metadata, not off package presence.

    cpu and tpu install a CPU-only torch, so nothing there declares an NCCL
    dependency and there is nothing to protect -- that must be a silent skip. But
    a flavor whose torch *does* pin nvidia-nccl-cu13 while the library is missing
    or disagrees has to fail loudly. Keying the guard on whether the distribution
    happens to be installed would collapse those two cases into one.
    """
    text = (REPO_ROOT / "base" / "install-common.sh").read_text()

    # No torch, or a torch that pins no NCCL -> print nothing and exit clean.
    assert "except md.PackageNotFoundError:\n    raise SystemExit(0)" in text
    assert "if not pins:\n    raise SystemExit(0)" in text
    # Empty pin is the skip branch, not an error branch.
    assert 'if [[ -z "${nccl_pin}" ]]; then' in text
    # But a declared pin with no installed distribution is a hard failure.
    assert "but it is not installed" in text


def test_every_flavor_pins_the_torch_build_it_installs() -> None:
    """An unpinned torch floats, and on cpu/tpu it floats all the way to CUDA.

    uv searches --extra-index-url before --index-url, so a bare `torch` against
    the PyTorch CPU index still resolves to PyPI's plain CUDA wheel and drags the
    whole nvidia-* stack -- cuDNN, cuBLAS, a 200 MB NCCL -- into an image that has
    no GPU. Every flavor that installs torch therefore names exact versions, and
    the accelerator-less flavors name +cpu builds and opt into best-match so that
    the local version is reachable at all.
    """
    cpu_only = {"cpu", "tpu"}
    seen: set[str] = set()
    for dockerfile in sorted((REPO_ROOT / "base").glob("*/Dockerfile")):
        flavor = dockerfile.parent.name
        text = dockerfile.read_text()
        if "TORCH_INDEX_URL" not in text:
            continue
        seen.add(flavor)

        match = re.search(r'ARG TORCH_PACKAGES="([^"]+)"', text)
        assert match, f"{flavor} installs torch but does not pin TORCH_PACKAGES"
        packages = match.group(1).split()
        assert packages, f"{flavor} has an empty TORCH_PACKAGES"
        for package in packages:
            assert "==" in package, f"{flavor} leaves {package} unpinned"

        if flavor in cpu_only:
            for package in packages:
                assert package.endswith("+cpu"), (
                    f"{flavor} has no accelerator, so {package} must be a +cpu "
                    "build or the CUDA wheel comes back"
                )
            assert "ENV TORCH_INDEX_STRATEGY=unsafe-best-match" in text, (
                f"{flavor} pins a +cpu local version, which uv cannot reach "
                "under its default first-index strategy"
            )

    assert cpu_only <= seen, f"cpu/tpu must install torch explicitly, saw {seen}"
    assert {"gpu", "gpu-blackwell", "cuda-graphics"} <= seen, seen


def test_install_common_seeds_pip_into_the_runtime_venv() -> None:
    """`uv venv` does not seed pip, so it must be installed explicitly.

    Without this, /usr/local/bin/pip is a dangling symlink, `python -m pip`
    fails, and the only pip3 on PATH belongs to the system CPython whose
    site-packages the runtime venv cannot see.
    """
    script = REPO_ROOT / "base" / "install-common.sh"
    completed = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    text = script.read_text()
    seed_pip = 'uv pip install --python "${RUNTIME_VENV_DIR}/bin/python" --no-cache pip'
    assert seed_pip in text
    # Both pip names must resolve to the runtime venv, not the system CPython.
    assert "rm -f /usr/local/bin/pip /usr/local/bin/pip3" in text
    assert 'ln -sf "${RUNTIME_VENV_DIR}/bin/pip" /usr/local/bin/pip\n' in text
    assert 'ln -sf "${RUNTIME_VENV_DIR}/bin/pip" /usr/local/bin/pip3\n' in text
    # pip must be seeded before the venv is made world-readable, or the agent
    # uid cannot execute it.
    world_readable = 'chmod -R a+rX "${UV_PYTHON_INSTALL_DIR}"'
    assert text.index(seed_pip) < text.index(world_readable)


def test_base_ships_a_headless_gl_backend() -> None:
    """mujoco ships in every flavor, so every flavor needs a usable GL backend.

    `libgl1` alone is only the client-side dispatch library; without a platform
    library mujoco.Renderer dies with "an OpenGL platform library has not been
    loaded". osmesa rasterizes on CPU, so it works on every flavor regardless of
    whether an accelerator is attached.
    """
    common = (REPO_ROOT / "base" / "install-common.sh").read_text()
    for package in ("libosmesa6", "libegl1"):
        assert f"  {package} \\\n" in common, f"{package} missing from apt list"

    # The osmesa backend is reached through PyOpenGL, and mujoco lives in the
    # runtime venv, so pyopengl has to be a runtime requirement too.
    runtime_reqs = (REPO_ROOT / "base" / "requirements-runtime.txt").read_text()
    assert "mujoco==" in runtime_reqs
    assert "pyopengl==" in runtime_reqs

    # Every flavor that installs the GL libraries must also pin the backend,
    # rather than let mujoco probe into one that cannot make a context. The set
    # is derived from the Dockerfiles instead of hardcoded so a newly added
    # flavor cannot inherit the libraries and silently skip the pin.
    flavors = sorted(
        path.parent.name
        for path in (REPO_ROOT / "base").glob("*/Dockerfile")
        if any(
            line.lstrip().startswith("RUN") and "install-common.sh" in line
            for line in path.read_text().splitlines()
        )
    )
    assert {"cpu", "cuda-graphics", "gpu", "gpu-blackwell", "tpu"} <= set(
        flavors
    ), flavors
    for flavor in flavors:
        dockerfile = (REPO_ROOT / "base" / flavor / "Dockerfile").read_text()
        # No exceptions, including cuda-graphics: its CUDA-native raster path
        # (pytorch3d/nvdiffrast) does not go through PyOpenGL, and hardware EGL
        # is not deployable under Taiga's gVisor + nvproxy sandbox anyway, so
        # pinning egl there only broke mujoco.Renderer.
        assert (
            "ENV PYOPENGL_PLATFORM=osmesa" in dockerfile
        ), f"{flavor} does not pin PYOPENGL_PLATFORM=osmesa"
        assert (
            "ENV MUJOCO_GL=osmesa" in dockerfile
        ), f"{flavor} does not pin MUJOCO_GL=osmesa"


def _fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "base" / "cpu").mkdir(parents=True)
    (tmp_path / "base" / "cpu" / "Dockerfile").write_text("FROM python:3.13-slim\n")
    (tmp_path / "base" / "install-common.sh").write_text("#!/bin/sh\n")
    (tmp_path / "base" / "requirements-common.txt").write_text("numpy\n")
    (tmp_path / "grader" / "src" / "grading").mkdir(parents=True)
    (tmp_path / "grader" / "src" / "grading" / "x.py").write_text("x = 1\n")
    (tmp_path / "grader" / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "taiga_runtime").mkdir(parents=True)
    (tmp_path / "taiga_runtime" / "y.py").write_text("y = 2\n")
    return tmp_path


def test_drift_hash_stable_and_12_hex(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    h1 = base_drift_hash(repo)
    h2 = base_drift_hash(repo)
    assert h1 == h2
    assert len(h1) == 12
    assert all(ch in "0123456789abcdef" for ch in h1)


def test_drift_hash_changes_on_base_input_change(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    before = base_drift_hash(repo)
    (repo / "base" / "cpu" / "Dockerfile").write_text(
        "FROM python:3.13-slim\nRUN echo changed\n"
    )
    assert base_drift_hash(repo) != before


def test_drift_hash_changes_on_grader_source_change(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    before = base_drift_hash(repo)
    (repo / "grader" / "src" / "grading" / "x.py").write_text("x = 2\n")
    assert base_drift_hash(repo) != before


def test_drift_hash_ignores_tests_and_pycache(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    before = base_drift_hash(repo)
    pycache = repo / "grader" / "src" / "grading" / "__pycache__"
    pycache.mkdir(parents=True, exist_ok=True)
    (pycache / "x.cpython.py").write_text("junk\n")
    tests = repo / "grader" / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "test_x.py").write_text("junk\n")
    assert base_drift_hash(repo) == before


def test_base_image_tag_uses_flavor_prefix(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    cpu_tag = base_image_tag(repo, "cpu")
    tpu_tag = base_image_tag(repo, "tpu")
    assert cpu_tag.startswith(f"{BASE_TAG_PREFIX}-")
    assert tpu_tag.startswith(f"{BASE_FLAVORS['tpu'].tag_prefix}-")
    assert cpu_tag.split("-")[-1] == tpu_tag.split("-")[-1]  # same drift hash


def test_all_flavors_have_distinct_suffixes() -> None:
    suffixes = [flavor.image_suffix for flavor in BASE_FLAVORS.values()]
    assert len(suffixes) == len(set(suffixes))
    assert BASE_FLAVORS["cpu"].image_suffix == ""
    assert BASE_FLAVORS["gpu-openroad"].image_suffix == "-gpu-openroad"
    assert BASE_FLAVORS["gpu-openroad"].parent == "gpu"
    assert BASE_FLAVORS["gpu-blackwell"].image_suffix == "-gpu-blackwell"
    assert BASE_FLAVORS["cuda-graphics"].image_suffix == "-cuda-graphics"


def test_resolve_base_flavor_auto() -> None:
    assert resolve_base_flavor("auto", 0) == "cpu"
    assert resolve_base_flavor("auto", 1) == "gpu"
    assert resolve_base_flavor("auto", 1, ["B200"]) == "gpu-blackwell"
    assert resolve_base_flavor("auto", 1, ["GB200"]) == "gpu-blackwell"
    assert resolve_base_flavor(None, 0) == "cpu"
    assert resolve_base_flavor("", 2) == "gpu"


def test_resolve_base_flavor_explicit_and_underscore_tolerant() -> None:
    assert resolve_base_flavor("cpu", 4) == "cpu"
    assert resolve_base_flavor("gpu_openroad", 1) == "gpu-openroad"
    assert resolve_base_flavor("gpu_blackwell", 1) == "gpu-blackwell"
    assert resolve_base_flavor("cuda-graphics", 1) == "cuda-graphics"
    assert resolve_base_flavor("cuda_graphics", 1) == "cuda-graphics"
    assert resolve_base_flavor("TPU", 0) == "tpu"


def test_resolve_base_flavor_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        resolve_base_flavor("quantum", 0)


def test_resolve_base_flavor_for_resource_auto() -> None:
    assert resolve_base_flavor_for_resource("auto", "4vcpu+16gib") == "cpu"
    assert resolve_base_flavor_for_resource("auto", "12vcpu+100gib+h100/2") == "gpu"
    assert (
        resolve_base_flavor_for_resource("auto", "12vcpu+100gib+h100/2+graphics")
        == "cuda-graphics"
    )
    assert resolve_base_flavor_for_resource("auto", "13vcpu+32gib+tpuv5e1x1") == "tpu"


def test_resolve_base_flavor_for_resource_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="requires a CPU"):
        resolve_base_flavor_for_resource("cpu", "12vcpu+100gib+h100/2")
    with pytest.raises(ValueError, match="requires a TPU"):
        resolve_base_flavor_for_resource("tpu", "4vcpu+16gib")
    with pytest.raises(ValueError, match="requires a graphics"):
        resolve_base_flavor_for_resource("cuda-graphics", "12vcpu+100gib+h100/2")
    with pytest.raises(ValueError, match="requires a non-graphics H100"):
        resolve_base_flavor_for_resource("gpu", "12vcpu+100gib+h100/2+graphics")
    with pytest.raises(ValueError, match="not supported"):
        resolve_base_flavor_for_resource("gpu-blackwell", "12vcpu+100gib+h100/2")


def test_expand_base_flavors_inserts_parents_once() -> None:
    assert expand_base_flavors(["gpu-openroad", "gpu"]) == ["gpu", "gpu-openroad"]


def test_exporter_selects_base_image_by_flavor() -> None:
    from alignerr_plugin.exporters.taiga import (
        CPU_BASE_IMAGE,
        CUDA_GRAPHICS_BASE_IMAGE,
        GPU_BASE_IMAGE,
        GPU_BLACKWELL_BASE_IMAGE,
        GPU_OPENROAD_BASE_IMAGE,
        TPU_BASE_IMAGE,
        _base_image_and_tag,
    )

    assert _base_image_and_tag("cpu")[0] == CPU_BASE_IMAGE
    assert _base_image_and_tag("gpu")[0] == GPU_BASE_IMAGE
    assert _base_image_and_tag("gpu-openroad")[0] == GPU_OPENROAD_BASE_IMAGE
    blackwell_image, blackwell_tag = _base_image_and_tag("gpu-blackwell")
    assert blackwell_image == GPU_BLACKWELL_BASE_IMAGE
    assert blackwell_tag.startswith("runtime-ml-blackwell-py313-")
    assert _base_image_and_tag("cuda-graphics")[0] == CUDA_GRAPHICS_BASE_IMAGE
    tpu_image, tpu_tag = _base_image_and_tag("tpu")
    assert tpu_image == TPU_BASE_IMAGE
    assert tpu_tag.startswith("runtime-ml-tpu-py312-")


def test_environment_resource_required_and_base_compatible() -> None:
    from alignerr_plugin.schemas import EnvironmentSection

    with pytest.raises(ValueError):
        EnvironmentSection()
    with pytest.raises(ValueError, match="Extra inputs"):
        EnvironmentSection(required_resources="4vcpu+16gib", cpus=4)
    with pytest.raises(ValueError, match="requires a TPU"):
        EnvironmentSection(required_resources="4vcpu+16gib", base_flavor="tpu")
    assert (
        EnvironmentSection(
            required_resources="12vcpu+100gib+h100/2+graphics",
            base_flavor="cuda-graphics",
        ).base_flavor
        == "cuda-graphics"
    )
    assert (
        EnvironmentSection(required_resources="13vcpu+32gib+tpuv5e1x1").base_flavor
        == "auto"
    )


def test_local_runtime_selects_flavor_dockerfile(tmp_path: Path) -> None:
    from alignerr_plugin.local_runtime import (
        LOCAL_CUDA_GRAPHICS_BASE_IMAGE,
        local_base_image_for_problem,
    )

    (tmp_path / "task.toml").write_text(
        "[task]\nname = 'x'\n"
        "[difficulty]\ntask_type = 'ml'\n"
        "domain = 'scientific_discovery_computational_science'\n"
        "reward_type = 'continuous_scoring_function'\nlicense = 'MIT'\n"
        "license_source = 'https://github.com/owner/dataset/blob/main/LICENSE'\n"
        "[environment]\nrequired_resources = '12vcpu+100gib+h100/2+graphics'\n"
        "base_flavor = 'cuda-graphics'\n"
    )
    base = local_base_image_for_problem(tmp_path)
    assert base.image == LOCAL_CUDA_GRAPHICS_BASE_IMAGE
    assert base.dockerfile == Path("base/cuda-graphics/Dockerfile")


def test_local_runtime_auto_selects_tpu(tmp_path: Path) -> None:
    from alignerr_plugin.local_runtime import (
        LOCAL_TPU_BASE_IMAGE,
        local_base_image_for_problem,
    )

    (tmp_path / "task.toml").write_text(
        "[task]\nname = 'x'\n"
        "[difficulty]\ntask_type = 'ml'\n"
        "domain = 'scientific_discovery_computational_science'\n"
        "reward_type = 'continuous_scoring_function'\nlicense = 'MIT'\n"
        "license_source = 'https://github.com/owner/dataset/blob/main/LICENSE'\n"
        "[environment]\nrequired_resources = '13vcpu+32gib+tpuv5e1x1'\n"
    )
    base = local_base_image_for_problem(tmp_path)
    assert base.image == LOCAL_TPU_BASE_IMAGE
    assert base.dockerfile == Path("base/tpu/Dockerfile")


def test_task_dockerfiles_do_not_reinstall_grader() -> None:
    dockerfiles = [
        path
        for path in REPO_ROOT.glob("**/environment/Dockerfile")
        if ".venv" not in path.parts
    ]
    assert dockerfiles
    offenders: list[str] = []
    for dockerfile in dockerfiles:
        text = dockerfile.read_text()
        if (
            "COPY grader/ /mcp_server/grading" in text
            or "-e /mcp_server/grading" in text
        ):
            offenders.append(dockerfile.relative_to(REPO_ROOT).as_posix())
    assert offenders == []
