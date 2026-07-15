from __future__ import annotations

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
