"""The slim base flavor is wired end-to-end: flavor registry, resolver, local +
Taiga image maps, and that the Dockerfile exists."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignerr_plugin import base_image, local_runtime
from alignerr_plugin.base_image import (
    BASE_FLAVOR_CHOICES,
    BASE_FLAVORS,
    resolve_base_flavor_for_resource,
)
from alignerr_plugin.exporters.taiga import MLENVS_SLIM_BASE_IMAGE, _base_image_and_tag

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_slim_registered_in_base_flavors():
    slim = BASE_FLAVORS["mlenvs-slim"]
    assert slim.image_suffix == "-mlenvs-slim"
    assert slim.dockerfile == "base/mlenvs-slim/Dockerfile"
    assert slim.tag_prefix == base_image.MLENVS_SLIM_BASE_TAG_PREFIX == "runtime-mlenvs-slim-py312"
    assert "mlenvs-slim" in BASE_FLAVOR_CHOICES
    assert "mlenvs-slim" in base_image.MLENVS_FLAVORS


def test_slim_dockerfile_exists():
    assert (REPO_ROOT / "base" / "mlenvs-slim" / "Dockerfile").is_file()


def test_explicit_slim_resolves_on_any_tier():
    # slim carries no accelerator stack; it is selectable on any tier.
    for tier in ("2vcpu+6gib", "12vcpu+100gib+h100/2", "13vcpu+32gib+tpuv5e1x1"):
        assert resolve_base_flavor_for_resource("mlenvs-slim", tier) == "mlenvs-slim"


def test_slim_local_and_registry_image_maps():
    image, dockerfile = local_runtime._LOCAL_BASE_BY_FLAVOR["mlenvs-slim"]
    assert image == local_runtime.LOCAL_MLENVS_SLIM_BASE_IMAGE == "lbx-tasks-base-mlenvs-slim"
    assert dockerfile == "base/mlenvs-slim/Dockerfile"

    reg_image, tag = _base_image_and_tag("mlenvs-slim")
    assert reg_image == MLENVS_SLIM_BASE_IMAGE
    assert tag.startswith("runtime-mlenvs-slim-py312")


def test_auto_resolution_unaffected_by_slim():
    # Adding slim/mlenvs flavors must not change auto inference for the real
    # Taiga tiers (native verticals are untouched).
    assert resolve_base_flavor_for_resource("auto", "12vcpu+100gib+h100/2") == "gpu"
    assert resolve_base_flavor_for_resource("auto", "13vcpu+32gib+tpuv5e1x1") == "tpu"
    assert resolve_base_flavor_for_resource("auto", "2vcpu+6gib") == "cpu"
    assert (
        resolve_base_flavor_for_resource("auto", "12vcpu+100gib+h100/2+graphics")
        == "cuda-graphics"
    )


# --- ML_Envs-specific flavors ---------------------------------------------


def test_mlenvs_flavors_registered_and_dockerfiles_exist():
    for flavor in ("mlenvs-gpu", "mlenvs-cuda-graphics", "mlenvs-tpu"):
        assert flavor in BASE_FLAVORS
        assert flavor in BASE_FLAVOR_CHOICES
        df = REPO_ROOT / BASE_FLAVORS[flavor].dockerfile
        assert df.is_file(), df
        assert BASE_FLAVORS[flavor].tag_prefix == "runtime-mlenvs-py312"


def test_mlenvs_flavor_parents():
    # cuda-graphics parent-chains on the compute base; tpu and gpu stand alone.
    assert BASE_FLAVORS["mlenvs-cuda-graphics"].parent == "mlenvs-gpu"
    assert BASE_FLAVORS["mlenvs-gpu"].parent is None
    assert BASE_FLAVORS["mlenvs-tpu"].parent is None
    # expand_base_flavors inserts the compute parent before its child.
    assert base_image.expand_base_flavors(["mlenvs-cuda-graphics"]) == [
        "mlenvs-gpu",
        "mlenvs-cuda-graphics",
    ]


def test_mlenvs_blackwell_flavors_registered_with_parents_and_dockerfiles():
    for flavor, suffix, parent in (
        ("mlenvs-gpu-blackwell", "-mlenvs-gpu-blackwell", "mlenvs-gpu"),
        (
            "mlenvs-cuda-graphics-blackwell",
            "-mlenvs-cuda-graphics-blackwell",
            "mlenvs-cuda-graphics",
        ),
    ):
        assert flavor in BASE_FLAVORS
        assert flavor in BASE_FLAVOR_CHOICES
        assert flavor in base_image.MLENVS_FLAVORS
        f = BASE_FLAVORS[flavor]
        assert f.image_suffix == suffix
        assert f.parent == parent
        assert f.tag_prefix == base_image.MLENVS_BLACKWELL_BASE_TAG_PREFIX
        assert f.tag_prefix == "runtime-mlenvs-blackwell-py312"
        assert (REPO_ROOT / f.dockerfile).is_file(), f.dockerfile
    # Non-blackwell mlenvs flavors keep the plain py312 prefix.
    assert BASE_FLAVORS["mlenvs-gpu"].tag_prefix == "runtime-mlenvs-py312"


def test_mlenvs_blackwell_flavors_are_local_only_on_any_taiga_tier():
    # The blackwell overlays are cu128 local-dev builds, invalid for any Taiga tier.
    for flavor in ("mlenvs-gpu-blackwell", "mlenvs-cuda-graphics-blackwell"):
        for tier in (
            "2vcpu+6gib",
            "12vcpu+100gib+h100/2",
            "12vcpu+100gib+h100/2+graphics",
            "13vcpu+32gib+tpuv5e1x1",
        ):
            with pytest.raises(ValueError, match="local-only"):
                resolve_base_flavor_for_resource(flavor, tier)


def test_mlenvs_blackwell_local_and_registry_image_maps():
    from alignerr_plugin.exporters import taiga as taiga_exporter

    for flavor, suffix in (
        ("mlenvs-gpu-blackwell", "-mlenvs-gpu-blackwell"),
        ("mlenvs-cuda-graphics-blackwell", "-mlenvs-cuda-graphics-blackwell"),
    ):
        image, dockerfile = local_runtime._LOCAL_BASE_BY_FLAVOR[flavor]
        assert image == f"lbx-tasks-base{suffix}"
        assert dockerfile == BASE_FLAVORS[flavor].dockerfile
        reg_image, tag = taiga_exporter._base_image_and_tag(flavor)
        assert reg_image.endswith(f"lbx-tasks-base{suffix}")
        assert tag.startswith("runtime-mlenvs-blackwell-py312")


def test_mlenvs_flavor_tier_compatibility():
    # mlenvs-gpu: any non-graphics, non-TPU tier (h100 or cpu).
    assert resolve_base_flavor_for_resource("mlenvs-gpu", "12vcpu+100gib+h100/2") == "mlenvs-gpu"
    assert resolve_base_flavor_for_resource("mlenvs-gpu", "2vcpu+6gib") == "mlenvs-gpu"
    for bad in ("13vcpu+32gib+tpuv5e1x1", "12vcpu+100gib+h100/2+graphics"):
        with pytest.raises(ValueError):
            resolve_base_flavor_for_resource("mlenvs-gpu", bad)
    # mlenvs-cuda-graphics requires a +graphics tier.
    assert (
        resolve_base_flavor_for_resource("mlenvs-cuda-graphics", "12vcpu+100gib+h100/2+graphics")
        == "mlenvs-cuda-graphics"
    )
    with pytest.raises(ValueError):
        resolve_base_flavor_for_resource("mlenvs-cuda-graphics", "12vcpu+100gib+h100/2")
    # mlenvs-tpu requires a TPU tier.
    assert resolve_base_flavor_for_resource("mlenvs-tpu", "13vcpu+32gib+tpuv5e1x1") == "mlenvs-tpu"
    with pytest.raises(ValueError):
        resolve_base_flavor_for_resource("mlenvs-tpu", "12vcpu+100gib+h100/2")


def test_mlenvs_flavor_registry_and_local_maps():
    from alignerr_plugin.exporters import taiga as taiga_exporter

    for flavor, suffix in (
        ("mlenvs-gpu", "-mlenvs-gpu"),
        ("mlenvs-cuda-graphics", "-mlenvs-cuda-graphics"),
        ("mlenvs-tpu", "-mlenvs-tpu"),
    ):
        image, _dockerfile = local_runtime._LOCAL_BASE_BY_FLAVOR[flavor]
        assert image == f"lbx-tasks-base{suffix}"
        reg_image, tag = taiga_exporter._base_image_and_tag(flavor)
        assert reg_image.endswith(f"lbx-tasks-base{suffix}")
        assert tag.startswith("runtime-mlenvs-py312")


# --- startup_command: --system rubric for ML_Envs, venv path for native -----


def _write_mlenvs_task_dir(root: Path) -> Path:
    """A minimal metadata-mode task dir (metadata.json + markers, no task.toml)."""
    task_dir = root / "demo-mlenvs_taiga"
    (task_dir / "data" / "public").mkdir(parents=True)
    (task_dir / "data" / "private").mkdir(parents=True)
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "ml_task_type": "dataset",
                "required_resources": "12vcpu+100gib+h100/2",
                "domain": "scientific_discovery_computational_science",
                "license": "CC0-1.0",
                "license_source": "https://creativecommons.org/publicdomain/zero/1.0/",
            }
        )
    )
    (task_dir / "prompt.md").write_text(
        "# Demo\n\nWrite predictions to `/tmp/output/submission.csv`.\n"
    )
    (task_dir / "test_file.py").write_text(
        "def compute_score():\n    return 0.5\n"
    )
    (task_dir / "data" / "public" / "train.csv").write_text("x,y\n1,2\n")
    (task_dir / "data" / "private" / "truth.csv").write_text("y\n2\n")
    return task_dir


def _write_native_task_dir(root: Path) -> Path:
    """A minimal native task dir (ships task.toml -> NOT ML_Envs mode)."""
    task_dir = root / "demo-native"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        "[task]\nname = 'x'\n"
        "[difficulty]\ntask_type = 'ml'\n"
        "domain = 'scientific_discovery_computational_science'\n"
        "reward_type = 'continuous_scoring_function'\nlicense = 'MIT'\n"
        "license_source = 'https://github.com/owner/dataset/blob/main/LICENSE'\n"
        "[environment]\nrequired_resources = '12vcpu+100gib+h100/2'\n"
    )
    # Native tasks ship the metadata.json envelope alongside task.toml.
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "taiga_task",
                "problem_data": {"instance_id": "demo-native", "description": ""},
            }
        )
    )
    (task_dir / "instruction.md").write_text(
        "Write /tmp/output/submission.csv.\n"
    )
    return task_dir


def test_taiga_startup_command_mlenvs_vs_native(tmp_path: Path):
    from alignerr_plugin import mlenvs
    from alignerr_plugin.exporters.taiga import (
        MLENVS_STARTUP_COMMAND,
        STARTUP_COMMAND,
        build_job_payload,
    )

    mlenvs_dir = _write_mlenvs_task_dir(tmp_path)
    native_dir = _write_native_task_dir(tmp_path)
    assert mlenvs.is_mlenvs_task(mlenvs_dir)
    assert not mlenvs.is_mlenvs_task(native_dir)

    # metadata-mode base has no venv -> console script on PATH.
    assert MLENVS_STARTUP_COMMAND == "rubric mcp"
    assert STARTUP_COMMAND == "/opt/lbx-runtime/.venv/bin/rubric mcp"

    mlenvs_problem = build_job_payload(
        mlenvs_dir, image_ref="gcr.io/example/img@sha256:abc"
    )["problems_metadata"]["problem_set"]["problems"][0]
    assert mlenvs_problem["startup_command"] == "rubric mcp"

    native_problem = build_job_payload(
        native_dir, image_ref="gcr.io/example/img@sha256:abc"
    )["problems_metadata"]["problem_set"]["problems"][0]
    assert native_problem["startup_command"] == "/opt/lbx-runtime/.venv/bin/rubric mcp"
