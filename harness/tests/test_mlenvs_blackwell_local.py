"""Local Blackwell auto-detection: swap the cu121 compute base for the cu128
blackwell overlay (local builds only), with the parent-chain build recursing."""

from __future__ import annotations

import json
from pathlib import Path

from alignerr_plugin import local_runtime
from alignerr_plugin.local_runtime import (
    LocalBaseImage,
    _ensure_parent_base_image,
    local_base_image_for_problem,
    local_gpu_is_blackwell,
)


def _write_mlenvs_task(root: Path, *, required_resources: str) -> Path:
    d = root / "task_taiga"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "ml_task_type": "dataset",
                "required_resources": required_resources,
                "domain": "scientific_discovery_computational_science",
                "license": "CC0-1.0",
                "license_source": "https://x",
            }
        )
    )
    (d / "prompt.md").write_text("Write /tmp/output/submission.csv.\n")
    (d / "test_file.py").write_text("def compute_score():\n    return 0.0\n")
    return d


def test_gpu_is_blackwell_env_override(monkeypatch):
    monkeypatch.setenv("LBX_RL_TASKS_LOCAL_BLACKWELL", "1")
    assert local_gpu_is_blackwell() is True
    monkeypatch.setenv("LBX_RL_TASKS_LOCAL_BLACKWELL", "0")
    assert local_gpu_is_blackwell() is False


def test_compute_task_swaps_to_gpu_blackwell(tmp_path, monkeypatch):
    monkeypatch.setenv("LBX_RL_TASKS_LOCAL_BLACKWELL", "1")
    task = _write_mlenvs_task(tmp_path, required_resources="12vcpu+100gib+h100/2")
    base = local_base_image_for_problem(task)
    assert base.image == "lbx-tasks-base-mlenvs-gpu-blackwell"
    assert base.dockerfile == Path("base/mlenvs-gpu-blackwell/Dockerfile")


def test_graphics_task_swaps_to_cuda_graphics_blackwell(tmp_path, monkeypatch):
    monkeypatch.setenv("LBX_RL_TASKS_LOCAL_BLACKWELL", "1")
    task = _write_mlenvs_task(tmp_path, required_resources="12vcpu+100gib+h100/2+graphics")
    base = local_base_image_for_problem(task)
    assert base.image == "lbx-tasks-base-mlenvs-cuda-graphics-blackwell"


def test_no_swap_when_blackwell_off(tmp_path, monkeypatch):
    monkeypatch.setenv("LBX_RL_TASKS_LOCAL_BLACKWELL", "0")
    task = _write_mlenvs_task(tmp_path, required_resources="12vcpu+100gib+h100/2")
    base = local_base_image_for_problem(task)
    assert base.image == "lbx-tasks-base-mlenvs-gpu"  # cu121 base, no overlay


def test_tpu_task_never_swaps(tmp_path, monkeypatch):
    monkeypatch.setenv("LBX_RL_TASKS_LOCAL_BLACKWELL", "1")
    task = _write_mlenvs_task(tmp_path, required_resources="13vcpu+32gib+tpuv5e1x1")
    base = local_base_image_for_problem(task)
    assert base.image == "lbx-tasks-base-mlenvs-tpu"


def test_parent_chain_builds_bottom_up(monkeypatch):
    # mlenvs-cuda-graphics-blackwell -> mlenvs-cuda-graphics -> mlenvs-gpu.
    built: list[str] = []
    monkeypatch.setattr(local_runtime, "_docker_image_exists", lambda ref: False)
    monkeypatch.setattr(
        local_runtime, "_build_base_image", lambda repo, base: built.append(base.image)
    )
    target = LocalBaseImage(
        image="lbx-tasks-base-mlenvs-cuda-graphics-blackwell",
        tag=local_runtime.LOCAL_BASE_TAG,
        dockerfile=Path("base/mlenvs-cuda-graphics-blackwell/Dockerfile"),
    )
    _ensure_parent_base_image(Path("/repo"), target)
    # Grandparent first, then parent; the target itself is built by the caller.
    assert built == [
        "lbx-tasks-base-mlenvs-gpu",
        "lbx-tasks-base-mlenvs-cuda-graphics",
    ]
