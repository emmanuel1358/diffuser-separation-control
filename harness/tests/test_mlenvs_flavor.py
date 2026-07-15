"""The ``--flavor`` compute-base mechanism: auto/heavy/slim selection, build-time
OOM detection, the heavy->slim auto-fallback, the slim-only-for-compute guard,
and TaskBuild + manifest telemetry. Docker is mocked throughout."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from alignerr_plugin import local_runtime as lr
from lbx_rl_tasks_harness import docker


def _write_mlenvs_task(root: Path, *, required_resources: str, name: str = "task_taiga") -> Path:
    d = root / name
    d.mkdir(parents=True)
    d.joinpath("metadata.json").write_text(
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
    d.joinpath("prompt.md").write_text("Write /tmp/output/submission.csv\n")
    d.joinpath("test_file.py").write_text("def compute_score():\n    return 0.0\n")
    return d


def _native_task(root: Path) -> Path:
    d = root / "native"
    d.mkdir(parents=True)
    d.joinpath("task.toml").write_text(
        "[task]\nname = 'labelbox/native'\n"
        "[environment]\nrequired_resources = '12vcpu+100gib+h100/2'\n"
        "[difficulty]\ntask_type = 'ml'\n"
    )
    return d


# --- constants + OOM detection --------------------------------------------


def test_valid_flavors_and_bonus():
    assert lr.VALID_FLAVORS == ("auto", "heavy", "slim")
    assert isinstance(lr.SLIM_FALLBACK_TURN_BONUS, int) and lr.SLIM_FALLBACK_TURN_BONUS > 0


@pytest.mark.parametrize(
    "output, rc, expected",
    [
        # Bare exit 137 with no OOM signature is NOT OOM (also a kill/timeout/Ctrl-C).
        ("", 137, False),
        # A genuine build-step OOM leaves a signature, matched regardless of exit code.
        ("some layer\n exit code: 137\n", 1, True),
        ("some layer\n exit code: 137\n", 137, True),
        ("cc1plus: out of memory allocating 64 bytes", 1, True),
        ("gcc: internal compiler error: Killed (program cc1plus)", 1, True),
        ("virtual memory exhausted: Cannot allocate memory", 1, True),
        ("ERROR: pip could not find a version", 1, False),
        ("generic build failure", 2, False),
    ],
)
def test_is_build_oom(output, rc, expected):
    assert lr._is_build_oom(output, rc) is expected


# --- slim availability -----------------------------------------------------


def test_slim_available_for_compute_task(tmp_path):
    task = _write_mlenvs_task(tmp_path, required_resources="12vcpu+100gib+h100/2")
    assert lr._slim_available(task) is True


def test_slim_unavailable_for_graphics_task(tmp_path):
    task = _write_mlenvs_task(tmp_path, required_resources="12vcpu+100gib+h100/2+graphics")
    assert lr._slim_available(task) is False


def test_slim_unavailable_for_tpu_task(tmp_path):
    task = _write_mlenvs_task(tmp_path, required_resources="13vcpu+32gib+tpuv5e1x1")
    assert lr._slim_available(task) is False


def test_slim_unavailable_for_native_task(tmp_path):
    assert lr._slim_available(_native_task(tmp_path)) is False


# --- ensure_local_base_image: flavor resolution + fallback ----------------


def _mock_docker(monkeypatch, *, exists=False, build_side_effect=None):
    monkeypatch.setattr(lr.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(lr, "_docker_image_exists", lambda _ref: exists)
    monkeypatch.setattr(lr, "_ensure_parent_base_image", lambda *_: None)
    monkeypatch.setattr(
        lr, "_build_base_image", mock.Mock(side_effect=build_side_effect)
    )


def test_auto_falls_back_to_slim_on_build_oom(tmp_path, monkeypatch):
    task = _write_mlenvs_task(tmp_path, required_resources="12vcpu+100gib+h100/2")
    _mock_docker(monkeypatch, build_side_effect=[lr.BuildOOMError("oom"), None])
    base = lr.ensure_local_base_image(Path("/repo"), task, flavor="auto")
    assert base.image == lr.LOCAL_MLENVS_SLIM_BASE_IMAGE
    assert base.flavor == "slim" and base.flavor_fallback is True


def test_auto_keeps_heavy_when_build_succeeds(tmp_path, monkeypatch):
    task = _write_mlenvs_task(tmp_path, required_resources="12vcpu+100gib+h100/2")
    _mock_docker(monkeypatch, build_side_effect=None)
    base = lr.ensure_local_base_image(Path("/repo"), task, flavor="auto")
    assert base.flavor == "heavy" and base.flavor_fallback is False
    assert base.image != lr.LOCAL_MLENVS_SLIM_BASE_IMAGE


def test_explicit_slim_on_compute_task(tmp_path, monkeypatch):
    task = _write_mlenvs_task(tmp_path, required_resources="12vcpu+100gib+h100/2")
    _mock_docker(monkeypatch)
    base = lr.ensure_local_base_image(Path("/repo"), task, flavor="slim")
    assert base.image == lr.LOCAL_MLENVS_SLIM_BASE_IMAGE and base.flavor == "slim"


def test_explicit_slim_rejected_for_graphics(tmp_path, monkeypatch):
    task = _write_mlenvs_task(tmp_path, required_resources="12vcpu+100gib+h100/2+graphics")
    _mock_docker(monkeypatch)
    with pytest.raises(RuntimeError, match="slim is only available"):
        lr.ensure_local_base_image(Path("/repo"), task, flavor="slim")


def test_auto_oom_reraises_when_slim_not_viable(tmp_path, monkeypatch):
    task = _write_mlenvs_task(tmp_path, required_resources="12vcpu+100gib+h100/2+graphics")
    _mock_docker(monkeypatch, build_side_effect=lr.BuildOOMError("oom"))
    with pytest.raises(lr.BuildOOMError):
        lr.ensure_local_base_image(Path("/repo"), task, flavor="auto")


def test_invalid_flavor_rejected(tmp_path, monkeypatch):
    task = _write_mlenvs_task(tmp_path, required_resources="12vcpu+100gib+h100/2")
    monkeypatch.setattr(lr.shutil, "which", lambda _: "/usr/bin/docker")
    with pytest.raises(ValueError, match="--flavor must be one of"):
        lr.ensure_local_base_image(Path("/repo"), task, flavor="medium")


# --- TaskBuild + flavor sidecar -------------------------------------------


def test_flavor_sidecar_roundtrip(tmp_path):
    problem_dir = tmp_path / "task_taiga"
    problem_dir.mkdir()
    build = docker.TaskBuild(
        image_tag="img:1", image_flavor="slim", flavor_requested="auto", flavor_fallback=True
    )
    docker._write_flavor_sidecar(problem_dir, build)
    recorded = docker.read_flavor_sidecar(problem_dir)
    assert recorded == {
        "image_flavor": "slim",
        "flavor_requested": "auto",
        "flavor_fallback": True,
    }


def test_read_flavor_sidecar_missing_returns_none(tmp_path):
    assert docker.read_flavor_sidecar(tmp_path) is None


# --- CLI wiring ------------------------------------------------------------


def test_run_command_advertises_flavor():
    from typer.testing import CliRunner

    from lbx_rl_tasks_harness.cli import app

    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--flavor" in result.output
