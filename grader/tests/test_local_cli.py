from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from alignerr_plugin.local_cli import app
from alignerr_plugin.utils import load_task_toml


def test_export_taiga_cli_writes_metadata_and_sidecar(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mujoco-pendulum"
    shutil.copytree(template_examples / "mujoco-pendulum", problem_dir)
    output = tmp_path / "problems-metadata.json"

    result = CliRunner().invoke(
        app,
        [
            "export-taiga",
            "--problem-dir",
            str(problem_dir),
            "--out",
            str(output),
            "--image",
            "local:test",
        ],
    )

    assert result.exit_code == 0, result.output
    metadata = json.loads(output.read_text())
    assert metadata["problem_set"]["problems"][0]["id"] == "mujoco-pendulum"

    sidecar = json.loads((problem_dir / ".taiga_submit.json").read_text())
    assert sidecar["task_id"] == "mujoco-pendulum"
    assert sidecar["image"] == "local:test"


def test_validate_cli_exits_nonzero_for_invalid_task(tmp_path: Path) -> None:
    problem_dir = tmp_path / "broken-task"
    problem_dir.mkdir()

    result = CliRunner().invoke(
        app,
        ["validate", "--problem-dir", str(problem_dir)],
    )

    assert result.exit_code == 1


def test_new_cli_scaffolds_prometheus_template(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "new",
            "--name",
            "labelbox/prometheus-smoke",
            "--template",
            "prometheus",
            "--out",
            str(tmp_path),
        ],
    )

    problem_dir = tmp_path / "prometheus-smoke"
    assert result.exit_code == 0, result.output
    assert (problem_dir / "task.toml").exists()
    assert load_task_toml(problem_dir).delivery.platform == "prometheus"


def test_new_cli_scaffolds_numerical_solver_prometheus_templates(tmp_path: Path) -> None:
    for template_name, task_type, is_eval in (
        ("prometheus-cfd", "cfd", False),
        ("prometheus-structures", "structures", False),
        ("prometheus-eval-cfd", "cfd", True),
        ("prometheus-eval-structures", "structures", True),
    ):
        result = CliRunner().invoke(
            app,
            [
                "new",
                "--name",
                f"labelbox/{template_name}-smoke",
                "--template",
                template_name,
                "--out",
                str(tmp_path),
            ],
        )

        problem_dir = tmp_path / f"{template_name}-smoke"
        assert result.exit_code == 0, result.output
        task = load_task_toml(problem_dir)
        assert task.delivery.platform == "prometheus"
        assert task.delivery.eval is is_eval
        assert task.difficulty.task_type == task_type


def test_export_harbor_cli_writes_harbor_task(
    template_examples: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "harbor"

    result = CliRunner().invoke(
        app,
        [
            "export-harbor",
            "--problem-dir",
            str(template_examples / "mujoco-pendulum"),
            "--out",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "task.toml").exists()
    assert (output_dir / "environment" / "Dockerfile").exists()
