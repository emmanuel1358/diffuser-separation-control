from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from alignerr_plugin.local_cli import app
from alignerr_plugin.utils import load_task_toml
from alignerr_plugin.validators.task.creator import TaskCreator
from grading.evaluation.plan import check_evaluation_plan
from typer.testing import CliRunner


def test_creator_replaces_software_engineering_identity(tmp_path: Path) -> None:
    problem = TaskCreator().create_structure(
        tmp_path,
        {
            "name": "example-org/retry-parser",
            "template": "software-engineering",
        },
    )

    assert problem == tmp_path / "retry-parser"
    task = load_task_toml(problem)
    metadata = json.loads((problem / "metadata.json").read_text(encoding="utf-8"))

    assert task.task.name == "example-org/retry-parser"
    assert metadata["problem_data"]["instance_id"] == "retry-parser"
    assert task.difficulty.task_type == "software_engineering"
    assert task.difficulty.domain == "repo_debugging"
    assert [output.path for output in task.outputs] == ["/tmp/output/repo"]
    assert (problem / "starter" / "normalizer.py").is_file()
    assert (problem / "scorer" / "data" / "candidate_driver.py").is_file()
    assert check_evaluation_plan(problem).status == "unchanged"


def test_create_cli_scaffolds_software_engineering_starter(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "create",
            "--name",
            "labelbox/cli-repo-debugging",
            "-t",
            "software-engineering",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    problem = tmp_path / "cli-repo-debugging"
    assert load_task_toml(problem).task.name == "labelbox/cli-repo-debugging"
    assert (problem / "solution" / "solve.sh").is_file()
    assert (problem / "baselines" / "noop.sh").is_file()
    assert (problem / "scorer" / "evaluation.plan.json").is_file()


def test_software_engineering_starter_host_grades_oracle_and_noop(
    tmp_path: Path,
) -> None:
    problem = TaskCreator().create_structure(
        tmp_path,
        {
            "name": "labelbox/host-grade-smoke",
            "template": "software-engineering",
        },
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PROBLEM_DIR": str(problem),
            "PYTHON_BIN": sys.executable,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    completed = subprocess.run(
        ["bash", str(problem / "tests" / "test.sh")],
        cwd=problem,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "oracle host grade: 1.0" in completed.stdout
    assert "no-op host grade: 0.0" in completed.stdout
