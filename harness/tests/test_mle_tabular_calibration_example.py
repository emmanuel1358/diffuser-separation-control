from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from grading.evaluation import (
    load_calibration_lock,
    load_task_registration,
    measure_task_module,
)
from grading.evaluation.author import load_task_module
from lbx_rl_tasks_harness.calibration import validate_committed_model_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "mle-tabular-classification"


def _run(script: Path, *, env: dict[str, str]) -> None:
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_example_models_reproduce_and_score(tmp_path, monkeypatch) -> None:
    data_dir = EXAMPLE / "data" / "public"
    private = EXAMPLE / "data" / "private"
    reference_models = tmp_path / "reference-model"
    naive_models = tmp_path / "naive-model"
    reference_output = tmp_path / "reference-output"
    naive_output = tmp_path / "naive-output"

    _run(
        EXAMPLE / "reference_solution" / "train.py",
        env={
            "LBT_DATA_DIR": str(data_dir),
            "LBT_MODEL_DIR": str(reference_models),
        },
    )
    _run(
        EXAMPLE / "baselines" / "naive" / "train.py",
        env={
            "LBT_DATA_DIR": str(data_dir),
            "LBT_MODEL_DIR": str(naive_models),
        },
    )
    assert (reference_models / "model.json").read_bytes() == (
        EXAMPLE / "reference_solution" / "model.json"
    ).read_bytes()
    assert (naive_models / "model.json").read_bytes() == (
        EXAMPLE / "baselines" / "naive" / "model.json"
    ).read_bytes()

    _run(
        EXAMPLE / "reference_solution" / "solution.py",
        env={
            "LBT_DATA_DIR": str(data_dir),
            "LBT_MODEL_DIR": str(reference_models),
            "LBT_OUTPUT_DIR": str(reference_output),
        },
    )
    _run(
        EXAMPLE / "baselines" / "naive" / "solution.py",
        env={
            "LBT_DATA_DIR": str(data_dir),
            "LBT_MODEL_DIR": str(naive_models),
            "LBT_OUTPUT_DIR": str(naive_output),
        },
    )

    module = load_task_module(EXAMPLE / "test_file.py")
    task = load_task_registration(EXAMPLE / "test_file.py")
    assert task is not None
    lock_path = EXAMPLE / "calibration.lock.json"
    lock = load_calibration_lock(lock_path, task_spec_sha256=task.spec_sha256)
    monkeypatch.setenv("LBX_CALIBRATION_LOCK_PATH", str(lock_path))

    reference_metrics = measure_task_module(
        module, workspace=reference_output, private=private
    )
    naive_metrics = measure_task_module(module, workspace=naive_output, private=private)
    assert task.score(reference_metrics, calibration_only=True)[
        "score"
    ] == pytest.approx(0.5)
    naive_score = task.score(naive_metrics, calibration_only=True)["score"]
    assert 0.0 < naive_score <= 0.10
    assert lock.payload["qualification"]["oracle_score"] == 1.0


def test_example_generator_reproduces_committed_data(tmp_path) -> None:
    isolated = tmp_path / "generated"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "data-generation" / "generate.py"),
            "--output-root",
            str(isolated),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for relative in (
        Path("data/public/train.parquet"),
        Path("data/public/test.parquet"),
        Path("data/private/test_target.parquet"),
    ):
        assert pd.read_parquet(EXAMPLE / relative).equals(
            pd.read_parquet(isolated / relative)
        )


def test_private_challenge_requires_uncommitted_trusted_seed() -> None:
    env = dict(os.environ)
    env.pop("LBX_PRIVATE_CHALLENGE_SEED", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "data-generation" / "generate_private_challenge.py"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "LBX_PRIVATE_CHALLENGE_SEED is required" in completed.stderr
    challenge = pd.read_parquet(EXAMPLE / "data" / "private" / "challenge.parquet")
    assert len(challenge) == 1000
    assert set(challenge) == {"x1", "x2", "x3", "t1", "t2", "label"}


def test_example_has_no_committed_score_copies() -> None:
    for strategy in (
        EXAMPLE / "reference_solution",
        EXAMPLE / "baselines" / "naive",
        EXAMPLE / "baselines" / "linear",
    ):
        assert not (strategy / "results.txt").exists()
        assert not (strategy / "submission.csv").exists()
    validate_committed_model_manifest(EXAMPLE / "reference_solution", role="reference")
    validate_committed_model_manifest(EXAMPLE / "baselines" / "naive", role="naive")
