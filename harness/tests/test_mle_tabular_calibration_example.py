from __future__ import annotations

import json
import math
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
from _fixture_guard import requires_examples

from grading.evaluation.author import load_task_module
from grading.evaluation.context import workspace_artifact_digest
from lbx_rl_tasks_harness.calibration import validate_committed_model_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "mle-tabular-classification"

pytestmark = requires_examples("mle-tabular-classification")


# Regenerated float artifacts are compared numerically, not byte for byte.
# `train.py` fits with `np.linalg.lstsq` and `generate.py` goes through `sin`/`cos`,
# and both LAPACK and libm differ in the last few ULPs between the Linux runner's
# OpenBLAS/glibc and the Accelerate build a macOS laptop uses -- measured at 1.2e-13
# relative on the reference fit. Byte equality therefore only ever held on the
# machine that produced the committed files: it passes locally and fails in CI,
# which is the same defect class as a test that assumes no runner environment
# variables are set.
#
# The tolerance is far tighter than any real change to a fit or a generator would
# produce, so these still fail if the model or the data actually changes. Non-float
# fields are still compared exactly, so schema, seed and dtypes remain locked down.
_FLOAT_RTOL = 1e-9
_FLOAT_ATOL = 1e-12


def _assert_model_json_matches(regenerated: Path, committed: Path) -> None:
    produced = json.loads(regenerated.read_text(encoding="utf-8"))
    expected = json.loads(committed.read_text(encoding="utf-8"))

    assert produced.keys() == expected.keys()
    for key, expected_value in expected.items():
        if isinstance(expected_value, float) or (
            isinstance(expected_value, list)
            and all(isinstance(item, float) for item in expected_value)
        ):
            assert produced[key] == pytest.approx(
                expected_value, rel=_FLOAT_RTOL, abs=_FLOAT_ATOL
            ), key
        else:
            assert produced[key] == expected_value, key


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


def _package(strategy: Path, *, model_dir: Path, output_dir: Path) -> Path:
    """Run a strategy's inference entrypoint over `model_dir`, as the grader does."""

    _run(
        strategy / "solution.py",
        env={
            "LBT_DATA_DIR": str(EXAMPLE / "data"),
            "LBT_MODEL_DIR": str(model_dir),
            "LBT_OUTPUT_DIR": str(output_dir),
        },
    )
    return output_dir


def test_example_models_reproduce_committed_weights(tmp_path) -> None:
    """`train.py` still yields the committed weights, to within LAPACK noise."""

    reference_models = tmp_path / "reference-model"
    naive_models = tmp_path / "naive-model"

    _run(
        EXAMPLE / "solution" / "train.py",
        env={
            "LBT_DATA_DIR": str(EXAMPLE / "data"),
            "LBT_MODEL_DIR": str(reference_models),
        },
    )
    _run(
        EXAMPLE / "baselines" / "naive" / "train.py",
        env={
            "LBT_DATA_DIR": str(EXAMPLE / "data"),
            "LBT_MODEL_DIR": str(naive_models),
        },
    )
    _assert_model_json_matches(
        reference_models / "model.json", EXAMPLE / "solution" / "model.json"
    )
    _assert_model_json_matches(
        naive_models / "model.json", EXAMPLE / "baselines" / "naive" / "model.json"
    )


def test_committed_weights_score_to_the_calibration_lock(tmp_path, monkeypatch) -> None:
    """The lock's anchors hold for the weights the lock was calibrated against.

    Deliberately scores the *committed* `model.json`, not the copy `train.py` just
    reproduced, because that is the only thing the lock speaks about and the only
    thing production ever grades: `validate_ml_strategy_contract` rejects a strategy
    whose inference entrypoint trains, and calibration runs that entrypoint over the
    committed weights -- it never invokes `train.py`.

    Scoring a retrained copy instead is not a stricter version of this check, it is a
    different and much weaker one. See
    `test_challenge_subsample_follows_the_submitted_artifact_digest` for why: the
    grader reseeds its private challenge draw from the submitted workspace digest, so
    a retrained copy that differs only in the last few ULPs is scored on a different
    subsample and lands percent-scale away from the anchor. That is what made this
    assertion fail on Linux while passing on macOS -- the anchor was fine, the
    artifact under test was not the one it describes.
    """

    private = EXAMPLE / "scorer" / "data"
    reference_output = _package(
        EXAMPLE / "solution",
        model_dir=EXAMPLE / "solution",
        output_dir=tmp_path / "reference-output",
    )
    naive_output = _package(
        EXAMPLE / "baselines" / "naive",
        model_dir=EXAMPLE / "baselines" / "naive",
        output_dir=tmp_path / "naive-output",
    )

    module = load_task_module(EXAMPLE / "scorer" / "compute_score.py")
    task = load_task_registration(EXAMPLE / "scorer" / "compute_score.py")
    assert task is not None
    lock_path = EXAMPLE / "calibration.lock.json"
    lock = load_calibration_lock(lock_path, task_spec_sha256=task.spec_sha256)
    monkeypatch.setenv("LBX_CALIBRATION_LOCK_PATH", str(lock_path))

    reference_metrics = measure_task_module(
        module, workspace=reference_output, private=private
    )
    naive_metrics = measure_task_module(module, workspace=naive_output, private=private)

    # The measured reference must land on the lock's own reference measurement, not
    # merely somewhere that happens to score 0.5.
    assert reference_metrics == pytest.approx(lock.reference_metrics)
    assert task.score(reference_metrics, calibration_only=True)[
        "score"
    ] == pytest.approx(0.5)
    naive_score = task.score(naive_metrics, calibration_only=True)["score"]
    assert 0.0 < naive_score <= 0.10
    assert lock.payload["qualification"]["oracle_score"] == 1.0


def test_challenge_subsample_follows_the_submitted_artifact_digest(
    tmp_path, monkeypatch
) -> None:
    """Pin the amplifier that turns ULP noise in the weights into percent of score.

    `ContinuousTask._load_model_challenge` seeds the private challenge draw from
    `workspace_artifact_digest(workspace)`, so changing the submitted bytes at all
    re-rolls which 400 of the 1000 challenge rows are scored. That is deliberate --
    it stops an agent precomputing answers for a fixed subsample -- but it means the
    committed weights are the only artifact for which the lock's anchors hold.

    This is the negative control for
    `test_committed_weights_score_to_the_calibration_lock`. If someone rewrites that
    test to score a retrained model again, this one explains why the resulting
    failure is not a calibration bug.
    """

    committed = json.loads(
        (EXAMPLE / "solution" / "model.json").read_text(encoding="utf-8")
    )
    # One ULP up on a single coefficient: a relative change of 2.2e-16, far below
    # anything that can move a metric arithmetically.
    perturbed = json.loads(json.dumps(committed))
    original = perturbed["t2_coef"][-1]
    perturbed["t2_coef"][-1] = math.nextafter(original, math.inf)
    assert perturbed["t2_coef"][-1] == pytest.approx(original, rel=1e-15)
    assert perturbed["t2_coef"][-1] != original

    model_dir = tmp_path / "perturbed-model"
    model_dir.mkdir()
    (model_dir / "model.json").write_text(
        json.dumps(perturbed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    baseline_output = _package(
        EXAMPLE / "solution",
        model_dir=EXAMPLE / "solution",
        output_dir=tmp_path / "baseline-output",
    )
    perturbed_output = _package(
        EXAMPLE / "solution",
        model_dir=model_dir,
        output_dir=tmp_path / "perturbed-output",
    )
    assert workspace_artifact_digest(baseline_output) != workspace_artifact_digest(
        perturbed_output
    )

    module = load_task_module(EXAMPLE / "scorer" / "compute_score.py")
    task = load_task_registration(EXAMPLE / "scorer" / "compute_score.py")
    assert task is not None
    monkeypatch.setenv(
        "LBX_CALIBRATION_LOCK_PATH", str(EXAMPLE / "calibration.lock.json")
    )
    private = EXAMPLE / "scorer" / "data"
    baseline_metrics = measure_task_module(
        module, workspace=baseline_output, private=private
    )
    perturbed_metrics = measure_task_module(
        module, workspace=perturbed_output, private=private
    )

    # A different subsample, so the metrics move by orders of magnitude more than the
    # 2.2e-16 change to the weights can account for.
    for name in ("t1", "t2"):
        moved = abs(perturbed_metrics[name] - baseline_metrics[name])
        assert moved > 1e-3 * abs(baseline_metrics[name]), name

    baseline_score = task.score(baseline_metrics, calibration_only=True)["score"]
    perturbed_score = task.score(perturbed_metrics, calibration_only=True)["score"]
    assert baseline_score == pytest.approx(0.5)
    assert abs(perturbed_score - 0.5) > 5e-3


def test_example_generator_reproduces_committed_data(tmp_path) -> None:
    isolated = tmp_path / "generated"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "data_generation" / "generate.py"),
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
        Path("data/train.parquet"),
        Path("data/test.parquet"),
        Path("scorer/data/test_target.parquet"),
    ):
        pd.testing.assert_frame_equal(
            pd.read_parquet(isolated / relative),
            pd.read_parquet(EXAMPLE / relative),
            check_exact=False,
            rtol=_FLOAT_RTOL,
            atol=_FLOAT_ATOL,
            obj=str(relative),
        )


def test_private_challenge_requires_uncommitted_trusted_seed() -> None:
    env = dict(os.environ)
    env.pop("LBX_PRIVATE_CHALLENGE_SEED", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "data_generation" / "generate_private_challenge.py"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "LBX_PRIVATE_CHALLENGE_SEED is required" in completed.stderr
    challenge = pd.read_parquet(EXAMPLE / "scorer" / "data" / "challenge.parquet")
    assert len(challenge) == 1000
    assert set(challenge) == {"x1", "x2", "x3", "t1", "t2", "label"}


def test_example_has_no_committed_score_copies() -> None:
    for strategy in (
        EXAMPLE / "solution",
        EXAMPLE / "baselines" / "naive",
        EXAMPLE / "baselines" / "linear",
    ):
        assert not (strategy / "results.txt").exists()
        assert not (strategy / "submission.csv").exists()
    validate_committed_model_manifest(EXAMPLE / "solution", role="reference")
    validate_committed_model_manifest(EXAMPLE / "baselines" / "naive", role="naive")
