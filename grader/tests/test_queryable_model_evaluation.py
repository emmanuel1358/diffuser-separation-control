from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grading.evaluation import (
    AnchorRationale,
    ContinuousTask,
    FloorAnchor,
    GeneratedCalibration,
    IIDPermutationEvidence,
    PopulationSRETarget,
    PrivateTableChallenge,
    PythonPredictor,
    write_calibration_lock_atomic,
)
from grading.faults import AgentFault


def _task() -> ContinuousTask:
    return ContinuousTask.model(
        artifact=PythonPredictor(),
        challenge=PrivateTableChallenge(
            "challenge.parquet",
            feature_columns=["x"],
            sample_size=120,
        ),
        targets=[
            PopulationSRETarget.lower(
                "target",
                weight=1.0,
                floor=FloorAnchor(
                    1.0,
                    AnchorRationale(
                        kind="theoretical",
                        summary="A population-mean constant has standardized RMSE one.",
                    ),
                ),
            )
        ],
        calibration=GeneratedCalibration(),
        evidence=IIDPermutationEvidence(
            family_alpha=0.01,
            permutations=199,
            min_units=32,
        ),
    )


def _prepare(tmp_path, monkeypatch):
    private = tmp_path / "private"
    workspace = tmp_path / "workspace"
    private.mkdir()
    workspace.mkdir()
    x = np.linspace(-2.0, 2.0, 300)
    pd.DataFrame({"x": x, "target": 2.0 * x + 0.3}).to_parquet(
        private / "challenge.parquet", index=False
    )
    task = _task()
    lock = task.build_lock(
        reference_metrics={"target": 0.05},
        naive_metrics={"target": 0.95},
        degenerate_metrics={"constant": {"target": 1.0}},
        input_digests={"challenge": "fixture"},
    )
    lock_path = tmp_path / "calibration.lock.json"
    write_calibration_lock_atomic(lock_path, lock)
    monkeypatch.setenv("LBX_CALIBRATION_LOCK_PATH", str(lock_path))
    return task, workspace, private


def test_queryable_model_is_evaluated_on_private_rows(tmp_path, monkeypatch) -> None:
    task, workspace, private = _prepare(tmp_path, monkeypatch)
    (workspace / "predictor.py").write_text(
        "def load_predictor():\n"
        "    class Predictor:\n"
        "        def predict(self, rows):\n"
        "            return {'target': [2.0*r['x'] + 0.3 for r in rows]}\n"
        "    return Predictor()\n",
        encoding="utf-8",
    )

    result = task.compute_score(workspace=workspace, private=private)

    assert result["score"] > 0.5
    assert result["metadata"]["security_tier"] == "sealed_challenge"
    assert result["metadata"]["evaluation"]["decisions"]["target"]["accepted"]


def test_queryable_constant_model_is_zero(tmp_path, monkeypatch) -> None:
    task, workspace, private = _prepare(tmp_path, monkeypatch)
    (workspace / "predictor.py").write_text(
        "def load_predictor():\n"
        "    class Predictor:\n"
        "        def predict(self, rows):\n"
        "            return {'target': [0.0 for _ in rows]}\n"
        "    return Predictor()\n",
        encoding="utf-8",
    )

    result = task.compute_score(workspace=workspace, private=private)

    assert result["score"] == pytest.approx(0.0)
    assert result["metadata"]["evaluation"]["decisions"]["target"] == {
        "accepted": False,
        "reason": "exact_constant",
    }


def test_last_bit_float_jitter_is_not_an_agent_fault(tmp_path, monkeypatch) -> None:
    """Thread-parallel ensembles reorder float sums between identical calls.

    Bit-exact repeatability failed those predictors with AgentFault, which the
    runner maps to a hard 0.0.
    """
    task, workspace, private = _prepare(tmp_path, monkeypatch)
    (workspace / "predictor.py").write_text(
        "def load_predictor():\n"
        "    class Predictor:\n"
        "        def __init__(self):\n"
        "            self.calls = 0\n"
        "        def predict(self, rows):\n"
        "            self.calls += 1\n"
        "            jitter = 0.0 if self.calls == 1 else 1e-16\n"
        "            return {'target': [2.0*r['x'] + 0.3 + jitter for r in rows]}\n"
        "    return Predictor()\n",
        encoding="utf-8",
    )

    result = task.compute_score(workspace=workspace, private=private)

    assert result["score"] > 0.5


def test_a_genuinely_unstable_predictor_still_faults(tmp_path, monkeypatch) -> None:
    task, workspace, private = _prepare(tmp_path, monkeypatch)
    (workspace / "predictor.py").write_text(
        "def load_predictor():\n"
        "    class Predictor:\n"
        "        def __init__(self):\n"
        "            self.calls = 0\n"
        "        def predict(self, rows):\n"
        "            self.calls += 1\n"
        "            shift = 0.0 if self.calls == 1 else 0.5\n"
        "            return {'target': [2.0*r['x'] + 0.3 + shift for r in rows]}\n"
        "    return Predictor()\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentFault, match="not deterministic"):
        task.compute_score(workspace=workspace, private=private)


def test_extreme_finite_predictions_are_kept_zero_not_internal_failure(
    tmp_path, monkeypatch
) -> None:
    task, workspace, private = _prepare(tmp_path, monkeypatch)
    (workspace / "predictor.py").write_text(
        "def load_predictor():\n"
        "    class Predictor:\n"
        "        def predict(self, rows):\n"
        "            return {'target': [1e308 if i % 2 else -1e308 for i, _ in enumerate(rows)]}\n"
        "    return Predictor()\n",
        encoding="utf-8",
    )

    result = task.compute_score(workspace=workspace, private=private)

    assert result["score"] == pytest.approx(0.0)


def test_private_nonce_changes_challenge_commitment(tmp_path, monkeypatch) -> None:
    task, workspace, private = _prepare(tmp_path, monkeypatch)
    (workspace / "predictor.py").write_text(
        "def load_predictor():\n"
        "    class Predictor:\n"
        "        def predict(self, rows):\n"
        "            return {'target': [2.0*r['x'] + 0.3 for r in rows]}\n"
        "    return Predictor()\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LBX_EVALUATION_PLAN_ATTESTED", "1")
    monkeypatch.setenv("LBX_EVALUATION_NONCE", "attempt-a")
    first = task.compute_score(workspace=workspace, private=private)
    monkeypatch.setenv("LBX_EVALUATION_NONCE", "attempt-b")
    second = task.compute_score(workspace=workspace, private=private)

    assert (
        first["metadata"]["evaluation"]["seed_commitment"]
        != second["metadata"]["evaluation"]["seed_commitment"]
    )
    assert first["metadata"]["evaluation"]["attested"] is True
    assert second["metadata"]["evaluation"]["attested"] is True


def test_private_truth_column_cannot_be_declared_as_feature() -> None:
    with pytest.raises(ValueError, match="must not include target truth"):
        ContinuousTask.model(
            artifact=PythonPredictor(),
            challenge=PrivateTableChallenge(
                "challenge.parquet",
                feature_columns=["x", "target"],
                sample_size=120,
            ),
            targets=_task().targets,
        )
