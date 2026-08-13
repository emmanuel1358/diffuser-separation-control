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
    OneHot,
    PopulationSRETarget,
    PrivateTableChallenge,
    PythonPredictor,
    Simplex,
    write_calibration_lock_atomic,
)
from grading.evaluation.author import _validate_prediction_contract
from grading.evaluation.context import EvaluationContext
from grading.faults import AgentFault


def _task() -> ContinuousTask:
    return ContinuousTask.model(
        artifact=PythonPredictor(),
        challenge=PrivateTableChallenge(
            "challenge.parquet",
            feature_columns=["x"],
            sample_size=120,
            selection_policy="stable_subset",
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


def _prepare(tmp_path, monkeypatch, *, task=None):
    private = tmp_path / "private"
    workspace = tmp_path / "workspace"
    private.mkdir()
    workspace.mkdir()
    x = np.linspace(-2.0, 2.0, 300)
    pd.DataFrame({"x": x, "target": 2.0 * x + 0.3}).to_parquet(
        private / "challenge.parquet", index=False
    )
    task = task or _task()
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
    _, first_truth, _ = task._load_model_challenge(workspace=workspace, private=private)
    monkeypatch.setenv("LBX_EVALUATION_NONCE", "attempt-b")
    second = task.compute_score(workspace=workspace, private=private)
    _, second_truth, _ = task._load_model_challenge(
        workspace=workspace, private=private
    )

    assert (
        first["metadata"]["evaluation"]["seed_commitment"]
        != second["metadata"]["evaluation"]["seed_commitment"]
    )
    assert first["metadata"]["evaluation"]["attested"] is True
    assert second["metadata"]["evaluation"]["attested"] is True
    pd.testing.assert_frame_equal(first_truth, second_truth)


def test_evaluation_context_replay_verifies_artifact_and_reuses_nonce(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LBX_EVALUATION_PLAN_ATTESTED", "1")
    monkeypatch.setenv("LBX_EVALUATION_NONCE", "private-attempt")
    original = EvaluationContext.create_from_artifact_digest(
        task_digest="task", candidate_digest="artifact"
    )

    replay = EvaluationContext.replay(
        task_digest="task",
        candidate_digest="artifact",
        replay={"nonce": "private-attempt", "artifact_digest": "artifact"},
    )

    assert replay.commitment == original.commitment
    assert replay.attested is False
    with pytest.raises(ValueError, match="artifact digest"):
        EvaluationContext.replay(
            task_digest="task",
            candidate_digest="different",
            replay={"nonce": "private-attempt", "artifact_digest": "artifact"},
        )


def test_grouped_prediction_constraints_reject_invalid_rows() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 0.3],
            "b": [0.0, 0.7],
            "label": [0, 2],
        }
    )

    with pytest.raises(AgentFault, match="one-hot"):
        _validate_prediction_contract(
            frame,
            value_domains={},
            constraints=(OneHot(["a", "b"]),),
        )
    with pytest.raises(AgentFault, match="declared domain"):
        _validate_prediction_contract(
            frame,
            value_domains={"label": (0, 1)},
            constraints=(),
        )
    _validate_prediction_contract(
        frame,
        value_domains={},
        constraints=(Simplex(["a", "b"]),),
    )


def test_row_independent_scope_rejects_batch_transduction(
    tmp_path, monkeypatch
) -> None:
    task = ContinuousTask.model(
        artifact=PythonPredictor(prediction_scope="row_independent"),
        challenge=PrivateTableChallenge(
            "challenge.parquet",
            feature_columns=["x"],
            sample_size=120,
            selection_policy="stable_subset",
        ),
        targets=_task().targets,
        evidence=_task().evidence,
    )
    task, workspace, private = _prepare(tmp_path, monkeypatch, task=task)
    (workspace / "predictor.py").write_text(
        "def load_predictor():\n"
        "    class Predictor:\n"
        "        def predict(self, rows):\n"
        "            mean = sum(r['x'] for r in rows) / len(rows)\n"
        "            return {'target': [r['x'] - mean for r in rows]}\n"
        "    return Predictor()\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentFault, match="row_independent"):
        task.compute_score(workspace=workspace, private=private)


def test_predictor_budgets_and_full_bank_policy_are_in_spec() -> None:
    predictor = PythonPredictor(
        predict_timeout_s=7,
        first_call_timeout_s=11,
        max_rows=321,
        max_reply_bytes=4096,
    )
    challenge = PrivateTableChallenge(
        "challenge.parquet",
        feature_columns=["x"],
    )

    assert predictor.spec_dict()["predict_timeout_s"] == 7.0
    assert predictor.spec_dict()["max_reply_bytes"] == 4096
    assert challenge.spec_dict()["selection_policy"] == "full_bank"
    assert challenge.spec_dict()["sample_size"] is None

    stable = PrivateTableChallenge(
        "challenge.parquet",
        feature_columns=["x"],
        sample_size=120,
    )
    assert stable.selection_policy == "stable_subset"
    assert stable.legacy_spec_dict() is None

    legacy = PrivateTableChallenge(
        "challenge.parquet",
        feature_columns=["x"],
        sample_size=120,
        selection_policy="artifact_digest",
    )
    assert legacy.selection_policy == "artifact_digest"
    assert legacy.legacy_spec_dict()["type"] == "private_table.v1"


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
