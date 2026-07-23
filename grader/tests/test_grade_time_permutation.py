"""Reward-hacking regression tests for Tier-B information certificates."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from grading.evaluation import (
    AnchorRationale,
    BinaryF1Target,
    ContinuousTask,
    FloorAnchor,
    GeneratedCalibration,
    IIDPermutationEvidence,
    SRETarget,
    write_calibration_lock_atomic,
)

N = 240
SEED = 20260720
RNG = np.random.default_rng(SEED)
TRUTH_VALUE = RNG.normal(loc=2.0, scale=1.5, size=N)
TRUTH_LABEL = (RNG.random(N) < 0.45).astype(float)


def _floor(value: float, summary: str) -> FloorAnchor:
    return FloorAnchor(
        value=value,
        rationale=AnchorRationale(kind="theoretical", summary=summary),
    )


def _task() -> ContinuousTask:
    return ContinuousTask.calibrated(
        targets=[
            SRETarget.lower(
                "value",
                weight=0.5,
                floor=_floor(
                    1.25,
                    "A deliberately generous reviewed SRE floor for test coverage.",
                ),
            ),
            BinaryF1Target.higher(
                "label",
                weight=0.5,
                floor=_floor(0.0, "Binary F1 is bounded below by zero."),
            ),
        ],
        calibration=GeneratedCalibration(),
        evidence=IIDPermutationEvidence(
            family_alpha=0.02,
            permutations=499,
            min_units=32,
        ),
    )


def _lock(task: ContinuousTask):
    return task.build_lock(
        reference_metrics={"value": 0.18, "label": 0.94},
        naive_metrics={"value": 0.98, "label": 0.50},
        degenerate_metrics={
            "constant": {"value": 1.0, "label": 0.48},
            "shuffle": {"value": 1.42, "label": 0.05},
        },
        input_digests={"fixture": "reward-hacking"},
    )


def _grade(tmp_path, monkeypatch, value, label, *, trace: bool = False):
    task = _task()
    lock_path = tmp_path / "calibration.lock.json"
    write_calibration_lock_atomic(lock_path, _lock(task))
    monkeypatch.setenv("LBX_CALIBRATION_LOCK_PATH", str(lock_path))
    if trace:
        monkeypatch.setenv(
            "LBX_EVALUATION_TRACE_PATH", str(tmp_path / "evaluation-details.json")
        )
    submission = pd.DataFrame({"value": value, "label": label})
    truth = pd.DataFrame({"value": TRUTH_VALUE, "label": TRUTH_LABEL})
    return task.grade(submission, truth)


@pytest.mark.parametrize("relative_sigma", [0.0, 1e-6, 1e-4, 1e-3, 1e-2])
def test_constant_and_perturbation_ladder_scores_zero(
    tmp_path, monkeypatch, relative_sigma
) -> None:
    rng = np.random.default_rng(71)
    value = np.full(N, float(TRUTH_VALUE.mean()))
    label = np.full(N, 1.0)
    if relative_sigma:
        value += rng.normal(0.0, relative_sigma * float(TRUTH_VALUE.std()), size=N)
        # Keep classification values on one side of the fixed threshold while
        # still adding the sort of epsilon perturbation used to evade identity.
        label += rng.normal(0.0, relative_sigma * 0.01, size=N)

    result = _grade(tmp_path, monkeypatch, value, label)

    assert result["score"] == pytest.approx(0.0)
    assert all(
        not decision["accepted"]
        for decision in result["metadata"]["evaluation"]["decisions"].values()
    )


def test_marginal_shuffle_and_row_index_attacks_score_zero(
    tmp_path, monkeypatch
) -> None:
    rng = np.random.default_rng(72)
    value = rng.permutation(TRUTH_VALUE)
    label = rng.permutation(TRUTH_LABEL)

    shuffled = _grade(tmp_path, monkeypatch, value, label)
    assert shuffled["score"] == pytest.approx(0.0)

    row = np.linspace(0.0, 1.0, N)
    indexed = _grade(tmp_path, monkeypatch, row, (row > 0.5).astype(float))
    assert indexed["score"] == pytest.approx(0.0)


def test_genuine_signal_keeps_original_quality_progress(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(73)
    value = TRUTH_VALUE + rng.normal(0.0, 0.45, size=N)
    label = TRUTH_LABEL.copy()
    flip = rng.random(N) < 0.08
    label[flip] = 1.0 - label[flip]
    result = _grade(tmp_path, monkeypatch, value, label)

    decisions = result["metadata"]["evaluation"]["decisions"]
    assert all(decision["accepted"] for decision in decisions.values())
    assert result["score"] > 0.0
    assert result["subscores"]["value_progress"] > 0.0
    assert result["subscores"]["label_progress"] > 0.0


def test_partial_target_failure_only_zeros_that_target(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(74)
    value = TRUTH_VALUE + rng.normal(0.0, 0.35, size=N)
    label = np.ones(N)

    result = _grade(tmp_path, monkeypatch, value, label)

    decisions = result["metadata"]["evaluation"]["decisions"]
    assert decisions["value"]["accepted"] is True
    assert decisions["label"] == {
        "accepted": False,
        "reason": "exact_constant",
    }
    assert result["subscores"]["value_progress"] > 0.0
    assert result["subscores"]["label_progress"] == 0.0
    assert result["score"] > 0.0


def test_public_receipt_is_redacted_and_private_trace_is_separate(
    tmp_path, monkeypatch
) -> None:
    rng = np.random.default_rng(75)
    value = TRUTH_VALUE + rng.normal(0.0, 0.4, size=N)
    label = TRUTH_LABEL.copy()
    result = _grade(tmp_path, monkeypatch, value, label, trace=True)

    receipt = result["metadata"]["evaluation"]
    serialized = json.dumps(receipt)
    for secret_field in (
        "p_value",
        "m_obs",
        "ceiling",
        "raw_metric",
        "raw_floor",
        "no_info_ceiling",
    ):
        assert secret_field not in serialized

    trace = json.loads((tmp_path / "evaluation-details.json").read_text())
    assert "p_value" in trace["targets"]["value"]
    assert "raw_metric" in trace["targets"]["value"]


def test_same_local_attempt_is_deterministic(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(76)
    value = TRUTH_VALUE + rng.normal(0.0, 0.4, size=N)
    label = TRUTH_LABEL.copy()

    first = _grade(tmp_path, monkeypatch, value, label)
    second = _grade(tmp_path, monkeypatch, value, label)

    assert first == second


def test_private_trace_refuses_existing_symlink(tmp_path, monkeypatch) -> None:
    victim = tmp_path / "victim.json"
    victim.write_text("do not overwrite", encoding="utf-8")
    trace = tmp_path / "trace.json"
    trace.symlink_to(victim)
    monkeypatch.setenv("LBX_EVALUATION_TRACE_PATH", str(trace))
    rng = np.random.default_rng(77)
    value = TRUTH_VALUE + rng.normal(0.0, 0.4, size=N)

    with pytest.raises(RuntimeError, match="securely"):
        _grade(tmp_path, monkeypatch, value, TRUTH_LABEL.copy())
    assert victim.read_text(encoding="utf-8") == "do not overwrite"


def test_small_effective_sample_fails_configuration(tmp_path, monkeypatch) -> None:
    task = _task()
    lock_path = tmp_path / "calibration.lock.json"
    write_calibration_lock_atomic(lock_path, _lock(task))
    monkeypatch.setenv("LBX_CALIBRATION_LOCK_PATH", str(lock_path))
    truth = pd.DataFrame({"value": [0.0] * 10, "label": [0.0, 1.0] * 5})
    submission = truth.copy()

    with pytest.raises(RuntimeError, match="independent units"):
        task.grade(submission, truth)
