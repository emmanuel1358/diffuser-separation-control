"""Calibration records no-information probes without replacing quality floors."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from grading.evaluation import (
    AnchorRationale,
    BinaryF1Target,
    ContinuousTask,
    CsvRows,
    FloorAnchor,
    GeneratedCalibration,
    SRETarget,
    load_calibration_lock,
    write_calibration_lock_atomic,
)


def _floor(value: float, summary: str) -> FloorAnchor:
    return FloorAnchor(
        value=value,
        rationale=AnchorRationale(kind="theoretical", summary=summary),
    )


def _task() -> ContinuousTask:
    return ContinuousTask.static(
        artifact=CsvRows("submission.csv", columns=["value", "label"]),
        targets=[
            SRETarget.lower(
                "value",
                weight=0.5,
                floor=_floor(
                    1.0,
                    "Population-standardized RMSE has a no-skill value of one.",
                ),
            ),
            BinaryF1Target.higher(
                "label",
                weight=0.5,
                floor=_floor(0.0, "Binary F1 is bounded below by zero."),
            ),
        ],
        calibration=GeneratedCalibration(
            naive_semantic_gap_acknowledgement=AnchorRationale(
                kind="reviewed_exception",
                summary=(
                    "The qualification floor intentionally excludes majority-class "
                    "F1 while runtime quality retains weak informative progress."
                ),
            )
        ),
    )


_DEGENERATE_METRICS = {
    "constant_mean": {"value": 1.0, "label": 0.30},
    "constant_median": {"value": 1.02, "label": 0.49},
    "row_index": {"value": 1.5, "label": 0.05},
    "shuffled_truth": {"value": 1.4, "label": 0.02},
}
_REFERENCE_METRICS = {"value": 0.2, "label": 0.9}
_NAIVE_METRICS = {"value": 0.9, "label": 0.35}


def _lock():
    return _task().build_lock(
        reference_metrics=_REFERENCE_METRICS,
        naive_metrics=_NAIVE_METRICS,
        degenerate_metrics=_DEGENERATE_METRICS,
        input_digests={"fixture": "digest"},
    )


def test_ceiling_is_audited_but_reviewed_quality_floor_is_preserved() -> None:
    lock = _lock()

    label = lock.payload["targets"]["label"]
    assert label["raw_floor"] == pytest.approx(0.0)
    assert label["no_info_ceiling"] == pytest.approx(0.49)
    assert label["floor"]["value"] == pytest.approx(0.0)

    value = lock.payload["targets"]["value"]
    assert value["raw_floor"] == pytest.approx(1.0)
    assert value["no_info_ceiling"] == pytest.approx(1.0)
    assert value["floor"]["value"] == pytest.approx(1.0)
    assert set(lock.payload["qualification"]["degenerate_scores"]) == set(
        _DEGENERATE_METRICS
    )
    qualification = lock.payload["qualification"]
    assert qualification["qualification_naive_score"] == pytest.approx(
        qualification["naive_score"]
    )
    assert (
        qualification["runtime_naive_quality_score"]
        > qualification["qualification_naive_score"]
    )
    assert qualification["naive_semantic_gap_acknowledgement"]["kind"] == (
        "reviewed_exception"
    )


def test_lock_validation_rejects_missing_semantic_gap_acknowledgement(
    tmp_path,
) -> None:
    task = _task()
    payload = json.loads(json.dumps(_lock().payload))
    payload["qualification"]["naive_semantic_gap_acknowledgement"] = None
    path = tmp_path / "missing-gap-ack.lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="naive_semantic_gap_acknowledgement"):
        load_calibration_lock(path, task_spec_sha256=task.spec_sha256)


def test_constant_quality_can_be_positive_but_evidence_zeros_it(
    tmp_path, monkeypatch
) -> None:
    task = _task()
    lock = _lock()
    scalar_score, scalar_progress, _ = task.score_metrics(
        {"value": 1.02, "label": 0.49}, lock
    )
    assert scalar_score > 0.0
    assert scalar_progress["label"] == pytest.approx(0.49)

    lock_path = tmp_path / "calibration.lock.json"
    write_calibration_lock_atomic(lock_path, lock)
    monkeypatch.setenv("LBX_CALIBRATION_LOCK_PATH", str(lock_path))

    n = 200
    truth = pd.DataFrame(
        {
            "value": np.linspace(-2.0, 2.0, n),
            "label": np.arange(n) % 2,
        }
    )
    submission = pd.DataFrame(
        {
            "value": np.full(n, 0.0),
            "label": np.full(n, 1.0),
        }
    )
    result = task.grade(submission, truth)

    assert result["score"] == pytest.approx(0.0)
    assert result["subscores"] == pytest.approx(
        {"value_progress": 0.0, "label_progress": 0.0}
    )
    decisions = result["metadata"]["evaluation"]["decisions"]
    assert decisions["value"]["reason"] == "exact_constant"
    assert decisions["label"]["reason"] == "exact_constant"
    assert "raw_metrics" not in result["metadata"]


def test_reference_and_weak_informative_quality_mapping_is_stable() -> None:
    task = _task()
    lock = _lock()

    reference_score, _, _ = task.score_metrics(_REFERENCE_METRICS, lock)
    assert reference_score == pytest.approx(0.5)

    weak_score, _, _ = task.score_metrics({"value": 0.6, "label": 0.6}, lock)
    assert 0.0 < weak_score < 0.5


def test_large_qualification_runtime_gap_requires_reviewed_acknowledgement() -> None:
    task = ContinuousTask.static(
        artifact=CsvRows("submission.csv", columns=["value", "label"]),
        targets=_task().targets,
        calibration=GeneratedCalibration(),
    )

    with pytest.raises(ValueError, match="naive_semantic_gap_acknowledgement"):
        task.build_lock(
            reference_metrics=_REFERENCE_METRICS,
            naive_metrics=_NAIVE_METRICS,
            degenerate_metrics=_DEGENERATE_METRICS,
            input_digests={},
        )


def test_no_information_ceiling_crossing_perfect_fails_fast() -> None:
    task = ContinuousTask.static(
        artifact=CsvRows("submission.csv", columns=["label"]),
        targets=[
            BinaryF1Target.higher(
                "label",
                weight=1.0,
                floor=_floor(0.0, "Binary F1 is bounded below by zero."),
                perfect=0.5,
            )
        ],
        calibration=GeneratedCalibration(),
    )
    with pytest.raises(ValueError, match="floor-vs-perfect invariant"):
        task.build_lock(
            reference_metrics={"label": 0.3},
            naive_metrics={"label": 0.1},
            degenerate_metrics={"majority": {"label": 0.6}},
            input_digests={},
        )


def test_lock_validation_recomputes_no_information_semantics(tmp_path) -> None:
    lock = _lock()
    payload = json.loads(json.dumps(lock.payload))
    payload["targets"]["label"]["no_info_ceiling"] = 0.1
    path = tmp_path / "tampered.lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="no-information ceiling"):
        load_calibration_lock(path, task_spec_sha256=_task().spec_sha256)


def test_runtime_rejects_lock_target_drift_even_with_copied_task_digest() -> None:
    task = _task()
    lock = _lock()
    lock.payload["targets"]["label"]["metric"]["formula"] = "forged"

    with pytest.raises(RuntimeError, match="does not match the TASK"):
        task.score_metrics(_REFERENCE_METRICS, lock)
