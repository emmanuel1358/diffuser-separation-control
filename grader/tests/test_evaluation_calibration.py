from __future__ import annotations

import json
from types import SimpleNamespace

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
    measure_task_module,
    write_calibration_lock_atomic,
)
from grading.evaluation.plan import validate_serialized_plan


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
                    1.0, "Population-standardized RMSE has a no-skill value of one."
                ),
            ),
            BinaryF1Target.higher(
                "label",
                weight=0.5,
                floor=_floor(0.0, "Binary F1 is bounded below by zero."),
            ),
        ],
        calibration=GeneratedCalibration(),
        naive_score_max=0.1,
    )


# Degenerate raw metrics equal to the author-declared floors: the no-info
# ceiling collapses onto the existing floor and leaves it unchanged, so the
# rest of this suite's assertions (written before the ceiling existed) still
# hold.
_NOOP_DEGENERATE = {"constant": {"value": 1.0, "label": 0.0}}


def test_generated_lock_maps_reference_and_oracle() -> None:
    task = _task()
    lock = task.build_lock(
        reference_metrics={"value": 0.2, "label": 0.9},
        naive_metrics={"value": 0.98, "label": 0.0},
        degenerate_metrics=_NOOP_DEGENERATE,
        input_digests={"model": "a" * 64, "private_data": "b" * 64},
    )

    reference, _, x_ref = task.score_metrics({"value": 0.2, "label": 0.9}, lock)
    oracle, _, x_oracle = task.score_metrics({"value": 0.0, "label": 1.0}, lock)
    null, _, x_null = task.score_metrics({"value": 1.0, "label": 0.0}, lock)

    assert reference == pytest.approx(0.5)
    assert oracle == pytest.approx(1.0)
    assert null == pytest.approx(0.0)
    assert (x_null, x_ref, x_oracle) == pytest.approx((0.0, 0.85, 1.0))
    assert 0.0 < lock.payload["qualification"]["naive_score"] <= 0.1


def test_lock_accepts_non_normalized_author_weights() -> None:
    task = ContinuousTask.static(
        artifact=CsvRows("submission.csv", columns=["value", "label"]),
        targets=[
            SRETarget.lower(
                "value",
                weight=2.0,
                floor=_floor(
                    1.0,
                    "Population-standardized RMSE has a no-skill value of one.",
                ),
            ),
            BinaryF1Target.higher(
                "label",
                weight=1.0,
                floor=_floor(0.0, "Binary F1 is bounded below by zero."),
            ),
        ],
    )
    lock = task.build_lock(
        reference_metrics={"value": 0.2, "label": 0.9},
        naive_metrics={"value": 0.98, "label": 0.0},
        degenerate_metrics=_NOOP_DEGENERATE,
        input_digests={},
    )

    score, _, _ = task.score_metrics({"value": 0.2, "label": 0.9}, lock)

    assert score == pytest.approx(0.5)
    assert lock.payload["targets"]["value"]["weight"] == pytest.approx(2 / 3)
    assert lock.payload["targets"]["label"]["weight"] == pytest.approx(1 / 3)


def test_generated_lock_rejects_noninformative_naive() -> None:
    task = _task()
    with pytest.raises(ValueError, match="weak but informative"):
        task.build_lock(
            reference_metrics={"value": 0.2, "label": 0.9},
            naive_metrics={"value": 1.0, "label": 0.0},
            degenerate_metrics=_NOOP_DEGENERATE,
            input_digests={},
        )


def test_lock_exposes_exact_metric_formula_and_floor_rationale() -> None:
    task = _task()
    lock = task.build_lock(
        reference_metrics={"value": 0.2, "label": 0.9},
        naive_metrics={"value": 0.98, "label": 0.0},
        degenerate_metrics=_NOOP_DEGENERATE,
        input_digests={},
    )

    value = lock.payload["targets"]["value"]
    assert value["metric"]["id"] == "sre.rmse_over_population_std.v1"
    assert "ddof=0" in value["metric"]["formula"]
    assert value["floor"] == {
        "value": 1.0,
        "rationale": {
            "kind": "theoretical",
            "summary": "Population-standardized RMSE has a no-skill value of one.",
        },
    }
    label = lock.payload["targets"]["label"]
    assert label["metric"]["id"] == "f1.binary_threshold_0_5.v1"
    assert "no macro/micro averaging" in label["metric"]["input_contract"]


def test_naive_is_qualification_not_curve_anchor() -> None:
    task = _task()
    common = {
        "reference_metrics": {"value": 0.2, "label": 0.9},
        "degenerate_metrics": _NOOP_DEGENERATE,
        "input_digests": {},
    }
    weaker = task.build_lock(naive_metrics={"value": 0.98, "label": 0.0}, **common)
    stronger = task.build_lock(naive_metrics={"value": 0.96, "label": 0.0}, **common)

    assert weaker.payload["curve"] == stronger.payload["curve"]
    assert weaker.payload["targets"] == stronger.payload["targets"]
    assert (
        weaker.payload["qualification"]["naive_score"]
        != stronger.payload["qualification"]["naive_score"]
    )
    worse_than_naive, _, _ = task.score_metrics({"value": 1.2, "label": 0.0}, weaker)
    assert worse_than_naive == 0.0


def test_floor_rationale_is_required_and_reviewable() -> None:
    with pytest.raises(ValueError, match="at least 20 characters"):
        AnchorRationale(kind="theoretical", summary="because")
    with pytest.raises(ValueError, match="kind must be one of"):
        AnchorRationale(
            kind="baseline_output",
            summary="The floor was copied from one measured baseline output.",
        )


def test_evaluation_paths_reject_traversal_and_custom_lock_names() -> None:
    with pytest.raises(ValueError, match="workspace-relative"):
        CsvRows("../submission.csv", columns=["value"])
    with pytest.raises(ValueError, match="filename is fixed"):
        GeneratedCalibration("../alternate.lock.json")


def test_evaluation_plan_digest_binds_security_tier() -> None:
    plan = _task().evaluation_plan
    payload = {**plan.to_dict(), "plan_sha256": plan.sha256}
    assert validate_serialized_plan(payload) == plan.sha256

    payload["security_tier"] = "sealed_challenge"
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_serialized_plan(payload)


def test_hand_authored_measurement_can_be_non_dataframe(tmp_path) -> None:
    task = ContinuousTask.calibrated(
        targets=_task().targets,
        calibration=GeneratedCalibration(),
    )
    module = SimpleNamespace(
        TASK=task,
        measure_submission=lambda workspace, private: {
            "value": 0.4 if workspace.name == "policy-rollout" else 0.5,
            "label": 0.75 if private.name == "hidden-env" else 0.5,
        },
    )

    measured = measure_task_module(
        module,
        workspace=tmp_path / "policy-rollout",
        private=tmp_path / "hidden-env",
    )

    assert measured == {"value": 0.4, "label": 0.75}


def test_lock_serialization_is_canonical_and_atomic(tmp_path) -> None:
    task = _task()
    lock = task.build_lock(
        reference_metrics={"value": 0.2, "label": 0.9},
        naive_metrics={"value": 0.98, "label": 0.0},
        degenerate_metrics=_NOOP_DEGENERATE,
        input_digests={"z": "last", "a": "first"},
    )
    path = tmp_path / "calibration.lock.json"

    write_calibration_lock_atomic(path, lock)
    first = path.read_bytes()
    write_calibration_lock_atomic(path, lock)

    assert path.read_bytes() == first
    assert json.loads(first)["inputs"] == {"a": "first", "z": "last"}
    assert not list(tmp_path.glob("*.tmp"))
    assert (
        load_calibration_lock(path, task_spec_sha256=task.spec_sha256).sha256
        == lock.sha256
    )


def test_lock_rejects_stale_task_registration(tmp_path) -> None:
    task = _task()
    lock = task.build_lock(
        reference_metrics={"value": 0.2, "label": 0.9},
        naive_metrics={"value": 0.98, "label": 0.0},
        degenerate_metrics=_NOOP_DEGENERATE,
        input_digests={},
    )
    path = tmp_path / "calibration.lock.json"
    write_calibration_lock_atomic(path, lock)

    with pytest.raises(ValueError, match="stale"):
        load_calibration_lock(path, task_spec_sha256="not-the-task")


def test_compute_score_reads_generated_lock(monkeypatch, tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    task = _task()
    private = tmp_path / "private"
    workspace = tmp_path / "workspace"
    private.mkdir()
    workspace.mkdir()
    # N=200, not the handful of rows a hand-written fixture would otherwise
    # use: compute_score now runs a grade-time permutation null test (see
    # ContinuousTask._grade_time_floors) on the raw arrays, and a tiny N
    # makes that test degenerate -- few enough row-permutations exist that
    # some reproduce a near-perfect match by chance, which spuriously
    # tightens the floor and breaks this reference-replay assertion. A
    # few hundred rows gives the permutation test the resolution it needs
    # to correctly recognize this reference submission as informative.
    n = 200
    truth = pd.DataFrame(
        {"value": [float(i) for i in range(n)], "label": [i % 2 for i in range(n)]}
    )
    truth.to_parquet(private / "test_target.parquet", index=False)
    pd.DataFrame(
        {
            "value": [float(i) + 0.5 for i in range(n)],
            "label": [i % 2 for i in range(n)],
        }
    ).to_csv(workspace / "submission.csv", index=False)
    reference_metrics = task.measure(workspace=workspace, private=private)
    lock = task.build_lock(
        reference_metrics=reference_metrics,
        naive_metrics={"value": 0.98, "label": 0.0},
        degenerate_metrics=_NOOP_DEGENERATE,
        input_digests={"test": "fixture"},
    )
    lock_path = tmp_path / "calibration.lock.json"
    write_calibration_lock_atomic(lock_path, lock)
    monkeypatch.setenv("LBX_CALIBRATION_LOCK_PATH", str(lock_path))

    result = task.compute_score(workspace=workspace, private=private)

    assert result["score"] == pytest.approx(0.5)
    assert result["metadata"]["calibration_lock_sha256"] == lock.sha256
    assert set(result["subscores"]) == {"value_progress", "label_progress"}
