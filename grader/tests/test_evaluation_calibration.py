from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from grading.evaluation import (
    AnchorRationale,
    BinaryF1Target,
    CalibrationMeasureContext,
    ContinuousTask,
    CsvRows,
    FloorAnchor,
    GeneratedCalibration,
    SRETarget,
    WorkspaceDegenerateProbes,
    WorkspaceProbe,
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


def test_reviewed_floor_tied_naive_can_qualify_at_zero() -> None:
    task = ContinuousTask.static(
        artifact=CsvRows("submission.csv", columns=["value", "label"]),
        targets=_task().targets,
        naive_score_min=0.0,
        naive_at_floor=AnchorRationale(
            kind="reviewed_exception",
            summary=(
                "The sanctioned no-op baseline exactly ties every measured "
                "no-information floor and no weak-positive baseline exists."
            ),
        ),
    )
    lock = task.build_lock(
        reference_metrics={"value": 0.2, "label": 0.9},
        naive_metrics={"value": 1.0, "label": 0.0},
        degenerate_metrics=_NOOP_DEGENERATE,
        input_digests={},
    )

    score_range = lock.payload["qualification"]["naive_score_range"]
    assert score_range["inclusive_min"] == 0.0
    assert "exclusive_min" not in score_range
    assert score_range["rationale"]["kind"] == "reviewed_exception"
    assert lock.payload["qualification"]["naive_score"] == 0.0


def test_reviewed_floor_naive_must_tie_effective_no_info_floor() -> None:
    task = ContinuousTask.static(
        artifact=CsvRows("submission.csv", columns=["value", "label"]),
        targets=_task().targets,
        naive_score_min=0.0,
        naive_at_floor=AnchorRationale(
            kind="reviewed_exception",
            summary=(
                "The sanctioned no-op baseline must tie rather than underperform "
                "the measured no-information family."
            ),
        ),
    )

    with pytest.raises(ValueError, match="weak but informative"):
        task.build_lock(
            reference_metrics={"value": 0.2, "label": 0.9},
            naive_metrics={"value": 1.1, "label": 0.0},
            degenerate_metrics=_NOOP_DEGENERATE,
            input_digests={},
        )


def test_naive_floor_exception_requires_zero_min_and_reviewed_rationale() -> None:
    with pytest.raises(ValueError, match="naive_score_min=0"):
        ContinuousTask.static(
            artifact=CsvRows("submission.csv", columns=["value", "label"]),
            targets=_task().targets,
            naive_at_floor=AnchorRationale(
                kind="reviewed_exception",
                summary="This reviewed exception has a sufficient explanation.",
            ),
        )
    with pytest.raises(ValueError, match="kind='reviewed_exception'"):
        ContinuousTask.static(
            artifact=CsvRows("submission.csv", columns=["value", "label"]),
            targets=_task().targets,
            naive_score_min=0.0,
            naive_at_floor=AnchorRationale(
                kind="domain_review",
                summary="This rationale uses the wrong semantic review category.",
            ),
        )


def test_default_calibration_and_naive_range_serialization_is_unchanged() -> None:
    task = _task()
    assert GeneratedCalibration().spec_dict() == {
        "type": "generated_lock.v1",
        "filename": "calibration.lock.json",
    }
    assert task.spec_dict()["naive_score_range"] == {
        "exclusive_min": 1e-6,
        "inclusive_max": 0.1,
    }


def test_workspace_probe_spec_is_bounded_and_canonical() -> None:
    provider = WorkspaceDegenerateProbes(
        probes=[
            WorkspaceProbe(
                name="seeded-random",
                path="baselines/degenerate/seeded-random",
                rationale="A seeded no-information policy over the public action API.",
            ),
            WorkspaceProbe(
                name="no-op",
                path="baselines/degenerate/no-op",
                rationale="A policy that always emits the documented neutral action.",
            ),
        ]
    )
    spec = GeneratedCalibration(degenerate_probes=provider).spec_dict()
    assert [probe["name"] for probe in spec["degenerate_probes"]["probes"]] == [
        "no-op",
        "seeded-random",
    ]
    with pytest.raises(ValueError, match="baselines/degenerate"):
        WorkspaceProbe(
            name="bad",
            path="../outside",
            rationale="This invalid path still has a sufficiently long rationale.",
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


def test_lock_is_found_when_the_runner_nests_the_bundle(tmp_path, monkeypatch) -> None:
    """The accelerator lane exports one absolute path; the runner picks the root.

    The lane assumed the bundle extracts at /workspace and the runner extracted
    it at /workspace/workspace, so every continuous task died on a missing lock
    and scored 0.0. Resolution now accepts either layout.
    """
    from grading.evaluation.lock import (
        CALIBRATION_LOCK_PATH_ENV,
        resolve_calibration_lock_path,
    )

    exported = tmp_path / "workspace" / "calibration" / "calibration.lock.json"
    actual = (
        tmp_path / "workspace" / "workspace" / "calibration" / "calibration.lock.json"
    )
    actual.parent.mkdir(parents=True)
    actual.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(CALIBRATION_LOCK_PATH_ENV, str(exported))

    assert resolve_calibration_lock_path() == actual

    # With the lock where the lane said it would be, that path still wins.
    exported.parent.mkdir(parents=True)
    exported.write_text("{}", encoding="utf-8")
    assert resolve_calibration_lock_path() == exported

    # And the mirror image: the lane now exports the nested path, so a runner
    # that stops nesting must not put us back where we started.
    monkeypatch.setenv(CALIBRATION_LOCK_PATH_ENV, str(actual))
    actual.unlink()
    assert resolve_calibration_lock_path() == exported


def test_missing_lock_names_every_path_it_looked_in(tmp_path, monkeypatch) -> None:
    from grading.evaluation.lock import CALIBRATION_LOCK_PATH_ENV, load_calibration_lock

    exported = tmp_path / "workspace" / "calibration" / "calibration.lock.json"
    monkeypatch.setenv(CALIBRATION_LOCK_PATH_ENV, str(exported))
    with pytest.raises(
        RuntimeError, match="searched .*workspace/workspace/calibration"
    ):
        load_calibration_lock()


def test_a_named_bundle_lock_never_falls_back_to_the_author_image_lock(
    tmp_path, monkeypatch
) -> None:
    """A missing sealed lock must fail, not grade against author calibration.

    RUNTIME_LOCK_ROOT on a task image is the author's own lock, shipped beside
    an .author-source marker. Searching it after an explicitly-named bundle lock
    missed would turn a loud failure into a silently wrong score.
    """
    from grading.evaluation import lock as lock_module

    baked = tmp_path / "image-calibration"
    baked.mkdir()
    (baked / "calibration.lock.json").write_text("{}", encoding="utf-8")
    (baked / ".author-source").write_text("author-image-fallback\n", encoding="utf-8")
    monkeypatch.setattr(lock_module, "RUNTIME_LOCK_ROOT", baked)

    exported = tmp_path / "workspace" / "calibration" / "calibration.lock.json"
    monkeypatch.setenv(lock_module.CALIBRATION_LOCK_PATH_ENV, str(exported))

    assert baked not in [p.parent for p in lock_module.calibration_lock_candidates()]
    with pytest.raises(RuntimeError, match="calibration lock is missing"):
        lock_module.load_calibration_lock()

    # With no lane naming a path, the in-image location is still the answer.
    monkeypatch.delenv(lock_module.CALIBRATION_LOCK_PATH_ENV)
    assert (
        lock_module.resolve_calibration_lock_path() == baked / "calibration.lock.json"
    )


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

    def measure_submission(workspace, private, context):
        assert context.derive_seed("cases") == CalibrationMeasureContext(7).derive_seed(
            "cases"
        )
        return {
            "value": 0.4 if workspace.name == "policy-rollout" else 0.5,
            "label": 0.75 if private.name == "hidden-env" else 0.5,
        }

    module = SimpleNamespace(
        TASK=task,
        measure_submission=measure_submission,
    )

    measured = measure_task_module(
        module,
        workspace=tmp_path / "policy-rollout",
        private=tmp_path / "hidden-env",
        context=CalibrationMeasureContext(7),
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


def test_lock_loader_rejects_ambiguous_naive_range(tmp_path) -> None:
    task = _task()
    lock = task.build_lock(
        reference_metrics={"value": 0.2, "label": 0.9},
        naive_metrics={"value": 0.98, "label": 0.0},
        degenerate_metrics=_NOOP_DEGENERATE,
        input_digests={},
    )
    payload = json.loads(json.dumps(lock.payload))
    payload["qualification"]["naive_score_range"]["inclusive_min"] = 0.0
    path = tmp_path / "ambiguous.lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        load_calibration_lock(path, task_spec_sha256=task.spec_sha256)


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


def test_custom_kernel_identity_survives_a_different_import_path() -> None:
    """The host and the grader worker import the same scorer under different
    module names. Embedding __module__ hashed one kernel two ways, so the sealed
    plan came back as a stale TASK."""
    from grading.evaluation.metrics import (
        MetricTarget,
        RegisteredMetric,
        registered_kernel_identity,
    )

    def custom_kernel(prediction, truth) -> float:
        return 0.0

    def _target() -> MetricTarget:
        return MetricTarget(
            name="value",
            metric=RegisteredMetric(
                id="custom.value.v1",
                formula="task-specific",
                input_contract="finite numeric arrays of identical shape",
            ),
            direction="lower",
            weight=1.0,
            floor=_floor(1.0, "A no-skill predictor scores one on this metric."),
            perfect=0.0,
            prediction_column="value",
            truth_column="value",
            kernel=custom_kernel,
        )

    baseline = registered_kernel_identity(_target())
    custom_kernel.__module__ = "compute_score"
    host_view = registered_kernel_identity(_target())
    custom_kernel.__module__ = "grader.compute_score"
    worker_view = registered_kernel_identity(_target())

    assert baseline == host_view == worker_view
    assert "compute_score" not in host_view


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
