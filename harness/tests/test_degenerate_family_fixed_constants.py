"""Fixed-class constants must be in the no-info degenerate family regardless
of which class the train split happens to favor.

Train-derived constants (constant-mean/median) only ever emit whichever class
is the *train* majority. If train and test class prevalence differ, that
train-derived constant can miss the best feature-blind constant on test,
understating no_info_ceiling and leaving a constant-class hack viable. Trying
both fixed classes directly closes that gap (flagged in PR review).
"""

from __future__ import annotations

from pathlib import Path

from grading.evaluation import (
    AnchorRationale,
    BinaryF1Target,
    ContinuousTask,
    CsvRows,
    FloorAnchor,
    PrivateTableChallenge,
    PythonPredictor,
    SRETarget,
)
from grading.evaluation.author import GeneratedCalibration
from lbx_rl_tasks_harness import calibration
from lbx_rl_tasks_harness.models import HarnessProblem


def _task() -> ContinuousTask:
    return ContinuousTask.static(
        artifact=CsvRows("submission.csv", columns=["value", "label"]),
        targets=[
            SRETarget.lower(
                "value",
                weight=0.5,
                floor=FloorAnchor(
                    1.0,
                    AnchorRationale(
                        kind="theoretical",
                        summary=(
                            "Population-standardized RMSE has a no-skill value of one."
                        ),
                    ),
                ),
            ),
            BinaryF1Target.higher(
                "label",
                weight=0.5,
                floor=FloorAnchor(
                    0.0,
                    AnchorRationale(
                        kind="metric_bound",
                        summary="Binary F1 is bounded below by zero.",
                    ),
                ),
            ),
        ],
        calibration=GeneratedCalibration(),
        truth_filename="test_target.csv",
    )


def _problem(tmp_path: Path) -> HarnessProblem:
    source = tmp_path / "problem"
    public = source / "data"
    private = source / "scorer" / "data"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    # Train label majority is skewed to 1 (three 1s, one 0): constant-mean and
    # constant-median can only ever propose the constant "1" strategy.
    public.joinpath("train.csv").write_text("value,label\n0.1,1\n0.3,1\n0.5,1\n0.7,0\n")
    private.joinpath("test_target.csv").write_text("value,label\n0.4,0\n0.6,1\n")
    return HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[],
        source_problem_dir=source,
    )


def test_family_includes_both_fixed_classes_independent_of_train_majority(
    monkeypatch, tmp_path
) -> None:
    written: dict[str, dict] = {}

    def fake_measure(_problem, workspace, _output, _transcript):
        del _problem, _output, _transcript
        import pandas as pd

        frame = pd.read_csv(workspace / "submission.csv")
        written[workspace.name] = {
            column: frame[column].tolist() for column in frame.columns
        }
        return {
            "schema_version": "raw-continuous-metrics.v1",
            "task_spec_sha256": task.spec_sha256,
            "calibration_seed": 0,
            "metrics": {
                column: float(frame[column].mean()) for column in frame.columns
            },
        }

    monkeypatch.setattr(calibration, "measure_workspace_in_container", fake_measure)

    problem = _problem(tmp_path)
    task = _task()
    run_dir = tmp_path / "run"

    measurements = calibration._measure_degenerate_family(problem, task, run_dir)

    assert set(measurements) >= {"constant-negative", "constant-positive"}

    # Both rows fixed to 0.0 for the negative-class strategy...
    assert written["degenerate-constant-negative"]["label"] == [0.0, 0.0]
    # ...and to 1.0 for the positive-class strategy -- regardless of train
    # majority (which is 1 here, so constant-mean/median never try "0").
    assert written["degenerate-constant-positive"]["label"] == [1.0, 1.0]

    # constant-mean/median are stuck on the train majority (1) and cannot
    # reach the "0" constant that constant-negative now covers.
    assert written["degenerate-constant-mean"]["label"] == [1.0, 1.0]
    assert written["degenerate-constant-median"]["label"] == [1.0, 1.0]

    # Non-classification columns fall back to the train mean, same as the
    # other constant strategies -- these strategies only add coverage for
    # the classification target.
    train_value_mean = (0.1 + 0.3 + 0.5 + 0.7) / 4
    assert written["degenerate-constant-negative"]["value"] == [
        train_value_mean,
        train_value_mean,
    ]
    assert written["degenerate-constant-positive"]["value"] == [
        train_value_mean,
        train_value_mean,
    ]


def test_queryable_family_uses_production_challenge_row_count(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "queryable-problem"
    public = source / "data"
    private = source / "scorer" / "data"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    public.joinpath("train.csv").write_text("value,label\n0.1,0\n0.3,0\n0.5,1\n0.7,1\n")
    private.joinpath("challenge.csv").write_text(
        "feature,value,label\n"
        + "".join(f"{index},{index / 10.0},{index % 2}\n" for index in range(48))
    )
    problem = HarnessProblem(
        id="queryable",
        source_format="problem-dir",
        prompt="demo",
        outputs=[],
        source_problem_dir=source,
    )
    static_task = _task()
    task = ContinuousTask.model(
        artifact=PythonPredictor(),
        challenge=PrivateTableChallenge(
            "challenge.csv",
            feature_columns=["feature"],
            sample_size=32,
        ),
        targets=static_task.targets,
    )
    measured_truth: list[list[float]] = []

    def fake_measure_registered(self, submission, truth):
        assert len(submission) == 32
        assert len(truth) == 32
        measured_truth.append(truth["value"].tolist())
        return {target.name: 0.0 for target in self.targets}

    monkeypatch.setattr(
        ContinuousTask,
        "measure_registered",
        fake_measure_registered,
    )

    measurements = calibration._measure_degenerate_family(
        problem,
        task,
        tmp_path / "run",
    )

    assert len(measurements) == 7
    assert len(measured_truth) == 7
    assert all(rows == measured_truth[0] for rows in measured_truth)
