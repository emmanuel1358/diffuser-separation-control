from __future__ import annotations

import json

import pytest
from grader_runner.raw_worker import main


def test_raw_worker_measures_task_without_calibration_lock(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    workspace = tmp_path / "workspace"
    grader_dir = tmp_path / "grader"
    private = tmp_path / "private"
    workspace.mkdir()
    grader_dir.mkdir()
    private.mkdir()
    pd.DataFrame({"target": [0.0, 1.0, 2.0]}).to_csv(
        workspace / "submission.csv", index=False
    )
    pd.DataFrame({"target": [0.0, 1.5, 2.0]}).to_parquet(
        private / "test_target.parquet", index=False
    )
    (grader_dir / "compute_score.py").write_text(
        "\n".join(
            [
                (
                    "from grading.evaluation import AnchorRationale, "
                    "ContinuousTask, FloorAnchor, GeneratedCalibration, SRETarget"
                ),
                (
                    "FLOOR = FloorAnchor(1.0, AnchorRationale('theoretical', "
                    "'Population-standardized RMSE has a no-skill value of one.'))"
                ),
                "TASK = ContinuousTask.calibrated(",
                "    targets=[SRETarget.lower('target', weight=1.0, floor=FLOOR)],",
                "    calibration=GeneratedCalibration(),",
                ")",
                "def measure_submission(workspace, private, context):",
                "    import pandas as pd, numpy as np",
                "    assert context.seed == 7",
                "    pred = pd.read_csv(workspace / 'submission.csv')['target'].to_numpy()",
                "    truth = pd.read_parquet(private / 'test_target.parquet')['target'].to_numpy()",
                "    return {'target': float(np.sqrt(np.mean((pred-truth)**2))/np.std(truth, ddof=0))}",
                "def compute_score(): return TASK.grade(None, None)",
                "",
            ]
        )
    )
    result_path = tmp_path / "metrics.json"

    rc = main(
        [
            "--workspace",
            str(workspace),
            "--grader-dir",
            str(grader_dir),
            "--private-dir",
            str(private),
            "--result-path",
            str(result_path),
            "--calibration-seed",
            "7",
        ]
    )

    payload = json.loads(result_path.read_text())
    assert rc == 0
    assert payload["schema_version"] == "raw-continuous-metrics.v1"
    assert payload["task_spec_sha256"]
    assert payload["calibration_seed"] == 7
    assert payload["metrics"]["target"] > 0.0


def test_raw_worker_rejects_legacy_grader(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    grader_dir = tmp_path / "grader"
    private = tmp_path / "private"
    workspace.mkdir()
    grader_dir.mkdir()
    private.mkdir()
    (grader_dir / "compute_score.py").write_text(
        "def compute_score():\n    return 0.0\n"
    )
    result_path = tmp_path / "metrics.json"

    rc = main(
        [
            "--workspace",
            str(workspace),
            "--grader-dir",
            str(grader_dir),
            "--private-dir",
            str(private),
            "--result-path",
            str(result_path),
        ]
    )

    payload = json.loads(result_path.read_text())
    assert rc == 1
    assert payload["error_type"] == "RuntimeError"
    assert "does not define" in payload["error_message"]
