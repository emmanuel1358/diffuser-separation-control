"""Composable sealed-challenge continuous grader.

The agent submits ``predictor.py``. The framework commits that artifact, samples
private challenge rows, runs the predictor in a sandbox, certifies row-level
information, and applies the generated PWL calibration.
"""

from pathlib import Path

from grading.evaluation import (
    AnchorRationale,
    ContinuousTask,
    FloorAnchor,
    GeneratedCalibration,
    PopulationSRETarget,
    PrivateTableChallenge,
    PythonPredictor,
)

TASK = ContinuousTask.model(
    artifact=PythonPredictor("predictor.py"),
    challenge=PrivateTableChallenge(
        "challenge.parquet",
        feature_columns=["feature_1", "feature_2"],  # TODO: your features
        sample_size=256,
        selection_policy="stable_subset",
    ),
    targets=[
        PopulationSRETarget.lower(
            "pred",
            truth_column="target",
            weight=1.0,
            floor=FloorAnchor(
                value=1.0,
                rationale=AnchorRationale(
                    kind="theoretical",
                    summary=(
                        "Explain why this reviewed metric value represents no "
                        "meaningful quality; do not copy one baseline output."
                    ),
                ),
            ),
            perfect=0.0,
        )
    ],
    calibration=GeneratedCalibration(
        "calibration.lock.json",
        quality_floor_mode="effective_no_info",
    ),
    naive="baselines/naive",
)


def compute_score(
    workspace: Path = Path("/tmp/output"),
    trajectory=None,
    private: Path = Path("/mcp_server/data"),
):
    del trajectory
    return TASK.compute_score(workspace=workspace, private=private)
