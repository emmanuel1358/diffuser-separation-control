"""Tier-A queryable tabular evaluator with private post-commit challenges."""

from pathlib import Path

from grading.evaluation import (
    AnchorRationale,
    BinaryF1Target,
    ContinuousTask,
    FloorAnchor,
    GeneratedCalibration,
    PopulationSRETarget,
    PrivateTableChallenge,
    PythonPredictor,
)

SRE_FLOOR = FloorAnchor(
    value=1.0,
    rationale=AnchorRationale(
        kind="theoretical",
        summary=(
            "For population-standardized RMSE, a constant prediction at the "
            "held-out population mean has SRE exactly 1."
        ),
        source="grading.evaluation metric sre.rmse_over_population_std.v1",
    ),
)
F1_FLOOR = FloorAnchor(
    value=0.0,
    rationale=AnchorRationale(
        kind="metric_bound",
        summary="Binary F1 is lower-bounded by zero for every finite prediction.",
        source="grading.evaluation metric f1.binary_threshold_0_5.v1",
    ),
)

TASK = ContinuousTask.model(
    artifact=PythonPredictor("predictor.py"),
    challenge=PrivateTableChallenge(
        "challenge.parquet",
        feature_columns=["x1", "x2", "x3"],
        sample_size=400,
        # Grandfathered lock compatibility. New tasks use stable_subset.
        selection_policy="artifact_digest",
    ),
    targets=[
        PopulationSRETarget.lower("t1", weight=0.35, perfect=0.0, floor=SRE_FLOOR),
        PopulationSRETarget.lower("t2", weight=0.35, perfect=0.0, floor=SRE_FLOOR),
        BinaryF1Target.higher("label", weight=0.30, perfect=1.0, floor=F1_FLOOR),
    ],
    calibration=GeneratedCalibration("calibration.lock.json"),
    naive="baselines/naive",
    naive_score_min=1e-6,
    naive_score_max=0.10,
)


def compute_score(
    workspace: Path = Path("/tmp/output"),
    trajectory=None,
    private: Path = Path("/mcp_server/data"),
):
    del trajectory
    return TASK.compute_score(workspace=workspace, private=private)
