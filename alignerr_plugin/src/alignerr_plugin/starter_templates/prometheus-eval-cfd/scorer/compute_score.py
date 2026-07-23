"""Starter declarative CFD rubric for Prometheus evaluation delivery."""

from grading.evaluation import JsonArtifact, RubricCriterion, RubricTask


def evaluate(context):
    return {"control_present": bool(context.candidate)}


TASK = RubricTask(
    artifact=JsonArtifact("control.json"),
    criteria=(
        RubricCriterion(
            id="control_present",
            weight=1.0,
            description="control.json exists, parses, and declares parameters",
            required=True,
        ),
    ),
    evaluate=evaluate,
)
