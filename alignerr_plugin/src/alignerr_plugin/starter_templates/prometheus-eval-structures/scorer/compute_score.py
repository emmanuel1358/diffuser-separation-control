"""Starter declarative structures rubric for Prometheus evaluation delivery."""

from grading.evaluation import JsonArtifact, RubricCriterion, RubricTask


def evaluate(context):
    return {"design_present": bool(context.candidate)}


TASK = RubricTask(
    artifact=JsonArtifact("design.json"),
    criteria=(
        RubricCriterion(
            id="design_present",
            weight=1.0,
            description="design.json exists, parses, and declares parameters",
            required=True,
        ),
    ),
    evaluate=evaluate,
)
