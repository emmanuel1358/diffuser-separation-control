"""Starter declarative CFD rubric.

The shared ``RubricTask`` owns submission I/O, malformed-input handling,
aggregation, fault attribution, and result serialization. Replace ``evaluate``
with pure domain logic and run external solvers through ``context.run_solver``.
"""

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
