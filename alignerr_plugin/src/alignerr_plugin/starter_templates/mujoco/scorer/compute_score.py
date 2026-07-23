"""Starter MuJoCo grader.

This scaffold intentionally keeps the rollout logic small. Real tasks should
replace `rollout_policy` with task-specific MuJoCo environment setup and hidden
evaluation episodes.
"""

from pathlib import Path

from grading import PolicyWorker
from grading.evaluation import (
    RegularFileArtifact,
    RubricCriterion,
    RubricTask,
)


def rollout_policy(policy: PolicyWorker, private: Path) -> float:
    """Return a normalized rollout score.

    Replace this with MuJoCo-specific hidden evaluation. The private directory
    can contain XML assets, initial states, seeds, and score normalization data.
    """
    _ = policy, private
    return 1.0


def evaluate(context):
    with context.policy(timeout_s=1.0) as policy:
        return {"rollout": rollout_policy(policy, context.private)}


TASK = RubricTask(
    artifact=RegularFileArtifact("policy.py"),
    criteria=(
        RubricCriterion(
            id="rollout",
            weight=1.0,
            description="Policy succeeds on the hidden MuJoCo rollout",
            required=True,
        ),
    ),
    evaluate=evaluate,
)
