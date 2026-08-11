from __future__ import annotations

import pytest

from grading.evaluation import PolicyEvaluationTask
from grading.faults import AgentFault


def test_policy_task_declares_required_control_families() -> None:
    task = PolicyEvaluationTask()

    assert task.spec_dict()["required_control_families"] == [
        "no_op",
        "constant",
        "open_loop",
    ]
    assert task.evaluation_plan.evidence["controls"]["required_families"] == [
        "no_op",
        "constant",
        "open_loop",
    ]


def test_policy_grade_rejects_missing_control_family(tmp_path) -> None:
    task = PolicyEvaluationTask()
    controls = {
        "never_act": lambda _seed: 0.0,
        "fixed": lambda _seed: 0.0,
    }

    with pytest.raises(RuntimeError, match="missing required.*open_loop"):
        task.grade(
            workspace=tmp_path,
            rollout=lambda _policy, _seed: 0.0,
            controls=controls,
            control_families={
                "never_act": "no_op",
                "fixed": "constant",
            },
        )


def test_policy_grade_requires_every_control_to_be_classified(tmp_path) -> None:
    task = PolicyEvaluationTask(required_control_families=("constant",))

    with pytest.raises(RuntimeError, match="classify every trusted control"):
        task.grade(
            workspace=tmp_path,
            rollout=lambda _policy, _seed: 0.0,
            controls={"fixed": lambda _seed: 0.0},
            control_families={},
        )


def test_policy_grade_keeps_legacy_unclassified_controls_compatible(tmp_path) -> None:
    task = PolicyEvaluationTask()

    with pytest.raises(AgentFault, match="missing submitted policy"):
        task.grade(
            workspace=tmp_path,
            rollout=lambda _policy, _seed: 0.0,
            controls={"fixed": lambda _seed: 0.0},
        )
