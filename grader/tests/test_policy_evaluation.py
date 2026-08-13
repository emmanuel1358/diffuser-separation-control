from __future__ import annotations

import time

import pytest
from grading.evaluation import PolicyEvaluationTask
from grading.evaluation.policy import (
    POLICY_CHALLENGE_PROTOCOL,
    SCENARIO_SEED_DERIVATION,
    _scenario_seeds,
)
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
    assert task.spec_dict()["schema_version"] == "policy-evaluation-task.v4"
    assert task.spec_dict()["scenario_seed_derivation"] == SCENARIO_SEED_DERIVATION
    assert task.evaluation_plan.evidence["type"] == POLICY_CHALLENGE_PROTOCOL
    assert (
        task.evaluation_plan.evidence["scenario_seed_derivation"]
        == SCENARIO_SEED_DERIVATION
    )
    assert task.spec_dict()["total_timeout_s"] == 3600.0
    assert task.evaluation_plan.evidence["budgets"] == {
        "per_call_timeout_s": 2.0,
        "total_policy_timeout_s": 3600.0,
    }


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


def test_policy_task_rejects_invalid_total_timeout() -> None:
    with pytest.raises(ValueError, match="total timeout"):
        PolicyEvaluationTask(call_timeout_s=2.0, total_timeout_s=1.0)


def test_policy_grade_maps_cumulative_timeout_to_agent_fault(tmp_path) -> None:
    (tmp_path / "policy.py").write_text(
        "import time\n"
        "class Policy:\n"
        "    def act(self, obs):\n"
        "        time.sleep(0.04)\n"
        "        return 0\n"
        "def load_policy():\n"
        "    return Policy()\n"
    )
    task = PolicyEvaluationTask(
        scenarios=8,
        call_timeout_s=0.1,
        total_timeout_s=0.12,
    )

    def rollout(policy, _seed):
        policy.act(None)
        policy.act(None)
        return 0.0

    started = time.monotonic()
    with pytest.raises(AgentFault, match="total compute budget"):
        task.grade(
            workspace=tmp_path,
            rollout=rollout,
            controls={"fixed": lambda _seed: 0.0},
        )
    assert time.monotonic() - started < 2.0


def test_policy_grade_maps_non_finite_candidate_quality_to_agent_fault(
    tmp_path,
) -> None:
    (tmp_path / "policy.py").write_text("def load_policy():\n" "    return object()\n")
    task = PolicyEvaluationTask(scenarios=8)

    with pytest.raises(AgentFault, match="non-finite policy quality"):
        task.grade(
            workspace=tmp_path,
            rollout=lambda _policy, _seed: float("nan"),
            controls={"fixed": lambda _seed: 0.0},
        )


def test_policy_grade_maps_missing_required_method_to_agent_fault(tmp_path) -> None:
    (tmp_path / "policy.py").write_text(
        "def load_policy():\n"
        "    class Policy:\n"
        "        pass\n"
        "    return Policy()\n"
    )
    task = PolicyEvaluationTask(scenarios=8)

    with pytest.raises(AgentFault, match="no attribute 'reset'"):
        task.grade(
            workspace=tmp_path,
            rollout=lambda policy, _seed: policy.reset(),
            controls={"fixed": lambda _seed: 0.0},
        )


def test_policy_scenario_seeds_keep_full_nonce_bound_width() -> None:
    class Context:
        @staticmethod
        def seed(label: str) -> int:
            return 2**63 + int(label.rsplit(":", 1)[-1])

    assert _scenario_seeds(Context(), 3) == [2**63, 2**63 + 1, 2**63 + 2]


def test_policy_evaluation_plan_refresh_uses_registered_task(tmp_path) -> None:
    from grading.evaluation.plan import refresh_evaluation_plan

    problem = tmp_path / "policy-task"
    scorer = problem / "scorer"
    scorer.mkdir(parents=True)
    (scorer / "compute_score.py").write_text(
        "from grading.evaluation import PolicyEvaluationTask\n"
        "TASK = PolicyEvaluationTask(scenarios=8)\n"
    )

    result = refresh_evaluation_plan(problem)

    assert result.status == "written"
    assert result.path.is_file()
    assert "paired-policy-challenge.v2" in result.path.read_text()
