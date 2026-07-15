"""Starter MuJoCo grader.

This scaffold intentionally keeps the rollout logic small. Real tasks should
replace `rollout_policy` with task-specific MuJoCo environment setup and hidden
evaluation episodes.
"""

from pathlib import Path
from typing import Any

from grading import AgentFault, PolicyWorker


def rollout_policy(policy: PolicyWorker, private: Path) -> float:
    """Return a normalized rollout score.

    Replace this with MuJoCo-specific hidden evaluation. The private directory
    can contain XML assets, initial states, seeds, and score normalization data.
    """
    _ = policy, private
    return 1.0


def compute_score(
    workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path
) -> dict[str, Any]:
    """Score a submitted MuJoCo policy."""
    _ = trajectory
    policy_path = workspace / "policy.py"
    if not policy_path.exists():
        return {"score": 0.0, "metadata": {"error": "missing /tmp/output/policy.py"}}
    try:
        with PolicyWorker(policy_path, timeout_s=1.0) as policy:
            score = rollout_policy(policy, private)
    except AgentFault:
        raise
    except Exception as exc:
        # The rollout executes agent code; a crash/timeout is the agent's fault and
        # must be kept as a real 0.0 via the typed AgentFault. Returning a homemade
        # 0.0 here would also swallow author/infra bugs into a kept score (over-keep)
        # and poison RL training. Let author/infra faults propagate to discard.
        raise AgentFault(
            f"policy rollout failed: {type(exc).__name__}: {exc}"
        ) from exc
    return {
        "score": max(0.0, min(1.0, float(score))),
        "subscores": {"rollout": float(score)},
        "weights": {"rollout": 1.0},
    }
