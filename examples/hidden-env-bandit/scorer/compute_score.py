"""Sealed policy challenge for the hidden bandit example."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grading import AgentFault, load_env_module
from grading.evaluation import PolicyEvaluationTask

PULL_BUDGET = 64
TASK = PolicyEvaluationTask(
    policy_path="policy.py",
    factory_name="load_policy",
    scenarios=128,
    alpha=0.01,
    # High-quantile reference calibration keeps stochastic challenge scores
    # inside the 0.5 +/- 0.05 ground-truth band without a steep upper segment.
    reference_quality=0.925,
    call_timeout_s=2.0,
)


def _normalized_quality(env, arm: int) -> float:
    if not 0 <= arm < env.n_arms():
        raise AgentFault(f"policy chose arm {arm} out of range [0, {env.n_arms()})")
    span = env.optimal_mean() - env.worst_mean()
    if span <= 0.0:
        return 0.0
    return float((env.mean(arm) - env.worst_mean()) / span)


def _challenge_quality(env, arm: int) -> float:
    """Expand near-optimal regret so the stochastic reference has headroom."""
    normalized = _normalized_quality(env, arm)
    return 1.0 - (1.0 - normalized) ** 0.25


def _submitted_arm(value: Any, *, method: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AgentFault(f"policy.{method}() must return an integer arm") from exc


def compute_score(
    workspace: Path,
    trajectory: list[dict[str, Any]] | None,
    private: Path,
) -> dict[str, Any]:
    del trajectory
    env_module = load_env_module(
        private / "env.py",
        trusted_roots=(private,),
    )

    def rollout(policy, seed: int) -> float:
        env = env_module.make_env(seed=seed)
        policy.reset(env.n_arms(), PULL_BUDGET)
        for _ in range(PULL_BUDGET):
            arm = _submitted_arm(policy.choose(), method="choose")
            if not 0 <= arm < env.n_arms():
                raise AgentFault(
                    f"policy chose arm {arm} out of range [0, {env.n_arms()})"
                )
            policy.observe(arm, env.pull(arm))
        return _challenge_quality(
            env,
            _submitted_arm(policy.recommend(), method="recommend"),
        )

    def fixed_zero(seed: int) -> float:
        env = env_module.make_env(seed=seed)
        return _challenge_quality(env, 0)

    def seed_indexed_open_loop(seed: int) -> float:
        env = env_module.make_env(seed=seed)
        return _challenge_quality(env, seed % env.n_arms())

    return TASK.grade(
        workspace=workspace,
        rollout=rollout,
        controls={
            "fixed_arm_zero": fixed_zero,
            "seed_indexed_open_loop": seed_indexed_open_loop,
        },
    )
