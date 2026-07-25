"""Grader for the hidden-env bandit example.

Grade-time flow (the env socket is already torn down by ``stop_env_server()``
before this runs):

1. Load the held-out env in-process via ``grading.load_env_module`` -- this is
   the SAME ``make_env(seed=0)`` the agent explored over the socket, so it
   carries the same hidden arm means.
2. Load the agent's submitted ``policy.py`` in a sandboxed worker via
   ``grading.load_submitted_policy`` (factory ``load_policy``); call
   ``policy.choose()`` to get the arm it believes is best.
3. Score continuously by how close the chosen arm's TRUE mean is to the optimal
   (1.0 for the best arm, ~0 for the worst). ``best_arm`` / ``mean`` are
   grader-only env methods (never socket-reachable), so the agent had to infer
   the best arm by pulling, not by asking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grading import AgentFault, load_env_module, load_submitted_policy


def compute_score(
    workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path
) -> float:
    _ = trajectory
    # Held-out env (root-only /mcp_server/data/env.py). Author data: a failure to
    # load it is infra (propagate), not the agent's fault.
    env = load_env_module(private / "env.py").make_env(seed=0)
    best = env.best_arm()
    optimal = env.optimal_mean()
    worst = env.worst_mean()

    # Agent submission: missing / non-regular / oversized / load-raise -> AgentFault.
    policy = load_submitted_policy(workspace / "policy.py", timeout_s=10.0)
    try:
        choice = policy.choose()
    except AgentFault:
        raise
    except Exception as exc:  # noqa: BLE001 - submitted-policy boundary
        raise AgentFault(
            f"policy.choose() failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        policy.close()

    try:
        choice = int(choice)
    except (TypeError, ValueError) as exc:
        raise AgentFault(f"policy.choose() must return an int arm; got {choice!r}") from exc
    if not 0 <= choice < env.n_arms():
        raise AgentFault(f"policy chose arm {choice} out of range [0, {env.n_arms()})")

    if choice == best:
        return 1.0
    denom = optimal - worst
    if denom <= 0:
        return 0.0
    return float(max(0.0, min(1.0, (env.mean(choice) - worst) / denom)))
