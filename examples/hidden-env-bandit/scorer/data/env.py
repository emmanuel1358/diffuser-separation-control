"""Hidden multi-armed bandit environment for the hidden-env example task.

Baked to ``/mcp_server/data/env.py`` (root-only, 0700) by the task Dockerfile.
The agent reaches a ``make_env()`` instance over ``/tmp/env.sock`` via the public
``/data/env_client.py`` but can never read THIS file -- the per-arm reward means
are the hidden dynamics it must infer by pulling arms.

``_env_public_methods`` pins the socket-reachable surface to exploration only;
``best_arm`` / ``mean`` are grader-only (the in-process scorer calls them
directly, never over the socket), so the agent cannot just ask for the answer.
"""

from __future__ import annotations

import numpy as np

N_ARMS = 8


class BanditEnv:
    # Only these names dispatch over the socket. best_arm / mean / optimal_mean
    # are deliberately omitted so the held-out answer never reaches the agent.
    _env_public_methods = frozenset({"reset", "pull", "n_arms"})

    def __init__(self, seed: int = 0, noise: float = 0.25) -> None:
        # Hidden per-arm means in [0, 1]; fixed by seed so the grader's env (same
        # seed) shares them with the instance the agent explored over the socket.
        self._means = np.random.default_rng(seed).uniform(0.0, 1.0, size=N_ARMS)
        self._noise = float(noise)
        self._pull_rng = np.random.default_rng(seed + 1)

    # ---- socket-reachable (exploration) ----
    def n_arms(self) -> int:
        return N_ARMS

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self._pull_rng = np.random.default_rng(int(seed))
        return {"n_arms": N_ARMS}

    def pull(self, arm: int) -> float:
        arm = int(arm)
        if not 0 <= arm < N_ARMS:
            raise ValueError(f"arm {arm} out of range [0, {N_ARMS})")
        return float(self._means[arm] + self._noise * self._pull_rng.standard_normal())

    # ---- grader-only (NOT socket-reachable) ----
    def best_arm(self) -> int:
        return int(np.argmax(self._means))

    def optimal_mean(self) -> float:
        return float(np.max(self._means))

    def worst_mean(self) -> float:
        return float(np.min(self._means))

    def mean(self, arm: int) -> float:
        return float(self._means[int(arm)])


def make_env(**env_kwargs) -> BanditEnv:
    """Factory the env server calls on ``__create__`` and the grader calls
    in-process. ``seed`` is the only accepted kwarg (pin via env_config.json's
    allowed_env_kwargs in production)."""
    return BanditEnv(seed=int(env_kwargs.get("seed", 0)))
