"""One-target Monte Carlo permutation statistic.

This module computes evidence only. The family-wide decision and reward
semantics live in :mod:`grading.evaluation.decision`: accepted targets retain
their reviewed-floor quality progress, while rejected targets become zero.

Validity requires the declared inferential units to be exchangeable under the
no-information null. IID row permutation is therefore a Tier-B compatibility
protocol, not a universal model/policy provenance test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from grading.evaluation.metrics import MetricTarget

# K=1999 (+ the 1 observed draw = 2000 total) gives a one-sided p-value
# resolution of 1/2000 = 5e-4, comfortably finer than PERM_ALPHA, and ~20
# order statistics below the alpha=0.01 tail so the tail quantile itself is
# stable run-to-run (see test_permutation_ceiling_is_stable_across_seeds).
# Fixed at module scope so every grade call spends the same, bounded amount
# of kernel recomputation -- zero authoring burden, not a per-task knob.
K_PERMUTATIONS = 1999
PERM_ALPHA = 0.01

# Compatibility default. Production evaluators pass a context-derived seed.
PERM_SEED = 0x5EED_1053


@dataclass(frozen=True)
class PermutationNullResult:
    """One target's permutation-null outcome for a single grade call."""

    m_obs: float
    ceiling: float
    p_value: float
    k: int
    alpha: float

    def to_metadata(self) -> dict[str, float | int]:
        return {
            "m_obs": self.m_obs,
            "ceiling": self.ceiling,
            "p_value": self.p_value,
            "k": self.k,
            "alpha": self.alpha,
        }


def permutation_null(
    target: MetricTarget,
    prediction: Any,
    truth: Any,
    *,
    k: int = K_PERMUTATIONS,
    alpha: float = PERM_ALPHA,
    seed: int = PERM_SEED,
) -> PermutationNullResult:
    """Row-permutation null test of ``target.kernel`` on this submission.

    Shuffles ``prediction`` row-wise ``k`` times (truth held fixed),
    recomputes the metric each time, and reports:

    - ``ceiling``: a diagnostic direction-aware tail order statistic.
    - ``p_value``: one-sided, ``(1 + count(null at least as favorable as
      m_obs)) / (k + 1)`` -- the standard permutation-test correction that
      keeps the estimate strictly positive.
    """
    import numpy as np

    if k < 1:
        raise ValueError("permutation count must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("permutation alpha must lie strictly between 0 and 1")
    pred = np.asarray(prediction, dtype=float)
    actual = np.asarray(truth, dtype=float)
    if pred.shape != actual.shape or pred.ndim < 1:
        raise ValueError(
            "permutation evidence requires prediction/truth arrays with the "
            f"same leading-axis shape; got {pred.shape} and {actual.shape}"
        )
    rng = np.random.default_rng(seed)
    m_obs = target.measure(pred, actual)
    null = np.empty(k, dtype=float)
    for i in range(k):
        null[i] = target.measure(rng.permutation(pred), actual)
    if target.direction == "lower":
        ceiling = float(np.quantile(null, alpha, method="nearest"))
        at_least_as_favorable = int(np.sum(null <= m_obs))
    else:
        ceiling = float(np.quantile(null, 1.0 - alpha, method="nearest"))
        at_least_as_favorable = int(np.sum(null >= m_obs))
    p_value = (1 + at_least_as_favorable) / (k + 1)
    return PermutationNullResult(
        m_obs=m_obs, ceiling=ceiling, p_value=p_value, k=k, alpha=alpha
    )


__all__ = [
    "K_PERMUTATIONS",
    "PERM_ALPHA",
    "PERM_SEED",
    "PermutationNullResult",
    "permutation_null",
]
