"""FLOOR / REF / PERFECT continuous calibration (the ML_Envs methodology).

Maps a raw task metric onto ``[0, 1]`` in two stages so that continuous-scored
tasks are provably *learnable* (a naive baseline cannot match the expert) and
not *trivially saturated* (near-perfect work stays expensive):

1. Per-target **progress**: linearly map a raw metric from ``FLOOR`` (the worst
   plausible value -> ``0.0``) to ``PERFECT`` (the theoretical optimum ->
   ``1.0``), clamped to ``[0, 1]``. Use :func:`progress_lower_better` when
   smaller is better (RMSE, error) and :func:`progress_higher_better` when
   larger is better (F1, accuracy).
2. Aggregate -> **score**: weight-combine the per-target progresses into one
   aggregate ``x in [0, 1]``, then map it through an anchor curve that passes
   through ``(0, 0)``, ``(x_ref, 0.5)`` and ``(1, 1)``, where ``x_ref`` is the
   aggregate progress the *reference* solution reaches. The curve lands the
   reference at exactly 0.5 and reserves 1.0 for the optimum.

   Use :class:`PiecewiseLinearCurve` -- it is the **only sanctioned curve**
   (per ``ReviewerInstructions`` 1.6.2): two straight segments, ``0 -> 0.5`` over
   ``[0, x_ref]`` and ``0.5 -> 1`` over ``[x_ref, 1]``. It keeps the
   sub-reference (stumped-agent) region uncompressed so partial progress is
   scored proportionally. :class:`ExponentialCurve` is **deprecated**: its convex
   shape compresses the sub-reference region and inflates near-perfect work; it
   is retained only for back-compat with tasks authored before the PWL mandate.

Anchor convention (per scored target):

* ``FLOOR``   = worst plausible raw metric (NOT a baseline's measured value;
  e.g. for SRE the constant-mean predictor gives RMSE == std -> SRE == 1.0).
* ``REF``     = the reference solution's measured raw metric (maps to 0.5).
* ``PERFECT`` = the metric's theoretical optimum (SRE 0.0, F1 1.0).

Promoted out of individual scorers into this shared, tested module so authors
do not re-derive the curve in every ``compute_score.py``.

Example::

    from grading import calibration

    xt1 = calibration.progress_lower_better(t1_sre, floor=1.0, perfect=0.0)
    xlb = calibration.progress_higher_better(label_f1, floor=0.0, perfect=1.0)
    x_agg = 0.5 * xt1 + 0.5 * xlb

    # x_ref is the aggregate the reference reaches; reused across submissions.
    curve = calibration.PiecewiseLinearCurve.from_reference(x_ref)
    score = curve.score(x_agg)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

# Shared nan-rejecting clamp: the bare max(0, min(1, x)) idiom returns 1.0 for
# NaN/+inf, so a metric an agent can drive to NaN (e.g. Pearson r of a constant
# prediction) would otherwise be laundered into a perfect score. grade imports
# none of the scoring modules, so this edge is cycle-free.
from grading.grade import _finite_clamp01

__all__ = [
    "progress_higher_better",
    "progress_lower_better",
    "piecewise_linear_score",
    "PiecewiseLinearCurve",
    "solve_exponential_constants",
    "exponential_score",
    "ExponentialCurve",
]


def progress_higher_better(value: float, floor: float, perfect: float) -> float:
    """Map a *higher-is-better* metric to progress in ``[0, 1]``.

    ``floor`` -> 0.0, ``perfect`` -> 1.0, clamped. When ``perfect <= floor``
    (a degenerate anchor pair) returns 1.0 iff ``value >= perfect``.
    """
    if perfect <= floor:
        return 1.0 if value >= perfect else 0.0
    return _finite_clamp01((value - floor) / (perfect - floor))


def progress_lower_better(value: float, floor: float, perfect: float) -> float:
    """Map a *lower-is-better* metric to progress in ``[0, 1]``.

    ``floor`` is the worst (highest) value -> 0.0, ``perfect`` is the best
    (lowest) -> 1.0, clamped. When ``floor <= perfect`` (degenerate) returns
    1.0 iff ``value <= perfect``.
    """
    if floor <= perfect:
        return 1.0 if value <= perfect else 0.0
    return _finite_clamp01((floor - value) / (floor - perfect))


def piecewise_linear_score(x: float, x_ref: float) -> float:
    """Evaluate the two-segment PWL anchor curve at ``x``, clamped to ``[0, 1]``.

    The curve passes through ``(0, 0)``, ``(x_ref, 0.5)`` and ``(1, 1)`` with a
    straight line on each side of ``x_ref``::

        S(x) = 0.5 * (x / x_ref)                     for x <= x_ref
        S(x) = 0.5 + 0.5 * (x - x_ref) / (1 - x_ref) for x >  x_ref

    Unlike :func:`exponential_score`, the sub-reference region ``[0, x_ref]`` is
    not compressed: an agent reaching half the reference's progress scores ~0.25,
    not ~0. ``x_ref`` must be in the open interval ``(0, 1)``.
    """
    if not 0.0 < x_ref < 1.0:
        raise ValueError(f"x_ref must be in the open interval (0, 1); got {x_ref!r}")
    # NaN-porous guards below (`nan <= 0.0` and `nan >= 1.0` are both False) would
    # let a NaN x fall through to the clamp and score 1.0; reject it up front.
    if not math.isfinite(x):
        return 0.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x <= x_ref:
        value = 0.5 * (x / x_ref)
    else:
        value = 0.5 + 0.5 * (x - x_ref) / (1.0 - x_ref)
    return _finite_clamp01(value)


@dataclass(frozen=True)
class PiecewiseLinearCurve:
    """Reusable two-segment PWL anchor curve through ``(0,0)``, ``(x_ref,0.5)``,
    ``(1,1)``.

    This is the sanctioned continuous-scoring curve. Solve once from the
    reference aggregate, then score every submission's aggregate through
    :meth:`score`.
    """

    x_ref: float

    @classmethod
    def from_reference(cls, x_ref: float) -> "PiecewiseLinearCurve":
        if not 0.0 < x_ref < 1.0:
            raise ValueError(f"x_ref must be in the open interval (0, 1); got {x_ref!r}")
        return cls(x_ref=x_ref)

    def score(self, x: float) -> float:
        return piecewise_linear_score(x, self.x_ref)


def _bisect(func, lo: float, hi: float, *, iters: int = 200) -> float:
    """Bisection for the root of ``func`` bracketed by ``[lo, hi]`` (works for
    either sign orientation of the endpoints)."""
    f_lo = func(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if abs(f_mid) < 1e-15 or (hi - lo) < 1e-15:
            return mid
        if (f_mid < 0.0) == (f_lo < 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_exponential_constants(x_ref: float) -> tuple[float, float]:
    """Solve ``(a, b)`` so ``S(x) = a * (exp(b*x) - 1)`` passes through
    ``(0, 0)``, ``(x_ref, 0.5)`` and ``(1, 1)``.

    The midpoint constraint ``(exp(b) - 1) == 2 * (exp(b*x_ref) - 1)`` has the
    trivial root ``b == 0`` (the linear limit ``S(x) = x``) plus one meaningful
    nonzero root. We bracket and bisect the nonzero root robustly (Newton can
    fall into the ``b == 0`` basin for extreme ``x_ref``). ``x_ref`` must be in
    ``(0, 1)``; in practice the reference aggregate sits above 0.5, giving a
    convex curve that makes near-perfect work expensive.
    """
    if not 0.0 < x_ref < 1.0:
        raise ValueError(f"x_ref must be in the open interval (0, 1); got {x_ref!r}")

    def constraint(b: float) -> float:
        return (math.exp(b) - 1.0) - 2.0 * (math.exp(b * x_ref) - 1.0)

    if abs(x_ref - 0.5) < 1e-9:
        b = 1e-6  # near-linear limit; S(x) ~ x
    elif x_ref > 0.5:
        lo, hi = 1e-9, 1.0
        while constraint(hi) < 0.0:
            hi *= 2.0
            if hi > 1e6:
                raise ValueError(f"could not bracket curve for x_ref={x_ref!r}")
        b = _bisect(constraint, lo, hi)
    else:
        lo, hi = -1.0, -1e-9
        while constraint(lo) < 0.0:
            lo *= 2.0
            if lo < -1e6:
                raise ValueError(f"could not bracket curve for x_ref={x_ref!r}")
        b = _bisect(constraint, lo, hi)

    denom = math.exp(b) - 1.0
    if abs(denom) < 1e-15:
        raise ValueError(f"degenerate curve for x_ref={x_ref!r} (b={b!r})")
    a = 1.0 / denom
    return a, b


def exponential_score(x: float, a: float, b: float) -> float:
    """Evaluate ``S(x) = a * (exp(b*x) - 1)`` clamped to ``[0, 1]``."""
    return _finite_clamp01(float(a * (math.exp(b * x) - 1.0)))


@dataclass(frozen=True)
class ExponentialCurve:
    """DEPRECATED exponential anchor curve through ``(0,0)``, ``(x_ref,0.5)``,
    ``(1,1)``.

    Deprecated in favour of :class:`PiecewiseLinearCurve`, the only sanctioned
    continuous-scoring curve. The convex exponential shape compresses the
    sub-reference region and inflates near-perfect work; new tasks must not use
    it. Retained only so tasks authored before the PWL mandate keep scoring.
    """

    x_ref: float
    a: float
    b: float

    @classmethod
    def from_reference(cls, x_ref: float) -> "ExponentialCurve":
        warnings.warn(
            "ExponentialCurve is deprecated; use PiecewiseLinearCurve "
            "(the only sanctioned continuous-scoring curve).",
            DeprecationWarning,
            stacklevel=2,
        )
        a, b = solve_exponential_constants(x_ref)
        return cls(x_ref=x_ref, a=a, b=b)

    def score(self, x: float) -> float:
        return exponential_score(x, self.a, self.b)
