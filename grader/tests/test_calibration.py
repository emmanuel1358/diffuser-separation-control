from __future__ import annotations

import pytest

from grading import calibration


def test_progress_lower_better_anchors_and_clamps() -> None:
    assert calibration.progress_lower_better(1.0, floor=1.0, perfect=0.0) == 0.0
    assert calibration.progress_lower_better(0.0, floor=1.0, perfect=0.0) == 1.0
    assert calibration.progress_lower_better(0.5, floor=1.0, perfect=0.0) == 0.5
    assert calibration.progress_lower_better(2.0, floor=1.0, perfect=0.0) == 0.0
    assert calibration.progress_lower_better(-1.0, floor=1.0, perfect=0.0) == 1.0


def test_progress_higher_better_anchors_and_clamps() -> None:
    assert calibration.progress_higher_better(0.0, floor=0.0, perfect=1.0) == 0.0
    assert calibration.progress_higher_better(1.0, floor=0.0, perfect=1.0) == 1.0
    assert calibration.progress_higher_better(0.25, floor=0.0, perfect=1.0) == 0.25
    assert calibration.progress_higher_better(5.0, floor=0.0, perfect=1.0) == 1.0


def test_progress_degenerate_anchor_pairs() -> None:
    # perfect <= floor (higher-better) and floor <= perfect (lower-better)
    assert calibration.progress_higher_better(1.0, floor=1.0, perfect=1.0) == 1.0
    assert calibration.progress_higher_better(0.9, floor=1.0, perfect=1.0) == 0.0
    assert calibration.progress_lower_better(0.0, floor=0.0, perfect=0.0) == 1.0
    assert calibration.progress_lower_better(0.1, floor=0.0, perfect=0.0) == 0.0


@pytest.mark.parametrize("x_ref", [0.05, 0.3, 0.5, 0.55, 0.66, 0.8, 0.95])
def test_piecewise_linear_curve_passes_through_anchors(x_ref: float) -> None:
    curve = calibration.PiecewiseLinearCurve.from_reference(x_ref)
    assert curve.score(0.0) == pytest.approx(0.0, abs=1e-12)
    assert curve.score(x_ref) == pytest.approx(0.5, abs=1e-12)
    assert curve.score(1.0) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_piecewise_linear_curve_is_proportional_below_reference() -> None:
    # Half the reference's progress maps to half of 0.5 (~0.25), NOT ~0 the way
    # the convex exponential curve compresses it -- the whole point of the PWL
    # mandate (uncompressed sub-reference / stumped-agent region).
    curve = calibration.PiecewiseLinearCurve.from_reference(0.66)
    assert curve.score(0.33) == pytest.approx(0.25, abs=1e-12)
    assert curve.score(0.33) > calibration.ExponentialCurve.from_reference(0.66).score(0.33)


def test_piecewise_linear_curve_monotonic_and_clamped() -> None:
    curve = calibration.PiecewiseLinearCurve.from_reference(0.66)
    xs = [i / 20 for i in range(21)]
    scores = [curve.score(x) for x in xs]
    assert scores == sorted(scores)
    assert curve.score(-1.0) == 0.0
    assert curve.score(2.0) == 1.0


def test_piecewise_linear_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        calibration.PiecewiseLinearCurve.from_reference(0.0)
    with pytest.raises(ValueError):
        calibration.PiecewiseLinearCurve.from_reference(1.0)
    with pytest.raises(ValueError):
        calibration.piecewise_linear_score(0.5, 1.0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("x_ref", [0.55, 0.66, 0.8, 0.95])
def test_exponential_curve_passes_through_anchors(x_ref: float) -> None:
    curve = calibration.ExponentialCurve.from_reference(x_ref)
    assert curve.score(0.0) == pytest.approx(0.0, abs=1e-9)
    assert curve.score(x_ref) == pytest.approx(0.5, abs=1e-6)
    assert curve.score(1.0) == pytest.approx(1.0, abs=1e-6)


def test_exponential_curve_emits_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning):
        calibration.ExponentialCurve.from_reference(0.66)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_exponential_curve_monotonic_and_clamped() -> None:
    curve = calibration.ExponentialCurve.from_reference(0.66)
    xs = [i / 20 for i in range(21)]
    scores = [curve.score(x) for x in xs]
    assert scores == sorted(scores)
    assert curve.score(-1.0) == 0.0
    assert curve.score(2.0) == 1.0


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_exponential_curve_is_convex_penalizing_naive() -> None:
    # x_ref > 0.5 means the curve sits below the diagonal: a naive aggregate
    # well below the reference clamps toward 0 (near-perfect work is expensive).
    curve = calibration.ExponentialCurve.from_reference(0.66)
    assert curve.score(0.3) < 0.3


def test_solve_exponential_constants_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        calibration.solve_exponential_constants(0.0)
    with pytest.raises(ValueError):
        calibration.solve_exponential_constants(1.0)


def test_exponential_score_matches_curve() -> None:
    a, b = calibration.solve_exponential_constants(0.66)
    assert calibration.exponential_score(0.66, a, b) == pytest.approx(0.5, abs=1e-6)


def test_progress_scores_reject_non_finite_value() -> None:
    # An agent-driven NaN / +inf metric must map to 0.0 (worst), not clamp to 1.0.
    assert calibration.progress_higher_better(float("nan"), 0.0, 1.0) == 0.0
    assert calibration.progress_higher_better(float("inf"), 0.0, 1.0) == 0.0
    assert calibration.progress_lower_better(float("nan"), 1.0, 0.0) == 0.0
    assert calibration.progress_lower_better(float("-inf"), 1.0, 0.0) == 0.0


def test_piecewise_linear_score_rejects_non_finite() -> None:
    # The `x <= 0` / `x >= 1` guards are both False for NaN, so without the finite
    # check a NaN x would fall through to the clamp and score 1.0.
    assert calibration.piecewise_linear_score(float("nan"), 0.5) == 0.0
    assert calibration.piecewise_linear_score(float("inf"), 0.5) == 0.0


def test_exponential_score_rejects_non_finite() -> None:
    a, b = calibration.solve_exponential_constants(0.66)
    assert calibration.exponential_score(float("nan"), a, b) == 0.0
    assert calibration.exponential_score(float("inf"), a, b) == 0.0
