from __future__ import annotations

import pytest

from grading.evaluation import (
    BaselineScore,
    evaluate_baseline_portfolio,
    evaluate_regrade_stability,
    evaluate_score_panel,
)


def test_baseline_portfolio_requires_families_and_reference_headroom() -> None:
    report = evaluate_baseline_portfolio(
        [
            BaselineScore("empty", "no_op", 0.0),
            BaselineScore("formula", "domain_heuristic", 0.18),
            BaselineScore("tree", "simple_fitted", 0.52),
        ],
        required_families=("no_op", "domain_heuristic", "simple_fitted"),
    )

    assert report.passed is False
    assert "tree" in report.issues[0]
    with pytest.raises(ValueError, match="release gate failed"):
        report.require_pass()


def test_score_panel_blocks_ceiling_saturation() -> None:
    saturated = evaluate_score_panel([1.0, 1.0, 1.0, 1.0, 0.99])
    healthy = evaluate_score_panel([0.1, 0.25, 0.5, 0.7, 0.9])

    assert saturated.passed is False
    assert saturated.statistics["ceiling_mass"] == 1.0
    assert healthy.passed is True


def test_regrade_stability_bounds_unchanged_artifact_drift() -> None:
    stable = evaluate_regrade_stability([0.5, 0.501, 0.499], max_range=0.01)
    unstable = evaluate_regrade_stability([0.4, 0.6], max_range=0.05)

    assert stable.passed is True
    assert unstable.passed is False
    assert unstable.statistics["score_range"] == pytest.approx(0.2)
