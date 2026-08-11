"""Blocking release gates for baseline strength and score-panel saturation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass


def _finite_scores(values: Iterable[float], *, label: str) -> tuple[float, ...]:
    scores = tuple(float(value) for value in values)
    if not scores or any(not math.isfinite(score) for score in scores):
        raise ValueError(f"{label} must contain finite scores")
    if any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError(f"{label} scores must lie in [0, 1]")
    return scores


@dataclass(frozen=True)
class ReleaseGateReport:
    """Auditable release decision with measured statistics."""

    passed: bool
    issues: tuple[str, ...]
    statistics: dict[str, float | int]

    def require_pass(self) -> None:
        if self.issues:
            raise ValueError("release gate failed: " + "; ".join(self.issues))


@dataclass(frozen=True)
class BaselineScore:
    """One named baseline measured through the production grader path."""

    name: str
    family: str
    score: float

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.family.strip():
            raise ValueError("baseline name and family must be non-empty")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("baseline score must be finite in [0, 1]")


def evaluate_baseline_portfolio(
    baselines: Iterable[BaselineScore],
    *,
    required_families: Iterable[str],
    reference_score: float = 0.5,
    min_baselines: int = 3,
) -> ReleaseGateReport:
    """Fail when obvious baseline families are absent or reach reference quality."""
    entries = tuple(baselines)
    required = tuple(dict.fromkeys(str(family) for family in required_families))
    if min_baselines <= 0 or not 0.0 < reference_score <= 1.0:
        raise ValueError("invalid baseline release-gate configuration")
    if not required or any(not family for family in required):
        raise ValueError("required baseline families must be non-empty")
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("baseline names must be unique")

    issues: list[str] = []
    if len(entries) < min_baselines:
        issues.append(
            f"only {len(entries)} baselines measured; at least {min_baselines} required"
        )
    present = {entry.family for entry in entries}
    missing = sorted(set(required) - present)
    if missing:
        issues.append(f"missing required baseline families: {missing}")
    competitive = sorted(
        entry.name for entry in entries if entry.score >= reference_score - 1e-12
    )
    if competitive:
        issues.append(
            "baseline(s) reach the reference band through the production grader: "
            f"{competitive}"
        )
    return ReleaseGateReport(
        passed=not issues,
        issues=tuple(issues),
        statistics={
            "baseline_count": len(entries),
            "family_count": len(present),
            "max_baseline_score": max(
                (entry.score for entry in entries), default=0.0
            ),
            "reference_score": reference_score,
        },
    )


def evaluate_score_panel(
    scores: Iterable[float],
    *,
    ceiling_threshold: float = 0.99,
    max_ceiling_mass: float = 0.5,
    min_variance: float = 1e-4,
    min_distinct_scores: int = 3,
) -> ReleaseGateReport:
    """Block a task whose representative agent panel is saturated."""
    values = _finite_scores(scores, label="score panel")
    if not 0.0 <= ceiling_threshold <= 1.0:
        raise ValueError("ceiling_threshold must lie in [0, 1]")
    if not 0.0 <= max_ceiling_mass <= 1.0:
        raise ValueError("max_ceiling_mass must lie in [0, 1]")
    if min_variance < 0.0 or min_distinct_scores <= 0:
        raise ValueError("invalid score-panel release-gate configuration")

    ceiling_mass = sum(score >= ceiling_threshold for score in values) / len(values)
    variance = statistics.pvariance(values)
    distinct = len({round(score, 12) for score in values})
    issues: list[str] = []
    if ceiling_mass > max_ceiling_mass:
        issues.append(
            f"ceiling mass {ceiling_mass:.3f} exceeds {max_ceiling_mass:.3f}"
        )
    if variance < min_variance:
        issues.append(f"score variance {variance:.6g} is below {min_variance:.6g}")
    if distinct < min_distinct_scores:
        issues.append(
            f"only {distinct} distinct scores; at least {min_distinct_scores} required"
        )
    return ReleaseGateReport(
        passed=not issues,
        issues=tuple(issues),
        statistics={
            "panel_size": len(values),
            "ceiling_mass": ceiling_mass,
            "score_variance": variance,
            "distinct_scores": distinct,
        },
    )


def evaluate_regrade_stability(
    scores: Iterable[float],
    *,
    max_range: float,
) -> ReleaseGateReport:
    """Block cross-nonce reward drift for one byte-identical artifact."""
    values = _finite_scores(scores, label="regrade panel")
    if max_range < 0.0 or not math.isfinite(max_range):
        raise ValueError("max_range must be finite and non-negative")
    observed_range = max(values) - min(values)
    issues = (
        (
            f"unchanged artifact score range {observed_range:.6g} exceeds "
            f"{max_range:.6g}"
        )
        if observed_range > max_range
        else None
    )
    normalized_issues = (issues,) if issues is not None else ()
    return ReleaseGateReport(
        passed=not normalized_issues,
        issues=normalized_issues,
        statistics={
            "regrade_count": len(values),
            "score_min": min(values),
            "score_max": max(values),
            "score_range": observed_range,
        },
    )


__all__ = [
    "BaselineScore",
    "ReleaseGateReport",
    "evaluate_baseline_portfolio",
    "evaluate_regrade_stability",
    "evaluate_score_panel",
]
