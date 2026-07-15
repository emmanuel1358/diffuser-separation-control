"""Tests for `normalize_compute_score_return`.

Especially the regression test that protects ML_Envs anchor mappings —
``dict["score"]`` MUST stay authoritative even when the per-target subscores
would weighted-average to a different number.
"""

from __future__ import annotations

import math

import pytest

from grading import Grade, normalize_compute_score_return


def test_bare_float_is_clamped() -> None:
    g = normalize_compute_score_return(1.5)
    assert g.score() == 1.0
    g = normalize_compute_score_return(-0.3)
    assert g.score() == 0.0


def test_bare_float_records_return_shape() -> None:
    g = normalize_compute_score_return(0.5)
    assert g.metadata["return_shape"] == "bare_float"
    assert g.subscores == {"score": 0.5}
    assert g.weights == {"score": 1.0}


def test_non_finite_scores_raise_instead_of_clamping_to_one() -> None:
    with pytest.raises(ValueError):
        normalize_compute_score_return(math.nan)
    with pytest.raises(ValueError):
        normalize_compute_score_return({"score": math.inf})


def test_score_dict_headline_is_authoritative() -> None:
    """ML_Envs anchor-mapping regression test.

    The headline (`dict["score"]`) MUST stay verbatim; weighted_avg of
    subscores would give 0.6 here.
    """
    raw = {
        "score": 0.42,
        "subscores": {"rmse": 0.9, "f1": 0.3},
        "weights": {"rmse": 0.5, "f1": 0.5},
        "metadata": {"x_ref": 0.7},
    }
    g = normalize_compute_score_return(raw)
    assert g.score() == 0.42
    assert g.weighted_total() == 0.6
    assert g.metadata["return_shape"] == "score_dict"
    assert g.metadata["x_ref"] == 0.7  # caller metadata preserved


def test_score_dict_diagnostic_subscores_are_not_clamped() -> None:
    raw = {
        "score": 0.25,
        "subscores": {"raw_loss": -3.5, "raw_return": 2.75},
        "weights": {"raw_loss": 0.0, "raw_return": 0.0},
    }

    g = normalize_compute_score_return(raw)

    assert g.score() == 0.25
    assert g.subscores == {"raw_loss": -3.5, "raw_return": 2.75}


def test_score_dict_without_weights_falls_back_to_equal() -> None:
    raw = {"score": 0.5, "subscores": {"a": 0.5, "b": 0.5}}
    g = normalize_compute_score_return(raw)
    assert g.weights == {"a": 0.5, "b": 0.5}


def test_score_dict_without_score_raises() -> None:
    with pytest.raises(TypeError):
        normalize_compute_score_return({"subscores": {"a": 1.0}})


def test_grade_pass_through() -> None:
    g_in = Grade(
        subscores={"a": 1.0},
        weights={"a": 1.0},
        scoring_mode="binary",
    )
    g_out = normalize_compute_score_return(g_in)
    assert g_out is g_in  # identity, not a copy


def test_serialized_grade_round_trip_preserves_per_criterion() -> None:
    """`to_dict()` -> `normalize` round-trip must NOT collapse per-criterion data.

    This is the regression that prevents `RubricBuilder.grade().to_dict()`
    returns from being silently flattened to a single headline subscore.
    """
    original = Grade(
        subscores={"compiled": 1.0, "stable": 0.8},
        weights={"compiled": 0.4, "stable": 0.6},
        criterion_logs={
            "compiled": {"reasoning": "ok", "grading_type": "deterministic"},
            "stable": {"reasoning": "rolled out fine", "grading_type": "deterministic"},
        },
    )
    serialized = original.to_dict()
    reconstructed = normalize_compute_score_return(serialized)

    # Headline preserved
    assert reconstructed.score() == original.score()
    # Per-criterion subscores recovered (NOT collapsed to one "score" key)
    assert reconstructed.subscores == {"compiled": 1.0, "stable": 0.8}
    assert reconstructed.weights == {"compiled": 0.4, "stable": 0.6}

    # criterion_logs reconstructed from rubric_breakdown
    assert "compiled" in (reconstructed.criterion_logs or {})
    assert reconstructed.metadata["return_shape"] == "rubric_grade"


def test_serialized_grade_round_trip_to_dict_is_idempotent_for_headline() -> None:
    """Round-tripping a Grade through to_dict + normalize must not drift the headline."""
    original = Grade(
        subscores={"a": 0.7, "b": 0.4, "c": 0.9},
        weights={"a": 0.3, "b": 0.3, "c": 0.4},
    )
    s1 = original.to_dict()["score"]
    reconstructed = normalize_compute_score_return(original.to_dict())
    s2 = reconstructed.to_dict()["score"]
    assert s1 == s2


def test_unsupported_return_type_raises() -> None:
    with pytest.raises(TypeError):
        normalize_compute_score_return("not a number")
    with pytest.raises(TypeError):
        normalize_compute_score_return([1.0, 0.0])


def test_grade_with_non_finite_score_is_rejected() -> None:
    # Defense-in-depth: Grade.score() already clamps non-finite to 0.0, but the
    # normalize Grade branch also calls _finite_float on the emitted headline, so
    # a Grade (here a subclass) that returns a non-finite score is rejected here
    # rather than laundered into the payload.
    class _NonFiniteGrade(Grade):
        def score(self) -> float:  # type: ignore[override]
            return math.inf

    with pytest.raises(ValueError):
        normalize_compute_score_return(
            _NonFiniteGrade(subscores={"a": 1.0}, weights={"a": 1.0})
        )


def test_finite_grade_passes_normalize_unchanged() -> None:
    g = Grade(subscores={"a": 1.0}, weights={"a": 1.0})
    assert normalize_compute_score_return(g) is g
