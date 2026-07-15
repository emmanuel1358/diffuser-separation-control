"""Tests for the canonical Grade dataclass + to_dict() schema."""

from __future__ import annotations

from grading.grade import PASS_THRESHOLD, Grade, _finite_clamp01


def test_weighted_total_is_clamped() -> None:
    g = Grade(subscores={"a": 1.0, "b": 1.0}, weights={"a": 0.6, "b": 0.6})
    # Raw sum is 1.2; should clamp to 1.0
    assert g.weighted_total() == 1.0


def test_weighted_total_subtracts_penalties() -> None:
    g = Grade(
        subscores={"a": 1.0},
        weights={"a": 1.0},
        penalties={"bad": -0.3},
    )
    assert g.weighted_total() == 0.7


def test_score_uses_headline_override_verbatim() -> None:
    """Critical: ML_Envs anchor mappings must NOT be recomputed."""
    g = Grade(
        subscores={"rmse": 0.9, "f1": 0.3},  # weighted_avg(0.9*0.5, 0.3*0.5) = 0.6
        weights={"rmse": 0.5, "f1": 0.5},
        headline_score_override=0.42,  # author's anchor mapping
    )
    assert g.score() == 0.42
    assert g.weighted_total() == 0.6  # weighted_total still computes raw


def test_score_binary_collapses_when_any_subscore_below_threshold() -> None:
    g = Grade(
        subscores={"a": 0.9, "b": 0.4},
        weights={"a": 0.5, "b": 0.5},
        scoring_mode="binary",
    )
    assert g.score() == 0.0  # b < PASS_THRESHOLD


def test_score_binary_returns_one_when_all_pass_and_no_penalty() -> None:
    g = Grade(
        subscores={"a": 0.7, "b": PASS_THRESHOLD},
        weights={"a": 0.5, "b": 0.5},
        scoring_mode="binary",
    )
    assert g.score() == 1.0


def test_score_binary_returns_zero_when_any_penalty_triggered() -> None:
    g = Grade(
        subscores={"a": 0.9},
        weights={"a": 1.0},
        scoring_mode="binary",
        penalties={"x": -0.1},
    )
    assert g.score() == 0.0


def test_to_dict_emits_taiga_canonical_shape() -> None:
    g = Grade(
        subscores={"a": 1.0, "b": 0.5},
        weights={"a": 0.5, "b": 0.5},
        criterion_logs={
            "a": {
                "reasoning": "passed cleanly",
                "grading_type": "deterministic",
                "criterion_id": "a",
                "description": "Readable A",
            },
            "b": {
                "reasoning": "partial",
                "grading_type": "llm",
                "criterion_id": "b",
                "description": "Readable B",
            },
        },
    )
    d = g.to_dict()

    # Top-level shape Boreal consumes
    assert set(d) == {
        "score",
        "subscores",
        "weights",
        "structured_subscores",
        "scoring_mode",
        "penalties",
        "metadata",
    }
    assert d["subscores"] == {"Readable A": 1.0, "Readable B": 0.5}
    assert d["weights"] == {"Readable A": 0.5, "Readable B": 0.5}
    assert len(d["structured_subscores"]) == 2

    sub_a = next(s for s in d["structured_subscores"] if s["id"] == "a")
    assert sub_a["name"] == "Readable A"
    assert sub_a["label"] == "Readable A"
    assert sub_a["description"] == "Readable A"
    assert sub_a["weight"] == 0.5
    assert sub_a["reasoning"] == "passed cleanly"

    # Computed metadata fields
    md = d["metadata"]
    assert md["scoring_mode"] == "weighted"
    assert "pass_threshold" not in md
    assert md["weighted_total"] == g.weighted_total()
    assert md["weighted_subscore_total"] == g.weighted_total()
    assert md["reported_final_score"] == d["score"]
    assert md["headline_score"] == d["score"]
    assert md["return_shape"] == "rubric_grade"
    assert md["rubric_weights"] == d["weights"]
    assert md["serialized_grade"]["subscores"] == d["subscores"]
    assert "pass_threshold" not in md["serialized_grade"]
    assert len(md["rubric_breakdown"]) == 2
    breakdown_a = next(s for s in md["rubric_breakdown"] if s["id"] == "a")
    assert breakdown_a["label"] == "Readable A"
    assert breakdown_a["description"] == "Readable A"


def test_duplicate_descriptions_are_deduped_with_ids() -> None:
    g = Grade(
        subscores={"a": 1.0, "b": 0.0},
        weights={"a": 0.5, "b": 0.5},
        criterion_logs={
            "a": {"description": "Same display name"},
            "b": {"description": "Same display name"},
        },
    )

    d = g.to_dict()

    assert d["subscores"] == {
        "Same display name [a]": 1.0,
        "Same display name [b]": 0.0,
    }
    assert d["structured_subscores"][0]["id"] == "a"
    assert d["structured_subscores"][1]["id"] == "b"


def test_description_takes_precedence_over_legacy_id_label() -> None:
    g = Grade(
        subscores={"single_dof": 1.0},
        weights={"single_dof": 1.0},
        criterion_logs={
            "single_dof": {
                "label": "single_dof",
                "description": "Exactly one degree of freedom (nv == 1)",
            }
        },
    )

    d = g.to_dict()

    assert d["subscores"] == {"Exactly one degree of freedom (nv == 1)": 1.0}
    assert d["weights"] == {"Exactly one degree of freedom (nv == 1)": 1.0}
    assert d["metadata"]["rubric_weights"] == d["weights"]
    assert d["structured_subscores"][0]["name"] == (
        "Exactly one degree of freedom (nv == 1)"
    )
    assert d["structured_subscores"][0]["label"] == (
        "Exactly one degree of freedom (nv == 1)"
    )
    assert d["structured_subscores"][0]["criterion_id"] == "single_dof"
    assert d["metadata"]["rubric_breakdown"][0]["criterion_id"] == "single_dof"
    assert d["metadata"]["rubric_breakdown"][0]["label"] == (
        "Exactly one degree of freedom (nv == 1)"
    )


def test_com_length_target_uses_description_keys_and_preserves_id() -> None:
    description = "Center-of-mass 0.5 m from hinge axis (+/-0.05)"
    g = Grade(
        subscores={"com_length_target": 1.0},
        weights={"com_length_target": 1.0},
        criterion_logs={
            "com_length_target": {
                "label": "com_length_target",
                "description": description,
                "criterion_id": "com_length_target",
            }
        },
    )

    d = g.to_dict()

    assert d["subscores"] == {description: 1.0}
    assert d["weights"] == {description: 1.0}
    assert d["metadata"]["rubric_weights"] == {description: 1.0}
    assert d["metadata"]["serialized_grade"]["subscores"] == {description: 1.0}
    assert d["structured_subscores"][0]["name"] == description
    assert d["structured_subscores"][0]["label"] == description
    assert d["structured_subscores"][0]["id"] == "com_length_target"
    assert d["structured_subscores"][0]["criterion_id"] == "com_length_target"


def test_structured_subscores_surface_judge_errors_as_reasoning_prefix() -> None:
    g = Grade(
        subscores={"q": 0.0},
        weights={"q": 1.0},
        criterion_logs={
            "q": {
                "error_type": "missing_api_key",
                "error_message": "no key",
                "reasoning": "should not be used",
            }
        },
    )
    sub = g.to_dict()["structured_subscores"][0]
    assert sub["reasoning"].startswith("[missing_api_key]")
    assert "no key" in sub["reasoning"]


def test_metadata_grading_errors_collected_only_when_present() -> None:
    g_clean = Grade(
        subscores={"a": 1.0},
        weights={"a": 1.0},
        criterion_logs={"a": {"reasoning": "ok"}},
    )
    assert "grading_errors" not in g_clean.to_dict()["metadata"]

    g_dirty = Grade(
        subscores={"a": 0.0},
        weights={"a": 1.0},
        criterion_logs={"a": {"error_type": "x", "error_message": "boom"}},
    )
    md = g_dirty.to_dict()["metadata"]
    assert "grading_errors" in md
    assert md["grading_errors"][0]["criterion"] == "a"


def test_env_internal_failure_surfaces_when_set() -> None:
    g = Grade(
        subscores={"score": 0.0},
        weights={"score": 1.0},
        env_internal_failure=True,
        env_internal_failure_logs=["OOM"],
    )

    d = g.to_dict()

    assert d["env_internal_failure"] is True
    assert d["env_internal_failure_logs"] == ["OOM"]
    assert d["metadata"]["env_internal_failure"] is True
    assert d["metadata"]["env_internal_failure_logs"] == ["OOM"]


def test_finite_clamp01_rejects_non_finite() -> None:
    # The NaN-porous max(0, min(1, x)) idiom returns 1.0 for NaN/+inf; the shared
    # clamp must map every non-finite value to 0.0 (worst) before clamping.
    assert _finite_clamp01(float("nan")) == 0.0
    assert _finite_clamp01(float("inf")) == 0.0
    assert _finite_clamp01(float("-inf")) == 0.0
    assert _finite_clamp01(0.7) == 0.7
    assert _finite_clamp01(2.0) == 1.0
    assert _finite_clamp01(-1.0) == 0.0


def test_headline_override_non_finite_scores_zero() -> None:
    # An agent-driven NaN headline (e.g. a correlation of a constant prediction)
    # must score 0.0, not be laundered to a perfect 1.0.
    assert Grade(headline_score_override=float("nan")).score() == 0.0
    assert Grade(headline_score_override=float("inf")).score() == 0.0


def test_binary_mode_non_finite_subscore_scores_zero() -> None:
    # `nan < PASS_THRESHOLD` is False, so without the finite guard a NaN subscore
    # would slip past the threshold check and score a perfect 1.0.
    g = Grade(scoring_mode="binary", subscores={"a": float("nan")})
    assert g.score() == 0.0


def test_weighted_total_non_finite_subscore_scores_zero() -> None:
    g = Grade(subscores={"a": float("nan")}, weights={"a": 1.0})
    assert g.weighted_total() == 0.0
    assert g.score() == 0.0
