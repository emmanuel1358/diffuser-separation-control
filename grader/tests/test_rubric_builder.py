"""Tests for `RubricBuilder` decorator API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from grading import RubricBuilder
from grading.judge import LLMJudge, JudgeResult


def test_criterion_decorator_registers_and_grades(workspace: Path) -> None:
    rb = RubricBuilder(workspace=workspace, trajectory=None, private=workspace)

    @rb.criterion(id="always_pass", weight=1.0)
    def _():
        return True

    @rb.criterion(id="always_fail", weight=1.0)
    def _():
        return False

    g = rb.grade()
    assert g.subscores == {"always_pass": 1.0, "always_fail": 0.0}
    # Equal weights normalize to 0.5/0.5
    assert g.weights == {"always_pass": 0.5, "always_fail": 0.5}
    assert g.score() == 0.5


def test_criterion_description_becomes_serialized_label(workspace: Path) -> None:
    rb = RubricBuilder(workspace=workspace, trajectory=None, private=workspace)

    @rb.criterion(
        id="always_pass",
        weight=1.0,
        description="Human-readable criterion label",
    )
    def _():
        return True

    d = rb.grade().to_dict()

    assert d["subscores"] == {"Human-readable criterion label": 1.0}
    assert d["weights"] == {"Human-readable criterion label": 1.0}
    assert d["structured_subscores"][0]["id"] == "always_pass"
    assert d["structured_subscores"][0]["description"] == (
        "Human-readable criterion label"
    )
    assert d["metadata"]["rubric_breakdown"][0]["criterion_id"] == "always_pass"


def test_criterion_async_predicate(workspace: Path) -> None:
    rb = RubricBuilder(workspace=workspace, trajectory=None, private=workspace)

    @rb.criterion(id="async_one", weight=1.0)
    async def _():
        await asyncio.sleep(0)
        return 0.75

    g = rb.grade()
    assert g.subscores["async_one"] == 0.75


def test_criterion_float_returns_are_clamped(workspace: Path) -> None:
    rb = RubricBuilder(workspace=workspace, trajectory=None, private=workspace)

    @rb.criterion(id="overflow", weight=1.0)
    def _():
        return 1.7

    @rb.criterion(id="negative", weight=1.0)
    def _():
        return -0.4

    g = rb.grade()
    assert g.subscores == {"overflow": 1.0, "negative": 0.0}


def test_criterion_predicate_exception_logs_and_returns_zero(workspace: Path) -> None:
    rb = RubricBuilder(workspace=workspace, trajectory=None, private=workspace)

    @rb.criterion(id="boom", weight=1.0)
    def _():
        raise RuntimeError("kaboom")

    g = rb.grade()
    assert g.subscores["boom"] == 0.0
    log = g.criterion_logs["boom"]
    assert log["error_type"] == "predicate_exception"
    assert "kaboom" in log["error_message"]


def test_duplicate_id_raises(workspace: Path) -> None:
    rb = RubricBuilder(workspace=workspace, trajectory=None, private=workspace)

    @rb.criterion(id="dup", weight=1.0)
    def _():
        return True

    with pytest.raises(ValueError, match="duplicate criterion/penalty id"):

        @rb.criterion(id="dup", weight=1.0)
        def _():
            return True


def test_zero_or_negative_weight_raises(workspace: Path) -> None:
    rb = RubricBuilder(workspace=workspace, trajectory=None, private=workspace)
    with pytest.raises(ValueError, match="weight must be > 0"):

        @rb.criterion(id="bad", weight=0)
        def _():
            return True


def test_penalty_value_must_be_non_positive(workspace: Path) -> None:
    rb = RubricBuilder(workspace=workspace, trajectory=None, private=workspace)
    with pytest.raises(ValueError, match="value must be <= 0"):

        @rb.penalty(id="bad", value=0.5)
        def _():
            return True


def test_penalty_triggers_only_when_predicate_truthy(workspace: Path) -> None:
    rb = RubricBuilder(workspace=workspace, trajectory=None, private=workspace)

    @rb.criterion(id="ok", weight=1.0)
    def _():
        return 1.0

    @rb.penalty(id="trigger_me", value=-0.4)
    def _():
        return True

    @rb.penalty(id="dont_trigger", value=-0.4)
    def _():
        return False

    g = rb.grade()
    assert g.penalties == {"trigger_me": -0.4}
    # weighted_total = 1.0 - |-0.4| = 0.6
    assert g.score() == pytest.approx(0.6)


def test_binary_scoring_collapses_when_any_subscore_below_threshold(
    workspace: Path,
) -> None:
    rb = RubricBuilder(
        workspace=workspace,
        trajectory=None,
        private=workspace,
        binary_scoring=True,
    )

    @rb.criterion(id="a", weight=1.0)
    def _():
        return 0.7

    @rb.criterion(id="b", weight=1.0)
    def _():
        return 0.4  # below PASS_THRESHOLD

    g = rb.grade()
    assert g.score() == 0.0


def test_llm_criterion_skipped_with_missing_api_key_logs_error(
    workspace: Path, monkeypatch
) -> None:
    # LLMJudge(api_key="") falls back to the ANTHROPIC_API_KEY env var, so the
    # test must clear it to stay hermetic (a configured judge would make a real
    # API call here instead of logging missing_api_key).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rb = RubricBuilder(
        workspace=workspace,
        trajectory=None,
        private=workspace,
        judge=LLMJudge(api_key=""),
    )
    rb.llm_criterion(
        id="judge_me",
        weight=1.0,
        prompt="Is this good?",
    )
    g = rb.grade()
    assert g.subscores["judge_me"] == 0.0
    log = g.criterion_logs["judge_me"]
    assert log["error_type"] == "missing_api_key"


def test_llm_criterion_uses_injected_judge(workspace: Path) -> None:
    """A passing judge result drives the criterion's score."""
    judge = LLMJudge(api_key="sk-fake")
    judge.judge = AsyncMock(  # type: ignore[method-assign]
        return_value=JudgeResult(score=1.0, explanation="looks great")
    )
    rb = RubricBuilder(
        workspace=workspace,
        trajectory=None,
        private=workspace,
        judge=judge,
    )
    rb.transcript = "Human: ...\n\nAssistant: my answer"
    rb.llm_criterion(
        id="quality",
        weight=1.0,
        prompt="Is this good?",
        scope="last_turn",
    )
    g = rb.grade()
    assert g.subscores["quality"] == 1.0
    assert g.criterion_logs["quality"]["reasoning"] == "looks great"


def test_empty_rubric_returns_zero_grade_with_warning(workspace: Path, caplog) -> None:
    rb = RubricBuilder(workspace=workspace, trajectory=None, private=workspace)
    g = rb.grade()
    assert g.score() == 0.0


def test_weights_normalize_to_sum_one(workspace: Path) -> None:
    rb = RubricBuilder(workspace=workspace, trajectory=None, private=workspace)

    @rb.criterion(id="a", weight=2.0)
    def _():
        return 1.0

    @rb.criterion(id="b", weight=3.0)
    def _():
        return 0.0

    g = rb.grade()
    assert g.weights["a"] == pytest.approx(2 / 5)
    assert g.weights["b"] == pytest.approx(3 / 5)
    assert sum(g.weights.values()) == pytest.approx(1.0)


def test_coerce_predicate_value_rejects_non_finite() -> None:
    # A predicate returning NaN / +inf (e.g. pearsonr of a constant prediction)
    # must map to 0.0, not the 1.0 the bare max(0, min(1, x)) idiom returns.
    from grading.rubric_builder import _coerce_predicate_value

    assert _coerce_predicate_value(float("nan")) == 0.0
    assert _coerce_predicate_value(float("inf")) == 0.0
    assert _coerce_predicate_value(float("-inf")) == 0.0
    assert _coerce_predicate_value(0.5) == 0.5
    assert _coerce_predicate_value(True) == 1.0
