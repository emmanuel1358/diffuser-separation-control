"""Normalize whatever `compute_score()` returned into a canonical `Grade`.

Three accepted return shapes (see plan + `docs/GRADING.md`):

  1. `float`                                          -> bare-float score (ML_Envs)
  2. `dict {score, subscores?, weights?, metadata?}`  -> headline `score` is authoritative
  3. `Grade` (e.g. from `RubricBuilder.grade()`)      -> pass through

This is the ONLY place the runtime branches on return shape. Both the
Boreal rubric MCP server (`runtime/rubric/server.py`) and the Harbor
`run_grader.py` call into here.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from grading.grade import Grade

logger = logging.getLogger(__name__)


def normalize_compute_score_return(value: Any) -> Grade:
    """Coerce a `compute_score()` return value into a `Grade`.

    Raises ``TypeError`` only if the value is genuinely unrecognizable.
    """
    if isinstance(value, Grade):
        # Defense-in-depth: Grade.score()/weighted_total() already reject
        # non-finite via grade._finite_clamp01, but validate the emitted headline
        # here too so NO return shape can launder a NaN/inf score into the payload
        # (mirrors the float/dict paths' _finite_float guard). Raises ValueError
        # if a future code path ever bypasses the Grade-level clamp.
        _finite_float(value.score(), field="grade score")
        return value

    if isinstance(value, (int, float)):
        score = _finite_score(value, field="score")
        return Grade(
            subscores={"score": score},
            weights={"score": 1.0},
            scoring_mode="weighted",
            headline_score_override=score,
            metadata={"return_shape": "bare_float"},
        )

    if isinstance(value, dict):
        return _grade_from_dict(value)

    raise TypeError(
        "compute_score() must return a float in [0, 1], a dict with at least "
        f"a 'score' key, or a `Grade` (got {type(value).__name__})."
    )


def _grade_from_dict(payload: dict[str, Any]) -> Grade:
    """Build a Grade from a dict return value.

    Two accepted shapes:

      1. ``{score, subscores, weights, metadata}`` — the
         "continuous_with_breakdown" hand-rolled shape (e.g. ML_Envs-style
         where the headline is an anchor-mapped float and the subscores
         are diagnostic). Preserved verbatim — `headline_score_override`
         pins the float so `Grade.score()` does not recompute.
      2. A previously-serialized ``Grade.to_dict()`` dict — recognized
         by the presence of ``structured_subscores`` (the per-criterion
         array). We reconstruct the underlying per-criterion subscores
         from there so the round-trip is lossless. Authors who do
         ``return rb.grade().to_dict()`` hit this path.
    """
    if "score" not in payload:
        raise TypeError(
            "compute_score() returned a dict without a 'score' key. "
            "Either include {'score': <float>} or return a `Grade` directly."
        )

    try:
        headline = _finite_score(payload["score"], field="dict.score")
    except TypeError as exc:
        raise TypeError(
            f"compute_score() returned dict.score that is not a float: "
            f"{payload['score']!r}"
        ) from exc

    # Case 2: a previously-serialized Grade.to_dict() — rebuild from
    # structured_subscores + rubric_breakdown so per-criterion data
    # survives the round-trip.
    structured = payload.get("structured_subscores")
    if isinstance(structured, list) and structured:
        return _grade_from_serialized_dict(payload, structured, headline)

    # Case 1: hand-rolled dict (ML_Envs migration target)
    raw_subscores = payload.get("subscores") or {}
    raw_weights = payload.get("weights") or {}
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("return_shape", "score_dict")
    if "subscores" in payload and not raw_weights:
        n = len(raw_subscores) or 1
        raw_weights = {k: 1.0 / n for k in raw_subscores}

    subscores = {
        k: _finite_float(v, field=f"subscores[{k!r}]") for k, v in raw_subscores.items()
    }
    weights = {
        k: _finite_float(v, field=f"weights[{k!r}]") for k, v in raw_weights.items()
    }

    return Grade(
        subscores=subscores,
        weights=weights,
        scoring_mode="weighted",
        headline_score_override=headline,
        metadata=metadata,
        env_internal_failure=payload.get("env_internal_failure"),
        env_internal_failure_logs=payload.get("env_internal_failure_logs"),
    )


def _grade_from_serialized_dict(
    payload: dict[str, Any],
    structured: list[dict[str, Any]],
    headline: float,
) -> Grade:
    """Reconstruct a Grade from a `Grade.to_dict()` round-trip.

    Pulls per-criterion `name`/`score`/`weight` from `structured_subscores`,
    re-hydrates `criterion_logs` from `metadata.rubric_breakdown`, and
    keeps the rest of `metadata` intact. The headline is pinned via
    `headline_score_override` so this is purely lossless.
    """
    subscores: dict[str, float] = {}
    weights: dict[str, float] = {}
    for entry in structured:
        if not isinstance(entry, dict):
            continue
        name = entry.get("criterion_id") or entry.get("id") or entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        subscores[name] = _finite_score(
            entry.get("score", 0.0), field=f"structured_subscores[{name!r}].score"
        )
        weights[name] = _finite_float(
            entry.get("weight", 0.0), field=f"structured_subscores[{name!r}].weight"
        )

    nested_meta = dict(payload.get("metadata") or {})
    breakdown = nested_meta.get("rubric_breakdown") or []

    criterion_logs: dict[str, dict[str, Any]] = {}
    for entry in breakdown:
        if not isinstance(entry, dict):
            continue
        cname = entry.get("criterion_id") or entry.get("id") or entry.get("criterion")
        if not isinstance(cname, str) or not cname:
            continue
        log = {
            k: v
            for k, v in entry.items()
            if k not in {"criterion", "id", "criterion_id"}
        }
        criterion_logs[cname] = log

    # Strip computed fields (Grade.to_dict adds them deterministically;
    # leaving them in would let the round-trip drift.)
    for k in (
        "scoring_mode",
        "reported_final_score",
        "pass_threshold",
        "weighted_total",
        "weighted_subscore_total",
        "headline_score",
        "rubric_weights",
        "rubric_breakdown",
        "grading_errors",
        "serialized_grade",
    ):
        nested_meta.pop(k, None)
    nested_meta.setdefault("return_shape", "rubric_grade")

    return Grade(
        subscores=subscores,
        weights=weights,
        scoring_mode=str(payload.get("scoring_mode", "weighted")),
        penalties=payload.get("penalties"),
        criterion_logs=criterion_logs or None,
        headline_score_override=headline,
        metadata=nested_meta,
        env_internal_failure=payload.get("env_internal_failure"),
        env_internal_failure_logs=payload.get("env_internal_failure_logs"),
    )


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"compute_score returned {field} that is not a float: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"compute_score returned non-finite {field}: {number!r}")
    return number


def _finite_score(value: Any, *, field: str) -> float:
    return max(0.0, min(1.0, _finite_float(value, field=field)))


__all__ = ["normalize_compute_score_return"]
