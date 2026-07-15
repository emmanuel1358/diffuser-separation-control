"""`Grade` dataclass + canonical `to_dict()` schema consumed by Boreal and Harbor.

Vendored (and trimmed) from the worldsim reference implementation at
``worldsim/worldsim-api/wsapi/_grading_vendored.py``. The shape this emits
is what both Boreal (`Grade.metadata` packed into the MCP `Grade` reply) and
Harbor (`reward-details.json`) ingest.

Critical correctness note: when `headline_score_override` is set, `score()`
returns it verbatim instead of recomputing the weighted total. This is how
ML_Envs-style continuous-with-breakdown graders (where the headline is e.g.
an exponential-anchored aggregate that does NOT equal a weighted average of
per-target subscores) survive the round-trip without their headline being
silently rewritten. See `normalize_compute_score_return`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

PASS_THRESHOLD = 0.5


def _finite_clamp01(value: float) -> float:
    """Clamp ``value`` into ``[0, 1]``, mapping any non-finite value to ``0.0``.

    The bare ``max(0.0, min(1.0, x))`` idiom is NaN-porous: ``nan < 1.0`` is
    False, so ``min(1.0, nan)`` returns 1.0 and ``max(0.0, 1.0)`` returns 1.0 --
    a NaN (or +inf) is laundered into a perfect 1.0. An agent that makes a metric
    return NaN (e.g. the Pearson r of a constant prediction vector) would
    otherwise score full marks. Reject non-finite to 0.0 (worst) BEFORE clamping.
    This is the single nan-rejecting clamp shared by the score-mapping sites
    (``Grade``, ``calibration``, ``rubric_builder``).
    """
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


@dataclass
class Grade:
    """Grader output. Use ``to_dict()`` to emit the canonical schema."""

    subscores: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    scoring_mode: str = "weighted"  # "weighted" or "binary"
    metadata: dict[str, Any] | None = None
    penalties: dict[str, float] | None = None
    criterion_logs: dict[str, dict[str, Any]] | None = None
    env_internal_failure: bool | None = None
    env_internal_failure_logs: list[str] | tuple[str, ...] | None = None
    # When set, `score()` returns this float verbatim instead of recomputing
    # from `subscores * weights`. Used by `continuous_with_breakdown` returns
    # to preserve ML_Envs anchor-mapping math.
    headline_score_override: float | None = None

    # ── Score math ────────────────────────────────────────────────

    def weighted_total(self) -> float:
        """Sum(subscore * weight) - sum(|penalty|), clamped to [0, 1]."""
        total = sum(
            self.subscores[k] * self.weights.get(k, 0.0) for k in self.subscores
        )
        if self.penalties:
            total -= sum(abs(v) for v in self.penalties.values())
        return _finite_clamp01(total)

    def score(self) -> float:
        """Final reported score in ``[0, 1]``.

        Precedence:

          1. ``headline_score_override`` if set (clamped, but otherwise
             passed through verbatim — this preserves ML_Envs anchor maps).
          2. Binary mode: ``1.0`` iff every subscore ``>= PASS_THRESHOLD``
             AND no penalty triggered. Otherwise ``0.0``.
          3. Weighted mode: ``weighted_total()``.
        """
        if self.headline_score_override is not None:
            return _finite_clamp01(float(self.headline_score_override))
        if self.scoring_mode == "binary":
            if not self.subscores:
                return 0.0
            # A non-finite subscore must NOT pass: `nan < PASS_THRESHOLD` is
            # False, so without this guard a NaN subscore slips past the
            # threshold check below and scores a perfect 1.0.
            if any(not math.isfinite(s) for s in self.subscores.values()):
                return 0.0
            if any(s < PASS_THRESHOLD for s in self.subscores.values()):
                return 0.0
            if self.penalties:
                return 0.0
            return 1.0
        return self.weighted_total()

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Single canonical serialize. Schema:

        Top-level:
          * ``score``                   -- float, the headline
          * ``subscores``               -- criterion display labels mapped to
                                          criterion scores
          * ``weights``                 -- criterion display labels mapped to
                                          normalized criterion weights
          * ``structured_subscores``    -- list of per-criterion dicts
                                          ``{name, score, max_score, weight,
                                          reasoning, grading_criteria}``
                                          (drives per-criterion rows in UIs)
          * ``scoring_mode``            -- ``"weighted"`` | ``"binary"``
          * ``penalties``               -- ``{name: float}`` or ``None``
          * ``metadata``:
              - ``rubric_breakdown``      -- list of richer per-criterion
                                            entries (grading_type, expected,
                                            actual, error_*, ...)
              - ``rubric_weights``        -- normalized weight map
              - ``weighted_total``        -- raw aggregate before binary collapse
              - ``reported_final_score``  -- mirror of top-level ``score``
              - ``scoring_mode``          -- mirror
              - ``grading_errors``        -- list (only present if any criterion errored)
        """
        final_score = self.score()
        weighted = self.weighted_total()
        structured = self._structured_subscores()
        breakdown = self._rubric_breakdown()
        grading_errors = self._collect_grading_errors()

        # Caller-supplied metadata is layered FIRST so reserved computed
        # fields authoritatively override anything the author also set.
        meta: dict[str, Any] = dict(self.metadata) if self.metadata else {}
        label_subscores, label_weights = self._labeled_subscores_and_weights()
        meta.update(
            {
                "scoring_mode": self.scoring_mode,
                "reported_final_score": final_score,
                "headline_score": final_score,
                "weighted_total": weighted,
                "weighted_subscore_total": weighted,
                "return_shape": "rubric_grade",
                "rubric_weights": label_weights,
                "rubric_breakdown": breakdown,
            }
        )
        if grading_errors:
            meta["grading_errors"] = grading_errors
        else:
            meta.pop("grading_errors", None)
        if self.env_internal_failure is not None:
            meta["env_internal_failure"] = self.env_internal_failure
        if self.env_internal_failure_logs:
            meta["env_internal_failure_logs"] = list(self.env_internal_failure_logs)

        result = {
            "score": final_score,
            "subscores": label_subscores or {"score": final_score},
            "weights": label_weights or {"score": 1.0},
            "structured_subscores": structured,
            "scoring_mode": self.scoring_mode,
            "penalties": self.penalties,
            "metadata": meta,
        }
        if self.env_internal_failure is not None:
            result["env_internal_failure"] = self.env_internal_failure
        if self.env_internal_failure_logs:
            result["env_internal_failure_logs"] = list(self.env_internal_failure_logs)
        meta["serialized_grade"] = {
            key: value for key, value in result.items() if key != "metadata"
        }
        return result

    # ── Helpers ──────────────────────────────────────────────────

    def _labeled_subscores_and_weights(
        self,
    ) -> tuple[dict[str, float], dict[str, float]]:
        labels = self._criterion_labels()
        return (
            {
                labels[criterion]: float(score)
                for criterion, score in self.subscores.items()
            },
            {
                labels[criterion]: float(self.weights.get(criterion, 0.0))
                for criterion in self.subscores
            },
        )

    def _criterion_labels(self) -> dict[str, str]:
        logs = self.criterion_logs or {}
        base_labels: dict[str, str] = {}
        counts: dict[str, int] = {}
        for criterion in self.subscores:
            log = logs.get(criterion) or {}
            raw_label = (
                log.get("description")
                or log.get("label")
                or log.get("criterion")
                or criterion
            )
            label = str(raw_label).strip() or criterion
            base_labels[criterion] = label
            counts[label] = counts.get(label, 0) + 1

        used: set[str] = set()
        labels: dict[str, str] = {}
        for criterion, base in base_labels.items():
            if counts[base] <= 1 and base not in used:
                label = base
            else:
                id_suffix = f"{base} [{criterion}]"
                if id_suffix not in used:
                    label = id_suffix
                else:
                    idx = 2
                    label = f"{base} ({idx})"
                    while label in used:
                        idx += 1
                        label = f"{base} ({idx})"
            used.add(label)
            labels[criterion] = label
        return labels

    def _structured_subscores(self) -> list[dict[str, Any]]:
        logs = self.criterion_logs or {}
        labels = self._criterion_labels()
        out: list[dict[str, Any]] = []
        for criterion, score in self.subscores.items():
            log = logs.get(criterion) or {}
            weight = float(self.weights.get(criterion, 0.0))
            label = labels[criterion]
            description = str(
                log.get("description") or log.get("criterion") or label
            ).strip()
            error_type = log.get("error_type")
            if error_type:
                msg = log.get("error_message") or log.get("reasoning") or ""
                reasoning = f"[{error_type}] {msg}".strip()
            else:
                reasoning = log.get("reasoning") or ""
            out.append(
                {
                    "name": label,
                    "label": label,
                    "id": criterion,
                    "criterion_id": criterion,
                    "description": description,
                    "score": float(score),
                    "max_score": 1.0,
                    "weight": weight,
                    "reasoning": reasoning,
                    "grading_criteria": description or criterion,
                }
            )
        return out

    def _rubric_breakdown(self) -> list[dict[str, Any]]:
        logs = self.criterion_logs or {}
        labels = self._criterion_labels()
        out: list[dict[str, Any]] = []
        for criterion, score in self.subscores.items():
            log = logs.get(criterion) or {}
            label = labels[criterion]
            description = str(
                log.get("description") or log.get("criterion") or label
            ).strip()
            entry: dict[str, Any] = {
                "criterion": criterion,
                "id": criterion,
                "criterion_id": criterion,
                "label": label,
                "description": description,
                "score": float(score),
                "weight": float(self.weights.get(criterion, 0.0)),
                "passed": float(score) >= PASS_THRESHOLD,
                "grading_type": log.get("grading_type", "unknown"),
                "reasoning": log.get("reasoning", ""),
                "expected": log.get("expected"),
                "actual": log.get("actual"),
            }
            for k in ("error_type", "error_status", "error_body", "error_message"):
                if log.get(k) is not None:
                    entry[k] = log[k]
            out.append(entry)
        return out

    def _collect_grading_errors(self) -> list[dict[str, Any]]:
        logs = self.criterion_logs or {}
        out: list[dict[str, Any]] = []
        for criterion, log in logs.items():
            if not isinstance(log, dict):
                continue
            err_type = log.get("error_type")
            if not err_type:
                continue
            out.append(
                {
                    "criterion": criterion,
                    "error_type": err_type,
                    "error_message": log.get("error_message")
                    or log.get("reasoning")
                    or "",
                    "error_status": log.get("error_status"),
                }
            )
        return out
