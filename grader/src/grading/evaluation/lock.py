"""Canonical generated calibration locks for continuous tasks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from grading.calibration import PiecewiseLinearCurve
from grading.evaluation.metrics import (
    RATIONALE_KINDS,
    MetricTarget,
    effective_floor,
    no_info_ceiling,
    normalize_weights,
    validate_metric_vector,
)

CALIBRATION_LOCK_SCHEMA = "3.1"
LEGACY_CALIBRATION_LOCK_SCHEMA = "3.0"
CALIBRATION_POLICY = "continuous-pwl-v3"
DEFAULT_LOCK_FILENAME = "calibration.lock.json"
CALIBRATION_LOCK_PATH_ENV = "LBX_CALIBRATION_LOCK_PATH"
RUNTIME_LOCK_ROOT = Path("/mcp_server/calibration")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Deterministic, reviewable JSON encoding used for locks and digests."""

    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class CalibrationLock:
    payload: dict[str, Any]
    sha256: str

    @property
    def targets(self) -> dict[str, dict[str, Any]]:
        return dict(self.payload["targets"])

    @property
    def x_ref(self) -> float:
        return float(self.payload["curve"]["x_ref"])

    @property
    def reference_metrics(self) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in self.payload["measurements"]["reference"].items()
        }

    @property
    def naive_metrics(self) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in self.payload["measurements"]["naive"].items()
        }

    @property
    def degenerate_metrics(self) -> dict[str, dict[str, float]]:
        return {
            str(strategy): {str(key): float(value) for key, value in values.items()}
            for strategy, values in self.payload["measurements"]["degenerate"].items()
        }

    @property
    def raw_floors(self) -> dict[str, float]:
        return {
            str(name): float(spec["raw_floor"])
            for name, spec in self.payload["targets"].items()
        }

    @property
    def no_info_ceilings(self) -> dict[str, float]:
        return {
            str(name): float(spec["no_info_ceiling"])
            for name, spec in self.payload["targets"].items()
        }


def _aggregate_progress(
    targets: tuple[MetricTarget, ...],
    metrics: Mapping[str, Any],
) -> float:
    finite = validate_metric_vector(targets, metrics)
    weights = normalize_weights(targets)
    return sum(
        weights[target.name] * target.progress(finite[target.name])
        for target in targets
    )


def _validate_naive_at_floor_rationale(
    rationale: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    if rationale is None:
        return None
    if (
        rationale.get("kind") != "reviewed_exception"
        or not isinstance(rationale.get("summary"), str)
        or len(str(rationale["summary"]).strip()) < 20
    ):
        raise ValueError(
            "inclusive naive floor requires a reviewed_exception rationale "
            "with a summary of at least 20 characters"
        )
    normalized = {
        "kind": "reviewed_exception",
        "summary": str(rationale["summary"]).strip(),
    }
    source = rationale.get("source")
    if source is not None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("inclusive naive floor rationale source must be non-empty")
        normalized["source"] = source.strip()
    return normalized


def _naive_ties_effective_floors(
    targets: tuple[MetricTarget, ...],
    ceilings: Mapping[str, float],
    naive: Mapping[str, float],
) -> bool:
    return all(
        math.isclose(
            float(naive[target.name]),
            effective_floor(target, float(ceilings[target.name])),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for target in targets
    )


def build_calibration_lock(
    *,
    task_spec_sha256: str,
    evaluation_plan_sha256: str,
    evaluation_plan: Mapping[str, Any],
    targets: tuple[MetricTarget, ...],
    reference_metrics: Mapping[str, Any],
    naive_metrics: Mapping[str, Any],
    degenerate_metrics: Mapping[str, Mapping[str, Any]],
    input_digests: Mapping[str, str],
    naive_score_min: float = 1e-6,
    naive_score_max: float = 0.10,
    naive_at_floor: Mapping[str, Any] | None = None,
    max_unacknowledged_naive_score_gap: float = 0.05,
    naive_semantic_gap_acknowledgement: Mapping[str, Any] | None = None,
) -> CalibrationLock:
    """Build a canonical quality lock plus auditable no-information probes.

    The reviewed author floor remains the quality anchor. Degenerate-family
    measurements are persisted for qualification and adversarial audit, but
    information eligibility is decided independently at grade time. This
    preserves low reward for weak candidates that carry real row-level signal.
    """

    reference = validate_metric_vector(targets, reference_metrics)
    naive = validate_metric_vector(targets, naive_metrics)
    if not degenerate_metrics:
        raise ValueError("degenerate strategy metrics must be non-empty")
    degenerate = {
        str(strategy): validate_metric_vector(targets, metrics)
        for strategy, metrics in degenerate_metrics.items()
    }
    weights = normalize_weights(targets)

    ceilings = {
        target.name: no_info_ceiling(
            target,
            (degenerate[strategy][target.name] for strategy in degenerate),
        )
        for target in targets
    }
    # A no-information result at/beyond perfect means the task/metric cannot
    # discriminate skill. Fail author-time calibration instead of silently
    # clamping or turning the no-information draw into a quality anchor.
    for target in targets:
        ceiling = ceilings[target.name]
        if (
            target.direction == "higher"
            and ceiling >= target.perfect
            or target.direction == "lower"
            and ceiling <= target.perfect
        ):
            raise ValueError(
                "no-information ceiling would violate the floor-vs-perfect "
                f"invariant for target {target.name!r}: {ceiling}"
            )

    qualification_targets = tuple(
        replace(
            target,
            floor=replace(
                target.floor,
                value=effective_floor(target, ceilings[target.name]),
            ),
        )
        for target in targets
    )
    x_ref = _aggregate_progress(targets, reference)
    if not 0.0 < x_ref < 1.0:
        raise ValueError(
            "reference aggregate progress must lie strictly between floor and "
            f"perfect; got {x_ref:.12g}"
        )
    curve = PiecewiseLinearCurve.from_reference(x_ref)
    naive_quality_progress = _aggregate_progress(targets, naive)
    naive_progress = _aggregate_progress(qualification_targets, naive)
    naive_score = curve.score(naive_progress)
    naive_quality_score = curve.score(naive_quality_progress)
    naive_semantic_gap = abs(naive_quality_score - naive_score)
    if (
        not math.isfinite(max_unacknowledged_naive_score_gap)
        or not 0 <= max_unacknowledged_naive_score_gap < 0.5
    ):
        raise ValueError(
            "max_unacknowledged_naive_score_gap must be finite in [0, 0.5)"
        )
    semantic_gap_acknowledgement = _validate_naive_at_floor_rationale(
        naive_semantic_gap_acknowledgement
    )
    if (
        naive_semantic_gap > max_unacknowledged_naive_score_gap
        and semantic_gap_acknowledgement is None
    ):
        raise ValueError(
            "qualification and runtime naive scores diverge by "
            f"{naive_semantic_gap:.6g}, above the unacknowledged threshold "
            f"{max_unacknowledged_naive_score_gap:.6g}; add a reviewed "
            "naive_semantic_gap_acknowledgement"
        )
    floor_rationale = _validate_naive_at_floor_rationale(naive_at_floor)
    if floor_rationale is not None:
        lower_ok = (
            naive_score_min == 0.0
            and math.isclose(naive_score, 0.0, abs_tol=1e-12)
            and _naive_ties_effective_floors(targets, ceilings, naive)
        )
        interval = f"[{naive_score_min:.6g}, {naive_score_max:.6g}]"
    else:
        lower_ok = naive_score_min < naive_score
        interval = f"({naive_score_min:.6g}, {naive_score_max:.6g}]"
    if not (math.isfinite(naive_score) and lower_ok and naive_score <= naive_score_max):
        raise ValueError(
            "naive baseline must be weak but informative: expected score in "
            f"{interval}, got {naive_score:.6g}"
        )
    naive_score_range: dict[str, Any] = {
        (
            "inclusive_min" if floor_rationale is not None else "exclusive_min"
        ): naive_score_min,
        "inclusive_max": naive_score_max,
    }
    if floor_rationale is not None:
        naive_score_range["rationale"] = floor_rationale

    payload: dict[str, Any] = {
        "schema_version": CALIBRATION_LOCK_SCHEMA,
        "policy": CALIBRATION_POLICY,
        "task_spec_sha256": task_spec_sha256,
        "evaluation_plan_sha256": evaluation_plan_sha256,
        "evaluation_plan": dict(evaluation_plan),
        "inputs": {
            str(key): str(value) for key, value in sorted(input_digests.items())
        },
        "targets": {
            target.name: {
                **target.spec_dict(),
                "weight": weights[target.name],
                "reference": reference[target.name],
                "raw_floor": target.floor.value,
                "no_info_ceiling": ceilings[target.name],
            }
            for target in targets
        },
        "curve": {
            "type": "piecewise_linear",
            "x_ref": x_ref,
            "knots": [[0.0, 0.0], [x_ref, 0.5], [1.0, 1.0]],
        },
        "measurements": {
            "reference": reference,
            "naive": naive,
            "degenerate": degenerate,
        },
        "qualification": {
            "naive_progress": naive_progress,
            "naive_quality_progress": naive_quality_progress,
            "naive_score": naive_score,
            "qualification_naive_score": naive_score,
            "runtime_naive_quality_score": naive_quality_score,
            "naive_semantic_gap": naive_semantic_gap,
            "max_unacknowledged_naive_score_gap": (max_unacknowledged_naive_score_gap),
            "naive_semantic_gap_acknowledgement": (semantic_gap_acknowledgement),
            "naive_score_range": naive_score_range,
            "reference_score": curve.score(x_ref),
            "oracle_score": curve.score(1.0),
            "null_score": curve.score(0.0),
            "degenerate_scores": {
                strategy: curve.score(
                    _aggregate_progress(qualification_targets, strategy_metrics)
                )
                for strategy, strategy_metrics in sorted(degenerate.items())
            },
        },
    }
    return CalibrationLock(payload=payload, sha256=canonical_sha256(payload))


def _close(actual: float, expected: float, *, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"calibration lock {field} is inconsistent: "
            f"expected {expected:.17g}, got {actual:.17g}"
        )


def _progress_from_spec(spec: Mapping[str, Any], value: float) -> float:
    floor = float(spec["floor"]["value"])
    perfect = float(spec["perfect"])
    direction = str(spec["direction"])
    if direction == "lower":
        if floor <= perfect:
            raise ValueError("lower-is-better lock target requires floor > perfect")
        raw = (floor - value) / (floor - perfect)
    elif direction == "higher":
        if floor >= perfect:
            raise ValueError("higher-is-better lock target requires floor < perfect")
        raw = (value - floor) / (perfect - floor)
    else:
        raise ValueError(f"unsupported lock target direction {direction!r}")
    return max(0.0, min(1.0, raw))


def _aggregate_from_payload(
    targets: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> float:
    return sum(
        float(spec["weight"]) * _progress_from_spec(spec, float(metrics[name]))
        for name, spec in targets.items()
    )


def _validate_naive_score_range(
    qualification: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any]],
    naive: Mapping[str, Any],
    *,
    naive_score: float,
) -> None:
    score_range = qualification.get("naive_score_range")
    if not isinstance(score_range, Mapping):
        raise ValueError("calibration lock is missing qualification.naive_score_range")
    has_exclusive = "exclusive_min" in score_range
    has_inclusive = "inclusive_min" in score_range
    if has_exclusive == has_inclusive:
        raise ValueError(
            "calibration lock naive score range must declare exactly one of "
            "exclusive_min or inclusive_min"
        )
    expected_keys = (
        {"exclusive_min", "inclusive_max"}
        if has_exclusive
        else {"inclusive_min", "inclusive_max", "rationale"}
    )
    if set(score_range) != expected_keys:
        raise ValueError(
            "calibration lock naive score range has unexpected fields: "
            f"{sorted(set(score_range) - expected_keys)}"
        )
    try:
        minimum = float(
            score_range["exclusive_min" if has_exclusive else "inclusive_min"]
        )
        maximum = float(score_range["inclusive_max"])
    except (TypeError, ValueError) as exc:
        raise ValueError("calibration lock naive score range must be numeric") from exc
    if not (
        math.isfinite(minimum)
        and math.isfinite(maximum)
        and 0.0 <= minimum < maximum < 0.5
    ):
        raise ValueError(
            "calibration lock naive score range must satisfy "
            "0 <= min < max < reference score 0.5"
        )

    if has_exclusive:
        lower_ok = minimum < naive_score
    else:
        _validate_naive_at_floor_rationale(score_range.get("rationale"))
        if minimum != 0.0:
            raise ValueError(
                "inclusive naive score minimum is allowed only at the zero boundary"
            )
        lower_ok = math.isclose(naive_score, minimum, abs_tol=1e-12)
        ties = all(
            math.isclose(
                float(naive[name]),
                (
                    min(
                        float(spec["raw_floor"]),
                        float(spec["no_info_ceiling"]),
                    )
                    if str(spec["direction"]) == "lower"
                    else max(
                        float(spec["raw_floor"]),
                        float(spec["no_info_ceiling"]),
                    )
                ),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for name, spec in targets.items()
        )
        if not ties:
            raise ValueError(
                "inclusive naive floor requires every naive metric to tie its "
                "effective no-information floor"
            )
    if not (math.isfinite(naive_score) and lower_ok and naive_score <= maximum):
        bracket = "[" if has_inclusive else "("
        raise ValueError(
            "calibration lock naive score does not satisfy its qualification "
            f"range {bracket}{minimum:.6g}, {maximum:.6g}]"
        )


def validate_calibration_lock(
    payload: Mapping[str, Any],
    *,
    task_spec_sha256: str | None = None,
    compatible_task_spec_sha256s: tuple[str, ...] = (),
) -> CalibrationLock:
    schema_version = payload.get("schema_version")
    if schema_version not in {
        CALIBRATION_LOCK_SCHEMA,
        LEGACY_CALIBRATION_LOCK_SCHEMA,
    }:
        raise ValueError(
            "unsupported calibration lock schema "
            f"{schema_version!r}; expected {CALIBRATION_LOCK_SCHEMA!r}"
        )
    if payload.get("policy") != CALIBRATION_POLICY:
        raise ValueError(f"unsupported calibration policy {payload.get('policy')!r}")
    accepted_task_digests = {
        digest
        for digest in (task_spec_sha256, *compatible_task_spec_sha256s)
        if digest is not None
    }
    if accepted_task_digests and payload.get("task_spec_sha256") not in (
        accepted_task_digests
    ):
        raise ValueError(
            "calibration lock is stale for this TASK registration: expected "
            f"one of {sorted(accepted_task_digests)}, got "
            f"{payload.get('task_spec_sha256')}"
        )
    evaluation_plan = payload.get("evaluation_plan")
    evaluation_plan_sha = payload.get("evaluation_plan_sha256")
    if not isinstance(evaluation_plan, Mapping) or not isinstance(
        evaluation_plan_sha, str
    ):
        raise ValueError("calibration lock is missing evaluation plan identity")
    _close_digest = canonical_sha256(evaluation_plan)
    if _close_digest != evaluation_plan_sha:
        raise ValueError(
            "calibration lock evaluation plan digest mismatch: "
            f"expected {_close_digest}, got {evaluation_plan_sha}"
        )
    if evaluation_plan.get("task_spec_sha256") != payload.get("task_spec_sha256"):
        raise ValueError("calibration lock evaluation plan TASK digest is stale")
    curve = payload.get("curve")
    if not isinstance(curve, Mapping):
        raise ValueError("calibration lock is missing curve")
    x_ref = float(curve.get("x_ref"))
    PiecewiseLinearCurve.from_reference(x_ref)
    targets = payload.get("targets")
    measurements = payload.get("measurements")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("calibration lock is missing targets")
    for name, target in targets.items():
        if not isinstance(target, Mapping):
            raise ValueError(f"calibration target {name!r} must be an object")
        metric = target.get("metric")
        if not isinstance(metric, Mapping) or not all(
            isinstance(metric.get(field), str) and metric.get(field)
            for field in ("id", "formula", "input_contract")
        ):
            raise ValueError(
                f"calibration target {name!r} is missing versioned metric semantics"
            )
        if ".v" not in str(metric["id"]):
            raise ValueError(f"calibration target {name!r} metric id is not versioned")
        floor = target.get("floor")
        rationale = floor.get("rationale") if isinstance(floor, Mapping) else None
        try:
            floor_value = (
                float(floor.get("value")) if isinstance(floor, Mapping) else math.nan
            )
        except (TypeError, ValueError):
            floor_value = math.nan
        if (
            not isinstance(floor, Mapping)
            or not math.isfinite(floor_value)
            or not isinstance(rationale, Mapping)
            or rationale.get("kind") not in RATIONALE_KINDS
            or not isinstance(rationale.get("summary"), str)
            or len(rationale["summary"].strip()) < 20
        ):
            raise ValueError(f"calibration target {name!r} has invalid floor rationale")
    if not isinstance(measurements, Mapping):
        raise ValueError("calibration lock is missing measurements")
    for role in ("reference", "naive"):
        values = measurements.get(role)
        if not isinstance(values, Mapping) or set(values) != set(targets):
            raise ValueError(
                f"calibration lock {role} measurements do not match targets"
            )
        for name, value in values.items():
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(
                    f"calibration lock contains non-finite {role} metric {name!r}"
                )
    degenerate = measurements.get("degenerate")
    if not isinstance(degenerate, Mapping) or not degenerate:
        raise ValueError("calibration lock is missing degenerate measurements")
    for strategy, values in degenerate.items():
        if not isinstance(values, Mapping) or set(values) != set(targets):
            raise ValueError(
                f"calibration lock degenerate strategy {strategy!r} measurements "
                "do not match targets"
            )
        for name, value in values.items():
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(
                    "calibration lock contains non-finite degenerate metric "
                    f"{name!r} for strategy {strategy!r}"
                )
    for name, spec in targets.items():
        if (
            not isinstance(spec, Mapping)
            or "raw_floor" not in spec
            or ("no_info_ceiling" not in spec)
        ):
            raise ValueError(
                f"calibration lock target {name!r} is missing raw_floor/"
                "no_info_ceiling"
            )
        floor = float(spec["floor"]["value"])
        raw_floor = float(spec["raw_floor"])
        _close(floor, raw_floor, field=f"target {name!r} reviewed floor")
        direction = str(spec["direction"])
        degenerate_values = [float(values[name]) for values in degenerate.values()]
        derived_ceiling = (
            min(degenerate_values) if direction == "lower" else max(degenerate_values)
        )
        _close(
            float(spec["no_info_ceiling"]),
            derived_ceiling,
            field=f"target {name!r} no-information ceiling",
        )

    weight_total = sum(float(spec["weight"]) for spec in targets.values())
    _close(weight_total, 1.0, field="normalized target weights")
    qualification_targets = {
        name: {
            **spec,
            "floor": {
                **spec["floor"],
                "value": (
                    min(
                        float(spec["raw_floor"]),
                        float(spec["no_info_ceiling"]),
                    )
                    if str(spec["direction"]) == "lower"
                    else max(
                        float(spec["raw_floor"]),
                        float(spec["no_info_ceiling"]),
                    )
                ),
            },
        }
        for name, spec in targets.items()
    }
    reference = measurements["reference"]
    naive = measurements["naive"]
    derived_x_ref = _aggregate_from_payload(targets, reference)
    _close(x_ref, derived_x_ref, field="curve.x_ref")
    qualification = payload.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ValueError("calibration lock is missing qualification")
    curve_obj = PiecewiseLinearCurve.from_reference(x_ref)
    expected_qualification = {
        "reference_score": curve_obj.score(derived_x_ref),
        "oracle_score": curve_obj.score(1.0),
        "null_score": curve_obj.score(0.0),
        "naive_progress": _aggregate_from_payload(qualification_targets, naive),
        "naive_quality_progress": _aggregate_from_payload(targets, naive),
    }
    expected_qualification["naive_score"] = curve_obj.score(
        expected_qualification["naive_progress"]
    )
    expected_qualification["qualification_naive_score"] = expected_qualification[
        "naive_score"
    ]
    expected_qualification["runtime_naive_quality_score"] = curve_obj.score(
        expected_qualification["naive_quality_progress"]
    )
    expected_qualification["naive_semantic_gap"] = abs(
        expected_qualification["runtime_naive_quality_score"]
        - expected_qualification["qualification_naive_score"]
    )
    stored_qualification = (
        expected_qualification
        if schema_version == CALIBRATION_LOCK_SCHEMA
        else {
            field: expected_qualification[field]
            for field in (
                "reference_score",
                "oracle_score",
                "null_score",
                "naive_progress",
                "naive_quality_progress",
                "naive_score",
            )
        }
    )
    for field, expected in stored_qualification.items():
        try:
            actual = float(qualification[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"calibration lock qualification is missing {field}"
            ) from exc
        _close(actual, expected, field=f"qualification.{field}")
    if schema_version == CALIBRATION_LOCK_SCHEMA:
        try:
            semantic_gap_threshold = float(
                qualification["max_unacknowledged_naive_score_gap"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "calibration lock qualification is missing "
                "max_unacknowledged_naive_score_gap"
            ) from exc
        if (
            not math.isfinite(semantic_gap_threshold)
            or not 0 <= semantic_gap_threshold < 0.5
        ):
            raise ValueError(
                "calibration lock naive semantic gap threshold must lie in [0, 0.5)"
            )
        semantic_gap_acknowledgement = qualification.get(
            "naive_semantic_gap_acknowledgement"
        )
        if (
            expected_qualification["naive_semantic_gap"] > semantic_gap_threshold
            and semantic_gap_acknowledgement is None
        ):
            raise ValueError(
                "calibration lock naive semantic gap exceeds its threshold without "
                "a naive_semantic_gap_acknowledgement"
            )
        if semantic_gap_acknowledgement is not None:
            _validate_naive_at_floor_rationale(semantic_gap_acknowledgement)
    _validate_naive_score_range(
        qualification,
        targets,
        naive,
        naive_score=float(expected_qualification["naive_score"]),
    )

    degenerate_scores = qualification.get("degenerate_scores")
    if not isinstance(degenerate_scores, Mapping) or set(degenerate_scores) != set(
        degenerate
    ):
        raise ValueError(
            "calibration lock qualification.degenerate_scores does not match "
            "degenerate measurements"
        )
    for strategy, strategy_metrics in degenerate.items():
        expected = curve_obj.score(
            _aggregate_from_payload(qualification_targets, strategy_metrics)
        )
        _close(
            float(degenerate_scores[strategy]),
            expected,
            field=f"qualification.degenerate_scores[{strategy!r}]",
        )
    normalized = json.loads(canonical_json_bytes(payload))
    return CalibrationLock(payload=normalized, sha256=canonical_sha256(normalized))


def calibration_lock_candidates(filename: str = DEFAULT_LOCK_FILENAME) -> list[Path]:
    """Every path the sealed lock may legitimately occupy, most specific first.

    The per-task image always has it at RUNTIME_LOCK_ROOT, but mothership's
    agent-service lane ships it inside an uploaded bundle and the runner decides
    where that bundle lands. The lane exported one absolute guess, the runner
    extracted a level deeper, and every continuous task scored 0.0 with
    "calibration lock is missing". Treat the exported path as the primary answer
    and the bundle-root variants as fallbacks, so neither side has to know the
    other's layout.

    The two sets are deliberately disjoint. RUNTIME_LOCK_ROOT on a task image is
    the author's own calibration, shipped beside an ``.author-source`` marker
    precisely so it is never mistaken for a sealed one. A caller that named a
    sealed bundle lock and did not get it must fail closed, not quietly grade
    against author numbers, so an override searches only bundle locations.
    """

    name = Path(filename).name
    candidates: list[Path] = []
    override = os.environ.get(CALIBRATION_LOCK_PATH_ENV)
    if override:
        primary = Path(override)
        candidates.append(primary)
        arc = Path(primary.parent.name) / name
        assumed_root = primary.parent.parent
        if str(assumed_root) not in {"", ".", "/"}:
            # The runner extracts the bundle into a subdirectory of the root the
            # lane assumed (/workspace -> /workspace/workspace).
            candidates.append(assumed_root / assumed_root.name / arc)
            # Or the lane over-qualified the path and the arc sits one up.
            candidates.append(assumed_root.parent / arc)
    else:
        candidates.append(RUNTIME_LOCK_ROOT / name)

    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def resolve_calibration_lock_path(filename: str = DEFAULT_LOCK_FILENAME) -> Path:
    candidates = calibration_lock_candidates(filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_calibration_lock(
    path: Path | None = None,
    *,
    filename: str = DEFAULT_LOCK_FILENAME,
    task_spec_sha256: str | None = None,
    compatible_task_spec_sha256s: tuple[str, ...] = (),
) -> CalibrationLock:
    resolved = path or resolve_calibration_lock_path(filename)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        searched = (
            [resolved] if path is not None else calibration_lock_candidates(filename)
        )
        locations = ", ".join(str(candidate) for candidate in searched)
        raise RuntimeError(
            f"calibration lock is missing (searched {locations}); "
            "run the ML ground-truth workflow"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"could not read calibration lock at {resolved}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"calibration lock at {resolved} must be a JSON object")
    return validate_calibration_lock(
        payload,
        task_spec_sha256=task_spec_sha256,
        compatible_task_spec_sha256s=compatible_task_spec_sha256s,
    )


def write_calibration_lock_atomic(path: Path, lock: CalibrationLock) -> None:
    """Atomically replace a lock only after canonical serialization succeeds."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(lock.payload)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


__all__ = [
    "CALIBRATION_LOCK_PATH_ENV",
    "CALIBRATION_LOCK_SCHEMA",
    "CALIBRATION_POLICY",
    "CalibrationLock",
    "build_calibration_lock",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_calibration_lock",
    "resolve_calibration_lock_path",
    "validate_calibration_lock",
    "write_calibration_lock_atomic",
]
