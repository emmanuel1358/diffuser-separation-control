"""Deterministic grader for the lead-rubber base-isolation design task.

The agent submits one file under /tmp/output:

  * isolation_design.json  -- the isolation-system design variables.

The grader NEVER imports, execs, or evals the submitted files as code. It only
reads JSON. It independently runs the disclosed solver-backed model on the six
private ground-motion records and scores the worst case.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from grading.evaluation import (
    JsonArtifact,
    NumericField,
    RubricCriterion,
    RubricEvaluation,
    RubricTask,
    TrustedJson,
)

DESIGN_FILENAME = "isolation_design.json"

SUBSCORE_KEYS = (
    "isolator_displacement_control",
    "floor_acceleration_control",
    "interstory_drift_control",
    "base_shear_control",
    "residual_displacement_control",
)
METRIC_FOR_SUBSCORE = {
    "isolator_displacement_control": "peak_isolator_disp_in",
    "floor_acceleration_control": "peak_floor_acceleration_g",
    "interstory_drift_control": "peak_interstory_drift_ratio",
    "base_shear_control": "peak_base_shear_coeff",
    "residual_displacement_control": "residual_isolator_disp_in",
}
DEFAULT_WEIGHTS = {
    "isolator_displacement_control": 0.22,
    "floor_acceleration_control": 0.25,
    "interstory_drift_control": 0.19,
    "base_shear_control": 0.18,
    "residual_displacement_control": 0.16,
}

# Allowed design ranges (mirrored from public_isolation_model; duplicated here so
# validation never depends on importing OpenSeesPy).
BOUNDS = {
    "Qd_kip": (80.0, 650.0),
    "Kd_kip_per_in": (10.0, 90.0),
    "Dy_in": (0.30, 1.50),
}


# ---------------------------------------------------------------------------
# Declarative entry point
# ---------------------------------------------------------------------------
def evaluate(context):
    design = context.candidate
    config = context.fixture("config")
    weights = numeric_weights(config)

    validation = validate_design(design)
    if validation["errors"]:
        context.reject_candidate(
            "isolation_design.json failed validation: "
            + "; ".join(validation["errors"])
        )

    model = import_model()
    if model is None:
        context.grader_failure(
            "OpenSeesPy / public_isolation_model is not importable in this "
            "grading environment"
        )

    hidden_cases = config["hidden_cases"]
    result = context.trusted_operation(
        "OpenSees response-history evaluation",
        model.evaluate_design,
        design,
        hidden_cases,
    )
    worst = result["worst_case"]
    if not result["all_converged"]:
        context.reject_candidate(
            "the submitted isolation design produced a non-converged response history "
            "on at least one design-basis ground motion"
        )

    scoring = context.candidate_operation(
        "rubric score calculation",
        compute_scoring,
        worst,
        config,
        weights,
    )
    weighted_total = scoring["weighted_total"]
    factor = context.ratio(
        scoring["score"],
        weighted_total,
        label="headline preservation factor",
        zero="zero",
    )
    adjusted = {
        key: clip01(value * factor) for key, value in scoring["subscores"].items()
    }
    return RubricEvaluation(
        subscores=adjusted,
        metadata=sanitize(
            {
                "validation": validation,
                "worst_case": worst,
                "targets": config["scoring"]["targets"],
                "moat_capacity_in": config["scoring"]["moat_capacity_in"],
                "moat_gate": scoring["moat_gate"],
                "recentering_capacity_in": config["scoring"]["recentering_capacity_in"],
                "recentering_gate": scoring["recentering_gate"],
                "weighted_subscore_total": scoring["weighted_total"],
                "final_score_exponent": scoring["exponent"],
                "metric_ratios": scoring["ratios"],
                "per_case": result["per_case"],
                "model_version": result["model_version"],
            }
        ),
    )


TASK = RubricTask(
    artifact=JsonArtifact(
        DESIGN_FILENAME,
        required_keys=("isolation_system",),
        numeric_fields=tuple(
            NumericField(
                f"isolation_system.{name}",
                minimum=bounds[0],
                maximum=bounds[1],
            )
            for name, bounds in BOUNDS.items()
        ),
        allow_extra_keys=False,
    ),
    fixtures={"config": TrustedJson("hidden_cases.json")},
    criteria=tuple(
        RubricCriterion(
            id=key,
            weight=DEFAULT_WEIGHTS[key],
            description=key.replace("_", " "),
        )
        for key in SUBSCORE_KEYS
    ),
    evaluate=evaluate,
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def compute_scoring(
    worst: dict[str, Any], config: dict[str, Any], weights: dict[str, float]
) -> dict[str, Any]:
    scoring = config["scoring"]
    targets = scoring["targets"]
    lower_curve = scoring["lower_ratio_curve"]
    moat_curve = scoring["moat_gate_curve"]
    recentering_curve = scoring["recentering_gate_curve"]
    moat_capacity = float(scoring["moat_capacity_in"])
    recentering_capacity = float(scoring["recentering_capacity_in"])
    exponent = max(1.0, float(scoring["final_score_exponent"]))

    subscores: dict[str, float] = {}
    ratios: dict[str, float] = {}
    for key in SUBSCORE_KEYS:
        metric = METRIC_FOR_SUBSCORE[key]
        value = safe_float(worst.get(metric))
        target = safe_float(targets.get(metric))
        if value is None or target is None or target <= 0.0:
            subscores[key] = 0.0
            ratios[metric] = None
            continue
        ratio = value / target
        ratios[metric] = ratio
        subscores[key] = interpolate(ratio, lower_curve)

    iso_value = safe_float(worst.get("peak_isolator_disp_in"))
    moat_ratio = (
        (iso_value / moat_capacity)
        if (iso_value is not None and moat_capacity > 0.0)
        else 2.0
    )
    moat_gate = interpolate(moat_ratio, moat_curve)

    residual_value = safe_float(worst.get("residual_isolator_disp_in"))
    recentering_ratio = (
        residual_value / recentering_capacity
        if (residual_value is not None and recentering_capacity > 0.0)
        else 2.0
    )
    recentering_gate = interpolate(recentering_ratio, recentering_curve)

    weighted_total = clip01(sum(weights[key] * subscores[key] for key in SUBSCORE_KEYS))
    score = clip01((weighted_total**exponent) * moat_gate * recentering_gate)
    return {
        "score": score,
        "subscores": subscores,
        "weighted_total": weighted_total,
        "exponent": exponent,
        "moat_gate": moat_gate,
        "recentering_gate": recentering_gate,
        "ratios": ratios,
    }


def interpolate(value: float, knots: list[list[float]]) -> float:
    points = [(float(x), float(y)) for x, y in knots]
    if value <= points[0][0]:
        return clip01(points[0][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if value <= x1:
            if x1 == x0:
                return clip01(y1)
            t = (value - x0) / (x1 - x0)
            return clip01(y0 + t * (y1 - y0))
    return clip01(points[-1][1])


# ---------------------------------------------------------------------------
# Design validation
# ---------------------------------------------------------------------------
def validate_design(design: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(design, dict):
        return {
            "errors": ["isolation_design.json must be a JSON object."],
            "system": None,
        }
    if set(design.keys()) != {"isolation_system"}:
        errors.append(
            "Top-level object must contain exactly the key 'isolation_system'."
        )
    system = design.get("isolation_system")
    if not isinstance(system, dict):
        return {
            "errors": errors + ["'isolation_system' must be a JSON object."],
            "system": None,
        }
    expected = {"Qd_kip", "Kd_kip_per_in", "Dy_in"}
    if set(system.keys()) != expected:
        errors.append(f"'isolation_system' must contain exactly {sorted(expected)}.")
    values: dict[str, float] = {}
    for key in ("Qd_kip", "Kd_kip_per_in", "Dy_in"):
        value = number(system.get(key))
        low, high = BOUNDS[key]
        if value is None:
            errors.append(f"{key} must be a finite number.")
        elif not (low <= value <= high):
            errors.append(
                f"{key}={system.get(key)} is outside the allowed range [{low}, {high}]."
            )
        else:
            values[key] = value
    return {"errors": errors, "system": values if not errors else None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def import_model():
    for candidate in (
        "/data",
        "/mcp_server/public_data",
        str(Path(__file__).resolve().parents[1] / "data"),
    ):
        if Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)
    try:
        import public_isolation_model  # type: ignore

        return public_isolation_model
    except Exception:
        return None


def numeric_weights(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("scoring", {}).get("weights", DEFAULT_WEIGHTS)
    weights = {key: float(raw.get(key, DEFAULT_WEIGHTS[key])) for key in SUBSCORE_KEYS}
    total = sum(weights.values())
    if total <= 0.0:
        return {key: 0.0 for key in SUBSCORE_KEYS}
    return {key: value / total for key, value in weights.items()}


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_float(value: Any) -> float | None:
    return number(value)


def clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    return str(value)
