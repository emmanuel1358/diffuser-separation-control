"""Deterministic grader for the lead-rubber base-isolation design task.

The agent submits one file under /tmp/output:

  * isolation_design.json  -- the isolation-system design variables.

The grader NEVER imports, execs, or evals the submitted files as code. It only
reads JSON. It independently runs the disclosed solver-backed model on the six
private ground-motion records and scores the worst case.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from grading import AgentFault
except Exception:  # pragma: no cover - host-side unit tests run without the grading pkg
    class AgentFault(Exception):
        """Fallback used only when the grading package is unavailable (e.g. host
        unit tests). In the container grader the real grading.AgentFault is used
        and the runtime keeps the rollout as a clean 0.0."""


DESIGN_FILENAME = "isolation_design.json"
FLOAT_TOL = 1.0e-5
INVALID_SCORE = 0.0

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
# Entry point
# ---------------------------------------------------------------------------
def compute_score(
    workspace: Path,
    trajectory: list[dict[str, Any]] | None = None,
    private: Path | None = None,
    transcript: str = "",
) -> dict[str, Any]:
    workspace = Path(workspace)
    private = Path(private) if private is not None else default_private_dir()

    config_result = load_hidden_config(private)
    if config_result["error"]:
        return failure("hidden_config_error", config_result["error"])
    config = config_result["config"]
    weights = numeric_weights(config)

    # Agent-caused problems (missing/unreadable/malformed submission, an
    # out-of-range design, or a non-converged
    # design) raise AgentFault so the runtime keeps a clean 0.0 for training.
    # Author/infra problems (missing fixture, un-importable solver) return a
    # failure dict / propagate so the runtime discards the rollout instead.
    design = read_agent_json(workspace / DESIGN_FILENAME, DESIGN_FILENAME)

    validation = validate_design(design)
    if validation["errors"]:
        raise AgentFault(
            "isolation_design.json failed validation: " + "; ".join(validation["errors"])
        )

    # Independent numeric evaluation on the private ground motions. A failure to
    # import OpenSeesPy or run the disclosed model is an infrastructure/author
    # problem; only return 0 (never crash) so degenerate host-side trivial
    # grades stay at 0. Real grading runs in-container with OpenSeesPy present.
    model = import_model()
    if model is None:
        return failure(
            "solver_unavailable",
            "OpenSeesPy / public_isolation_model is not importable in this grading environment.",
            metadata={"validation": validation},
            weights=weights,
        )

    hidden_cases = config["hidden_cases"]
    result = model.evaluate_design(design, hidden_cases)
    worst = result["worst_case"]
    if not result["all_converged"]:
        raise AgentFault(
            "the submitted isolation design produced a non-converged response history "
            "on at least one design-basis ground motion"
        )

    scoring = compute_scoring(worst, config, weights)
    return sanitize(
        {
            "score": scoring["score"],
            "subscores": scoring["subscores"],
            "weights": weights,
            "metadata": {
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
                "message": (
                    "Score is the worst case over six private design-basis ground "
                    "motions. A balanced isolation design meets the displacement, "
                    "acceleration, drift, base-shear, and residual-displacement "
                    "targets simultaneously; "
                    "too-soft designs are gated by the moat-capacity (pounding) "
                    "limit, too-stiff designs lose acceleration and drift credit."
                ),
            },
        }
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def compute_scoring(worst: dict[str, Any], config: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
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
    moat_ratio = (iso_value / moat_capacity) if (iso_value is not None and moat_capacity > 0.0) else 2.0
    moat_gate = interpolate(moat_ratio, moat_curve)

    residual_value = safe_float(worst.get("residual_isolator_disp_in"))
    recentering_ratio = (
        residual_value / recentering_capacity
        if (residual_value is not None and recentering_capacity > 0.0)
        else 2.0
    )
    recentering_gate = interpolate(recentering_ratio, recentering_curve)

    weighted_total = clip01(sum(weights[key] * subscores[key] for key in SUBSCORE_KEYS))
    score = clip01((weighted_total ** exponent) * moat_gate * recentering_gate)
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
        return {"errors": ["isolation_design.json must be a JSON object."], "system": None}
    if set(design.keys()) != {"isolation_system"}:
        errors.append("Top-level object must contain exactly the key 'isolation_system'.")
    system = design.get("isolation_system")
    if not isinstance(system, dict):
        return {"errors": errors + ["'isolation_system' must be a JSON object."], "system": None}
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
            errors.append(f"{key}={system.get(key)} is outside the allowed range [{low}, {high}].")
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


def canonical_hash(design: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(design, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_private_dir() -> Path:
    runtime_private = Path("/mcp_server/data")
    if (runtime_private / "hidden_cases.json").exists():
        return runtime_private
    return Path(__file__).resolve().parent / "data"


def load_hidden_config(private: Path) -> dict[str, Any]:
    candidates = [
        private / "hidden_cases.json",
        Path("/mcp_server/data/hidden_cases.json"),
        Path(__file__).resolve().parent / "data" / "hidden_cases.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return {"config": json.loads(path.read_text(encoding="utf-8")), "error": None}
            except json.JSONDecodeError as exc:
                return {"config": None, "error": f"Could not parse hidden_cases.json: {exc}"}
            except OSError as exc:
                return {"config": None, "error": f"Could not read hidden_cases.json: {exc}"}
    return {"config": None, "error": "Could not find hidden_cases.json."}


def numeric_weights(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("scoring", {}).get("weights", DEFAULT_WEIGHTS)
    weights = {key: float(raw.get(key, DEFAULT_WEIGHTS[key])) for key in SUBSCORE_KEYS}
    total = sum(weights.values())
    if total <= 0.0:
        return {key: 0.0 for key in SUBSCORE_KEYS}
    return {key: value / total for key, value in weights.items()}


def read_agent_json(path: Path, label: str) -> Any:
    """Read an agent-submitted JSON file. A missing/unreadable file (including a
    planted directory or FIFO) or malformed JSON is an agent-controlled fault, so
    raise AgentFault to keep a clean 0.0 rather than crashing or returning a
    homebrew score."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AgentFault(f"missing required output file: {label}") from exc
    except OSError as exc:
        raise AgentFault(f"could not read {label}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentFault(f"{label} is not valid JSON: {exc}") from exc


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


def int_or_none(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def floats_close(left: Any, right: Any, tol: float = FLOAT_TOL) -> bool:
    a, b = number(left), number(right)
    if a is None or b is None:
        return False
    return abs(a - b) <= tol + tol * abs(b)


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


def failure(reason: str, message: str, metadata: dict[str, Any] | None = None, weights: dict[str, float] | None = None) -> dict[str, Any]:
    weights = weights or dict(DEFAULT_WEIGHTS)
    return sanitize(
        {
            "score": INVALID_SCORE,
            "subscores": {key: 0.0 for key in SUBSCORE_KEYS},
            "weights": weights,
            "metadata": {"reason": reason, "message": message, **(metadata or {})},
        }
    )
