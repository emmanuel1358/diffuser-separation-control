#!/usr/bin/env python3
"""Oracle: derive a balanced isolation design by running structural analyses on public,
in-band ground motions, scored with the disclosed scoring rule.

This does NOT read the private hidden_cases.json. It searches the disclosed
design space, evaluates each candidate with the public structural model over a
set of representative ground motions that span the disclosed band (including the
long-predominant-period end that governs the worst case), and keeps the design
with the highest disclosed score. It then writes the design.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import product
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", "/data"))
OUT_DIR = Path(os.environ.get("TASK_OUTPUT_DIR", "/tmp/output"))
for candidate in (DATA_DIR, Path(__file__).resolve().parents[1] / "data"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import public_isolation_model as model  # noqa: E402


def load_scoring() -> dict[str, Any]:
    envelope = json.loads((DATA_DIR / "public_analysis_envelope.json").read_text(encoding="utf-8"))
    scoring = dict(envelope["scoring"])
    scoring["targets"] = envelope["performance_targets"]
    scoring["moat_capacity_in"] = envelope["moat_capacity_in"]
    scoring["recentering_capacity_in"] = envelope["recentering_capacity_in"]
    return scoring


# In-band ground motions spanning the design-basis range up to the
# maximum-considered (MCE) corner. The grader scores the worst case over the
# band, and isolator displacement grows with intensity and period, so the oracle
# must evaluate the demanding corner (not just the milder public probe records).
SEARCH_CASES = [
    {"case_id": "s_dbe_short_a", "pga_g": 0.30, "tp_sec": 1.5, "duration_sec": 24.0, "phase_seed": 109},
    {"case_id": "s_dbe_short_b", "pga_g": 0.30, "tp_sec": 1.5, "duration_sec": 24.0, "phase_seed": 229},
    {"case_id": "s_strong_mid_a", "pga_g": 0.33, "tp_sec": 2.6, "duration_sec": 26.0, "phase_seed": 337},
    {"case_id": "s_strong_mid_b", "pga_g": 0.33, "tp_sec": 2.6, "duration_sec": 26.0, "phase_seed": 457},
    {"case_id": "s_mce_long_a", "pga_g": 0.35, "tp_sec": 3.15, "duration_sec": 28.0, "phase_seed": 557},
    {"case_id": "s_mce_long_b", "pga_g": 0.35, "tp_sec": 3.15, "duration_sec": 28.0, "phase_seed": 677},
    {"case_id": "s_mce_corner_a", "pga_g": 0.36, "tp_sec": 3.35, "duration_sec": 30.0, "phase_seed": 787},
    {"case_id": "s_mce_corner_b", "pga_g": 0.36, "tp_sec": 3.35, "duration_sec": 30.0, "phase_seed": 887},
]

METRIC_FOR_KEY = {
    "isolator_displacement_control": "peak_isolator_disp_in",
    "floor_acceleration_control": "peak_floor_acceleration_g",
    "interstory_drift_control": "peak_interstory_drift_ratio",
    "base_shear_control": "peak_base_shear_coeff",
    "residual_displacement_control": "residual_isolator_disp_in",
}


def interpolate(value: float, knots: list[list[float]]) -> float:
    pts = [(float(x), float(y)) for x, y in knots]
    if value <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if value <= x1:
            return y1 if x1 == x0 else y0 + (y1 - y0) * (value - x0) / (x1 - x0)
    return pts[-1][1]


def disclosed_score(worst: dict[str, float], scoring: dict[str, Any]) -> float:
    weights = scoring["weights"]
    targets = scoring["targets"]
    lower = scoring["lower_ratio_curve"]
    total = 0.0
    for key, weight in weights.items():
        metric = METRIC_FOR_KEY[key]
        ratio = worst[metric] / float(targets[metric])
        total += float(weight) * interpolate(ratio, lower)
    gate = interpolate(worst["peak_isolator_disp_in"] / float(scoring["moat_capacity_in"]), scoring["moat_gate_curve"])
    recentering_gate = interpolate(
        worst["residual_isolator_disp_in"] / float(scoring["recentering_capacity_in"]),
        scoring["recentering_gate_curve"],
    )
    return (total ** float(scoring["final_score_exponent"])) * gate * recentering_gate


def main() -> None:
    scoring = load_scoring()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best_design: dict[str, Any] | None = None
    best_score = -1.0
    best_worst: dict[str, float] = {}

    qd_grid = [380.0, 420.0, 460.0, 500.0, 540.0]
    kd_grid = [16.0, 18.0, 20.0, 22.0, 24.0]
    dy_grid = [0.5, 0.6, 0.7]

    for qd, kd, dy in product(qd_grid, kd_grid, dy_grid):
        design = {"isolation_system": {"Qd_kip": qd, "Kd_kip_per_in": kd, "Dy_in": dy}}
        result = model.evaluate_design(design, SEARCH_CASES)
        if not result["all_converged"]:
            continue
        score = disclosed_score(result["worst_case"], scoring)
        if score > best_score:
            best_score = score
            best_design = design
            best_worst = result["worst_case"]

    if best_design is None:
        raise RuntimeError("oracle search failed to find any converged design")

    design_path = OUT_DIR / "isolation_design.json"
    design_path.write_text(json.dumps(best_design, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "oracle_design": best_design["isolation_system"],
        "disclosed_public_score": round(best_score, 4),
        "search_worst_case": {k: round(float(v), 4) for k, v in best_worst.items()},
        "wrote": [str(design_path)],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
