from __future__ import annotations

import json
import math
from pathlib import Path

from grading.faults import AgentFault
from grading.helpers import load_json


def compute_score(workspace: str, trajectory=None, private=None):
    submission_path = Path(workspace) / "output" / "diffuser_design.json"
    data = load_json(submission_path)

    try:
        angle = float(data["half_angle_deg"])
        length = float(data["length_ratio"])
        inlet_ext = float(data["inlet_extension_m"])
    except KeyError as e:
        raise AgentFault(f"missing field: {e}")
    except (TypeError, ValueError) as e:
        raise AgentFault(f"invalid numeric value: {e}")

    # Load public data for physics-based scoring
    data_dir = Path("/data")
    try:
        with open(data_dir / "baseline_diffuser.json") as f:
            baseline = json.load(f)
        with open(data_dir / "public_operating_conditions.json") as f:
            ops = json.load(f)
    except FileNotFoundError:
        # Fallback for local testing without /data
        baseline = {"inlet_height_m": 0.1, "outlet_height_m": 0.2}
        ops = {"inlet_velocity_m_per_s": 15.0, "kinematic_viscosity_m2_per_s": 1.5e-5}

    h1 = baseline["inlet_height_m"]
    h2 = baseline.get("outlet_height_m", 0.2)
    AR = h2 / h1  # Area ratio (2.0 for baseline)

    # --- Physics-based surrogate scoring ---

    # 1. Pressure Recovery
    Cp_ideal = 1.0 - (1.0 / AR**2)
    theta_rad = math.radians(angle)
    K_loss = 0.08 * (math.tan(theta_rad))**2 * (AR - 1.0)**2
    length_factor = min(1.0, length / 6.0)
    Cp = max(0.0, Cp_ideal - K_loss) * length_factor
    pressure_recovery = Cp / Cp_ideal if Cp_ideal > 0 else 0.0

    # 2. Separation Penalty
    theta_stall = 10.0
    if angle <= theta_stall:
        separation_penalty = 0.0
    else:
        separation_penalty = min(1.0, (angle - theta_stall) / 5.0)

    # 3. Outlet Uniformity
    L_opt = 6.0
    uniformity = math.exp(-0.5 * ((length - L_opt) / 2.5)**2)

    # 4. Robustness
    robustness = max(0.0, 1.0 - abs(angle - 7.0) / 8.0)

    # 5. Inlet extension bonus
    inlet_bonus = min(1.0, inlet_ext / 0.5)

    # Combined weighted score, calibrated to [0.1, 0.7] range
    raw_score = (
        0.30 * pressure_recovery +
        0.25 * (1.0 - separation_penalty) +
        0.20 * uniformity +
        0.15 * robustness +
        0.10 * inlet_bonus
    )
    score = 0.1 + 0.6 * raw_score
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "subscores": {
            "pressure_recovery": round(pressure_recovery, 4),
            "separation_penalty": round(separation_penalty, 4),
            "outlet_uniformity": round(uniformity, 4),
            "robustness": round(robustness, 4),
            "inlet_extension_bonus": round(inlet_bonus, 4),
        },
    }


if __name__ == "__main__":
    import sys

    workspace = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    print(json.dumps(compute_score(workspace), indent=2))