"""
Diffuser Separation Control -- Grader
=====================================

Physics-based scoring for the diffuser-separation-control CFD task.
The oracle (optimal design: 7 deg half-angle, length ratio 6.0,
inlet extension >= 0.5 m) must score exactly 1.0 for ground-truth
validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from grading.faults import AgentFault
from grading.helpers import load_json
from grading.evaluation import (
    JsonArtifact,
    NumericField,
    RubricCriterion,
    RubricEvaluation,
    RubricTask,
)


def _evaluate(context):
    """Evaluate diffuser design using RubricTask protocol."""
    ws = context.workspace
    submission_path = ws / "output" / "diffuser_design.json"
    if not submission_path.exists():
        submission_path = ws / "diffuser_design.json"

    data = load_json(submission_path)

    try:
        angle = float(data["half_angle_deg"])
        length = float(data["length_ratio"])
        inlet_ext = float(data["inlet_extension_m"])
    except KeyError as exc:
        raise AgentFault(f"missing field: {exc}")
    except (TypeError, ValueError) as exc:
        raise AgentFault(f"invalid numeric value: {exc}")

    # Load public baseline / operating data
    data_dir = Path("/data")
    try:
        with open(data_dir / "baseline_diffuser.json") as fh:
            baseline = json.load(fh)
        with open(data_dir / "public_operating_conditions.json") as fh:
            ops = json.load(fh)          # noqa: F841 -- reserved for future use
    except FileNotFoundError:
        baseline = {"inlet_height_m": 0.1, "outlet_height_m": 0.2}
        ops = {"inlet_velocity_m_per_s": 15.0,
               "kinematic_viscosity_m2_per_s": 1.5e-5}

    h1 = baseline["inlet_height_m"]
    h2 = baseline.get("outlet_height_m", 0.2)
    AR = h2 / h1                     # Baseline area ratio = 2.0

    # Physics surrogate
    Cp_ideal = 1.0 - (1.0 / AR ** 2)
    theta_rad = math.radians(angle)

    # Loss is zero at and below the optimal 7 deg half-angle
    theta_opt = math.radians(7.0)
    deviation = max(0.0, theta_rad - theta_opt)
    K_loss = 0.08 * (math.tan(deviation)) ** 2 * (AR - 1.0) ** 2

    length_factor = min(1.0, length / 6.0)
    Cp = max(0.0, Cp_ideal - K_loss) * length_factor
    pressure_recovery = Cp / Cp_ideal if Cp_ideal > 0 else 0.0

    # Separation penalty (raw: 0=good, 1=bad)
    theta_stall = 10.0
    separation_penalty_raw = (
        0.0 if angle <= theta_stall
        else min(1.0, (angle - theta_stall) / 5.0)
    )

    # Outlet uniformity
    L_opt = 6.0
    uniformity = math.exp(-0.5 * ((length - L_opt) / 2.5) ** 2)

    # Robustness
    robustness = max(0.0, 1.0 - abs(angle - 7.0) / 8.0)

    # Inlet extension bonus
    inlet_bonus = min(1.0, inlet_ext / 0.5)

    # Subscores in "higher is better" form.
    # separation_penalty is inverted so 1.0 = no penalty (good).
    subscores = {
        "pressure_recovery": round(pressure_recovery, 4),
        "separation_penalty": round(1.0 - separation_penalty_raw, 4),
        "outlet_uniformity": round(uniformity, 4),
        "robustness": round(robustness, 4),
        "inlet_extension_bonus": round(inlet_bonus, 4),
    }

    # Headline score (weighted sum -- oracle hits 1.0)
    headline = (
        0.30 * subscores["pressure_recovery"]
        + 0.25 * subscores["separation_penalty"]
        + 0.20 * subscores["outlet_uniformity"]
        + 0.15 * subscores["robustness"]
        + 0.10 * subscores["inlet_extension_bonus"]
    )

    return RubricEvaluation(
        subscores=subscores,
        metadata={
            "reported_final_score": round(headline, 4),
            "scoring_mode": "weighted",
        },
    )


TASK = RubricTask(
    artifact=JsonArtifact(
        "diffuser_design.json",
        required_keys=["half_angle_deg", "length_ratio", "inlet_extension_m"],
        numeric_fields=[
            NumericField("half_angle_deg"),
            NumericField("length_ratio"),
            NumericField("inlet_extension_m"),
        ],
        allow_extra_keys=False,
    ),
    fixtures={},
    criteria=tuple(
        RubricCriterion(
            id=key,
            weight=weight,
            description=key.replace("_", " ").title(),
            required=False,
        )
        for key, weight in {
            "pressure_recovery": 0.30,
            "separation_penalty": 0.25,
            "outlet_uniformity": 0.20,
            "robustness": 0.15,
            "inlet_extension_bonus": 0.10,
        }.items()
    ),
    evaluate=_evaluate,
)
