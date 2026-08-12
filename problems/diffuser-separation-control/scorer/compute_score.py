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
from grading.evaluation import RubricTask


def compute_score(workspace, trajectory=None, private=None):
    """Grade a diffuser-design submission.

    Parameters
    ----------
    workspace : str
        Absolute path to the agent workspace directory.
    trajectory : str | None
        Agent transcript (unused for this static task).
    private : pathlib.Path | None
        Path to grader-private fixtures (unused; all inputs are public).

    Returns
    -------
    dict
        Normalized result with ``score`` in [0.0, 1.0] and per-criterion
        ``subscores``.
    """
    ws = Path(workspace)

    # The harness may place the submission directly in the workspace
    # or in a nested ``output/`` sub-directory.
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

    # ------------------------------------------------------------------
    # Load public baseline / operating data
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Physics surrogate
    # ------------------------------------------------------------------
    Cp_ideal = 1.0 - (1.0 / AR ** 2)
    theta_rad = math.radians(angle)

    # Loss is zero at and below the optimal 7 deg half-angle; it only
    # accumulates when the angle exceeds the optimum (where stall risk
    # begins to dominate).
    theta_opt = math.radians(7.0)
    deviation = max(0.0, theta_rad - theta_opt)
    K_loss = 0.08 * (math.tan(deviation)) ** 2 * (AR - 1.0) ** 2

    length_factor = min(1.0, length / 6.0)
    Cp = max(0.0, Cp_ideal - K_loss) * length_factor
    pressure_recovery = Cp / Cp_ideal if Cp_ideal > 0 else 0.0

    # Separation penalty
    theta_stall = 10.0
    separation_penalty = (
        0.0 if angle <= theta_stall
        else min(1.0, (angle - theta_stall) / 5.0)
    )

    # Outlet uniformity (Gaussian around optimal length 6.0)
    L_opt = 6.0
    uniformity = math.exp(-0.5 * ((length - L_opt) / 2.5) ** 2)

    # Robustness (proximity to 7 deg sweet-spot)
    robustness = max(0.0, 1.0 - abs(angle - 7.0) / 8.0)

    # Inlet extension bonus
    inlet_bonus = min(1.0, inlet_ext / 0.5)

    # ------------------------------------------------------------------
    # Aggregate -- oracle must be able to reach exactly 1.0
    # ------------------------------------------------------------------
    score = (
        0.30 * pressure_recovery
        + 0.25 * (1.0 - separation_penalty)
        + 0.20 * uniformity
        + 0.15 * robustness
        + 0.10 * inlet_bonus
    )
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


class _DiffuserRubricTask(RubricTask):
    """Protocol wrapper required for multi_deterministic_rubrics."""

    def grade(self, workspace, trajectory=None, private=None):
        return compute_score(workspace, trajectory, private)


TASK = _DiffuserRubricTask()
