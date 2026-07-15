"""Lightweight public constants and equivalent-linear screening helpers.

Use these closed-form relations to screen candidate (Qd, Kd, Dy) designs
quickly. Final credit is based on nonlinear response demands in the hidden
grader, so equivalent-linear estimates alone are only a coarse guide.
"""

from __future__ import annotations

import math
from typing import Any

import public_isolation_model as model

G_IN_S2 = model.G_IN_S2
SEISMIC_WEIGHT_KIP = model.TOTAL_WEIGHT_KIP
SEISMIC_MASS = model.TOTAL_WEIGHT_KIP / model.G_IN_S2

QD_RANGE = (model.QD_MIN_KIP, model.QD_MAX_KIP)
KD_RANGE = (model.KD_MIN_KIP_IN, model.KD_MAX_KIP_IN)
DY_RANGE = (model.DY_MIN_IN, model.DY_MAX_IN)

# Disclosed performance targets and moat capacity (mirror public_analysis_envelope.json).
PERFORMANCE_TARGETS = {
    "peak_isolator_disp_in": 43.6,
    "peak_interstory_drift_ratio": 0.00865,
    "peak_floor_acceleration_g": 0.720,
    "peak_base_shear_coeff": 0.366,
    "residual_isolator_disp_in": 8.2,
}
MOAT_CAPACITY_IN = 45.0
RECENTERING_CAPACITY_IN = 9.0


def effective_properties(qd_kip: float, kd_kip_in: float, displacement_in: float) -> dict[str, float]:
    """Equivalent-linear effective stiffness, period, and damping of the system."""
    return model.effective_period_and_damping(float(qd_kip), float(kd_kip_in), float(displacement_in))


def post_yield_period_sec(kd_kip_in: float) -> float:
    """Isolation period if only the post-yield (rubber) stiffness resisted."""
    return 2.0 * math.pi * math.sqrt(SEISMIC_MASS / max(float(kd_kip_in), 1.0e-9))


def characteristic_strength_ratio(qd_kip: float) -> float:
    """Qd / W -- the normalized characteristic strength."""
    return float(qd_kip) / SEISMIC_WEIGHT_KIP


def in_bounds(design: dict[str, Any]) -> bool:
    try:
        system = model.extract_system(design)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        QD_RANGE[0] <= system["Qd_kip"] <= QD_RANGE[1]
        and KD_RANGE[0] <= system["Kd_kip_per_in"] <= KD_RANGE[1]
        and DY_RANGE[0] <= system["Dy_in"] <= DY_RANGE[1]
    )


if __name__ == "__main__":
    # Quick screening demonstration.
    for qd, kd in ((200, 16), (360, 30), (500, 45)):
        eff = effective_properties(qd, kd, 25.0)
        print(
            f"Qd={qd:4d} Kd={kd:3d}  Qd/W={characteristic_strength_ratio(qd):.3f}  "
            f"T_eff(@25in)={eff['effective_period_sec']:.2f}s  "
            f"beta_eff={eff['effective_damping_ratio']:.3f}  "
            f"T_postyield={post_yield_period_sec(kd):.2f}s"
        )
