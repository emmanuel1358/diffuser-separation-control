"""Lead-rubber base-isolation model used by the grader and oracle.

This module is the SINGLE source of truth for the structural physics of the
task. The grader differs ONLY in which deterministic ground-motion cases it
evaluates (private records inside the disclosed band); the building model, the
lead-rubber isolator model, the synthetic ground-motion generator, and the
response-metric definitions are identical.

Units: kip, inch, second.  Acceleration of gravity g = 386.089 in/s^2.

Model
-----
A three-story shear-building superstructure sits on a base mat that is supported
by a lead-rubber isolation system. The isolation layer is modeled with two
parallel uniaxial springs at a zeroLength element:

    * a lead core  : elastic-perfectly-plastic (ElasticPP), yield force = Qd,
                     initial stiffness = Qd / Dy ;
    * a rubber unit: linear elastic, stiffness = Kd .

This reproduces the standard bilinear lead-rubber bearing backbone with
characteristic strength ``Qd`` and post-yield stiffness ``Kd``. The
superstructure stories are linear shear springs; the superstructure carries
2% Rayleigh damping anchored to two fixed-base modes, while the isolation
energy dissipation comes entirely from the lead core hysteresis.

Each ground motion is a deterministic Kanai-Tajimi accelerogram whose spectral
peak sits near ``1 / tp_sec`` and whose peak ground acceleration equals
``pga_g`` (in g). A disclosed integer ``phase_seed`` selects the Fourier-phase
realization. The frequency grid is fixed, so a given
``(pga_g, tp_sec, duration_sec, phase_seed)`` always produces the same record.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

MODEL_VERSION = "2026-06-22.isolation-rha-v2"

# ---------------------------------------------------------------------------
# Fixed structural properties (fully disclosed).
# ---------------------------------------------------------------------------
G_IN_S2 = 386.089

# Seismic weight (kip) lumped at the base mat and the three floors above it.
FLOOR_WEIGHT_KIP = (1200.0, 1000.0, 1000.0, 900.0)
TOTAL_WEIGHT_KIP = float(sum(FLOOR_WEIGHT_KIP))

# Superstructure story shear stiffness (kip/in) between consecutive levels.
STORY_STIFFNESS_KIP_IN = (1000.0, 800.0, 620.0)
# Story heights (in) used to turn relative floor displacement into drift ratio.
STORY_HEIGHT_IN = (156.0, 144.0, 144.0)

SUPERSTRUCTURE_DAMPING_RATIO = 0.02

# Transient integration step (s).
DT = 0.01

# Ground-motion generator constants (disclosed, fixed).
GM_DEFAULT_PHASE_SEED = 101
GM_FREQ_HZ = tuple(float(v) for v in np.linspace(0.15, 8.0, 90))
GM_GROUND_DAMPING = 0.55
GM_ENV_RISE_FRAC = 0.15
GM_ENV_HOLD_FRAC = 0.55
GM_ENV_DECAY = 0.30

# Allowed design-variable ranges (closed intervals, disclosed).
QD_MIN_KIP, QD_MAX_KIP = 80.0, 650.0
KD_MIN_KIP_IN, KD_MAX_KIP_IN = 10.0, 90.0
DY_MIN_IN, DY_MAX_IN = 0.30, 1.50


def design_bounds() -> dict[str, tuple[float, float]]:
    return {
        "Qd_kip": (QD_MIN_KIP, QD_MAX_KIP),
        "Kd_kip_per_in": (KD_MIN_KIP_IN, KD_MAX_KIP_IN),
        "Dy_in": (DY_MIN_IN, DY_MAX_IN),
    }


def canonical_design_hash(design: dict[str, Any]) -> str:
    payload = json.dumps(design, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_system(design: dict[str, Any]) -> dict[str, float]:
    """Pull the three isolation-system design variables out of a design dict."""
    system = design.get("isolation_system", {})
    return {
        "Qd_kip": float(system["Qd_kip"]),
        "Kd_kip_per_in": float(system["Kd_kip_per_in"]),
        "Dy_in": float(system["Dy_in"]),
    }


def effective_period_and_damping(qd_kip: float, kd_kip_in: float, displacement_in: float) -> dict[str, float]:
    """Closed-form effective-period / effective-damping screening helper.

    These are the standard equivalent-linear lead-rubber bearing relations at a
    trial displacement ``displacement_in``. They are provided for screening; the
    graded demands come from the nonlinear OpenSeesPy response history, not from
    these formulas.
    """
    disp = max(float(displacement_in), 1.0e-6)
    k_eff = kd_kip_in + qd_kip / disp
    total_mass = TOTAL_WEIGHT_KIP / G_IN_S2
    period = 2.0 * math.pi * math.sqrt(total_mass / max(k_eff, 1.0e-9))
    # Equivalent viscous damping of a bilinear loop, beta = 2 Q (D - Dy) / (pi Keff D^2).
    energy = 4.0 * qd_kip * disp  # area of bilinear loop (Dy small relative to D)
    beta = energy / (2.0 * math.pi * k_eff * disp * disp)
    return {
        "effective_stiffness_kip_per_in": float(k_eff),
        "effective_period_sec": float(period),
        "effective_damping_ratio": float(max(0.0, min(beta, 0.60))),
    }


def synthesize_ground_motion(
    pga_g: float,
    tp_sec: float,
    duration_sec: float,
    phase_seed: int = GM_DEFAULT_PHASE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Kanai-Tajimi accelerogram scaled to ``pga_g``.

    ``phase_seed`` selects a realization from the disclosed record family.  It
    changes only the Fourier phases; the spectrum, envelope, scaling, and all
    structural equations remain unchanged.  This lets a design be checked for
    record-to-record robustness instead of against one repeated phase pattern.
    """
    n = int(round(float(duration_sec) / DT)) + 1
    t = np.arange(n) * DT
    freqs = np.array(GM_FREQ_HZ, dtype=float)
    seed = int(phase_seed)
    if seed < 1 or seed > 999:
        raise ValueError("phase_seed must be an integer in [1, 999]")
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=freqs.shape)
    w = 2.0 * math.pi * freqs
    wg = 2.0 * math.pi / float(tp_sec)
    zg = GM_GROUND_DAMPING
    num = wg ** 4 + (2.0 * zg * wg * w) ** 2
    den = (wg ** 2 - w ** 2) ** 2 + (2.0 * zg * wg * w) ** 2
    amp = np.sqrt(num / den) / np.sqrt(w + 1.0)
    sig = np.zeros(n, dtype=float)
    for a_k, w_k, ph_k in zip(amp, w, phases):
        sig += a_k * np.sin(w_k * t + ph_k)
    # Saragoni-Hart intensity envelope.
    t1 = GM_ENV_RISE_FRAC * float(duration_sec)
    t2 = GM_ENV_HOLD_FRAC * float(duration_sec)
    env = np.ones(n, dtype=float)
    rise = t < t1
    env[rise] = (t[rise] / t1) ** 2
    decay = t > t2
    env[decay] = np.exp(-GM_ENV_DECAY * (t[decay] - t2))
    sig *= env
    peak = float(np.max(np.abs(sig)))
    if peak > 0.0:
        sig *= (float(pga_g) * G_IN_S2) / peak
    return t, sig


def run_case(design: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Build the isolated model and run one nonlinear response history.

    Returns peak isolator displacement (in), peak interstory drift ratio, peak
    total floor acceleration (g), peak base-shear coefficient (isolator force
    over seismic weight), residual isolator displacement (in), and convergence.
    """
    import openseespy.opensees as ops

    system = extract_system(design)
    qd = system["Qd_kip"]
    kd = system["Kd_kip_per_in"]
    dy = system["Dy_in"]

    t, ag = synthesize_ground_motion(
        float(case["pga_g"]),
        float(case["tp_sec"]),
        float(case["duration_sec"]),
        int(case.get("phase_seed", GM_DEFAULT_PHASE_SEED)),
    )
    nsteps = len(t) - 1

    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    for nd in range(5):
        ops.node(nd, 0.0)
    ops.fix(0, 1)
    masses = [w / G_IN_S2 for w in FLOOR_WEIGHT_KIP]
    for nd in range(1, 5):
        ops.mass(nd, float(masses[nd - 1]))

    # Isolator: parallel lead core (ElasticPP) and rubber (Elastic).
    k_lead = qd / max(dy, 1.0e-6)
    ops.uniaxialMaterial("ElasticPP", 1, float(k_lead), float(dy))
    ops.uniaxialMaterial("Elastic", 2, float(kd))
    ops.element("zeroLength", 1, 0, 1, "-mat", 1, 2, "-dir", 1, 1)

    # Superstructure story springs.
    for i in range(3):
        ops.uniaxialMaterial("Elastic", 10 + i, float(STORY_STIFFNESS_KIP_IN[i]))
        ops.element("zeroLength", 10 + i, 1 + i, 2 + i, "-mat", 10 + i, "-dir", 1)

    analyze_first_code = -1
    try:
        eigvals = ops.eigen("-fullGenLapack", 4)
    except Exception:
        eigvals = []
    omegas = sorted(math.sqrt(e) for e in eigvals if e > 0.0)
    if len(omegas) >= 4:
        wi, wj = omegas[1], omegas[3]
        a0 = SUPERSTRUCTURE_DAMPING_RATIO * 2.0 * wi * wj / (wi + wj)
        a1 = SUPERSTRUCTURE_DAMPING_RATIO * 2.0 / (wi + wj)
        ops.rayleigh(a0, 0.0, a1, 0.0)

    ops.timeSeries("Path", 1, "-dt", DT, "-values", *[float(v) for v in ag])
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

    ops.wipeAnalysis()
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 25)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    peak_iso = 0.0
    peak_drift = [0.0, 0.0, 0.0]
    peak_acc = [0.0, 0.0, 0.0, 0.0, 0.0]
    peak_force = 0.0
    completed = 0
    for k in range(nsteps):
        code = ops.analyze(1, DT)
        if code != 0:
            ops.algorithm("ModifiedNewton")
            code = ops.analyze(1, DT)
            ops.algorithm("Newton")
        if analyze_first_code < 0:
            analyze_first_code = int(code)
        if code != 0:
            break
        completed += 1
        disp = [ops.nodeDisp(nd, 1) for nd in range(5)]
        peak_iso = max(peak_iso, abs(disp[1] - disp[0]))
        for i in range(3):
            drift = abs(disp[2 + i] - disp[1 + i]) / STORY_HEIGHT_IN[i]
            peak_drift[i] = max(peak_drift[i], drift)
        ag_k = ag[k + 1]
        for nd in range(1, 5):
            total_acc = ops.nodeAccel(nd, 1) + ag_k
            peak_acc[nd] = max(peak_acc[nd], abs(total_acc))
        peak_force = max(peak_force, abs(ops.eleForce(1, 1)))

    converged = completed == nsteps
    residual_iso = abs(ops.nodeDisp(1, 1)) if converged else float("nan")
    ops.wipe()

    return {
        "case_id": str(case.get("case_id", case.get("name", "case"))),
        "phase_seed": int(case.get("phase_seed", GM_DEFAULT_PHASE_SEED)),
        "peak_isolator_disp_in": float(peak_iso),
        "peak_interstory_drift_ratio": float(max(peak_drift)),
        "peak_floor_acceleration_g": float(max(peak_acc[1:]) / G_IN_S2),
        "peak_base_shear_coeff": float(peak_force / TOTAL_WEIGHT_KIP),
        "residual_isolator_disp_in": float(residual_iso) if converged else None,
        "analyze_return_code": int(analyze_first_code),
        "converged": bool(converged),
        "completed_steps": int(completed),
        "total_steps": int(nsteps),
    }


WORST_METRIC_KEYS = (
    "peak_isolator_disp_in",
    "peak_interstory_drift_ratio",
    "peak_floor_acceleration_g",
    "peak_base_shear_coeff",
    "residual_isolator_disp_in",
)

# Finite sentinel used only in a diagnostic envelope when a case did not
# produce a metric.  The grader rejects any design with all_converged=false;
# keeping the envelope finite prevents non-convergence from becoming a Python
# exception or non-standard JSON Infinity value first.
FAILED_METRIC_VALUE = 1.0e12


def _finite_metric_or_failure(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return FAILED_METRIC_VALUE
    return result if math.isfinite(result) else FAILED_METRIC_VALUE


def evaluate_design(design: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Run every case and return per-case results plus worst-case envelope."""
    per_case = [run_case(design, case) for case in cases]
    all_converged = bool(per_case) and all(bool(c.get("converged", False)) for c in per_case)
    worst = {
        key: max((_finite_metric_or_failure(c.get(key)) for c in per_case), default=FAILED_METRIC_VALUE)
        for key in WORST_METRIC_KEYS
    }
    return {
        "model_version": MODEL_VERSION,
        "per_case": per_case,
        "worst_case": worst,
        "all_converged": bool(all_converged),
    }
