"""OpenFOAM case helper for openfoam-hydrofoil-flap.

# _rev: 2026-06-16.b7prc04t3_batch6

The verifier uses a deterministic two-dimensional marine hydrofoil flap test
section. The submitted JSON controls only aft-flap geometry. Mesh generation,
solver settings, operating cases, and scoring anchors stay fixed by this module
and private scorer data. OpenFOAM is used as a reproducible mesh/solver health
check; the aerodynamic metrics are deterministic engineering summaries of the
resulting geometry and hidden operating point so grading remains cheap and
stable while still requiring meshable, solver-compatible OpenFOAM cases. Hidden
cases remain private, while public calibration data gives coarse design guidance.
"""

from __future__ import annotations

import math
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIELDS = ("flap_deflection_deg", "hinge_gap_m", "flap_chord_fraction", "blend_radius_m")
OF_ENV = "source /etc/solver-envs.d/openfoam.sh >/dev/null 2>&1"
SPAN = 0.05


@dataclass
class Geometry:
    flap_deflection_deg: float
    hinge_gap_m: float
    flap_chord_fraction: float
    blend_radius_m: float
    chord_m: float
    upstream_length_m: float
    wake_length_m: float
    test_section_height_m: float
    flap_length_m: float
    hinge_x_m: float
    trailing_edge_offset_m: float
    blockage_fraction: float
    gap_ratio: float
    blend_fraction: float
    minimum_clearance_m: float
    total_length_m: float


@dataclass
class CaseResult:
    ok: bool
    mesh_ok: bool
    solver_ok: bool
    lift_coefficient: float = 0.0
    drag_coefficient: float = 0.0
    separation_index: float = 1.0
    wake_uniformity: float = 0.0
    trailing_edge_offset_m: float = 0.0
    blockage_fraction: float = 0.0
    reason: str = ""


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def to_geometry(design: dict[str, Any], public: dict[str, Any]) -> Geometry:
    packaging = public["packaging_context"]
    chord = float(packaging["chord_m"])
    upstream = float(packaging["upstream_length_m"])
    wake = float(packaging["wake_length_m"])
    height = float(packaging["test_section_height_m"])
    deflection = float(design["flap_deflection_deg"])
    gap = float(design["hinge_gap_m"])
    chord_fraction = float(design["flap_chord_fraction"])
    blend = float(design["blend_radius_m"])
    flap_len = chord * chord_fraction
    hinge_x = upstream + chord - flap_len
    te_offset = flap_len * math.tan(math.radians(deflection))
    min_clearance = height - te_offset - gap
    return Geometry(
        flap_deflection_deg=deflection,
        hinge_gap_m=gap,
        flap_chord_fraction=chord_fraction,
        blend_radius_m=blend,
        chord_m=chord,
        upstream_length_m=upstream,
        wake_length_m=wake,
        test_section_height_m=height,
        flap_length_m=flap_len,
        hinge_x_m=hinge_x,
        trailing_edge_offset_m=te_offset,
        blockage_fraction=te_offset / max(height, 1e-9),
        gap_ratio=gap / max(flap_len, 1e-9),
        blend_fraction=blend / max(flap_len, 1e-9),
        minimum_clearance_m=min_clearance,
        total_length_m=upstream + chord + wake,
    )


def feasibility_scores(design: dict[str, Any] | None, public: dict[str, Any]) -> tuple[dict[str, float], list[str], Geometry | None]:
    scores = {"required_fields": 0.0, "finite_numeric": 0.0, "public_ranges": 0.0, "derived_geometry": 0.0}
    errors: list[str] = []
    if not isinstance(design, dict):
        return scores, ["hydrofoil_flap.json must contain a JSON object"], None

    present = [field for field in FIELDS if field in design]
    scores["required_fields"] = len(present) / len(FIELDS)
    if len(present) != len(FIELDS):
        errors.append("missing one or more required hydrofoil flap geometry fields")
        return scores, errors, None

    numeric: dict[str, float] = {}
    finite_count = 0
    for field in FIELDS:
        try:
            value = float(design[field])
        except (TypeError, ValueError):
            errors.append(f"{field} is not numeric")
            continue
        numeric[field] = value
        if math.isfinite(value):
            finite_count += 1
        else:
            errors.append(f"{field} is not finite")
    scores["finite_numeric"] = finite_count / len(FIELDS)
    if finite_count != len(FIELDS):
        return scores, errors, None

    bounds = public["design_bounds"]
    range_parts = []
    for field in FIELDS:
        lo = float(bounds[field]["min"])
        hi = float(bounds[field]["max"])
        val = numeric[field]
        span = hi - lo
        if lo <= val <= hi:
            range_parts.append(1.0)
        else:
            distance = min(abs(val - lo), abs(val - hi))
            range_parts.append(_clamp(1.0 - distance / (0.35 * span)))
            errors.append(f"{field}={val} outside public range [{lo}, {hi}]")
    scores["public_ranges"] = sum(range_parts) / len(range_parts)

    geom = to_geometry(numeric, public)
    packaging = public["packaging_context"]
    min_clearance = float(packaging["minimum_clearance_m"])
    max_te = float(packaging["maximum_trailing_edge_offset_m"])
    block_lo, block_hi = [float(x) for x in packaging["target_blockage_fraction_band"]]

    derived_parts = []
    if geom.minimum_clearance_m >= min_clearance:
        derived_parts.append(1.0)
    else:
        derived_parts.append(_clamp(geom.minimum_clearance_m / max(min_clearance, 1e-9)))
        errors.append("flap geometry violates minimum test-section clearance")

    if 0.0 <= geom.trailing_edge_offset_m <= max_te:
        derived_parts.append(1.0)
    else:
        derived_parts.append(_clamp(1.0 - (geom.trailing_edge_offset_m - max_te) / 0.050))
        errors.append("trailing-edge offset exceeds packaging limit")

    if geom.blend_radius_m <= 0.22 * geom.flap_length_m:
        derived_parts.append(1.0)
    else:
        derived_parts.append(_clamp(1.0 - (geom.blend_radius_m - 0.22 * geom.flap_length_m) / 0.030))
        errors.append("blend radius too large for the submitted flap chord")

    if 0.015 <= geom.gap_ratio <= 0.14:
        derived_parts.append(1.0)
    else:
        distance = min(abs(geom.gap_ratio - 0.015), abs(geom.gap_ratio - 0.14))
        derived_parts.append(_clamp(1.0 - distance / 0.10))
        errors.append("hinge gap ratio is outside the meshable flap range")

    if block_lo * 0.35 <= geom.blockage_fraction <= block_hi * 1.35:
        derived_parts.append(1.0)
    else:
        derived_parts.append(_clamp(1.0 - abs(geom.blockage_fraction - 0.24) / 0.24))
        errors.append("flap blockage is outside the plausible control-authority envelope")

    scores["derived_geometry"] = sum(derived_parts) / len(derived_parts)
    return scores, errors, geom


def response_metrics(geom: Geometry, case: dict[str, Any]) -> dict[str, float]:
    """Smooth deterministic hydrofoil-flap response model for one hidden case."""
    speed = float(case["tow_speed_m_per_s"])
    trim = float(case["trim_bias_deg"])
    submergence = float(case.get("submergence_multiplier", 1.0))
    re_eff = float(case.get("effective_reynolds", 6500.0))
    speed_factor = speed / 3.4
    reynolds_factor = re_eff / 6500.0

    effective_deflection = max(0.0, geom.flap_deflection_deg + 0.55 * trim)
    clearance = geom.minimum_clearance_m * submergence
    block = geom.blockage_fraction
    gap = geom.hinge_gap_m
    gap_ratio = geom.gap_ratio
    blend = geom.blend_radius_m
    blend_fraction = geom.blend_fraction

    deflection_limit = 8.15 + 2.8 * (blend_fraction - 0.085) / 0.09
    deflection_limit -= 1.2 * max(0.0, gap_ratio - 0.070) / 0.050
    deflection_limit += 0.35 * (submergence - 1.0)
    deflection_limit -= 0.48 * (speed_factor - 1.0) + 0.18 * (reynolds_factor - 1.0)

    separation = max(0.0, (effective_deflection - deflection_limit) / 4.6) ** 1.55
    separation += max(0.0, (block - 0.32) / 0.20) ** 1.45
    separation += max(0.0, (0.105 - clearance) / 0.050) ** 1.35 * 0.55
    separation += max(0.0, (0.006 - gap) / 0.007) ** 1.20 * 0.18
    separation += max(0.0, (gap - 0.014) / 0.012) ** 1.10 * 0.10
    separation -= min(0.16, max(0.0, (blend - 0.010) / 0.018) * 0.12)
    separation = _clamp(separation)

    lift = 0.165 + 0.052 * effective_deflection + 1.05 * (geom.flap_chord_fraction - 0.18) + 0.46 * block
    lift -= 0.36 * separation + 1.40 * abs(gap - 0.0085) + 0.030 * abs(speed_factor - 1.0)
    lift_coefficient = _clamp(lift, 0.0, 0.92)

    drag = 0.048 + 0.0064 * (effective_deflection / 6.8) ** 2
    drag += 0.046 * (geom.flap_chord_fraction - 0.22) ** 2 / 0.012
    drag += 0.073 * separation + 0.38 * max(0.0, gap - 0.011) + 0.17 * max(0.0, 0.006 - gap)
    drag += 0.014 * abs(speed_factor - 1.0)
    drag_coefficient = _clamp(drag, 0.0, 0.40)

    wake = 0.925 - 0.36 * separation - 0.52 * abs(gap - 0.0085)
    wake -= 0.11 * max(0.0, block - 0.30) + 0.025 * abs(speed_factor - 1.0)
    wake_uniformity = _clamp(wake)

    return {
        "lift_coefficient": lift_coefficient,
        "drag_coefficient": drag_coefficient,
        "separation_index": separation,
        "wake_uniformity": wake_uniformity,
        "trailing_edge_offset_m": geom.trailing_edge_offset_m,
        "blockage_fraction": geom.blockage_fraction,
    }


def _fh(cls: str, obj: str) -> str:
    return f"FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}"


def _block_mesh_dict(geom: Geometry) -> str:
    """Return a conformal 2D water-tunnel blockMeshDict.

    The bottom wall represents a simplified hydrofoil/flap surface. It is flat
    through the forward chord, then rises smoothly through a blend station and
    aft flap station according to the submitted deflection. The far-field top
    wall is fixed, so aggressive flaps reduce clearance and can fail mesh or
    solver health instead of receiving free authority.
    """
    x0 = 0.0
    x1 = geom.upstream_length_m
    x2 = geom.hinge_x_m
    blend_dx = max(0.015, min(geom.blend_radius_m * 2.0, geom.flap_length_m * 0.45))
    x3 = min(geom.upstream_length_m + geom.chord_m - 0.010, x2 + blend_dx)
    x4 = geom.upstream_length_m + geom.chord_m
    x5 = geom.total_length_m
    yb0 = 0.0
    yb1 = 0.0
    yb2 = geom.hinge_gap_m * 0.25
    frac = max(0.0, min(1.0, (x3 - x2) / max(geom.flap_length_m, 1e-9)))
    yb3 = geom.hinge_gap_m * 0.25 + frac * geom.trailing_edge_offset_m
    yb4 = geom.trailing_edge_offset_m
    yb5 = geom.trailing_edge_offset_m * 0.42
    top = geom.test_section_height_m
    xs = [x0, x1, x2, x3, x4, x5]
    ybs = [yb0, yb1, yb2, yb3, yb4, yb5]
    yts = [top] * len(xs)

    verts = []
    for z in (0.0, SPAN):
        for x, yb, yt in zip(xs, ybs, yts):
            verts.append((x, yb, z))
            verts.append((x, yt, z))

    def b(j: int, zlayer: int) -> int:
        return zlayer * len(xs) * 2 + j * 2

    def t(j: int, zlayer: int) -> int:
        return b(j, zlayer) + 1

    vlines = "\n".join(f"    ({x:.8f} {y:.8f} {z:.8f})" for x, y, z in verts)
    blocks = []
    hydrofoil_faces = []
    farfield_faces = []
    front_faces = []
    back_faces = []
    min_height = max(0.030, min(yt - yb for yb, yt in zip(ybs, yts)))
    global_ny = max(14, min(34, int(round((geom.test_section_height_m - min(ybs)) / 0.006))))

    for j in range(len(xs) - 1):
        dx = xs[j + 1] - xs[j]
        nx = max(8, min(44, int(round(dx / 0.024))))
        blocks.append(
            f"    hex ({b(j,0)} {b(j+1,0)} {t(j+1,0)} {t(j,0)} {b(j,1)} {b(j+1,1)} {t(j+1,1)} {t(j,1)}) "
            f"({nx} {global_ny} 1) simpleGrading (1 1 1)"
        )
        hydrofoil_faces.append(f"        ({b(j,0)} {b(j+1,0)} {b(j+1,1)} {b(j,1)})")
        farfield_faces.append(f"        ({t(j,0)} {t(j,1)} {t(j+1,1)} {t(j+1,0)})")
        front_faces.append(f"        ({b(j,0)} {t(j,0)} {t(j+1,0)} {b(j+1,0)})")
        back_faces.append(f"        ({b(j,1)} {b(j+1,1)} {t(j+1,1)} {t(j,1)})")

    inlet_face = f"        ({b(0,0)} {t(0,0)} {t(0,1)} {b(0,1)})"
    outlet_face = f"        ({b(len(xs)-1,0)} {b(len(xs)-1,1)} {t(len(xs)-1,1)} {t(len(xs)-1,0)})"
    front_back = "\n".join(front_faces + back_faces)
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices
(
{vlines}
);
blocks
(
{chr(10).join(blocks)}
);
edges ();
boundary
(
    inlet {{ type patch; faces (\n{inlet_face}\n    ); }}
    outlet {{ type patch; faces (\n{outlet_face}\n    ); }}
    hydrofoilFlap {{ type wall; faces (\n{chr(10).join(hydrofoil_faces)}\n    ); }}
    farfield {{ type wall; faces (\n{chr(10).join(farfield_faces)}\n    ); }}
    frontAndBack {{ type empty; faces (\n{front_back}\n    ); }}
);
mergePatchPairs ();
"""


def write_case(case_dir: Path, design: dict[str, Any], public: dict[str, Any], case: dict[str, Any]) -> Geometry:
    case_dir = Path(case_dir)
    for sub in ("system", "constant", "0"):
        (case_dir / sub).mkdir(parents=True, exist_ok=True)
    geom = to_geometry(design, public)
    u = float(case["tow_speed_m_per_s"])
    re_eff = float(case.get("effective_reynolds", 6500.0))
    nu = u * max(geom.chord_m, 1e-4) / max(re_eff, 1.0)
    iterations = int(case.get("iterations", 90))

    (case_dir / "system/blockMeshDict").write_text(_block_mesh_dict(geom))
    (case_dir / "constant/transportProperties").write_text(
        _fh("dictionary", "transportProperties") + f"\ntransportModel Newtonian;\nnu {nu:.10e};\n"
    )
    (case_dir / "constant/turbulenceProperties").write_text(
        _fh("dictionary", "turbulenceProperties") + "\nsimulationType laminar;\n"
    )
    (case_dir / "0/U").write_text(
        _fh("volVectorField", "U") + f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({u:.8f} 0 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform ({u:.8f} 0 0); }}
    outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({u:.8f} 0 0); }}
    hydrofoilFlap {{ type noSlip; }}
    farfield {{ type slip; }}
    frontAndBack {{ type empty; }}
}}
"""
    )
    (case_dir / "0/p").write_text(
        _fh("volScalarField", "p") + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    hydrofoilFlap { type zeroGradient; }
    farfield { type zeroGradient; }
    frontAndBack { type empty; }
}
"""
    )
    (case_dir / "system/controlDict").write_text(
        _fh("dictionary", "controlDict") + f"""
application simpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {iterations};
deltaT 1; writeControl timeStep; writeInterval {iterations}; purgeWrite 0;
writeFormat ascii; writePrecision 8; timeFormat general; timePrecision 8;
runTimeModifiable false;
"""
    )
    (case_dir / "system/fvSchemes").write_text(
        _fh("dictionary", "fvSchemes") + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
    )
    (case_dir / "system/fvSolution").write_text(
        _fh("dictionary", "fvSolution") + """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.05; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    residualControl { p 1e-5; U 1e-6; }
}
relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; }
}
"""
    )
    return geom


def _run(case_dir: Path, cmd: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", f"{OF_ENV}; cd {shlex.quote(str(case_dir))}; {cmd}"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _case_result(ok: bool, mesh_ok: bool, solver_ok: bool, metrics: dict[str, float] | None = None, *, reason: str = "") -> CaseResult:
    metrics = metrics or {}
    return CaseResult(
        ok=ok,
        mesh_ok=mesh_ok,
        solver_ok=solver_ok,
        lift_coefficient=float(metrics.get("lift_coefficient", 0.0)),
        drag_coefficient=float(metrics.get("drag_coefficient", 0.0)),
        separation_index=float(metrics.get("separation_index", 1.0)),
        wake_uniformity=float(metrics.get("wake_uniformity", 0.0)),
        trailing_edge_offset_m=float(metrics.get("trailing_edge_offset_m", 0.0)),
        blockage_fraction=float(metrics.get("blockage_fraction", 0.0)),
        reason=reason,
    )


def build_and_run(
    case_dir: Path,
    design: dict[str, Any],
    public: dict[str, Any],
    case: dict[str, Any],
    *,
    timeout: int = 105,
    block_timeout: int | None = None,
    solver_timeout: int | None = None,
) -> CaseResult:
    """Build and run one bounded OpenFOAM case.

    Batch6 keeps a bounded wall-clock profile. blockMesh and simpleFoam have
    separate budgets so four sequential hidden cases cannot consume the whole
    verifier timeout before setup and teardown overhead.
    """
    try:
        geom = write_case(case_dir, design, public, case)
    except Exception as exc:  # noqa: BLE001 - metadata should capture cause
        return _case_result(False, False, False, reason=f"case write failed: {exc}")
    metrics = response_metrics(geom, case)
    timeout = max(1, int(timeout))
    block_budget = max(1, min(int(block_timeout if block_timeout is not None else 35), timeout))
    solver_budget = max(1, min(int(solver_timeout if solver_timeout is not None else max(1, timeout - block_budget)), timeout))
    r = _run(case_dir, "blockMesh", block_budget)
    if r.returncode != 0:
        return _case_result(False, False, False, metrics, reason=f"blockMesh failed: {(r.stderr or r.stdout)[-400:]}")
    r = _run(case_dir, "simpleFoam", solver_budget)
    blob = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        return _case_result(False, True, False, metrics, reason=f"simpleFoam failed: {blob[-400:]}")
    if "FOAM FATAL" in blob or re.search(r"\b(nan|inf|-nan)\b", blob.lower()):
        return _case_result(False, True, False, metrics, reason="solver emitted fatal/NaN/inf")
    return _case_result(True, True, True, metrics, reason="ok")
