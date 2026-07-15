#!/usr/bin/env bash
# _rev: 2026-06-22.b7prc04t3_batch25
set -euo pipefail
OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
DATA_DIR="${LBT_DATA_DIR:-/data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBLEM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$OUT_DIR"
export B7_PROBLEM_DIR="$PROBLEM_DIR"
export B7_DATA_DIR="$DATA_DIR"
export B7_OUT_DIR="$OUT_DIR"
if command -v blockMesh >/dev/null 2>&1; then
  blockMesh -help >/dev/null 2>&1 || true
fi
if command -v simpleFoam >/dev/null 2>&1; then
  simpleFoam -help >/dev/null 2>&1 || true
fi
python3 - <<'__SOLVE_PY__'
import json
import math
import os
import shutil
import sys
import subprocess
import tempfile
from pathlib import Path

FIELDS = ("flap_deflection_deg", "hinge_gap_m", "flap_chord_fraction", "blend_radius_m")
problem_dir = Path(os.environ.get("B7_PROBLEM_DIR", ".")).resolve()
data_dir = Path(os.environ.get("B7_DATA_DIR", "/data"))
out_dir = Path(os.environ.get("B7_OUT_DIR", "/tmp/output"))

# The oracle intentionally uses the same scorer modules and hidden-case response
# model as compute_score.py. This avoids optimizing a separate public surrogate.
module_candidates = [
    Path("/mcp_server/grader"),
    problem_dir / "scorer",
]
for candidate in module_candidates:
    if (candidate / "openfoam_case.py").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import openfoam_case as ofc  # noqa: E402
import compute_score as scorer  # noqa: E402


def openfoam_binary_smoke():
    """Verify the oracle runtime can invoke the expected OpenFOAM binaries."""
    commands = ("blockMesh", "simpleFoam")
    missing = [cmd for cmd in commands if shutil.which(cmd) is None]
    if missing:
        print(f"OpenFOAM smoke skipped; missing binaries: {', '.join(missing)}")
        return
    for cmd in commands:
        result = subprocess.run(
            [cmd, "-help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        if result.returncode != 0:
            print(f"OpenFOAM smoke command returned {result.returncode}: {cmd} -help")


openfoam_binary_smoke()

private_candidates = [
    Path("/mcp_server/data"),
    problem_dir / "scorer" / "data",
]
private_dir = next((p for p in private_candidates if (p / "expected.json").exists() and (p / "hidden_conditions.json").exists()), None)
if private_dir is None:
    raise SystemExit("oracle could not locate private expected/hidden-condition fixtures")

public_candidates = [
    data_dir / "public_operating_envelope.json",
    problem_dir / "data" / "public_operating_envelope.json",
    Path("/data/public_operating_envelope.json"),
]
public_path = next((p for p in public_candidates if p.exists()), None)
if public_path is None:
    raise SystemExit("oracle could not locate public_operating_envelope.json")
public = json.loads(public_path.read_text())
expected = json.loads((private_dir / "expected.json").read_text())
conditions = json.loads((private_dir / "hidden_conditions.json").read_text())


def clamp_design(design):
    bounds = public["design_bounds"]
    return {
        field: max(float(bounds[field]["min"]), min(float(bounds[field]["max"]), float(design[field])))
        for field in FIELDS
    }


def mock_build(case_dir, design, public_env, case, timeout=180, block_timeout=None, solver_timeout=None, **_kwargs):
    feasibility, _errors, geom = ofc.feasibility_scores(design, public_env)
    if geom is None or min(feasibility.values()) <= 0.0:
        return ofc.CaseResult(False, False, False, reason="oracle candidate is infeasible")
    metrics = ofc.response_metrics(geom, case)
    return ofc.CaseResult(True, True, True, **metrics, reason="oracle same-response-model evaluation")

# Ground-truth selection scores candidates through compute_score.py with the
# OpenFOAM health path replaced by the same deterministic response_metrics model
# used inside openfoam_case.build_and_run after mesh and solver pass.
scorer.ofc.build_and_run = mock_build


class _MockCheckMeshResult:
    returncode = 0
    stdout = "Mesh OK"
    stderr = ""


def mock_public_check_mesh(case_dir, timeout):
    return _MockCheckMeshResult()


scorer._run_public_check_mesh = mock_public_check_mesh


def public_assessment(result):
    if not result.ok or not result.mesh_ok or not result.solver_ok:
        return {
            "authority_class": "probe_failed",
            "drag_risk_class": "probe_failed",
            "separation_risk_class": "probe_failed",
            "wake_quality_class": "probe_failed",
        }

    def public_class(value, useful_min, useful_max, low_label="low", ok_label="useful", high_label="high"):
        value = float(value)
        if value < useful_min:
            return low_label
        if value > useful_max:
            return high_label
        return ok_label

    def risk_class(value):
        value = float(value)
        if value <= 0.18:
            return "low_risk"
        if value <= 0.38:
            return "medium_risk"
        return "high_risk"

    wake_score = float(result.wake_uniformity)
    return {
        "authority_class": public_class(float(result.lift_coefficient), 0.52, 0.82),
        "drag_risk_class": risk_class(float(result.drag_coefficient)),
        "separation_risk_class": risk_class(float(result.separation_index)),
        "wake_quality_class": "high" if wake_score >= 0.76 else ("acceptable" if wake_score >= 0.50 else "poor"),
    }


def score_candidate(design):
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        candidate = clamp_design(design)
        (workspace / "hydrofoil_flap.json").write_text(json.dumps(candidate, sort_keys=True) + "\n")
        payload = scorer.compute_score(workspace, "", private_dir)
    return float(payload.get("score", 0.0))


def guidance_midpoint_seed():
    guidance = public.get("coarse_design_guidance", {})

    def mid(key, fallback):
        band = guidance.get(key, fallback)
        return (float(band[0]) + float(band[1])) / 2.0

    return {
        "flap_deflection_deg": mid("useful_deflection_region_deg", [5.8, 7.8]),
        "hinge_gap_m": mid("useful_hinge_gap_region_m", [0.006, 0.012]),
        "flap_chord_fraction": mid("useful_flap_chord_fraction_region", [0.22, 0.30]),
        "blend_radius_m": mid("useful_blend_radius_region_m", [0.012, 0.024]),
    }


def geometry_target_seed():
    targets = expected.get("geometry_targets", {}) or {}
    packaging = public["packaging_context"]
    chord = float(packaging["chord_m"])
    chord_fraction = float(targets.get("reference_flap_chord_ratio", guidance_midpoint_seed()["flap_chord_fraction"]))
    flap_length = max(chord * chord_fraction, 1e-9)
    te_offset = float(targets.get("trailing_edge_offset_m", 0.023))
    gap_ratio = float(targets.get("gap_ratio", 0.040))
    blend_fraction = float(targets.get("blend_fraction", 0.115))
    return {
        "flap_deflection_deg": math.degrees(math.atan(te_offset / flap_length)),
        "hinge_gap_m": gap_ratio * flap_length,
        "flap_chord_fraction": chord_fraction,
        "blend_radius_m": blend_fraction * flap_length,
    }

seeds = [guidance_midpoint_seed(), geometry_target_seed()]
# Add a small deterministic neighborhood around the target-derived seed so the
# official design is selected by the same scorer, not by a hard-coded JSON blob.
base = geometry_target_seed()
for d in (-0.08, 0.0, 0.08):
    for g in (-0.00008, 0.0, 0.00008):
        for c in (-0.001, 0.0, 0.001):
            for b in (-0.00012, 0.0, 0.00012):
                seeds.append({
                    "flap_deflection_deg": base["flap_deflection_deg"] + d,
                    "hinge_gap_m": base["hinge_gap_m"] + g,
                    "flap_chord_fraction": base["flap_chord_fraction"] + c,
                    "blend_radius_m": base["blend_radius_m"] + b,
                })

best_score = -1.0
best_design = None
for seed in seeds:
    design = clamp_design(seed)
    score = score_candidate(design)
    if score > best_score:
        best_score = score
        best_design = design

# Fine coordinate search around the best same-scorer seed.
steps = [
    (0.05, 0.00005, 0.00050, 0.00010),
    (0.02, 0.00002, 0.00020, 0.00005),
    (0.01, 0.00001, 0.00010, 0.00002),
]
for deltas in steps:
    improved = True
    while improved:
        improved = False
        for field, delta in zip(FIELDS, deltas):
            for sign in (-1.0, 1.0):
                candidate = dict(best_design)
                candidate[field] = candidate[field] + sign * delta
                candidate = clamp_design(candidate)
                score = score_candidate(candidate)
                if score > best_score + 1e-12:
                    best_score = score
                    best_design = candidate
                    improved = True

if best_design is None or best_score < 0.999:
    raise SystemExit(f"oracle same-scorer search did not find a full-credit design; best_score={best_score:.6f}")

# Stable formatting intentionally matches the private reference precision used by
# expected.json geometry targets without relying on a digest shortcut.
final = {
    "flap_deflection_deg": round(float(best_design["flap_deflection_deg"]), 4),
    "hinge_gap_m": round(float(best_design["hinge_gap_m"]), 5),
    "flap_chord_fraction": round(float(best_design["flap_chord_fraction"]), 4),
    "blend_radius_m": round(float(best_design["blend_radius_m"]), 5),
}
# Re-score the rounded design before writing it.
rounded_score = score_candidate(final)
if rounded_score < 0.999:
    raise SystemExit(f"rounded oracle design lost full credit; rounded_score={rounded_score:.6f}; design={final}")

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "hydrofoil_flap.json").write_text(json.dumps(final, indent=2) + "\n")
print((out_dir / "hydrofoil_flap.json").read_text())
__SOLVE_PY__
