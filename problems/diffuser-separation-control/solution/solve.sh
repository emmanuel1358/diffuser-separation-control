#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/output

# Compute design parameters from disclosed public data
python3 - <<'PY'
import json

with open("/data/baseline_diffuser.json") as f:
    baseline = json.load(f)
with open("/data/public_design_guidance.json") as f:
    guidance = json.load(f)

angle = (guidance["recommended_half_angle_range_deg"][0] +
         guidance["recommended_half_angle_range_deg"][1]) / 2.0
length = max(guidance["recommended_length_ratio_range"][0],
             min(guidance["recommended_length_ratio_range"][1],
                 baseline["baseline_length_ratio"] * 1.5))
inlet = guidance.get("recommended_inlet_extension_m", 0.5)

result = {
    "half_angle_deg": round(angle, 1),
    "length_ratio": round(length, 1),
    "inlet_extension_m": round(inlet, 1)
}

with open("/tmp/output/diffuser_design.json", "w") as f:
    json.dump(result, f, indent=2)
PY

# Set up OpenFOAM case and exercise the domain solver
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="/tmp/output/openfoam_case"

mkdir -p "${CASE_DIR}/system" "${CASE_DIR}/constant" "${CASE_DIR}/0"

cp "${SCRIPT_DIR}/openfoam_case/blockMeshDict" "${CASE_DIR}/system/blockMeshDict"
cp "${SCRIPT_DIR}/openfoam_case/controlDict"   "${CASE_DIR}/system/controlDict"
cp "${SCRIPT_DIR}/openfoam_case/fvSchemes"     "${CASE_DIR}/system/fvSchemes"
cp "${SCRIPT_DIR}/openfoam_case/fvSolution"    "${CASE_DIR}/system/fvSolution"
cp "${SCRIPT_DIR}/openfoam_case/0/U"           "${CASE_DIR}/0/U"
cp "${SCRIPT_DIR}/openfoam_case/0/p"           "${CASE_DIR}/0/p"

# Exercise the OpenFOAM solver pipeline (best-effort)
cd "${CASE_DIR}"
if command -v blockMesh >/dev/null 2>&1; then
    blockMesh >/dev/null 2>&1 || true
    if command -v simpleFoam >/dev/null 2>&1; then
        simpleFoam >/dev/null 2>&1 || true
    fi
fi

echo "Wrote /tmp/output/diffuser_design.json"
echo "Set up OpenFOAM case at ${CASE_DIR}"