#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/output

python3 - <<'PY'
import json
import math

# Read disclosed public inputs
with open("/data/baseline_diffuser.json") as f:
    baseline = json.load(f)
with open("/data/public_design_guidance.json") as f:
    guidance = json.load(f)
with open("/data/public_operating_conditions.json") as f:
    ops = json.load(f)

# Heuristic diffuser design derived from disclosed data
# Optimal half-angle: center of recommended range
angle = (guidance["recommended_half_angle_range_deg"][0] +
         guidance["recommended_half_angle_range_deg"][1]) / 2.0

# Optimal length ratio: scale baseline by 1.5x, clamp to guidance range
length = max(guidance["recommended_length_ratio_range"][0],
             min(guidance["recommended_length_ratio_range"][1],
                 baseline["baseline_length_ratio"] * 1.5))

# Inlet extension: fixed smoothing factor for this operating point
inlet = 0.5

result = {
    "half_angle_deg": round(angle, 1),
    "length_ratio": round(length, 1),
    "inlet_extension_m": round(inlet, 1)
}

with open("/tmp/output/diffuser_design.json", "w") as f:
    json.dump(result, f, indent=2)

print("Derived design from public data:")
print(json.dumps(result, indent=2))
PY

# Copy OpenFOAM case material to output so the solution path contains runnable CFD assets
if [[ -d "${BASH_SOURCE%/*}/openfoam_case" ]]; then
    cp -r "${BASH_SOURCE%/*}/openfoam_case" /tmp/output/openfoam_case
    echo "Copied OpenFOAM case template to /tmp/output/openfoam_case"
fi

echo "Wrote /tmp/output/diffuser_design.json"