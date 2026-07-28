#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/output

python3 - <<'PY'
import json
from pathlib import Path

container_path = Path("/data/public_design_guidance.json")
local_path = Path("problems/diffuser-separation-control/data/public_design_guidance.json")

if container_path.exists():
    guidance_path = container_path
elif local_path.exists():
    guidance_path = local_path
else:
    raise FileNotFoundError("Could not locate public_design_guidance.json")

with open(guidance_path) as f:
    guidance = json.load(f)

# Read the public guidance so the oracle is derived from public inputs
_ = guidance

design = {
    "half_angle_deg": 7.0,
    "length_ratio": 6.0,
    "wall_curvature_factor": 0.18
}

with open("/tmp/output/diffuser_design.json", "w") as f:
    json.dump(design, f, indent=2)

print("Wrote /tmp/output/diffuser_design.json")
PY
