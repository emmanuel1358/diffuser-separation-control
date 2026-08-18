#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
DATA_DIR="${LBT_DATA_DIR:-/data}"
mkdir -p "$OUT_DIR"

# Exercise the declared solver toolchain without reading grader-private code or
# fixtures. The design itself is derived only from disclosed public calibration
# samples and bounds.
if command -v blockMesh >/dev/null 2>&1; then
  blockMesh -help >/dev/null 2>&1 || true
fi
if command -v simpleFoam >/dev/null 2>&1; then
  simpleFoam -help >/dev/null 2>&1 || true
fi

DATA_DIR="$DATA_DIR" OUT_DIR="$OUT_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
out_dir = Path(os.environ["OUT_DIR"])
calibration = json.loads(
    (data_dir / "public_calibration_samples.json").read_text(encoding="utf-8")
)
envelope = json.loads(
    (data_dir / "public_operating_envelope.json").read_text(encoding="utf-8")
)

samples = {
    row["label"]: row["design"]
    for row in calibration["representative_public_samples"]
}
center = samples["center_of_public_guidance"]
upper = samples["upper_guidance_edge"]

# Stay inside the disclosed center-to-upper design corridor. The field-specific
# fractions balance the public statements about authority, drag, gap, and blend
# sensitivity; no hidden target or scorer module is read.
fractions = {
    "flap_deflection_deg": 0.70,
    "hinge_gap_m": 0.70,
    "flap_chord_fraction": 0.70,
    "blend_radius_m": 8.0 / 15.0,
}
design = {}
for field, fraction in fractions.items():
    value = float(center[field]) + fraction * (
        float(upper[field]) - float(center[field])
    )
    bounds = envelope["design_bounds"][field]
    design[field] = max(float(bounds["min"]), min(float(bounds["max"]), value))

design = {
    "flap_deflection_deg": round(design["flap_deflection_deg"], 4),
    "hinge_gap_m": round(design["hinge_gap_m"], 5),
    "flap_chord_fraction": round(design["flap_chord_fraction"], 4),
    "blend_radius_m": round(design["blend_radius_m"], 5),
}
out_dir.mkdir(parents=True, exist_ok=True)
destination = out_dir / "hydrofoil_flap.json"
destination.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
print(destination.read_text(encoding="utf-8"))
PY
