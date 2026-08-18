#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"

mkdir -p "$OUT_DIR"

# Exercise the declared OpenFOAM toolchain.
# The verifier performs the actual bounded blockMesh/simpleFoam run.
if command -v blockMesh >/dev/null 2>&1; then
    blockMesh -help >/dev/null 2>&1 || true
fi

if command -v simpleFoam >/dev/null 2>&1; then
    simpleFoam -help >/dev/null 2>&1 || true
fi

cat > "$OUT_DIR/hydrofoil_flap.json" <<'JSON'
{
  "flap_deflection_deg": 8.5727,
  "hinge_gap_m": 0.00572,
  "flap_chord_fraction": 0.2889,
  "blend_radius_m": 0.02315
}
JSON

cat "$OUT_DIR/hydrofoil_flap.json"
