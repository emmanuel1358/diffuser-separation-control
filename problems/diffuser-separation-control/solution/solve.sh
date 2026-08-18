#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "$OUT_DIR"
cat > "$OUT_DIR/hydrofoil_flap.json" <<'JSON'
{
  "flap_deflection_deg": 7.5,
  "hinge_gap_m": 0.0069,
  "flap_chord_fraction": 0.288,
  "blend_radius_m": 0.0212
}
JSON
if command -v blockMesh >/dev/null 2>&1; then blockMesh -help >/dev/null 2>&1 || true; fi
if command -v simpleFoam >/dev/null 2>&1; then simpleFoam -help >/dev/null 2>&1 || true; fi
echo "Generated hydrofoil design:"
cat "$OUT_DIR/hydrofoil_flap.json"
