#!/usr/bin/env bash
# _rev: 2026-06-16.b7prc04t3_batch6
set -euo pipefail
OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "$OUT_DIR"
cat > "$OUT_DIR/hydrofoil_flap.json" <<'JSON'
{
  "flap_deflection_deg": 6.8,
  "hinge_gap_m": 0.009,
  "flap_chord_fraction": 0.26,
  "blend_radius_m": 0.018
}
JSON
cat "$OUT_DIR/hydrofoil_flap.json"
