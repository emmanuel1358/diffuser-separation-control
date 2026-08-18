#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "$OUT_DIR"

if [ -f "$OUT_DIR/hydrofoil_flap.json" ]; then
    echo "=== Hydrofoil Design ==="
    cat "$OUT_DIR/hydrofoil_flap.json"
else
    echo "No hydrofoil design found in: $OUT_DIR"
    exit 1
fi
