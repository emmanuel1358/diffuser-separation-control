#!/usr/bin/env bash
# _rev: 2026-06-16.b7prc04t3_batch6
set -euo pipefail
OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --design) shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$OUT_DIR"
cat > "$OUT_DIR/render_disabled.txt" <<'TXT'
Render is intentionally disabled for openfoam-hydrofoil-flap batch6.
The benchmark is scored through required JSON outputs and hidden deterministic OpenFOAM verifier metrics. No reviewer MP4 artifact is declared in task.toml.
TXT
