#!/usr/bin/env bash
# Weak baseline: copy the intentionally too-soft starter design and run the
# public probe on it. This isolation system minimizes superstructure demand but
# its isolator displacement exceeds the seismic moat, so the moat-pounding gate
# drives the score to ~0. Used to confirm a trivial submission scores low.
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "$OUT_DIR"
export PYTHONPATH="/data:${PYTHONPATH:-}"

PY=/opt/lbx-runtime/.venv/bin/python
[[ -x "$PY" ]] || PY=python3

cp /data/isolation_starter.json "$OUT_DIR/isolation_design.json"

echo "Naive baseline wrote too-soft design to $OUT_DIR"
