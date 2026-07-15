#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "$OUT_DIR"
export TASK_OUTPUT_DIR="$OUT_DIR"
export TASK_DATA_DIR="${TASK_DATA_DIR:-/data}"
export PYTHONPATH="/data:${PYTHONPATH:-}"

if [[ -d /tmp/verifier ]]; then
  exec > >(tee -a /tmp/verifier/transcript.txt) 2> >(tee -a /tmp/verifier/transcript.txt >&2)
fi

PY=/opt/lbx-runtime/.venv/bin/python
[[ -x "$PY" ]] || PY=python3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Oracle: searching base-isolation designs on in-band ground motions..."
"$PY" "$SCRIPT_DIR/oracle_search.py"

echo "Oracle outputs:"
ls -l "$OUT_DIR/isolation_design.json"
