#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBLEM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

rm -rf "$OUT_DIR/repo"
mkdir -p "$OUT_DIR"
cp -R "$PROBLEM_DIR/starter" "$OUT_DIR/repo"
