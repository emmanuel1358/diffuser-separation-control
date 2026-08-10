#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
if [[ -n "${LBX_SOLUTION_DIR:-}" ]]; then
    TASK_DIR="$(cd "$LBX_SOLUTION_DIR/.." && pwd)"
else
    TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
REPO_DIR="$OUT_DIR/repo"

rm -rf "$REPO_DIR"
mkdir -p "$REPO_DIR"
cp -a "$TASK_DIR/starter/repo/." "$REPO_DIR/"
cp -a "$TASK_DIR/solution/files/." "$REPO_DIR/"
