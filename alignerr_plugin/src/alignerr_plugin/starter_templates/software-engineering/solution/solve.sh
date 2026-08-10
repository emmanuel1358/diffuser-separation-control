#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$OUT_DIR/repo"

if [[ -d /opt/software-engineering-starter ]]; then
  STARTER_DIR=/opt/software-engineering-starter
else
  STARTER_DIR="$TASK_DIR/starter"
fi

mkdir -p "$OUT_DIR" "$REPO_DIR"
shopt -s dotglob nullglob
rm -rf "${REPO_DIR:?}/"*
cp -a "$STARTER_DIR/." "$REPO_DIR/"
cp -a "$SCRIPT_DIR/files/." "$REPO_DIR/"
shopt -u dotglob nullglob

find "$REPO_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$REPO_DIR" -type f -name '*.pyc' -delete

if [[ "$(id -u)" -eq 0 ]]; then
  chown -R "${RUBRIC_AGENT_UID:-1000}:${RUBRIC_AGENT_GID:-1000}" "$REPO_DIR"
fi

echo "Prepared repaired repository at $REPO_DIR"
