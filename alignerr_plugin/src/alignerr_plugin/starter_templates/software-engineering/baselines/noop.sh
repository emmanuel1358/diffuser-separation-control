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
shopt -u dotglob nullglob

if [[ "$(id -u)" -eq 0 ]]; then
  chown -R "${RUBRIC_AGENT_UID:-1000}:${RUBRIC_AGENT_GID:-1000}" "$REPO_DIR"
fi

echo "Prepared unchanged starter baseline at $REPO_DIR"
