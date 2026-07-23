#!/usr/bin/env bash
# Trusted-CI helper: package a verified calibration lock as a dedicated
# full-digest read-only mount and merge it into the preloaded manifest.
set -euo pipefail

PROBLEM_DIR="${1:?usage: sync_calibration_mount.sh <problem-dir> [bundle]}"
BUNDLE="${2:-${PROBLEM_DIR}/calibration.lock.json}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPLOAD="${REPO_ROOT}/scripts/upload_squashfs.sh"
STAMP="${REPO_ROOT}/scripts/stamp_preloaded_files.py"
BUCKET_PREFIX="${TAIGA_PRELOADED_BUCKET_PREFIX:-gs://anthropic-argonrl-dog-bowl-us-central1-0/biome/environment_files}"

[[ "${CALIBRATION_TRUSTED_CI:-}" == "1" ]] || {
  echo "ERROR: CALIBRATION_TRUSTED_CI=1 is required; author-side locks are not promoted directly." >&2
  exit 1
}
[[ -f "$BUNDLE" ]] || { echo "ERROR: calibration bundle not found: $BUNDLE" >&2; exit 1; }
[[ -n "${TAIGA_ENV_ID:-}" ]] || { echo "ERROR: TAIGA_ENV_ID is required" >&2; exit 1; }

LOCK_SHA="$(python3 - "$BUNDLE" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
)"

if [[ -n "${CALIBRATION_EXPECTED_SHA256:-}" && "$LOCK_SHA" != "$CALIBRATION_EXPECTED_SHA256" ]]; then
  echo "ERROR: calibration lock digest mismatch: expected ${CALIBRATION_EXPECTED_SHA256}, got ${LOCK_SHA}" >&2
  exit 1
fi

TASK_ID="$(basename "$PROBLEM_DIR")"
TASK_ID="${TASK_ID%_taiga}"
REMOTE_NAME="${TASK_ID}/calibration-${LOCK_SHA}.squashfs"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
STAGING="${WORK}/calibration"
SQUASHFS="${WORK}/calibration-${LOCK_SHA}.squashfs"
mkdir -p "$STAGING"
cp "$BUNDLE" "${STAGING}/calibration.lock.json"

mksquashfs "$STAGING" "$SQUASHFS" \
  -noappend -quiet -comp zstd -Xcompression-level 3 -all-time 0 -mkfs-time 0
bash "$UPLOAD" "$SQUASHFS" "$REMOTE_NAME" >/dev/null
REMOTE_PATH="${BUCKET_PREFIX}/${TAIGA_ENV_ID}/${REMOTE_NAME}"

python3 "$STAMP" \
  --problem-dir "$PROBLEM_DIR" \
  --trusted-entry "/mcp_server/calibration::${REMOTE_PATH}::true"

echo "promoted calibration ${LOCK_SHA} -> ${REMOTE_PATH}"
