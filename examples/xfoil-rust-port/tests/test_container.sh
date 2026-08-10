#!/usr/bin/env bash
set -euo pipefail

PROBLEM_DIR="${PROBLEM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${XFOIL_TEST_IMAGE:-local/xfoil-rust-port:refresh}"

docker run --rm \
  --platform linux/amd64 \
  --cap-add SYS_PTRACE \
  --mount "type=bind,src=$PROBLEM_DIR,dst=/test-task,readonly" \
  "$IMAGE" \
  python3 -I -B /test-task/tests/test_grading.py
