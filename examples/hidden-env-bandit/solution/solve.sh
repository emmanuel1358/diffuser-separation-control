#!/usr/bin/env bash
set -euo pipefail

# Reference solution: package the committed UCB policy. The grader commits this
# artifact, then runs it on fresh hidden bandit seeds; no best-arm value is
# baked in.
OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
SOLUTION_DIR="${LBX_SOLUTION_DIR:-$(dirname "$0")}"
mkdir -p "${OUTPUT_DIR}"
cp "${SOLUTION_DIR}/policy.py" "${OUTPUT_DIR}/policy.py"
echo "reference wrote a queryable UCB policy"
