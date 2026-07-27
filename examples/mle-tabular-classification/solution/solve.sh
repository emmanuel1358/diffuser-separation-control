#!/usr/bin/env bash
set -euo pipefail

# Reference solution: package the committed model as a queryable predictor. The
# grader commits that artifact, then evaluates it on a fresh private challenge,
# so no challenge answer is baked in here.
SOLUTION_DIR="${LBX_SOLUTION_DIR:-$(dirname "$0")}"
exec python "${SOLUTION_DIR}/solution.py"
