#!/usr/bin/env bash
set -euo pipefail

# Ground-truth oracle entrypoint. It packages the committed reference model as a
# queryable predictor under /tmp/output; it must never read the held-out data in
# /mcp_server or bake a challenge answer key into the artifact.
SOLUTION_DIR="${LBX_SOLUTION_DIR:-$(dirname "$0")}"
exec python "${SOLUTION_DIR}/solution.py"
