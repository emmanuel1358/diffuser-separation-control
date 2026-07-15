#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier
python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "/mcp_server")
from grader.compute_score import compute_score

score = compute_score(Path("/tmp/output"), None, Path("/mcp_server/data"))
Path("/logs/verifier/reward.txt").write_text(str(score))
PY
