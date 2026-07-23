#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier
python - <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, "/mcp_server")
sys.path.insert(0, "/mcp_server/grader")
from grader.compute_score import TASK

result = TASK.grade(
    workspace=Path("/tmp/output"),
    trajectory=None,
    private=Path("/mcp_server/data"),
).to_dict()
Path("/logs/verifier/reward.json").write_text(json.dumps(result))
PY
