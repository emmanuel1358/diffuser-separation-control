#!/usr/bin/env bash
set -euo pipefail

cat > /tmp/output/policy.py <<'PY'
def act(obs):
    # Replace with a real reference policy for each task.
    return 0
PY
