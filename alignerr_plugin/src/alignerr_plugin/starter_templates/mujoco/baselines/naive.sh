#!/usr/bin/env bash
set -euo pipefail

cat > /tmp/output/policy.py <<'PY'
def act(obs):
    return 0
PY
