#!/usr/bin/env bash
set -euo pipefail

# Reference solution: explore the hidden bandit over the socket, find the best
# arm empirically, and bake it into the submitted policy.
python - <<'PY'
import sys
sys.path.insert(0, "/data")
from env_client import estimate_best_arm

best = estimate_best_arm(pulls_per_arm=400)
with open("/tmp/output/policy.py", "w") as fh:
    fh.write(
        "def load_policy():\n"
        "    class Policy:\n"
        "        def choose(self):\n"
        f"            return {best}\n"
        "    return Policy()\n"
    )
print(f"reference policy picks arm {best}")
PY
