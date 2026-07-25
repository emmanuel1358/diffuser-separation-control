#!/usr/bin/env bash
set -euo pipefail

# Naive baseline: pick arm 0 without exploring. Scores low unless arm 0 happens
# to be near-optimal -- the point is that "submit without interacting" earns ~0.
cat > /tmp/output/policy.py <<'PY'
def load_policy():
    class Policy:
        def choose(self):
            return 0
    return Policy()
PY
