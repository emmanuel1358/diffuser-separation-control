#!/usr/bin/env bash
set -euo pipefail

# Naive baseline: pick arm 0 without exploring. Scores low unless arm 0 happens
# to be near-optimal -- the point is that "submit without interacting" earns ~0.
cat > /tmp/output/policy.py <<'PY'
def load_policy():
    class Policy:
        def reset(self, n_arms, budget):
            self.n_arms = int(n_arms)

        def choose(self):
            return 0

        def observe(self, arm, reward):
            pass

        def recommend(self):
            return 0

    return Policy()
PY
