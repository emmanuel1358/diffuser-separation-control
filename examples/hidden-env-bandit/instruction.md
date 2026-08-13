# Hidden Bandit: find the best arm

You are connected to a hidden multi-armed bandit. You CANNOT read its source or
its reward distribution; you can only interact with it over a local socket.

## Interact with the environment

A client library is provided at `/data/env_client.py`. The bandit listens on a
Unix socket at `/tmp/env.sock`.

```python
import sys
sys.path.insert(0, "/data")
from env_client import BanditEnv

with BanditEnv() as env:
    n = env.n_arms()            # number of arms
    reward = env.pull(0)        # noisy reward for pulling arm 0
```

Each `pull(arm)` returns a noisy sample. Use this training environment to develop
and test an exploration algorithm; the grader evaluates the submitted algorithm
on fresh hidden bandit instances selected after your artifact is committed.

## What to submit

Write a Python module to:

```text
/tmp/output/policy.py
```

defining a `load_policy()` factory that returns an object with:

- `reset(n_arms, budget)`
- `choose() -> int`
- `observe(arm, reward)`
- `recommend() -> int`

```python
def load_policy():
    class Policy:
        def reset(self, n_arms, budget):
            self.n_arms = n_arms

        def choose(self):
            return 0

        def observe(self, arm, reward):
            pass

        def recommend(self):
            return 0
    return Policy()
```

## Scoring

The grader commits `policy.py`, derives secret challenge seeds, and gives the
policy a fixed pull budget on each fresh bandit. Quality is the final recommended
arm's normalized true mean. The policy must beat trusted fixed/open-loop controls
across the paired scenarios; constant, seed-indexed, crashing, or observation-
independent policies receive zero. Certified policies retain their continuous
quality reward. Each policy method call has a 2-second deadline, and all policy
method calls across grading share a 900-second cumulative compute budget.
