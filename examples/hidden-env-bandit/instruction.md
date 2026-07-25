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

Each `pull(arm)` returns a NOISY sample of that arm's hidden mean reward. Pull
arms enough times to estimate which arm has the highest mean.

## What to submit

Write a Python module to:

```text
/tmp/output/policy.py
```

defining a `load_policy()` factory that returns an object with a `choose()`
method returning the integer index of the arm you believe is best:

```python
def load_policy():
    class Policy:
        def choose(self):
            return 3   # the arm you found to be best
    return Policy()
```

## Scoring

The grader runs your submitted `policy.choose()` against the held-out bandit
(the same hidden means you explored) and scores by how close the chosen arm's
true mean is to the optimal: `1.0` for the best arm, down toward `0.0` for the
worst. The env socket is closed before grading, so bake your decision into
`policy.py` (e.g. hard-code the arm you found, or re-derive it from data you
saved during exploration).
