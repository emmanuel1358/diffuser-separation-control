# hidden-env-bandit (example)

A complete, minimal **hidden-environment** task: the agent must infer the best
arm of a multi-armed bandit it can only interact with over a socket, then submit
a policy. It is the reference for `[environment].hidden_env` tasks; read it
alongside [docs/HIDDEN_ENV.md](../../docs/HIDDEN_ENV.md).

## What it demonstrates

- `[environment].hidden_env = "env"` activates the `env_server` (the rubric MCP
  server spawns `python -m env_server` at boot).
- The hidden env lives at `scorer/data/env.py` (baked root-only to
  `/mcp_server/data/env.py`); the agent can call `make_env()` over
  `/tmp/env.sock` but cannot read the source or the per-arm means.
- `BanditEnv._env_public_methods` pins the socket surface to `reset` / `pull` /
  `n_arms`; the grader-only `best_arm` / `mean` are never reachable by the agent.
- The agent uses the public `data/env_client.py` (baked to `/data/env_client.py`).
- The grader (`scorer/compute_score.py`) loads the held-out env in-process with
  `grading.load_env_module` and the agent policy with
  `grading.load_submitted_policy`, then scores by regret. The socket is torn
  down (`stop_env_server`) before grading.

## Run the reference locally

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir examples/hidden-env-bandit
```

The oracle (`solution/solve.sh`) explores the bandit and bakes the best arm into
`policy.py` (scores ~1.0); the naive baseline (`baselines/naive.sh`) submits arm
0 without exploring (scores low).
