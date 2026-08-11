# hidden-env-bandit (example)

A complete, minimal **hidden-environment** task: the agent must infer the best
arm of a multi-armed bandit it can only interact with over a socket, then submit
a policy. It is the reference for `[environment].hidden_env` tasks; read it
alongside [docs/HIDDEN_ENV.md](../../docs/HIDDEN_ENV.md).

## What it demonstrates

- `[environment].hidden_env = "env"` activates the `env_server` (the rubric MCP
  server safely spawns `python -P -m env_server` from root-owned `/` at boot).
- The hidden env lives at `scorer/data/env.py` (baked root-only to
  `/mcp_server/data/env.py`); the agent can call `make_env()` over
  `/tmp/env.sock` but cannot read the source or the per-arm means.
- `BanditEnv._env_public_methods` pins the socket surface to `reset` / `pull` /
  `n_arms`; the grader-only `best_arm` / `mean` are never reachable by the agent.
- The agent uses the public `data/env_client.py` (baked to `/data/env_client.py`).
- The grader uses `PolicyEvaluationTask` to commit the submitted algorithm,
  derive fresh hidden bandit seeds, run paired candidate/control scenarios, and
  score only policies that beat fixed/open-loop controls.

## Run the reference locally

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir examples/hidden-env-bandit
```

The oracle (`solution/solve.sh`) submits a UCB learner and scores near the 0.5
reference anchor. The naive baseline submits a fixed-arm policy and is
certificate-gated to zero.
