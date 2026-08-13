---
name: hidden-env-tasks
description: Authors simulation and interaction tasks where agents probe hidden dynamics over an RPC socket and submit a policy. Use for hidden_env env/hybrid, env_server, /tmp/env.sock, env.py, env_client.py, or load_submitted_policy; not static-data tasks.
---

# Hidden-environment (simulation) tasks

Use this for **simulation / interaction based** tasks: the agent must *probe* a
hidden environment over an episode (call `reset`/`step`/custom methods, observe
responses, infer dynamics) and then submit a policy that is graded against the
held-out env. Full guide: `docs/HIDDEN_ENV.md`. Reference task:
`examples/hidden-env-bandit/`.

## When to use this

ALL of:
- the task is interaction/simulation based (probing the env is the task, not
  transforming a static input);
- the env's dynamics / reward MUST stay hidden (reading them would shortcut it);
- the agent explores live during the rollout, then its decision is frozen + graded.

Do NOT use it for static-data tasks (ML on a fixed dataset, CFD/structures on a
fixed case, a one-shot artifact) -- ship data under `data/` and grade the
artifact; the env server is pure overhead there. `hidden_env` is **orthogonal to
`task_type`**: an ml / mujoco / cfd / structures / software_engineering task can
opt in.

For software tasks, do not use `hidden_env` as a substitute for a service graph.
Multi-service tasks declare service-owned artifacts and an isolated verifier,
then trusted tooling packages them as an outer capsule. Declared SSE MCP works
on Taiga only through that capsule's audited service-DNS proxy; unbundled SSE is
rejected.

## Author contract

1. Enable it: `task.toml`
```toml
[environment]
hidden_env = "env"      # or "hybrid" (also ship static data/ files)
```
2. Hidden env at `scorer/data/env.py` (baked root-only to `/mcp_server/data/env.py`):
```python
def make_env(**env_kwargs):
    return MyEnv(**env_kwargs)        # any public methods: reset/step/custom
```
3. Hide the answer: keep oracle / target / budget methods OFF the socket and
   declare the required surface (`_env_public_methods` fails CLOSED under the
   strict config). Exposed methods use explicit keyword signatures, not
   unrestricted `**kwargs`:
```python
class MyEnv:
    _env_public_methods = frozenset({"reset", "step"})
    def best_arm(self): ...           # grader-only; never socket-reachable
```
4. Ship the client: copy `grader/src/env_server/env_client.py` to
   `data/env_client.py` (baked to `/data/env_client.py`); add a thin subclass.
5. Grade in-process (`scorer/compute_score.py`):
```python
from grading import AgentFault, load_env_module, load_submitted_policy

def compute_score(workspace, trajectory, private):
    env = load_env_module(private / "env.py").make_env(seed=0)   # held-out, root-only
    policy = load_submitted_policy(workspace / "policy.py")      # sandboxed worker
    try:
        action = policy.choose()
    except AgentFault:
        raise
    except Exception as exc:           # submitted-policy boundary
        raise AgentFault(f"policy failed: {exc}") from exc
    finally:
        policy.close()
    return score_from(env, action)
```
6. Pin the RPC contract in `scorer/data/env_config.json`:
```json
{
  "allowed_env_kwargs": ["seed"],
  "require_public_methods_allowlist": true,
  "max_instances": 64,
  "max_instances_per_connection": 16
}
```

If this grader uses `ContinuousTask.calibrated()` and generated locks (rather
than `PolicyEvaluationTask` paired controls), declare task-specific committed
no-op/fixed/open-loop/seeded-random output workspaces with
`WorkspaceDegenerateProbes` under `baselines/degenerate/`. There is no generic
hidden-env policy interface, so never ask the framework to infer these probes
from `hidden_env` or `ml_task_type`.

## What runs where (don't fight it)

- Boot: the rubric MCP server calls `supervise_if_enabled()`, reads
  `[environment].hidden_env` from `/task/task.toml`, and spawns
  `python -P -m env_server` from root-owned `/` with inherited Python import
  paths removed and a root-owned executable `PATH`.
- Rollout: agent -> `/data/env_client.py` -> `/tmp/env.sock` -> `make_env()` instance.
- Grade: `stop_env_server()` closes the socket FIRST (no grade-time env access),
  then the grader loads the held-out env in-process and runs the policy.

## Reward-hacking discipline (built in)

Env source is root-only; `_`/non-allow-listed methods are rejected over the
wire; explicit `allowed_env_kwargs` pins construction; global/per-connection
instance caps bound live state; socket-exposed methods cannot accept
unrestricted `**kwargs`; socket closes before grading; policy results return
over a dedicated pipe (never stdout); error replies carry no server traceback.
`raise AgentFault` for agent faults (kept 0.0); let author/infra faults
propagate (discarded). See `docs/REWARD_HACKING.md`.

## load_submitted_policy parity

Named factory (`load_policy` default), source mode (a grader-supplied LITERAL
wrapper for an agent data artifact -- never f-string agent data into it), and
multi-policy factories (return a dict / list -> dict / list of proxy handles).
Raises `AgentFault` for missing / non-regular / oversized policy or a load crash.

## Verify and submit

Reference runs start the env server inside the container before executing
`solution/solve.sh` (same as production agent sessions):

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
uv run lbx-rl-template validate --problem-dir problems/<task_id>   # blocking hidden_env stage
```

Read `reward-hacking-security` for submitted-policy loaders and fault
classification.
