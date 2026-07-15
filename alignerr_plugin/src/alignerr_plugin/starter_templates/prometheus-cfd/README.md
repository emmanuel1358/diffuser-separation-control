# Prometheus CFD Starter Template

Use this scaffold for `task_type = "cfd"` tasks that should run the normal
numerical-solver trusted-CI gates and then submit to **Prometheus** and an
independent Taiga mirror. The layout matches the `cfd` starter; the difference is
`[delivery].platform = "prometheus"` and `[delivery].eval = false` in
`task.toml`.

Use this for non-eval Prometheus CFD projects. Eval Prometheus CFD projects use
the sibling `prometheus-eval-cfd` starter; both starters follow the same CI and
Prometheus submission route.

After creating a task, update:

- `instruction.md` with the flow problem and the exact `/tmp/output/...` artifact.
- `task.toml` with resources, timeouts, the `cfd` `domain`, and required outputs.
- `scorer/compute_score.py` with task-specific hidden evaluation. Read the
  agent submission through a guarded loader (`except OSError: raise AgentFault(...)`);
  run the solver against held-out conditions and score deterministically.
- `data/` with public assets (case templates, schemas, probes).
- `scorer/data/` with private hidden conditions / target specifications.
- `solution/solve.sh` with the reference (oracle) design.
- `solution/render.sh` with a reviewer flow-field video generator.

Notes:

- Trusted CI still treats the task as CFD: solver-agnostic instruction checks,
  CFD grader QA, solver-backed oracle validation, render artifact validation,
  local agent scoring, and score-bounds gating all run before delivery.
- Passing tasks are exported to Harbor for the Prometheus Agent Service runner
  and submitted independently to Taiga. Taiga carries OpenFOAM availability as
  a native hint rather than changing the authored instruction.
- Harbor/Prometheus runs the agent as uid 1000 (`[agent].user = "agent"`) and the
  verifier as root (`[verifier].user = "root"`) so private scorer data under
  `/mcp_server/data` stays out of the agent sandbox.
- Do not submit the task for review until the full Prometheus workflow passes:
  `submit-prometheus` must pass with average target score `<= 0.5`, target score
  standard deviation `>= 0.1`, required attempts present, and a passing
  trainability auditor score. Only then should a non-eval row move into review.
  Submitting a non-passing row for review violates fair practices and may remove
  the tasker from the project.
- Submit a new, original problem. Problems already submitted to the original CFD
  or structures projects, or previously submitted to Boreal, must not be
  resubmitted to Prometheus. Those submissions will be rejected, count as
  cheating, and may warrant removal from the project.
- The `eval` flag is internal metadata only. It identifies non-eval versus eval
  submissions; it does not change the route.
- OpenFOAM ships in the base image conda env (reached via
  `/etc/solver-envs.d/openfoam.sh`); no task-level solver install is needed.
- `reward_type` defaults to `multi_deterministic_rubrics`. The oracle
  (`solution/solve.sh`) must score ~1.0 and a do-nothing baseline ~0.

Before submitting, run:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```
