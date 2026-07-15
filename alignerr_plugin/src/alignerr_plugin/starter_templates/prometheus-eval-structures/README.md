# Prometheus Eval Structures Starter Template

Use this scaffold for `task_type = "structures"` eval tasks that should run the
normal numerical-solver trusted-CI gates and then submit to **Prometheus** and
an independent Taiga mirror. The layout matches the `prometheus-structures`
starter; the difference is `[delivery].eval = true` in `task.toml`.

Use this for eval Prometheus structures projects. Non-eval Prometheus structures
projects use the sibling `prometheus-structures` starter with
`[delivery].eval = false`; both starters follow the same CI and dual-delivery
route.

After creating a task, update:

- `instruction.md` with the structural problem and the exact `/tmp/output/...` artifact.
- `task.toml` with resources, timeouts, the `structures` `domain`, and required outputs.
- `scorer/compute_score.py` with task-specific hidden evaluation. Read the
  agent submission through a guarded loader (`except OSError: raise AgentFault(...)`);
  run OpenSeesPy against held-out load cases and score deterministically.
- `data/` with public assets (model summary, schema, public probe).
- `scorer/data/` with private hidden load cases / target specifications.
- `solution/solve.sh` with the reference (oracle) design.
- `solution/render.sh` with a reviewer FE-response video generator.

Notes:

- Trusted CI still treats the task as structures: solver-agnostic instruction
  checks, structures grader QA, solver-backed oracle validation, render artifact
  validation, local agent scoring, and score-bounds gating all run before
  delivery.
- Passing tasks are exported to Harbor for the Prometheus Agent Service runner
  and submitted independently to Taiga. Taiga carries OpenSees availability as
  a native hint rather than changing the authored instruction.
- The eval row is accepted only after the full Prometheus workflow passes:
  `submit-prometheus` must pass with average target score `<= 0.5`, target score
  standard deviation `>= 0.1`, required attempts present, and a passing
  trainability auditor score.
- The `eval` flag is internal metadata only. It identifies eval versus non-eval
  submissions; it does not change the route.
- The environment installs `openseespy` (pip). Add any extra deps your scorer
  needs in `environment/Dockerfile`.
- `reward_type` defaults to `multi_deterministic_rubrics`. The oracle
  (`solution/solve.sh`) must score ~1.0 and a do-nothing baseline ~0.

Before submitting, run:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```
