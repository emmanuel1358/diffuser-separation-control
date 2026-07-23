# Prometheus Structures Starter Template

Use this scaffold for `task_type = "structures"` tasks that should run the
normal numerical-solver trusted-CI gates and then submit to **Prometheus** and
an independent Taiga mirror. The layout matches the `structures` starter; the
difference is
`[delivery].platform = "prometheus"` and `[delivery].eval = false` in
`task.toml`.

Use this for non-eval Prometheus structures projects. Eval Prometheus structures
projects use the sibling `prometheus-eval-structures` starter; both starters
follow the same CI and dual-delivery route.

After creating a task, update:

- `instruction.md` with the structural problem and the exact `/tmp/output/...` artifact.
- `task.toml` with resources, timeouts, the `structures` `domain`, and required outputs.
- `scorer/compute_score.py` with `TASK = RubricTask(...)` and pure hidden
  evaluation. Declare `JsonArtifact`/`TrustedJson`; run trusted structural
  analysis through `RubricContext`. Do not hand-write loaders/error handling.
- `scorer/evaluation.plan.json`: sealed plan refreshed from `TASK` by harness reference/ground-truth (commit; never hand-edit).
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
- Harbor/Prometheus runs the agent as uid 1000 (`[agent].user = "agent"`) and the
  verifier as root (`[verifier].user = "root"`) so private scorer data under
  `/mcp_server/data` stays out of the agent sandbox.
- Review readiness: green `trusted-ci/grade` (Prometheus average `<= 0.5`
  included), Boreal required QA complete on the LBx Validation comment or
  dashboard, and no unresolved critical findings. Warnings/info are fine;
  Boreal average is not a blocker. Self-iterate on clear criticals; submit for
  coaching when stuck or acceptance when gates pass, and name which Boreal
  surface is latest.
- Submit a new, original problem. Problems already submitted to the original CFD
  or structures projects, or previously submitted to Boreal, must not be
  resubmitted to Prometheus. Those submissions will be rejected, count as
  cheating, and may warrant removal from the project.
- The `eval` flag is internal metadata only. It identifies non-eval versus eval
  submissions; it does not change the route.
- The environment installs `openseespy` (pip). Add any extra deps your scorer
  needs in `environment/Dockerfile`.
- `reward_type` defaults to `multi_deterministic_rubrics`. The oracle
  (`solution/solve.sh`) must score ~1.0 and a do-nothing baseline ~0.

Before submitting, run:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```
