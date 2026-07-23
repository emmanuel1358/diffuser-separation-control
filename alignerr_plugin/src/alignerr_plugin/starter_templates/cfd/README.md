# CFD Starter Template

Use this scaffold for `task_type = "cfd"` tasks (OpenFOAM-style flow problems
graded by running the solver against hidden conditions). It is intentionally
minimal; for a complete working reference, read
`examples/openfoam-hydrofoil-flap/`.

After creating a task, update:

- `instruction.md` with the flow problem and the exact `/tmp/output/...` artifact.
- `task.toml` with resources, timeouts, the `cfd` `domain`, and required outputs.
- `scorer/compute_score.py` with `TASK = RubricTask(...)` and pure hidden
  evaluation. Declare `JsonArtifact`; run solvers through
  `context.run_solver(...)`. Do not hand-write loaders or exception handling.
- `scorer/evaluation.plan.json`: sealed plan refreshed from `TASK` by harness reference/ground-truth (commit; never hand-edit).
- `data/` with public assets (case templates, schemas, probes).
- `scorer/data/` with private hidden conditions / target specifications.
- `solution/solve.sh` with the reference (oracle) design.
- `solution/render.sh` with a reviewer flow-field video generator.

Notes:

- OpenFOAM ships in the base image conda env (reached via
  `/etc/solver-envs.d/openfoam.sh`); no task-level solver install is needed.
- `reward_type` defaults to `multi_deterministic_rubrics`. The oracle
  (`solution/solve.sh`) must score ~1.0 and a do-nothing baseline ~0.

Before submitting, run:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```
