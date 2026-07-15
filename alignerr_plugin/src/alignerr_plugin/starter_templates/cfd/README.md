# CFD Starter Template

Use this scaffold for `task_type = "cfd"` tasks (OpenFOAM-style flow problems
graded by running the solver against hidden conditions). It is intentionally
minimal; for a complete working reference, read
`examples/openfoam-hydrofoil-flap/`.

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

- OpenFOAM ships in the base image conda env (reached via
  `/etc/solver-envs.d/openfoam.sh`); no task-level solver install is needed.
- `reward_type` defaults to `multi_deterministic_rubrics`. The oracle
  (`solution/solve.sh`) must score ~1.0 and a do-nothing baseline ~0.

Before submitting, run:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```
