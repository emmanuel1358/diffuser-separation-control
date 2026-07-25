# Structures Starter Template

Use this scaffold for `task_type = "structures"` tasks (OpenSeesPy-style
finite-element problems graded by running an analysis against hidden load
cases). It is intentionally minimal; for a complete working reference, read
`examples/opensees-base-isolation/`.

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

- The environment installs `openseespy` (pip). Add any extra deps your scorer
  needs in `environment/Dockerfile`.
- `reward_type` defaults to `multi_deterministic_rubrics`. The oracle
  (`solution/solve.sh`) must score ~1.0 and a do-nothing baseline ~0.

Before submitting, run:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```
