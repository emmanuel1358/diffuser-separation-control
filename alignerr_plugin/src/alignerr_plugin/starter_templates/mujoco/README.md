# MuJoCo Starter Template

Use this scaffold for robotics simulation tasks. It is intentionally
minimal; for a complete working reference, read
`examples/mujoco-pendulum/`.

After creating a task, update:

- `instruction.md` with the model, policy, or control objective.
- `task.toml` with resources, timeouts, and required `/tmp/output/...`
  artifacts.
- `scorer/compute_score.py` with task-specific hidden evaluation.
- `data/` with public examples, starter assets, or environment clients.
- `scorer/data/` with private test fixtures, seeds, XML assets, or target
  specifications.
- `solution/solve.sh` with a reference solution when possible.

Common additions:

- MuJoCo XML assets under `scorer/data/assets/`.
- Hidden seeds under `scorer/data/seeds.json`.
- Public environment clients or examples under `data/`.
- GPU requirements in `task.toml` when training or evaluation needs CUDA.

Before submitting, run:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```
