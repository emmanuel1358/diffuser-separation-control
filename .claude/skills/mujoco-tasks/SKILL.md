---
name: mujoco-tasks
description: Authors MuJoCo robotics tasks with MJCF artifacts, deterministic rollout rubrics, policy submissions, oracle rendering, and reviewer videos. Use for task_type mujoco or examples/mujoco-pendulum patterns.
---

# MuJoCo Tasks

## Contract

```toml
[difficulty]
task_type = "mujoco"
domain = "model_environment_construction"
reward_type = "multi_deterministic_rubrics"
```

Choose the narrowest MuJoCo domain from `alignerr_plugin.task_metadata`.

Typical artifacts:

- MJCF/XML model;
- `policy.py` or `controller.py`;
- controller parameters; or
- a combined model/environment artifact.

Policies expose `act(obs)` or `class Policy` with `act(obs)`. Optional
`reset(seed=None, metadata=None)` is supported.

## Task design

- State the physical/control objective and constraints publicly.
- Pin timestep, integrator, initial state, perturbations, control limits, and
  RNG seeds.
- Grade multiple deterministic initial states or disturbances.
- Separate model validity, behavior, safety, robustness, and efficiency.
- Reject invalid physics, NaN, unstable simulation, and policy timeouts as
  candidate faults.
- Do not reward XML shape alone when rollout behavior is the capability.

## Grading

Use `RubricTask` and shared policy/model execution. Never import submitted
Python in the root grader. Use `context.policy()` or the shared dropped-worker
helpers.

The oracle scores `1.0`; required correctness/safety criteria gate optional
efficiency points.

Read `deterministic-grading`, `rubric-design`, and
`reward-hacking-security`.

## Reviewer video

MuJoCo tasks require a reviewer video:

```toml
[ground_truth]
render_command = "bash solution/render.sh"
render_outputs = [
  { path = "/tmp/output/rendering.mp4", required = true, description = "Oracle rollout" },
]
```

The video must be H.264, exactly `1280x720`, non-empty, and representative of
the oracle behavior. Prefer the shared renderer:

```bash
uv run python -m lbx_rl_tasks_harness.render_mujoco
```

Keep only task-specific rollout, observation, camera, and overlay hooks in
`solution/render_config.py`.

## Verification

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
uv run lbx-rl-template check --problem-dir problems/<task_id>
```

Inspect oracle score, perturbation coverage, video metadata, no-op policy score,
and world-integrity/adversarial checks.

## References

- `project_guidelines/mujoco/mujoco_environments.md`
- `examples/mujoco-pendulum/`
- `docs/GROUND_TRUTH.md`
- `docs/RUBRIC_EVALUATION.md`
