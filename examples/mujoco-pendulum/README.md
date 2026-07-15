# MuJoCo Pendulum Example

This is the canonical reference task for the template repo. It follows
[`project_guidelines/mujoco_environments.md`](../../project_guidelines/mujoco_environments.md)
and provides a real deterministic MuJoCo grader.

Use this task as a pattern when creating your own task under `problems/`.
Do not submit work by editing this example directory.

## What The Agent Must Do

The prompt in `instruction.md` asks the agent to create a damped
single-pendulum MJCF and write it to:

```text
/tmp/output/model.xml
```

Because this is a static MJCF artifact, the grader does not execute
agent-authored Python during scoring and does not need `PolicyWorker`. Use
`PolicyWorker` for MuJoCo tasks that ask for executable `/tmp/output/policy.py`
controllers.

That same output path is declared in `task.toml`:

```toml
[[outputs]]
path = "/tmp/output/model.xml"
required = true
description = "MJCF XML model for the damped pendulum"
```

## What The Grader Checks

`scorer/compute_score.py` uses `RubricBuilder` and returns
`rb.grade().to_dict()`. The grader compiles the submitted MJCF once and
then scores ten criteria:

- The MJCF compiles.
- There is exactly one hinge joint.
- There is exactly one degree of freedom.
- There is exactly one moving body.
- The moving mass is near the target.
- The center of mass is near the target.
- A `jointpos` sensor exists.
- A `jointvel` sensor exists.
- A rollout from `qpos=pi/2` stays bounded.
- The rollout has no NaN values.

The criteria are equally weighted, so Boreal and Harbor can show both the
headline score and per-criterion subscores.

## Files To Notice

- `task.toml`: task metadata, resources, timeout, verifier env, required
  outputs, and ground-truth video declaration.
- `instruction.md`: the agent-facing prompt.
- `environment/Dockerfile`: installs the shared `grader/` package and
  copies this task's public data, private scorer data, and scorer code.
- `scorer/compute_score.py`: the grader implementation.
- `solution/solve.sh`: the ground-truth solution that scores `1.0`.
- `solution/render.sh`: required reviewer video generation for that solution.
- `baselines/naive.sh`: a weak baseline script for comparison.

## Creating Your Own MuJoCo Task

From the repo root:

```bash
mkdir -p problems
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/mujoco problems/my-task
```

Then update `problems/my-task/` using this example as your guide. Before
opening a PR, run:

```bash
uv run lbx-rl-harness run --problem-dir problems/my-task
uv run lbx-rl-harness run --problem-dir problems/my-task
```
