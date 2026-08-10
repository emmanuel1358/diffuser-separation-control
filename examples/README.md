# Example Tasks

`examples/` contains template-owned reference tasks. Use these as working
examples when authoring your own task, but put your submitted work under
`problems/<task_id>/`.

## Canonical Examples

The current end-to-end examples are:

- [`mujoco-pendulum`](mujoco-pendulum/): rubric-style deterministic
  scoring with declarative `RubricTask`.
- [`mle-tabular-classification`](mle-tabular-classification/):
  the canonical continuous-scored `ml` task (static held-out data;
  no rubric).
- [`openfoam-hydrofoil-flap`](openfoam-hydrofoil-flap/):
  CFD/OpenFOAM scoring with solver-backed oracle/grader logic.
- [`opensees-base-isolation`](opensees-base-isolation/):
  structures/OpenSeesPy scoring with solver-backed oracle/grader logic.
- [`wal-recovery-ordering`](wal-recovery-ordering/):
  the canonical repository-debugging example, migrated from FrontierBench with
  hidden concurrency checks, performance gates, determinism, and attack tests.
- [`xfoil-rust-port`](xfoil-rust-port/):
  a long-horizon legacy-modernization task with public differential tooling,
  hidden behavioral suites, and subprocess-delegation defenses.
- [`frontier-service-cutover`](frontier-service-cutover/):
  nested services, init jobs, health gates, ordered state capture, and a
  separate verifier.
- [`frontier-mcp-workspace`](frontier-mcp-workspace/):
  shared service state plus an audited task-local MCP/SSE tool bridge.

### `mujoco-pendulum`

This example demonstrates:

- A complete task directory with `task.toml`, `instruction.md`,
  `environment/Dockerfile`, `scorer/compute_score.py`, `data/`,
  `solution/`, and `baselines/`.
- A declarative `RubricTask` (`TASK`). Harness reference/ground-truth write
  sealed `scorer/evaluation.plan.json` (commit it; never hand-edit).
- Ten equally weighted criteria, including MuJoCo compile checks,
  structural checks, sensor checks, and rollout checks.
- The required `/tmp/output` convention for agent-created artifacts.

### `mle-tabular-classification`

The canonical continuous-scored **`ml`** example (`docs/ML_TASKS.md`). It
demonstrates:

- Public parquet in `data/` and held-out truth in
  `scorer/data/`.
- The dependency-channel split: agent-visible `environment/requirements.txt`
  versus the root-only channels the grader and hidden env server use.
- A `compute_score` that loads the submission via `grading.helpers` and
  calibrates with `FLOOR/REF/PERFECT` + the sanctioned `PiecewiseLinearCurve`
  (reference scores ~0.5).
- A `{score, subscores}` return with no `RubricTask`.

## How To Use This Example

Try the reference harness against any example (Docker required for ML, CFD,
structures, and hidden-env tasks):

```bash
uv run lbx-rl-harness reference --problem-dir examples/mujoco-pendulum
uv run lbx-rl-harness reference --problem-dir examples/mle-tabular-classification
uv run lbx-rl-harness reference --problem-dir examples/openfoam-hydrofoil-flap
uv run lbx-rl-harness reference --problem-dir examples/opensees-base-isolation
uv run lbx-rl-harness reference --problem-dir examples/wal-recovery-ordering
```

Expected scores on the checked-in references:

| Example | Execution | Expected score |
| --- | --- | --- |
| `mujoco-pendulum` | host | `1.0` |
| `mle-tabular-classification` | container | `~0.5` (continuous) |
| `hidden-env-bandit` | container | `1.0` |
| `openfoam-hydrofoil-flap` | container | `1.0` |
| `opensees-base-isolation` | container | `1.0` |
| `wal-recovery-ordering` | host/container | `1.0` |
| `frontier-service-cutover` | capsule | `1.0` |
| `frontier-mcp-workspace` | capsule | `1.0` |

Read the closest example before creating your own task:

```bash
ls examples/mujoco-pendulum
ls examples/mle-tabular-classification
ls examples/openfoam-hydrofoil-flap
ls examples/opensees-base-isolation
ls examples/wal-recovery-ordering
ls examples/frontier-service-cutover
ls examples/frontier-mcp-workspace
```

Then scaffold a new task in `problems/` by copying the starter that matches your
`task_type` (`ml`, `mujoco`, `cfd`, `structures`, or
`software_engineering`):

```bash
mkdir -p problems
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/mujoco problems/my-task
```

Update `problems/my-task/metadata.json` and `task.toml` after copying the
starter. Copy patterns from the examples, not the example directories themselves.
The `examples/` tree is maintained by the template repo to test the full
pipeline and to document known-good task shapes.

## PR Behavior

The `examples/` tree is maintained by the template repo to test the full
pipeline and to document known-good task shapes. Example changes are typically
made by maintainers via direct commits or internal PRs here, not via labeler
fork PRs.

Labeler task submissions use the fork + trusted CI flow described in
[`../README.md`](../README.md): open a PR in your assigned fork, and trusted CI
posts `trusted-ci/grade` automatically.

## Where To Read More

- [`../README.md`](../README.md): beginner workflow and command reference.
- [`../docs/AUTHORING.md`](../docs/AUTHORING.md): task authoring guide.
- [`../docs/GRADING.md`](../docs/GRADING.md): grader package and
  `compute_score.py` contract.
