# Ground Truth Requirements

Every task must include a reproducible ground-truth submission under
`solution/`. This is the quality gate that proves a human-authored oracle or
reference solution reaches the expected score under the exact same grader used
for agents.

## Oracle proof by task type

All tasks commit `.alignerr/build_proof.json`. The **oracle proof** is the
`ground_truth_result` block written by `--runtime ground-truth`. It is **not**
the same as **agent proof** (`harness_result`), which records an optional local
agent run and is populated by `--runtime claude-code`, `--runtime deepagents`, or
trusted CI.

| | Rubric tasks (`multi_deterministic_rubrics`) | ML / continuous tasks (`continuous_scoring_function`) |
| --- | --- | --- |
| **Typical `task_type`** | `mujoco`, `cfd`, `structures` | `ml` |
| **Expected oracle score** | **1.0** (within `[ground_truth].score_epsilon`) | **0.5 ± 0.05** (within `[ground_truth].continuous_score_epsilon`) |
| **What proof means** | Reference is a perfect oracle under deterministic rubrics | Reference is a **calibrated anchor**: competent, not optimal; agents should beat it toward 1.0 |
| **Reviewer video** | Required for `mujoco`; optional for other types if `[ground_truth].render_outputs` is declared | Usually none unless you explicitly declare render outputs |
| **Static artifacts in `solution/`** | Common (MJCF, control JSON, policies) | Common (CSV, weights); `solve.sh` may copy a checked-in artifact rather than retrain |
| **What ground-truth verifies** | Solve → grade → (render) → commit proof | Same pipeline: proves Dockerfile, grader, loaders, and **calibration** are correct |

For ML tasks, oracle proof does **not** mean “we retrained the model in CI.” It
means `solution/solve.sh` runs in the task environment, the grader accepts the
submission, and the reference lands at the calibration midpoint (~0.5). Training
code may live in `solution/` for provenance; the validation path should stay
deterministic (precomputed weights, cached CSV, etc.).

Learnability (baselines scoring clearly below the reference) is validated
separately by `lbx-rl-template validate`, not by the oracle score alone.

## Contract

- `solution/solve.sh` is required.
- Running `solution/solve.sh` must create all required `[[outputs]]` artifacts.
- `[difficulty].reward_type = "multi_deterministic_rubrics"` tasks must score
  `1.0` within `[ground_truth].score_epsilon`.
- `[difficulty].reward_type = "continuous_scoring_function"` tasks must score
  `0.5 ± [ground_truth].continuous_score_epsilon` (default `0.05`). This is the
  ML_Envs reference-solution convention: the checked-in reference is competent
  but leaves headroom for better agents.
- If the task requires a trained policy or model, include the trained artifact
  in `solution/` and make `solve.sh` copy or export it into `/tmp/output`.
- Training code may be included for provenance, but validation should use the
  already-trained artifact so the oracle run is deterministic and fast.
- Every task must declare `[difficulty].task_type`, `[difficulty].domain`, and
  `[difficulty].reward_type`. These are enum fields defined in
  `alignerr_plugin.task_metadata`.
- Reviewer video is required for `mujoco` tasks. Declare it in `[ground_truth]`
  and create it during ground-truth verification.
- Reviewer videos are MuJoCo-specific and must be 16:9 720p MP4-compatible
  assets. The harness enforces `1280x720` dimensions for `mujoco` tasks.

Example layout (MuJoCo rubric task):

```text
solution/
├── solve.sh              # required: emits outputs that score 1.0 under the rubric
├── render.sh             # MuJoCo: required reviewer video hook
├── policy.pt             # allowed: trained oracle policy/model
├── export_policy.py      # allowed: converts/checks policy artifacts
└── train.py              # optional: documents how the policy was produced
```

Example layout (ML continuous task):

```text
solution/
├── solve.sh              # required: writes /tmp/output artifacts; should score ~0.5
├── submission.csv        # allowed: precomputed reference submission
├── model.pt              # allowed: precomputed weights
└── train.py              # optional: provenance only; not run during CI proof
```

## `task.toml`

Declare the task metadata for every task:

```toml
[difficulty]
task_type = "mujoco" # ml | mujoco | cfd | structures
domain = "model_environment_construction"  # enum scoped by task_type
reward_type = "multi_deterministic_rubrics" # or continuous_scoring_function
```

Current enum values are defined in `alignerr_plugin.task_metadata`. The broad
sets are:

| Field | Values |
| --- | --- |
| `task_type` | `ml`, `mujoco`, `cfd`, `structures` |
| `reward_type` | `continuous_scoring_function`, `multi_deterministic_rubrics` |
| `domain` for `ml` | Production taxonomy subjects/subdomains such as `physical_sciences`, `physical_chemistry`, `robotics_embodied_ai`, `robot_dynamics_system_identification`, `multimodal_vision_perception`, `world_models`, `online_system_identification`, etc. |
| `domain` for `mujoco` | `policy_training_improvement`, `model_environment_construction`, `manipulation`, `locomotion`, `balance_recovery`, etc. |
| `domain` for `cfd` | `aerodynamics`, `airfoil_design`, `flow_control`, `vortex_suppression`, `rans_simulation`, etc. |
| `domain` for `structures` | `seismic_retrofit`, `structural_mechanics`, `topology_optimization`, `truss_design`, `modal_analysis`, etc. |

For `mujoco` tasks, declare the render command and video output:

```toml
[ground_truth]
render_command = "bash solution/render.sh"
render_outputs = [
  { path = "/tmp/output/rendering.mp4", required = true, description = "Reviewer video of the oracle rollout" },
]
```

The harness maps logical `/tmp/output/...` paths to its local workspace and
sets `LBT_OUTPUT_DIR` for render commands. Prefer writing render scripts like:

```bash
#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
uv run python -m lbx_rl_tasks_harness.render_mujoco \
  --model "${OUTPUT_DIR}/model.xml" \
  --policy "${OUTPUT_DIR}/policy.py" \
  --output "${OUTPUT_DIR}/rendering.mp4" \
  --config solution/render_config.py
```

Use `solution/render_config.py` only for task-specific rollout setup, such as
initial state, target trajectories, controller observations, or camera setup.

MuJoCo policy artifacts should define either `act(obs)` or `class Policy` with
an `act(obs)` method. The renderer passes a base observation containing `time`,
`step`, `qpos`, `qvel`, `sensordata`, and `ctrl`; task configs may add target
or goal fields before calling the policy.

## Local Verification

### While authoring: reference iteration

While iterating on `solution/solve.sh` (especially ML training), use the
reference loop. It is faster than the PR gate and keeps a persistent cache:

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness reference --problem-dir problems/<task_id> --skip-solve
```

See `harness/README.md` for `--no-grade`, `--clean-cache`, and optional
`[reference]` settings in `task.toml`.

`reference` does **not** write `ground_truth_result` or `harness_result`. It
writes `.alignerr/reference_run/latest/manifest.json` and (by default) caches
solve outputs under `.alignerr/reference_cache/output`. On **container** tasks,
the first `reference` run may still rebuild the task image and refresh
`image_digest` / `task_dir_sha256` in `build_proof.json` (without adding oracle
or agent scores). Use `--runtime ground-truth` when you need a committable
oracle proof.

### Before PR: ground-truth proof

Run the ground-truth gate before opening a PR:

```bash
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime ground-truth
```

This command:

1. runs `solution/solve.sh`;
2. grades the produced artifacts with `scorer/compute_score.py`;
3. fails unless the normalized score matches the `reward_type` target:
   `1.0` for deterministic rubrics, `0.5 ± 0.05` for continuous scoring;
4. refreshes build metadata in `.alignerr/build_proof.json` when the task image
   or source tree changed (host refresh or in-container build);
5. for tasks with declared `[ground_truth].render_outputs`, runs the render
   command and copies reviewer videos into `.alignerr/ground_truth/` (MuJoCo
   tasks require `1280x720` h264 video named `rendering.mp4`);
6. records `ground_truth_result` in `.alignerr/build_proof.json` (score,
   subscores, and review artifact metadata when renders are expected).

Commit the proof for every task:

```bash
git add problems/<task_id>/.alignerr/build_proof.json
```

For tasks with reviewer videos (typically `mujoco`), also commit:

```bash
git add problems/<task_id>/.alignerr/ground_truth/
```

ML tasks usually commit only `build_proof.json` unless you explicitly declare
render outputs.

The template validation workflow checks that each committed video exists and
matches the checksum recorded in `build_proof.json`.

### What goes in `build_proof.json`

| Key | Written by | Purpose |
| --- | --- | --- |
| `task_dir_sha256`, `image_digest`, `base_image_ref` | Docker build (ground-truth host refresh or container image build) | Proves the task source and image match |
| `ground_truth_result` | `--runtime ground-truth` | **Oracle proof** — reference score and optional review artifacts |
| `harness_result` | `--runtime claude-code` / `--runtime deepagents` (local) or trusted CI | **Agent proof** — agent score, rubric quality, optional Auto QA |

Ground-truth verification is separate from the agent difficulty signal. A good
task should have a verified oracle/reference, weak baselines near `0.0`, and
local/LLM agent attempts that remain below the target difficulty threshold.
