---
name: ground-truth-oracle
description: Guides Alignerr RL task ground-truth/oracle submissions. Use when creating or reviewing solution/ artifacts, verify-ground-truth runs, reviewer videos, build_proof.json ground_truth_result, or .alignerr/ground_truth assets.
---

# Ground Truth Oracle

## Contract

- Every task needs `solution/solve.sh`; it is the oracle submission.
- Every task must declare enum-backed `[difficulty].task_type`, `[difficulty].domain`, and `[difficulty].reward_type`; values live in `alignerr_plugin.task_metadata`.
- `reward_type = "multi_deterministic_rubrics"` oracles must score `1.0` under the same `scorer/compute_score.py` used for agents.
- `reward_type = "continuous_scoring_function"` references must score `0.5 ± 0.05`.
- If a MuJoCo task needs a trained policy/model, commit the trained artifact or deterministic exporter under `solution/`; full training code is optional provenance, not the validation path.
- Reviewer video is required for `mujoco` tasks. Declare it in `[ground_truth]` and generate it with `solution/render.sh`.
- Reviewer videos are MuJoCo-specific and must be 16:9 720p: exactly `1280x720`.

## Required `task.toml`

```toml
[difficulty]
task_type = "mujoco"
domain = "model_environment_construction"
reward_type = "multi_deterministic_rubrics"
```

For `mujoco` tasks:

```toml
[ground_truth]
render_command = "bash solution/render.sh"
render_outputs = [
  { path = "/tmp/output/rendering.mp4", required = true, description = "Reviewer video of the oracle rollout" },
]
```

## Verification Loop

### While authoring

Iterate on the reference without refreshing build proof:

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness reference --problem-dir problems/<task_id> --skip-solve
```

### Before PR

Run:

```bash
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime ground-truth
```

This command must:

1. run `solution/solve.sh`;
2. grade with `scorer/compute_score.py`;
3. fail unless score matches the declared `reward_type` target;
4. for `mujoco` tasks, run `[ground_truth].render_command`;
5. for `mujoco` tasks, fail unless each video exists, is non-empty, and is `1280x720`;
6. for `mujoco` tasks, copy videos to `problems/<task_id>/.alignerr/ground_truth/`;
7. record `ground_truth_result.review_artifacts[]` in `.alignerr/build_proof.json` with `path`, `logical_path`, `sha256`, `bytes`, `width`, and `height`.

Commit both:

```bash
git add problems/<task_id>/.alignerr/build_proof.json
git add problems/<task_id>/.alignerr/ground_truth/
```

## Do Not

- Do not treat agent difficulty scores as oracle/reference scores. Agents should generally remain below the task difficulty threshold, while ground-truth submissions must match the declared `reward_type` target.
- Do not store generated review videos in `solution/` or `.harness-runs/` as the PR artifact.
- Do not require rendering videos for `ml` tasks unless the task explicitly needs one.
- Do not accept videos without checksum/size/dimension metadata in `build_proof.json`.

## References

- `docs/GROUND_TRUTH.md`
- `examples/mujoco-pendulum/solution/render_config.py`

- `lbx_rl_tasks_harness.render_mujoco`
- MuJoCo policies should expose `act(obs)` or `class Policy` with `act(obs)`.
