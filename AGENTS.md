# Agent Instructions

These instructions are for OpenAI Codex and other repository-aware agents. They
mirror the Cursor and Claude ground-truth/oracle skill content.

## Ground Truth Oracle

Use these rules when creating or reviewing `solution/` artifacts,
`verify-ground-truth` runs, reviewer videos, `build_proof.json`
`ground_truth_result`, or `.alignerr/ground_truth` assets.

## Contract

- Every task needs `solution/solve.sh`; it is the oracle submission.
- Every task must declare `[difficulty].task_type`, `[difficulty].domain`, and `[difficulty].reward_type`; enum values live in `alignerr_plugin.task_metadata`.
- `task_type` is one of `ml`, `mujoco`, `cfd`, `structures`. `domain` is an enum scoped by `task_type` and is used by the dashboard/Supabase index for diversity tracking.
- Declare `[environment].required_resources` as one of the Taiga enum values in `alignerr_plugin.taiga_resources`. **`ml` tasks deploy on TPU by default** (`13vcpu+32gib+tpuv5e1x1`; the `ml` starter ships this) because TPU is faster on Taiga, while the reference still runs on GPU; an `ml` task should request an `h100/*` enum only when it genuinely cannot run on a TPU. Other task types that can use acceleration should request an H100 enum; pick a CPU enum only when the task genuinely cannot use one. `base_flavor = "auto"` infers CPU/GPU/graphics/TPU from the enum; use `gpu-openroad` only for EDA/OpenROAD tasks with a non-graphics H100 enum. Keep Dockerfiles generic (`ARG BASE_IMAGE`/`ARG BASE_TAG`) so the harness/mothership inject the right base.
- Available resource enums: CPU `2vcpu+6gib`, `4vcpu+16gib`, `6vcpu+32gib`, `8vcpu+64gib`, `16vcpu+64gib`, `16vcpu+128gib`; CPU perf `2vcpu+6gib+perf`, `4vcpu+16gib+perf`, `6vcpu+32gib+perf`, `8vcpu+64gib+perf`, `16vcpu+64gib+perf`, `16vcpu+128gib+perf`; TPU `13vcpu+32gib+tpuv5e1x1`, `16vcpu+64gib+tpuv5e2x2`, `50vcpu+128gib+tpuv5e2x2`; H100 `3vcpu+25gib+h100/8`, `6vcpu+50gib+h100/4`, `12vcpu+100gib+h100/2`, `24vcpu+200gib+h100/1`; H100 graphics `3vcpu+25gib+h100/8+graphics`, `6vcpu+50gib+h100/4+graphics`, `12vcpu+100gib+h100/2+graphics`, `24vcpu+200gib+h100/1+graphics`.
- `ml` tasks must declare `[difficulty].license` (a permissive SPDX id from `task_metadata.LICENSES`: MIT, Apache-2.0, BSD-2/3-Clause, ISC, Unlicense, CC0-1.0, CC-BY-4.0, PDDL-1.0, UPL-1.0), or `not_applicable` for edge cases with no applicable external-dataset license. Trace the dataset license upstream; copyleft / non-commercial / research-only data is rejected. Other task types use solver-generated data and may omit it.
- `reward_type = "multi_deterministic_rubrics"` oracles must create all required `[[outputs]]` artifacts and score `1.0` under the same `scorer/compute_score.py` used for agents.
- `reward_type = "continuous_scoring_function"` references should score `0.5 ± 0.05`; this preserves ML_Envs continuous-score/reference behavior.
- Any task type (not just `mujoco`) may attach a reviewer video by declaring `[ground_truth].render_outputs`; doing so generalizes the same video contract (video named `rendering.mp4`, exactly `1280x720` h264, committed to `.alignerr/ground_truth/`).
- Numerical-solver tasks (`cfd` / `structures`) use the base-image solver stack; a grader/renderer that needs a base-only engine (OpenFOAM/SU2/OpenSees) must set `[ground_truth].in_container = true` so solve+grade+render run inside the task image. `instruction.md` stays solver-agnostic, and the core solution implementation (`solution/solve.sh` or solution files it calls) must include solver-specific runnable material for the submitted ground-truth oracle. Do not require agent transcript checks or solver-evidence artifacts. See `docs/NUMERICAL_SOLVERS.md` and `project_guidelines/cfd/cfd_environments.md`.
- Simulation / interaction tasks where the agent must probe a black-box environment (hidden dynamics it cannot read) set `[environment].hidden_env = "env"|"hybrid"` (orthogonal to `task_type`). The rubric server spawns an RPC env server the agent reaches over `/tmp/env.sock`; ship the hidden env at `scorer/data/env.py` (`make_env`, baked root-only) and a public `data/env_client.py`, and grade with `grading.load_env_module` + `grading.load_submitted_policy`. Do NOT use it for static-data tasks. See `docs/HIDDEN_ENV.md` and `examples/hidden-env-bandit/`.
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

While authoring, iterate on the reference without refreshing build proof:

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness reference --problem-dir problems/<task_id> --skip-solve
```

Before opening or updating a task PR, run the deterministic ground-truth
workflow:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```

The ground-truth runtime must:

1. run `solution/solve.sh`;
2. grade with `scorer/compute_score.py`;
3. fail unless score matches the `reward_type` target (`1.0` or `0.5 ± 0.05`);
4. for `mujoco` tasks, run `[ground_truth].render_command`;
5. for `mujoco` tasks, fail unless each video exists, is non-empty, and is `1280x720`;
6. for `mujoco` tasks, copy videos to `problems/<task_id>/.alignerr/ground_truth/`;
7. record `ground_truth_result.review_artifacts[]` in `.alignerr/build_proof.json` with `path`, `logical_path`, `sha256`, `bytes`, `width`, and `height`.

The AI agent harness, rubric-quality review, Auto QA, and ground-truth artifact
checks run in trusted CI when you open or update a fork PR. Passing QA posts
`trusted-ci/grade` and submits a fire-and-forget Taiga eval for advisory
feedback on the same PR.

Focused modes are for iteration only:

```bash
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime ground-truth
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime rubric-quality
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime claude-code
```

Commit both:

```bash
git add problems/<task_id>/.alignerr/build_proof.json
git add problems/<task_id>/.alignerr/ground_truth/
```

For `ml` tasks, `.alignerr/ground_truth/` may be absent; still commit the
updated `build_proof.json`.

## Do Not

- Do not treat agent difficulty scores as oracle/reference scores. Agents should generally remain below the task difficulty threshold, while ground-truth submissions must match the declared `reward_type` target.
- Do not store generated review videos in `solution/` or `.harness-runs/` as the PR artifact.
- Do not require rendering videos for `ml` tasks unless the task explicitly needs one.
- Do not accept videos without checksum/size/dimension metadata in `build_proof.json`.
- Do not lower the oracle score to make a task harder; task difficulty comes from local/LLM/Boreal agent attempts remaining below the desired threshold.

## References

- `docs/GROUND_TRUTH.md`
- `docs/NUMERICAL_SOLVERS.md`
- `project_guidelines/cfd/cfd_environments.md`
- `examples/mujoco-pendulum/solution/render_config.py`
- `examples/openfoam-hydrofoil-flap` (CFD/OpenFOAM with solver-backed oracle/grader scoring)

- `lbx_rl_tasks_harness.render_mujoco`
- MuJoCo policies should expose `act(obs)` or `class Policy` with `act(obs)`.
