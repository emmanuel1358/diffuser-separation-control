---
name: alignerr-task-authoring
description: Helps author Alignerr RL tasks in this repository. Use when creating or editing task directories, task.toml, instruction.md, environment Dockerfiles, local harness runs, or submission docs.
---

# Alignerr Task Authoring

> **Authoring an ML task?** Prefer **ML_Envs mode** — the minimal contract (`metadata.json` + `prompt.md` + `test_file.py` + `data/{public,private}/`, no `task.toml` / per-task Dockerfile / `tests/test.sh`). See the `mlenvs-tasks` skill and `docs/MLENVS_TASKS.md`. The workflow below is the native contract for mujoco / cfd / structures.

## Workflow

1. Create tasks under `problems/<task_id>/`, not `examples/`.
2. Required files: `task.toml`, `metadata.json`, `instruction.md`, `environment/Dockerfile`, `scorer/compute_score.py`.
3. Final agent artifacts must be written under `/tmp/output` and declared in `task.toml` `[[outputs]]`.
4. Put public agent-readable files in `data/`; hidden grader fixtures go in `scorer/data/`.
5. Ensure `environment/Dockerfile` keeps `/workdir` and `/tmp/output` writable for uid `1000`:

```dockerfile
WORKDIR /workdir
RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output
```

6. Prefer GPU -- we want GPU-accelerated tasks. **`ml` tasks default to GPU** (the `ml` starter ships `[environment].required_resources = "12vcpu+100gib+h100/2"`); keep an H100 resource enum unless the task genuinely cannot use one (then pick a CPU enum and explain why). Do not put `cpus`, `memory_mb`, `gpus`, or `gpu_types` in `task.toml`. `base_flavor = "auto"` infers CPU/GPU/graphics/TPU from `required_resources`; use `gpu-openroad` only for EDA/OpenROAD tasks with a non-graphics H100 enum. Keep Dockerfiles generic with `ARG BASE_IMAGE`, `ARG BASE_TAG`, and `FROM ${BASE_IMAGE}:${BASE_TAG}` so the harness and mothership inject the right base image.

Available resource enums:

```text
CPU: 2vcpu+6gib, 4vcpu+16gib, 6vcpu+32gib, 8vcpu+64gib, 16vcpu+64gib, 16vcpu+128gib
CPU perf: 2vcpu+6gib+perf, 4vcpu+16gib+perf, 6vcpu+32gib+perf, 8vcpu+64gib+perf, 16vcpu+64gib+perf, 16vcpu+128gib+perf
TPU: 13vcpu+32gib+tpuv5e1x1, 16vcpu+64gib+tpuv5e2x2, 50vcpu+128gib+tpuv5e2x2
H100: 3vcpu+25gib+h100/8, 6vcpu+50gib+h100/4, 12vcpu+100gib+h100/2, 24vcpu+200gib+h100/1
H100 graphics: 3vcpu+25gib+h100/8+graphics, 6vcpu+50gib+h100/4+graphics, 12vcpu+100gib+h100/2+graphics, 24vcpu+200gib+h100/1+graphics
```

7. While developing `solution/solve.sh`, use the reference loop (no build-proof refresh):

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness reference --problem-dir problems/<task_id> --skip-solve
```

Run the deterministic ground-truth workflow before opening or updating a PR:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```

This requires `solution/solve.sh` to match the declared `reward_type` target. Rendered tasks must generate
reviewer artifacts under `.alignerr/ground_truth/`. When you open or update a
fork PR, trusted CI runs the full QA stack and posts `trusted-ci/grade`. Every task must declare enum-backed `[difficulty].task_type`, `[difficulty].domain`, and `[difficulty].reward_type`; values live in `alignerr_plugin.task_metadata`. `ml` tasks must also declare `[difficulty].license` (a permissive SPDX id or `not_applicable`).

8. Use focused modes only while iterating:

```bash
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime ground-truth
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime rubric-quality
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime claude-code
```

9. Commit the generated `problems/<task_id>/.alignerr/build_proof.json` and `problems/<task_id>/.alignerr/ground_truth/` with the task. If task files change after proof generation, rerun the ground-truth verifier before committing.
10. Open a GitHub PR in your assigned fork (not this template repo); trusted CI runs automatically and posts `trusted-ci/grade`.

## Local Harness

- `.env.local` is loaded automatically and must never be read, printed, or committed.
- `uv run lbx-rl-harness reference --problem-dir problems/<task_id>` runs solve + grade with a persistent cache (author iteration; does not update build proof).
- `uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>` creates or overwrites the required task build proof at `problems/<task_id>/.alignerr/build_proof.json`.
- If `task.toml` requests GPUs, the local harness builds `lbx-tasks-base-gpu:runtime-ml-core-py313-local`; local GPU execution still requires Docker/NVIDIA GPU support on the host.
- Use `--runtime ground-truth` for no-LLM oracle checks; it records `ground_truth_result` in build proof.
- Debug failures with `.harness-runs/<run>/trajectory.json` and `verifier/reward-details.json`.

## Numerical-Solver / non-MuJoCo tasks

For CFD or structures tasks, the
base image bundles an open-source solver stack (OpenFOAM, SU2, AeroSandbox,
scikit-fem/PyNite/CalculiX/OpenSeesPy). Use `task_type = "cfd"` with
`domain = "aerodynamics"` or `task_type = "structures"` with
`domain = "structural_mechanics"`, and follow the `numerical-solver-tasks` skill. A grader that needs a
base-only engine must set `[ground_truth].in_container = true`. Any task type can
attach a reviewer video via `[ground_truth].render_outputs` (`rendering.mp4`,
`1280x720` h264). See `docs/NUMERICAL_SOLVERS.md`,
`project_guidelines/cfd/cfd_environments.md`, and `examples/openfoam-hydrofoil-flap`.

## ML_Envs-style tasks and reward hacking

For dataset/CSV, model-module, executable, HDF5, or k-fold submissions scored by
a continuous calibrated function of held-out truth, follow the `mlenvs-tasks`
skill and `docs/MLENVS_TASKS.md`: read submissions via the sanctioned loaders
(`helpers.load_submission_or_fault`, `run_model_module`,
`run_submitted_executable`, `load_submission_h5_or_fault`, `score_kfold_cv`),
calibrate with `grading.calibration` (FLOOR/REF/PERFECT), and ship `solution/`
plus `baselines/`. Every scorer must respect `docs/REWARD_HACKING.md`: never
trust agent stdout, never exec/pickle agent artifacts in the grader, and use the
AgentFault keep-vs-discard discipline. `lbx-rl-template validate` enforces the
blocking `agent_fault` stage.

## References

- `README.md`
- `docs/AUTHORING.md`
- `docs/GRADING.md`
- `docs/MLENVS_TASKS.md`
- `docs/REWARD_HACKING.md`
- `docs/NUMERICAL_SOLVERS.md`
- `project_guidelines/mujoco_environments.md`
- `project_guidelines/cfd/cfd_environments.md`

## Submission Loop

Alignerrs submit GitHub PRs in their assigned fork:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
git add problems/<task_id> problems/<task_id>/.alignerr/build_proof.json problems/<task_id>/.alignerr/ground_truth/
git commit -m "Add <task_id> task"
git push
```

Trusted CI runs on every fork PR open or update. After edits, rerun the
ground-truth verifier and push to the same fork PR. Each push reruns trusted CI,
including the agent harness, rubric QA, Auto QA, and `trusted-ci/grade`. Taiga
results arrive later as advisory comments on the fork PR.
