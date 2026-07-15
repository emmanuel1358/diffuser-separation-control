# Porting an existing ML_Envs task

ML_Envs mode reproduces the ML_Envs authoring contract almost 1:1, so porting a task from an ML_Envs repo is mostly a copy. See [`MLENVS_TASKS.md`](MLENVS_TASKS.md) for the full contract.

## File mapping

Drop the `_taiga` directory suffix: ML_Envs tasks live in `tasks/<name>_taiga/`, but here they are plain `problems/<name>/` directories. (A `_taiga` suffix is stripped from the task name if you leave it on, so a straight copy still deploys with a clean name — but prefer the plain directory.)

| ML_Envs (`tasks/<name>_taiga/`) | This template (`problems/<name>/`) |
| --- | --- |
| `metadata.json` | `metadata.json` (same 8-key surface, minus the timeout keys — timeouts are pinned here) |
| `prompt.md` | `prompt.md` (verbatim) |
| `test_file.py` | `test_file.py` (verbatim; the no-arg `compute_score()` runs unchanged) |
| `data/public/` | `data/public/` |
| `data/private/` | `data/private/` |
| `reference_solution/` | `reference_solution/` |
| `baselines/` | `baselines/` |
| `data-generation/` | `data-generation/` |

Nothing else is authored: `problems-metadata.json`, `pyproject.toml`, and the per-task Dockerfile were all generated/centralized in ML_Envs too, and are here.

## What changes

- **Timeouts.** Drop `tool_timeout_seconds` / `setup_timeout_seconds` / `grading_timeout_seconds` from `metadata.json` — they are pinned centrally (grading is pinned to the Taiga maximum).
- **Grader imports: flatten the `grading.helpers.<submodule>` paths.** This template keeps all the submission/executable/HDF5/model helpers in a single flat `grading.helpers` module (its own convention), with `env_loading`, `policy_eval`, and `kfold` as their own top-level modules. So adjust ML_Envs's nested imports:
  - `from grading.helpers.submission import load_submission_or_fault` -> `from grading.helpers import load_submission_or_fault`
  - `from grading.helpers.executable import run_submitted_executable` -> `from grading.helpers import run_submitted_executable`
  - `from grading.helpers.hdf5_loading import read_submitted_h5` -> `from grading.helpers import load_submission_h5_or_fault`
  - `from grading.helpers.model_loading import load_submitted_model` -> `from grading.helpers import load_submitted_model`
  - `from grading.helpers.env_loading import load_env_module` -> `from grading.env_loading import load_env_module`
  - `from grading.helpers.policy_eval import run_seeds, aggregate` -> `from grading.policy_eval import run_seeds, aggregate`
  - `from grading.helpers.kfold_eval import score_kfold_cv` -> `from grading.kfold import score_kfold_cv`
  - `from grading.faults import AgentFault` and `from env_server.policy_loader import load_submitted_policy` are unchanged.
  The helper implementations (including the working `run_submitted_executable` streaming/stdin/env modes and the disk-isolated `score_kfold_cv`) are ported from ML_Envs, so only the import path changes.
- **Grading runtime is this repo's, not ML_Envs's.** The bases bake this template's `rubric` + `grader` runtime (not `shared/grading`); the exporter points the task's `startup_command` at it. The `compute_score()` contract is identical, so graders don't notice.
- **Curve.** If the port used `ExponentialCurve`, switch to `PiecewiseLinearCurve` (the only sanctioned curve here). Anchors and weights are unchanged; the reference still lands at 0.5.
- **`env`/`hybrid` simulator deps -> `env_dependencies` (lbx-only hardening).** ML_Envs installs a hidden env's simulator (e.g. `myosuite`) into the shared site-packages, so the agent can `import` it and drive the raw env, bypassing the RPC constraints. When porting an `env`/`hybrid` task, move that package out of `dependencies` into lbx's `env_dependencies` (a new optional key): it installs root-only at `/mcp_server/env_deps` on the env server's `sys.path`, so the agent can't reach it. See [`HIDDEN_ENV.md`](HIDDEN_ENV.md).
- **`hf_resources` port verbatim.** The `metadata.json:hf_resources` list (bare `"org/name"` strings or `{repo_id, revision, repo_type, allow_patterns, ignore_patterns}` objects) carries over unchanged. This template ships the same faithful packer (`scripts/pack_hf_resource.py`): each repo is sha-content-addressed, fetched into the HF hub-cache layout, and mounted read-only at `/tmp/.cache/huggingface/hub/<folder>` for offline `from_pretrained`. Not available on the TPU base. See [`MLENVS_TASKS.md`](MLENVS_TASKS.md) section 8.

## Model loaders

Both model paradigms are available from `grading.helpers`: `load_submitted_model` (ML_Envs's sandboxed pickle/joblib-proxy loader for a `model.pkl` exposing `predict(...)`, ported verbatim — it reuses this template's `load_submitted_policy` sandbox) and `run_model_module` (this template's module-callable runner for a `model.py`). Import them as `from grading.helpers import load_submitted_model, run_model_module` (ML_Envs's `grading.helpers.model_loading` path flattens to `grading.helpers`).

## Local reference runs on Blackwell

The `mlenvs-gpu` / `mlenvs-cuda-graphics` bases are cu121 (Ampere/Ada/Hopper); their CUDA can't run on a Blackwell (sm_120) dev GPU. The local harness auto-detects this: on a GPU with compute capability >= 10.0 (via `nvidia-smi`) it swaps `mlenvs-gpu` -> `mlenvs-gpu-blackwell` and `mlenvs-cuda-graphics` -> `mlenvs-cuda-graphics-blackwell` (cu128) for local builds only — the Taiga export always uses the cu121 base for the H100 runners. Force it with `LBX_RL_TASKS_LOCAL_BLACKWELL=1` / `0` (e.g. on an nvidia-smi without the `compute_cap` field).
