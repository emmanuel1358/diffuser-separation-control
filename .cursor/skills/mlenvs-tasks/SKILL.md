---
name: mlenvs-tasks 
description: Author ML_Envs-mode tasks (the minimal metadata.json + prompt.md + test_file.py contract) in this template. Use when creating or migrating an ML_Envs-style continuous-scored task with held-out truth — the no-arg compute_score() grader, the submission loaders, FLOOR/REF/PERFECT + PiecewiseLinearCurve calibration, dataset licensing, ml_task_type paradigms (dataset/env/hybrid/sim_policy), or the mlenvs-* base flavors.
---

# ML_Envs-mode Tasks

The **minimal authoring contract** for ML tasks. The author edits only `metadata.json` + `prompt.md` + `test_file.py` + `data/` + `reference_solution/`
+ `baselines/`. **No `task.toml`, no per-task Dockerfile, no `tests/test.sh`** — all pinned centrally. Full guide: `docs/MLENVS_TASKS.md`. Canonical example: `examples/mle-tabular-classification/`. This is additive: mujoco/cfd/structures and native `ml` tasks keep the `task.toml` contract (`docs/AUTHORING.md`).

## Layout

```text
problems/<task_id>/
├── metadata.json          # minimal config (below)
├── prompt.md              # agent prompt (no anchors, no internals)
├── test_file.py           # no-arg compute_score() reading /tmp/output + /mcp_server/data
├── data/{public,private}/ # public -> /data/ (read-only); private -> /mcp_server/data/ (root)
├── reference_solution/    # train.py + solution.py + model + model.manifest.json (scores 0.5)
├── baselines/<name>/      # naive: same committed-model contract; scores below reference
└── data-generation/       # provenance
```

Detected as ML_Envs mode when there is no `task.toml` and a `test_file.py` (or `metadata.json` has `ml_task_type`). Synthesis + pinned constants: `alignerr_plugin.mlenvs`.

## `metadata.json`

Required: `ml_task_type` (`dataset`|`env`|`hybrid`|`sim_policy`), `required_resources` (a Taiga enum), `domain` (ml-scoped), `license` (permissive SPDX or `self_generated`), `license_source`. Optional: `docker-base` (`default`|`cuda-graphics`|`tpu`), `dependencies` (pip, agent-visible), `apt_extras` (apt), `env_dependencies` (pip for the hidden env server ONLY — env/hybrid; installed root-only so the agent can't import them; see below), `description` (one-line human blurb), `hf_resources` (read-only HuggingFace mounts — see below). Everything else (`task_type=ml`, `reward_type=continuous_scoring_function`, `allow_internet=false`, timeouts, runner knobs, `/tmp/output`) is pinned — never author it.

```json
{
  "ml_task_type": "dataset",
  "required_resources": "12vcpu+100gib+h100/2",
  "domain": "scientific_discovery_computational_science",
  "license": "CC0-1.0",
  "license_source": "https://creativecommons.org/publicdomain/zero/1.0/"
}
```

## `test_file.py` (no-arg grader)

```python
from grading.evaluation import ContinuousTask, PrivateTableChallenge, PythonPredictor

TASK = ContinuousTask.model(
    artifact=PythonPredictor("predictor.py"),
    challenge=PrivateTableChallenge(
        "challenge.parquet", feature_columns=["x1", "x2"], sample_size=256
    ),
    targets=[...],
)

def compute_score():
    return TASK.compute_score()
```

- Follow `docs/CONTINUOUS_EVALUATION.md`; new tasks use queryable Tier-A artifacts.
- **No arguments** remains valid for metadata-mode graders.
- `raise AgentFault` for agent faults (kept 0.0); let author/infra faults propagate (discarded). No broad `except: return 0.0`; no exec/pickle of agent artifacts in the grader.
- Read agent output only via the sanctioned loaders (all flat): `grading.helpers` (`load_submission_or_fault` CSV, `run_submitted_executable`, `load_submission_h5_or_fault`, `load_submitted_model`), `grading.policy_eval` (`run_seeds`/`aggregate` for sim_policy), `grading.env_loading` (`load_env_module`), `grading.kfold` (`score_kfold_cv`); `env_server.policy_loader.load_submitted_policy` for env/hybrid.
- Use registered targets, reviewed `FloorAnchor`s, and generated lock schema v3. Never call `TASK.score(metrics)` from production.

## Paradigms (`ml_task_type`)

- `dataset` — agent submits a queryable predictor evaluated on private challenge rows.
- `sim_policy` — agent submits `policy.py`, graded over held-out seeds (`grading.policy_eval.run_seeds` + `aggregate`, `failed_fill` = each key's FLOOR).
- `env`/`hybrid` — hidden env over `/tmp/env.sock`; ship `data/private/env.py` (`make_env`) + public `data/public/env_client.py`. If the env is built on a pip simulator, put it in `env_dependencies` (NOT `dependencies`) so it installs root-only at `/mcp_server/env_deps` and the agent can't `import` it to bypass the RPC. See `hidden-env-tasks` skill / `docs/HIDDEN_ENV.md`.

## Bases

`docker-base` + tier -> `mlenvs-gpu` / `mlenvs-cuda-graphics` / `mlenvs-tpu` (ML_Envs-pinned, py3.12/torch2.4.1+cu121, not shared with native flavors). `+graphics` tier -> `mlenvs-cuda-graphics`; TPU tier needs `docker-base="tpu"`.

Local runs take `harness run --flavor {auto,heavy,slim}` (default `auto`): `heavy` = the production base; `slim` = stripped `mlenvs-slim` for low-RAM hosts (compute tasks only — the agent pip-installs ML wheels at runtime); `auto` builds heavy and falls back to slim on a build-time OOM (exit 137 / compiler OOM), granting +50 agent turns. Manifest records `flavor_requested`/`image_flavor`/`flavor_fallback`. Local-only; the Taiga export always uses heavy. (No pkg-manager unblocking needed — lbx's local bash tool doesn't sandbox pip/uv.)

## HuggingFace resources (offline)

No internet at runtime. To ship pretrained weights / HF datasets, declare `hf_resources` in `metadata.json`: a list of bare `"org/name"` strings (model, `main`) or objects `{repo_id, revision, repo_type: model|dataset, allow_patterns, ignore_patterns}`. `scripts/sync_mount.sh` sha-content-addresses each repo, fetches it into the HF hub-cache layout, and mounts it read-only at `/tmp/.cache/huggingface/hub/<repo_type>s--<org>--<name>` (bases set `HF_HOME=/tmp/.cache/huggingface`), so `from_pretrained("org/name")` resolves offline. Downloaded once, shared across tasks. Gated repos need `HF_TOKEN` on the deploy host. Not available on the TPU base. Faithful ML_Envs pipeline (`scripts/pack_hf_resource.py`); see `docs/MLENVS_TASKS.md` §8.

## Calibration gate (committed-model hard contract)

Trusted CI / ground-truth / seal **never train**. For `reference_solution/` and `TASK.naive` (usually `baselines/naive/`) authors **must** commit:

1. `train.py` (provenance only — not executed by CI)
2. trained model artifact(s)
3. `model.manifest.json` with matching digests
4. inference-only `solution.py` (load weights → `/tmp/output`)

Optional `submission.csv` is Tier-B additive only — never a substitute. `results.txt` must not be committed. Reference must score `0.5 ± 0.05`; declared baselines must score clearly below it. See `docs/MLENVS_TASKS.md` §4 and `examples/mle-tabular-classification/`.

Model submissions for agents: both `grading.helpers.load_submitted_model` (ML_Envs pickle/joblib-proxy for `model.pkl` with `predict(...)`) and `grading.helpers.run_model_module` (module-callable `model.py`) are available.

## Reward-hacking discipline

See `docs/REWARD_HACKING.md`. The validator enforces the blocking `agent_fault`, grader-sandbox, and determinism gates over `test_file.py`.
