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
├── reference_solution/    # solution.py + captured output + results.txt (scores 0.5)
├── baselines/<name>/      # naive output scoring below the reference (learnability gate)
└── data-generation/       # provenance
```

Detected as ML_Envs mode when there is no `task.toml` and a `test_file.py` (or `metadata.json` has `ml_task_type`). Synthesis + pinned constants: `alignerr_plugin.mlenvs`.

## `metadata.json`

Required: `ml_task_type` (`dataset`|`env`|`hybrid`|`sim_policy`), `required_resources` (a Taiga enum), `domain` (ml-scoped), `license` (permissive SPDX or `self_generated`), `license_source`. Optional: `docker-base` (`default`|`cuda-graphics`|`tpu`), `dependencies` (pip, agent-visible), `apt_extras` (apt), `env_dependencies` (pip for the hidden env server ONLY — env/hybrid; installed root-only so the agent can't import them; see below), `grading_dependencies` (pip for the GRADER ONLY — any task type; installed root-only under `/mcp_server/grading_deps` and prepended to the grader's `sys.path`, so the root grader can import them but the uid-1000 agent cannot — for a scoring/reference library that would leak the intended approach if it were agent-visible; a package in both `dependencies` and `grading_dependencies` fails validation), `description` (one-line human blurb), `hf_resources` (read-only HuggingFace mounts — see below). Everything else (`task_type=ml`, `reward_type=continuous_scoring_function`, `allow_internet=false`, timeouts, runner knobs, `/tmp/output`) is pinned — never author it.

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
from pathlib import Path
from grading import calibration
from grading.faults import AgentFault
from grading.helpers import load_submission_or_fault

SUBMISSION_DIR = Path("/tmp/output")
PRIVATE_DATA = Path("/mcp_server/data")

def compute_score() -> float:
    truth = _load_truth(PRIVATE_DATA)                 # author data: propagate on failure (discard)
    try:
        sub = load_submission_or_fault(SUBMISSION_DIR / "submission.csv",
                                       required_columns=["id", "pred"])
    except AgentFault:
        raise                                         # agent fault -> kept 0.0
    x = calibration.progress_lower_better(_metric(sub, truth), floor=1.0, perfect=0.0)
    return calibration.PiecewiseLinearCurve.from_reference(X_REF).score(x)
```

- **No arguments**; read `/tmp/output` + `/mcp_server/data` directly. Return a float in `[0,1]` (or a `{score, subscores}` dict).
- `raise AgentFault` for agent faults (kept 0.0); let author/infra faults propagate (discarded). No broad `except: return 0.0`; no exec/pickle of agent artifacts in the grader.
- Read agent output only via the sanctioned loaders (all flat): `grading.helpers` (`load_submission_or_fault` CSV, `run_submitted_executable`, `load_submission_h5_or_fault`, `load_submitted_model`), `grading.policy_eval` (`run_seeds`/`aggregate` for sim_policy), `grading.env_loading` (`load_env_module`), `grading.kfold` (`score_kfold_cv`); `env_server.policy_loader.load_submitted_policy` for env/hybrid.
- Calibrate with `FLOOR/REF/PERFECT` + `PiecewiseLinearCurve` (`grading.calibration`). `ExponentialCurve` is deprecated/rejected. `FLOOR` = worst plausible raw metric (not a baseline's value); `REF` -> 0.5.

## Paradigms (`ml_task_type`)

- `dataset` — static held-out data; agent writes a submission file.
- `sim_policy` — agent submits `policy.py`, graded over held-out seeds (`grading.policy_eval.run_seeds` + `aggregate`, `failed_fill` = each key's FLOOR).
- `env`/`hybrid` — hidden env over `/tmp/env.sock`; ship `data/private/env.py` (`make_env`) + public `data/public/env_client.py`. If the env is built on a pip simulator, put it in `env_dependencies` (NOT `dependencies`) so it installs root-only at `/mcp_server/env_deps` and the agent can't `import` it to bypass the RPC. See `hidden-env-tasks` skill / `docs/HIDDEN_ENV.md`.

## Bases

`docker-base` + tier -> `mlenvs-gpu` / `mlenvs-cuda-graphics` / `mlenvs-tpu` (ML_Envs-pinned, py3.12/torch2.4.1+cu121, not shared with native flavors). `+graphics` tier -> `mlenvs-cuda-graphics`; TPU tier needs `docker-base="tpu"`.

Local runs take `harness run --flavor {auto,heavy,slim}` (default `auto`): `heavy` = the production base; `slim` = stripped `mlenvs-slim` for low-RAM hosts (compute tasks only — the agent pip-installs ML wheels at runtime); `auto` builds heavy and falls back to slim on a build-time OOM (exit 137 / compiler OOM), granting +50 agent turns. Manifest records `flavor_requested`/`image_flavor`/`flavor_fallback`. Local-only; the Taiga export always uses heavy. (No pkg-manager unblocking needed — lbx's local bash tool doesn't sandbox pip/uv.)

## HuggingFace resources (offline)

No internet at runtime. To ship pretrained weights / HF datasets, declare `hf_resources` in `metadata.json`: a list of bare `"org/name"` strings (model, `main`) or objects `{repo_id, revision, repo_type: model|dataset, allow_patterns, ignore_patterns}`. `scripts/sync_mount.sh` sha-content-addresses each repo, fetches it into the HF hub-cache layout, and mounts it read-only at `/tmp/.cache/huggingface/hub/<repo_type>s--<org>--<name>` (bases set `HF_HOME=/tmp/.cache/huggingface`), so `from_pretrained("org/name")` resolves offline. Downloaded once, shared across tasks. Gated repos need `HF_TOKEN` on the deploy host. Not available on the TPU base. Faithful ML_Envs pipeline (`scripts/pack_hf_resource.py`); see `docs/MLENVS_TASKS.md` §8.

## Calibration gate

Reference (`reference_solution/`) must score `0.5 ± 0.05`; committed baselines must score clearly below it. Model submissions: both `grading.helpers.load_submitted_model` (ML_Envs pickle/joblib-proxy for `model.pkl` with `predict(...)`) and `grading.helpers.run_model_module` (module-callable `model.py`) are available.

## Reward-hacking discipline

See `docs/REWARD_HACKING.md`. The validator enforces the blocking `agent_fault`, grader-sandbox, and determinism gates over `test_file.py`.
