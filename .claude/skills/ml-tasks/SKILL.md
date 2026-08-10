---
name: ml-tasks
description: Authors continuous-scored ml tasks with held-out truth, sanctioned submission loaders, generated calibration, dataset licensing, hidden environments, dependency channels, and accelerator base flavors. Use for task_type ml.
---

# ML Tasks

`task_type = "ml"` is continuous-scored machine-learning work. It uses the same
native contract as every other task type, plus the committed-strategy
calibration contract. Full guide: `docs/ML_TASKS.md`.
Canonical examples: `examples/mle-tabular-classification/` (static data),
`examples/hidden-env-bandit/` (hidden env). Scaffold with
`uv run lbx-rl-template create --name labelbox/<task_id> --template ml --out problems`.

## Layout

```text
problems/<task_id>/
├── task.toml              # native config (below)
├── metadata.json          # native envelope (benchmark + problem_data)
├── instruction.md         # agent prompt (no anchors, no internals)
├── environment/Dockerfile # builds FROM a flagship base
├── scorer/compute_score.py # the grader
├── scorer/data/           # held-out truth -> /mcp_server/data/ (root-only)
├── data/                  # agent-visible -> /data/ (read-only)
├── solution/              # committed strategy + inference entrypoint + manifest (scores 0.5)
├── baselines/naive/       # weak committed strategy
├── baselines/degenerate/  # explicit non-tabular no-information workspaces
└── data_generation/       # provenance
```

## `task.toml`

Standard sections only — there is no ml-specific section:

```toml
[difficulty]
task_type = "ml"
domain = "scientific_discovery_computational_science"
reward_type = "continuous_scoring_function"
license = "CC0-1.0"
license_source = "https://creativecommons.org/publicdomain/zero/1.0/"
```

Timeouts are pinned centrally for `ml` (grading at the Taiga maximum), and
`allow_internet = true` is rejected — never author either.

## Dependency channels

Declared as files, not `task.toml` fields, so private package names never reach
the agent-visible `/task/task.toml`. `base/install-task-deps.sh` routes each to
its isolation boundary:

| File | Installs into | Visible to |
| --- | --- | --- |
| `environment/apt.txt` | system apt | agent |
| `environment/requirements.txt` | `/opt/lbx-runtime/.venv` | agent |
| `scorer/requirements.txt` | `/mcp_server/grading_deps` (0700 root) | grader only |
| `scorer/env-requirements.txt` | `/mcp_server/env_deps` (0700 root) | hidden env server only |

The grader-only channel is for a scoring/reference library that would leak the
intended approach if agent-visible; the env-only channel is for a simulator the
agent must reach only through the RPC. A package in both an agent-visible and a
private channel fails validation.

## `scorer/compute_score.py`

```python
from grading.evaluation import ContinuousTask, PrivateTableChallenge, PythonPredictor

TASK = ContinuousTask.model(
    artifact=PythonPredictor("predictor.py"),
    challenge=PrivateTableChallenge(
        "challenge.parquet", feature_columns=["x1", "x2"], sample_size=256
    ),
    targets=[...],
)


def compute_score(workspace, trajectory, private):
    return TASK.compute_score(workspace=workspace, private=private)
```

- Follow `docs/CONTINUOUS_EVALUATION.md`; new tasks use queryable Tier-A artifacts.
- A no-arg `compute_score()` is also accepted; it reads the baked paths directly.
- `raise AgentFault` for agent faults (kept 0.0); let author/infra faults propagate (discarded). No broad `except: return 0.0`; no exec/pickle of agent artifacts in the grader.
- Read agent output only via sanctioned component-safe loaders (all flat):
  `grading.helpers` (`load_submission_or_fault` CSV,
  `load_submission_npz_or_fault` NumPy,
  `open_submission_file_or_fault` custom file-like parsers,
  `run_submitted_executable`, `load_submission_h5_or_fault`,
  `load_submitted_model`), `grading.policy_eval` (`run_seeds`/`aggregate` for
  sim_policy), `grading.env_loading` (`load_env_module`), and `grading.kfold`
  (`score_kfold_cv`); `env_server.policy_loader.load_submitted_policy` for
  env/hybrid. Never use raw `lstat` or leaf-only `O_NOFOLLOW`.
- Keep submitted-code privilege dropping enabled and NumPy
  `allow_pickle=False`. Whole-object H5AD cannot be handed back to the root
  grader; evaluate it inside a dropped worker or expose bounded primitive HDF5
  datasets.
- Use registered targets, reviewed `FloorAnchor`s, and generated lock schema v3. Never call `TASK.score(metrics)` from production.

## Shapes (nothing to declare)

You never classify the task; grading follows from what you build, so a task that
mixes these needs no extra bookkeeping.

- **Static data** — agent submits a queryable predictor evaluated on private challenge rows. This is the default.
- **Policy over seeds** — agent submits `policy.py` and the grader declares `PolicyEvaluationTask` (`grading.policy_eval.run_seeds` + `aggregate`, `failed_fill` = each key's FLOOR). The TASK class is the declaration.
- **Hidden env** — set `[environment].hidden_env` to `env` or `hybrid` for an env over `/tmp/env.sock`; ship `scorer/data/env.py` (`make_env`) + public `data/env_client.py`. Put a pip simulator in `scorer/env-requirements.txt`, NOT `environment/requirements.txt`. See `hidden-env-tasks` skill / `docs/HIDDEN_ENV.md`.

## Bases

The resource tier resolves the flagship base flavor: H100 tier -> `gpu`,
`+graphics` tier -> `cuda-graphics`, TPU tier -> `tpu`, otherwise `cpu`.
Override with `[environment].base_flavor` only when the tier is ambiguous; an
incompatible pair is rejected. `gpu`/`cpu` are py3.13; `cuda-graphics` and `tpu`
pin py3.12 for wheel availability. Each flavor has its own drift-hashed tag.

## HuggingFace resources (offline)

No internet at runtime. To ship pretrained weights / HF datasets, declare
`[[preloaded_files]]` entries with `hf_repo` (plus optional `hf_revision`,
`repo_type` = `model`|`dataset`, `allow_patterns`, `ignore_patterns`); leave
`mount_path` unset. `scripts/sync_mount.sh` sha-content-addresses each repo,
fetches it into the HF hub-cache layout, and mounts it read-only at
`/tmp/hf-cache/hub/<repo_type>s--<org>--<name>` (bases set
`HF_HOME=/tmp/hf-cache`), so `from_pretrained("org/name")` resolves offline.
Downloaded once, shared across tasks. Gated repos need `HF_TOKEN` on the deploy
host. Not available on the TPU base. See `docs/ML_TASKS.md` §8.

## Calibration gate (committed-strategy contract)

Trusted CI / ground-truth / seal **never train**. For `solution/` and
`TASK.naive` (usually `baselines/naive/`) authors commit an inference-only
entrypoint plus one explicit manifest:

- `model.manifest.json` v1 for existing trained models; or
- `strategy.manifest.json` with `kind=trained_model` (digest-bound training inputs/artifacts) or `kind=committed_artifact` (hand-authored policy/static artifact, no fake training fields).

Numeric tabular tasks use the built-in constant/shuffle family. `ContinuousTask` policy, env/hybrid, executable, and non-numeric callback tasks declare `WorkspaceDegenerateProbes` whose paths are ready-to-measure workspaces under `baselines/degenerate/`; `PolicyEvaluationTask` retains its separate paired controls. Every workspace probe is digest-bound, measured twice under one calibration seed, and must succeed deterministically.

Reference must score `0.5 ± 0.05`. Naive is weak-positive by default; an exact tie with every effective no-information floor requires `naive_score_min=0.0` plus `naive_at_floor=AnchorRationale(kind="reviewed_exception", ...)`. See `docs/ML_TASKS.md` §4.

The image bakes the calibration lock at `/mcp_server/calibration/` alongside an
`.author-source` marker; that marker is what lets the rubric server refuse the
author's fallback when the trusted-CI promoted mount is required.

Model submissions for agents: both `grading.helpers.load_submitted_model`
(sandboxed pickle/joblib proxy for `model.pkl` with `predict(...)`) and
`grading.helpers.run_model_module` (module-callable `model.py`) are available.

## Reward-hacking discipline

See `docs/REWARD_HACKING.md`. The validator enforces the blocking `agent_fault`, grader-sandbox, and determinism gates over `scorer/compute_score.py`.
