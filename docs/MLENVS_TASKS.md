# Authoring ML_Envs-mode Tasks

ML_Envs mode is the **minimal authoring contract** for ML tasks in this template. The contributor edits a tiny `metadata.json`, the prompt, the grader, and the data — nothing else. There is **no `task.toml`, no per-task `environment/Dockerfile`, and no `tests/test.sh`**: every operational field is pinned centrally, the image builds from the shared `base/task.mlenvs.Dockerfile`, and QA fixes that should be universal are made in one place instead of task-by-task.

Canonical example: [`examples/mle-tabular-classification/`](../examples/mle-tabular-classification/).

> This is additive. MuJoCo / CFD / structures and hand-authored `ml` tasks keep the native `task.toml` contract (see [`AUTHORING.md`](AUTHORING.md)); ML_Envs mode does not affect them.

## 1. Layout

```text
problems/<task_id>/
├── metadata.json          # minimal config (section 2)
├── prompt.md              # agent-facing prompt (no anchors, no internals)
├── test_file.py           # no-arg compute_score() (section 3)
├── data/
│   ├── public/            # agent-visible  -> /data/
│   └── private/           # root-only truth -> /mcp_server/data/
├── reference_solution/    # solution.py + captured outputs + results.txt
├── baselines/             # >=1 naive baseline whose output scores below the reference
└── data-generation/       # provenance for the data
```

A task dir is detected as ML_Envs mode when it has **no `task.toml`** and ships a `test_file.py` (or a `metadata.json` with `ml_task_type`). Detection, synthesis, and the pinned constants live in `alignerr_plugin.mlenvs`.

## 2. `metadata.json`

Required keys:

| Key | Meaning |
| --- | --- |
| `ml_task_type` | `dataset` \| `env` \| `hybrid` \| `sim_policy` (the grading paradigm) |
| `required_resources` | one Taiga resource enum, verbatim |
| `domain` | one ml-scoped domain (diversity tracking) |
| `license` | permissive SPDX id, or `self_generated` |
| `license_source` | upstream license URL, or a justification for `self_generated` |

Optional keys (defaults shown):

| Key | Default | Meaning |
| --- | --- | --- |
| `docker-base` | `default` | `default` -> `mlenvs-gpu`; `cuda-graphics` -> `mlenvs-cuda-graphics`; `tpu` -> `mlenvs-tpu` |
| `dependencies` | `[]` | extra pip requirements installed system-wide (**agent-visible**) |
| `apt_extras` | `[]` | extra apt packages baked into the task image |
| `env_dependencies` | `[]` | pip requirements for the hidden env server **only** (`env`/`hybrid`); installed root-only so the agent can't import them (see [`HIDDEN_ENV.md`](HIDDEN_ENV.md)) |
| `description` | `""` | one-line human description (surfaced in the Taiga payload) |
| `hf_resources` | `[]` | read-only Hugging Face repos mounted for offline `from_pretrained` (section 8) |

```json
{
  "ml_task_type": "dataset",
  "required_resources": "12vcpu+100gib+h100/2",
  "domain": "scientific_discovery_computational_science",
  "license": "CC0-1.0",
  "license_source": "https://creativecommons.org/publicdomain/zero/1.0/"
}
```

**Pinned centrally (never authored):** `task_type = "ml"`, `reward_type = "continuous_scoring_function"`, `allow_internet = false`, all timeouts (grading pinned to the Taiga max), the runner knobs, the model name (obscured at submit), and the `/tmp/output` submission convention. To change a pinned value for all tasks, edit `alignerr_plugin.mlenvs` — do not add a per-task override.

### `ml_task_type` -> paradigm

- `dataset` — static held-out data; the agent writes a submission file.
- `sim_policy` — the agent submits a `policy.py`, evaluated over held-out seeds (`grading.policy_eval.run_seeds` + `aggregate`).
- `env` / `hybrid` — the agent probes a hidden env over `/tmp/env.sock`; ship the held-out env at `data/private/env.py` (`make_env`) and a public `data/public/env_client.py`. See [`HIDDEN_ENV.md`](HIDDEN_ENV.md).

### Resource / base selection

`docker-base` + the resource tier pick the ML_Envs-specific base flavor (`mlenvs-gpu` / `mlenvs-cuda-graphics` / `mlenvs-tpu`) — rebuilt on ML_Envs's H100-validated pins and **not shared** with the native verticals. A `+graphics` resource tier routes to `mlenvs-cuda-graphics`; a TPU tier requires `docker-base = "tpu"`.

## 3. `test_file.py`

`compute_score()` takes **no arguments** and reads the baked runtime paths:

```python
from pathlib import Path

from grading import calibration
from grading.faults import AgentFault
from grading.helpers import load_submission_or_fault

SUBMISSION_DIR = Path("/tmp/output")     # agent's submission
PRIVATE_DATA = Path("/mcp_server/data")  # root-only held-out truth (data/private/)


def compute_score() -> float:
    truth = _load_truth(PRIVATE_DATA)                    # author data: propagate on failure
    try:
        sub = load_submission_or_fault(SUBMISSION_DIR / "submission.csv",
                                       required_columns=["id", "pred"])
    except AgentFault:
        raise                                            # agent fault -> kept 0.0
    rmse = _metric(sub, truth)
    x = calibration.progress_lower_better(rmse, floor=1.0, perfect=0.0)
    return calibration.PiecewiseLinearCurve.from_reference(X_REF).score(x)
```

Rules:

- **Return a float in `[0, 1]`** (a score dict with `score` + `subscores` is also accepted; the headline `score` is authoritative).
- **Never raise for author/infra faults** — let them propagate so the runner discards the attempt (`env_internal_failure`). `raise AgentFault` only for agent-controlled failures (missing/malformed submission, wrong row count); those are kept as a clean 0.0.
- **Read agent artifacts only through the sanctioned loaders** — `grading.helpers` (`load_submission_or_fault` CSV, `load_submission_npz_or_fault` .npz/.npy, `load_submission_h5_or_fault` HDF5, `run_submitted_executable`, `load_submitted_model`), `grading.policy_eval` (`run_seeds` / `aggregate`), `grading.env_loading` (`load_env_module`), `grading.kfold` (`score_kfold_cv`) — never by hand and never `exec`/`pickle` of agent code in the (root) grader. A bare `except OSError` does **not** stop a symlink to the held-out truth (which the agent can re-plant after the pre-grade scrub) — the read *succeeds* and scores the truth as the submission. Neither does an `lstat` + `S_ISREG` check: it is check-then-use on the path, and a uid-1000 process that survives a mid-grade `run_submitted_executable` / `run_policy` races it. If no loader fits, open the descriptor yourself with `os.open(path, os.O_RDONLY | os.O_NOFOLLOW)` (a symlink leaf fails atomically at open) and read *that* fd — never re-open the path.
- **Calibrate with `FLOOR / REF / PERFECT` + `PiecewiseLinearCurve`** (`grading.calibration`). `ExponentialCurve` is deprecated and rejected. `FLOOR` = worst plausible raw metric (not a baseline's measured value); `REF` (the reference's metric) -> 0.5; `PERFECT` = the optimum.

The image bakes `test_file.py` as the grader; the no-arg signature and the `grading.*` helper import surface are handled by the grader runtime.

## 4. Reference + baselines (calibration gate)

- `reference_solution/` — the reference (`solution.py` + its captured output + `results.txt`). It must score `0.5 ± 0.05`.
- `baselines/<name>/` — naive solutions (constant / mean predictor, ...) whose committed outputs score clearly below the reference. This is the **learnability gate**: a constant guess must not match the expert.

## 5. Licensing

`license` must be a permissive SPDX id (`MIT`, `Apache-2.0`, `BSD-2/3-Clause`, `ISC`, `Unlicense`, `CC0-1.0`, `CC-BY-4.0`, `PDDL-1.0`, `UPL-1.0`) or `self_generated`. Trace the dataset license **upstream**; copyleft / non-commercial / research-only data is rejected. `license_source` is the upstream http(s) URL where you confirmed it (or a justification for `self_generated`). Validation checks the allowlist mechanically; a licensing code-owner approves the PR.

## 6. Prompt guardrails

Name installed tools/libraries directly in `prompt.md`; do **not** tell the agent to inspect the Docker/base image, `metadata.json`, or "declared dependencies" to discover runtime packages, and do **not** mention the anchors or how `compute_score` is structured. GPU/TPU availability and dedicated-`tmux` guidance are appended automatically at export.

## 7. Base images

ML_Envs-mode tasks build on the `mlenvs-*` bases (`base/mlenvs-gpu/`, `base/mlenvs-cuda-graphics/`, `base/mlenvs-tpu/`, `base/mlenvs-slim/`, + local-only Blackwell overlays) — faithful ports of ML_Envs's H100-validated images (Python 3.12, torch 2.4.1+cu121, ...) that bake this template's grader runtime. They are rebuilt/pushed via `base/build_and_push.sh --flavors mlenvs-gpu,mlenvs-cuda-graphics,mlenvs-tpu,mlenvs-slim`.

On a Blackwell dev GPU (sm_100/sm_120, e.g. an RTX 5090), the local harness auto-detects it and swaps in the cu128 blackwell overlay for local builds only; the Taiga export always targets the cu121 base for the H100 runners. Override with `LBX_RL_TASKS_LOCAL_BLACKWELL=1`/`0`.

### Compute flavor (`--flavor`) for low-RAM local hosts

`lbx-rl-tasks-harness run` takes `--flavor {auto,heavy,slim}` (default `auto`) for ML_Envs tasks — a port of ML_Envs's flavor mechanism for dev machines that can't build the full heavy base:

- **`heavy`** — the production-equivalent `mlenvs-gpu` / `mlenvs-cuda-graphics` / `mlenvs-tpu` base the Taiga runners use (with the local Blackwell overlay swap). This is what a real run resolves to.
- **`slim`** — the stripped `mlenvs-slim` base (no torch / ML wheels); the agent pip-installs what it needs at runtime. Only valid for ML_Envs GPU base — cuda-graphics and TPU tasks need their heavy base, so `--flavor slim` is rejected for them.
- **`auto`** (default) — build heavy, and on a **build-time OOM** (exit 137 or a from-source-wheel compiler OOM, e.g. LightGBM) fall back to slim when the task allows it, granting the agent `+50` turns to offset the runtime install cost. On a capable host `auto` is always heavy, so behavior is unchanged.

Unlike ML_Envs, the lbx local bash tool does not sandbox package managers, so nothing needs "unblocking" under slim — the agent can `pip`/`uv install` directly. The run manifest records `flavor_requested`, `image_flavor`, and `flavor_fallback`. This only affects **local** runs; the Taiga export always uses the heavy base for the production runners.

A slim run is for iterating on task plumbing only: its score is not guaranteed to be calibration-equivalent to the heavy production base (different base image, agent-installed wheels), so confirm the reference/baseline anchors on `--flavor heavy` (or the Taiga runners) before trusting them.

## 8. Hugging Face resources (offline weights / datasets)

There is **no internet at runtime**, so a task that needs pretrained weights or a HuggingFace dataset declares them in `metadata.json:hf_resources`. Each repo is fetched once at deploy time, packed into a content-addressed squashfs, and mounted read-only into the agent's HF hub cache, so `from_pretrained("org/name")` (or `load_dataset(...)`) resolves entirely from the mount.

```json
{
  "hf_resources": [
    "sentence-transformers/all-MiniLM-L6-v2",
    { "repo_id": "org/name", "repo_type": "dataset", "revision": "v2",
      "allow_patterns": ["*.json"], "ignore_patterns": ["*.bin"] }
  ]
}
```

Each entry is either a bare `"org/name"` string (a model, `main`) or an object: `repo_id` (required), `revision` (default `main`; a tag, branch, or 40-hex commit sha), `repo_type` (`model` | `dataset`), and optional `allow_patterns` / `ignore_patterns` to narrow what is downloaded. The mount lands at `/tmp/.cache/huggingface/hub/<repo_type>s--<org>--<name>`, matching HF's own cache scheme; the bases set `HF_HOME=/tmp/.cache/huggingface`.

Mechanics: `scripts/sync_mount.sh` resolves each repo's immutable commit sha (a cheap metadata call, no weights), so the remote object is content-addressed by `repo@sha` under a shared `cache/huggingface/` prefix and downloaded/packed **once** across all tasks that reference it. `hf_resources` is **not** available on the TPU base. This is the faithful ML_Envs HF pipeline (`scripts/pack_hf_resource.py`).

**Token (gated repos only).** Ungated permissive repos (the common case, and the only ones that pass licensing) need no token. A **gated** repo needs `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) in the environment where `sync_mount.sh` runs — never at task runtime. Two sources, mirroring ML_Envs:

- **CI / deploy:** the grade workflow injects it from a repo secret — `HF_TOKEN: ${{ secrets.HF_TOKEN }}` (as ML_Envs's `submit-to-taiga.yml` does).
- **Local:** `export HF_TOKEN=hf_...` before running `sync_mount.sh`.

`pack_hf_resource.py` reads it from the env, so both paths work with no code change.

## Data mounts (`dataset` tasks)

For `ml_task_type = "dataset"`, `sync_mount.sh` also packs `data/public` and `data/private` into content-addressed squashfs and mounts them read-only — `data/public -> /data` (agent-visible) and `data/private -> /mcp_server/data` (root-only, since `/mcp_server` is baked mode `0700`). This mirrors ML_Envs's `pack_squashfs.sh`. `env` / `hybrid` / `sim_policy` tasks bake their data instead. The task image also bakes the data so **local** harness runs work; at **deploy** the squashfs overrides that layer and dedups across task versions.

## References

- [`examples/mle-tabular-classification/`](../examples/mle-tabular-classification/) — canonical ML_Envs-mode task.
- [`REWARD_HACKING.md`](REWARD_HACKING.md) — mitigations every scorer must respect.
- [`GRADING.md`](GRADING.md) — full grader contract + calibration.
- [`HIDDEN_ENV.md`](HIDDEN_ENV.md) — `env` / `hybrid` tasks.
- [`ML_ENVS_MIGRATION.md`](ML_ENVS_MIGRATION.md) — porting an existing ML_Envs task.
