# Authoring ML_Envs-mode Tasks

ML_Envs mode is the **minimal authoring contract** for ML tasks in this template. The contributor edits a tiny `metadata.json`, the prompt, the grader, and the data — nothing else. There is **no `task.toml`, no per-task `environment/Dockerfile`, and no `tests/test.sh`**: every operational field is pinned centrally, the image builds from the shared `base/task.mlenvs.Dockerfile`, and QA fixes that should be universal are made in one place instead of task-by-task.

Canonical example: [`examples/mle-tabular-classification/`](../examples/mle-tabular-classification/).

> This is additive. MuJoCo / CFD / structures and hand-authored `ml` tasks keep the native `task.toml` contract (see [`AUTHORING.md`](AUTHORING.md)); ML_Envs mode does not affect them.

## 1. Layout

```text
problems/<task_id>/
├── metadata.json          # minimal config (section 2)
├── prompt.md              # agent-facing prompt (no anchors, no internals)
├── test_file.py           # declarative grading.evaluation TASK (section 3)
├── calibration.lock.json  # ignored local cache; CI generates production lock
├── .alignerr/calibration.evidence.json # ignored local cache identity
├── data/
│   ├── public/            # agent-visible  -> /data/
│   └── private/           # root-only truth -> /mcp_server/data/
├── reference_solution/    # committed model + train.py + solution.py + manifest
├── baselines/naive/       # weak committed model + reproducible recipe
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

## 3. `test_file.py` and generated calibration

New ML tasks use the sealed challenge API described in
[`CONTINUOUS_EVALUATION.md`](CONTINUOUS_EVALUATION.md). The agent submits a
queryable predictor and the grader selects private rows after commitment:

```python
from grading.evaluation import (
    AnchorRationale,
    ContinuousTask,
    FloorAnchor,
    GeneratedCalibration,
    PopulationSRETarget,
    PrivateTableChallenge,
    PythonPredictor,
)

TASK = ContinuousTask.model(
    artifact=PythonPredictor("predictor.py"),
    challenge=PrivateTableChallenge(
        "challenge.parquet",
        feature_columns=["feature_1", "feature_2"],
        sample_size=256,
    ),
    targets=[
        PopulationSRETarget.lower(
            "pred",
            truth_column="target",
            weight=1.0,
            floor=FloorAnchor(
                value=1.0,
                rationale=AnchorRationale(
                    kind="theoretical",
                    summary=(
                        "A constant prediction at the population mean has "
                        "population-standardized RMSE exactly one."
                    ),
                ),
            ),
            perfect=0.0,
        )
    ],
    calibration=GeneratedCalibration("calibration.lock.json"),
    naive="baselines/naive",
)

def compute_score():
    return TASK.compute_score()
```

Rules:

- **Return a float in `[0, 1]`** (a score dict with `score` + `subscores` is also accepted; the headline `score` is authoritative).
- **Never raise for author/infra faults** — let them propagate so the runner discards the attempt (`env_internal_failure`). `raise AgentFault` only for agent-controlled failures (missing/malformed submission, wrong row count); those are kept as a clean 0.0.
- **Read agent artifacts only through the sanctioned loaders** — `grading.helpers` (`load_submission_or_fault` CSV, `load_submission_npz_or_fault` .npz/.npy, `load_submission_h5_or_fault` HDF5, `run_submitted_executable`, `load_submitted_model`), `grading.policy_eval` (`run_seeds` / `aggregate`), `grading.env_loading` (`load_env_module`), `grading.kfold` (`score_kfold_cv`) — never by hand and never `exec`/`pickle` of agent code in the (root) grader. A bare `except OSError` does **not** stop a symlink to the held-out truth (which the agent can re-plant after the pre-grade scrub) — the read *succeeds* and scores the truth as the submission. Neither does an `lstat` + `S_ISREG` check: it is check-then-use on the path, and a uid-1000 process that survives a mid-grade `run_submitted_executable` / `run_policy` races it. If no loader fits, open the descriptor yourself with `os.open(path, os.O_RDONLY | os.O_NOFOLLOW)` (a symlink leaf fails atomically at open) and read *that* fd — never re-open the path.
- **Run ground truth locally only when you need calibration feedback.** The
  framework measures committed reference/naive models, writes an ignored
  development lock/evidence bundle, and replays no-op/reference/oracle
  contracts. Trusted CI independently generates or restores the production
  bundle and mounts it read-only in Taiga. Do not edit or commit generated
  calibration values.
- **Floors are reviewed semantic choices, not baseline measurements.** Every
  floor carries an `AnchorRationale`; baselines only prove weak genuine work
  remains above floor and below reference.
- **Metric names are not formulas.** Use exact versioned definitions such as
  `sre.rmse_over_population_std.v1`; documentation and the generated lock expose
  denominator, `ddof`, threshold, label, and averaging conventions.
- **Never call `TASK.score(metrics)` from production.** It is calibration-only.
  Legacy static tasks temporarily call `TASK.grade(submission, truth)`; follow
  the version-by-version migration guide in `CONTINUOUS_EVALUATION.md`.

The image bakes `test_file.py` as the grader; the no-arg signature and the `grading.*` helper import surface are handled by the grader runtime.

## 4. Reference + baselines (calibration gate)

- `reference_solution/` — committed trained model, `train.py`, inference
  `solution.py`, and `model.manifest.json`. It must score `0.5 ± 0.05`.
- `baselines/naive/` — committed weak but input-dependent model plus the same
  reproducibility surface. It must earn a small positive score below the
  reference; constant/mean strategies remain null probes at zero.
- Generated submissions and `results.txt` are not committed.

Finalize in one command:

```bash
uv run lbx-rl-harness run \
  --runtime ground-truth \
  --problem-dir problems/<task_id>
```

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

## Public payloads and data mounts

For every `ml_task_type`, `sync_mount.sh` exposes `data/public -> /data` in
Taiga's payload inventory. Public trees up to 256 files and 1 GiB are uploaded
as explicit regular `preloaded_files`; when a tree is larger, regular files
named verbatim in `prompt.md` are still exposed as payload-only copies under
`/lbx-public-files`, while the complete canonical tree remains at `/data`. This
lets Taiga Data Quality resolve logical filenames such as `column_mapping.json`
instead of seeing only an opaque image or squashfs. Root-side setup re-owns
both public locations and changes them to `0444`/`0555`, preserving the
immutable public-data contract.

For `ml_task_type = "dataset"`, a public tree too large for bounded expansion
falls back to a content-addressed read-only squashfs, and `data/private` is
always mounted read-only behind the baked `0700` `/mcp_server` boundary.
`env`, `hybrid`, and `sim_policy` retain their baked data so environment startup
semantics do not change; their explicit public entries are identical
QA-visible overlays. The baked copy also keeps local harness runs working.

## References

- [`examples/mle-tabular-classification/`](../examples/mle-tabular-classification/) — canonical ML_Envs-mode task.
- [`REWARD_HACKING.md`](REWARD_HACKING.md) — mitigations every scorer must respect.
- [`GRADING.md`](GRADING.md) — full grader contract + calibration.
- [`HIDDEN_ENV.md`](HIDDEN_ENV.md) — `env` / `hybrid` tasks.
- [`ML_ENVS_MIGRATION.md`](ML_ENVS_MIGRATION.md) — porting an existing ML_Envs task.
