# Authoring ML Tasks

`ml` is the task type for continuous-scored machine-learning work: static-dataset
prediction, submitted policies, and hidden-environment interaction. It uses the
same native contract as every other task type (`task.toml`, `instruction.md`,
`scorer/compute_score.py`, `environment/Dockerfile`, `solution/`) plus the
committed-strategy calibration contract below.

Canonical example: [`examples/mle-tabular-classification/`](../examples/mle-tabular-classification/).
Scaffold a new one with `lbx-rl-template new <name> --template ml`.

## 1. Layout

```text
problems/<task_id>/
├── task.toml              # native config (section 2)
├── metadata.json          # native envelope (benchmark + problem_data)
├── instruction.md         # agent-facing prompt (no anchors, no internals)
├── environment/
│   ├── Dockerfile         # builds FROM a flagship base (section 7)
│   ├── requirements.txt   # optional: agent-visible pip deps
│   └── apt.txt            # optional: agent-visible apt packages
├── scorer/
│   ├── compute_score.py   # declarative grading.evaluation TASK (section 3)
│   ├── data/              # root-only truth -> /mcp_server/data/
│   ├── requirements.txt   # optional: grader-only pip deps (root-only)
│   └── env-requirements.txt # optional: hidden-env-only pip deps (root-only)
├── data/                  # agent-visible -> /data/
├── calibration.lock.json  # ignored local cache; CI generates production lock
├── .alignerr/calibration.evidence.json # ignored local cache identity
├── solution/              # committed strategy + inference entrypoint + manifest
├── baselines/naive/       # weak committed strategy
├── baselines/degenerate/  # explicit non-tabular no-information workspaces
└── data_generation/       # provenance for the data
```

## 2. `task.toml`

An ml task fills in the standard sections; there is no ml-specific section:

```toml
[environment]
required_resources = "12vcpu+100gib+h100/2"

[difficulty]
task_type = "ml"
domain = "scientific_discovery_computational_science"
reward_type = "continuous_scoring_function"
license = "CC0-1.0"
license_source = "https://creativecommons.org/publicdomain/zero/1.0/"
```

**You do not classify your task.** How the submission is graded follows from what
you actually build, so there is nothing to keep in sync and nothing to get wrong
on a task that does several of these at once:

- **Static held-out data** — the agent submits a queryable predictor scored
  against `scorer/data/`. This is the default: write the grader and ship the data.
- **Policy over held-out seeds** — the agent submits a `policy.py` and your
  grader declares `PolicyEvaluationTask` (`grading.policy_eval.run_seeds` +
  `aggregate`). The grader's TASK class *is* the declaration.
- **Hidden env** — the agent probes an env over `/tmp/env.sock`. Set
  `[environment].hidden_env` to `env` (socket only) or `hybrid` (socket plus
  static `data/` files), ship the held-out env at `scorer/data/env.py`
  (`make_env`) and a public `data/env_client.py`. See
  [`HIDDEN_ENV.md`](HIDDEN_ENV.md).

A task can combine these freely — a hidden-env task whose grader rolls out a
policy needs no extra bookkeeping.

**Pinned centrally for `task_type = "ml"` (authored values are ignored):** all
timeouts, with grading pinned to the Taiga maximum. `allow_internet = true` is
rejected outright — the sandbox is offline, so fetch everything at build time.

### Dependency channels

Dependencies are declared as **files**, not `task.toml` fields, so a private
package name never reaches the agent-visible `/task/task.toml`. Each file is
optional; `base/install-task-deps.sh` routes each one to its isolation boundary:

| File | Installs into | Visible to |
| --- | --- | --- |
| `environment/apt.txt` | system apt | agent |
| `environment/requirements.txt` | `/opt/lbx-runtime/.venv` | agent |
| `scorer/requirements.txt` | `/mcp_server/grading_deps` (0700 root) | grader only |
| `scorer/env-requirements.txt` | `/mcp_server/env_deps` (0700 root) | hidden env server only |

The two private channels matter for `env`/`hybrid`: installing a simulator like
`myosuite` agent-visibly would let the agent drive the raw env and bypass the RPC
constraints entirely.

### Resource / base selection

The resource tier picks the base flavor automatically: an H100 tier routes to
`gpu`, a `+graphics` tier to `cuda-graphics`, a TPU tier to `tpu`, and anything
else to `cpu`. Override with `[environment].base_flavor` only when the tier alone
is ambiguous; an incompatible pair is rejected.

## 3. `scorer/compute_score.py` and generated calibration
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
    artifact=PythonPredictor(
        "predictor.py",
        predict_timeout_s=60,
        first_call_timeout_s=120,
        max_rows=100_000,
        prediction_scope="row_independent",
    ),
    challenge=PrivateTableChallenge(
        "challenge.parquet",
        feature_columns=["feature_1", "feature_2"],
        # Omit sample_size to evaluate the full private bank. For a large bank,
        # set sample_size=N and selection_policy="stable_subset".
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
    calibration=GeneratedCalibration(
        "calibration.lock.json",
        quality_floor_mode="effective_no_info",
    ),
    naive="baselines/naive",
)

def compute_score():
    return TASK.compute_score()
```

Rules:

- **Return a float in `[0, 1]`** (a score dict with `score` + `subscores` is also accepted; the headline `score` is authoritative).
- **Never raise for author/infra faults** — let them propagate so the runner discards the attempt (`env_internal_failure`). `raise AgentFault` only for agent-controlled failures (missing/malformed submission, wrong row count); those are kept as a clean 0.0.
- **Read agent artifacts only through the sanctioned loaders** —
  `grading.helpers` (`load_submission_or_fault` CSV,
  `load_submission_npz_or_fault` .npz/.npy,
  `load_submission_h5_or_fault` HDF5,
  `open_submission_file_or_fault` custom file-like parsers,
  `run_submitted_executable`, `load_submitted_model`),
  `grading.policy_eval` (`run_seeds` / `aggregate`),
  `grading.env_loading` (`load_env_module`), and `grading.kfold`
  (`score_kfold_cv`). Never read or execute agent artifacts by hand in the root
  grader. A bare `except OSError`, `lstat` + `S_ISREG`, or
  `os.open(..., O_NOFOLLOW)` is insufficient: the first two are check/use
  races, while `O_NOFOLLOW` protects only the leaf and still follows a swapped
  `/tmp/output` or nested parent directory. If no format loader fits, parse the
  immutable file object yielded by `open_submission_file_or_fault`.
- **Make prediction semantics declarative.** `CsvRows` and `PythonPredictor`
  accept per-column `value_domains` plus opt-in `OneHot([...])` and
  `Simplex([...])` constraints. Use `prediction_scope="row_independent"` when
  each output must depend only on its own row; the grader compares the full
  batch with shuffled fresh-worker partitions.
- **Declare the real inference budget.** `PythonPredictor` commits
  `predict_timeout_s`, `first_call_timeout_s`, `max_rows`, and
  `max_reply_bytes` into the task digest. State those exact limits in
  `instruction.md`; reference inference must retain headroom under them.
- **Use stable private selection.** `PrivateTableChallenge` evaluates the full
  bank by default and a sized challenge now defaults to `stable_subset`, so
  calibration and production score identical rows. Explicit
  `selection_policy="artifact_digest"` is retained only for legacy `3.0` lock
  migration.
- **Never preserve undeclared CSV fields just to be permissive.** Use
  `extra_columns="drop"` for safe tolerance, or `"reject"` (the default).
  `"preserve"` is an explicit expert-only escape hatch. For keyed static CSV
  comparisons, declare `CsvRows(..., join_key="id")`; it uses
  `join_submission_to_truth_or_fault`, which projects declared columns before
  merging.
- **Run ground truth locally only when you need calibration feedback.** The
  framework measures committed reference/naive strategies, writes an ignored
  development lock/evidence bundle, and replays no-op/reference/oracle
  contracts. Trusted CI independently generates or restores the production
  bundle and mounts it read-only in Taiga. Do not edit or commit generated
  calibration values.
- **Floors are reviewed semantic choices, not baseline measurements.** Every
  floor carries an `AnchorRationale`; baselines normally prove weak genuine
  work remains above floor and below reference. A baseline that genuinely ties
  the no-information floor needs an explicit `naive_at_floor` reviewed
  exception.
- **Acknowledge floor-semantic divergence.** Generated locks expose both
  `qualification_naive_score` and `runtime_naive_quality_score`. If their gap
  exceeds the configured threshold, calibration fails until
  `GeneratedCalibration(naive_semantic_gap_acknowledgement=...)` records a
  reviewed exception.
- **Use effective no-information floors for new classification-heavy tasks.**
  `GeneratedCalibration(quality_floor_mode="effective_no_info")` starts
  grade-time credit above the measured no-information ceiling. Existing
  author-floor locks remain supported until explicitly regenerated.
- **Metric names are not formulas.** Use exact versioned definitions such as
  `sre.rmse_over_population_std.v1`; documentation and the generated lock expose
  denominator, `ddof`, threshold, label, and averaging conventions.
- **Never call `TASK.score(metrics)` from production.** It is calibration-only.
  Legacy static tasks temporarily call `TASK.grade(submission, truth)`; follow
  the version-by-version migration guide in `CONTINUOUS_EVALUATION.md`.

The image bakes `scorer/` at root-only `/mcp_server/grader/`; both the no-arg and the `(workspace, trajectory, private)` signatures are handled by the grader runtime.

## 4. Reference + baselines (committed-strategy contract)

Trusted CI, ground-truth, and Taiga seal generation **must never train**. For
`solution/` and every calibration-declared baseline (`TASK.naive`,
normally `baselines/naive/`), authors commit an inference-only strategy and one
of these explicit contracts:

- **Trained model:** existing `model.manifest.json` v1 remains supported, or
  use `strategy.manifest.json` with `kind = "trained_model"`, digest-bound
  `training_inputs`, seed, training entrypoint, inference entrypoint, and
  trained artifact digests.
- **Hand-authored policy/static artifact:** use `strategy.manifest.json` with
  `kind = "committed_artifact"`, an inference entrypoint, and artifact digests.
  Do not invent a training script or dummy `train.csv`.

Rules:

- `solution.py` / `solve.sh` must **not** train (no `fit` / epoch loops that
  produce the scored artifact). Loading weights + predict/package only.
- Training may live in-repo for `trained_model`; validate and CI treat “train
  on the seal path” as an error.
- Optional: committed `submission.csv` is allowed as a **Tier-B** static
  artifact in a declared strategy — it does not replace its manifest.
- Generated score logs (`results.txt`) must not be committed.
- Hand-edited `calibration.lock.json` remains forbidden; Trusted CI generates
  or restores the production lock.

Example hand-authored policy manifest:

```json
{
  "schema_version": "1.0",
  "role": "reference",
  "kind": "committed_artifact",
  "inference_entrypoint": "solution.py",
  "artifacts": [
    {"path": "policy.py", "sha256": "<sha256>"}
  ]
}
```

A `trained_model` strategy additionally declares `training_entrypoint`, integer
`seed`, and non-empty `training_inputs: [{path, sha256}, ...]`. Input paths may
refer to task-root-relative public simulator/config/data files (for example
`data/train.csv`); artifact paths remain strategy-relative.

`solution/` must score `0.5 ± 0.05`. The declared naive baseline must
normally earn a small positive score below the reference.

Strict publication (`LBX_STRICT_RELEASE_GATES=1`) also requires at least three
named baselines and `baselines/portfolio.json`:

```json
{
  "schema_version": "baseline-portfolio.v1",
  "required_families": ["no_op", "domain_heuristic", "simple_fitted"],
  "baselines": [
    {
      "name": "naive",
      "family": "no_op",
      "path": "baselines/naive",
      "rationale": "A no-information constant prediction baseline."
    },
    {
      "name": "formula",
      "family": "domain_heuristic",
      "path": "baselines/formula",
      "rationale": "The obvious domain formula available from the prompt."
    },
    {
      "name": "linear",
      "family": "simple_fitted",
      "path": "baselines/linear",
      "rationale": "A simple fitted model over the raw public features."
    }
  ]
}
```

Declare every required family and add enough entries to reach the three-baseline
minimum. Grade the portfolio through the production path. The shared
`evaluate_baseline_portfolio`, `evaluate_score_panel`, and
`evaluate_regrade_stability` gates block an obvious baseline reaching the
reference band, score-panel ceiling saturation, or cross-nonce drift for an
unchanged artifact.

### Non-tabular no-information probes

Built-in constant/jitter/shuffle/row-index probes remain the default for
numeric tabular tasks. `ContinuousTask` policy, env/hybrid, executable, and
non-numeric callback tasks declare committed ready-to-measure workspaces.
`PolicyEvaluationTask` keeps its separate paired-control protocol:

```python
from grading.evaluation import (
    GeneratedCalibration,
    WorkspaceDegenerateProbes,
    WorkspaceProbe,
)

calibration = GeneratedCalibration(
    degenerate_probes=WorkspaceDegenerateProbes(
        probes=(
            WorkspaceProbe(
                name="no-op",
                path="baselines/degenerate/no-op",
                rationale="A policy that always emits the documented neutral action.",
            ),
            WorkspaceProbe(
                name="seeded-random",
                path="baselines/degenerate/seeded-random",
                rationale="A seeded action policy with no learned state or signal.",
            ),
        )
    )
)
```

Each directory is copied as the complete `/tmp/output` workspace and measured
twice under the same calibration seed. All files must be regular, bounded, and
digest-bound; every probe must return a complete deterministic metric vector.
Never put private truth in a probe or provide hand-written raw metrics.

`PolicyEvaluationTask` additionally declares `required_control_families`
(default: `no_op`, `constant`, `open_loop`). New and migrated tasks pass a
`control_families` mapping to `grade(...)`; missing declared families then fail
before candidate evaluation. Legacy calls without the mapping remain supported
as unclassified controls. Override the required tuple only when a family is
genuinely meaningless for the domain, and keep that choice reviewable in the
task spec.

Policy tasks also declare `call_timeout_s` and `total_timeout_s`. The first
limits one policy-worker RPC; the second charges cumulative time waiting for
submitted policy calls so thousands of individually compliant sleeps cannot
reach the outer grader timeout and discard the run. State both limits in
`instruction.md`, measure the reference under them, and keep the total at or
below 80% of the effective grading timeout.

If the honest naive exactly ties every effective no-information floor and no
weak-positive baseline exists, opt in explicitly with
`naive_score_min=0.0` plus `naive_at_floor=AnchorRationale(
kind="reviewed_exception", ...)`. This is not a default for bounded metrics.

Finalize in one command:

```bash
uv run lbx-rl-harness run \
  --runtime ground-truth \
  --problem-dir problems/<task_id>
```

That command validates manifests, runs **inference only**, measures anchors, and
writes the development lock/evidence bundle. It never shells out to `train.py`.
## 5. Licensing

`license` must be a permissive SPDX id (`MIT`, `Apache-2.0`, `BSD-2/3-Clause`, `ISC`, `Unlicense`, `CC0-1.0`, `CC-BY-4.0`, `PDDL-1.0`, `UPL-1.0`) or `self_generated`. Trace the dataset license **upstream**; copyleft / non-commercial / research-only data is rejected. `license_source` is the upstream http(s) URL where you confirmed it (or a justification for `self_generated`). Validation checks the allowlist mechanically; a licensing code-owner approves the PR.

## 6. Prompt guardrails

Name installed tools/libraries directly in `instruction.md`; do **not** tell the agent to inspect the Docker/base image, `task.toml`, or the requirements files to discover runtime packages, and do **not** mention the anchors or how `compute_score` is structured. GPU/TPU availability and dedicated-`tmux` guidance are appended automatically at export.

## 7. Base images

Every task builds `FROM` one of the flagship bases — `cpu`, `gpu`,
`gpu-openroad`, `gpu-blackwell`, `cuda-graphics`, `tpu` (`base/<flavor>/`). They
all run `base/install-common.sh`, so they share the `/opt/lbx-runtime/.venv`
runtime, the uid-1000 `agent` account, the baked grader runtime, and the 0700
`/mcp_server` boundary. The task `Dockerfile` adds only the task's own layers:
its dependency channels, its data, its scorer, and its calibration lock.

`cpu`, `gpu`, `gpu-openroad`, `gpu-blackwell` and `cuda-graphics` are all Python
3.13; only `tpu` pins Python 3.12 (the JAX TPU stack publishes no 3.13 wheels).
The three CUDA flavors share one CUDA and one torch: CUDA 13.0.3 on Ubuntu 24.04
with torch 2.9.1+cu130. Kaolin and Open3D are not available on `cuda-graphics` —
neither publishes wheels for that matrix. Each flavor carries its own
drift-hashed tag prefix, so its tag names the flavor it was built from.

Bases are rebuilt and pushed via `base/build_and_push.sh --flavors <list>`.

## 8. Hugging Face resources (offline weights / datasets)

There is **no internet at runtime**, so a task that needs pretrained weights or a HuggingFace dataset declares them as `[[preloaded_files]]` entries with an `hf_repo`. Each repo is fetched once at deploy time, packed into a content-addressed squashfs, and mounted read-only into the agent's HF hub cache, so `from_pretrained("org/name")` (or `load_dataset(...)`) resolves entirely from the mount.

```toml
[[preloaded_files]]
hf_repo = "sentence-transformers/all-MiniLM-L6-v2"

[[preloaded_files]]
hf_repo = "org/name"
repo_type = "dataset"
hf_revision = "v2"
allow_patterns = ["*.json"]
ignore_patterns = ["*.bin"]
```

`hf_repo` is required; `hf_revision` defaults to `main` (a tag, branch, or 40-hex commit sha), `repo_type` is `model` or `dataset`, and `allow_patterns` / `ignore_patterns` narrow what is downloaded. Leave `mount_path` unset: it is derived as `/tmp/hf-cache/hub/<repo_type>s--<org>--<name>`, matching HF's own cache scheme, and the bases set `HF_HOME=/tmp/hf-cache`.

Mechanics: `scripts/sync_mount.sh` resolves each repo's immutable commit sha (a cheap metadata call, no weights), so the remote object is content-addressed by `repo@sha` under a shared `cache/huggingface/` prefix and downloaded/packed **once** across all tasks that reference it (`scripts/pack_hf_resource.py`). `hf_repo` mounts are **not** available on the TPU base; validation rejects the combination.

**Token (gated repos only).** Ungated permissive repos (the common case, and the only ones that pass licensing) need no token. A **gated** repo needs `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) in the environment where `sync_mount.sh` runs — never at task runtime. Two sources:

- **CI / deploy:** the grade workflow injects it from a repo secret — `HF_TOKEN: ${{ secrets.HF_TOKEN }}`.
- **Local:** `export HF_TOKEN=hf_...` before running `sync_mount.sh`.

`pack_hf_resource.py` reads it from the env, so both paths work with no code change.

## Public payloads and data mounts

For every ml task, `sync_mount.sh` exposes `data/ -> /data` in
Taiga's payload inventory. Public trees up to 256 files and 1 GiB are uploaded
as explicit regular `preloaded_files`; when a tree is larger, regular files
named verbatim in `instruction.md` are still exposed as payload-only copies under
`/lbx-public-files`, while the complete canonical tree remains at `/data`. This
lets Taiga Data Quality resolve logical filenames such as `column_mapping.json`
instead of seeing only an opaque image or squashfs. Root-side setup re-owns
both public locations and changes them to `0444`/`0555`, preserving the
immutable public-data contract.

When no hidden env server is configured, a public tree too large for bounded
expansion falls back to a content-addressed read-only squashfs, and `scorer/data`
is always mounted read-only behind the baked `0700` `/mcp_server` boundary.
Hidden-env tasks retain their baked data so environment startup
semantics do not change; their explicit public entries are identical
QA-visible overlays. The baked copy also keeps local harness runs working.

## References

- [`examples/mle-tabular-classification/`](../examples/mle-tabular-classification/) — canonical continuous `dataset` task.
- [`examples/hidden-env-bandit/`](../examples/hidden-env-bandit/) — canonical `env` task.
- [`REWARD_HACKING.md`](REWARD_HACKING.md) — mitigations every scorer must respect.
- [`GRADING.md`](GRADING.md) — full grader contract + calibration.
- [`HIDDEN_ENV.md`](HIDDEN_ENV.md) — `env` / `hybrid` tasks.
- [`LEGACY_ML_LAYOUT.md`](LEGACY_ML_LAYOUT.md) — converting a removed-layout task.
