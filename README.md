# LBX RL Tasks Template

Author workspace for Alignerr RL tasks. Clone this repo (or your assigned
fork), scaffold under `problems/`, validate locally, and open a fork PR.
Trusted CI and post-merge integration live in
[`lbx-rl-tasks-iso-mothership`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-mothership);
you do not need mothership access to author tasks.

Before upgrading or regenerating a task, review the
[`Task Author Changelog`](docs/CHANGELOG.md) for migrations, schema changes,
and required image or calibration rebuilds.

## Quick start

```bash
git clone https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template.git
cd lbx-rl-tasks-iso-template
uv sync
uv run lbx-rl-harness --help
uv run lbx-rl-template --help
```

`uv sync` installs the shared `grading` package, Harbor `run-grader`, local
harness, validators/exporters, and the `lbx-rl-template` CLI. Prefer
`uv run ...` from the repo root.

Set `domain` in `.labelbox/problem.json` before you open a PR (Labelbox contract;
leave the file in place).

Scaffold a task (preferred):

```bash
uv run lbx-rl-template create \
  --name labelbox/my-task \
  --template ml \
  --out problems
```

Starters: `ml`, `mujoco`, `cfd`, `structures`, `software-engineering`,
`prometheus-cfd`, `prometheus-structures`, `prometheus-eval-cfd`,
`prometheus-eval-structures`. Manual `cp -R` from
`alignerr_plugin/.../starter_templates/` still works; see
[`docs/AUTHORING.md`](docs/AUTHORING.md).

Update `metadata.json` and `task.toml` so ids match the directory:

```json
{
  "benchmark": "taiga_task",
  "problem_data": {
    "instance_id": "my-task",
    "description": "Your concise task description"
  }
}
```

```toml
[task]
name = "labelbox/my-task"
description = "Your concise task description"

[difficulty]
task_type = "ml"  # ml | mujoco | cfd | structures | software_engineering
domain = "scientific_discovery_computational_science"
reward_type = "continuous_scoring_function"  # or multi_deterministic_rubrics
```

## Required vs conditional files

**Always required** (validator schema stage):

| File | Notes |
| --- | --- |
| `metadata.json` | Required. `benchmark` must be `"taiga_task"`; `problem_data.instance_id` must match the task id. |
| `task.toml` | Config, resources, `[difficulty]`, and at least one `[[outputs]]`. |
| `instruction.md` | Agent-facing prompt. |
| `scorer/compute_score.py` | Grader entrypoint (`compute_score` and/or `TASK = RubricTask(...)`). |

**Effectively required for submitted tasks:**

| File / artifact | When |
| --- | --- |
| `environment/Dockerfile` | Local/CI image build and build-proof stages. |
| Oracle under `solution/` | Ground-truth proof (`solve.sh`, or ML committed-strategy layout). |

**Conditional:**

| File / artifact | When |
| --- | --- |
| `solution/render.sh` + `[ground_truth].render_*` | MuJoCo, or any task that declares `render_outputs`. |
| `scorer/evaluation.plan.json` | Rubric / `RubricTask` tasks (sealed; do not hand-edit). |
| ML manifests / committed artifacts | Continuous `ml` tasks — see [`docs/ML_TASKS.md`](docs/ML_TASKS.md). |
| `scorer/data/env.py` + `data/env_client.py` | `[environment].hidden_env` tasks. |
| `baselines/`, `data/`, task `README.md` | Recommended; not schema-required. Layout varies by starter. |
| `.alignerr/build_proof.json` | Local feedback only under `problems/` (gitignored). Trusted CI regenerates the authoritative proof. |
| `.alignerr/ground_truth/` | Commit reviewer media when render is expected. |

Typical layout (not every path is always present):

```text
problems/<task_id>/
├── metadata.json          # required
├── task.toml              # required
├── instruction.md         # required
├── environment/Dockerfile # required for submit
├── scorer/compute_score.py
├── data/                  # optional public inputs
├── solution/              # oracle / render scripts as needed
├── baselines/             # optional
└── README.md              # optional
```

## Author loop

1. Edit the task under `problems/<task_id>/`.
2. Iterate on the oracle: `uv run lbx-rl-harness reference --problem-dir problems/<task_id>`.
3. Optional preflight: `uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>`.
4. Pre-submit check: `uv run lbx-rl-template check --problem-dir problems/<task_id>` (or `validate` for full JSON).
5. Optional local agent / Auto QA (see below).
6. Commit task source and, when applicable, `.alignerr/ground_truth/` media. **Do not** commit `build_proof.json` or generated calibration locks under `problems/`.
7. Open a PR in **your assigned fork** (not this template repo). Keep one task per PR.

## Artifacts policy

Local `--runtime ground-truth` writes development evidence under
`problems/<task_id>/.alignerr/`. That path is gitignored except
`.alignerr/ground_truth/` reviewer media.

- **Do not commit** `build_proof.json` or generated calibration lock/evidence for
  authored tasks under `problems/`.
- **Do commit** reviewed media under `.alignerr/ground_truth/` when render is
  expected (typically MuJoCo).
- Trusted CI checks out the immutable PR revision, runs ground-truth, then
  verifies the CI-generated proof before grading and Taiga submission.

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
git add problems/<task_id>
# If render artifacts exist:
git add problems/<task_id>/.alignerr/ground_truth/
```

Details: [`docs/GROUND_TRUTH.md`](docs/GROUND_TRUTH.md).

## What CI runs where

| Surface | What runs |
| --- | --- |
| This template repo | Shared package tests (`python-ci.yml`) and example contract checks. No full task grade. |
| Your fork PR | Labelbox relays to mothership trusted CI (`trusted-ci/grade`): ground-truth, proof verify, validate, harness/rubric QA, Auto QA, advisory Taiga. |

Gate map for maintainers:
[mothership `docs/QA_PIPELINE.md`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-mothership/blob/main/docs/QA_PIPELINE.md).

## Repo layout

```text
lbx-rl-tasks-iso-template/
├── grader/                 # shared grading package + Harbor runner
├── alignerr_plugin/        # validators, exporters, starter_templates/
├── harness/                # local Boreal-like agent runner
├── examples/               # reference tasks (patterns only)
├── problems/               # put authored tasks here
├── docs/                   # authoring / grading / domain guides
├── project_guidelines/     # long-form domain handbooks
├── base/                   # local base-image build context
└── taiga_runtime/          # rubric runtime used by local images
```

On first local run the harness builds a repo-local base from `base/`,
`taiga_runtime/`, and `grader/` (no Artifact Registry needed). CPU:
`lbx-tasks-base:runtime-ml-core-py313-local`. GPU:
`lbx-tasks-base-gpu:runtime-ml-core-py313-local`.

## Examples

Use `examples/` as patterns; submit new work under `problems/`.

| Example | `task_type` | `reward_type` |
| --- | --- | --- |
| [`mujoco-pendulum`](examples/mujoco-pendulum/) | `mujoco` | `multi_deterministic_rubrics` |
| [`mle-tabular-classification`](examples/mle-tabular-classification/) | `ml` | `continuous_scoring_function` |
| [`hidden-env-bandit`](examples/hidden-env-bandit/) | `ml` | `continuous_scoring_function` |
| [`openfoam-hydrofoil-flap`](examples/openfoam-hydrofoil-flap/) | `cfd` | `multi_deterministic_rubrics` |
| [`opensees-base-isolation`](examples/opensees-base-isolation/) | `structures` | `multi_deterministic_rubrics` |
| [`wal-recovery-ordering`](examples/wal-recovery-ordering/) | `software_engineering` | `multi_deterministic_rubrics` |
| [`xfoil-rust-port`](examples/xfoil-rust-port/) | `software_engineering` | `multi_deterministic_rubrics` |
| [`frontier-service-cutover`](examples/frontier-service-cutover/) | `software_engineering` | `multi_deterministic_rubrics` |
| [`frontier-mcp-workspace`](examples/frontier-mcp-workspace/) | `software_engineering` | `multi_deterministic_rubrics` |

See [`examples/README.md`](examples/README.md) for expected scores and how to run them.

## Task metadata and delivery

Every task declares enum-backed `[difficulty]` metadata (`task_type`, `domain`,
`reward_type`). Master enums live in `alignerr_plugin.task_metadata`.

- Continuous references should score about `0.5 ± 0.05`.
- Deterministic rubric oracles should score `1.0`.

Omit `[delivery]` for default Taiga. Set `platform = "prometheus"` only for
Prometheus CFD/structures routes (non-eval vs eval starters differ by
`[delivery].eval`). Software-engineering acceptance still requires trusted CI
green and Boreal aggregate `<= 0.4`. Non-eval CFD/structures may submit for
review through **either** score lane — Prometheus (CI + mean `<= 0.6` + stddev
`>= 0.08`) or Achilles (CI + Boreal mean `<= 0.4`). Both lanes additionally
require completed Boreal QA with no undocumented criticals; see
[`docs/CFD_STRUCTURES_DUAL_LANE_REVIEW.md`](docs/CFD_STRUCTURES_DUAL_LANE_REVIEW.md)
and [`docs/AUTHORING.md`](docs/AUTHORING.md).

**Prefer GPU.** `ml` defaults to H100 (`12vcpu+100gib+h100/2`). Use a CPU enum
only when acceleration is impossible. Keep Dockerfiles generic
(`ARG BASE_IMAGE` / `ARG BASE_TAG`) so harness/CI can inject the correct base.
Resource enums and base flavors: [`docs/AUTHORING.md`](docs/AUTHORING.md).

ML tasks must keep `allow_internet = false` and declare
`[difficulty].license` (prefer `self_generated` for synthetic data, or a
permissive SPDX id with `license_source`). `not_applicable` remains a
back-compat alias for `self_generated`.

## Output directory and grading

Ask the agent to write final artifacts under `/tmp/output`. The grader receives
that path as `workspace`.

- Continuous / legacy: implement `compute_score(workspace, trajectory, private)`.
- Rubric tasks: declare `TASK = RubricTask(...)` (no author-owned score plumbing);
  commit sealed `scorer/evaluation.plan.json`.

Details: [`docs/GRADING.md`](docs/GRADING.md),
[`docs/RUBRIC_EVALUATION.md`](docs/RUBRIC_EVALUATION.md),
[`docs/RUBRIC_GUIDANCE.md`](docs/RUBRIC_GUIDANCE.md). Criteria must be
deterministic Python — no LLM judges.

## Local agent and Auto QA (optional)

Local agent runs default to `claude-code` (Claude CLI + `claude /login`). Without
that subscription, use `--runtime deepagents` and `.env.local` from
`.env.example`.

```bash
uv run lbx-rl-harness run --problem-dir problems/<task_id>
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime deepagents
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime rubric-quality
uv run lbx-rl-harness autoqa --problem-dir problems/<task_id>
```

## Open a fork PR

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
uv run lbx-rl-template check --problem-dir problems/<task_id>
git add problems/<task_id>
git add problems/<task_id>/.alignerr/ground_truth/   # only if render media exists
git status --short   # confirm no build_proof.json / secrets
git commit -m "Add problems/<task_id>"
git push
```

Do not commit `.env.local`, `.harness-runs/`, provider keys, or credentials.
Push fixes to the same fork PR; trusted CI reruns on each sync. Feedback appears
on the fork PR (`trusted-ci/grade`, harness/rubric/Auto QA, advisory Taiga).

## Command reference

### `lbx-rl-harness`

| Command | Purpose |
| --- | --- |
| `reference` | Day-to-day solve/grade loop for the oracle. |
| `run` | Full local run; use `--runtime ground-truth`, `rubric-quality`, `claude-code`, or `deepagents`. |
| `verify-ground-truth` | Convenience wrapper for the ground-truth runtime. |
| `autoqa` | Local Auto QA from an existing proof. |
| `run-taiga` | Local Boreal metadata / job payload. |
| `run-harbor` | Local Harbor export directory. |
| `calibration-key` | Print continuous-ML calibration cache key. |

### `lbx-rl-template`

| Command | Purpose |
| --- | --- |
| `create` / `new` | Scaffold from a starter template. |
| `check` | Human-friendly pre-submit preflight. |
| `validate` | Full validator JSON (CI-shaped). |
| `export-taiga` | Boreal/Taiga metadata export. |
| `export-harbor` | Harbor directory export. |
| `export-capsule` | Outer Taiga capsule bundle for multi-service tasks. |
| `lint-reward-hacks` | Advisory reward-hacking lint. |
| `migrate-legacy-mujoco` | Legacy MuJoCo path migration. |

Also: `run-grader` (Harbor in-container grader) after `uv sync`.

## Testing shared package changes

```bash
uv sync --all-packages
uv run --no-sync pytest -ra grader/tests harness/tests
```

Root discovery excludes `problems/` and `examples/`. Task validation is via
harness + `lbx-rl-template` + trusted CI.

## Docs index

| Audience | Start here |
| --- | --- |
| Shared framework changes | [`docs/CHANGELOG.md`](docs/CHANGELOG.md) |
| First task | [`docs/AUTHORING.md`](docs/AUTHORING.md), [`problems/README.md`](problems/README.md) |
| Grading / rubrics | [`docs/GRADING.md`](docs/GRADING.md), [`docs/RUBRIC_EVALUATION.md`](docs/RUBRIC_EVALUATION.md), [`docs/RUBRIC_GUIDANCE.md`](docs/RUBRIC_GUIDANCE.md) |
| Ground truth / proof | [`docs/GROUND_TRUTH.md`](docs/GROUND_TRUTH.md) |
| Continuous ML | [`docs/ML_TASKS.md`](docs/ML_TASKS.md), [`docs/CONTINUOUS_EVALUATION.md`](docs/CONTINUOUS_EVALUATION.md) |
| Hidden env | [`docs/HIDDEN_ENV.md`](docs/HIDDEN_ENV.md) |
| Software engineering | [`docs/SOFTWARE_ENGINEERING_FRAMEWORK.md`](docs/SOFTWARE_ENGINEERING_FRAMEWORK.md), [`project_guidelines/software_engineering/`](project_guidelines/software_engineering/) |
| CFD / structures | [`docs/NUMERICAL_SOLVERS.md`](docs/NUMERICAL_SOLVERS.md), [`project_guidelines/cfd/`](project_guidelines/cfd/), [`project_guidelines/strctural_engineering/`](project_guidelines/strctural_engineering/) |
| CFD / structures submit lanes | [`docs/CFD_STRUCTURES_DUAL_LANE_REVIEW.md`](docs/CFD_STRUCTURES_DUAL_LANE_REVIEW.md) |
| Migration | [`docs/TASK_MIGRATION.md`](docs/TASK_MIGRATION.md) |
| Reward hacking | [`docs/REWARD_HACKING.md`](docs/REWARD_HACKING.md), [`docs/POLICY_ISOLATION.md`](docs/POLICY_ISOLATION.md) |
| Agent rules | [`AGENTS.md`](AGENTS.md) |

Historical snapshot (not current policy):
[`docs/QA_TRIAGE_2026-07-24.md`](docs/QA_TRIAGE_2026-07-24.md).
