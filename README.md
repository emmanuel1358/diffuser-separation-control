# LBX RL Tasks Template

This repository is the starting point for authoring Alignerr RL tasks.
It includes the shared grading package, local authoring utilities, starter
templates, the local Boreal-like harness, and complete example tasks you can
copy from while building your own task under `problems/`.

You do not need mothership repo access to use this repo. The normal workflow is:

1. Install the local workspace with `uv sync`.
2. Read the examples in `examples/` (MuJoCo, tabular ML, CFD/OpenFOAM, and
   structural/OpenSees).
3. Set `domain` in `.labelbox/problem.json` — Labelbox reads this when your PR
   opens and shows it in the Tasks table. (Path + field are a contract with
   Labelbox; leave the file in place.)
4. Create your own task in `problems/<task_id>/`.
5. Implement `scorer/compute_score.py`.
6. Run the ground-truth verifier before creating a PR.
7. Commit `problems/<task_id>/.alignerr/build_proof.json` and any generated
   `problems/<task_id>/.alignerr/ground_truth/` reviewer artifacts.
8. Open a GitHub PR in **your assigned fork** (not this template repo).
   Labelbox relays the PR to trusted-side CI, which runs the agent harness,
   rubric QA, and Auto QA and posts a `trusted-ci/grade` check back on your PR.
9. Read feedback on your PR, fix the same task, and push updates. Your reviewer
   merges the PR once the check passes; submission happens after acceptance.

## What This Repo Contains

```text
lbx-rl-tasks-template/
├── pyproject.toml
├── grader/
│   ├── src/grading/           # Grade, deterministic RubricBuilder, helpers
│   ├── src/grader_runner/     # run-grader console script
│   └── tests/                 # shared grader tests
├── alignerr_plugin/
│   └── src/alignerr_plugin/
│       ├── commands.py        # internal authoring/export helpers
│       ├── validators/        # task validation helpers used by CI
│       └── starter_templates/ # ml, mujoco, cfd, structures, Prometheus scaffolds
├── harness/                   # local Boreal-like agent runner and docs
├── examples/
│   ├── mle-tabular-classification/      # continuous ML scoring example
│   ├── mujoco-pendulum/                 # MuJoCo robotics reference task
│   ├── openfoam-hydrofoil-flap/          # CFD/OpenFOAM numerical-solver example
│   └── opensees-base-isolation/          # structural/OpenSees numerical-solver example
├── problems/                  # put your authored tasks here
├── docs/
│   ├── AUTHORING.md           # task creation walkthrough
│   ├── GRADING.md             # compute_score and grader package guide
│   ├── GROUND_TRUTH.md        # oracle solution and reviewer video requirements
│   ├── HIDDEN_ENV.md          # simulation/interaction tasks via the hidden-env RPC server
│   ├── MLENVS_TASKS.md        # authoring ML_Envs-style (dataset/model/exec/hdf5/kfold) tasks
│   ├── REWARD_HACKING.md      # reward-hacking mitigations every scorer must respect
│   ├── NUMERICAL_SOLVERS.md   # installed solver inventory + invocation guide
│   └── RUBRIC_GUIDANCE.md     # rubric design principles and examples
└── project_guidelines/
    ├── cfd/cfd_environments.md
    ├── mujoco_environments.md
    └── strctural_engineering/STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md
```

The `grader/` package is installed into task Docker images so your
task-local `scorer/compute_score.py` can import `grading`. The `harness/`
package lets you run a local Boreal-like LLM attempt before opening a PR.
Template PR validation only requires the deterministic ground-truth proof in
`problems/<task_id>/.alignerr/build_proof.json` plus any renderer artifacts
under `.alignerr/ground_truth/`. The expensive agent harness, rubric QA, and Auto QA
run in trusted-side CI on your fork PR and post a `trusted-ci/grade` check.

## First-Time Setup

Install `uv` if you do not already have it, then install the workspace:

```bash
git clone https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template.git
cd lbx-rl-tasks-iso-template
uv sync
uv run lbx-rl-harness --help
```

`uv sync` installs:

- `lbx-rl-tasks-grading`, the shared `grading` package from `grader/`.
- `run-grader`, the Harbor-compatible grader runner.
- `lbx-rl-tasks-harness`, the local Boreal-like runner.
- `lbx-rl-tasks-alignerr-plugin`, internal task validation/export helpers used by CI.
- `lbx-rl-template`, a self-contained CLI for template validation/export helpers.

You should normally run commands through `uv run ...` from the repo root.

For task submission, run the deterministic ground-truth verifier:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```

While developing the reference solution (especially ML training), use the
lighter reference loop — it does not refresh build proof:

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness reference --problem-dir problems/<task_id> --skip-solve
```

That command records `ground_truth_result` in
`problems/<task_id>/.alignerr/build_proof.json`. For MuJoCo tasks it also writes
reviewer render artifacts under `.alignerr/ground_truth/`.

Optional local modes are available when you want earlier feedback before
spending CI/model budget:

```bash
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime ground-truth
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime rubric-quality
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime claude-code
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime deepagents
uv run lbx-rl-harness run --problem-dir problems/<task_id>
```

Local agent runs default to `claude-code`. Use this path if you have Claude Code
available through your subscription: install the `claude` CLI, run
`claude /login`, and the harness will use that local OAuth session without
needing provider API keys. If you do not have a Claude Code subscription/login,
use `--runtime deepagents` instead and configure provider keys in `.env.local`
as shown in `.env.example`.

Local agent harness runs are self-contained. On first use, the harness builds
the required repo-local base image from this repo's `base/`, `taiga_runtime/`,
and `grader/` directories, then builds your task image from that local base.
CPU tasks use `lbx-tasks-base:runtime-ml-core-py313-local`; GPU tasks use
`lbx-tasks-base-gpu:runtime-ml-core-py313-local`. This first Docker build can
take a while because it downloads Python and ML dependencies, but it does not
require Google Artifact Registry credentials.

## The Beginner Workflow

Start by reading the reference tasks. Each demonstrates a different domain and
scoring shape:

- [`examples/mujoco-pendulum/`](examples/mujoco-pendulum/) — MuJoCo robotics,
  `task_type = "mujoco"`, `domain = "model_environment_construction"`,
  `reward_type = "multi_deterministic_rubrics"`, required rollout video.
- [`examples/mle-tabular-classification/`](examples/mle-tabular-classification/) —
  tabular ML, `task_type = "ml"`,
  `domain = "scientific_discovery_computational_science"`,
  `reward_type = "continuous_scoring_function"`.
- [`examples/openfoam-hydrofoil-flap/`](examples/openfoam-hydrofoil-flap/) —
  CFD/OpenFOAM, `task_type = "cfd"`, `domain = "aerodynamics"`,
  deterministic engineering rubric with solver-agnostic instructions and
  solver-backed oracle/scorer code.
- [`examples/opensees-base-isolation/`](examples/opensees-base-isolation/) —
  structural/OpenSeesPy, `task_type = "structures"`,
  `domain = "structural_mechanics"`, deterministic engineering score with
  solver-backed oracle/grader scoring.

These tasks show the expected files, Dockerfile copy paths, `task.toml` shape,
`/tmp/output` convention, oracle scripts, and grader patterns.

Create a new task under `problems/` by copying the starter that matches your
`task_type` (`ml`, `mujoco`, `cfd`, or `structures`). Use the `prometheus-*`
starters when the task should keep its task type and trusted-CI gates but route
passing delivery to Prometheus and an independent Taiga mirror. Prometheus
supports both non-eval and eval submissions for CFD and structures. They use
the same CI and dual-delivery route; the only starter-level difference is
`[delivery].eval = false`
for non-eval starters and `[delivery].eval = true` for eval starters.

```bash
mkdir -p problems
# ml (continuous scoring); also: mujoco | cfd | structures
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/ml problems/my-task
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/mujoco problems/reacher-control
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/cfd problems/my-cfd-case
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/structures problems/my-frame
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/prometheus-cfd problems/my-prometheus-cfd-case
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/prometheus-structures problems/my-prometheus-frame
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/prometheus-eval-cfd problems/my-prometheus-eval-cfd-case
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/prometheus-eval-structures problems/my-prometheus-eval-frame
```

Use the matching project guide:

- Non-eval CFD: [`project_guidelines/cfd/prometheus_cfd_environments.md`](project_guidelines/cfd/prometheus_cfd_environments.md)
- Eval CFD: [`project_guidelines/cfd/prometheus_eval_cfd_environments.md`](project_guidelines/cfd/prometheus_eval_cfd_environments.md)
- Non-eval structures: [`project_guidelines/strctural_engineering/PROMETHEUS_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md`](project_guidelines/strctural_engineering/PROMETHEUS_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md)
- Eval structures: [`project_guidelines/strctural_engineering/PROMETHEUS_EVAL_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md`](project_guidelines/strctural_engineering/PROMETHEUS_EVAL_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md)

Each starter is intentionally minimal; copy task-specific patterns from the
matching `examples/` reference (not the example directory itself).

Then update `metadata.json` and `task.toml` so the task id and name match the
directory:

```json
{
  "benchmark": "taiga_task",
  "problem_data": {
    "instance_id": "reacher-control",
    "description": "Your concise task description"
  }
}
```

```toml
[task]
name = "labelbox/reacher-control"
description = "Your concise task description"
```

Then edit the generated files, especially:

- `instruction.md`: the prompt the agent sees.
- `task.toml`: resources, outputs, timeouts, and metadata.
- `environment/Dockerfile`: how the task image copies public data,
  private scorer data, and the grader into the container.
- `scorer/compute_score.py`: the Python grader.
- `data/`: public files available to the agent at `/data/`.
- `scorer/data/`: private files available only to the grader.
- `solution/solve.sh`: required ground-truth solution used by validation.
- `solution/render.sh`: required reviewer video generation for `mujoco` ground-truth solutions.
- `baselines/naive.sh`: optional weak baseline.

### Task metadata

Every task must declare enum-backed `[difficulty]` metadata. The master enum
values live in `alignerr_plugin.task_metadata`.

```toml
[difficulty]
task_type = "mujoco" # ml | mujoco | cfd | structures
domain = "model_environment_construction"  # enum scoped by task_type
reward_type = "multi_deterministic_rubrics" # or continuous_scoring_function
```

`task_type` is the broad category. `domain` is the finer diversity label used by
the dashboard/Supabase index. `reward_type` controls the ground-truth score
target: continuous scoring-function references should score `0.5 ± 0.05`, while
deterministic rubric oracles should score `1.0`.

### Delivery platform

Omit `[delivery]` for the default Taiga path. Set it only when the task should
use a non-default post-CI route:

```toml
[delivery]
platform = "prometheus"
```

For `cfd` and `structures`, Prometheus delivery does not change the task type or
the numerical-solver quality gates. Trusted CI still runs the normal CFD or
structures validation, grader QA, oracle validation, local agent score gates,
and Auto QA; only the final delivery job changes from Taiga submission to the
Prometheus Agent Service runner. Numerical-solver Prometheus submissions also
require the Prometheus target average score to be present and `<= 0.5`, with
target reward standard deviation `>= 0.1` across the required attempts, and a
passing trajectory trainability auditor score. The full `submit-prometheus`
workflow must pass before an eval row is accepted or a non-eval row moves into
review. Submitting a non-passing non-eval row for review violates fair practices
and may remove the tasker from the project.

## Requesting GPUs

**Prefer GPU.** We want GPU-accelerated tasks: `ml` tasks default to GPU (the
`ml` starter ships `required_resources = "12vcpu+100gib+h100/2"`), and any task
that can use acceleration should request an H100 resource enum. Pick a CPU enum
only when a task genuinely cannot use acceleration.

Pick one Taiga-supported resource enum in `task.toml` under `[environment]`:

```toml
[environment]
required_resources = "12vcpu+100gib+h100/2"
storage_mb = 50000
allow_internet = true
```

Available resource enums:

```text
CPU: 2vcpu+6gib, 4vcpu+16gib, 6vcpu+32gib, 8vcpu+64gib, 16vcpu+64gib, 16vcpu+128gib
CPU perf: 2vcpu+6gib+perf, 4vcpu+16gib+perf, 6vcpu+32gib+perf, 8vcpu+64gib+perf, 16vcpu+64gib+perf, 16vcpu+128gib+perf
TPU: 13vcpu+32gib+tpuv5e1x1, 16vcpu+64gib+tpuv5e2x2, 50vcpu+128gib+tpuv5e2x2
H100: 3vcpu+25gib+h100/8, 6vcpu+50gib+h100/4, 12vcpu+100gib+h100/2, 24vcpu+200gib+h100/1
H100 graphics: 3vcpu+25gib+h100/8+graphics, 6vcpu+50gib+h100/4+graphics, 12vcpu+100gib+h100/2+graphics, 24vcpu+200gib+h100/1+graphics
```

The default GPU tier is `12vcpu+100gib+h100/2` (2x H100), shipped by the `ml`
starter. When `required_resources` is an H100 tier, the local harness
automatically builds and uses the repo-local clean ML GPU base image,
`lbx-tasks-base-gpu:runtime-ml-core-py313-local`. Graphics H100 tiers ending in
`+graphics` select `cuda-graphics`; TPU tiers select the TPU base. Use
`base_flavor = "gpu-openroad"` only for OpenROAD/EDA tasks with a non-graphics
H100 enum.

Keep `environment/Dockerfile` generic so the harness and mothership can inject
the correct CPU or GPU base image:

```dockerfile
ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=problems/<task_id>
FROM ${BASE_IMAGE}:${BASE_TAG}
```

Local GPU image builds do not require Artifact Registry access, but running a
GPU-dependent task locally requires a Docker host with NVIDIA GPU support. The
local harness adds a runtime note for GPU/TPU tasks: the agent should make one
quick accelerator availability probe, then fall back to CPU-compatible work
instead of spending time debugging CUDA/TPU drivers. It must still write every
required artifact under `/tmp/output`.

Trusted CI routes GPU-configured tasks to the org `gpu` runner. TPU-configured
tasks stay on the default CPU runner, so they should also have a quick CPU
fallback path for the local/GitHub harness. Taiga submission remains accelerator
backed: exported jobs keep the structured GPU/TPU runtime notice and request the
configured accelerator resources.

## Required Task Layout

Each authored task should live in one directory:

```text
problems/<task_id>/
├── task.toml
├── metadata.json
├── instruction.md
├── environment/
│   └── Dockerfile
├── scorer/
│   ├── compute_score.py
│   └── data/
├── data/
├── solution/
│   └── solve.sh
├── baselines/
│   └── naive.sh
└── README.md
```

The validator requires `metadata.json`, `task.toml`, `instruction.md`,
and `scorer/compute_score.py`. It also expects `task.toml` to declare at
least one `[[outputs]]` entry and a `[ground_truth]` render command/video.

## Output Directory

Always ask the agent to write final artifacts under `/tmp/output`.

Good output paths:

```text
/tmp/output/model.xml
/tmp/output/policy.py
/tmp/output/submission.csv
/tmp/output/answer.md
```

Avoid `/workspace` for final answers, submissions, policies, MJCF files,
or other graded artifacts. The grader receives `/tmp/output` as its
`workspace` argument.

## Writing `compute_score.py`

Every task has exactly one grader entrypoint:

```python
from pathlib import Path

def compute_score(workspace: Path, trajectory, private: Path):
    ...
```

The arguments are:

- `workspace`: the agent output directory, usually `/tmp/output`.
- `trajectory`: reserved for future trajectory data; it is usually `None`
  today.
- `private`: the hidden grader data directory, copied from `scorer/data/`.

`compute_score()` may return one of three deterministic shapes:

- A `float` in `[0, 1]` for a single continuous score.
- A `dict` with at least `score`, plus optional `subscores`, `weights`,
  and `metadata`.
- `RubricBuilder.grade().to_dict()` for weighted deterministic criteria
  and penalties.

For continuous reward functions, follow the ML_Envs calibration pattern in
[`docs/GRADING.md`](docs/GRADING.md#continuous-reward-functions-from-ml_envs):
anchor naive baselines near `0`, the reference solution near `0.5`, and
perfect performance at `1`. Use a score dict when you want the calibrated
headline reward plus diagnostic metric progress values.

Most new tasks should use `RubricBuilder` because it gives reviewers,
Boreal, and Harbor per-criterion detail:

```python
from pathlib import Path
from grading import RubricBuilder, helpers

def compute_score(workspace: Path, trajectory, private: Path):
    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)

    @rb.criterion(
        id="answer_present",
        weight=1.0,
        description="answer.txt exists and is non-empty",
    )
    def _():
        return helpers.file_exists(workspace / "answer.txt", non_empty=True)

    @rb.penalty(
        id="forbidden_file",
        value=-0.5,
        description="Agent wrote a forbidden file",
    )
    def _():
        return (workspace / "forbidden.txt").exists()

    return rb.grade().to_dict()
```

Read [`docs/GRADING.md`](docs/GRADING.md) for the full grading contract,
helper API, return shapes, Harbor outputs, and common patterns. Read
[`docs/RUBRIC_GUIDANCE.md`](docs/RUBRIC_GUIDANCE.md) before writing
`RubricBuilder` criteria so the rubric stays task-specific, measurable, and
not overly prescriptive.

Important: rubric criteria must be deterministic Python checks. Do not use
LLMs as judges in `compute_score.py`; model calls make rewards non-reproducible
and are not allowed for submitted task rubrics.

## Using The Example Tasks

`examples/mujoco-pendulum/` is the canonical rubric-style reference. It
demonstrates:

- A complete `task.toml` with `/tmp/output/model.xml` as the required
  output.
- A Dockerfile that installs the shared `grader/` package and copies the
  task scorer into `/mcp_server/grader/`.
- Public `data/` versus private `scorer/data/`.
- A deterministic MuJoCo `compute_score.py` with ten criteria.
- Optional solution and baseline scripts.

`examples/mle-tabular-classification/` is the canonical continuous
ML_Envs-style reference. It demonstrates:

- A tabular train/test task adapted from
  `ML_Envs/examples/dataset-example-task_taiga`.
- Public parquet files in `data/` and hidden ground truth in
  `scorer/data/`.
- A `compute_score.py` that computes raw ML metrics, anchor-normalizes
  them, and applies the original exponential score curve.
- No rubrics and no `RubricBuilder`; the score is a continuous function
  of `/tmp/output/submission.csv`.

Use examples as patterns, but create your actual work under `problems/`,
not under `examples/`.

## Validate Before Opening A PR

Run the ground-truth verifier before opening or updating a PR:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```

The successful ground-truth run creates or overwrites:

```text
problems/<task_id>/.alignerr/build_proof.json
```

Commit that `build_proof.json` file and any generated
`.alignerr/ground_truth/` artifacts with your task. PR validation rejects
submitted tasks when the ground-truth proof is missing, stale, not scored at
the `reward_type` target, or missing required reviewer render artifacts.

Optional local modes are available while iterating:

```bash
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime ground-truth
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime rubric-quality
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime claude-code
```

The same commands work for checked-in examples by replacing `problems/<task_id>`
with `examples/<example_id>`. For example, to verify the MuJoCo pendulum example
end to end, including the required reviewer video render:

```bash
uv run lbx-rl-harness run --problem-dir examples/mujoco-pendulum --runtime ground-truth
uv run lbx-rl-harness run --problem-dir examples/mujoco-pendulum --runtime rubric-quality
uv run lbx-rl-harness run --problem-dir examples/mujoco-pendulum --runtime claude-code
```

For MuJoCo tasks, `--runtime ground-truth` runs `solution/solve.sh`, grades the
oracle output, and runs the configured ground-truth render command. A passing
run prints `review_artifact: .alignerr/ground_truth/...` for each generated
review artifact.

## Trusted CI On Your Fork PR

When you open or update a PR in your assigned fork, Labelbox relays it to
trusted-side CI in `lbx-rl-tasks-iso-mothership`. That workflow:

- reruns proof checks, ground-truth checks, static task validation, and Boreal
  metadata export;
- runs the DeepAgents AI harness using org-level secrets;
- runs rubric QA and Auto QA from the CI-generated proof;
- posts a `trusted-ci/grade` check and summary comment on your fork PR;
- submits a fire-and-forget Taiga eval job for advisory downstream feedback.

You do not need mothership access. Push fixes to the same fork PR; trusted CI
reruns automatically on each sync.

## Optional Local Agent And Auto QA Runs

You can still run the full agent harness locally if you want earlier feedback:

```bash
uv run lbx-rl-harness run --problem-dir problems/<task_id>
```

For the default `claude-code` runtime, use your Claude Code subscription: install
the `claude` CLI and run `claude /login` first. If you do not have Claude Code
subscription access, run the provider-key path instead:

```bash
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime deepagents
```

Configure `.env.local` from `.env.example` for `--runtime deepagents`; CI uses
org secrets.

Auto QA can also be run locally after a local agent proof exists:

```bash
uv run lbx-rl-harness autoqa --problem-dir problems/<task_id>
```

To keep the submitted proof untouched and write local artifacts under
`.harness-runs/`, use:

```bash
uv run lbx-rl-harness autoqa \
  --problem-dir problems/<task_id> \
  --output-proof-path .harness-runs/autoqa/<task_id>/build_proof.json \
  --output-json .harness-runs/autoqa/<task_id>/auto_qa.json
```

## Open A Pull Request

When your task is ready, open a GitHub PR in **your assigned fork** (not this
template repo). Keep each task PR focused on exactly one task directory under
`problems/<task_id>/`.

Before opening the PR:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
git status --short
git add problems/<task_id> problems/<task_id>/.alignerr/build_proof.json problems/<task_id>/.alignerr/ground_truth/
```

Do not open the PR until `build_proof.json` is present in `git status` and
included in your commit. If you edit task files after generating the proof,
rerun the ground-truth verifier so the proof hash matches the final task
contents.
Do not commit `.env.local`, `.harness-runs/`, provider keys, or other secrets.

Trusted CI runs automatically when your fork PR opens or updates.

## Feedback On Your PR

Alignerrs do not need mothership access. All feedback appears on the fork PR
you opened.

Feedback can come from:

- the `trusted-ci/grade` check and its summary comment,
- CI agent harness trajectory and score summaries,
- rubric QA and Auto QA summaries,
- Taiga/Boreal polling comments (advisory),
- reviewer comments before merge.

The intended feedback loop is:

1. Open or update your fork PR with a fresh ground-truth proof.
2. Trusted CI runs the full QA stack and posts `trusted-ci/grade`.
3. Taiga results arrive later as advisory comments on the same fork PR.
4. Your reviewer merges the fork PR once the check passes.
5. After merge, accepted task content integrates into the solution repo.

If feedback requests changes, do this:

```bash
# Edit files under problems/<task_id>/
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
git status --short
git add problems/<task_id> problems/<task_id>/.alignerr/build_proof.json problems/<task_id>/.alignerr/ground_truth/
git commit -m "Address review feedback"
git push
```

Each push to your fork PR reruns trusted CI automatically.

## Command Reference

Common local commands:

- `lbx-rl-harness run --problem-dir problems/<task_id>`:
  optional local Claude Code agent harness and rubric quality review.
- `lbx-rl-harness run --problem-dir problems/<task_id> --runtime ground-truth`:
  required pre-PR check. Runs only the checked-in solution/reference path. For
  `mujoco` tasks this records `ground_truth_result` and rendering artifacts.
- `lbx-rl-harness run --problem-dir problems/<task_id> --runtime claude-code`:
  run the AI agent harness through the local Claude Code CLI/OAuth path.
- `lbx-rl-harness run --problem-dir problems/<task_id> --runtime deepagents`:
  run the AI agent harness through DeepAgents and provider API keys.
- `lbx-rl-harness run --problem-dir problems/<task_id> --runtime rubric-quality`:
  run the no-agent grading/rubric-quality path. Alias: `--runtime noop`.
- `lbx-rl-harness autoqa --problem-dir problems/<task_id>`:
  optional local Auto QA review using an existing local agent proof.
- `lbx-rl-template validate --problem-dir problems/<task_id>`:
  internal static template validation used by CI; authors normally rely on PR
  validation rather than running this directly.
- `lbx-rl-template export-taiga --problem-dir problems/<task_id> --out problems-metadata.json --image PLACEHOLDER`:
  internal Boreal/Taiga metadata export used by template and mothership CI.
- `lbx-rl-harness run-taiga --metadata problems-metadata.json --source-problem-dir problems/<task_id>`:
  run a local Boreal payload while using the source task scorer for grading.
- `lbx-rl-harness run-harbor --task-dir harbor-export --source-problem-dir problems/<task_id>`:
  run a local Harbor export and write Harbor-style verifier logs.

## Testing Shared Package Changes

If you change `grader/`, `harness/`, or task helper packages, run the Python tests:

```bash
uv run pytest
```

The root test configuration intentionally excludes `problems/` and
`examples/` from pytest discovery and Ruff linting. Task-specific validation
happens through the local harness and PR validation.

## Pull Request Expectations

For task work, keep each PR focused on exactly one task directory under
`problems/<task_id>/`. Shared changes to docs, `grader/`, `harness/`, or
task helper packages may be included when they are required for that task.
You may also open focused PRs to improve any part of this codebase, including
shared grading utilities, harness behavior, validators, examples, starter
templates, documentation, or CI workflows.

Opening a PR in your assigned fork that changes a task under `problems/<task_id>/`
runs trusted CI automatically. This template repo
should not contain checked-in Boreal, GCP, Artifact Registry, Harbor, or provider
API secrets.

## More Reading

- [`docs/AUTHORING.md`](docs/AUTHORING.md): task creation walkthrough.
- [`docs/GRADING.md`](docs/GRADING.md): detailed grader package guide.
- [`docs/MLENVS_TASKS.md`](docs/MLENVS_TASKS.md): authoring ML_Envs-style tasks
  (dataset/CSV, model module, executable, HDF5, k-fold) -- submission loaders,
  calibration, licensing, mounts, and base flavors.
- [`docs/HIDDEN_ENV.md`](docs/HIDDEN_ENV.md): simulation / interaction tasks where
  the agent probes a black-box environment over an RPC socket
  (`[environment].hidden_env`, the env server, `load_submitted_policy`).
- [`docs/REWARD_HACKING.md`](docs/REWARD_HACKING.md): the reward-hacking
  mitigations the grader enforces and what every scorer must do to stay inside
  them.
- [`docs/NUMERICAL_SOLVERS.md`](docs/NUMERICAL_SOLVERS.md): installed
  numerical-solver stack and invocation patterns.
- [`docs/POLICY_ISOLATION.md`](docs/POLICY_ISOLATION.md): safe out-of-process
  execution for submitted `policy.py` / `controller.py` code.
- [`docs/RUBRIC_GUIDANCE.md`](docs/RUBRIC_GUIDANCE.md): rubric design
  principles, examples, and common pitfalls.
- [`examples/mle-tabular-classification/`](examples/mle-tabular-classification/):
  continuous ML scoring reference task.
- [`examples/mujoco-pendulum/`](examples/mujoco-pendulum/): complete
  MuJoCo robotics reference task.
- [`examples/openfoam-hydrofoil-flap/`](examples/openfoam-hydrofoil-flap/):
  CFD/OpenFOAM numerical-solver reference task.
- [`examples/opensees-base-isolation/`](examples/opensees-base-isolation/):
  structural/OpenSeesPy numerical-solver reference task.
- [`project_guidelines/mujoco_environments.md`](project_guidelines/mujoco_environments.md):
  MuJoCo task design guidance.
- [`project_guidelines/cfd/cfd_environments.md`](project_guidelines/cfd/cfd_environments.md):
  CFD task design guidance for AVL and OpenFOAM.
- [`project_guidelines/cfd/prometheus_cfd_environments.md`](project_guidelines/cfd/prometheus_cfd_environments.md):
  non-eval Prometheus CFD task guidance.
- [`project_guidelines/cfd/prometheus_eval_cfd_environments.md`](project_guidelines/cfd/prometheus_eval_cfd_environments.md):
  eval Prometheus CFD task guidance.
- [`project_guidelines/strctural_engineering/STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md`](project_guidelines/strctural_engineering/STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md):
  structural engineering task guidance for OpenSeesPy.
- [`project_guidelines/strctural_engineering/PROMETHEUS_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md`](project_guidelines/strctural_engineering/PROMETHEUS_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md):
  non-eval Prometheus structures task guidance.
- [`project_guidelines/strctural_engineering/PROMETHEUS_EVAL_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md`](project_guidelines/strctural_engineering/PROMETHEUS_EVAL_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md):
  eval Prometheus structures task guidance.
