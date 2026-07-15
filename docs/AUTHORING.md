# Authoring Tasks

## Setup

```bash
git clone https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-template.git
cd lbx-rl-tasks-template
uv sync                # grading library + harness + local helpers (one shot)
uv run lbx-rl-harness --help # confirm the local harness is installed
```

The `uv sync` installs:

- the shared `grading` library (your `compute_score.py` will `from grading import RubricBuilder`),
- the `grader_runner` (provides `/runtime/run_grader.py` for Harbor),
- the `alignerr_plugin` package with task validation/export helpers,
- the `lbx-rl-template` CLI used by CI for local validation and Boreal/Taiga metadata export.

You don't need any access to the mothership repo or Google Artifact Registry
for local authoring. The local harness builds the required repo-local base image
from this repo on first use, then builds and runs your task image from that
local base. CPU tasks use `lbx-tasks-base:runtime-ml-core-py313-local`; GPU
tasks use `lbx-tasks-base-gpu:runtime-ml-core-py313-local`. The first Docker
build can be slow because it downloads Python and ML dependencies.

## Scaffold a task

There is one starter per supported `task_type` (`ml`, `mujoco`, `cfd`,
`structures`). Copy the one that matches your task. Use the `prometheus-*`
starters when a `cfd` or `structures` task should keep the same local/CI checks
and submit to Prometheus while also running an independent Taiga mirror.
Prometheus has two
starter families per domain: non-eval starters set `[delivery].eval = false`,
and eval starters set `[delivery].eval = true`. Both families follow the same
CI, Prometheus, and Taiga submission routes; the flag is internal metadata.

```bash
mkdir -p problems
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

- Non-eval CFD: `project_guidelines/cfd/prometheus_cfd_environments.md`
- Eval CFD: `project_guidelines/cfd/prometheus_eval_cfd_environments.md`
- Non-eval structures: `project_guidelines/strctural_engineering/PROMETHEUS_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md`
- Eval structures: `project_guidelines/strctural_engineering/PROMETHEUS_EVAL_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md`

After copying a starter, update `metadata.json` and `task.toml` so
`problem_data.instance_id` and `[task].name` match your directory. Copy
task-specific patterns from the matching `examples/` reference (not the example
directory itself).

Before opening or updating a PR, run:

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```

The **reference** command is for day-to-day iteration (solve + grade, persistent
cache, no build-proof refresh). **Ground-truth** is the PR gate: it updates
`.alignerr/build_proof.json` with `ground_truth_result`.

The ground-truth runtime checks the reference solution without calling an LLM.
It creates or overwrites `problems/<task_id>/.alignerr/build_proof.json` and,
for rendered tasks, `.alignerr/ground_truth/` reviewer artifacts. Commit those
files before opening or updating a PR in your assigned fork. When the fork PR
opens or updates, the template repo dispatches trusted CI in
`lbx-rl-tasks-iso-mothership`, but the template PR stays the author-facing
surface. The template workflow posts a handoff comment, and mothership posts the
`trusted-ci/grade` check, dashboard link, diagnostics, and later Taiga/Boreal
feedback comments back on the same fork PR. Authors and labelers do not need to
open mothership to see pipeline status. See
[`GROUND_TRUTH.md`](GROUND_TRUTH.md) for the oracle and reviewer video contract.

For MuJoCo tasks, add the `run_qa`, `run_adversarial`, or
`run_mujoco_adversarial` label when you want the non-blocking adversarial
score-shaping review. That review also runs in mothership and posts its
`MuJoCo Adversarial AutoQA (shadow)` comment back on the template PR.

There are no task categories. Add the files your task needs; the validator
runs checks based on what is present.

## Layout

```
problems/<task_id>/
├── task.toml             # config + [runner] + [difficulty]
├── metadata.json         # task identity and description metadata
├── instruction.md        # task prompt the agent sees
├── environment/
│   └── Dockerfile        # FROM lbx-tasks-base, COPY scorer/ -> /mcp_server/grader/
├── scorer/
│   ├── compute_score.py  # YOUR scorer (see docs/GRADING.md)
│   └── data/             # private hidden test set
├── data/                 # public data the agent sees at /data/
├── solution/solve.sh     # required reference/oracle solution
├── solution/render.sh    # required reviewer video generation for mujoco tasks
├── baselines/naive.sh    # optional weak baseline
└── README.md
```

### Runtime Prompt Guidance

Write `instruction.md` for the agent, not for reviewers. Name installed tools and
libraries directly (for example, “NumPy, pandas, and scikit-learn are installed”)
instead of telling the agent to inspect the Docker/base image, task
`metadata.json`, or “declared dependencies”. Those source/build internals are
not part of the task contract and the validator rejects prompts that point agents
at them as package or dependency sources. Dataset metadata files such as
`/data/metadata.json` are fine when they are part of the problem data.

Taiga export appends runtime notices automatically when needed: GPU/TPU
availability and guidance to use the dedicated `tmux` tool for long-running
training. You may write domain-specific accelerator guidance yourself; the
exporter avoids duplicating notices when your prompt already mentions GPU/TPU or
`tmux`.

## Task Metadata

Every task must declare three enum-backed metadata fields in `[difficulty]`:

```toml
[difficulty]
task_type = "ml"                         # ml | mujoco | cfd | structures
domain = "scientific_discovery_computational_science" # enum scoped by task_type
reward_type = "continuous_scoring_function" # or multi_deterministic_rubrics
license = "MIT"                           # required for ml tasks; permissive SPDX id
license_source = "https://github.com/owner/dataset/blob/main/LICENSE" # required for ml; upstream URL
```

`ml` tasks must additionally declare `[difficulty].license`: a permissive SPDX
identifier from `task_metadata.LICENSES` (`MIT`, `Apache-2.0`, `BSD-2-Clause`,
`BSD-3-Clause`, `ISC`, `Unlicense`, `CC0-1.0`, `CC-BY-4.0`, `PDDL-1.0`,
`UPL-1.0`), or `self_generated` when the task produces its own data
(e.g. fully synthetic data generated at runtime). Trace the dataset license
**upstream** before picking one - Hugging Face and Kaggle mirrors routinely
carry incorrect license claims. Copyleft, share-alike, non-commercial, and
research-only data is rejected outright because it cannot ship in a commercial
delivery. (`not_applicable` is still accepted as a back-compat alias for
`self_generated`.) `mujoco`/`cfd`/`structures` tasks use solver-generated or
self-authored data and may omit the field.

`ml` tasks must also set `[difficulty].license_source` -- the provenance pointer
a licensing code-owner verifies by hand. For a real license it must be the
upstream-most **http(s) URL** where the license was confirmed; for
`self_generated` it is a short written reason the task makes its own data. The
gate is two-step: validation mechanically checks the allowlist + that the source
is present (a URL for real licenses), then a licensing reviewer **approves the
PR** as the human sign-off. There is deliberately no self-typed "reviewer" field
-- the code-owner approval is the immutable, identity-backed sign-off.

### Delivery platform

Omit `[delivery]` for the default Taiga path. Set it only when a task should use
a non-default post-CI route:

```toml
[delivery]
platform = "prometheus"
```

Prometheus delivery is independent of `task_type`. For numerical-solver tasks,
keep `task_type = "cfd"` or `task_type = "structures"` and use the matching
Prometheus starter. Trusted CI still runs the same CFD/structures checks before
delivery: solver-agnostic prompt checks, grader QA, solver-backed oracle
validation, local agent score gates, Auto QA, and build-proof validation. Only
the final delivery job changes from Taiga submission to the Prometheus Agent
Service runner. The Prometheus result must include target reward metrics, and
CFD/structures submissions fail if the Prometheus target average score is above
`0.5` or target reward standard deviation is below `0.1` across the required
attempts, or if the trajectory trainability auditor does not pass. The full
`submit-prometheus` workflow must pass before an eval row is accepted or a
non-eval row moves into review. Submitting a non-passing non-eval row for review
violates fair practices and may remove the tasker from the project.

Prometheus Harbor exports require agent/verifier user separation in
``task.toml``: ``[agent].user = "agent"`` (non-root uid 1000) and
``[verifier].user = "root"`` so grading can read ``/mcp_server/data`` while the
agent cannot. The Prometheus starters and Harbor exporter set these values
automatically; keep them when editing a Prometheus task.

### Base image flavor

`[environment].required_resources` selects one Taiga-supported resource enum.
Authors must pick one of the values in `alignerr_plugin.taiga_resources`; do not
declare arbitrary `cpus`, `memory_mb`, `gpus`, or `gpu_types` in `task.toml`.

Available resource enums:

```text
CPU: 2vcpu+6gib, 4vcpu+16gib, 6vcpu+32gib, 8vcpu+64gib, 16vcpu+64gib, 16vcpu+128gib
CPU perf: 2vcpu+6gib+perf, 4vcpu+16gib+perf, 6vcpu+32gib+perf, 8vcpu+64gib+perf, 16vcpu+64gib+perf, 16vcpu+128gib+perf
TPU: 13vcpu+32gib+tpuv5e1x1, 16vcpu+64gib+tpuv5e2x2, 50vcpu+128gib+tpuv5e2x2
H100: 3vcpu+25gib+h100/8, 6vcpu+50gib+h100/4, 12vcpu+100gib+h100/2, 24vcpu+200gib+h100/1
H100 graphics: 3vcpu+25gib+h100/8+graphics, 6vcpu+50gib+h100/4+graphics, 12vcpu+100gib+h100/2+graphics, 24vcpu+200gib+h100/1+graphics
```

`[environment].base_flavor` selects the base image: `auto` (default; inferred
from `required_resources`), or an explicit compatible `cpu` / `gpu` /
`gpu-openroad` / `cuda-graphics` / `tpu`.

- `gpu` is the canonical ML GPU stack for H100/Hopper tasks.
- `gpu-openroad` adds OpenROAD/EDA binaries on top of the clean ML GPU base.
- `cuda-graphics` ships CUDA-native rasterization (pytorch3d / nvdiffrast). Use
  it for rendering tasks: hardware GL/EGL does **not** work under Taiga's gVisor
  sandbox, so CUDA-native raster is the only deployable render path.
- `tpu` ships `jax[tpu]` on Python 3.12.

Base images are built and pushed with drift-hashed tags by
`base/build_and_push.sh` (the tag is a content hash over every base build input,
so a stale base is never silently reused). The exporter points each task at the
matching tag automatically.

### Large datasets and model weights (preloaded_files)

Do not bake large datasets or model weights into the task image (bloat, no
cross-task dedup, network + token at every build). They are mounted read-only at
deploy time instead.

For `ml` tasks the raw dataset is mounted **automatically**: trusted CI packs
`data/` -> `/data` and `scorer/data/` -> `/mcp_server/data` into
content-addressed squashfs and mounts them, so just commit those dirs as usual --
nothing to declare.

Declare `[[preloaded_files]]` only for *extra* mounts (Hugging Face weights or
trees outside the conventional dirs):

```toml
[[preloaded_files]]
source = "data/big_dataset"     # local tree -> content-addressed squashfs
mount_path = "/data/big_dataset"

[[preloaded_files]]
hf_repo = "org/model-name"       # Hugging Face weights, fetched + packed
hf_revision = "<commit-sha>"     # pin for immutability
mount_path = "/tmp/hf-cache"     # HF_HOME, so from_pretrained resolves offline
```

Trusted CI runs `scripts/sync_mount.sh <problem-dir>` at submit time: it packs
each source into a content-addressed squashfs, uploads it once to the shared
cache (reused across tasks), and stamps `.alignerr/preloaded_files.json`, which
the exporter emits so the platform mounts each object read-only. You do **not**
need Taiga upload credentials and do **not** commit the stamp -- CI generates it
(run the script locally only if you want to preview the mounts). The base image
sets `HF_HOME=/tmp/hf-cache` so mounted weights resolve without a network fetch.

The master enum definitions live in `alignerr_plugin.task_metadata`; do not add
new values ad hoc in docs or task files. Domains are scoped by `task_type`; for
example, ML includes the production taxonomy (`physical_sciences`,
`healthcare_bioinformatics`, `robot_dynamics_system_identification`) plus planned
expansion areas (`world_models`, `active_perception`, `long_horizon_planning`).
MuJoCo includes `policy_training_improvement` and `manipulation`, CFD includes
`aerodynamics` and `vortex_suppression`, and structures includes
`seismic_retrofit` and `topology_optimization`.

`task_type` is the broad domain family. `domain` is the finer diversity label
used by the dashboard/Supabase index. `reward_type` controls the ground-truth
score target: deterministic rubric tasks must score `1.0`, while continuous
scoring-function tasks should score `0.5 ± 0.05`.

MuJoCo tasks require a reviewer video. The other task types do not require a
rendering video unless they declare one (see Reviewer Artifacts). The shared
base image already bundles a verified open-source solver stack
(AeroSandbox, PySpice/ngspice, scikit-fem, PyNite, anastruct, scikit-rf,
Cantera, CoolProp, KLayout); see
[`docs/NUMERICAL_SOLVERS.md`](NUMERICAL_SOLVERS.md) for the full list, import
names, determinism notes, and per-task recipes for heavier engines (Meep,
OpenFOAM/SU2, OpenROAD).

For `cfd` and `structures` tasks, `instruction.md` must be solver-agnostic. The
prompt must describe the engineering objective, deliverable, schema, units, and
constraints without mentioning solvers, naming solver software, requiring solver
commands, asking for solver evidence, or hinting at solver/tool implementation
details. Physics-grounded scoring details are still required and should be written in
solver-agnostic terms. The core solution implementation (`solution/solve.sh` or
solution files it calls) must include solver-specific runnable material, and the
hidden oracle/scorer may still run the real solver.

### Hidden-environment (simulation) tasks

For **simulation / interaction** tasks where the agent must probe a black-box
environment whose dynamics it cannot read (a simulator, market, bandit, game
opponent), set `[environment].hidden_env = "env"` (socket only) or `"hybrid"`
(socket + static `data/`). The rubric server then spawns an RPC env server the
agent reaches over `/tmp/env.sock`. This is orthogonal to `task_type` (any type
can opt in) and is the wrong tool for static-data tasks. Ship the hidden env at
`scorer/data/env.py` (`make_env`) and a public `data/env_client.py`; grade with
`grading.load_env_module` + `grading.load_submitted_policy`. Full guide:
[`docs/HIDDEN_ENV.md`](HIDDEN_ENV.md); reference: `examples/hidden-env-bandit/`.

## Output Directory

Use `/tmp/output` for all final agent-created artifacts, and declare each final
artifact in `task.toml` with `[[outputs]]`:

```toml
[[outputs]]
path = "/tmp/output/model.xml"
required = true
description = "Final MJCF model the agent writes."
```

This contract is task-type agnostic: `ml`, `mujoco`, `cfd`, `structures`, and any
hidden-env variant all use the same `[[outputs]]` shape. The Taiga exporter
copies these declarations into the problem payload as `outputs` and mirrors them
under `extra_fields.task_metadata.outputs` so Taiga QA, local Taiga-format runs,
and reviewers see the same required artifact list as the authoring validator.

Do not use `/workspace` in task prompts or `task.toml` output paths.

## Dockerfile Output Permissions

Task tools run as a non-root user in Boreal. Every task Dockerfile should keep
`/workdir` and `/tmp/output` writable for uid `1000`; otherwise agents can
reason correctly but fail when saving required files such as
`/tmp/output/model.xml`.

Keep this pattern near the top of `environment/Dockerfile`, after the `FROM`
and `ARG PROBLEM_DIR` lines:

```dockerfile
WORKDIR /workdir
RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output
```

The starter templates already include this. Preserve it when copying or
rewriting Dockerfiles.

Conversely, the **private** roots that hold your scorer and answer key
(`/mcp_server/grader`, `/mcp_server/data`) must stay root-owned and unreadable
to uid `1000`. A plain `COPY scorer/data/ /mcp_server/data/` leaves them
world-readable (`0644`) — the agent can `cat` the answer. Always harden them:

```dockerfile
COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} +
```

`uv run lbx-rl-template validate` now fails any Dockerfile that copies into the
private roots without this hardening, and an in-image probe confirms uid `1000`
cannot read them.

## GPU Tasks

**Prefer GPU.** We want GPU-accelerated tasks. Default to a Taiga H100 enum
whenever the work can use acceleration -- training, deep models, RAPIDS/cuDF,
CUDA solvers, batched evaluation. **`ml` tasks default to GPU**: the `ml`
starter ships `required_resources = "12vcpu+100gib+h100/2"`. Only pick a CPU
enum when a task genuinely cannot use a GPU, and note why.

Set GPU requirements in `task.toml`:

```toml
[environment]
required_resources = "12vcpu+100gib+h100/2"
storage_mb = 50000
allow_internet = true
```

When `required_resources` is an H100 tier, the local harness builds and uses
`lbx-tasks-base-gpu:runtime-ml-core-py313-local` instead of the CPU base.
Graphics tiers ending in `+graphics` select the `cuda-graphics` base when
`base_flavor = "auto"`. TPU tiers select the TPU base.

For local and trusted-CI agent runs, do not make hardware presence itself the
reward target. The harness tells agents to probe GPU/TPU availability once and
fall back quickly to a CPU-compatible path if accelerator access is unavailable
or broken. Optional reports may record backend/device details, but graders should
score the required output quality rather than requiring `cuda`, `tpu: true`, or
other hardware-only side effects. Taiga submission still requests the configured
accelerator resources.

Keep `environment/Dockerfile` generic:

```dockerfile
ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=problems/<task_id>
FROM ${BASE_IMAGE}:${BASE_TAG}
```

Do not hard-code the production GPU image in task Dockerfiles. The harness and
mothership workflows pass the correct `BASE_IMAGE` and `BASE_TAG` build args.

Local GPU image builds do not require Google Artifact Registry access, but
executing GPU code locally requires Docker/NVIDIA GPU support on the host. If
your machine cannot run GPU containers, still run the harness when possible for
build-proof generation and rely on trusted CI on your fork PR for the
author-visible runtime check before acceptance.

## Grading — pick your shape

Your `scorer/compute_score.py` returns one of three shapes. The runtime
detects the type and packs the appropriate canonical `Grade.to_dict()`
for both Boreal UI and Harbor `reward.json`.

All grading must be deterministic. The agent can be an LLM; the verifier
cannot. Do not call LLM providers from `compute_score.py`.

| Return shape | When to use |
| --- | --- |
| `float` in `[0, 1]` | ML_Envs-style continuous metric (RMSE/F1, anchor-mapped). Single number, no per-criterion breakdown. |
| `dict {score, subscores, weights, metadata}` | Custom anchor-mapped headline + diagnostic per-target rows in Boreal UI. The headline is `dict["score"]`, NOT a recomputed weighted average. |
| `RubricBuilder.grade().to_dict()` | Weighted deterministic criteria + optional penalties. The natural shape when "did this pass?" is decomposable into code-checkable checks. |

### Bare float (simplest, ML_Envs migration)

```python
# scorer/compute_score.py
from pathlib import Path

def compute_score(workspace: Path, trajectory, private: Path) -> float:
    f1 = compute_my_f1(workspace / "submission.csv", private / "test_target.csv")
    return anchor_map(f1, floor=0.0, perfect=1.0)
```

This shape is intended for ML_Envs migrations where only the headline
score matters. See
[`examples/mle-tabular-classification`](../examples/mle-tabular-classification/)
for a complete continuous-scoring dataset task.

For the full ML_Envs calibration pattern, including baseline/reference/perfect
anchors and difficulty targets for continuous reward functions, see
[`docs/GRADING.md`](GRADING.md#continuous-reward-functions-from-ml_envs).

### Score dict (preserve a custom anchor map AND surface diagnostics)

```python
def compute_score(workspace, trajectory, private) -> dict:
    return {
        "score": exp_curve(x_agg),                # YOUR headline math
        "subscores": {"rmse_progress": x1, "f1": x2},
        "weights":   {"rmse_progress": 0.5, "f1": 0.5},
        "metadata":  {"raw_rmse": rmse},
    }
```

The headline `score` stays exactly what you returned — the runtime does NOT
recompute it from `subscores * weights`.

Use this shape for continuous ML tasks when you want to preserve the
original ML_Envs headline math and also expose diagnostic target
progress values. This is still not a rubric: the subscores are
continuous metrics, not pass/fail criteria.

### RubricBuilder (typed deterministic criteria + penalties)

```python
from pathlib import Path
from grading import RubricBuilder, helpers   # baked into lbx-tasks-base

def compute_score(workspace: Path, trajectory, private: Path):
    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)

    @rb.criterion(id="answer_present", weight=1.0,
                  description="answer.txt exists and is non-empty")
    def _():
        return helpers.file_exists(workspace / "answer.txt", non_empty=True)

    @rb.penalty(id="wrote_outside", value=-0.3)
    def _():
        return any((workspace / p).exists() for p in ["/etc/passwd"])

    return rb.grade().to_dict()
```

See [`examples/mujoco-pendulum`](../examples/mujoco-pendulum/) for a
deterministic-only RubricBuilder.

Do not use LLM judges in `compute_score.py`. Rubrics must be deterministic:
same submitted files, same hidden fixtures, same fixed seeds, same score.
If a criterion cannot be checked by code, rewrite it as a measurable property.

For verifier environment variables, keep the default empty list unless your
deterministic grader needs a non-secret runtime setting:

```toml
[verifier]
env = []
```

## Submission knobs (`task.toml [runner]`)

```toml
[runner]
attempts = 3                    # n_attempts_per_problem
turn_limit = 1500               # null = unlimited
max_ctx = 1_000_000
context_mode = "none"           # none / autocompact / memory
api_model_name = "claude-fable-5"
required_tools = ["bash", "str_replace_editor", "tmux"]

[runner.timeouts]
setup_sec = 600
grading_sec = 600
tool_sec = 120
max_episode_sec = 3600
```

These become per-problem and job-level fields in the Boreal submission.
Defaults are sensible — leave the section out and the exporter uses them.

## Holistic grading guide

The full author guide is at [`docs/GRADING.md`](GRADING.md). Highlights:

- API reference for `RubricBuilder`, `helpers`
- Decision tree for picking a return shape
- How `Grade.to_dict()` flows to Boreal UI and Harbor `reward.json`
- Customer RL training flow (reading `reward.json` in your trainer)
- Common patterns and gotchas

## Submit and iterate through fork PRs

Open a PR in **your assigned fork** (not this template repo). Keep the PR focused
on one task directory under `problems/<task_id>/`.

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
git status --short
git add problems/<task_id> problems/<task_id>/.alignerr/build_proof.json problems/<task_id>/.alignerr/ground_truth/
```

Do not open the PR until the generated `build_proof.json` is committed. If
you edit task files after generating the proof, rerun the ground-truth verifier
so the proof hash matches the final task contents.

Trusted CI runs automatically on every fork PR open or update. The expensive
agent harness, rubric QA, Auto QA, and `trusted-ci/grade` check all run in
`lbx-rl-tasks-iso-mothership`. Alignerrs do not need mothership access;
feedback appears on the fork PR, including advisory Taiga/Boreal comments after
submission.

If feedback requests changes, edit the same `problems/<task_id>/`, rerun the
ground-truth verifier, and push to the same fork PR. Each push reruns trusted CI:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
git add problems/<task_id> problems/<task_id>/.alignerr/build_proof.json problems/<task_id>/.alignerr/ground_truth/
git commit -m "Address review feedback"
git push
```
