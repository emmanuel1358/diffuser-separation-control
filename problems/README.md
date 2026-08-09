# Problems

Put authored tasks in this directory. Each task gets exactly one directory:

```text
problems/<task_id>/
```

Prefer scaffolding from the repo root:

```bash
uv run lbx-rl-template create \
  --name labelbox/my-task \
  --template ml \
  --out problems
# templates: ml | mujoco | cfd | structures | software-engineering
#            prometheus-cfd | prometheus-structures
#            prometheus-eval-cfd | prometheus-eval-structures
```

Manual copy still works (`cp -R alignerr_plugin/.../starter_templates/<starter>
problems/<task_id>`). Use `examples/` as patterns; put submitted work under
`problems/`.

## Expected Layout

```text
problems/<task_id>/
├── task.toml                 # required
├── metadata.json             # required
├── instruction.md            # required
├── environment/
│   └── Dockerfile            # required for submit / build proof
├── scorer/
│   ├── compute_score.py      # required
│   └── data/                 # as needed
├── data/                     # optional public inputs
├── solution/                 # oracle / render as needed
├── baselines/                # optional (layout varies by starter)
└── README.md                 # optional
```

Validation always requires `metadata.json`, `task.toml`, `instruction.md`, and
`scorer/compute_score.py`. The local/CI build-proof path also needs
`environment/Dockerfile`. For submitted tasks, an oracle under `solution/` is
part of the contract (typically `solve.sh`; ML may use committed-strategy
layouts). `baselines/` is optional but recommended, especially for continuous
scoring. Render scripts are required only for MuJoCo or when
`[ground_truth].render_outputs` is declared.

Keep `metadata.json` aligned with the directory:

```json
{
  "benchmark": "taiga_task",
  "problem_data": {
    "instance_id": "<task_id>"
  }
}
```

## `task.toml` Contract

Start from the copied starter and keep `schema_version = "1.1"`. At minimum,
`task.toml` must identify the task, declare resources and outputs, and provide
enum-backed difficulty metadata:

```toml
schema_version = "1.1"

[task]
name = "labelbox/<task_id>"
description = "Concise task description"

[environment]
required_resources = "12vcpu+100gib+h100/2"
storage_mb = 50000
allow_internet = true

[verifier]
env = []

[difficulty]
task_type = "ml"                         # ml | mujoco | cfd | structures | software_engineering
domain = "scientific_discovery_computational_science"
reward_type = "continuous_scoring_function" # or multi_deterministic_rubrics
license = "MIT"                         # required for ml tasks

[[outputs]]
path = "/tmp/output/submission.csv"
required = true
description = "Final artifact graded by scorer/compute_score.py"
```

Important validation details:

- `[task].name` should be `labelbox/<task_id>` and match
  `metadata.json`'s `problem_data.instance_id`.
- Every `[[outputs]]` path must be absolute and under `/tmp/output`. Tell the
  agent to write final graded artifacts there, not under `/workspace`.
- `[difficulty].task_type`, `[difficulty].domain`, and
  `[difficulty].reward_type` are required. Valid values live in
  `alignerr_plugin.task_metadata`; `domain` is scoped by `task_type`.
- `reward_type = "multi_deterministic_rubrics"` means the oracle must score
  `1.0` within `[ground_truth].score_epsilon`.
- `reward_type = "continuous_scoring_function"` means the reference solution
  should score `0.5 +/- [ground_truth].continuous_score_epsilon` (default `0.05`),
  leaving headroom for stronger agents.
- `ml` tasks must declare `[difficulty].license` as a permissive SPDX id from
  `task_metadata.LICENSES` (`MIT`, `Apache-2.0`, `BSD-2-Clause`,
  `BSD-3-Clause`, `ISC`, `Unlicense`, `CC0-1.0`, `CC-BY-4.0`, `PDDL-1.0`,
  `UPL-1.0`) or `self_generated` when the task generates its own data (no
  external dataset license). `not_applicable` remains a back-compat alias for
  `self_generated`. Copyleft, non-commercial, research-only, and share-alike
  data is rejected.
- `[runner]` and `[runner.timeouts]` are optional Boreal/Taiga knobs. The
  exporter has defaults, but starters include useful values for attempts,
  context mode, model, tools, and timeouts.
- `[[preloaded_files]]` is only for extra large mounts such as Hugging Face
  weights or trees outside the conventional task data dirs. ML task `data/` and
  `scorer/data/` are mounted automatically by trusted CI.

## Ground Truth And Reviewer Artifacts

Every submitted task must include a deterministic reference at
`solution/solve.sh`. It must create every required `[[outputs]]` artifact and
score at the target implied by `[difficulty].reward_type`.

Tasks that need reviewer video declare `[ground_truth]`. MuJoCo tasks require
one; other task types may opt in. Declared render outputs must be required video
files under `/tmp/output` and the enforced logical name is
`/tmp/output/rendering.mp4`:

```toml
[ground_truth]
render_command = "bash solution/render.sh"
render_outputs = [
  { path = "/tmp/output/rendering.mp4", required = true, description = "Reviewer video" },
]
```

For solvers or renderers that only exist inside the task image, set
`[ground_truth].in_container = true`.

## Data And Dockerfile Rules

- Put public agent-readable files in `data/`; the task image exposes them at
  `/data`.
- Put hidden grader fixtures in `scorer/data/`; the task image exposes them to
  the grader at `/mcp_server/data`.
- Do not duplicate private fixtures under public paths, and do not copy
  `scorer/` or `scorer/data/` into `/data`, `/workdir`, `/tmp/output`, `/app`,
  or `/workspace`.
- Keep `/workdir` and `/tmp/output` writable by uid `1000`.
- Keep `/mcp_server/data` and `/mcp_server/grader` root-owned and unreadable by
  uid `1000`; the validator checks both Dockerfile copy patterns and the built
  image.
- Keep Dockerfiles generic with `ARG BASE_IMAGE`, `ARG BASE_TAG`, and
  `FROM ${BASE_IMAGE}:${BASE_TAG}` so local validation and mothership export can
  inject the correct CPU/GPU base. `[environment].base_flavor` may be `auto`
  (default), `cpu`, `gpu`, `gpu-blackwell`, `gpu-openroad`, `cuda-graphics`, or
  `tpu`.

## Scorer Rules

For continuous tasks, implement `scorer/compute_score.py` as:

```python
def compute_score(workspace, trajectory, private):
    ...
```

Return a finite score in `[0, 1]` as a `float` or a score `dict`.

For `reward_type = "multi_deterministic_rubrics"`, declare
`TASK = RubricTask(...)` instead (no `compute_score()`). Harness
reference/ground-truth (and Trusted CI's seal step) write
`scorer/evaluation.plan.json` from `TASK`; validate only checks it.
Commit the generated file. See
[`docs/RUBRIC_EVALUATION.md`](../docs/RUBRIC_EVALUATION.md).

Grading must be deterministic: do not use LLM judges or external services in
the scorer. If the submitted artifact is malformed, unreadable, or crashes,
raise `grading.AgentFault` (or use shared artifact/`helpers` APIs). Do not
catch grader bugs as `AgentFault`, and never import, exec, pickle-load, or
otherwise run agent-authored code in the root grader process. Use the
sanctioned sandbox helpers for submitted Python policies or modules.

## Hidden Environment Tasks

Only use `[environment].hidden_env = "env"` or `"hybrid"` for simulation or
interaction tasks where the agent must probe a black-box environment over
`/tmp/env.sock`. Static data tasks should not use it.

Hidden environment tasks must provide:

- `scorer/data/env.py` with a `make_env` factory, kept private under
  `/mcp_server/data`.
- `data/env_client.py`, copied from the canonical env client and adapted for the
  task.
- A scorer that loads the held-out env through `grading.load_env_module` and
  submitted policies through `grading.load_submitted_policy`.

## Validate Before PR

Before opening or updating a PR, run:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
uv run lbx-rl-template check --problem-dir problems/<task_id>
```

Do **not** commit `problems/<task_id>/.alignerr/build_proof.json` (gitignored
under `problems/`). Trusted CI regenerates the authoritative proof from the
immutable PR revision. Do commit reviewer media under
`problems/<task_id>/.alignerr/ground_truth/` when render is expected. Rerun
ground-truth after task file, Dockerfile, scorer, data, or solution changes so
local feedback stays fresh.

Use `--runtime rubric-quality`, `--runtime claude-code`, `--runtime deepagents`,
or the default local run only when you want optional local feedback before
opening your fork PR. Trusted CI in `lbx-rl-tasks-iso-mothership` runs
ground-truth, proof verify, static validation, agent harness, rubric QA, Auto
QA, and advisory Taiga/Boreal submission on your assigned fork PR.

## No Secrets

Do not commit credentials, refresh tokens, service account keys, provider API
keys, or generated secret files. Files under `scorer/data/` are hidden from the
agent at runtime, but they are still committed to this repository and must not
contain secrets.

For the full guide, read `docs/AUTHORING.md`, `docs/GRADING.md`,
`docs/GROUND_TRUTH.md`, `docs/HIDDEN_ENV.md`, and `docs/REWARD_HACKING.md`.
