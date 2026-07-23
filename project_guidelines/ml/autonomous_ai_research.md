# Autonomous AI Research ML Task Authoring Guide

This guide is for authors creating continuous ML tasks in the ISO template for
the Autonomous AI Research Taiga environment. It uses the current
`lbx-rl-tasks-iso-template` metadata-mode contract.

An ML task here is not a one-off Kaggle clone. It is a verifiable research
problem where an agent reads a prompt, uses public data or a public simulator,
writes a final artifact under `/tmp/output`, and receives a deterministic
continuous score from hidden truth. The score becomes reinforcement-learning
feedback, so the grader must be correct, deterministic, and resistant to
shortcuts.

## 1. Current ISO ML layout

New continuous ML tasks use the template's metadata-mode ML contract. Operational
fields are synthesized centrally, while authors keep task-specific grading in
`test_file.py`:

```text
problems/<task_id>/
|-- metadata.json
|-- prompt.md
|-- test_file.py
|-- calibration.lock.json         # generated, never hand-edited
|-- data-generation/
|   `-- generate.py
|-- data/
|   |-- public/                    # mounted read-only at /data
|   `-- private/                   # root-only at /mcp_server/data
|-- reference_solution/
|   |-- train.py
|   |-- solution.py
|   |-- model.*
|   `-- model.manifest.json
|-- baselines/
|   `-- naive/
|       |-- train.py
|       |-- solution.py
|       |-- model.*
|       `-- model.manifest.json
`-- README.md
```

`task_type = "ml"` remains the broad platform category. Dataset, model,
executable, HDF5, k-fold, hidden-environment, and learned-policy tasks are
evaluation paradigms expressed inside the hand-authored grader.

Native `task.toml` ML tasks remain supported for migration and specialized
images, but new authors should start from the metadata-mode ML starter and
`docs/MLENVS_TASKS.md`.

## 2. The Mental Model

Every strong ML task has five pieces:

1. A domain-faithful problem that actually requires machine learning.
2. Public artifacts under `data/` that are sufficient for a capable agent to
   train, adapt, probe, or infer a useful solution without seeing hidden truth.
3. Hidden fixtures under `data/private/` that define the held-out evaluation.
4. A deterministic hand-authored evaluator in `test_file.py`, composed with
   `ContinuousTask` for reviewed anchors and generated PWL scoring.
5. A committed reference model and inference path in `reference_solution/` that proves the problem is
   solvable and scores `0.5 +/- 0.05` for continuous scoring tasks.

The goal is not to make the reference perfect. The reference is the expert
anchor at score `0.5`. A theoretical perfect answer maps to `1.0`; weak
baselines should be far below the reference but usually above the raw floor.
Trusted CI checks this shape.

## 3. Scaffold The Task

Start from the ML starter:

```bash
mkdir -p problems
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/ml problems/<task_id>
```

Then update:

- `metadata.json`: set ML paradigm, resources, domain, license, and description.
- `prompt.md`: the agent-facing task prompt.
- `data/public/`: public training data, public simulator, public schema, or public
  helper files.
- `data/private/`: hidden labels, hidden seeds, held-out cases, private eval
  distributions, and grader-only metadata.
- `test_file.py`: hand-authored loading and raw evaluation composed with `TASK`.
- `reference_solution/`: committed model, train/inference scripts, and manifest.
- `baselines/naive/`: weak input-dependent model and reproducible recipe.

Do not edit shared template code unless the task genuinely needs a reusable
tooling improvement. Authored task work should stay inside one
`problems/<task_id>/` directory.

## 4. `metadata.json` contract for ML tasks

The starter synthesizes `task_type = "ml"`,
`reward_type = "continuous_scoring_function"`, offline runtime, timeouts, and
output conventions. Authors provide:

```json
{
  "ml_task_type": "dataset",
  "required_resources": "12vcpu+100gib+h100/2",
  "domain": "scientific_discovery_computational_science",
  "license": "CC0-1.0",
  "license_source": "https://example.com/upstream-license",
  "description": "Concise description of the ML task."
}
```

Important details:

- `ml_task_type` selects `dataset`, `env`, `hybrid`, or `sim_policy`.
- `domain` must be one of the ML domains in
  `alignerr_plugin.task_metadata.DOMAINS_BY_TASK_TYPE["ml"]`.
- The generated reference score must remain `0.5 +/- 0.05`.
- `license` and `license_source` must identify cleared upstream provenance.
- Agent output remains under `/tmp/output`; the specific artifact is declared
  by the grader and prompt.
- ML tasks default to GPU in this template. Keep GPU resources unless the task
  genuinely cannot use acceleration.

Allowed ML license ids are:

```text
MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Unlicense,
CC0-1.0, CC-BY-4.0, PDDL-1.0, UPL-1.0, not_applicable
```

Copyleft, share-alike, non-commercial, research-only, and unlicensed datasets are
not acceptable.

## 5. Trusted CI Reality

A fork PR is relayed to `lbx-rl-tasks-iso-mothership`, which runs trusted CI. The
important gates for ML authors are:

- It expects exactly one changed `problems/<task_id>/` directory.
- It runs an environment QA pre-flight from mothership-owned scripts.
- It restores or runs `uv run lbx-rl-harness run --runtime ground-truth
  --problem-dir <task>` and generates authoritative evidence from the immutable
  PR revision.
- It runs `uv run lbx-rl-template validate --problem-dir <task>`.
- It runs reward-hacking lint. Some findings are advisory, but blocking
  validator stages such as `agent_fault`, private data layout, metadata, output
  paths, and proof checks must pass.
- It runs the agent harness and Auto QA.
- It gates the local harness score against configured bounds. In the current
  mothership config, the local score must be inside `[0.1, 0.7]` unless an infra
  override is explicitly allowed.
- If the sandbox job succeeds, ML tasks are submitted to the Autonomous AI
  Research ISO Taiga environment.
- Mothership applies ML/GPU overrides at submit time: ML tasks route to the
  Autonomous AI Research environment and use the `12vcpu+100gib+h100/2` GPU tier
  with the GPU model override.
- For metadata-mode `ml` tasks, trusted CI packs `data/public/` and
  `data/private/` into read-only
  content-addressed mounts before Taiga submission, then slims them out of the
  image build context. Do not rely on large data being baked into the Docker
  image.

The practical consequence: a task that "works locally" but uses a broad
`except Exception: return 0.0`, reads agent outputs unsafely, hides public data in
the image, omits its generated calibration lock/build proof, or carries stale
model manifests will fail CI.

## 6. Choosing A Research-Grade Problem

The best Autonomous AI Research tasks test whether agents can do real ML
research work, not whether they can exploit a synthetic formula. Strong task
families include:

- world models;
- test-time adaptation;
- active perception;
- multimodal sensor fusion;
- online system identification;
- long-horizon planning;
- robotics and embodied AI;
- computer vision and 3D perception;
- STEM ML tasks with genuine scientific or engineering structure;
- RL or policy-learning tasks when the policy is learned rather than hand-coded.

Prefer domains where you have enough expertise to defend the data generator,
simulator, targets, and reference approach. A reviewer should be able to inspect
`data/public/`, `data/private/`, `test_file.py`, and the generator/provenance
and agree that the task is an instance of the named domain.

Strong ML tasks usually have these properties:

- ML is load-bearing. Deterministic feature engineering or a closed-form rule
  does not account for most of the reference's progress.
- The data or environment is domain-faithful. Equations, constraints, dynamics,
  observation models, labels, and citations match the domain claims.
- There is a real train/test or public/hidden distribution shift.
- Public data plus the prompt give a domain expert enough context to discover a
  good approach through normal EDA or experimentation.
- Naive baselines fail for understandable reasons.
- The reference demonstrates domain knowledge and substantially beats baselines.
- The scoring curve preserves headroom: reference near `0.5`, perfect at `1.0`,
  weak baselines far below reference.

Avoid tasks where:

- a simple formula, lookup table, prompt leak, filename, row order, seed, or
  public helper exposes the answer;
- the data is random numbers with domain labels attached;
- the target name implies a physical, biological, or geometric operation that
  the generator does not actually compute;
- a citation is used as decoration rather than implemented methodology;
- the reference is mostly deterministic preprocessing with a trivial fitted
  model at the end;
- the agent can submit a constant, empty file, or copied prompt example and score
  near reference.

## 7. Domain Faithfulness Bar

Domain faithfulness is binary. Passing validation is not enough.

A task is domain-faithful when:

- The substrate carries the domain's structure: constraints, symmetries,
  topology, conservation laws, dynamics, geometry, observational noise, or other
  invariants implied by the prompt.
- Targets are produced by the operation their names imply. If a target is a
  simulator outcome, value function, physical observable, geometric measurement,
  biological label, or perception output, the generator or private evaluator
  actually computes that thing.
- Public descriptions match the code. `prompt.md`, `data/public/column_mapping.json`,
  schemas, and README text describe what is actually generated and scored.
- Citations describe implemented methods, not just vocabulary.
- The public information is sufficient for an expert to recognize the task's
  structure, while still withholding labels, hidden seeds, anchors, and scoring
  constants.

Not acceptable:

- invented equations marketed as real physics or biology;
- random graphs, random matrices, random point clouds, or random trajectories
  relabeled as domain objects without the domain's invariants;
- hidden targets computed by arbitrary closed-form functions while the prompt
  claims they are simulator or experimental outcomes;
- prompt claims that imply unimplemented structure;
- "stylized proxy" disclaimers used to lower the scientific bar.

If you are unsure whether your equations or simulator are valid for the domain,
choose a domain you know better.

## 8. Public And Private Data Design

Use the public/private split deliberately:

```text
data/
|-- public/
|   |-- train.parquet
|   |-- test.parquet
|   |-- column_mapping.json
|   `-- public_sim.py
`-- private/
    |-- test_target.parquet
    |-- hidden_cases.json
    `-- eval_seeds.json
```

Public data should contain everything the agent needs to work, except answers
and grader-only mechanisms. Hidden data should contain ground truth, held-out
parameters, private seeds, hidden cases, and reference calibration artifacts.

For dataset tasks:

- Generate or collect enough training data for non-trivial ML.
- Use a held-out split by parameter range, environment, time, instrument,
  geography, domain shift, simulator regime, or other meaningful axis. Avoid
  random splits unless the task is explicitly about IID generalization.
- Public test features must not contain targets, target-derived columns, row ids
  that key into hidden truth, or reversible encodings of labels.
- Include target diversity and a meaningful dynamic range.
- Use compressed, appropriate formats: Parquet + Zstd for tabular data, Zarr or
  HDF5 for dense arrays, WebDataset shards for large CV corpora, native codecs
  for images/video, and domain-standard formats where needed.
- Add public schemas or `column_mapping.json` with high-level semantics and
  units, but not the reference transforms or scoring anchors.

For public simulators or learned-policy tasks:

- Put inspectable training simulators or helpers under `data/public/`.
- Keep hidden evaluation dynamics, hidden seeds, or private scenarios under
  `data/private/`.
- If the agent must interact live with a hidden black-box environment during the
  solve, use `[environment].hidden_env = "env" | "hybrid"` and follow
  `docs/HIDDEN_ENV.md`.
- If a policy is evaluated by the grader, run submitted Python through
  `helpers.run_policy`, `grading.load_submitted_policy`, or `PolicyWorker`, not a
  direct import.

For external data:

- Get licensing cleared before doing data work.
- Trace the license upstream, not only through mirrors.
- Record sources, license ids, and transformation provenance in the README or PR
  description.
- Do not use datasets without a license, non-commercial datasets, copyleft or
  share-alike data, or "research use only" data.

## 9. Submission Paradigms And Safe Loaders

The prompt and grader must define an artifact under `/tmp/output`. Match it to a
sanctioned loader:

| Submission paradigm | Output path example | Scorer loader |
| --- | --- | --- |
| CSV/dataframe predictions | `/tmp/output/submission.csv` | `helpers.load_submission_or_fault` |
| HDF5 arrays | `/tmp/output/submission.h5` | `helpers.load_submission_h5_or_fault` |
| Python policy or callable | `/tmp/output/policy.py` | `helpers.run_policy` or `grading.load_submitted_policy` |
| Python model module | `/tmp/output/model.py` | `helpers.run_model_module` |
| Executable solver/trainer | `/tmp/output/run` | `helpers.run_submitted_executable` |
| K-fold model module | `/tmp/output/model.py` | `grading.score_kfold_cv` |
| `sim_policy` held-out rollout | `/tmp/output/policy.py` | `grading.policy_eval.run_seeds` + `aggregate` |

For `sim_policy` grading, use `grading.policy_eval.run_seeds` + `aggregate` with
`failed_fill=FLOOR` as the default. `run_seeds(roll_one_seed, n_seeds=..., ...)`
loads the policy once through the sandboxed `load_submitted_policy` and rolls
deterministic held-out seeds; a crashed (or soft-timed-out) seed is recorded as a
failure rather than aborting the run. Reduce with
`aggregate(per_seed, keys, failed_fill={key: FLOOR})` so a failed seed contributes
each key's FLOOR (worst legitimate value) instead of being skipped. Flooring
crashes — not skipping them — preserves granularity and removes the incentive to
strategically crash out of hard seeds to lift the mean.

Do not load an agent-controlled pickle, joblib, torch checkpoint, or arbitrary
Python module directly in the root grader process. The validator scans for
unsafe patterns because an agent artifact can execute code as root or read hidden
truth if imported incorrectly.

For simple first tasks, CSV submissions are fine. For richer ML tasks, prefer
model modules or policies when the grader should evaluate on hidden cases the
agent never saw. The safe pattern is to ask for a source module with a narrow API
such as:

```python
def predict(X):
    ...
```

Then call it from `test_file.py` with:

```python
from grading import AgentFault, helpers

def measure_submission(workspace=Path("/tmp/output"), private=Path("/mcp_server/data")):
    X_hidden, y_hidden = load_hidden_data(private)
    try:
        pred = helpers.run_model_module(
            workspace / "model.py",
            "predict",
            X_hidden,
            timeout_s=60.0,
        )
    except AgentFault:
        raise
    except Exception as exc:
        raise AgentFault(f"submitted model failed: {type(exc).__name__}: {exc}") from exc
    return {"prediction_quality": score_predictions(pred, y_hidden)}

def compute_score():
    prediction, truth = load_prediction_and_truth()
    return TASK.grade(prediction, truth)
```

For submitted executables, never trust stdout lines such as `RUBRIC_SCORE=`.
`helpers.run_submitted_executable` scrubs score-forging markers, caps output, and
runs the executable in a non-root sandbox.

## 10. Scorer Contract

Metadata-mode ML `test_file.py` must define the no-argument platform entrypoint
and should separate raw evaluation:

```python
def measure_submission(
    workspace=Path("/tmp/output"),
    private=Path("/mcp_server/data"),
):
    ...

def compute_score():
    submission, truth = load_submission_and_truth()
    return TASK.grade(submission, truth)
```

Where:

- `workspace` points to the final output directory, normally `/tmp/output`.
- `private` points to `/mcp_server/data`.

Return one of:

- a finite `float` in `[0, 1]`;
- a score dict with `score`, optional `subscores`, `weights`, and `metadata`;
- for `multi_deterministic_rubrics`, a declarative `RubricTask` (see
  [`docs/RUBRIC_EVALUATION.md`](../../docs/RUBRIC_EVALUATION.md)).

`TASK.grade()` returns the authoritative protected score dict. It computes raw
registered metrics, information evidence, per-target progress, and the final
curve. `TASK.score(metrics)` is calibration-only and must not be called by
production `compute_score`.

```python
return TASK.grade(submission, truth)
```

The headline is `dict["score"]`. The runtime does not recompute it from
`subscores`.

Scorer rules:

- Be deterministic. Fix seeds, hidden case lists, fold splits, and rollout
  protocols.
- Do not call LLM providers, external APIs, or live services from the grader.
- Do not read agent artifacts by hand when a helper exists.
- Do not import or execute agent-authored Python in the root grader process.
- Do not broadly catch author/infra failures and turn them into zeros.
- Do not use hidden data under public paths.
- Keep score finite and clamped to `[0, 1]`.

## 11. AgentFault Discipline

The grader distinguishes two failure classes:

- Agent fault: missing output, malformed submission, wrong shape, non-finite
  predictions, policy crashes, out-of-bounds actions. These should become a real
  kept `0.0` for training.
- Author/infra fault: missing hidden truth, broken scorer import, invalid private
  data, simulator crash caused by the task, dependency missing from the image.
  These should propagate as `env_internal_failure` and be discarded.

Signal agent faults by raising `grading.AgentFault` or by using helpers that
raise it for you. Do not write:

```python
try:
    ...
except Exception:
    return 0.0
```

Use this shape instead:

```python
from grading import AgentFault, helpers

def compute_score(workspace, trajectory, private):
    truth = load_truth(private)  # author data; let failures propagate

    try:
        sub = helpers.load_submission_or_fault(
            workspace / "submission.csv",
            required_columns=["id", "prediction"],
            numeric_columns=["prediction"],
            n_rows=len(truth),
        )
    except AgentFault:
        raise
    except Exception as exc:
        raise AgentFault(
            f"could not load submission: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        return score_submission(sub, truth)
    except AgentFault:
        raise
    except Exception as exc:
        raise AgentFault(
            f"could not score submitted values: {type(exc).__name__}: {exc}"
        ) from exc
```

Keep hidden-truth loading outside the agent-controlled `try`. If hidden truth is
corrupt, the task is broken; it should not become a kept zero.

## 12. Continuous Calibration

Keep the full hand-authored evaluator and compose it with
`grading.evaluation.ContinuousTask`:

1. The author declares exact versioned metric formulas, weights, perfect values,
   and reviewed floor rationales.
2. A Tier-A adapter commits and evaluates a queryable model/policy on private
   challenges; legacy Tier-B code loads submission/truth arrays explicitly.
3. `TASK.grade(...)` verifies the lock, certifies information, keeps quality
   progress only for accepted targets, aggregates it, and applies the PWL curve.

See [`docs/CONTINUOUS_EVALUATION.md`](../../docs/CONTINUOUS_EVALUATION.md) for
the full API and migration recipes.

Floor anchors are not generated from baselines. They are theoretical,
metric-bound, or domain-reviewed choices represented by:

```python
FloorAnchor(
    value=1.0,
    rationale=AnchorRationale(
        kind="theoretical",
        summary="A constant prediction at the population mean has SRE exactly one.",
    ),
)
```

Metric names are insufficient: use exact IDs such as
`sre.rmse_over_population_std.v1` or `f1.binary_threshold_0_5.v1`, whose formulas
and input contracts are serialized into the lock.

Ground truth measures the reference, computes `x_ref`, and creates the PWL knots
`(0,0)`, `(x_ref,0.5)`, `(1,1)`. The weak naive model is a qualification
witness only and cannot move floors or curve knots.

Rerun ground truth after changes to generation, hidden truth, split, target
definitions, metric formula, weights, floor/perfect/rationale, reference model,
naive model, feature semantics, dependencies, or simulator dynamics.

## 13. Baselines

Baselines are calibration evidence and a review tool. They are not the floor
anchor. Put them under:

```text
baselines/
|-- naive/
|   |-- train.py
|   |-- solution.py
|   |-- model.*
|   `-- model.manifest.json
|-- linear/
|   `-- solution.py
`-- gbt/
    `-- solution.py
```

For static dataset tasks, ship a useful diagnostic trio:

- Naive qualification model: deliberately weak but input-dependent, committed
  with a reproducible training/inference recipe. It should score low but
  positive.
- Linear/logistic: raw feature columns only, no domain feature engineering.
- Untuned GBT: XGBoost, LightGBM, sklearn gradient boosting, or similar on raw
  features only. No tuning or domain transforms.

Constant mean/median/majority strategies are null probes and should score zero;
they are not the positive naive qualification model and never define the floor.

For policy or environment-style ML tasks, ship the analogue trio:

- random-action policy;
- no-op/idle policy;
- weak-trained baseline, such as default PPO on raw observations, narrow
  behavior cloning, or a shallow heuristic using observations without the expert
  insight.

The key asymmetry is intentional: baselines stay on raw public information and
generic modeling. The reference may use domain-informed features, architecture,
adaptation, world modeling, or system identification.

Trusted validation checks naive ordering and provenance. A baseline near the
reference indicates the task is too easy or the reference is too weak; a floor
problem is diagnosed from its rationale and null probes, not by fitting the
floor to a baseline.

### PR Boreal stumping target

After you open the task PR, the automated LBx validation comment reports
**Boreal Results**: a strong hidden model runs the task 5 times (PASS@5). For ML
tasks the difficulty gate is the **maximum** score across those attempts:

```text
max(all 5 Boreal attempt scores) < 0.5   (strict)
```

A single attempt scoring `0.5` or higher fails the gate even if the average is
low — calibrate to the worst-case (max), not the average. If the max reaches
`0.5`, the task is too easy for the target SOTA model even when local validation
passes; tighten the engineering challenge fairly and rerun QA.

## 14. Reference Solution

The reference solution is the expert anchor and must be checked in under
`reference_solution/`. Commit the trained model and complete reproduction
surface:

```text
reference_solution/
|-- train.py
|-- solution.py
|-- model.*
`-- model.manifest.json
```

`solution.py` reads only public data and the committed model, then writes the
declared artifact under `/tmp/output`.

Reference expectations:

- It should be ML-based for ML tasks.
- It should demonstrate domain knowledge.
- It should be deterministic under fixed seeds.
- It should use only public files available to the agent at solve time, plus
  normal package dependencies. Do not read `data/private/`.
- It should run within the task's timeouts and resource allocation.
- It should score `0.5 +/- 0.05` for `continuous_scoring_function`.
- It should beat all weak baselines by a meaningful margin.
- It should include enough provenance for reviewers to understand what was
  trained or produced.

Trained weights are required provenance. Use Git LFS or approved immutable
storage for large artifacts, and bind them to scripts/data/config/seeds through
`model.manifest.json`. Never train from private hidden truth.

## 15. Prompt guidelines for `prompt.md`

The prompt is the only natural-language assignment the agent sees. It should
describe what to do, not how to solve it.

Include:

- the domain context at a high level;
- public file paths under `/data`;
- public data schema, row counts, array shapes, units, observation/action spaces,
  or API contracts;
- the required final artifact path under `/tmp/output`;
- exact output columns, file format, API function names, or executable contract;
- metric names used for scoring, such as SRE, RMSE, F1, AUC, success rate,
  regret, tracking error, or constraint violation rate;
- a truthful statement that ML is required and direct formula-based approaches
  alone are not competitive;
- the final sentence:

```text
For long-running training, you may use the dedicated tmux tool, not tmux inside the bash tool, or an equivalent persistent session to avoid losing work.
```

Do not include:

- floor/reference/perfect anchors;
- `X_REF`, score mapping equations, curve parameters, calibration constants, or
  any discussion of how raw metrics become final reward;
- feature-engineering hints;
- target transforms;
- architecture recommendations;
- "hints" sections;
- descriptions of private hidden cases or eval seeds;
- statements that reveal why targets behave the way they do;
- scoring code details not needed to produce the output.

There is a tension between "domain-faithful prompt" and "no hints." Resolve it
this way: give enough domain context for an expert to recognize the problem and
apply standard practice, but do not reveal the reference's specific tricks,
anchors, hidden split, or engineered transforms.

## 16. Dockerfile Rules

Metadata-mode ML uses the shared `base/task.mlenvs.Dockerfile`; authors do not
write a per-task Dockerfile. Declare additional pip/apt/grader dependencies in
`metadata.json`.

Rules:

- Keep `/workdir` and `/tmp/output` writable by uid `1000`.
- Keep `/mcp_server/data`, `/mcp_server/grader`, and
  `/mcp_server/calibration` root-owned and unreadable by
  uid `1000`.
- Do not copy private data or grader/calibration files into `/data`, `/workdir`, `/tmp/output`,
  `/app`, or `/workspace`.
- Put task-specific Python dependencies in `metadata.json:dependencies`; use
  `grading_dependencies` only for root-only grader packages.
- Prefer packages already in the base image. Add heavy dependencies only when
  the task needs them.
- Do not bake large datasets or model weights into the image. Use conventional
  `data/public/` and `data/private/` mounts plus approved model storage.

For GPU rendering or differentiable graphics, select matching resources and
base in `metadata.json`:

```json
{
  "required_resources": "12vcpu+100gib+h100/2+graphics",
  "docker-base": "cuda-graphics"
}
```

Hardware GL/EGL/Vulkan is not deployable under the Taiga gVisor sandbox. Use
CUDA-native stacks such as PyTorch3D, nvdiffrast, gsplat, nerfacc, kaolin,
MJX, Warp, or other image-base-supported libraries.

## 17. Large Data And Preloaded Files

For metadata-mode `ml` tasks, `data/public/`, `data/private/`, and the promoted
calibration lock are mounted automatically by trusted CI. Declare Hugging Face
resources in `metadata.json`:

```json
{
  "hf_resources": [
    {
      "repo_id": "org/model-name",
      "revision": "<commit-sha>",
      "repo_type": "model"
    }
  ]
}
```

Pin `revision` to a commit for reproducibility. The base image sets
`HF_HOME=/tmp/hf-cache`, so `from_pretrained` can resolve mounted weights
offline.

Do not commit `.alignerr/preloaded_files.json`; trusted CI generates the stamp at
submit time.

## 18. Hidden Environment Pattern

Set `metadata.json:ml_task_type` to `env` or `hybrid` only when the agent must
interact with a live black-box environment. Static datasets and public
simulators use `dataset`.

Minimal shape:

```json
{
  "ml_task_type": "env"
}
```

Files:

```text
data/public/env_client.py          # public RPC client
data/private/env.py                # hidden env with make_env()
data/private/env_config.json       # optional factory/config override
```

The agent connects to `/tmp/env.sock`; the env source stays root-only under
`/mcp_server/data`. Before grading, the server is stopped and the scorer loads
the held-out env in-process. Grade submitted policies with sanctioned helpers.

Hidden env is a runtime capability, not a replacement for `task_type = "ml"`.
An ML task can opt into it when interaction is part of the benchmark.

## 19. Data Generation And Provenance

For v3 continuous ML, keep deterministic synthetic-data provenance under
`data-generation/`. Reviewers must be able to audit how public and hidden data
were produced.

Required provenance:

- Include generation scripts under `data-generation/`.
- Keep scripts deterministic with fixed seeds.
- Document commands used to regenerate public and hidden fixtures.
- Record external data sources, versions, licenses, and transformations.
- Keep generated hidden targets out of public `data/`.
- Avoid committing intermediate junk, local caches, notebooks with outputs, or
  host-specific absolute paths.

For synthetic scientific data, reviewers will inspect the equations and
sampling distributions. Treat generation code as part of the scientific claim.

## 20. Quality Gates Before PR

While iterating on committed reference/naive models, use their task-local
training scripts and the reference harness without grading:

```bash
uv sync
uv run python problems/<task_id>/reference_solution/train.py
uv run python problems/<task_id>/baselines/naive/train.py
uv run lbx-rl-harness reference --problem-dir problems/<task_id> --no-grade
```

ML and accelerator tasks run the reference path **inside the task container** by
default (`task_type = "ml"` or an accelerator `required_resources` enum).
Host-only MuJoCo-style tasks run on the host.

Before opening or updating a PR:

```bash
uv sync
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
uv run lbx-rl-template validate --problem-dir problems/<task_id>
```

Commit:

```text
problems/<task_id>/calibration.lock.json
problems/<task_id>/.alignerr/build_proof.json
```

For ML tasks, `.alignerr/ground_truth/` is usually absent unless you explicitly
declare reviewer artifacts.

Then, when you want optional local agent feedback:

```bash
uv run lbx-rl-harness run --runtime claude-code --problem-dir problems/<task_id>
uv run lbx-rl-harness autoqa --problem-dir problems/<task_id> --require
```

Rerun ground truth and validation after any change to:

- `metadata.json`;
- `prompt.md` when it changes task requirements rather than prose only;
- `data-generation/`;
- `data/public/` or `data/private/`;
- `test_file.py`;
- `reference_solution/`;
- `baselines/`;
- calibration floor/perfect values or rationales.

## 21. Common Failure Modes

### The reference scores 1.0

For `continuous_scoring_function`, this is wrong. The reference should score
near `0.5`, not perfect. Rerun ground truth; never hand-edit `x_ref`.

### A naive baseline scores near reference

The task may be too easy, the split may be IID, target leakage may exist, the
reference may be weak. Review floor rationale separately; do not move the floor
to force the naive score down.

### `except Exception: return 0.0`

This fails the AgentFault discipline. Replace with typed `AgentFault` for
agent-controlled failures and let author/infra failures propagate.

### The Grader Imports `model.py` Directly

This can leak hidden truth or execute agent code as root. Use
`helpers.run_model_module`, `helpers.run_policy`, `PolicyWorker`,
`load_submitted_policy`, or another sanctioned helper.

### Hidden truth is in `data/public/`

Anything under `data/public/` is public to the agent. Move labels, eval seeds, private
cases, and grader-only fixtures to `data/private/`; only `data/public/` is
mounted for the agent in metadata-mode ML.

### The Prompt Reveals Anchors

Metric names are allowed. Floor/reference/perfect constants, curve formulas, and
weighting details should stay hidden.

### Public Data Is Too Sparse

If a domain expert cannot infer what the data represents or what the targets
measure from `prompt.md`, public schemas, and normal EDA, the prompt is too
vague or the data is not domain-faithful.

### The Task Needs GPU But Uses CPU Base

Keep an H100 `required_resources` enum and `base_flavor = "auto"` for H100
tasks. Use `base_flavor = "gpu-openroad"` only for EDA/OpenROAD tasks with a
non-graphics H100 enum. Do not hard-code production base image tags in the
Dockerfile.

### The Image Bakes Large Data

ML `data/public/` and `data/private/` are mounted by trusted CI. Extra large resources
should use `[[preloaded_files]]`, not image layers.

## 22. Review Checklist

Use this checklist before asking for review:

- [ ] The task lives under one `problems/<task_id>/` directory.
- [ ] `metadata.json` declares valid `ml_task_type`, resources, domain, and license provenance.
- [ ] `prompt.md` states public files, output format, metric names, and the
      required tmux final sentence.
- [ ] `prompt.md` does not reveal anchors, transforms, architecture hints,
      or hidden evaluation details.
- [ ] Public data is in `data/public/`; hidden truth is in `data/private/`.
- [ ] External data licensing is cleared and documented.
- [ ] `data-generation/` deterministically reproduces the committed data.
- [ ] The domain claims match the equations, simulator, labels, and citations.
- [ ] `test_file.py` keeps task-specific evaluation logic and uses sanctioned loaders.
- [ ] Agent faults raise `AgentFault`; author/infra faults propagate.
- [ ] The scorer is deterministic and returns a finite score in `[0, 1]`.
- [ ] Exact versioned metric formulas are declared.
- [ ] Every floor has a substantive `AnchorRationale` and is not copied from a baseline.
- [ ] Reference and naive trained models, scripts, configs, seeds, and manifests are committed.
- [ ] `reference_solution/solution.py` scores `0.5 +/- 0.05`.
- [ ] The weak input-dependent naive model scores low but positive; null probes score zero.
- [ ] `calibration.lock.json` is generated, canonical, and unedited.
- [ ] Large data is mounted rather than baked into image layers.
- [ ] Generated submissions and `results.txt` are not committed.
- [ ] `uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>` passes.
- [ ] `uv run lbx-rl-harness run --runtime ground-truth --calibration-check --problem-dir problems/<task_id>` passes.
- [ ] `uv run lbx-rl-template validate --problem-dir problems/<task_id>` passes.
- [ ] Generated calibration locks/proofs are not committed; trusted CI evidence is authoritative.

## 23. References In This Repo

Read these before authoring:

- `docs/AUTHORING.md`: universal task layout and PR workflow.
- `docs/MLENVS_TASKS.md`: concise metadata-mode ML authoring and calibration guide.
- `docs/GRADING.md`: scorer return shapes, calibration, and grader contract.
- `docs/REWARD_HACKING.md`: AgentFault and loader discipline.
- `docs/POLICY_ISOLATION.md`: safe policy/model execution.
- `docs/HIDDEN_ENV.md`: live hidden environment RPC pattern.
- `examples/mle-tabular-classification/`: complete continuous ML example.
- `alignerr_plugin/src/alignerr_plugin/starter_templates/ml/`: ML starter.
- `alignerr_plugin/src/alignerr_plugin/task_metadata.py`: valid task type,
  reward type, domain, and license enums.

When in doubt, copy patterns from `examples/mle-tabular-classification/` and
then strengthen the problem, data, reference, and baselines until the task is a
research-grade ML benchmark rather than a toy prediction exercise.
