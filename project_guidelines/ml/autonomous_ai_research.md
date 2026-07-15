# Autonomous AI Research ML Task Authoring Guide

This guide is for authors creating ML_Envs-style tasks in the ISO template for
the Autonomous AI Research Taiga environment. It translates the older ML_Envs
task-authoring rules into the current `lbx-rl-tasks-iso-template` contract.

An ML task here is not a one-off Kaggle clone. It is a verifiable research
problem where an agent reads a prompt, uses public data or a public simulator,
writes a final artifact under `/tmp/output`, and receives a deterministic
continuous score from hidden truth. The score becomes reinforcement-learning
feedback, so the grader must be correct, deterministic, and resistant to
shortcuts.

## 1. What Changed From ML_Envs

The old ML_Envs layout used `prompt.md`, `test_file.py`, `data/public`,
`data/private`, `reference_solution/`, and `metadata.json` fields such as
`_task_type = "dataset" | "env" | "hybrid" | "sim_policy"`.

The ISO template uses a universal task layout instead:

```text
problems/<task_id>/
|-- task.toml
|-- metadata.json
|-- instruction.md
|-- environment/
|   `-- Dockerfile
|-- data/                    # public, mounted at /data
|-- scorer/
|   |-- compute_score.py      # deterministic grader
|   `-- data/                 # private, mounted at /mcp_server/data
|-- solution/
|   `-- solve.sh              # reference/oracle submission
|-- baselines/
|   `-- <baseline>/solve.sh
`-- README.md
```

Use this translation table when migrating old instructions:

| Old ML_Envs concept | ISO template concept |
| --- | --- |
| `tasks/<id>_taiga/` | `problems/<task_id>/` |
| `prompt.md` | `instruction.md` |
| `test_file.py` with `compute_score()` | `scorer/compute_score.py` with `compute_score(workspace, trajectory, private)` |
| `data/public/` | `data/`, mounted read-only at `/data` |
| `data/private/` | `scorer/data/`, mounted root-only at `/mcp_server/data` |
| `reference_solution/solution.py` | `solution/solve.sh` plus any helper files it needs |
| `reference_solution/submission.csv` | The artifact written by `solution/solve.sh` under `/tmp/output` |
| `baselines/<name>/solution.py` | `baselines/<name>/solve.sh` plus committed artifact/results if useful |
| `_task_type = "dataset"` | `[difficulty].task_type = "ml"` plus a CSV/model/HDF5/executable output |
| `_task_type = "env"` or `"hybrid"` | Usually `[environment].hidden_env = "env" | "hybrid"` if the agent must probe a hidden live environment |
| `_task_type = "sim_policy"` | Usually `task_type = "ml"` or `mujoco` with a submitted policy, depending on domain; if dynamics must be hidden, use `hidden_env` |
| `metadata.json` resources/dependencies | `task.toml [environment]`, `[[outputs]]`, and `environment/Dockerfile` |

In ISO, `task_type = "ml"` is the broad metadata category. "Dataset",
"model submission", "HDF5", "executable", "k-fold", and "policy" are
submission paradigms, not task types. Pick the paradigm by declaring
`[[outputs]]` and using the matching `grading.helpers` loader.

## 2. The Mental Model

Every strong ML task has five pieces:

1. A domain-faithful problem that actually requires machine learning.
2. Public artifacts under `data/` that are sufficient for a capable agent to
   train, adapt, probe, or infer a useful solution without seeing hidden truth.
3. Hidden fixtures under `scorer/data/` that define the held-out evaluation.
4. A deterministic scorer in `scorer/compute_score.py` that converts the
   submission into a reward in `[0, 1]`.
5. A reference solution in `solution/solve.sh` that proves the problem is
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

- `metadata.json`: set `problem_data.instance_id` to `<task_id>`.
- `task.toml`: set `[task].name = "labelbox/<task_id>"`, description, resources,
  difficulty metadata, output declarations, and timeouts.
- `instruction.md`: the agent-facing task prompt.
- `environment/Dockerfile`: task dependencies and copy paths.
- `data/`: public training data, public simulator, public schema, or public
  helper files.
- `scorer/data/`: hidden labels, hidden seeds, held-out cases, private eval
  distributions, and grader-only metadata.
- `scorer/compute_score.py`: the deterministic scorer.
- `solution/solve.sh`: the reference submission path.
- `baselines/`: weak submissions used to calibrate and prove non-triviality.

Do not edit shared template code unless the task genuinely needs a reusable
tooling improvement. Authored task work should stay inside one
`problems/<task_id>/` directory.

## 4. `task.toml` Contract For ML Tasks

Every ML task must declare enum-backed difficulty metadata:

```toml
schema_version = "1.1"

[task]
name = "labelbox/<task_id>"
description = "Concise description of the ML task."

[environment]
required_resources = "12vcpu+100gib+h100/2"
storage_mb = 50000
allow_internet = false  # locked off; validation hard-fails true

[agent]
timeout_sec = 21600

[verifier]
timeout_sec = 5400
env = []

[ground_truth]
continuous_score_epsilon = 0.05

[difficulty]
task_type = "ml"
domain = "scientific_discovery_computational_science"
reward_type = "continuous_scoring_function"
license = "CC0-1.0"

[[outputs]]
path = "/tmp/output/submission.csv"
required = true
description = "CSV predictions for the held-out test rows."
```

Important details:

- `schema_version` should remain `"1.1"`.
- `[difficulty].task_type` is `ml` for ML_Envs-style static data, model,
  executable, HDF5, k-fold, and most learned-policy tasks.
- `[difficulty].domain` must be one of the ML domains in
  `alignerr_plugin.task_metadata.DOMAINS_BY_TASK_TYPE["ml"]`.
- `[difficulty].reward_type = "continuous_scoring_function"` means the reference
  in `solution/` should score `0.5 +/- [ground_truth].continuous_score_epsilon`.
- `[difficulty].license` is required for `ml`. Use a cleared permissive SPDX id
  or `not_applicable` only when no external dataset license applies.
- Every `[[outputs]].path` must be absolute and under `/tmp/output`.
- Keep `[verifier].env = []` unless the deterministic grader needs a non-secret
  runtime setting. Do not put API keys or provider credentials into verifier env.
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
- It runs `uv run lbx-rl-harness run --runtime ground-truth --problem-dir <task>`
  and verifies the committed `.alignerr/build_proof.json`.
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
- For `ml` tasks, trusted CI packs `data/` and `scorer/data/` into read-only
  content-addressed mounts before Taiga submission, then slims them out of the
  image build context. Do not rely on large data being baked into the Docker
  image.

The practical consequence: a task that "works locally" but uses a broad
`except Exception: return 0.0`, reads agent outputs unsafely, hides public data in
the image, omits `build_proof.json`, or forgets `[[outputs]]` will fail CI.

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
`data/`, `scorer/data/`, `scorer/compute_score.py`, and the generator/provenance
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
- Public descriptions match the code. `instruction.md`, `data/column_mapping.json`,
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
|-- train.parquet
|-- test.parquet
|-- column_mapping.json
|-- public_sim.py              # optional, if agent may inspect/use it
`-- README.md                  # optional public data notes

scorer/data/
|-- test_target.parquet
|-- hidden_cases.json
|-- eval_seeds.json
`-- reference_anchors.json      # optional, if scorer reads anchors from data
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

- Put inspectable training simulators or helpers under `data/`.
- Keep hidden evaluation dynamics, hidden seeds, or private scenarios under
  `scorer/data/`.
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

The agent's final artifact must be declared in `task.toml` and written under
`/tmp/output`. Match the artifact to a sanctioned loader:

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

Then call it from `compute_score.py` with:

```python
from grading import AgentFault, helpers

def compute_score(workspace, trajectory, private):
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
    return score_predictions(pred, y_hidden)
```

For submitted executables, never trust stdout lines such as `RUBRIC_SCORE=`.
`helpers.run_submitted_executable` scrubs score-forging markers, caps output, and
runs the executable in a non-root sandbox.

## 10. Scorer Contract

`scorer/compute_score.py` must define:

```python
def compute_score(workspace, trajectory, private):
    ...
```

Where:

- `workspace` points to the final output directory, normally `/tmp/output`.
- `trajectory` is the agent transcript/tool trajectory; many ML tasks ignore it.
- `private` points to `/mcp_server/data`.

Return one of:

- a finite `float` in `[0, 1]`;
- a score dict with `score`, optional `subscores`, `weights`, and `metadata`;
- `RubricBuilder.grade().to_dict()` for deterministic rubric-style tasks.

For ML_Envs-style continuous tasks, prefer a score dict when you want diagnostic
per-target progress in the UI:

```python
return {
    "score": final,
    "subscores": {
        "rmse_progress": x_rmse,
        "f1_progress": x_f1,
    },
    "weights": {
        "rmse_progress": 0.6,
        "f1_progress": 0.4,
    },
    "metadata": {
        "raw_rmse": rmse,
        "raw_f1": f1,
        "aggregate_progress": x_agg,
    },
}
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

Use the shared calibration toolkit. Do not rederive the curve in each task.

The ISO template's ML continuous scorer uses the ML_Envs FLOOR / REF / PERFECT
method with an exponential anchor curve:

1. Convert each raw metric to progress `x_i in [0, 1]`.
2. Compute a weighted aggregate `x_agg`.
3. Map `x_agg` through `grading.calibration.ExponentialCurve.from_reference(X_REF)`.

Example:

```python
from grading import calibration

xt1 = calibration.progress_lower_better(t1_sre, floor=T1_FLOOR, perfect=0.0)
xt2 = calibration.progress_lower_better(t2_sre, floor=T2_FLOOR, perfect=0.0)
xlb = calibration.progress_higher_better(label_f1, floor=LABEL_FLOOR, perfect=1.0)

x_agg = 0.35 * xt1 + 0.35 * xt2 + 0.30 * xlb
curve = calibration.ExponentialCurve.from_reference(X_REF)
final = curve.score(x_agg)
```

Anchor meanings:

- `FLOOR`: a worst plausible raw metric, not the measured baseline unless the
  baseline truly represents the metric floor.
- `REF`: the reference solution's measured raw metric. The aggregate reference
  progress becomes `X_REF`; the final score should be `0.5`.
- `PERFECT`: the theoretical optimum, such as SRE `0.0`, F1 `1.0`, success rate
  `1.0`, regret `0.0`, or constraint violation rate `0.0`.

For lower-is-better metrics:

```python
x = calibration.progress_lower_better(value, floor=FLOOR, perfect=PERFECT)
```

For higher-is-better metrics:

```python
x = calibration.progress_higher_better(value, floor=FLOOR, perfect=PERFECT)
```

Weights should sum to `1.0`. Weight harder, more discriminative targets more
heavily. Drop or down-weight targets that every baseline and reference solve
equally well.

Recompute anchors whenever you change:

- data generation;
- hidden test cases;
- train/test split;
- target definitions;
- metrics;
- reference solution;
- baseline submissions;
- feature/schema semantics;
- simulator dynamics.

Old ML_Envs instructions may mention a hand-coded `test_file.py` or
piecewise-linear score mapping. In this ISO template, use `scorer/compute_score.py`
and `grading.calibration` unless project leadership explicitly instructs
otherwise.

## 13. Baselines

Baselines are calibration evidence and a review tool. They are not the floor
anchor. Put them under:

```text
baselines/
|-- naive/
|   `-- solve.sh
|-- linear/
|   `-- solve.sh
`-- gbt/
    `-- solve.sh
```

For static dataset tasks, ship the required trio (trusted validation emits an
advisory warning when fewer than three baselines are committed):

- Naive: mean/median for regression targets, majority class or random predictor
  for classification targets. No training beyond summary statistics.
- Linear/logistic: raw feature columns only, no domain feature engineering.
- Untuned GBT: XGBoost, LightGBM, sklearn gradient boosting, or similar on raw
  features only. No tuning or domain transforms.

For policy or environment-style ML tasks, ship the analogue trio:

- random-action policy;
- no-op/idle policy;
- weak-trained baseline, such as default PPO on raw observations, narrow
  behavior cloning, or a shallow heuristic using observations without the expert
  insight.

The key asymmetry is intentional: baselines stay on raw public information and
generic modeling. The reference may use domain-informed features, architecture,
adaptation, world modeling, or system identification.

Trusted validation includes a learnability gate for continuous tasks: committed
baselines that score near the reference indicate the task is too easy, the
reference is too weak, or the floor/anchors are miscalibrated.

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
`solution/`. The required entrypoint is:

```text
solution/solve.sh
```

It must create every required `[[outputs]]` artifact under `/tmp/output`. For a
CSV task:

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /tmp/output
python solution/solution.py --out /tmp/output/submission.csv
```

Reference expectations:

- It should be ML-based for ML tasks.
- It should demonstrate domain knowledge.
- It should be deterministic under fixed seeds.
- It should use only public files available to the agent at solve time, plus
  normal package dependencies. Do not read `scorer/data/`.
- It should run within the task's timeouts and resource allocation.
- It should score `0.5 +/- 0.05` for `continuous_scoring_function`.
- It should beat all weak baselines by a meaningful margin.
- It should include enough provenance for reviewers to understand what was
  trained or produced.

It is acceptable for `solution/solve.sh` to call helper scripts under
`solution/`. The old "one self-contained `solution.py`" rule was an ML_Envs
packaging rule; in ISO, `solve.sh` is the contract.

If you commit trained weights or precomputed reference artifacts, keep them small
or mount them through `[[preloaded_files]]` if they are large. Do not store
secrets or private hidden truth in `solution/`.

## 15. Prompt Guidelines For `instruction.md`

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

The ML starter's Dockerfile pattern should remain generic:

```dockerfile
ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=problems/<task_id>
FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workdir
RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output

RUN env -u UV_SYSTEM_PYTHON uv pip install --python /opt/lbx-runtime/.venv/bin/python --no-cache <task-deps>

COPY ${PROBLEM_DIR}/data/ /data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} +

COPY ${PROBLEM_DIR}/task.toml ${PROBLEM_DIR}/instruction.md /task/
```

Rules:

- Keep `/workdir` and `/tmp/output` writable by uid `1000`.
- Keep `/mcp_server/data` and `/mcp_server/grader` root-owned and unreadable by
  uid `1000`.
- Do not copy `scorer/` or hidden data into `/data`, `/workdir`, `/tmp/output`,
  `/app`, or `/workspace`.
- Keep `ARG BASE_IMAGE`, `ARG BASE_TAG`, and `FROM ${BASE_IMAGE}:${BASE_TAG}` so
  local harness and mothership can inject the right base.
- Add task-specific Python dependencies in the Dockerfile install line unless
  the template or project policy later moves them into another field.
- Prefer packages already in the base image. Add heavy dependencies only when
  the task needs them.
- Do not bake large datasets or model weights into the image. Use conventional
  `data/` and `scorer/data/` mounts for ML, or declare extra `[[preloaded_files]]`.

For GPU rendering or differentiable graphics, set:

```toml
[environment]
required_resources = "12vcpu+100gib+h100/2+graphics"
base_flavor = "cuda-graphics"
```

Hardware GL/EGL/Vulkan is not deployable under the Taiga gVisor sandbox. Use
CUDA-native stacks such as PyTorch3D, nvdiffrast, gsplat, nerfacc, kaolin,
MJX, Warp, or other image-base-supported libraries.

## 17. Large Data And Preloaded Files

For `ml` tasks, `data/` and `scorer/data/` are mounted automatically by trusted
CI. You usually do not need `[[preloaded_files]]`.

Use `[[preloaded_files]]` only for extra large mounts outside conventional dirs,
or for Hugging Face weights:

```toml
[[preloaded_files]]
source = "data/big_dataset"
mount_path = "/data/big_dataset"

[[preloaded_files]]
hf_repo = "org/model-name"
hf_revision = "<commit-sha>"
mount_path = "/tmp/hf-cache"
```

Pin `hf_revision` to a commit for reproducibility. The base image sets
`HF_HOME=/tmp/hf-cache`, so `from_pretrained` can resolve mounted weights
offline.

Do not commit `.alignerr/preloaded_files.json`; trusted CI generates the stamp at
submit time.

## 18. Hidden Environment Pattern

Use `[environment].hidden_env` only when the agent must interact with a live
black-box environment during the solve. Static datasets and public simulators do
not need it.

Minimal shape:

```toml
[environment]
hidden_env = "env"     # or "hybrid" if static data/ is also shipped

[[outputs]]
path = "/tmp/output/policy.py"
required = true
description = "Policy module with a load_policy() factory."
```

Files:

```text
data/env_client.py          # public RPC client
scorer/data/env.py          # hidden env with make_env()
scorer/data/env_config.json # optional factory/config override
```

The agent connects to `/tmp/env.sock`; the env source stays root-only under
`/mcp_server/data`. Before grading, the server is stopped and the scorer loads
the held-out env in-process. Grade submitted policies with sanctioned helpers.

Hidden env is a runtime capability, not a replacement for `task_type = "ml"`.
An ML task can opt into it when interaction is part of the benchmark.

## 19. Data Generation And Provenance

Unlike old ML_Envs, ISO does not require a specific `data-generation/` directory
for every task. Still, reviewers need to audit how data was produced.

Recommended provenance:

- Include generation scripts under `solution/`, `data_generation/`, or a clearly
  named task-local directory when the data is synthetic and reproducible.
- Keep scripts deterministic with fixed seeds.
- Document commands used to regenerate public and hidden fixtures.
- Record external data sources, versions, licenses, and transformations.
- Keep generated hidden targets out of public `data/`.
- Avoid committing intermediate junk, local caches, notebooks with outputs, or
  host-specific absolute paths.

For synthetic scientific data, reviewers will inspect the equations and
sampling distributions. Treat generation code as part of the scientific claim.

## 20. Quality Gates Before PR

While iterating on `solution/solve.sh` (training, calibration, baselines), use
the reference harness. It runs solve + grade with a persistent cache and skips
build-proof refresh:

```bash
uv sync
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness reference --problem-dir problems/<task_id> --skip-solve
```

ML and accelerator tasks run the reference path **inside the task container** by
default (`task_type = "ml"` or an accelerator `required_resources` enum).
Host-only MuJoCo-style tasks run on the host. Override with
`[reference].execution = "host"|"container"` in `task.toml` when needed.

Before opening or updating a PR:

```bash
uv sync
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
uv run lbx-rl-template validate --problem-dir problems/<task_id>
```

Commit:

```text
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

- `task.toml`;
- `metadata.json`;
- `instruction.md`;
- `environment/Dockerfile`;
- `data/`;
- `scorer/`;
- `solution/`;
- `baselines/`;
- calibration anchors.

## 21. Common Failure Modes

### The Reference Scores 1.0

For `continuous_scoring_function`, this is wrong. The reference should score
near `0.5`, not perfect. Recalibrate `X_REF` from reference aggregate progress.

### A Naive Baseline Scores Near Reference

The task may be too easy, the split may be IID, target leakage may exist, the
reference may be weak, or the floor may be miscalibrated. Redesign before PR.

### `except Exception: return 0.0`

This fails the AgentFault discipline. Replace with typed `AgentFault` for
agent-controlled failures and let author/infra failures propagate.

### The Grader Imports `model.py` Directly

This can leak hidden truth or execute agent code as root. Use
`helpers.run_model_module`, `helpers.run_policy`, `PolicyWorker`,
`load_submitted_policy`, or another sanctioned helper.

### Hidden Truth Is In `/data`

Anything under `data/` is public to the agent. Move labels, eval seeds, private
cases, and anchors to `scorer/data/`.

### The Prompt Reveals Anchors

Metric names are allowed. Floor/reference/perfect constants, curve formulas, and
weighting details should stay hidden.

### Public Data Is Too Sparse

If a domain expert cannot infer what the data represents or what the targets
measure from `instruction.md`, public schemas, and normal EDA, the prompt is too
vague or the data is not domain-faithful.

### The Task Needs GPU But Uses CPU Base

Keep an H100 `required_resources` enum and `base_flavor = "auto"` for H100
tasks. Use `base_flavor = "gpu-openroad"` only for EDA/OpenROAD tasks with a
non-graphics H100 enum. Do not hard-code production base image tags in the
Dockerfile.

### The Image Bakes Large Data

ML `data/` and `scorer/data/` are mounted by trusted CI. Extra large resources
should use `[[preloaded_files]]`, not image layers.

## 22. Review Checklist

Use this checklist before asking for review:

- [ ] The task lives under one `problems/<task_id>/` directory.
- [ ] `metadata.json` instance id matches the directory.
- [ ] `task.toml [task].name` matches `labelbox/<task_id>`.
- [ ] `[difficulty].task_type = "ml"`.
- [ ] `[difficulty].domain` is a valid ML domain enum.
- [ ] `[difficulty].reward_type = "continuous_scoring_function"` for ML_Envs-style scoring.
- [ ] `[difficulty].license` is permissive or deliberately `not_applicable`.
- [ ] Every output is declared under `/tmp/output`.
- [ ] `instruction.md` states public files, output format, metric names, and the
      required tmux final sentence.
- [ ] `instruction.md` does not reveal anchors, transforms, architecture hints,
      or hidden evaluation details.
- [ ] Public data is in `data/`; hidden truth is in `scorer/data/`.
- [ ] External data licensing is cleared and documented.
- [ ] Data generation is reproducible or provenance is documented.
- [ ] The domain claims match the equations, simulator, labels, and citations.
- [ ] `scorer/compute_score.py` uses sanctioned loaders.
- [ ] Agent faults raise `AgentFault`; author/infra faults propagate.
- [ ] The scorer is deterministic and returns a finite score in `[0, 1]`.
- [ ] Calibration uses `grading.calibration`.
- [ ] The reference in `solution/solve.sh` scores `0.5 +/- 0.05`.
- [ ] Baselines exist and score clearly below the reference.
- [ ] The Dockerfile preserves private root permissions and output writability.
- [ ] Large data is mounted rather than baked into image layers.
- [ ] `uv run lbx-rl-harness reference --problem-dir problems/<task_id>` reaches the expected reference score during development.
- [ ] `uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>` passes.
- [ ] `uv run lbx-rl-template validate --problem-dir problems/<task_id>` passes.
- [ ] `.alignerr/build_proof.json` is committed and fresh.

## 23. References In This Repo

Read these before authoring:

- `docs/AUTHORING.md`: universal task layout and PR workflow.
- `docs/MLENVS_TASKS.md`: concise ML_Envs migration guide.
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
