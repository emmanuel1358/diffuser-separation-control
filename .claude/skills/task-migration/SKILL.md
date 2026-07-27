---
name: task-migration
description: Migrate a legacy ML RL task to the sealed evaluation format and drive it to green validation. Use when a task still ships a hand-rolled compute_score, RubricBuilder/LLMJudge, or a pre-v3 calibration lock, or when Taiga QA findings demand the sealed rubric/continuous stack.
---

# Task Migration (sealed evaluation)

Operationalizes `docs/TASK_MIGRATION.md` (bundled alongside as `TASK_MIGRATION.md`).
Every class, function, and command below is verified against the grader source
and CLI; do not substitute invented names. Scope: ML tasks — Path A (rubric) and
Path B (continuous).

## When to use

A task under `problems/<task_id>/` (or a fixture) that: uses a legacy
`compute_score(workspace, trajectory, private)`, `RubricBuilder`, `LLMJudge`, a
hand-authored FLOOR/REF/PERFECT curve, `ExponentialCurve`, or a hand-edited
`calibration.lock.json`; and/or carries Taiga QA findings (no-info floor payout,
seed/probe design, data leak) that require the sealed stack to fix.

## Step 0 — detect format and choose the path

Do **not** mix paths. Decide from config + grader shape:

- `task.toml` `[difficulty] reward_type = "multi_deterministic_rubrics"`, or a
  `scorer/compute_score.py` building rubric criteria → **Path A** (`RubricTask`).
- `reward_type = "continuous_scoring_function"` → **Path B** (`ContinuousTask`).
- The removed ML-only layout (`metadata.json` with `ml_task_type` + `test_file.py`,
  no `task.toml`) → convert the layout first (`docs/LEGACY_ML_LAYOUT.md`), then
  Path B.

All APIs import from the single package `grading.evaluation`
(`grader/src/grading/evaluation/__init__.py`); faults from `grading.faults`.

---

## Path A — rubric → `RubricTask`

Target: `scorer/compute_score.py` declares `TASK = RubricTask(...)` with a pure
`evaluate(context)`; the sealed `scorer/evaluation.plan.json` is generated.

1. **Inventory** the old grader: every artifact path, criterion, weight,
   required gate, trusted fixture, subprocess, and each `try/except` that returns
   `0`.
2. **Delete legacy scaffolding**: `RubricBuilder`, `LLMJudge`, custom weight
   normalizers, hand-built `Grade`/failure dicts, and any author-owned
   `compute_score()`. Both `RubricBuilder` and `LLMJudge` raise
   `DeprecationWarning` on construct — remove, do not keep.
3. **Declare `TASK`.** Map each file to an artifact descriptor
   (`JsonArtifact` / `TextArtifact` / `RegularFileArtifact`, from
   `artifacts.py`), each old criterion to a `RubricCriterion`, each trusted
   fixture to `TrustedJson` passed in `fixtures={...}`.
   `RubricCriterion(id, weight=1.0, description="", required=False,
   pass_threshold=0.5)` — weight must be positive; `required=True` gates the
   headline to `0.0` on failure (`rubric.py:60`, `rubric.py:409`).
4. **Move domain logic into `evaluate(context)`.** It returns a criterion→score
   mapping (or `RubricEvaluation(subscores, metadata)`); keys must match the
   declared criterion ids exactly (`rubric.py:374`). Use `RubricContext`
   operations (`rubric.py:99`): `context.candidate` (loaded artifact),
   `context.fixture(name)`, `context.candidate_operation(label, fn, *a)` /
   `context.trusted_operation(...)` for typed fault boundaries,
   `context.number/ratio/mean`, `context.run_solver(cmd, ...)`, and
   `context.policy(...)` (requires the artifact be `RegularFileArtifact`, runs it
   in a sandboxed worker — the sealed replacement for direct `PolicyWorker`).
   Raise `AgentFault` / `GraderFault` / `InfrastructureFault` (from
   `grading.faults`) — never a bare `except: return 0.0`.
5. **Reseal the plan** (writes `scorer/evaluation.plan.json` from `TASK` via
   `refresh_evaluation_plan`, `plan.py:282`):

   ```bash
   uv run lbx-rl-harness reference --problem-dir problems/<task_id>
   # or, plan only:
   uv run python scripts/write_evaluation_plan.py problems/<task_id>
   git add problems/<task_id>/scorer/evaluation.plan.json
   ```

Never hand-edit the plan; `validate` only runs `check_evaluation_plan`
(`plan.py:288`) and fails on a digest/payload mismatch.

### Before → after (rubric)

Legacy:

```python
def compute_score(workspace, trajectory, private):
    try:
        design = json.loads((workspace / "design.json").read_text())
    except Exception:
        return {"score": 0.0}
    ...  # hand weights, float casts, custom aggregation
```

Sealed:

```python
from grading.evaluation import JsonArtifact, RubricCriterion, RubricTask

def evaluate(context):
    design = context.candidate
    return {"valid_span": design["span_m"] >= 1.0}

TASK = RubricTask(
    artifact=JsonArtifact("design.json", required_keys=("span_m",)),
    criteria=(RubricCriterion("valid_span", weight=1.0, required=True),),
    evaluate=evaluate,
)
```

`RubricTask.grade(...)` is framework-owned; do not define `compute_score`.

---

## Path B — continuous → sealed evaluation

`reward_type = "continuous_scoring_function"` has **three** sealed shapes. Pick by
the agent's output artifact before writing any code:

| Output artifact | Task | Constructor |
|---|---|---|
| `submission.csv` (tabular predictions) | tabular / CSV | `ContinuousTask.static` |
| `predictor.py` queried against a private table | predictor + private table | `ContinuousTask.model` |
| `policy.py` / controller rolled out in an env | paired fresh-scenario challenge | `PolicyEvaluationTask` |
| `policy.py`, executable, or opaque artifact with the existing task metric | generated-lock compatibility | `ContinuousTask.calibrated` + workspace probes |

- If the agent submits **rows of predictions** and truth is a held-out table →
  `ContinuousTask.static` (`CsvRows`) or, preferred when a queryable predictor is
  feasible, `ContinuousTask.model` (`PythonPredictor` + `PrivateTableChallenge`).
  Follow **Path B-tabular** below.
- If the agent submits a **policy/controller that is executed** and the task can
  express candidate-vs-control paired evidence (a
  `reset/choose/observe/recommend` or `act(obs)` module, rolled out over fresh
  hidden scenarios — e.g. a MuJoCo `PolicyWorker` loop) → `PolicyEvaluationTask`.
  Follow **Path B-policy** below.
- If the existing methodology cannot be represented as a paired
  challenge, use `ContinuousTask.calibrated()` with its normal
  `measure_submission` callback and declare task-specific ready-to-measure
  `WorkspaceDegenerateProbes`. Do not force it into `PolicyEvaluationTask` or
  synthesize a generic random policy.

`PolicyEvaluationTask` is first-classed by the continuous-scoring validator
(`alignerr_plugin/src/alignerr_plugin/validators/task/validator.py:928-972`): a
`continuous_scoring_function` grader that declares `TASK = PolicyEvaluationTask(...)`
passes when it calls `TASK.grade(...)` and ships a matching sealed
`scorer/evaluation.plan.json`. Canonical example: `examples/hidden-env-bandit`.

### Path B-tabular — `ContinuousTask` sealed calibration

Target: registered metric targets + reviewed floors + `GeneratedCalibration`;
production calls `TASK.grade(...)`; the lock is schema v3
(`CALIBRATION_LOCK_SCHEMA = "3.0"`, `lock.py:24`) and generated, never edited.

1. **Register targets** with reviewed floors. Use the platform factories
   (`metrics.py`): `PopulationSRETarget.lower(name, weight=, floor=, perfect=0.0,
   prediction_column=, truth_column=)` (SRE, lower-better) or
   `BinaryF1Target.higher(name, weight=, floor=, perfect=1.0, ...)`. Each `floor`
   is a `FloorAnchor(value, rationale=AnchorRationale(kind, summary, source=None))`
   — `kind` ∈ {theoretical, metric_bound, domain_review, reviewed_exception},
   `summary` ≥ 20 chars (`metrics.py:29-72`). Floors carry human rationale; they
   are not fitted to one baseline run. Only **two** platform kernels are
   registered — `sre.rmse_over_population_std.v1` and
   `f1.binary_threshold_0_5.v1` (`metrics.py:99-115`, `_REGISTERED_KERNELS`
   `metrics.py:244-250`). Any other metric (MCC, AUC, macro-F1, …) has no
   factory: hand-construct a `MetricTarget` — see **Custom metric escape hatch**
   below.
2. **Declare `TASK`** (`author.py`). Prefer **Tier A** queryable
   `ContinuousTask.model(artifact=PythonPredictor("predictor.py"),
   challenge=PrivateTableChallenge("challenge.parquet", feature_columns=[...],
   sample_size=256), targets=[...])`. Bridge with
   `ContinuousTask.static(artifact=CsvRows("submission.csv", columns=[...]),
   targets=[...], truth_filename="test_target.parquet")` only when a queryable
   predictor is infeasible. `calibration` defaults to `GeneratedCalibration()`
   (filename fixed to `calibration.lock.json`, `author.py:145`).
3. **Wire production** to `TASK.grade(submission, truth)` (`author.py:708`) —
   which applies quality progress **and** the permutation no-info gate — or
   `TASK.compute_score(...)` which loads then calls `grade`. Do **not** call
   `TASK.score(metrics)` in production: it raises when
   `LBX_EVALUATION_PRODUCTION=1` (`author.py:852`); it is calibration-only.
4. **Curve.** Quality mapping is the framework PWL
   (`PiecewiseLinearCurve.from_reference(lock.x_ref)`, `author.py:705`). Delete
   any `ExponentialCurve` or hand `progress_*` + local curve copy.
5. **Reseal & feedback (two separable concerns).** FORMAT validation is local
   and needs **no Docker and no ground-truth bake**; the calibration-lock bake is
   INFRA and CI-side.
   - **Local / an agent can finish this:** `lbx-rl-template validate` (and
     `check`, `lint-reward-hacks`). These verify TASK shape, plan digests, and
     lint. They reach `status == "valid"` without Docker. The trusted
     calibration bundle is only demanded when
     `REQUIRE_TRUSTED_CONTINUOUS` (`REQUIRE_TRUSTED_CONTINUOUS_ENV`) is set —
     off in local runs (`validator.py:996-1003`).
   - **Deferred to trusted CI (Docker, INFRA):** the ground-truth calibration
     bake:

     ```bash
     uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
     ```

     **Prerequisite — reference/naive strategy manifests.** Both reference and
     naive ship an inference-only entrypoint plus either existing
     `model.manifest.json` v1 or `strategy.manifest.json` with explicit
     `trained_model` / `committed_artifact` kind. Trained strategies bind every
     training input; hand-authored policies do not invent training data.
     Non-tabular callbacks additionally declare deterministic workspaces under
     `baselines/degenerate/` with `WorkspaceDegenerateProbes`.

   Stop committing generated locks/proofs under `problems/**`
   (`git rm --cached problems/<task_id>/calibration.lock.json`).

#### Custom metric escape hatch (unregistered metrics)

`PopulationSRETarget` and `BinaryF1Target` are convenience factories over the
two platform-registered kernels. For any other metric there is no factory — but
you are **not** forced to `custom_static` / `custom_reviewed`. Build a
`MetricTarget` by hand and keep the normal `sealed_rescore` path (default
`ContinuousTask.static`), which retains the permutation no-info gate. A
`MetricTarget` is a plain frozen dataclass (`metrics.py:118-176`), constructed
keyword-only:

- `name: str`, `direction: "lower"|"higher"`, `weight>0`, `perfect: float`,
  `floor: FloorAnchor`, `prediction_column`, `truth_column`.
- `metric: RegisteredMetric(id, formula, input_contract)` (`metrics.py:75-96`) —
  `id` **must contain an explicit version** (`.v`, e.g. `"mcc.multiclass.v1"`);
  `formula` and `input_contract` must be non-empty. This is a documentation
  record, not a lookup key: an id outside `_REGISTERED_KERNELS` is accepted.
- `kernel: Callable[[prediction, truth], float]` — your metric. `measure`
  wraps it in `_finite`, so it must return a finite float (`metrics.py:154`).
- Direction/anchor rule (`metrics.py:141-148`): `lower` needs `floor > perfect`;
  `higher` needs `floor < perfect`.

Why this stays safe and validates (verify in source):

- **Tier stays `sealed_rescore`.** `ContinuousTask.static` defaults
  `security_tier="sealed_rescore"` and `evidence=IIDPermutationEvidence()`
  (`author.py:214,228`); a custom kernel changes neither.
- **Permutation gate is kept.** `grade` populates per-target `(prediction,
  truth)` `raw_arrays` for every standard-adapter target and runs the IID
  permutation null on `target.kernel` (`author.py:729-735,772-783`;
  `permutation.py:52-88` calls `target.measure`). The gate zeros progress a
  shuffled prediction could reach — for your custom kernel too. (Only a
  `custom_static` `raw_evaluator` collapses to scalars and skips the gate,
  `author.py:615-617` — do not use it here.)
- **Not rejected for being unregistered.** `registered_kernel_identity` returns
  `"custom:<qualname>"` for a non-registered kernel and does **not**
  raise (`metrics.py:263-277`); `is_platform_registered_target` just returns
  `False` (`metrics.py:280-284`). The continuous-scoring validator never checks
  platform registration — it only requires a sealed tier to call
  `TASK.grade`/`compute_score` and the lock digests to match
  (`validator.py:979-994`). Floor bounding (`no_info_ceiling` /
  `effective_floor`) is direction-aware and metric-agnostic
  (`metrics.py:184-202`, `lock.py:143,170`), so the no-info ceiling still clamps
  your author floor.

Real snippet — 6-class MCC (higher-is-better; no-skill line is `0.0`):

```python
from grading.evaluation import (
    AnchorRationale, ContinuousTask, CsvRows, FloorAnchor,
    GeneratedCalibration, MetricTarget, RegisteredMetric,
)

def _mcc_multiclass(prediction, truth) -> float:
    import numpy as np
    from sklearn.metrics import matthews_corrcoef
    pred = np.clip(np.rint(np.asarray(prediction, dtype=float)), 0, 5).astype(int)
    actual = np.asarray(truth, dtype=float).astype(int)
    return float(matthews_corrcoef(actual, pred))

MCC = RegisteredMetric(
    id="mcc.multiclass.v1",
    formula="matthews_corrcoef(round/clip(pred, 0..5), truth)",
    input_contract="Finite numeric prediction/truth arrays, same shape; "
                   "predictions rounded and clipped to the 0..5 class ids.",
)

TASK = ContinuousTask.static(
    artifact=CsvRows("submission.csv", columns=["sample_id", "regime"]),
    targets=[
        MetricTarget(
            name="regime",
            metric=MCC,
            direction="higher",
            weight=1.0,
            perfect=1.0,
            floor=FloorAnchor(
                0.0,
                AnchorRationale(
                    "metric_bound",
                    "Matthews correlation is 0 for any class-blind constant or "
                    "random guesser; that is the no-skill line for MCC.",
                ),
            ),
            prediction_column="regime",
            truth_column="regime_class",
            kernel=_mcc_multiclass,
        ),
    ],
    calibration=GeneratedCalibration("calibration.lock.json"),
    truth_filename="test_target.parquet",
)
```

### Before → after (continuous)

Legacy (`test_file.py`, hand FLOOR/REF/PERFECT + local SRE + curve):

```python
from grading import calibration
Y1_FLOOR, Y1_REF, Y1_PERFECT = 1.0, 0.51, 0.0
_CURVE = calibration.PiecewiseLinearCurve.from_reference(_reference_aggregate_x())

def compute_score():
    ...
    x = calibration.progress_lower_better(y1_sre, Y1_FLOOR, Y1_PERFECT)
    return {"score": _CURVE.score(x), ...}
```

Sealed:

```python
from grading.evaluation import (
    AnchorRationale, ContinuousTask, CsvRows, FloorAnchor,
    GeneratedCalibration, PopulationSRETarget,
)

TASK = ContinuousTask.static(
    artifact=CsvRows("submission.csv", columns=["sample_id", "y1"]),
    targets=[
        PopulationSRETarget.lower(
            "y1", weight=1.0,
            floor=FloorAnchor(
                1.0,
                AnchorRationale(
                    "metric_bound",
                    "Constant-mean predictor scores population SRE ~= 1.0 on the "
                    "held-out fold; that is the no-skill line.",
                ),
            ),
            prediction_column="y1", truth_column="rul_log10_cycles",
        ),
    ],
    calibration=GeneratedCalibration("calibration.lock.json"),
    truth_filename="test_target.parquet",
)

def compute_score():
    return TASK.compute_score()
```

### Path B-policy — `PolicyEvaluationTask`

Target: `scorer/compute_score.py` declares `TASK = PolicyEvaluationTask(...)` and a
`compute_score(workspace, trajectory, private)` that builds a `rollout` closure plus
trusted `controls` and delegates to `TASK.grade(...)`. The dataclass lives at
`grading/evaluation/policy.py:44`; import from `grading.evaluation`.

1. **Map the legacy scorer.** The old `PolicyWorker` rollout loop → the `rollout`
   closure. `calibration.progress_higher_better` + a hand
   `PiecewiseLinearCurve.from_reference(X_REF)` → the single `reference_quality`
   **float** (the curve is rebuilt internally by
   `PiecewiseLinearCurve.from_reference(self.reference_quality)`, `policy.py:214`;
   you pass a scalar, not a curve). Policy contract violations → `AgentFault`
   (`grading.faults`); submitted-worker crashes surface as `PolicyWorkerError`
   (`grading.policy_runner`) and are converted to `AgentFault` inside `grade`.

2. **Declare `TASK`.** Params (`policy.py:47-70`, all validated in `__post_init__`):
   - `policy_path="policy.py"`, `factory_name="load_policy"` — the submitted module
     and its factory (loaded in a sandboxed worker via `load_submitted_policy`).
   - `scenarios` — independent hidden rollouts (**≥ 8**); each scenario is one
     paired unit, not a timestep.
   - `alpha` ∈ (0,1) — paired sign-test family threshold.
   - `reference_quality` ∈ (0,1) — the mean bounded quality that maps to the
     reference score; the PWL anchor.
   - `call_timeout_s` — **defaults to 2.0s**, far too low for planning policies.
     Set it to the real per-call budget (legacy wheelslip planning used `300.0`).

3. **Write `compute_score(workspace, trajectory, private)`.** Define a
   `rollout(policy, seed) -> float` closure over the private protocol/env that
   returns one bounded quality in `[0, 1]` per scenario, and a
   `controls: Mapping[str, Callable[[int], float]]` of **trusted open-loop
   baselines** (e.g. no-torque, constant-drive) scored on the *same* seeds.
   `grade` requires ≥ 1 control (`policy.py:135`): the candidate must beat every
   control under a paired sign test (`p ≤ alpha` **and** mean difference > 0,
   `policy.py:186-201`); otherwise the score is `0.0`. This is the sealed
   replacement for a fixed-answer / no-info gate. Then `return TASK.grade(...)`.

Legacy (`PolicyWorker` scorer):

```python
def compute_score(workspace, trajectory, private):
    worker = PolicyWorker(workspace / "policy.py", timeout_s=300.0)
    total = 0.0
    for seed in EVAL_SEEDS:
        env = make_private_env(private, seed)
        total += rollout_return(worker, env)      # hand loop
    x = calibration.progress_higher_better(total / len(EVAL_SEEDS), FLOOR, PERFECT)
    return {"score": _CURVE.score(x)}              # hand FLOOR/REF/PERFECT + curve
```

Sealed:

```python
from pathlib import Path
from grading.faults import AgentFault
from grading.evaluation import PolicyEvaluationTask

TASK = PolicyEvaluationTask(
    policy_path="policy.py",
    factory_name="load_policy",
    scenarios=32,
    alpha=0.01,
    reference_quality=0.72,
    call_timeout_s=300.0,          # planning budget; NOT the 2.0s default
)

def compute_score(workspace: Path, trajectory, private: Path):
    del trajectory

    def rollout(policy, seed: int) -> float:
        env = make_private_env(private, seed)      # trusted protocol from data/private
        action = policy.act(env.observe())         # AgentFault on contract breach
        return bounded_return(env.step(action))    # one score in [0, 1]

    def no_torque(seed: int) -> float:
        return bounded_return(make_private_env(private, seed).step(ZERO))

    def constant_drive(seed: int) -> float:
        return bounded_return(make_private_env(private, seed).step(CONST))

    return TASK.grade(
        workspace=workspace,
        rollout=rollout,
        controls={"no_torque": no_torque, "constant_drive": constant_drive},
    )
```

4. **Seal the plan manually.** `refresh_evaluation_plan` /
   `lbx-rl-harness reference` / `scripts/write_evaluation_plan.py` only handle
   `RubricTask` — for any non-rubric TASK they return `not_rubric` and write
   nothing (`plan.py:187,200-212`; `write_evaluation_plan.py:23`). Seal the
   policy plan yourself with the exported atomic writer (validator compares the
   file digest to `TASK.evaluation_plan.sha256`, `validator.py:964`):

   ```python
   from pathlib import Path
   from grading.evaluation import evaluation_plan_path, write_evaluation_plan_atomic
   from scorer.compute_score import TASK   # or import by path

   problem = Path("problems/<task_id>")
   write_evaluation_plan_atomic(evaluation_plan_path(problem), TASK.evaluation_plan)
   ```

   `write_evaluation_plan_atomic` canonically serializes
   `{**TASK.evaluation_plan.to_dict(), "plan_sha256": ...}` (`plan.py:127-155`) to
   `scorer/evaluation.plan.json` (schema `continuous-evaluation-plan.v1`, tier
   `sealed_challenge`). Re-run it after changing any `TASK` param, then
   `git add` the plan. Do not hand-edit it.

---

## The validate loop (both paths)

Iterate until green:

```bash
uv sync
uv run lbx-rl-harness reference --problem-dir problems/<task_id>     # reseal plan/lock
uv run lbx-rl-template validate --problem-dir problems/<task_id>     # check-only gates
uv run lbx-rl-template check --problem-dir problems/<task_id>        # preflight + advisory lint
uv run lbx-rl-template lint-reward-hacks --problem-dir problems/<task_id>
```

`validate` prints a JSON result and exits non-zero unless `status == "valid"`
(`local_cli.py:71`). On a plan/lock digest failure, re-run `reference` (reseal) —
except for `PolicyEvaluationTask`, whose plan is **not** resealed by `reference`;
re-run the manual `write_evaluation_plan_atomic` snippet (Path B-policy step 4).
Then re-validate. `validate`/`check`/`lint-reward-hacks` are the FORMAT gates:
they run locally without Docker and are what an agent drives to green.

**Proving anchors is a separate, INFRA/Docker step** (trusted CI), not part of
reaching `status == "valid"`:

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>              # reference scores as designed
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```

For an **ML continuous** task the ground-truth bake fails before Docker when a
required reference/naive strategy has neither a valid `model.manifest.json` nor
`strategy.manifest.json`, or when a non-tabular task omits its declared probe
workspaces. Complete FORMAT validation locally; defer the anchor-proof bake to
CI if Docker is unavailable.

Rubric reference must score `1.0`; continuous reference ≈ `0.5` after
calibration; naive/constant/shuffle/malformed probes must behave (rubric
required-gate → `0.0`; continuous no-info gate zeros progress).

## QA-finding remediations relevant to migration

- **No-info floor payout** (e.g. QA "y1 SRE floor = sqrt(2) pays a constant
  predictor"): set the `FloorAnchor.value` to the true no-skill line for the
  metric (SRE ≈ 1.0). The framework additionally bounds author floors by the
  measured degenerate ceiling (`no_info_ceiling` / `effective_floor`,
  `metrics.py:184-202`) and the grade-time permutation gate
  (`ContinuousTask.grade`) zeros progress that a shuffled prediction could reach.
- **Opaque policy/custom artifact:** declare fixed/open-loop/no-op/seeded-random
  artifacts in their real output shape with `WorkspaceDegenerateProbes`.
  Every probe must measure successfully twice under the shared context.
- **Naive exactly ties no-info:** keep the exclusive weak-positive default unless
  no such baseline exists; then set `naive_score_min=0.0` and add a reviewed
  `naive_at_floor` rationale. Never use this to admit a naive worse than no-info.
- **Seed inversion / probe design** (unidentifiable episodes, disclosed seed):
  fix the private evaluation protocol / probe seed in the trusted fixture or
  `data/private/`, not in the grader; keep eval seeds root-only.
- **Data leak / world-readable truth**: held-out truth belongs under
  `data/private/` → `/mcp_server/data` (root-only), loaded via `TrustedJson`
  (rubric) or the `truth_filename`/challenge parquet (continuous); never in
  `data/public/` or agent-visible files. Challenge feature columns must not
  include target truth columns (`author.py:314`).

## Definition of done

- `lbx-rl-template validate` → `status == "valid"` (local FORMAT gate; no Docker).
- Path A: `TASK = RubricTask(...)`, no author `compute_score`,
  `scorer/evaluation.plan.json` present + committed + matching `TASK`.
- Path B-tabular: platform-factory **or** hand-built custom `MetricTarget`s (any
  is fine, all stay `sealed_rescore` + keep the permutation gate) +
  `GeneratedCalibration`, production uses `TASK.grade`, lock schema v3, no
  committed production lock under `problems/**`. Reference and naive strategies
  ship explicit model/strategy manifests; non-tabular tasks ship declared
  `baselines/degenerate/` workspaces. The ground-truth calibration bake is a
  trusted-CI/Docker step, not a local blocker.
- Path B-policy: `TASK = PolicyEvaluationTask(...)`, `compute_score` delegates to
  `TASK.grade(workspace=, rollout=, controls=)` with ≥ 1 trusted control and a
  realistic `call_timeout_s`; `scorer/evaluation.plan.json` sealed via
  `write_evaluation_plan_atomic`, committed, digest matching `TASK`.
- Reference scores as designed; adversarial probes behave.
- Legacy markers gone: no `RubricBuilder`, `LLMJudge`, `ExponentialCurve`,
  hand-edited sealed files, or `TASK.score(...)` in production.

## References

- `TASK_MIGRATION.md` (bundled) — full authoritative guide.
- `grader/src/grading/evaluation/{rubric,author,metrics,artifacts,plan,lock,policy}.py`
- `examples/hidden-env-bandit` — canonical `PolicyEvaluationTask` reference
- `docs/RUBRIC_EVALUATION.md`, `docs/CONTINUOUS_EVALUATION.md`,
  `docs/REWARD_HACKING.md`
