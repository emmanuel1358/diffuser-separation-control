# Continuous Evaluation API

This guide is the source of truth for continuous-scoring graders, including
new sealed-challenge tasks and migrations from older ML_Envs or static
`submission.csv` graders.

This guide applies only to `continuous_scoring_function`. Deterministic rubric
tasks use the mandatory [`RubricTask` protocol](RUBRIC_EVALUATION.md), which
reuses artifact commitments, plans, attestation, traces, and fault semantics
without continuous calibration or permutation evidence.

## Why the API exists

A continuous metric alone can reward feature-blind constants, tiny perturbations
of constants, row-index vectors, marginal shuffles, or open-loop policies. The
evaluation API separates:

1. **Quality:** the task metric mapped through reviewed
   FLOOR/REFERENCE/PERFECT anchors.
2. **Information evidence:** whether the output carries task-relevant signal
   under a declared challenge protocol.

A target that passes evidence keeps its ordinary quality progress. A target that
fails receives exactly zero. P-values, action variance, and confidence are never
used as reward multipliers.

## Security tiers

### Tier A: `sealed_challenge`

Use for every new continuous task when possible.

- The agent submits a queryable model, executable predictor, or policy.
- The runtime commits the artifact before selecting hidden rows/scenarios.
- Production injects a secret attempt nonce; the stored commitment makes a
  recorded attempt reproducible.
- Metric kernels, evidence decisions, weights, and final PWL aggregation are
  framework-owned.

After verifying the promoted plan, production runners generate a private
cryptographic `LBX_EVALUATION_NONCE` after artifact commitment. The nonce is
scrubbed from submitted workers and stored only in the private replay trace.
Local runs without an attested plan remain deterministic and the public receipt
reports `attested: false`.

### Tier B: `sealed_rescore`

Compatibility for existing static prediction artifacts.

- `ContinuousTask.static()` or `ContinuousTask.calibrated()`
- `TASK.grade(submission, truth)` is the protected production entry.
- IID permutation evidence is available only when rows are genuinely
  exchangeable.

Tier B cannot establish prediction provenance or prevent unlimited adaptive
queries against one fixed holdout.

### Tier C: `custom_reviewed`

Use only when no valid Tier-A/Tier-B protocol applies. It requires explicit
review and must not claim sealed protection.

## Registered quality metrics

Targets combine a versioned metric implementation with stable quality anchors:

```python
from grading.evaluation import (
    AnchorRationale,
    FloorAnchor,
    PopulationSRETarget,
)

target = PopulationSRETarget.lower(
    "temperature",
    prediction_column="temperature",
    truth_column="temperature",
    weight=1.0,
    floor=FloorAnchor(
        value=1.0,
        rationale=AnchorRationale(
            kind="theoretical",
            summary=(
                "A held-out population-mean constant has standardized RMSE one."
            ),
        ),
    ),
    perfect=0.0,
)
```

Built-ins currently include:

- `sre.rmse_over_population_std.v1`
- `f1.binary_threshold_0_5.v1`

Tier-A targets must resolve to platform-owned kernels. A descriptive
`RegisteredMetric` paired with an arbitrary callback is not a Tier-A metric.

The reviewed floor is a quality decision; do not copy one baseline score into
it. Generated degenerate-family measurements are audit/qualification evidence,
not a replacement for the reviewed quality floor.

## Tier-A queryable tabular model

### Grader registration

```python
from grading.evaluation import (
    ContinuousTask,
    GeneratedCalibration,
    PrivateTableChallenge,
    PythonPredictor,
)

TASK = ContinuousTask.model(
    artifact=PythonPredictor(
        "predictor.py",
        factory_name="load_predictor",
        method="predict",
    ),
    challenge=PrivateTableChallenge(
        "challenge.parquet",
        feature_columns=["x1", "x2", "x3"],
        sample_size=400,
    ),
    targets=[target],
    calibration=GeneratedCalibration("calibration.lock.json"),
    naive="baselines/naive",
)


def compute_score():
    return TASK.compute_score()
```

`data/private/challenge.parquet` is root-only and contains the declared feature
columns plus every target truth column. It must have more rows than
`sample_size`. The runtime:

1. validates and hashes `predictor.py`;
2. derives a private challenge selection;
3. calls the predictor in the submitted-policy sandbox;
4. validates target shapes/finiteness;
5. evaluates quality and family-wide information evidence;
6. returns a calibrated score and redacted receipt.

### Agent artifact contract

```python
def load_predictor():
    class Predictor:
        def predict(self, rows):
            return {
                "temperature": [model(row) for row in rows],
            }

    return Predictor()
```

`rows` is a list of feature dictionaries. The result is either a mapping from
target columns to equally sized lists, or a list of row mappings. Additional
model files may be placed beside `predictor.py`.

Predictors must be deterministic for the same rows and must not require network
access.

## Tier-A policy challenge

Use `PolicyEvaluationTask` when the grader can produce fresh hidden scenarios.

```python
from grading.evaluation import PolicyEvaluationTask

TASK = PolicyEvaluationTask(
    policy_path="policy.py",
    factory_name="load_policy",
    scenarios=32,
    alpha=0.01,
    reference_quality=0.72,
)


def compute_score(workspace, trajectory, private):
    del trajectory

    def rollout(policy, seed):
        env = make_private_env(seed)
        # Query the policy and return one bounded score in [0, 1].
        ...

    return TASK.grade(
        workspace=workspace,
        rollout=rollout,
        controls={
            "fixed_action": fixed_action_quality,
            "open_loop": open_loop_quality,
        },
    )
```

Candidate and trusted controls run on identical secret scenario seeds. The
candidate must beat every control under paired return evidence. Candidate worker
failures become kept zeros; environment/grader failures propagate as internal
failures.

Policy tasks must:

- use scenarios—not timesteps—as independent units;
- floor every failed episode rather than skipping it;
- impose hard worker-call deadlines;
- include all task-relevant fixed/open-loop controls;
- reject a task whose reference cannot reliably beat those controls.

See `examples/hidden-env-bandit`.

## Tier-B static migration

Existing tabular tasks may retain their loader:

```python
TASK = ContinuousTask.calibrated(targets=[...])


def _load():
    truth = ...
    submission = ...
    return submission, truth


def measure_submission(workspace, private):
    submission, truth = _load(workspace, private)
    return TASK.measure_registered(submission, truth)


def compute_score():
    submission, truth = _load()
    return TASK.grade(submission, truth)
```

Do **not** use `TASK.score(measure_submission())` in production. `TASK.score`
is scalar-only and exists for calibration/reference tooling. Production workers
reject it.

`custom_static()` is Tier C because arbitrary callbacks do not expose trusted
per-target evidence units.

## Calibration lock v3

`calibration.lock.json` is generated, never hand-edited. Schema v3 binds:

- TASK and evaluation-plan digests;
- registered metric and kernel identities;
- reviewed floors and perfect anchors;
- reference, naive, and generated degenerate measurements;
- challenge/data/model/image input roots;
- PWL reference aggregate and qualification results.

The v3 lock preserves reviewed floors for quality. Degenerate measurements are
used to qualify/audit no-information behavior; grade-time evidence determines
eligibility.

For local score feedback after changing data, metrics, targets, models,
challenge protocol, or grader:

```bash
uv run lbx-rl-harness run \
  --runtime ground-truth \
  --problem-dir problems/<task_id>
```

This writes a development `calibration.lock.json` and
`.alignerr/calibration.evidence.json`. They are local cache state for
`problems/**` and must not be hand-edited or committed. Commit the reproducible
reference/naive models, manifests, data provenance, and task source.

For a task that tracked generated evidence under the previous workflow, remove
it from Git once; local files remain available and are ignored afterward:

```bash
git rm --cached problems/<task_id>/calibration.lock.json \
  problems/<task_id>/.alignerr/build_proof.json
```

Trusted CI is authoritative:

1. It computes a cache key from the immutable task specification, grader, data,
   runtime configuration, reference/naive strategies, and trusted framework
   revision.
2. It restores an exact previously trusted bundle or runs ground truth to
   generate a fresh canonical lock and evidence file.
3. The separate Taiga submission job downloads that trusted artifact, verifies
   its full digest, and uploads it as a content-addressed read-only mount at
   `/mcp_server/calibration`.
4. Taiga rejects continuous grading when the expected lock/plan evidence or
   trusted mount is absent or mismatched.

Thus authors run local calibration only when they want local feedback; every
submitted revision receives CI-generated production evidence. A v2 lock must
still be regenerated rather than repaired manually.

## Public receipt and private trace

`reward-details.json` includes only a redacted
`metadata.evaluation` receipt:

- protocol and plan identity;
- seed commitment;
- attestation status;
- challenge count and family alpha;
- per-target/control accepted status and coarse reason.

Exact raw metrics, null statistics, p-values, floors, scenario returns, seeds,
and exceptions are written only to a root-owned `evaluation-details.json`.
Taiga persists it after the agent phase under
`/tmp/output/.lbx-evaluation/` for authorized replay/debugging; it is never
included in public reward metadata or exposed to submitted workers.

## Failure semantics

- Missing/malformed artifact, bad predictor output, or submitted-worker crash:
  `AgentFault`, score `0`, rollout kept.
- Valid but insufficiently informative output: score `0`, rollout kept.
- Missing/stale lock, invalid plan, metric bug, private challenge failure, or
  signature/runtime mismatch: internal failure, rollout discarded.
- A protocol error must never fall back to unguarded progress.

## Migrating prior versions

For a single authoritative checklist that also covers rubric migrations and
Trusted CI seal/promote behavior, see [`TASK_MIGRATION.md`](TASK_MIGRATION.md).

### Hand-written FLOOR/REF/PERFECT grader

1. Replace local metric/curve copies with registered targets and
   `GeneratedCalibration`.
2. Commit reproducible reference and weak input-dependent naive models.
3. Add `TASK`.
4. Choose Tier A where a queryable artifact is possible; otherwise use the
   temporary Tier-B `TASK.grade` shape.
5. Run ground truth to generate lock v3.

### Calibration-lock v2 task

1. Keep existing target names, weights, and reviewed `FloorAnchor` rationales.
2. Replace `TASK.score(measure_submission())` with `TASK.grade(...)`.
3. Prefer `ContinuousTask.model()` and a private challenge bank.
4. Regenerate local development calibration for feedback; trusted CI generates
   the production lock/evidence bundle. V2 and v3 are intentionally not
   interchangeable.

### Static `submission.csv` to queryable predictor

1. Move held-out challenge features and truth to one private table.
2. Change the prompt/output contract to `predictor.py`.
3. Change reference/naive inference scripts to package queryable predictors.
4. Replace `ContinuousTask.calibrated()` with `ContinuousTask.model()`.
5. Retain metric targets and quality anchors.
6. Regenerate calibration and run the adversarial tests.

### Fixed-decision hidden policy to fresh challenges

1. Ask for an adaptive policy algorithm, not one baked answer.
2. Randomize private scenarios after artifact commitment.
3. Define bounded per-scenario quality and trusted fixed/open-loop controls.
4. Use `PolicyEvaluationTask`.
5. Verify reference near `0.5`, controls/constant policies at `0`, and crashes
   as kept zeros.

## Required reward-hacking tests

Every task family must test through the real grader:

- exact constants and tuned per-target constants;
- relative perturbation ladder around constants;
- marginal shuffles and row-index/nonlinear-index outputs;
- partial-target and decoy-field attacks;
- malformed/non-finite/oversized outputs;
- adaptive replay across attempt nonces;
- model call-position and duplicate-input attacks where applicable;
- fixed/open-loop/perturbed/crashing policies for policy tasks;
- reference unchanged and weak informative candidate still positive.

Run at minimum:

```bash
uv run pytest grader/tests harness/tests
uv run lbx-rl-template validate --problem-dir problems/<task_id>
uv run lbx-rl-harness run --runtime ground-truth \
  --problem-dir problems/<task_id>
```

## Security limits

No finite grader can perfectly separate every arbitrarily weak learned model
from every no-information strategy on one fixed holdout. Guarantees depend on
declared exchangeability/generative assumptions, effective independent units,
false-pass budget, power target, fresh challenge capacity, and bounded adaptive
queries. When those assumptions do not hold, use a reviewed custom protocol
rather than weakening or mislabeling the tier.
