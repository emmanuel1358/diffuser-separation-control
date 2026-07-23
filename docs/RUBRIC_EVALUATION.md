# Declarative Rubric Evaluation

`multi_deterministic_rubrics` tasks use a mandatory `RubricTask` registration.
The shared grader owns artifact I/O, malformed-input handling, numeric safety,
criterion aggregation, subprocess timeouts, fault attribution, result
serialization, plan identity, and private replay traces.

Authors declare the contract and write pure domain evaluation. Do not define a
task-owned `compute_score()` for a rubric task.

## Minimal grader

```python
from grading.evaluation import (
    JsonArtifact,
    NumericField,
    RubricCriterion,
    RubricTask,
)


def evaluate(context):
    design = context.candidate
    return {
        "valid_span": design["span_m"] >= 1.0,
        "efficiency": context.ratio(
            design["useful_work"],
            design["input_work"],
            label="efficiency",
            zero="agent_fault",
        ),
    }


TASK = RubricTask(
    artifact=JsonArtifact(
        "design.json",
        required_keys=("span_m", "useful_work", "input_work"),
        numeric_fields=(
            NumericField("span_m", minimum=0.0, maximum=100.0),
            NumericField("useful_work", minimum=0.0),
            NumericField("input_work", minimum=0.0),
        ),
        allow_extra_keys=False,
    ),
    criteria=(
        RubricCriterion(
            "valid_span",
            weight=1.0,
            description="Span satisfies the declared physical range",
            required=True,
        ),
        RubricCriterion(
            "efficiency",
            weight=2.0,
            description="Useful work divided by input work",
        ),
    ),
    evaluate=evaluate,
)
```

The sealed file `scorer/evaluation.plan.json` is **generated, never
hand-edited** (same model as continuous `calibration.lock.json`).

**Who writes vs who checks** (call-site explicit; no CI env sniffing):

| Caller | API | Behavior |
|--------|-----|----------|
| `lbx-rl-harness reference` / `ground-truth` | `refresh_evaluation_plan` | writes from `TASK` |
| `lbx-rl-template new` / `scripts/write_evaluation_plan.py` | `refresh_evaluation_plan` | writes from `TASK` |
| Trusted CI (mothership) | `refresh_evaluation_plan` | explicit seal step before validate |
| `lbx-rl-template validate` / `check` | `check_evaluation_plan` | verify-only |

Commit the generated file after local reference/ground-truth. Trusted CI
reseals from `TASK` in the grade workspace (mirroring continuous calibration
staging) so a forgotten commit does not invent plans via ambient `CI=1`
heuristics.

The writer serializes:

```python
payload = {
    **TASK.evaluation_plan.to_dict(),
    "plan_sha256": TASK.evaluation_plan.sha256,
}
```

Validate and Taiga verify this plan against the image-baked registration.

## Shared ownership boundary

Rubric task code may:

- declare artifact and trusted-fixture schemas;
- declare criterion IDs, descriptions, weights, required gates, and thresholds;
- compute deterministic domain quantities from `context.candidate` and declared
  fixtures;
- call `context.candidate_operation(...)` for candidate-dependent parsers or
  simulators;
- call `context.trusted_operation(...)` for trusted model/configuration work;
- call `context.policy(...)` for agent Python;
- call `context.run_solver(...)` for bounded trusted executables;
- return a criterion mapping or `RubricEvaluation`.

Rubric task code must not:

- open, decode, or parse the agent artifact itself;
- use `json.loads`, `Path.read_text`, `open`, pickle, or raw dataframe readers on
  `/tmp/output`;
- call `float`/`int` directly on candidate values;
- divide or average without an explicit empty/zero policy;
- normalize weights or aggregate the headline;
- launch `subprocess` directly;
- catch exceptions to return a score;
- construct `env_internal_failure` or failure payloads;
- define its own `compute_score()`.

The validator hard-blocks legacy rubric graders and stale/missing plans.

## Artifact descriptors

### `JsonArtifact`

`JsonArtifact` opens with `O_NOFOLLOW | O_NONBLOCK`, validates a regular file,
caps bytes before reading, decodes strict UTF-8, bounds JSON depth/node count,
checks object shape, and converts declared numeric fields through finite,
overflow-safe coercion.

All submission-content failures become `AgentFault`, including:

- missing, empty, directory, FIFO, device, or symlink artifacts;
- invalid UTF-8;
- malformed or deeply recursive JSON;
- oversized JSON;
- wrong top-level type;
- missing or unexpected keys;
- booleans where numbers are required;
- huge integers that overflow `float`;
- NaN/Inf and out-of-range numbers.

Nested numeric fields use dotted paths:

```python
JsonArtifact(
    "isolation_design.json",
    required_keys=("isolation_system",),
    numeric_fields=(
        NumericField("isolation_system.Qd_kip", minimum=80, maximum=650),
        NumericField("isolation_system.Kd_kip_per_in", minimum=10, maximum=90),
    ),
    allow_extra_keys=False,
)
```

### `TextArtifact`

Use for XML or other bounded UTF-8 source. The framework owns the secure read;
compile it through a candidate boundary:

```python
model = context.candidate_operation(
    "MJCF compilation",
    mujoco.MjModel.from_xml_string,
    context.candidate,
)
```

### `RegularFileArtifact`

Use for submitted Python policies/models handled by an existing sandbox:

```python
with context.policy(timeout_s=2.0) as policy:
    score = rollout(policy)
```

The artifact is checked for regular-file and size constraints before the
privilege-dropped worker opens it.

### `TrustedJson`

Declare root-only fixtures instead of opening private paths in evaluator code:

```python
TASK = RubricTask(
    ...,
    fixtures={"cases": TrustedJson("hidden_cases.json")},
)

def evaluate(context):
    cases = context.fixture("cases")
```

A missing or malformed trusted fixture is a `GraderFault`, never an agent zero.

## Numeric safety

`context.number`, `context.ratio`, and `context.mean` centralize finite checks.
Every empty or zero denominator needs an explicit policy.

```python
context.ratio(a, b, label="coverage", zero="zero")
context.ratio(a, b, label="physical ratio", zero="agent_fault")
context.ratio(a, b, label="trusted normalization", zero="grader_fault")

context.mean(values, label="case quality", empty="zero")
context.mean(values, label="required cases", empty="grader_fault")
```

Criterion returns accept booleans or finite numbers and are clamped to `[0, 1]`.
Weights are validated as finite, strictly positive, unique by criterion ID, and
normalized by the framework.

## Domain operation boundaries

Use `candidate_operation` when malformed but schema-valid candidate content can
make a trusted parser reject:

```python
geometry = context.candidate_operation(
    "geometry compilation",
    compile_geometry,
    context.candidate,
)
```

Any ordinary exception becomes `AgentFault` and a kept zero.

Use `trusted_operation` when failure means the grader, dependency, or private
configuration is broken:

```python
response = context.trusted_operation(
    "response-history analysis",
    model.evaluate,
    context.candidate,
    context.fixture("cases"),
)
```

Any ordinary exception becomes `GraderFault` and the rollout is discarded.

Prefer a structured result for expected candidate nonconvergence instead of
raising. The evaluator can map that result to criterion zero.

## Bounded solver execution

Never call `subprocess.run` from rubric code. Use:

```python
result = context.run_solver(
    ["blockMesh"],
    cwd=case_dir,
    timeout_s=30,
    max_output_bytes=4 * 1024 * 1024,
)

mesh_score = 1.0 if result.ok else 0.0
```

The shared runner:

- creates a separate process group;
- kills the group on timeout;
- caps combined stdout/stderr while streaming;
- returns typed nonzero/timeout/output-limit outcomes;
- raises `GraderFault` for launch/environment failures.

No `TimeoutExpired` escapes into the grader process.

## Fault and episode semantics

- `AgentFault`: score `0.0`, `env_internal_failure=False`, episode remains valid
  training signal.
- `GraderFault` or `InfrastructureFault`: score `0.0`,
  `env_internal_failure=True`, episode is discarded.
- unclassified exception inside declarative candidate evaluation: score `0.0`,
  `env_internal_failure=False`, `critical_operator_alert=True`. This prevents a
  free episode/group veto; trusted CI must fail the adversarial probe and force
  the author to classify/fix it.
- process signal/OOM/platform timeout: `env_internal_failure=True`.
- import, plan, attestation, private fixture, or normalization failure:
  `env_internal_failure=True`.

Harbor and Taiga/Boreal consume the same typed result semantics.

## Criteria, required gates, and metadata

`RubricCriterion(required=True)` gates the headline to zero when its score is
below `pass_threshold`. Non-required criteria contribute their normalized
weight.

Return `RubricEvaluation` to include non-sensitive diagnostics:

```python
return RubricEvaluation(
    subscores={"mesh": mesh_score, "quality": quality},
    metadata={"case_count": len(cases)},
)
```

Do not put hidden cases, seeds, thresholds, answer keys, or exact private
statistics in public metadata. The framework writes exact criterion decisions
to a root-only evaluation trace.

## Attestation and replay

The task spec hashes:

- artifact schema and resource limits;
- criterion IDs, weights, descriptions, gates, and thresholds;
- trusted fixture declarations;
- scoring mode and security tier.

Production combines the spec hash, full workspace artifact digest, and
post-commit nonce. Public metadata contains only the plan/spec identity,
commitment, attestation status, and redacted decisions. The private trace
contains the replay nonce, artifact digest, exact criterion scores, and weights.

## Required adversarial probes

The validator automatically checks each shipped rubric artifact boundary with
missing/empty, invalid UTF-8 where applicable, malformed/deep JSON, and symlink
inputs. Task tests must additionally cover:

- huge integer and non-finite numeric values;
- every required key/type/range boundary;
- zero denominator and empty case list policies;
- candidate parser rejection;
- policy crash/timeout;
- solver nonzero/timeout/output flood;
- missing private fixture/dependency as a grader fault;
- no-op and naive submissions below `max_trivial_score`;
- deterministic repeatability;
- reference score exactly `1.0`.

## Migration

For the full checklist across continuous and rubric families (including sealed
plans, calibration lock v3, and Trusted CI promotion), see
[`TASK_MIGRATION.md`](TASK_MIGRATION.md).

Legacy:

```python
def compute_score(workspace, trajectory, private):
    try:
        design = json.loads((workspace / "design.json").read_text())
    except Exception:
        return {"score": 0.0}
    ...
```

Declarative:

```python
def evaluate(context):
    design = context.candidate
    return {"criterion": domain_check(design)}


TASK = RubricTask(
    artifact=JsonArtifact(...),
    criteria=(RubricCriterion("criterion"),),
    evaluate=evaluate,
)
```

Remove all author-owned loader, `safe_float`, denominator, weight normalization,
failure-payload, and exception-handling utilities. Let harness
reference/ground-truth refresh and commit matching
`scorer/evaluation.plan.json` (`refresh_evaluation_plan`), run task validation
(`check_evaluation_plan` only), run the reference and malformed-input tests,
then regenerate the ground-truth proof.

Canonical migrations are under:

- `examples/mujoco-pendulum/`
- `examples/opensees-base-isolation/`
- `examples/openfoam-hydrofoil-flap/`
