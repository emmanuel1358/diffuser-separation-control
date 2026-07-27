# Migrating Existing Tasks to the Current Evaluation Stack

This is the authoritative migration guide for tasks authored before the sealed
evaluation and declarative grading work landed in the ISO template. Use it when
an older `problems/<task_id>/` (or a forked example) still relies on hand-rolled
loaders, manual anchors, `RubricBuilder`, `LLMJudge`, or a pre-v3 calibration
lock.

Deep API references remain elsewhere:

| Topic | Source of truth |
| --- | --- |
| Deterministic rubrics | [`RUBRIC_EVALUATION.md`](RUBRIC_EVALUATION.md) |
| Continuous / sealed challenges | [`CONTINUOUS_EVALUATION.md`](CONTINUOUS_EVALUATION.md) |
| Removed ML-only layout conversions | [`LEGACY_ML_LAYOUT.md`](LEGACY_ML_LAYOUT.md) |
| Rubric design principles | [`RUBRIC_GUIDANCE.md`](RUBRIC_GUIDANCE.md) |
| Reward-hacking expectations | [`REWARD_HACKING.md`](REWARD_HACKING.md) |

This document answers **what changed**, **which path you are on**, and **the
minimum sequence to become current**.

## What recently shipped

Approximate template history (ISO template PRs):

| Change | Why it matters for migrations |
| --- | --- |
| **#107** — framework continuous calibration | Replaces fragile hand-authored FLOOR/REF/PERFECT wiring with generated `calibration.lock.json`, reviewed floors, and one-command finalization. |
| **#110 / #109** — no-info / permutation gates | Grade-time evidence so constant and shuffle attacks cannot coast on quality alone. |
| **#111** — sealed continuous evaluation | Prefer queryable post-commit challenges; trusted CI owns production lock promotion to Taiga. |
| **#115** — declarative `RubricTask` | Mandatory for `multi_deterministic_rubrics`; shared artifact I/O, faults, aggregation, and sealed `evaluation.plan.json`. |
| Trusted CI seal + submit overlay (mothership) | Sandbox reseals plans/locks and promotes them into the Taiga image so local git and production cannot silently diverge. |

If your task still works under `lbx-rl-template validate` only because legacy
shapes are tolerated, treat that as **debt**: production hardening and Trusted
CI increasingly reject unprotected continuous graders and non-declarative
rubrics.

## Choose your path

```text
task.toml scoring.mode / return shape
        |
        +-- multi_deterministic_rubrics  -->  Path A: RubricTask + evaluation.plan.json
        |
        +-- continuous_scoring_function  -->  Path B: ContinuousTask + calibration.lock.json
        |
        +-- removed ML-only layout        -->  Convert layout first (LEGACY_ML_LAYOUT.md),
                                              then Path B (or A if you convert to rubrics)
```

Do **not** mix paths. A rubric task must not keep a production
`compute_score()` that bypasses `TASK`. A continuous task must not invent a
second hand-written PWL curve beside `GeneratedCalibration`.

---

## Path A — Deterministic rubrics → `RubricTask`

### Target state

- `scorer/compute_score.py` declares `TASK = RubricTask(...)` and a pure
  `evaluate(context)` (or equivalent domain helpers).
- **No** author-owned `compute_score()` for production grading.
- Artifact loading uses descriptors (`JsonArtifact`, `TextArtifact`,
  `RegularFileArtifact`, …), not `open` / `json.loads` on `/tmp/output`.
- Numeric work goes through `context.ratio` / `NumericField` / grading numeric
  helpers — not bare `float`/`/` with silent `except`.
- Faults use `AgentFault` / `GraderFault` / `InfrastructureFault` semantics
  owned by the framework.
- Sealed file: **`scorer/evaluation.plan.json`** (generated, never hand-edited).

### Why the sealed plan exists

Same trust model as continuous locks:

1. Local `reference` / `ground-truth` / scaffold calls `refresh_evaluation_plan`
   and writes the plan from `TASK`.
2. `lbx-rl-template validate` only **checks** (`check_evaluation_plan`).
3. Trusted CI reseals from `TASK` in the sandbox, stages the plan into the
   grade-result artifact, and overlays it onto the submit checkout before
   Taiga export/image bake.

Sandbox validation and the image that Taiga grades therefore agree on criterion
IDs, weights, required gates, and plan digests.

### Migration steps

1. **Inventory the old grader.** List every artifact path, criterion, weight,
   required gate, private fixture, subprocess, and `try/except` that returns
   score `0`.
2. **Delete legacy scaffolding.** Remove `RubricBuilder`, `LLMJudge`, custom
   weight normalizers, hand-built `Grade` / failure dicts, and direct artifact
   I/O. Both `RubricBuilder` and `LLMJudge` emit `DeprecationWarning` on
   construct — do not keep them “temporarily.”
3. **Declare `TASK`.** Map each old criterion to `RubricCriterion`, each file to
   an artifact descriptor, each trusted fixture to `TrustedJson` (or equivalent).
4. **Move domain logic into `evaluate`.** Use
   `context.candidate` / `context.candidate_operation` /
   `context.trusted_operation` / `context.policy` / `context.run_solver` as
   appropriate. Return a criterion map or `RubricEvaluation`.
5. **Refresh and commit the plan:**

   ```bash
   uv run lbx-rl-harness reference --problem-dir problems/<task_id>
   # or:
   uv run python scripts/write_evaluation_plan.py problems/<task_id>
   git add problems/<task_id>/scorer/evaluation.plan.json
   ```

6. **Validate (verify-only for the plan):**

   ```bash
   uv run lbx-rl-template validate --problem-dir problems/<task_id>
   ```

7. **Prove reference and adversarial cases.** Reference must score `1.0`.
   Cover malformed/missing artifacts, zero denominators, policy crashes, and
   naive/no-op below `max_trivial_score` (see
   [`RUBRIC_EVALUATION.md`](RUBRIC_EVALUATION.md#required-adversarial-probes)).

### Canonical examples

- `examples/mujoco-pendulum/`
- `examples/opensees-base-isolation/`
- `examples/openfoam-hydrofoil-flap/`

Starters under `alignerr_plugin/.../starter_templates/` already ship
`RubricTask` + plan generation for rubric families.

### Before / after sketch

Legacy:

```python
def compute_score(workspace, trajectory, private):
    try:
        design = json.loads((workspace / "design.json").read_text())
    except Exception:
        return {"score": 0.0}
    # hand weights, float casts, custom aggregation...
```

Current:

```python
def evaluate(context):
    design = context.candidate
    return {"valid_span": design["span_m"] >= 1.0}


TASK = RubricTask(
    artifact=JsonArtifact("design.json", required_keys=("span_m",), ...),
    criteria=(RubricCriterion("valid_span", weight=1.0, required=True),),
    evaluate=evaluate,
)
```

---

## Path B — Continuous scorers → sealed calibration

PR **#107** replaced a fragile manual calibration process with a reproducible
framework workflow. Later PRs added sealed challenges, grade-time no-info
gates, and Trusted CI promotion. Migrating means adopting that whole stack, not
only regenerating one JSON file.

### Benefits you should expect after migration

- **No copied reference anchors:** reference metrics and `x_ref` are generated
  automatically.
- **Reviewed floor semantics:** floors carry explicit human rationale and are
  never derived from one baseline run.
- **Custom graders preserved where they belong:** authors retain loading,
  rollouts, k-fold, hidden-env, and raw metric measurement; `TASK` owns
  calibration binding and final quality mapping.
- **Unambiguous metrics:** SRE/F1 (and other registered kernels) are exact,
  versioned, and recorded in the lock.
- **Reproducible strategies:** trained reference/naive assets bind training
  inputs/code, configs, seeds, manifests, and artifacts; hand-authored
  `committed_artifact` strategies bind their source without fake training data.
- **One-command finalization:** ground truth runs inference, measures metrics,
  generates the lock, verifies 0 / 0.5 / 1 anchors, and updates proof evidence.
- **Automatic invalidation:** changes to data, models, metrics, rationales,
  runtime, or dependencies make calibration stale.
- **Failure atomicity:** unsuccessful calibration preserves the previous trusted
  lock and proof.
- **No committed score copies:** generated submissions and `results.txt` are
  removed from the authoring contract.
- **Efficient rollout grading:** reference and baseline solutions are not
  re-executed during ordinary agent grading.
- **Stronger trust boundary:** local/Harbor images may use a marked fallback;
  Taiga requires the trusted promoted calibration mount from CI.
- **Backward compatible entry:** legacy hand-authored continuous graders can
  keep working while you migrate, but unprotected continuous shapes are
  increasingly blocked at validate/CI.
- **Roadmap compatible:** the same metric schema and PWL calibration feed sealed
  challenge evaluation.

Overall, calibration becomes reproducible, reviewable, dynamically refreshed,
and much harder to accidentally or intentionally misconfigure.

### Target state

Prefer **Tier A** (`ContinuousTask.model()` / policy challenges) whenever the
agent can submit a queryable predictor or policy. Use Tier B
(`ContinuousTask.calibrated()` / `static()` + `TASK.grade`) only as a
compatibility bridge. Tier C (`custom_reviewed`) needs explicit review and must
not claim sealed protection.

Required pieces:

- Registered targets (`PopulationSRETarget`, F1 helpers, …) with
  `FloorAnchor` + `AnchorRationale`.
- `GeneratedCalibration("calibration.lock.json")` — **never hand-edit** the
  lock.
- Production entry: `TASK.grade(...)` (not `TASK.score(metrics)`).
- Local ground-truth for feedback; **Trusted CI** generates and promotes the
  production lock/evidence bundle to `/mcp_server/calibration`.

Schema **v3** locks bind TASK/plan digests, metric kernels, reviewed floors,
reference/naive/degenerate measurements, challenge roots, and PWL qualification.
V2 locks are not patched in place — regenerate.

### Migration steps

#### B1. Hand-written FLOOR / REF / PERFECT grader

1. Replace local metric and curve copies with registered targets +
   `GeneratedCalibration`.
2. Commit reproducible reference and naive strategies. Use
   `model.manifest.json` v1 for existing trained models or
   `strategy.manifest.json` with explicit `trained_model` /
   `committed_artifact` kind.
3. Add `TASK` (`ContinuousTask.model()` if possible).
4. Wire production `compute_score()` to `TASK.grade(...)`.
5. For non-tabular callbacks, declare ready-to-measure workspaces under
   `baselines/degenerate/` with `WorkspaceDegenerateProbes`.
6. If the honest naive ties every effective no-information floor, add a
   reviewed `naive_at_floor` exception; otherwise retain the strict
   weak-positive default.
7. Run local ground truth for feedback; open/update the fork PR so Trusted CI
   seals production evidence.

#### B2. Calibration lock v2 → v3

1. Keep target names, weights, and reviewed floor rationales.
2. Replace `TASK.score(measure_submission())` with `TASK.grade(...)`.
3. Prefer a queryable `ContinuousTask.model()` and private challenge bank.
4. Stop committing generated locks/proofs under `problems/**` if you still track
   them:

   ```bash
   git rm --cached problems/<task_id>/calibration.lock.json \
     problems/<task_id>/.alignerr/build_proof.json
   ```

5. Regenerate local development calibration; CI owns production.

#### B3. Static `submission.csv` → queryable predictor

1. Move held-out features + truth into one private challenge table.
2. Change the prompt/output contract to `predictor.py` (or the family’s
   queryable artifact).
3. Package reference/naive as queryable predictors.
4. Switch `ContinuousTask.calibrated()` → `ContinuousTask.model()`.
5. Retain metric targets and quality anchors; regenerate calibration; run
   reward-hacking probes in
   [`CONTINUOUS_EVALUATION.md`](CONTINUOUS_EVALUATION.md#required-reward-hacking-tests).

### Canonical example

- `examples/mle-tabular-classification/`

### Local vs Trusted CI (do not confuse these)

| Context | What happens |
| --- | --- |
| Author laptop | `lbx-rl-harness run --runtime ground-truth` writes **development** lock/evidence for feedback (`attested: false` locally is expected). |
| Fork PR / Trusted CI | Cache key from immutable task + framework revision; restore or regenerate canonical lock; submit job verifies digest and mounts read-only at `/mcp_server/calibration`. |
| Taiga production | Rejects continuous grading when expected lock/plan evidence or trusted mount is missing/mismatched. |

Authors never “finalize” production calibration by committing a hand-tweaked
lock.

---

## Cross-cutting requirements (both paths)

Apply these even if the score shape already looks modern:

1. **Policy isolation.** Never `import` / `exec` agent Python in the grader
   process. Use `PolicyWorker` / `helpers.run_policy` /
   `context.policy(...)`. See [`POLICY_ISOLATION.md`](POLICY_ISOLATION.md).
2. **Hidden env deps.** Simulator packages the agent must not import belong in
   `env_dependencies`, not agent-visible `dependencies`. See
   [`HIDDEN_ENV.md`](HIDDEN_ENV.md).
3. **Public vs private data.** Public mounts are QA-visible; private fixtures
   stay root-only. Do not smuggle answers into `data/public/`.
4. **Determinism.** No LLM judges in scorers. Rubric and continuous rewards must
   be reproducible from files, seeds, and hidden fixtures.
5. **One task per PR** under `problems/<task_id>/` unless you are changing shared
   framework code intentionally.

## Deprecations to delete during migration

| Old | Replacement |
| --- | --- |
| `RubricBuilder` | `RubricTask` + `evaluate` |
| `LLMJudge` | Deterministic checks only |
| Hand-edited `calibration.lock.json` | `GeneratedCalibration` + CI seal |
| Hand-edited `evaluation.plan.json` | `refresh_evaluation_plan` from `TASK` |
| `TASK.score(...)` in production | `TASK.grade(...)` |
| `ExponentialCurve` | Framework PWL via calibration |
| Custom weight normalization / failure dicts in rubrics | Framework aggregation + faults |

## Verification checklist

Before requesting review on a migrated task:

```bash
uv sync
uv run lbx-rl-template validate --problem-dir problems/<task_id>
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```

Confirm:

- [ ] Path A: `TASK = RubricTask(...)` present; no production `compute_score()`.
- [ ] Path A: `scorer/evaluation.plan.json` present, generated, committed, and
      matching `TASK` digests under validate.
- [ ] Path B: registered targets + `GeneratedCalibration`; production uses
      `TASK.grade`.
- [ ] Path B: prefer Tier-A sealed challenge when feasible; no committed
      production lock under `problems/**`.
- [ ] Reference scores as designed (rubric `1.0`, continuous reference ≈ `0.5`
      after calibration).
- [ ] Naive / constant / shuffle / malformed probes behave as required.
- [ ] No `RubricBuilder` / `LLMJudge` / hand-edited sealed files.
- [ ] Fork PR Trusted CI green for the single task directory.

## Quick decision table

| You have today | Migrate to |
| --- | --- |
| Manual rubric `compute_score` + JSON load | `RubricTask` + descriptors + `evaluation.plan.json` |
| `RubricBuilder` / LLM judge | `RubricTask` only |
| Hand FLOOR/REF/PERFECT floats | Registered targets + generated lock v3 |
| Lock schema v2 + `TASK.score` | Lock v3 + `TASK.grade` |
| Static CSV holdout only | Prefer `ContinuousTask.model()` + private challenge |
| Fixed open-loop policy answer | Fresh post-commit policy challenge |
| Removed ML-only layout | [`LEGACY_ML_LAYOUT.md`](LEGACY_ML_LAYOUT.md) then Path B |

## See also

- [`AUTHORING.md`](AUTHORING.md) — scaffold, reference, ground-truth loop
- [`GRADING.md`](GRADING.md) — contract, helpers, Harbor/Boreal flow
- [`CONTINUOUS_EVALUATION.md`](CONTINUOUS_EVALUATION.md) — tiers, lock v3, sealed
  challenges
- [`RUBRIC_EVALUATION.md`](RUBRIC_EVALUATION.md) — declarative API and plan seal
- [`LEGACY_ML_LAYOUT.md`](LEGACY_ML_LAYOUT.md) — converting the removed ML-only layout
- [`REWARD_HACKING.md`](REWARD_HACKING.md) — adversarial expectations
