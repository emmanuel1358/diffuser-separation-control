---
name: deterministic-grading
description: Guides deterministic Alignerr scorer implementation with RubricTask, ContinuousTask, sealed evaluation plans, fault semantics, and shared artifact APIs. Use when writing or reviewing scorer code.
---

# Deterministic Grading

## Choose the scoring contract

### Deterministic rubrics

For `reward_type = "multi_deterministic_rubrics"`:

- declare `TASK = RubricTask(...)`;
- do not define author-owned `compute_score()`;
- return raw criterion subscores from `evaluate(context)`;
- define weights and required gates only in `RubricCriterion`; and
- generate `scorer/evaluation.plan.json` from `TASK`.

### Continuous scoring

For `reward_type = "continuous_scoring_function"`:

- use `ContinuousTask`/shared calibration where supported;
- keep `compute_score()` a thin adapter to the declared task;
- use a committed inference-only reference strategy; and
- enforce the reference target `0.5 ± 0.05`.

## Shared APIs only

Use descriptors such as:

- `WorkspaceArtifact`
- `JsonArtifact`
- `TextArtifact`
- `RegularFileArtifact`
- `TrustedJson`

Use `RubricContext` operations:

- `fixture`
- `candidate_operation`
- `trusted_operation`
- `number`, `ratio`, `mean`
- `run_candidate`, `run_candidate_suite`
- `run_solver` for trusted tools only
- `reject_candidate`, `grader_failure`

For non-rubric formats, use sanctioned helpers from `grading.helpers`,
`grading.policy_eval`, `grading.env_loading`, and `grading.kfold`.

Never parse candidate artifacts with raw `open`, pickle, unsafe `torch.load`,
leaf-only `O_NOFOLLOW`, or direct subprocess execution in the root grader.

## Faults

- Candidate-controlled malformed input, failure, or timeout -> `AgentFault`
  (kept rollout, normally score `0.0`).
- Invalid hidden fixtures, trusted reference bugs, or environment failures ->
  propagate grader/infrastructure failure (discarded rollout).
- Never use `except Exception: return 0.0`.

## Determinism

Pin seeds, solver settings, state, operation order, and tolerances. Generate
trusted expected values before candidate execution. Repeat identical committed
submissions and require identical results.

Do not use LLM judges, provider APIs, wall-clock scoring, or unseeded randomness.

## Evaluation plan

```bash
uv run python scripts/write_evaluation_plan.py problems/<task_id>
uv run lbx-rl-template validate --problem-dir problems/<task_id>
```

Commit the generated plan and never hand-edit it. Trusted CI reseals and verifies
the plan.

## Verification

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
uv run lbx-rl-template check --problem-dir problems/<task_id>
```

Also read `rubric-design` for criterion quality and
`reward-hacking-security` for the trust boundary.

## References

- `docs/RUBRIC_EVALUATION.md`
- `docs/CONTINUOUS_EVALUATION.md`
- `docs/GRADING.md`
- `examples/mujoco-pendulum/scorer/compute_score.py`
- `examples/mle-tabular-classification/scorer/compute_score.py`
