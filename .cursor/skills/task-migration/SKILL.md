---
name: task-migration
description: Converts existing Alignerr task implementations to current sealed RubricTask, ContinuousTask, policy, calibration, and artifact contracts. Use only for maintaining legacy tasks, not for designing brand-new problems.
---

# Task Migration

Use this skill only when an existing task must move to the current framework.
Brand-new tasks use `alignerr-task-authoring` and a domain skill.

## Determine the destination

- Deterministic criteria -> `RubricTask` +
  `multi_deterministic_rubrics`.
- Continuous held-out metric -> `ContinuousTask` +
  `continuous_scoring_function`.
- Submitted policy over seeds -> shared policy evaluation.
- Black-box interaction -> `hidden-env-tasks`.
- Repository software -> `software-engineering-tasks`.

## Preserve the product contract

Before changing code, record:

- agent-visible prompt/inputs;
- submission artifact;
- hidden behavior and oracle;
- baseline/floor behavior;
- runtime dependencies;
- fault semantics; and
- existing evidence/provenance.

Do not preserve unsafe implementation details merely for compatibility.

## Migration sequence

1. Update enum-backed task metadata and output declarations.
2. Replace raw submission reads with shared descriptors/loaders.
3. Replace root execution/import with UID-dropped shared APIs.
4. Convert scorer to the selected task declaration.
5. Classify candidate vs grader/infrastructure faults.
6. Add deterministic oracle and baseline contracts.
7. Generate the sealed evaluation plan or calibration lock.
8. Add malformed/no-op/attack regressions.
9. Run local reference, validation, and ground-truth preflight.
10. Let trusted CI regenerate authoritative evidence.

## Never

- Catch broad exceptions and return zero.
- Hand-edit evaluation plans/calibration locks.
- Keep pickle/dynamic import of candidate artifacts in root.
- Train in trusted CI ground-truth.
- Change hidden requirements while calling the change mechanical.
- Mark migration complete while validator stages are skipped.

## Verify

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-template check --problem-dir problems/<task_id>
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
uv run lbx-rl-template validate --problem-dir problems/<task_id>
```

## References

- `docs/TASK_MIGRATION.md`
- `docs/RUBRIC_EVALUATION.md`
- `docs/CONTINUOUS_EVALUATION.md`
- `docs/REWARD_HACKING.md`
