---
name: rubric-design
description: Designs measurable deterministic rubric criteria for Alignerr tasks. Use when choosing RubricCriterion sets, required gates, weights, hidden coverage, or responding to rubric-quality feedback.
---

# Rubric Design

## Design from outcomes

Each criterion should represent one observable requirement from the public
specification. Prefer:

- behavioral compatibility;
- safety or validity;
- state/recovery;
- deterministic replay;
- robustness over declared variation;
- performance in named units; and
- security invariants.

Avoid criteria based mainly on filenames, symbol presence, line counts, oracle
diff similarity, or one expected library.

## Required gates

Mark a criterion `required=True` when failure invalidates the solution:

- artifact/protocol does not parse;
- project cannot build;
- mandatory operation is missing;
- data integrity or security fails;
- deterministic behavior is required but absent.

Grant optional partial credit only after required validity holds.

## Weights

- Define weights once in `RubricCriterion`.
- Return raw criterion scores; do not aggregate in `evaluate()`.
- Make weights reflect product importance, not test implementation cost.
- Ensure a cosmetic/static subset cannot outscore broken core behavior.

## Hidden coverage

Build semantic families tied to public requirements. Vary values, scale,
ordering, seeds, schedules, restart, and errors without adding hidden
requirements.

Public tooling should use the same full-credit tolerances as hidden grading.

## Calibration

Grade:

- oracle;
- unchanged starter;
- no-op;
- representative partial solution;
- public-case hardcode; and
- task-specific attacks.

If a weak solution earns significant reward, fix gates/criteria before adding
more hidden cases.

## Review questions

- Can an engineer explain exactly what each score measures?
- Can two independent implementations earn full credit?
- Is every hidden behavior publicly specified?
- Are tolerances justified?
- Are results deterministic?
- Does the rubric distinguish progress from shape?
- Can candidate output forge any score input?

Use `deterministic-grading` for API mechanics and
`reward-hacking-security` for security.

## References

- `docs/RUBRIC_GUIDANCE.md`
- `docs/RUBRIC_EVALUATION.md`
- `docs/REWARD_HACKING.md`
