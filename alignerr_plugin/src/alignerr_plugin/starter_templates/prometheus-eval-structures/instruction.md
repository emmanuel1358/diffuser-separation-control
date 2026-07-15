# Structures Task

Replace this with the structural problem the agent must solve (e.g. size members
to meet a stress/displacement target under a budget, or repair a finite-element
model so the analysis converges).

## What to submit

Write a JSON file to:

```text
/tmp/output/design.json
```

containing the design your grader expects, for example:

```json
{
  "example_param": 1.0
}
```

The grader evaluates the submitted design against hidden load cases and scores
the result deterministically using physics-grounded structural metrics.
