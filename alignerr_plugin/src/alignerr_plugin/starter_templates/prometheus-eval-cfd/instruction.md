# CFD Task

Replace this with the flow problem the agent must solve (e.g. design a
geometry/control that achieves a target aerodynamic objective, or repair a
broken case so it converges).

## What to submit

Write a JSON file to:

```text
/tmp/output/control.json
```

containing the design/control parameters your grader expects, for example:

```json
{
  "example_param": 1.0
}
```

The grader evaluates the submitted design against hidden flow conditions and
scores the result deterministically using physics-grounded metrics.
