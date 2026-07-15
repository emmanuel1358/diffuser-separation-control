---
name: deterministic-grading
description: Guides deterministic compute_score.py and RubricBuilder authoring. Use when writing, reviewing, or debugging task scorers, rubric criteria, continuous scoring, or MuJoCo reward checks.
---

# Deterministic Grading

## Contract

Every task scorer defines:

```python
def compute_score(workspace, trajectory, private):
    ...
```

Return one supported shape:

- `float` in `[0, 1]` for headline-only continuous scoring.
- `dict` with authoritative `score` plus optional `subscores`, `weights`, `metadata`.
- `RubricBuilder.grade().to_dict()` for weighted deterministic criteria and penalties.

## Rules

- Rubric criteria must be deterministic Python checks.
- Do not use `rb.llm_criterion`, `LLMJudge`, provider SDKs, or external model APIs in `compute_score.py`.
- Rewrite subjective requirements as measurable checks: parse success, numeric tolerance, hidden fixture match, simulator rollout result, or structural inspection.
- For score dicts, `score` is authoritative; do not recompute the headline from subscores.
- For MuJoCo scoring, pin timestep, integrator, initial state, controls, perturbations, and RNG seeds.
- Before opening or updating a task PR, run `uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>` and commit the generated `problems/<task_id>/.alignerr/build_proof.json` plus any `.alignerr/ground_truth/` artifacts.

## Preferred Pattern

```python
from grading import RubricBuilder

def compute_score(workspace, trajectory, private):
    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)

    @rb.criterion(id="compiled", weight=0.1, description="MJCF compiles")
    def _():
        return deterministic_check(workspace / "model.xml")

    return rb.grade().to_dict()
```

## References

- `docs/GRADING.md`
- `examples/mujoco-pendulum/scorer/compute_score.py`
- `examples/mle-tabular-classification/scorer/compute_score.py`
