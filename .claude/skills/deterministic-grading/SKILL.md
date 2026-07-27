---
name: deterministic-grading
description: Guides deterministic scorer authoring with RubricTask (rubrics) or compute_score (continuous). Use when writing, reviewing, or debugging task scorers, rubric criteria, continuous scoring, or MuJoCo reward checks.
---

# Deterministic Grading

## Contract

### Continuous scoring (`continuous_scoring_function`)

```python
def compute_score(workspace, trajectory, private):
    ...
```

Return one supported shape:

- `float` in `[0, 1]` for headline-only continuous scoring.
- `dict` with authoritative `score` plus optional `subscores`, `weights`, `metadata`.

### Deterministic rubrics (`multi_deterministic_rubrics`)

Declare `TASK = RubricTask(...)` in `scorer/compute_score.py` (do **not** define
`compute_score`). `scorer/evaluation.plan.json` is written by harness
reference/ground-truth (or `scripts/write_evaluation_plan.py`) and verified
by validate; Trusted CI reseals it with an explicit refresh step. Never
hand-edit. See `docs/RUBRIC_EVALUATION.md`.

```bash
# optional explicit regenerate
uv run python scripts/write_evaluation_plan.py problems/<task_id>
```

## Rules

- Rubric criteria must be deterministic Python checks.
- Do not use LLM judges, provider SDKs, or external model APIs in scorers.
- Rewrite subjective requirements as measurable checks: parse success, numeric
  tolerance, hidden fixture match, simulator rollout result, or structural
  inspection.
- For continuous score dicts, `score` is authoritative; do not recompute the
  headline from subscores.
- For MuJoCo scoring, pin timestep, integrator, initial state, controls,
  perturbations, and RNG seeds.
- Load agent artifacts through descriptors (`JsonArtifact`, `TextArtifact`) or
  sanctioned component-safe helpers. Use `open_submission_file_or_fault` for
  custom file-like parsers; never raw `open`, `lstat`, leaf-only `O_NOFOLLOW`,
  `json.loads`, or pickle on `/tmp/output`.
- Never disable worker privilege dropping or enable NumPy pickle loading.
  Whole-object H5AD evaluation stays entirely inside a dropped worker; the root
  grader may consume only bounded primitive HDF5 datasets.
- Before opening or updating a task PR, run
  `uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>`
  and commit the generated `problems/<task_id>/.alignerr/build_proof.json` plus
  any `.alignerr/ground_truth/` artifacts.

## Preferred Rubric Pattern

```python
from grading.evaluation import RubricCriterion, RubricTask, TextArtifact

def evaluate(context):
    model = context.candidate_operation(
        "MJCF compilation",
        mujoco.MjModel.from_xml_string,
        context.candidate,
    )
    return {"compiled": model is not None}

TASK = RubricTask(
    artifact=TextArtifact("model.xml"),
    criteria=(RubricCriterion("compiled", required=True),),
    evaluate=evaluate,
)
```

## References

- `docs/RUBRIC_EVALUATION.md`
- `docs/GRADING.md`
- `examples/mujoco-pendulum/scorer/compute_score.py`
- `examples/mle-tabular-classification/scorer/compute_score.py`
