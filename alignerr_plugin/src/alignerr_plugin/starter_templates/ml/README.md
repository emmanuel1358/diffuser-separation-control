# ML task template

This scaffolds a v3 sealed-challenge continuous ML task. Authors declare a
queryable predictor, private challenge bank, registered targets, and reviewed
quality anchors through `grading.evaluation.ContinuousTask`.

## Required authored surface

```text
metadata.json
prompt.md
test_file.py
data/public/
data/private/
data-generation/
reference_solution/
  train.py
  solution.py
  <trained model artifacts>
  model.manifest.json
baselines/naive/
  train.py
  solution.py
  <trained model artifacts>
  model.manifest.json
```

The reference and naive models, training/inference scripts, configs, dependency
locks, and seeds are committed. Generated submissions and score logs are not.
`data/private/challenge.parquet` contains private features and target truth.

## Finalize the task

After filling the TODOs and training the committed models:

```bash
uv run lbx-rl-harness run \
  --runtime ground-truth \
  --problem-dir problems/<task_id>
```

For `task_type=ml`, this command:

1. Validates committed model manifests.
2. Runs reference and naive inference in isolated containers.
3. Measures raw metrics against private truth.
4. Generates and qualifies the three-anchor curve.
5. Atomically writes ignored local calibration lock/evidence state.
6. Replays no-op/reference/oracle score contracts.
7. Produces local diagnostics for author feedback.

Commit the trained models, manifests, and scripts—not generated calibration
files. Trusted CI independently generates or restores the production lock and
evidence bundle, verifies its digest, and mounts it read-only in Taiga. Do not
edit measured anchors or copy results into `test_file.py`.
Floor values are not measured from baselines: each is a reviewed semantic
choice captured by `FloorAnchor` and `AnchorRationale`. Use exact versioned
metric formulas rather than relying on ambiguous names such as “SRE” or “F1”.

The scaffold uses the Tier-A `ContinuousTask.model()` adapter. For legacy
`submission.csv` migration, read
`docs/CONTINUOUS_EVALUATION.md`; never call `TASK.score(metrics)` from
production.
