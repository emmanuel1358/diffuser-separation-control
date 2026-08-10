# ML task template

This scaffolds a sealed-challenge continuous ML task. Authors declare a
queryable predictor, a held-out challenge bank, registered targets, and reviewed
quality anchors through `grading.evaluation.ContinuousTask`.

## Required authored surface

```text
task.toml                     contract: [difficulty].task_type = "ml"
instruction.md                agent-facing prompt
environment/
  Dockerfile                  thin layer on a native base image
  requirements.txt            agent-visible pip deps (optional)
  apt.txt                     agent-visible apt packages (optional)
data/                         public data, mounted read-only at /data
data_generation/              deterministic regeneration scripts
scorer/
  compute_score.py            the grader, root-only at /mcp_server/grader
  data/                       held-out truth, root-only at /mcp_server/data
  requirements.txt            grader-only pip deps (optional)
solution/
  solve.sh                    oracle entrypoint
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
`scorer/data/` holds the held-out features and target truth; nothing under it is
readable by the uid-1000 agent.

Dependencies are declared per isolation boundary, never in `task.toml`.
`install-task-deps.sh` installs `environment/requirements.txt` into the
agent-visible runtime venv, and `scorer/requirements.txt` into a root-only
target the agent cannot import.

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
3. Measures raw metrics against held-out truth.
4. Generates and qualifies the three-anchor curve.
5. Atomically writes ignored local calibration lock/evidence state.
6. Replays no-op/reference/oracle score contracts.
7. Produces local diagnostics for author feedback.

Commit the trained models, manifests, and scripts, not generated calibration
files. Trusted CI independently generates or restores the production lock and
evidence bundle, verifies its digest, and mounts it read-only in Taiga. Do not
edit measured anchors or copy results into `scorer/compute_score.py`.
Floor values are not measured from baselines: each is a reviewed semantic
choice captured by `FloorAnchor` and `AnchorRationale`. Use exact versioned
metric formulas rather than relying on ambiguous names such as "SRE" or "F1".

The scaffold uses the Tier-A `ContinuousTask.model()` adapter. For legacy
`submission.csv` migration, read
`docs/CONTINUOUS_EVALUATION.md`; never call `TASK.score(metrics)` from
production.
