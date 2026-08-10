# MLE Tabular Classification

This is the canonical v3 continuous-ML example. The agent submits a queryable
`predictor.py`; `ContinuousTask.model()` commits it, selects private challenge
rows, runs it in a sandbox, applies family-wide information evidence, and then
uses the generated PWL quality calibration.

The example demonstrates the Tier-A migration endpoint while retaining
registered metrics, committed reference/naive models, reviewed floors, and
calibration provenance.

The floors are reviewed metric semantics, not baseline measurements:

- `t1` and `t2` use `sre.rmse_over_population_std.v1`; predicting the held-out
  population mean gives SRE exactly `1`, so that theoretical point is floor.
- `label` uses `f1.binary_threshold_0_5.v1`; binary F1 is bounded below by `0`.

The naive model is only a qualification witness that weak input-dependent work
earns a small positive score. Changing it cannot move either floor or the PWL
curve.

## Authored layout

```text
task.toml
instruction.md
calibration.lock.json
environment/
  Dockerfile
  requirements.txt
data_generation/generate.py
data/                       public data, mounted read-only at /data
scorer/
  compute_score.py
  data/
    challenge.parquet
    test_target.parquet
solution/
  solve.sh
  train.py
  solution.py
  model.json
  model.manifest.json
baselines/
  naive/
    train.py
    solution.py
    model.json
    model.manifest.json
  null/solution.py
  linear/solution.py
  gbt/solution.py
```

Authors commit trained reference/naive models and the complete recipes needed
to reproduce them. Generated submissions and copied score logs are not
committed.

## One-command finalization

For a task with `difficulty.task_type = "ml"`, ground-truth validation owns
calibration:

```bash
uv run lbx-rl-harness run \
  --runtime ground-truth \
  --problem-dir examples/mle-tabular-classification
```

The workflow validates model manifests, runs reference and naive inference in
isolated containers, measures raw metrics against held-out truth, generates the
three-anchor curve, runs qualification checks, atomically replaces
`calibration.lock.json`, replays reference/no-op/oracle contracts, and updates
the build proof.

The reference must score `0.5 +/- 0.05`, the theoretical optimum maps to `1`,
and the weak input-dependent naive model must remain in the configured small
positive band. The mean/majority strategy is a null probe and maps to zero.

## Regenerating data and models

```bash
uv run python \
  examples/mle-tabular-classification/data_generation/generate.py

LBX_PRIVATE_CHALLENGE_SEED='<trusted secret>' uv run python \
  examples/mle-tabular-classification/data_generation/generate_private_challenge.py

uv run python \
  examples/mle-tabular-classification/solution/train.py

uv run python \
  examples/mle-tabular-classification/baselines/naive/train.py
```

After any generator, split, metric, model, or inference change, rerun the
ground-truth command when local calibration feedback is useful. Commit the
refreshed models/manifests and task source. For normal `problems/**` authoring,
the generated lock/evidence remain ignored local state; trusted CI generates
the authoritative Taiga bundle. This example commits its bundle only as a
framework regression fixture.

The private challenge seed is supplied by trusted CI and is never committed or
shown to the solving agent. The committed challenge artifact is mounted
root-only at runtime.

## Runtime grading

During an agent rollout, baseline and reference solutions never run. The grader
commits `predictor.py`, samples hidden challenge rows, certifies task-relevant
information, verifies the promoted calibration release, and applies the PWL
quality mapping.

Agent-controlled malformed output is a kept zero. Missing/stale calibration,
metric-worker failures, or infrastructure faults are marked internal so the
rollout is discarded.
