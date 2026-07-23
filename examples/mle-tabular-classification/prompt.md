# Synthetic Tabular Regression + Classification

You have a small tabular dataset with three continuous input features and three
prediction targets. Train on the provided rows, then submit a queryable predictor
that the grader can evaluate on fresh hidden rows after your artifact is committed.

## Inputs

The public data files live under `/data/`:

- `/data/train.parquet`: 500 rows, columns `x1`, `x2`, `x3`, `t1`, `t2`, `label`. The first three are features, and the last three are targets.
- `/data/column_mapping.json`: high-level descriptions of each column. It contains no formulas and no hints about relationships between columns.

The hidden challenge has a wider `x1` range than training. Models must generalize
over `x1` rather than replaying a fixed output vector.

## What to produce

Write:

```text
/tmp/output/predictor.py
```

It must define `load_predictor()`, returning an object with:

```python
predict(rows: list[dict]) -> dict[str, list]
```

Each input row contains `x1`, `x2`, and `x3`. Return equally sized lists named
`t1`, `t2`, and `label`; labels must be `0` or `1`.

```python
def load_predictor():
    class Predictor:
        def predict(self, rows):
            return {
                "t1": [0.0 for _ in rows],
                "t2": [0.0 for _ in rows],
                "label": [0 for _ in rows],
            }
    return Predictor()
```

## Constraints

- The predictor must not require network access.
- Calls must be deterministic for the same rows.
- You may place additional model files beside `predictor.py`.

## Scoring

Each target is scored against its own metric and anchored between a naive baseline and the theoretical optimum:

- `t1`: SRE, standardized RMSE (`RMSE / std(true)`); lower is better. Perfect is `0`.
- `t2`: SRE; lower is better. Perfect is `0`.
- `label`: binary F1; higher is better. Perfect is `1`.

The grader first commits your artifact, then selects a private challenge and
calls your predictor in a sandbox. Each target must show row-level information
under a family-wide permutation certificate; constants, perturbed constants,
row-index outputs, and marginally shuffled outputs receive zero for that target.
Certified targets retain their ordinary calibrated quality score.
