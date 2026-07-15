"""Naive baseline for the tabular continuous-scoring example.

Weak-baseline trio (1/3). Summary statistics only -- no training: predict the
train-set mean for the regression targets t1/t2 and the majority class for
`label`, on every test row. Writes `/tmp/output/submission.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TRAIN_PATH = Path("/data/train.parquet")
TEST_PATH = Path("/data/test.parquet")
OUT_PATH = Path("/tmp/output/submission.csv")


def main() -> None:
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    t1_mean = float(train["t1"].mean())
    t2_mean = float(train["t2"].mean())
    majority = int(train["label"].mode().iloc[0])

    submission = pd.DataFrame(
        {
            "t1": [t1_mean] * len(test),
            "t2": [t2_mean] * len(test),
            "label": [majority] * len(test),
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)
    print(f"[baseline:naive] wrote {len(submission)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
