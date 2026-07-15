"""Untuned gradient-boosted-trees baseline for the tabular example.

Weak-baseline trio (3/3). Trains sklearn gradient boosting on the RAW feature
columns only -- no tuning, no domain feature engineering -- and writes
`/tmp/output/submission.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

TRAIN_PATH = Path("/data/train.parquet")
TEST_PATH = Path("/data/test.parquet")
OUT_PATH = Path("/tmp/output/submission.csv")

FEATURE_COLS = ["x1", "x2", "x3"]


def main() -> None:
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    x_train = train[FEATURE_COLS].values
    x_test = test[FEATURE_COLS].values

    t1 = GradientBoostingRegressor().fit(x_train, train["t1"].values).predict(x_test)
    t2 = GradientBoostingRegressor().fit(x_train, train["t2"].values).predict(x_test)
    label = (
        GradientBoostingClassifier()
        .fit(x_train, train["label"].values)
        .predict(x_test)
    )

    submission = pd.DataFrame({"t1": t1, "t2": t2, "label": label.astype(int)})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)
    print(f"[baseline:gbt] wrote {len(submission)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
