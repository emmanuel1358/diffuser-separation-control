"""Linear/logistic baseline for the tabular continuous-scoring example.

This mirrors the original ML_Envs baseline. It trains scikit-learn
linear models on raw features only and writes `/tmp/output/submission.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

TRAIN_PATH = Path("/data/train.parquet")
TEST_PATH = Path("/data/test.parquet")
OUT_PATH = Path("/tmp/output/submission.csv")

FEATURE_COLS = ["x1", "x2", "x3"]


def main() -> None:
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    x_train = train[FEATURE_COLS].values
    x_test = test[FEATURE_COLS].values

    t1_model = LinearRegression().fit(x_train, train["t1"].values)
    t2_model = LinearRegression().fit(x_train, train["t2"].values)
    label_model = LogisticRegression(max_iter=1000).fit(x_train, train["label"].values)

    submission = pd.DataFrame(
        {
            "t1": t1_model.predict(x_test),
            "t2": t2_model.predict(x_test),
            "label": label_model.predict(x_test).astype(int),
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)
    print(f"[baseline:linear] wrote {len(submission)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
