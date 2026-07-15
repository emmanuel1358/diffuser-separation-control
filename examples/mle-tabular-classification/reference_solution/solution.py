"""Reference solution for the tabular continuous-scoring example.

Trains a domain-aware linear model for `t1` and XGBoost models for `t2` and
`label`, then writes `/tmp/output/submission.csv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

TRAIN_PATH = Path("/data/train.parquet")
TEST_PATH = Path("/data/test.parquet")
OUT_PATH = Path("/tmp/output/submission.csv")

SEED = 20260417


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["x1", "x2", "x3"]].copy()
    out["x2_sq"] = df["x2"] ** 2
    out["sin_x3"] = np.sin(df["x3"])
    out["cos_x3"] = np.cos(df["x3"])
    out["x1_sin_x3"] = df["x1"] * np.sin(df["x3"])
    return out


def main() -> None:
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    x_train_t1 = pd.DataFrame(
        {
            "x1": train["x1"],
            "x2": train["x2"],
            "x3": train["x3"],
            "x2_sq": train["x2"] ** 2,
        }
    )
    x_test_t1 = pd.DataFrame(
        {
            "x1": test["x1"],
            "x2": test["x2"],
            "x3": test["x3"],
            "x2_sq": test["x2"] ** 2,
        }
    )
    t1_model = LinearRegression().fit(x_train_t1, train["t1"].values)

    x_train_eng = _engineer(train)
    x_test_eng = _engineer(test)

    t2_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        n_jobs=1,
        verbosity=0,
    )
    t2_model.fit(x_train_eng, train["t2"].values)

    label_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        n_jobs=1,
        verbosity=0,
        eval_metric="logloss",
    )
    label_model.fit(x_train_eng, train["label"].values)

    submission = pd.DataFrame(
        {
            "t1": t1_model.predict(x_test_t1),
            "t2": t2_model.predict(x_test_eng),
            "label": label_model.predict(x_test_eng).astype(int),
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)
    print(f"[reference] wrote {len(submission)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
