"""Platform null probe: training means and majority label."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("LBT_DATA_DIR", "/data"))
OUTPUT_DIR = Path(os.environ.get("LBT_OUTPUT_DIR", "/tmp/output"))


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    submission = pd.DataFrame(
        {
            "t1": [float(train["t1"].mean())] * len(test),
            "t2": [float(train["t2"].mean())] * len(test),
            "label": [int(train["label"].mode().iloc[0])] * len(test),
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)


if __name__ == "__main__":
    main()
