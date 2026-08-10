#!/usr/bin/env python3
"""Deterministically regenerate the tabular example's public and held-out data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260511
N_TRAIN = 500
N_TEST = 200
X1_TRAIN_LOW, X1_TRAIN_HIGH = 0.0, 2.0
X1_TEST_LOW, X1_TEST_HIGH = -1.0, 3.0
NOISE_SCALE_T1 = 0.05
NOISE_SCALE_T2 = 0.10


def _features(
    rng: np.random.Generator,
    n: int,
    x1_low: float,
    x1_high: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": rng.uniform(x1_low, x1_high, size=n),
            "x2": rng.uniform(-1.0, 1.0, size=n),
            "x3": rng.uniform(0.0, 2.0 * np.pi, size=n),
        }
    )


def _targets(features: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    t1_signal = 0.5 * features["x1"] ** 2 + 0.3 * features["x1"] * features["x2"]
    t1 = t1_signal + rng.normal(scale=NOISE_SCALE_T1, size=len(features))
    t2_signal = features["x1"] * np.sin(features["x3"])
    t2 = t2_signal + rng.normal(scale=NOISE_SCALE_T2, size=len(features))
    label = (features["x2"] * np.cos(features["x3"]) > 0).astype(np.int64)
    return pd.DataFrame({"t1": t1, "t2": t2, "label": label})


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    print(f"wrote {path} ({len(frame)} rows)")


def generate(output_root: Path) -> None:
    rng = np.random.default_rng(SEED)
    train_features = _features(rng, N_TRAIN, X1_TRAIN_LOW, X1_TRAIN_HIGH)
    train = pd.concat([train_features, _targets(train_features, rng)], axis=1)
    test_features = _features(rng, N_TEST, X1_TEST_LOW, X1_TEST_HIGH)
    test_targets = _targets(test_features, rng)
    _write(output_root / "data" / "train.parquet", train)
    _write(output_root / "data" / "test.parquet", test_features)
    _write(
        output_root / "scorer" / "data" / "test_target.parquet",
        test_targets,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = parser.parse_args()
    generate(args.output_root.resolve())


if __name__ == "__main__":
    main()
