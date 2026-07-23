"""Train the committed weak, input-dependent naive model."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260417
SHRINKAGE = 0.05
HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parents[1]
DATA_DIR = Path(os.environ.get("LBT_DATA_DIR", TASK_ROOT / "data" / "public"))
MODEL_DIR = Path(os.environ.get("LBT_MODEL_DIR", HERE))
MODEL_PATH = MODEL_DIR / "model.json"
MANIFEST_PATH = MODEL_DIR / "model.manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _features(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(frame)),
            frame["x1"],
            frame["x2"],
            frame["x3"],
        ]
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    features = _features(train)
    model = {
        "schema_version": "tabular-naive-model.v1",
        "seed": SEED,
        "shrinkage": SHRINKAGE,
        "t1_mean": float(train["t1"].mean()),
        "t2_mean": float(train["t2"].mean()),
        "t1_coef": np.linalg.lstsq(features, train["t1"].to_numpy(), rcond=None)[
            0
        ].tolist(),
        "t2_coef": np.linalg.lstsq(features, train["t2"].to_numpy(), rcond=None)[
            0
        ].tolist(),
        "majority_label": int(train["label"].mode().iloc[0]),
    }
    MODEL_PATH.write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "role": "naive",
        "training_entrypoint": "train.py",
        "inference_entrypoint": "solution.py",
        "seed": SEED,
        "public_training_data": {
            "path": "../../data/public/train.parquet",
            "sha256": _sha256(DATA_DIR / "train.parquet"),
        },
        "artifacts": [{"path": "model.json", "sha256": _sha256(MODEL_PATH)}],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {MODEL_PATH}")
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
