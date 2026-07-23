"""Train the committed reference model and refresh its provenance manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260417
HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parent
DATA_DIR = Path(os.environ.get("LBT_DATA_DIR", TASK_ROOT / "data" / "public"))
MODEL_DIR = Path(os.environ.get("LBT_MODEL_DIR", HERE))
MODEL_PATH = MODEL_DIR / "model.json"
MANIFEST_PATH = MODEL_DIR / "model.manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regression_features(frame: pd.DataFrame, *, target: str) -> np.ndarray:
    if target == "t1":
        values = [
            np.ones(len(frame)),
            frame["x1"],
            frame["x2"],
            frame["x1"] ** 2,
            frame["x1"] * frame["x2"],
        ]
    else:
        values = [
            np.ones(len(frame)),
            frame["x1"],
            frame["x2"],
            np.sin(frame["x3"]),
            np.cos(frame["x3"]),
            frame["x1"] * np.sin(frame["x3"]),
        ]
    return np.column_stack(values)


def train() -> dict:
    train_frame = pd.read_parquet(DATA_DIR / "train.parquet")
    t1_coef, *_ = np.linalg.lstsq(
        _regression_features(train_frame, target="t1"),
        train_frame["t1"].to_numpy(),
        rcond=None,
    )
    t2_coef, *_ = np.linalg.lstsq(
        _regression_features(train_frame, target="t2"),
        train_frame["t2"].to_numpy(),
        rcond=None,
    )
    label_feature = train_frame["x2"].to_numpy() * np.cos(train_frame["x3"].to_numpy())
    negative = label_feature[train_frame["label"].to_numpy() == 0]
    positive = label_feature[train_frame["label"].to_numpy() == 1]
    threshold = 0.5 * (float(np.max(negative)) + float(np.min(positive)))
    return {
        "schema_version": "tabular-reference-model.v1",
        "seed": SEED,
        "t1_coef": t1_coef.tolist(),
        "t2_coef": t2_coef.tolist(),
        "label_threshold": threshold,
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(
        json.dumps(train(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "role": "reference",
        "training_entrypoint": "train.py",
        "inference_entrypoint": "solution.py",
        "seed": SEED,
        "public_training_data": {
            "path": "../data/public/train.parquet",
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
