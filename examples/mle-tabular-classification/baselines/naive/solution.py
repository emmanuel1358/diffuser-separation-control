"""Package the committed weak model as a queryable predictor."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("LBT_OUTPUT_DIR", "/tmp/output"))
MODEL_DIR = Path(os.environ.get("LBT_MODEL_DIR", HERE))
MODEL_PATH = MODEL_DIR / "model.json"

PREDICTOR_SOURCE = r"""
import json
from pathlib import Path


def load_predictor():
    model = json.loads(Path(__file__).with_name("model.json").read_text())

    class Predictor:
        @staticmethod
        def _dot(coef, values):
            return sum(float(a) * float(b) for a, b in zip(coef, values))

        def predict(self, rows):
            t1, t2, label = [], [], []
            shrinkage = float(model["shrinkage"])
            for row in rows:
                values = [1.0, float(row["x1"]), float(row["x2"]), float(row["x3"])]
                raw_t1 = self._dot(model["t1_coef"], values)
                raw_t2 = self._dot(model["t2_coef"], values)
                t1_mean = float(model["t1_mean"])
                t2_mean = float(model["t2_mean"])
                t1.append(t1_mean + shrinkage * (raw_t1 - t1_mean))
                t2.append(t2_mean + shrinkage * (raw_t2 - t2_mean))
                label.append(int(model["majority_label"]))
            return {"t1": t1, "t2": t2, "label": label}

    return Predictor()
""".lstrip()


def main() -> None:
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            "committed naive model is missing; run baselines/naive/train.py"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_PATH, OUTPUT_DIR / "model.json")
    (OUTPUT_DIR / "predictor.py").write_text(PREDICTOR_SOURCE, encoding="utf-8")
    print("[baseline:naive] wrote queryable predictor.py + model.json")


if __name__ == "__main__":
    main()
