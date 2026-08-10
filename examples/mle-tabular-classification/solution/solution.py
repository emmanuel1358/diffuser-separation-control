"""Package the committed model as a queryable predictor artifact."""

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
import math
from pathlib import Path


def load_predictor():
    model = json.loads(Path(__file__).with_name("model.json").read_text())

    class Predictor:
        @staticmethod
        def _dot(coef, values):
            return sum(float(a) * float(b) for a, b in zip(coef, values))

        def predict(self, rows):
            t1, t2, label = [], [], []
            for row in rows:
                x1 = float(row["x1"])
                x2 = float(row["x2"])
                x3 = float(row["x3"])
                t1.append(
                    self._dot(
                        model["t1_coef"],
                        [1.0, x1, x2, x1 * x1, x1 * x2],
                    )
                )
                t2.append(
                    self._dot(
                        model["t2_coef"],
                        [1.0, x1, x2, math.sin(x3), math.cos(x3), x1 * math.sin(x3)],
                    )
                )
                feature = x2 * math.cos(x3)
                label.append(int(feature > float(model["label_threshold"])))
            return {"t1": t1, "t2": t2, "label": label}

    return Predictor()
""".lstrip()


def main() -> None:
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            "committed reference model is missing; run solution/train.py"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_PATH, OUTPUT_DIR / "model.json")
    (OUTPUT_DIR / "predictor.py").write_text(PREDICTOR_SOURCE, encoding="utf-8")
    print("[reference] wrote queryable predictor.py + model.json")


if __name__ == "__main__":
    main()
