"""Reference solution: package a trained queryable predictor.

The expert reference whose metric anchors the 0.5 score (REF in test_file.py).
It must produce /tmp/output/predictor.py plus any sibling model artifacts. Keep
it a strong, domain-aware solution, not a shortcut that games the scorer.

    uv run lbx-rl-tasks-harness reference --problem-dir problems/<task_id>
"""

from __future__ import annotations

from pathlib import Path

PUBLIC_DATA = Path("/data")  # data/public/ is mounted here in the task image
OUTPUT = Path("/tmp/output")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # TODO: package your committed model and write predictor.py implementing
    # load_predictor() -> object with predict(rows) -> target-column lists.
    raise NotImplementedError("fill in the reference solution")


if __name__ == "__main__":
    main()
