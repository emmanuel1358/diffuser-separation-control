"""Reference solution: train on the public data and write a submission.

The expert reference whose metric anchors the 0.5 score (REF in test_file.py).
It must run inside the task image and produce /tmp/output/submission.csv. Keep it
a strong, domain-aware solution, not a shortcut that games the scorer. Iterate:

    uv run lbx-rl-tasks-harness reference --problem-dir problems/<task_id>
"""

from __future__ import annotations

from pathlib import Path

PUBLIC_DATA = Path("/data")  # data/public/ is mounted here in the task image
OUTPUT = Path("/tmp/output")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # TODO: load PUBLIC_DATA, train, and write predictions for every held-out row
    # to OUTPUT / "submission.csv" with the columns your grader requires.
    raise NotImplementedError("fill in the reference solution")


if __name__ == "__main__":
    main()
