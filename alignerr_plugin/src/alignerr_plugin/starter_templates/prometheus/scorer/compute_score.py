"""Reference grader for the ml task template (continuous scoring)."""

import os
from pathlib import Path
from typing import Any

from grading import AgentFault
from grading.helpers import open_submission_file_or_fault


def compute_score(
    workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path
) -> float:
    """Return the continuous reference score when the expected output exists."""
    _ = trajectory, private
    answer = workspace / "answer.txt"
    if not os.path.lexists(answer):
        return 0.0
    try:
        with open_submission_file_or_fault(answer, allow_empty=True) as handle:
            text = handle.read().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AgentFault(f"could not read answer.txt: {exc}") from exc
    return 0.5 if text.strip() else 0.0
