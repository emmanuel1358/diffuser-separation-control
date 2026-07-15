"""Reference grader for the ml task template (continuous scoring)."""

from pathlib import Path
from typing import Any

from grading import AgentFault


def compute_score(
    workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path
) -> float:
    """Return the continuous reference score when the expected output exists."""
    _ = trajectory, private
    answer = workspace / "answer.txt"
    if not answer.exists():
        return 0.0
    try:
        text = answer.read_text()
    except OSError as exc:
        # A directory/FIFO planted at answer.txt is an agent fault, not infra:
        # raise the typed AgentFault so it is kept as a real 0.0 instead of
        # crashing the grader (which would discard the score the agent earned).
        raise AgentFault(f"could not read answer.txt: {exc}") from exc
    return 0.5 if text.strip() else 0.0
