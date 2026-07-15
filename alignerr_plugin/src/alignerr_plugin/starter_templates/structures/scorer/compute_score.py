"""Starter structures grader.

Minimal scaffold for ``task_type = "structures"``. Replace the placeholder
criterion with real hidden evaluation that runs a finite-element analysis
(OpenSeesPy ships in the task image) against held-out load cases. For a complete
working reference, read ``examples/opensees-base-isolation/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grading import (  # noqa: F401  -- helpers exposed for authors
    AgentFault,
    RubricBuilder,
    helpers,
)


def _load_design(workspace: Path) -> dict[str, Any]:
    """Read the agent's ``design.json`` as an AgentFault-guarded JSON object.

    A missing/dir/FIFO path or invalid JSON is the agent's fault: raise the
    typed ``AgentFault`` (kept 0.0) instead of letting a bare ``OSError`` escape
    and discard the rollout.
    """
    path = workspace / "design.json"
    try:
        text = path.read_text()
    except OSError as exc:
        raise AgentFault(f"could not read design.json: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentFault(f"design.json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentFault("design.json must be a JSON object")
    return data


def compute_score(
    workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path
) -> dict[str, Any]:
    """Score a submitted structural design (placeholder rubric -- replace me)."""
    _ = private
    design = _load_design(workspace)

    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)

    @rb.criterion(
        id="design_present",
        weight=1.0,
        description="design.json exists, parses, and declares parameters",
    )
    def _():
        return len(design) > 0

    # TODO: add task-specific criteria that run OpenSeesPy against hidden load
    # cases and score the response (see
    # examples/opensees-base-isolation/scorer/compute_score.py).
    return rb.grade().to_dict()
