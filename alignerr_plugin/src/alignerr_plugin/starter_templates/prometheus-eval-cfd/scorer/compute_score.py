"""Starter CFD grader.

Minimal scaffold for ``task_type = "cfd"``. Replace the placeholder criterion
with real hidden evaluation that runs the solver (OpenFOAM ships in the base
image conda env, reached via ``/etc/solver-envs.d/openfoam.sh``) against
held-out flow conditions. For a complete working reference, read
``examples/openfoam-hydrofoil-flap/``.
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


def _load_control(workspace: Path) -> dict[str, Any]:
    """Read the agent's ``control.json`` as an AgentFault-guarded JSON object.

    A missing/dir/FIFO path or invalid JSON is the agent's fault: raise the
    typed ``AgentFault`` (kept 0.0) instead of letting a bare ``OSError`` escape
    and discard the rollout.
    """
    path = workspace / "control.json"
    try:
        text = path.read_text()
    except OSError as exc:
        raise AgentFault(f"could not read control.json: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentFault(f"control.json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentFault("control.json must be a JSON object")
    return data


def compute_score(
    workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path
) -> dict[str, Any]:
    """Score a submitted CFD design (placeholder rubric -- replace me)."""
    _ = private
    control = _load_control(workspace)

    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)

    @rb.criterion(
        id="control_present",
        weight=1.0,
        description="control.json exists, parses, and declares parameters",
    )
    def _():
        return len(control) > 0

    # TODO: add task-specific criteria that run the solver against hidden
    # conditions and score the result (see
    # examples/openfoam-hydrofoil-flap/scorer/compute_score.py).
    return rb.grade().to_dict()
