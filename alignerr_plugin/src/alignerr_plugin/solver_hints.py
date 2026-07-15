"""Shared solver availability hints for downstream runners."""

_STRUCTURES_HINT = """\
## OpenSees Availability

OpenSees is available in the environment as an optional tool.
"""

_CFD_HINT = """\
## OpenFOAM Availability

OpenFOAM is available in the environment as an optional tool.
"""


def task_type_solver_hint(task_type: str) -> str:
    """Return the optional solver availability hint for a task type."""
    normalized = task_type.strip().lower()
    if normalized == "structures":
        return _STRUCTURES_HINT
    if normalized == "cfd":
        return _CFD_HINT
    return ""
