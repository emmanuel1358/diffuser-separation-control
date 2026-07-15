from __future__ import annotations

LOCAL_SYSTEM_PROMPT = (
    "You are an autonomous agent inside a sandboxed Linux task container. "
    "Solve the task described in the user message by exploring available files, "
    "writing code, and producing the required output artifacts under /tmp/output. "
    "Use bash for short shell commands, str_replace_based_edit_tool for file edits, "
    "and tmux for long-running processes."
)

_STRUCTURES_HINT = """\
## OpenSees Availability

OpenSees is available in the environment as an optional tool.
"""

_CFD_HINT = """\
## OpenFOAM Availability

OpenFOAM is available in the environment as an optional tool.
"""


def task_type_hint(task_type: str) -> str:
    normalized = task_type.strip().lower()
    if normalized == "structures":
        return _STRUCTURES_HINT
    if normalized == "cfd":
        return _CFD_HINT
    return ""
