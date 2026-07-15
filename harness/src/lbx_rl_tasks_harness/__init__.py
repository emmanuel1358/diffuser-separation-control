"""Local testing harness for lbx RL tasks."""

from __future__ import annotations

import warnings

_ORIGINAL_SHOWWARNING = warnings.showwarning
_ALLOWED_OBJECTS_WARNING = (
    "The default value of `allowed_objects` will change in a future version"
)


def _showwarning_without_langgraph_noise(
    message,
    category,
    filename,
    lineno,
    file=None,
    line=None,
):
    if _ALLOWED_OBJECTS_WARNING in str(message):
        return
    _ORIGINAL_SHOWWARNING(message, category, filename, lineno, file=file, line=line)


warnings.filterwarnings(
    "ignore",
    message=(
        "The default value of `allowed_objects` will change in a future version.*"
    ),
    category=Warning,
)
warnings.showwarning = _showwarning_without_langgraph_noise

__all__ = ["__version__"]

__version__ = "0.1.0"
