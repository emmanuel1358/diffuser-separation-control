"""Repository slug normalization."""

from __future__ import annotations

import re

_SEPARATOR_RUN = re.compile(r"[^a-z0-9]+")


def normalize_slug(value: str) -> str:
    """Normalize ASCII text into a stable lowercase slug."""

    normalized = _SEPARATOR_RUN.sub("-", value.strip().lower()).strip("-")
    return normalized or "untitled"
