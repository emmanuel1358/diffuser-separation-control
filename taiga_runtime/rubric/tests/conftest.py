"""Import local runtime and grading packages without a wheel build."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

for source_root in (
    REPO_ROOT / "taiga_runtime" / "rubric" / "src",
    REPO_ROOT / "grader" / "src",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
