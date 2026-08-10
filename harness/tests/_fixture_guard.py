"""Skip tests whose fixtures exist only in the template repo.

``sync-shared-from-template.yml`` copies ``grader/`` and ``harness/`` into consumer
repos wholesale, but it does not copy ``examples/`` or ``.github/``. Tests that read
those trees therefore cannot pass downstream: in the mothership they failed with a
bare ``FileNotFoundError`` that reads like a real regression.

``grader/tests/conftest.py`` already applies this policy through the session-scoped
``template_examples`` fixture, which is why roughly 45 grader tests skip cleanly
downstream today. The guards here express the same policy for tests that resolve
their fixture paths at import time instead of taking a fixture, so the check has to
happen at collection.

Prefer the narrowest guard that covers a test, and keep the reason specific: a
developer reading a skip downstream should learn which path is missing and why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"


def missing_repo_paths(*relative_paths: str) -> list[str]:
    """Return the given repo-root-relative paths that are absent here."""
    return [rel for rel in relative_paths if not (REPO_ROOT / rel).exists()]


def requires_repo_paths(*relative_paths: str) -> pytest.MarkDecorator:
    """Skip unless every repo-root-relative path exists in this checkout."""
    missing = missing_repo_paths(*relative_paths)
    return pytest.mark.skipif(
        bool(missing),
        reason=(
            "template-only fixture missing from this checkout: "
            + ", ".join(missing)
            + " (sync-shared-from-template copies grader/ and harness/, "
            "but not examples/ or .github/)"
        ),
    )


def requires_examples(*names: str) -> pytest.MarkDecorator:
    """Skip unless every named ``examples/<name>`` task directory exists."""
    return requires_repo_paths(*(f"examples/{name}" for name in names))
