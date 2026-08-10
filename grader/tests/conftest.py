"""Pytest fixtures shared across the test suite.

Tests live in the template repo alongside the runtime packages they
cover (`grading`, `grader_runner`, `alignerr_plugin`). All packages are
workspace members of this repo's pyproject.toml — `uv sync` from the
repo root installs them in editable mode and pytest picks them up via
the normal import machinery.

The path tweaks below are a fallback for `python -m pytest` invocations
that bypass the workspace install (e.g. `uv run --with httpx pytest`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from _fixture_guard import EXAMPLES as TEMPLATE_EXAMPLES

REPO_ROOT = Path(__file__).resolve().parents[2]
GRADING_SRC = REPO_ROOT / "grader" / "src"
GRADER_RUNNER_SRC = REPO_ROOT / "grader" / "src"
ALIGNERR_PLUGIN_SRC = REPO_ROOT / "alignerr_plugin" / "src"
RUBRIC_SRC = REPO_ROOT / "taiga_runtime" / "rubric" / "src"


for _p in (GRADING_SRC, GRADER_RUNNER_SRC, ALIGNERR_PLUGIN_SRC, RUBRIC_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# GitHub Actions injects these into every step, and code that reads them behaves
# differently when they are set: `GITHUB_ACTIONS` gates CI-only branches, and the
# other four name files that code appends to, so a test reaching one of those code
# paths would write into the environment of the job running it.
#
# Nothing under `grader/` reads any of them today, so clearing them changes no
# current result. It is here for the same reason as the copy in
# `harness/tests/conftest.py`, which does fix a real failure: both suites are
# template-owned, and `sync-shared-from-template.yml` copies them into consumer
# repos that run them under Actions, so neither should depend on whether a runner
# is present. A test that wants the Actions branch sets the variable itself with
# `monkeypatch`, because this fixture runs first.
_RUNNER_ENV_VARS = (
    "GITHUB_ACTIONS",
    "GITHUB_ENV",
    "GITHUB_OUTPUT",
    "GITHUB_PATH",
    "GITHUB_STEP_SUMMARY",
)


@pytest.fixture(autouse=True)
def _isolate_from_runner_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _RUNNER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def template_examples() -> Path:
    """Path to the example tasks directory in this same repo."""
    if not TEMPLATE_EXAMPLES.exists():
        pytest.skip(f"examples directory not present at {TEMPLATE_EXAMPLES}")
    return TEMPLATE_EXAMPLES


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A clean workspace directory the agent would write to."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """A clean directory for run_grader output (reward.json, etc.)."""
    od = tmp_path / "output"
    od.mkdir()
    return od
