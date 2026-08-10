"""Shared pytest configuration for the harness test suite.

This suite was written on developer machines, where none of the environment
variables GitHub Actions injects into every step are present. One of them steers
production code down a CI-only branch:

* ``GITHUB_ACTIONS`` puts ``StreamingTrajectoryRenderer`` into its Actions branch,
  where it prints ``::group::`` markers to stdout instead of writing to the ``rich``
  Console the caller injected. ``test_trajectory.py`` asserts against that Console,
  so the test failed whenever the suite ran inside any Actions context.

``GITHUB_ENV``, ``GITHUB_OUTPUT``, ``GITHUB_PATH`` and ``GITHUB_STEP_SUMMARY`` are
cleared alongside it. Nothing under ``harness/`` reads them today, so clearing them
changes no current result; it is here because code that reads them appends to
whichever path they name, and a unit test that grew such a call would otherwise
write into the environment of the job running it.

Clearing them gives the suite one behaviour everywhere it runs. A test that wants
the Actions branch sets the variable itself with ``monkeypatch`` -- this fixture
runs before the test body, so opting back in still works, and
``test_trajectory.py::test_streaming_renderer_emits_github_actions_groups`` does
exactly that to cover the ``::group::`` path.

Mirrors the equivalent fixture in the mothership's root ``tests/conftest.py``. It
lives here, inside a template-owned directory, because
``sync-shared-from-template.yml`` replaces ``grader/`` and ``harness/`` wholesale in
consumer repos: a conftest added downstream would be deleted by the next sync.
"""

from __future__ import annotations

import pytest

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
