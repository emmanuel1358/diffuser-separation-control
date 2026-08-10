from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from _fixture_guard import requires_repo_paths

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / ".github" / "scripts" / "check_workflow_boundary.py"
WORKFLOWS = ROOT / ".github" / "workflows"

pytestmark = requires_repo_paths(
    ".github/scripts/check_workflow_boundary.py",
    ".github/workflows",
)


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_workflow_boundary", CHECKER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_template_workflows_respect_trust_boundary() -> None:
    checker = _load_checker()

    assert checker.audit_workflows(WORKFLOWS) == []


def test_checker_rejects_secrets_and_cross_repo_dispatch(tmp_path: Path) -> None:
    checker = _load_checker()
    workflow = tmp_path / "unsafe.yml"
    workflow.write_text(
        "on: pull_request_target\n"
        "env:\n"
        "  TOKEN: ${{ secrets.WORKFLOW_TOKEN }}\n"
        "run: curl https://api.github.com/repos/example/repo/dispatches\n"
    )

    violations = checker.audit_workflows(tmp_path)

    assert len(violations) == 3
    assert any("privileged fork-PR execution" in item for item in violations)
    assert any("must not consume repository secrets" in item for item in violations)
    assert any("cross-repository dispatch" in item for item in violations)


@pytest.mark.parametrize(
    "secret_reference",
    [
        "${{secrets.WORKFLOW_TOKEN}}",
        "${{ secrets['WORKFLOW_TOKEN'] }}",
        '${{ secrets["WORKFLOW_TOKEN"] }}',
        "${{ toJSON(secrets) }}",
        "${{\n  secrets.WORKFLOW_TOKEN\n}}",
        "secrets: inherit",
    ],
)
def test_checker_rejects_all_actions_secret_forms(
    tmp_path: Path, secret_reference: str
) -> None:
    checker = _load_checker()
    workflow = tmp_path / "unsafe.yml"
    if secret_reference == "secrets: inherit":
        workflow.write_text("name: unsafe\njobs:\n  call:\n    secrets: inherit\n")
    else:
        workflow.write_text(f"name: unsafe\nvalue: {secret_reference}\n")

    violations = checker.audit_workflows(tmp_path)

    assert len(violations) == 1
    assert "must not consume repository secrets" in violations[0]


def test_checker_allows_unprivileged_github_context(tmp_path: Path) -> None:
    checker = _load_checker()
    workflow = tmp_path / "safe.yml"
    workflow.write_text("name: safe\nvalue: ${{ github.token }}\n")

    assert checker.audit_workflows(tmp_path) == []
