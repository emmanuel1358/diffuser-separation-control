from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / ".github" / "scripts" / "template_ci_handoff_comment.py"
TAIGA_DEPLOY_WORKFLOW = ROOT / ".github" / "workflows" / "dispatch-taiga-deploy.yml"


def _load_helper():
    spec = importlib.util.spec_from_file_location("template_ci_handoff_comment", HELPER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_taiga_deploy_handoff_comment_renders_reviewer_guidance() -> None:
    helper = _load_helper()

    body = helper.render_body(
        kind="taiga-deploy",
        state="dispatched",
        head_sha="abcdef1234567890",
        target_repo="Alignerr-Code-Labeling/lbx-rl-tasks-iso-mothership",
        target_workflow="grade-fork-pr.yml",
        actions_run_url="https://github.com/example/actions/runs/1",
    )

    assert "<!-- lbx-template-ci-handoff:taiga-deploy -->" in body
    assert "## Taiga Deploy Handoff: Dispatched" in body
    assert "reruns trusted CI for this head SHA" in body
    assert "submits Taiga only after that run passes" in body
    assert "Runner workflow: `grade-fork-pr.yml`" in body
    assert "fresh `trusted-ci/grade` run" in body


def test_taiga_deploy_dispatch_forwards_requester_actor() -> None:
    workflow = TAIGA_DEPLOY_WORKFLOW.read_text()

    assert "REQUESTED_BY: ${{ github.actor }}" in workflow
    assert '"requested_by": os.environ["REQUESTED_BY"]' in workflow
