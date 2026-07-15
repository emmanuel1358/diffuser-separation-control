"""The grader runner accepts no-arg ``compute_score()`` graders: the worker
calls the no-arg signature, resolves the helper imports, and still honors
AgentFault, all via the baked ``compute_score.py`` name (the production path)."""

from __future__ import annotations

import json
from pathlib import Path

from grader_runner import worker

NOARG_GRADER = (
    "from grading.helpers import load_submission_or_fault  # noqa: F401\n"
    "from grading.env_loading import load_env_module  # noqa: F401\n"
    "from grading.faults import AgentFault  # noqa: F401\n\n\n"
    "def compute_score():\n"
    "    return 0.5\n"
)

NOARG_AGENT_FAULT = (
    "from grading.faults import AgentFault\n\n\n"
    "def compute_score():\n"
    "    raise AgentFault('bad submission')\n"
)

NATIVE_GRADER = (
    "def compute_score(workspace, trajectory, private):\n"
    "    return 1.0\n"
)


def _run(grader_dir: Path, tmp_path: Path) -> dict:
    result = tmp_path / "result.json"
    rc = worker.main(
        [
            "--workspace", str(tmp_path / "ws"),
            "--grader-dir", str(grader_dir),
            "--private-dir", str(tmp_path / "priv"),
            "--result-path", str(result),
        ]
    )
    payload = json.loads(result.read_text())
    return {"rc": rc, "payload": payload}


def test_noarg_grader_is_discovered_and_called(tmp_path: Path):
    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "compute_score.py").write_text(NOARG_GRADER)
    out = _run(gdir, tmp_path)
    assert out["rc"] == 0
    assert out["payload"]["score"] == 0.5


def test_noarg_agent_fault_is_kept_zero(tmp_path: Path):
    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "compute_score.py").write_text(NOARG_AGENT_FAULT)
    out = _run(gdir, tmp_path)
    assert out["rc"] == 0
    # AgentFault contract: kept 0.0 (not discarded) with the message surfaced.
    assert out["payload"]["score"] == 0.0
    assert out["payload"]["env_internal_failure"] is False
    assert out["payload"]["metadata"]["agent_fault"] == "bad submission"


def test_native_three_arg_still_works(tmp_path: Path):
    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "compute_score.py").write_text(NATIVE_GRADER)
    out = _run(gdir, tmp_path)
    assert out["rc"] == 0
    assert out["payload"]["score"] == 1.0


def test_missing_grader_source_reports_failure(tmp_path: Path):
    gdir = tmp_path / "g"
    gdir.mkdir()
    out = _run(gdir, tmp_path)
    assert out["rc"] == 1
    assert out["payload"]["env_internal_failure"] is True
