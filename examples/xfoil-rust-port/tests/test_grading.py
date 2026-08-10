"""Container-only end-to-end grading checks for XFOIL attack resistance."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

TASK_ROOT = Path("/test-task")
SCORER_PATH = Path("/mcp_server/grader/compute_score.py")
PRIVATE_DATA = Path("/mcp_server/data")


def _load_scorer():
    spec = importlib.util.spec_from_file_location("xfoil_container_scorer", SCORER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_fixture(script: Path, output: Path, *, as_agent: bool) -> None:
    output.mkdir(parents=True)
    if as_agent:
        os.chown(output, 1000, 1000)

    def demote() -> None:
        os.setgroups([])
        os.setgid(1000)
        os.setuid(1000)

    subprocess.run(
        ["bash", str(script)],
        check=True,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "LC_ALL": "C",
            "LBT_OUTPUT_DIR": str(output),
        },
        preexec_fn=demote if as_agent else None,
        timeout=300,
    )


def main() -> None:
    scorer = _load_scorer()
    with tempfile.TemporaryDirectory(prefix="xfoil-grading-") as temporary:
        root = Path(temporary)
        os.chmod(root, 0o755)
        oracle = root / "oracle"
        noop = root / "noop"
        cheat = root / "cheat"
        relay = root / "relay"

        _run_fixture(TASK_ROOT / "solution/solve.sh", oracle, as_agent=False)
        oracle_grade = scorer.TASK.grade(workspace=oracle, private=PRIVATE_DATA)

        _run_fixture(TASK_ROOT / "baselines/naive.sh", noop, as_agent=True)
        noop_grade = scorer.TASK.grade(workspace=noop, private=PRIVATE_DATA)

        _run_fixture(
            TASK_ROOT / "attacks/encoded-oracle/solve.sh",
            cheat,
            as_agent=True,
        )
        cheat_grade = scorer.TASK.grade(workspace=cheat, private=PRIVATE_DATA)

        _run_fixture(
            TASK_ROOT / "attacks/build-script-relay/solve.sh",
            relay,
            as_agent=True,
        )
        relay_grade = scorer.TASK.grade(workspace=relay, private=PRIVATE_DATA)

        assert abs(oracle_grade.score() - 1.0) <= 1e-6, oracle_grade
        assert noop_grade.score() == 0.0, noop_grade
        assert cheat_grade.score() == 0.0, cheat_grade
        assert relay_grade.score() == 0.0, relay_grade
        # Spaced Command tokens are collapsed by WorkspaceArtifact scanning.
        cheat_error = str(
            cheat_grade.metadata.get("error")
            or cheat_grade.metadata.get("agent_fault")
            or cheat_grade
        ).lower()
        assert (
            "forbidden pattern" in cheat_error or "command" in cheat_error
        ), cheat_grade
        relay_error = str(
            relay_grade.metadata.get("error")
            or relay_grade.metadata.get("agent_fault")
            or relay_grade
        ).lower()
        assert "build" in relay_error or "forbidden" in relay_error, relay_grade
        assert not (Path("/tmp") / "forged-monitor-trace").exists()

    shutil.rmtree("/tmp/output", ignore_errors=True)
    print("XFOIL TASK.grade oracle=1, no-op=0, encoded-oracle=0, build-script-relay=0")


if __name__ == "__main__":
    main()
