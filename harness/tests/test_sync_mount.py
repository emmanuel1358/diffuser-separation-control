"""Coverage for scripts/sync_mount.sh's Hugging Face branch.

The function is extracted and run on its own with a deliberately missing packer,
so the tests assert on whether the packer is reached at all rather than on what
it would have done.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_SYNC_MOUNT = Path(__file__).resolve().parents[2] / "scripts" / "sync_mount.sh"


def _run_sync_hf_resources(problem_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run just sync_hf_resources with a packer path that fails if invoked."""
    source = _SYNC_MOUNT.read_text(encoding="utf-8")
    start = source.index("sync_hf_resources() {")
    end = source.index("\n}\n", start) + len("\n}\n")
    harness = problem_dir / "harness.sh"
    harness.write_text(
        source[start:end]
        + '\nPROBLEM_DIR="$1" TASK_ID=demo-task PACK_HF=/nonexistent/pack.py'
        " sync_hf_resources\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(harness), str(problem_dir)],
        capture_output=True,
        text=True,
    )


def test_hf_sync_is_a_no_op_when_no_task_declares_a_repo(tmp_path: Path) -> None:
    """A task with no Hugging Face mounts must not reach the packer at all.

    sync_hf_resources ran unconditionally and its first act was a fatal packer
    call, so a mismatch between the trusted caller and a fork's stale packer
    failed submit-taiga for tasks that declare nothing to mount.
    """
    (tmp_path / "task.toml").write_text(
        '[task]\nname = "demo"\n\n[[preloaded_files]]\nlocal_path = "data/x.parquet"\n',
        encoding="utf-8",
    )

    result = _run_sync_hf_resources(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "declares no Hugging Face mounts" in result.stdout


def test_hf_sync_still_runs_the_packer_when_a_repo_is_declared(tmp_path: Path) -> None:
    (tmp_path / "task.toml").write_text(
        '[[preloaded_files]]\nhf_repo = "org/dataset"\nlocal_path = "data/x.parquet"\n',
        encoding="utf-8",
    )

    result = _run_sync_hf_resources(tmp_path)

    assert result.returncode != 0
    assert "failed to read [[preloaded_files]]" in result.stderr
