from __future__ import annotations

import shutil
import subprocess

import pytest

from lbx_rl_tasks_harness.tmux_tool import _tmux_exec_args


def test_tmux_exec_runs_as_agent_with_memory_and_thread_limits() -> None:
    args = _tmux_exec_args("container-123", "new-session -d -s train python fit.py")

    assert args[:3] == ["docker", "exec", "container-123"]
    assert args[-3:-1] == ["bash", "-lc"]
    script = args[-1]
    assert "child_memory_limit_bytes" in script
    assert 'ulimit -v "$limit_kib"' in script
    assert "setpriv --reuid=1000 --regid=1000 --clear-groups --no-new-privs" in script
    assert " su " not in script
    assert "HOME=/home/agent" in script
    assert "OMP_NUM_THREADS=1" in script
    assert "MKL_NUM_THREADS=1" in script
    assert "OPENBLAS_NUM_THREADS=1" in script
    assert "NUMEXPR_NUM_THREADS=1" in script
    assert "exec tmux new-session -d -s train python fit.py" in script


@pytest.mark.skipif(shutil.which("setpriv") is None, reason="setpriv is Linux-only")
def test_setpriv_preserves_inherited_address_space_limit() -> None:
    result = subprocess.run(
        [
            "bash",
            "-lc",
            "ulimit -v 262144 && "
            "exec setpriv --no-new-privs /bin/bash -lc 'ulimit -v'",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "262144"
