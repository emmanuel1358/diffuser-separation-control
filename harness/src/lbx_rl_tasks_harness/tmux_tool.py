from __future__ import annotations

import asyncio
import shlex
import subprocess


def _tmux_exec_args(container_id: str, command: str) -> list[str]:
    """Run tmux as the model uid with the shared per-process memory limit.

    The ordinary rubric bash/editor tools apply RLIMIT_AS before dropping to uid
    1000. The local harness tmux tool bypasses that MCP boundary via
    ``docker exec``, so a short root launcher reads the sealed runtime limit,
    applies it, then uses ``setpriv`` to exec tmux as uid/gid 1000 without
    resetting RLIMIT_AS (modern ``su`` resets it). This keeps detached sweeps
    unprivileged and under the inherited cap.
    """
    agent_command = (
        "export HOME=/home/agent USER=agent LOGNAME=agent "
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1; exec tmux "
        f"{command}"
    )
    limited = (
        'limit_kib="$(python -c '
        "'from grading.runtime_hardening import child_memory_limit_bytes; "
        "print(child_memory_limit_bytes() // 1024)'"
        ')" && ulimit -v "$limit_kib" '
        "&& exec setpriv --reuid=1000 --regid=1000 --clear-groups "
        "--no-new-privs /bin/bash -lc "
        f"{shlex.quote(agent_command)}"
    )
    return [
        "docker",
        "exec",
        container_id,
        "bash",
        "-lc",
        limited,
    ]


def build_tmux_tool(container_id: str):
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise RuntimeError(
            "tmux tool dependencies are missing. Install with `uv sync`."
        ) from exc

    @tool
    async def tmux(command: str) -> str:
        """Run a tmux command inside the task container."""

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                _tmux_exec_args(container_id, command),
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )

        proc = await asyncio.get_running_loop().run_in_executor(None, _run)
        return (
            f"$ tmux {command}\n"
            f"exit_code={proc.returncode}\n"
            f"stdout:\n{proc.stdout[-6000:]}\n"
            f"stderr:\n{proc.stderr[-6000:]}"
        )

    return tmux
