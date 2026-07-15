from __future__ import annotations

import asyncio
import subprocess


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
                ["docker", "exec", container_id, "bash", "-lc", f"tmux {command}"],
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
