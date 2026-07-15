from __future__ import annotations

import os
import sys
from pathlib import Path


def harness_subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Child-process env that resolves Python to the harness's own interpreter.

    Task scripts (``solution/solve.sh``, the ground-truth render command) run
    with their CWD set to the task directory, which the local runtimes copy to
    a temp location outside any uv workspace. A bare ``uv run`` / ``python``
    there finds no project and falls back to an arbitrary uv-managed interpreter
    that lacks ``lbx_rl_tasks_harness`` and the task's grading deps — e.g.
    ``uv run python -m lbx_rl_tasks_harness.render_mujoco`` dies with
    ``ModuleNotFoundError``. Activating the harness venv (``VIRTUAL_ENV`` +
    ``PATH``) makes ``uv run``, ``python``, and ``python -m`` all select the
    interpreter that is actually running the harness.

    Pass per-call additions (e.g. ``LBT_OUTPUT_DIR``) via ``extra``.
    """
    env = os.environ.copy()
    executable = Path(sys.executable) if sys.executable else None
    if executable is not None and executable.exists():
        bin_dir = executable.parent
        env["PATH"] = os.pathsep.join(
            part for part in (str(bin_dir), env.get("PATH", "")) if part
        )
        # Only advertise a virtual environment when the interpreter actually
        # lives in one, so uv doesn't warn about an invalid VIRTUAL_ENV.
        if (bin_dir.parent / "pyvenv.cfg").exists():
            env["VIRTUAL_ENV"] = str(bin_dir.parent)
            # A stale interpreter pin would otherwise override the venv.
            env.pop("UV_PYTHON", None)
    if extra:
        env.update(extra)
    return env


def load_env_file(path: Path = Path(".env.local")) -> bool:
    """Load simple KEY=VALUE pairs from a local env file if it exists.

    Existing process env values win. The parser intentionally supports the
    common dotenv subset we need for API keys without adding another dependency.
    """
    if not path.exists():
        return False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True
