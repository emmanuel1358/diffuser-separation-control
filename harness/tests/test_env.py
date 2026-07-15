from __future__ import annotations

import os
import sys
from pathlib import Path

from lbx_rl_tasks_harness.env import harness_subprocess_env, load_env_file
from lbx_rl_tasks_harness.models_config import default_model_from_env, required_env_var


def test_load_env_file_does_not_override_existing_value(tmp_path: Path) -> None:
    path = tmp_path / ".env.local"
    path.write_text(
        "ANTHROPIC_API_KEY=file-value\n"
        "OPENAI_API_KEY='quoted-value'\n"
        "LBX_RL_HARNESS_MODEL=openai:gpt-5.1\n"
    )
    os.environ["ANTHROPIC_API_KEY"] = "existing-value"
    try:
        assert load_env_file(path)
        assert os.environ["ANTHROPIC_API_KEY"] == "existing-value"
        assert os.environ["OPENAI_API_KEY"] == "quoted-value"
        assert default_model_from_env("claude-opus-4-7") == "openai:gpt-5.1"
        assert (
            required_env_var(default_model_from_env("claude-opus-4-7"))
            == "OPENAI_API_KEY"
        )
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("LBX_RL_HARNESS_MODEL", None)


def test_provider_specific_model_env(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    assert default_model_from_env("claude-opus-4-7") == "claude-sonnet-4-5"


def test_harness_subprocess_env_activates_the_running_interpreter() -> None:
    """Task scripts run from a temp dir must resolve to the harness venv.

    Otherwise `uv run python -m lbx_rl_tasks_harness.render_mujoco` falls back
    to a bare managed interpreter without the package (the render cpython bug).
    """
    bin_dir = Path(sys.executable).parent
    env = harness_subprocess_env({"LBT_OUTPUT_DIR": "/tmp/x"})

    # The interpreter's bin dir is first on PATH, so `python` / `uv run` resolve
    # to the harness interpreter rather than a managed fallback.
    assert env["PATH"].split(os.pathsep)[0] == str(bin_dir)
    assert env["LBT_OUTPUT_DIR"] == "/tmp/x"
    # When running inside a venv (the CI/dev case), it is advertised so uv uses
    # it from a no-project directory.
    if (bin_dir.parent / "pyvenv.cfg").exists():
        assert env["VIRTUAL_ENV"] == str(bin_dir.parent)


def test_harness_subprocess_env_overrides_stale_uv_python(monkeypatch) -> None:
    """A leaked UV_PYTHON pin must not override the activated venv."""
    monkeypatch.setenv("UV_PYTHON", "3.12")
    env = harness_subprocess_env()
    bin_dir = Path(sys.executable).parent
    if (bin_dir.parent / "pyvenv.cfg").exists():
        assert "UV_PYTHON" not in env
