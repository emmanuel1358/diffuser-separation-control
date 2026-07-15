"""Env server configuration and activation gate.

The env server is opt-in per task. Activation is driven by a single field in the
task's ``task.toml`` (baked into the image at ``/task/task.toml`` by the task
Dockerfile)::

    [environment]
    hidden_env = "env"      # env server runs; agent interacts ONLY via the socket
    # or
    hidden_env = "hybrid"   # env server runs; task ALSO ships static data/ files
    # or
    hidden_env = ""         # (default) no env server -- classical static task

If ``hidden_env`` is "env" or "hybrid" and ``env_config.json`` is present at
``/mcp_server/data/env_config.json``, it overrides the defaults (``env.py`` /
``make_env``). Shape::

    {
      "module":  "env.py",        // path relative to /mcp_server/data/
      "factory": "make_env",      // module-level callable returning an instance
      "allowed_env_kwargs": ["split", "seed"]  // optional allow-list
    }

``module`` / ``factory`` are optional (use them when your env is a package, e.g.
``"module": "envs/__init__.py"``, or ``make_env`` is the wrong name).

``allowed_env_kwargs`` is an optional reward-hacking guard: the agent supplies
``env_kwargs`` on ``__create__`` and the server forwards them to
``make_env(**env_kwargs)``; if declared, any key not in this list is rejected
before the factory runs. Pin it to exactly the kwargs your factory expects. Omit
it to accept any keys (the always-on ``SERVER_PRIVATE_ROOT`` path guard in
``server.py`` still applies regardless).
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from pathlib import Path

ENV_DATA_DIR = Path("/mcp_server/data")
ENV_CONFIG_PATH = ENV_DATA_DIR / "env_config.json"
# The task Dockerfile bakes task.toml here (COPY task.toml ... /task/). There is
# no in-container problems-metadata.json, so activation is read from task.toml.
TASK_TOML_PATH = Path("/task/task.toml")
SOCKET_PATH = Path("/tmp/env.sock")

# Root of the server-private filesystem. The task Dockerfile chmods /mcp_server
# to 0700, so a UID-1000 agent cannot read held-out targets under
# /mcp_server/data or the grader under /mcp_server/grader. No legitimate
# agent-supplied env_kwarg should ever name a path here, so the env server
# rejects __create__ kwargs whose string values resolve beneath this root (see
# EnvServer._validate_env_kwargs).
SERVER_PRIVATE_ROOT = Path("/mcp_server")

# Root-only site-packages for env-server-ONLY deps, prepended to sys.path before
# the hidden env loads so a simulator is importable by the (root) server but not
# the agent (which must use the RPC). Skipped when absent.
ENV_DEPS_DIR = SERVER_PRIVATE_ROOT / "env_deps"

# Root-only site-packages for GRADER-ONLY deps, prepended to the grader worker's
# sys.path before compute_score loads so a scoring/reference library is importable
# by the (root) grader but not the uid-1000 agent (0700 /mcp_server blocks it, as
# for the held-out truth). Unlike ENV_DEPS_DIR this serves every task type, incl.
# dataset graders that never load an env module. Skipped when absent.
GRADING_DEPS_DIR = SERVER_PRIVATE_ROOT / "grading_deps"

# [environment].hidden_env values that activate the env server. Everything else
# -- including "" and a missing task.toml -- leaves the server off.
_ACTIVATING_VALUES = {"env", "hybrid"}


@dataclasses.dataclass
class EnvConfig:
    module_path: Path
    factory_name: str
    allowed_env_kwargs: frozenset[str] | None = None

    @classmethod
    def load(
        cls,
        data_dir: Path = ENV_DATA_DIR,
        config_path: Path = ENV_CONFIG_PATH,
    ) -> "EnvConfig":
        """Load env_config.json if present, else fall back to defaults
        (``env.py`` / ``make_env``). Missing keys use defaults."""
        module = "env.py"
        factory = "make_env"
        allowed: frozenset[str] | None = None
        if config_path.exists():
            try:
                raw = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise RuntimeError(f"Could not parse {config_path}: {exc}") from exc
            if not isinstance(raw, dict):
                raise RuntimeError(f"{config_path} must be a JSON object")
            module = raw.get("module", module)
            factory = raw.get("factory", factory)
            if not isinstance(module, str) or not isinstance(factory, str):
                raise RuntimeError(
                    f"{config_path}: 'module' and 'factory' must be strings"
                )
            declared = raw.get("allowed_env_kwargs")
            if declared is not None:
                if not isinstance(declared, list) or not all(
                    isinstance(k, str) for k in declared
                ):
                    raise RuntimeError(
                        f"{config_path}: 'allowed_env_kwargs' must be a list of strings"
                    )
                allowed = frozenset(declared)
        return cls(
            module_path=(data_dir / module).resolve(),
            factory_name=factory,
            allowed_env_kwargs=allowed,
        )


def hidden_env_mode(task_toml_path: Path = TASK_TOML_PATH) -> str | None:
    """Read ``[environment].hidden_env`` from task.toml. Returns None if the
    file is missing, malformed, or the field is unset/empty.

    Normalizes with ``strip().lower()`` to match ``EnvironmentSection``'s
    validation: a task that passes validate with ``hidden_env = "ENV"`` bakes
    that raw value into ``/task/task.toml``, so the runtime gate must normalize
    it the same way or the env server would never start.
    """
    if not task_toml_path.exists():
        return None
    try:
        data = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return None
    env = data.get("environment")
    if not isinstance(env, dict):
        return None
    mode = env.get("hidden_env")
    if not isinstance(mode, str):
        return None
    return mode.strip().lower() or None


def _is_env_task(task_toml_path: Path = TASK_TOML_PATH) -> bool:
    return hidden_env_mode(task_toml_path) in _ACTIVATING_VALUES
