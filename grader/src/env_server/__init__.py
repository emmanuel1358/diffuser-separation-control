"""Optional per-task environment RPC server.

For ``[environment].hidden_env = "env"|"hybrid"`` tasks where the agent must
interact with a black-box environment through a Unix socket (``/tmp/env.sock``)
whose source it can never read. See ``server.py`` for the protocol and contract,
``config.py`` for activation, ``supervisor.py`` for startup/teardown, and
``grading.load_env_module`` for the trusted in-process env loader the grader
uses at grade time.

Agent-facing client code is shipped per task as ``data/env_client.py`` (a
canonical copy lives at ``env_server/env_client.py``); the server side never
imports it.
"""

from .config import EnvConfig, _is_env_task, hidden_env_mode
from .protocol import _pack, _unpack
from .server import EnvServer, serve
from .supervisor import stop_env_server, supervise_if_enabled

__all__ = [
    "EnvConfig",
    "EnvServer",
    "serve",
    "supervise_if_enabled",
    "stop_env_server",
    "hidden_env_mode",
    "_is_env_task",
    "_pack",
    "_unpack",
]
