"""Supervisor that spawns and restarts the env server subprocess.

Called from the rubric MCP server's startup. If the task opts in via
``[environment].hidden_env = "env"|"hybrid"``, spawns the env server as a child
process and restarts it on unexpected exit (up to a small budget). If it cannot
stay up, the supervisor kills the MCP so the orchestrator surfaces the failure
rather than leaving the agent connected to a broken socket.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from typing import Callable

from .config import SOCKET_PATH, EnvConfig, _is_env_task

# Restart budget: after this many unexpected exits the supervisor gives up and
# kills the MCP so the failure is surfaced rather than served as a broken socket.
_MAX_RESTARTS = 3
_RESTART_COOLDOWN_S = 0.5

logger = logging.getLogger(__name__)

# Handle to the live subprocess + a stop latch so the grading path can tear the
# server down BEFORE grading (closing the agent's grade-time socket) without the
# supervisor racing a respawn that would re-bind the socket.
_proc_lock = threading.Lock()
_current_proc: subprocess.Popen | None = None
_stop_requested = threading.Event()


def stop_env_server(timeout: float = 5.0) -> None:
    """Stop the supervised env server (if any) and remove its socket.

    Called from the grading path before the grader subprocess runs.
    ``/tmp/env.sock`` is the agent's training channel and is NOT part of grading
    (graders read the held-out env in-process via
    ``grading.load_env_module``), so stopping the server removes the agent's
    grade-time access to the live env (free reset() seed fingerprinting, a
    bit-exact twin-env oracle, private-method probes) without affecting grading.

    Idempotent; a no-op when no server was started (static tasks). Sets the stop
    latch FIRST so the supervisor will not restart, then SIGTERMs (SIGKILL
    fallback on timeout) and unlinks the socket as a backstop.
    """
    _stop_requested.set()
    with _proc_lock:
        proc = _current_proc
    if proc is not None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "[ENV_SERVER] env server did not exit on SIGTERM within "
                    "%.1fs; sending SIGKILL",
                    timeout,
                )
                proc.kill()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logger.error("[ENV_SERVER] env server survived SIGKILL wait")
        except Exception:
            logger.exception("[ENV_SERVER] error stopping env server")
    try:
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
    except OSError:
        pass


def supervise_if_enabled(
    max_restarts: int = _MAX_RESTARTS,
    on_give_up: Callable[[], None] | None = None,
) -> threading.Thread | None:
    """Spawn + supervise the env server if this task opts in, else no-op.

    If the subprocess cannot stay up after ``max_restarts`` attempts,
    ``on_give_up`` is called (default ``os._exit(1)`` so the container dies hard
    and the orchestrator restarts it rather than serving a broken MCP). Returns
    the supervisor thread, or None when the server was not started (task is not
    env/hybrid).

    ``os._exit(1)`` is deliberate (NOT ``sys.exit(1)``): a broken MCP should die
    hard, skipping atexit handlers. Tests that need to assert the give-up path
    pass an explicit ``on_give_up``.
    """
    if not _is_env_task():
        logger.debug(
            "[ENV_SERVER] hidden_env not env/hybrid; env server not started"
        )
        return None

    try:
        config = EnvConfig.load()
    except Exception:
        logger.exception("[ENV_SERVER] Failed to parse env_config.json")
        if on_give_up is None:
            os._exit(1)
        else:
            on_give_up()
            return None
    if not config.module_path.exists():
        logger.error(
            "[ENV_SERVER] hidden_env is env/hybrid but env module %s is missing; "
            "aborting so the setup bug is surfaced rather than running a broken "
            "agent session",
            config.module_path,
        )
        if on_give_up is None:
            os._exit(1)
        else:
            on_give_up()
            return None

    # Pre-flight: import the env module in the parent (already root) to surface
    # SyntaxError / ImportError cleanly with one traceback, rather than spawning
    # the subprocess and burning the restart budget failing the same way.
    try:
        from grading.env_loading import load_env_module

        load_env_module(
            config.module_path,
            factory_name=config.factory_name,
            module_name="_supervisor_preflight",
            trusted_roots=(config.module_path.parent.resolve(),),
        )
    except Exception as exc:
        logger.error(
            "[ENV_SERVER] env module %s failed pre-flight import: %s: %s; "
            "aborting before spawning the worker. Fix the module and redeploy.",
            config.module_path,
            type(exc).__name__,
            exc,
        )
        logger.debug("[ENV_SERVER] pre-flight traceback:", exc_info=exc)
        if on_give_up is None:
            os._exit(1)
        else:
            on_give_up()
            return None

    def _supervise():
        global _current_proc
        restarts = 0
        while restarts <= max_restarts:
            logger.info(
                "[ENV_SERVER] Spawning env server subprocess (env module: %s, "
                "attempt %d/%d)",
                config.module_path,
                restarts + 1,
                max_restarts + 1,
            )
            # `python -m env_server` runs the package __main__.py -> serve().
            # Both streams routed to the parent's stderr so the child never
            # writes to fd 1 -- the rubric MCP server uses stdio transport which
            # owns fd 1 for JSON-RPC framing; stray bytes there corrupt the wire.
            with _proc_lock:
                if _stop_requested.is_set():
                    return
                proc = subprocess.Popen(
                    [sys.executable, "-m", "env_server"],
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                )
                _current_proc = proc
            rc = proc.wait()
            with _proc_lock:
                _current_proc = None
            if _stop_requested.is_set():
                logger.info(
                    "[ENV_SERVER] stop requested; env server not restarted"
                )
                return
            if rc == 0:
                logger.info(
                    "[ENV_SERVER] env server exited cleanly; not restarting"
                )
                return
            restarts += 1
            if restarts > max_restarts:
                break
            logger.warning(
                "[ENV_SERVER] env server exited with code %d; restart %d/%d "
                "after %.1fs",
                rc,
                restarts,
                max_restarts,
                _RESTART_COOLDOWN_S,
            )
            threading.Event().wait(_RESTART_COOLDOWN_S)

        logger.error(
            "[ENV_SERVER] env server failed %d times; giving up and killing MCP",
            max_restarts,
        )
        if on_give_up is not None:
            on_give_up()
        else:
            os._exit(1)

    thread = threading.Thread(target=_supervise, daemon=True, name="env_supervisor")
    thread.start()
    return thread
