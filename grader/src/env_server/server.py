"""EnvServer: threaded Unix-socket server that proxies method calls to
task-supplied env instances.

For tasks where the agent must interact with a black-box environment -- the
agent can call methods on it but should NOT be able to read the env source,
reward function, or hidden dynamics -- this module runs a local Unix-socket
server that proxies method calls to a task-supplied instance.

The env's Python source lives at ``/mcp_server/data/env.py`` (root-only, because
the canonical task Dockerfile chmods /mcp_server to 0700). The env server runs
inside the rubric's container as root and dispatches method-call messages to a
task-supplied instance. The agent (UID 1000) can open a Unix socket at
``/tmp/env.sock`` but cannot open the env source.

--- Design: generic RPC, not fixed ops -----------------------------------

The server is a thin method-call proxy: the agent says "call method X on
instance Y with these kwargs," the server does ``getattr(instance, X)(**kwargs)``,
and the return value comes back msgpack-encoded. Gymnasium works without a shim
because ``env.reset(seed=...)`` / ``env.step(action=...)`` accept keyword calls.

--- Task-author contract -------------------------------------------------

Ship ``scorer/data/env.py`` (baked to ``/mcp_server/data/env.py``) with a
factory::

    def make_env(**env_kwargs) -> instance

The returned instance can have any public methods. Gymnasium's
``reset`` / ``step`` / ``close`` work out of the box; custom APIs
(``query_price``, ``buy``, ``evaluate_structure``) work equally well. A task can
narrow the socket-dispatchable surface with a class-level
``_env_public_methods`` allow-list. Set ``[environment].hidden_env = "env"``
(no static data) or ``"hybrid"`` (static data + env) in task.toml.

--- Wire protocol --------------------------------------------------------

See ``env_server/protocol.py`` for framing and type coercion.

Request body (map)::

    {"method": "<name>", "instance_id": <int>, "args": {...}}

Response body (success)::

    {"ok": true, "result": <decoded return value>, "gen": "<hex>"}

Response body (failure)::

    {"ok": false, "error": {"type": "...", "message": "...", "traceback": ""},
     "gen": "<hex>"}

The ``traceback`` field is always empty in replies to the agent: the server runs
as root and can read the 0700 /mcp_server tree, so a real traceback would carry
private source lines. The full traceback is logged server-side instead.

The ``gen`` field is a per-process server-generation token (fresh UUID each
subprocess start). A changed ``gen`` means the supervisor restarted the
subprocess and all prior ``instance_id`` registrations are gone.

--- Reserved method names ------------------------------------------------

``__create__`` creates the instance via ``make_env(**args.env_kwargs)`` and
stores it under ``instance_id``. ``env_kwargs`` is screened by
``_validate_env_kwargs`` (optional per-task allow-list + always-on rejection of
string values resolving beneath ``/mcp_server``). ``__destroy__`` removes the
instance (calling ``close()`` if present). Any other ``method`` whose name
starts with ``_`` is rejected: a private/underscore method can expose internal
state (an oracle, hidden parameters) the rollout is meant to infer.

--- Public-method allow-list (opt-in) -----------------------------------

By default every PUBLIC method is dispatchable. A task can narrow that by
declaring a class-level ``_env_public_methods`` (set/frozenset/list of names):
only listed names dispatch. A declared-but-malformed value fails CLOSED
(rejects every method), so an env that tried to lock its surface never silently
falls open.

--- Concurrency ----------------------------------------------------------

Threaded: each connection runs in its own thread; requests on the same
``instance_id`` serialize via a per-instance lock, different ids run in
parallel. A class-level ``_env_concurrent_methods`` opt-in lets named methods
dispatch without the per-instance lock (for long-polling / multi-agent waits);
the env is then responsible for that method's thread-safety.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from .config import ENV_DEPS_DIR, SERVER_PRIVATE_ROOT, SOCKET_PATH, EnvConfig
from .protocol import _FRAME_HEADER, _MAX_FRAME_BYTES, _frame, _pack, _unpack

_RECV_CHUNK_BYTES = 65536
_LISTEN_BACKLOG = 16

_OP_CREATE = "__create__"
_OP_DESTROY = "__destroy__"

logger = logging.getLogger(__name__)


def _resolves_under_private_root(value: str) -> bool:
    """True if ``value`` is an absolute path that resolves (following symlinks)
    to ``SERVER_PRIVATE_ROOT`` or anything beneath it. Absolute-only by design:
    a bare relative token like ``"test"`` is a normal env_kwarg (a split name)."""
    if not isinstance(value, str) or not os.path.isabs(value):
        return False
    try:
        resolved = os.path.realpath(value)
    except ValueError:
        return True
    root = str(SERVER_PRIVATE_ROOT)
    return resolved == root or resolved.startswith(root + os.sep)


def _reject_private_paths(value: Any) -> None:
    """Recursively raise ``ValueError`` if any string inside ``value`` resolves
    beneath the server-private root."""
    if isinstance(value, str):
        if _resolves_under_private_root(value):
            raise ValueError(
                f"env_kwargs may not reference a server-private path: {value!r}"
            )
    elif isinstance(value, dict):
        for k, v in value.items():
            _reject_private_paths(k)
            _reject_private_paths(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _reject_private_paths(v)


class EnvServer:
    """Threaded Unix-socket server that proxies method calls to task-supplied
    env instances. Instantiable with custom paths for testability."""

    def __init__(self, env_config: EnvConfig, socket_path: Path):
        self.env_config = env_config
        self.socket_path = socket_path
        self._make_env = self._load_make_env()
        self._instances: dict[int, Any] = {}
        self._instances_lock = threading.Lock()
        self._instance_locks: dict[int, threading.Lock] = {}
        self._shutdown = threading.Event()
        self._listen_sock: socket.socket | None = None
        self._gen: str = uuid.uuid4().hex

    # ---- loading ------------------------------------------------------

    def _load_make_env(self):
        from grading.env_loading import load_env_module  # lazy: avoid import cycle

        path = self.env_config.module_path
        mod = load_env_module(
            path,
            factory_name=self.env_config.factory_name,
            module_name="task_env",
            trusted_roots=(path.parent.resolve(),),
        )
        return getattr(mod, self.env_config.factory_name)

    # ---- per-instance lock helper -------------------------------------

    def _get_instance_lock(self, instance_id: int) -> threading.Lock:
        with self._instances_lock:
            lock = self._instance_locks.get(instance_id)
            if lock is None:
                lock = threading.Lock()
                self._instance_locks[instance_id] = lock
            return lock

    # ---- request dispatch --------------------------------------------

    def _destroy_instance(self, instance_id: int) -> None:
        """Pop the instance + lock and call close() if defined. Idempotent;
        logs but swallows close() exceptions."""
        lock = self._get_instance_lock(instance_id)
        with lock:
            with self._instances_lock:
                instance = self._instances.pop(instance_id, None)
                self._instance_locks.pop(instance_id, None)
        if instance is None:
            return
        close = getattr(instance, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception(
                    "[ENV_SERVER] close() raised for instance %d; discarding",
                    instance_id,
                )

    def _validate_env_kwargs(self, env_kwargs: dict) -> None:
        """Reward-hacking guard on agent-supplied ``__create__`` kwargs.

        1. Optional allow-list: reject any key outside
           ``env_config.allowed_env_kwargs`` when declared.
        2. Always-on path guard: reject any string value (recursively) that
           resolves to an absolute path beneath SERVER_PRIVATE_ROOT.
        """
        allowed = self.env_config.allowed_env_kwargs
        if allowed is not None:
            extra = sorted(k for k in env_kwargs if k not in allowed)
            if extra:
                raise ValueError(
                    f"env_kwargs key(s) {extra!r} not in this task's "
                    f"allowed_env_kwargs {sorted(allowed)!r}"
                )
        _reject_private_paths(env_kwargs)

    def _handle(
        self, msg: dict, created_instance_ids: set[int] | None = None
    ) -> dict:
        if not isinstance(msg, dict):
            raise ValueError("request body must be a msgpack map")
        method = msg.get("method")
        instance_id = msg.get("instance_id")
        args = msg.get("args") or {}

        if not isinstance(method, str) or not method:
            raise ValueError("method must be a non-empty string")
        # bool is an int subclass; reject it so `instance_id: true` can't shadow
        # the integer-1 instance.
        if not isinstance(instance_id, int) or isinstance(instance_id, bool):
            raise ValueError("instance_id must be an integer (not a bool)")
        if not isinstance(args, dict):
            raise ValueError("args must be a map (dict)")

        if method == _OP_CREATE:
            env_kwargs = args.get("env_kwargs") or {}
            if not isinstance(env_kwargs, dict):
                raise ValueError("args.env_kwargs must be a map (dict)")
            self._validate_env_kwargs(env_kwargs)
            lock = self._get_instance_lock(instance_id)
            with lock:
                with self._instances_lock:
                    if instance_id in self._instances:
                        raise ValueError(
                            f"instance_id {instance_id} already exists; call "
                            f"{_OP_DESTROY!r} first or use a new id"
                        )
                try:
                    new_instance = self._make_env(**env_kwargs)
                except BaseException:
                    with self._instances_lock:
                        self._instance_locks.pop(instance_id, None)
                    raise
                with self._instances_lock:
                    self._instances[instance_id] = new_instance
            if created_instance_ids is not None:
                created_instance_ids.add(instance_id)
            return {"ok": True, "result": None}

        if method == _OP_DESTROY:
            self._destroy_instance(instance_id)
            if created_instance_ids is not None:
                created_instance_ids.discard(instance_id)
            return {"ok": True, "result": None}

        if method.startswith("_"):
            raise ValueError(
                f"method {method!r} is not callable from the env protocol "
                f"(names starting with '_' are private)"
            )

        with self._instances_lock:
            instance = self._instances.get(instance_id)
        if instance is None:
            raise KeyError(
                f"no instance for instance_id {instance_id}; "
                f"call {_OP_CREATE!r} first"
            )

        # Per-task public-method allow-list (opt-in). Fail CLOSED on a malformed
        # declaration so an env that tried to lock its surface never falls open.
        raw_allow = getattr(instance, "_env_public_methods", None)
        if raw_allow is not None:
            if isinstance(raw_allow, str):
                logger.warning(
                    "[ENV_SERVER] instance %d declared _env_public_methods as a "
                    "bare string (%r); failing closed (rejecting all methods)",
                    instance_id,
                    raw_allow,
                )
                allow: frozenset[str] = frozenset()
            else:
                try:
                    allow = frozenset(raw_allow)
                except TypeError:
                    logger.warning(
                        "[ENV_SERVER] instance %d has non-iterable "
                        "_env_public_methods (%r); failing closed",
                        instance_id,
                        type(raw_allow).__name__,
                    )
                    allow = frozenset()
            if method not in allow:
                raise ValueError(
                    f"method {method!r} is not in this env's public API "
                    f"(not listed in _env_public_methods)"
                )

        raw_concurrent = getattr(instance, "_env_concurrent_methods", None)
        try:
            concurrent = (
                frozenset(raw_concurrent) if raw_concurrent else frozenset()
            )
        except TypeError:
            logger.warning(
                "[ENV_SERVER] instance %d has non-iterable _env_concurrent_methods "
                "(%r); dispatching with the per-instance lock",
                instance_id,
                type(raw_concurrent).__name__,
            )
            concurrent = frozenset()
        if method in concurrent:
            fn = getattr(instance, method, None)
            if fn is None:
                raise AttributeError(f"instance has no method {method!r}")
            if not callable(fn):
                raise TypeError(f"attribute {method!r} on instance is not callable")
            return {"ok": True, "result": fn(**args)}

        lock = self._get_instance_lock(instance_id)
        with lock:
            # Stale-lock detection: another thread may have destroyed (and a
            # third re-created) this instance_id while we waited for the lock.
            with self._instances_lock:
                current_lock = self._instance_locks.get(instance_id)
                if current_lock is not lock:
                    raise KeyError(
                        f"instance {instance_id} was destroyed (and possibly "
                        f"re-created) during dispatch; lock rotation detected. "
                        f"Re-issue with the current generation token."
                    )
                instance = self._instances.get(instance_id)
            if instance is None:
                raise KeyError(
                    f"no instance for instance_id {instance_id}; "
                    f"call {_OP_CREATE!r} first"
                )
            fn = getattr(instance, method, None)
            if fn is None:
                raise AttributeError(f"instance has no method {method!r}")
            if not callable(fn):
                raise TypeError(f"attribute {method!r} on instance is not callable")
            result = fn(**args)
        return {"ok": True, "result": result}

    def _handle_frame(
        self, body: bytes, created_instance_ids: set[int] | None = None
    ) -> bytes:
        """Decode a frame body, dispatch, return the packed reply body. Every
        reply is stamped with the current server generation token."""
        try:
            msg = _unpack(body)
        except Exception as e:
            return _pack(
                {
                    "ok": False,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": "",
                    },
                    "gen": self._gen,
                }
            )

        try:
            reply = self._handle(msg, created_instance_ids=created_instance_ids)
        except Exception as e:
            # Don't return the traceback to the agent: it carries /mcp_server
            # source lines (server runs as root over 0700). Log it server-side.
            logger.warning(
                "env RPC error: %s: %s", type(e).__name__, e, exc_info=True
            )
            reply = {
                "ok": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": "",
                },
            }
        reply["gen"] = self._gen
        try:
            return _pack(reply)
        except Exception as e:
            logger.warning(
                "env reply encode error: %s: %s", type(e).__name__, e, exc_info=True
            )
            return _pack(
                {
                    "ok": False,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": "",
                    },
                    "gen": self._gen,
                }
            )

    # ---- connection handling -----------------------------------------

    def _handle_connection(self, conn: socket.socket) -> None:
        """One thread per connection. Reads framed requests, replies, and on any
        exit path destroys the instances THIS connection created and did not
        explicitly ``__destroy__`` (creator-scoped lifetime)."""
        buf = b""
        created_instance_ids: set[int] = set()
        try:
            while not self._shutdown.is_set():
                while len(buf) < _FRAME_HEADER.size:
                    chunk = conn.recv(_RECV_CHUNK_BYTES)
                    if not chunk:
                        return
                    buf += chunk
                (length,) = _FRAME_HEADER.unpack(buf[: _FRAME_HEADER.size])
                buf = buf[_FRAME_HEADER.size :]

                if length > _MAX_FRAME_BYTES:
                    logger.warning(
                        "[ENV_SERVER] frame length %d exceeds cap %d; closing",
                        length,
                        _MAX_FRAME_BYTES,
                    )
                    return

                while len(buf) < length:
                    chunk = conn.recv(_RECV_CHUNK_BYTES)
                    if not chunk:
                        return
                    buf += chunk
                body = buf[:length]
                buf = buf[length:]

                reply_body = self._handle_frame(
                    body, created_instance_ids=created_instance_ids
                )
                try:
                    conn.sendall(_frame(reply_body))
                except (BrokenPipeError, ConnectionResetError):
                    return
        finally:
            for instance_id in tuple(created_instance_ids):
                try:
                    self._destroy_instance(instance_id)
                except Exception:
                    logger.exception(
                        "[ENV_SERVER] error destroying instance %d during cleanup",
                        instance_id,
                    )
            created_instance_ids.clear()
            try:
                conn.close()
            except OSError:
                pass

    # ---- socket / shutdown -------------------------------------------

    def _bind_socket(self) -> socket.socket:
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError as exc:
                logger.warning(
                    "[ENV_SERVER] Could not remove stale socket %s: %s",
                    self.socket_path,
                    exc,
                )
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.socket_path))
        os.chmod(str(self.socket_path), 0o666)
        srv.listen(_LISTEN_BACKLOG)
        return srv

    def stop(self) -> None:
        """Signal the accept loop to exit. Safe from a signal handler / thread."""
        self._shutdown.set()
        if self._listen_sock is not None:
            try:
                self._listen_sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def serve_forever(self) -> None:
        self._listen_sock = self._bind_socket()
        logger.info("[ENV_SERVER] Listening on %s", self.socket_path)
        try:
            while not self._shutdown.is_set():
                try:
                    conn, _ = self._listen_sock.accept()
                except OSError:
                    if self._shutdown.is_set():
                        break
                    raise
                t = threading.Thread(
                    target=self._handle_connection, args=(conn,), daemon=True
                )
                t.start()
        finally:
            try:
                self._listen_sock.close()
            except OSError:
                pass
            if self.socket_path.exists():
                try:
                    self.socket_path.unlink()
                except OSError:
                    pass


def _install_signal_handlers(server: EnvServer) -> None:
    def _shutdown(signum, _frame):
        logger.info("[ENV_SERVER] Received signal %d; shutting down", signum)
        server.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)


def _prepend_env_deps() -> None:
    """Put the root-only env-only site-packages (``/mcp_server/env_deps``) on
    ``sys.path`` so the hidden env can import server-only deps the agent cannot.
    Idempotent; a no-op when the directory is absent."""
    deps = str(ENV_DEPS_DIR)
    if ENV_DEPS_DIR.is_dir() and deps not in sys.path:
        sys.path.insert(0, deps)
        logger.info("[ENV_SERVER] env-only deps on sys.path: %s", deps)


def serve() -> None:
    """Entry point for ``python -m env_server``: set up logging + signal handlers
    and loop until told to stop. A startup failure logs and exits non-zero."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    try:
        # Server-only deps must be importable BEFORE the hidden env loads.
        _prepend_env_deps()
        config = EnvConfig.load()
        server = EnvServer(config, SOCKET_PATH)
    except Exception:
        logger.exception("[ENV_SERVER] Failed to start")
        sys.exit(1)

    _install_signal_handlers(server)
    server.serve_forever()
