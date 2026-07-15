"""Out-of-process runner for submitted policy modules.

Use this helper when a task asks the agent to submit executable Python such as
``/tmp/output/policy.py``. The grader process keeps hidden fixtures and scoring
state, while the submitted policy runs in a child process that only receives
public observations and returns actions.

The worker speaks a length-prefixed MessagePack wire (the same
``env_server.protocol`` used by the hidden-env RPC server), so observations and
actions can carry numpy arrays, torch/JAX tensors, gym Spaces, pandas frames,
pydantic models, and dataclasses without a base64 / pickle hop. The response
channel is a dedicated pipe (separate from stdout) so a submitted policy cannot
forge a result by printing to stdout.

Two entry points:

* :class:`PolicyWorker` -- a long-lived single-object handle exposing
  ``act`` / ``call`` (used by most sim-control tasks).
* :func:`load_submitted_policy` -- ML_Envs-style loader that supports a named
  factory (``load_policy``), a source mode (grader-supplied wrapper for an agent
  data artifact), and multi-policy factories returning a dict / list of
  proxies. Returns a :class:`PolicyHandle` (or dict / list of them).
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import pwd
import queue
import select
import stat
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from env_server.protocol import _FRAME_HEADER, _frame, _pack, _unpack
from grading.faults import AgentFault

logger = logging.getLogger(__name__)


class PolicyWorkerError(RuntimeError):
    """Raised when the submitted policy worker fails for non-agent reasons."""


class PolicyTimeoutError(PolicyWorkerError, TimeoutError):
    """Raised when a submitted policy call (load or predict) exceeds its timeout.

    Subclasses BOTH ``PolicyWorkerError`` (a ``RuntimeError``) and the builtin
    ``TimeoutError`` (an ``OSError``) so a per-call hang and a worker death
    converge to a single catchable type: a migrated grader's ``except
    RuntimeError: return 0.0`` (the ML_Envs idiom), the h5 helpers' ``except
    TimeoutError``, and the init handshake's ``except PolicyWorkerError`` all
    catch it. A bare builtin ``TimeoutError`` here would escape ``except
    RuntimeError`` and be routed to ``env_internal_failure`` -> the rollout is
    DISCARDED instead of kept at a real 0.0, handing the agent a free veto for a
    model that deliberately hangs.
    """


# CLONE_NEWIPC: a private SysV-IPC/POSIX-message namespace. Used by the k-fold
# loader so a submitted model cannot stash cross-fold state (e.g. the held-out
# labels it just saw) in a shared-memory segment that survives a fresh worker.
_CLONE_NEWIPC = 0x08000000
try:
    _LIBC: ctypes.CDLL | None = ctypes.CDLL(None, use_errno=True)
    _CLONE_NEWIPC_ARG = ctypes.c_int(_CLONE_NEWIPC)
except Exception:  # pragma: no cover - libc always present on Linux
    _LIBC = None
    _CLONE_NEWIPC_ARG = None


_AGENT_USER = "agent"
_AGENT_USER_ENV = "RUBRIC_AGENT_USER"
_AGENT_UID_ENV = "RUBRIC_AGENT_UID"
_AGENT_GID_ENV = "RUBRIC_AGENT_GID"
_AGENT_HOME_ENV = "RUBRIC_AGENT_HOME"
# Overridable home used to locate the agent's user-site (where a policy may
# import sibling modules it pip-installed --user during the rollout).
_AGENT_HOME_OVERRIDE_ENV = "TAIGA_AGENT_HOME"
_SANDBOX_UID_FALLBACK = 1000

# Size cap so an agent cannot plant a huge file that OOM-kills the root grader
# when it is read whole (mirrors helpers.load_submission_or_fault's 64 MiB cap).
_MAX_POLICY_BYTES = 64 * 1024 * 1024

# Secrets provisioned for the grading-side LLM judge (the Anthropic proxy
# credentials) must never reach the submitted policy child.
_SECRET_ENV_PREFIXES = ("ANTHROPIC_",)
_SECRET_ENV_SUBSTRINGS = (
    "API_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
)


def _scrubbed_environ() -> dict[str, str]:
    """Copy of os.environ with grading-side secrets removed."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_SECRET_ENV_PREFIXES)
        and not any(token in key.upper() for token in _SECRET_ENV_SUBSTRINGS)
    }


def _isolated_python_environ() -> dict[str, str]:
    """Environment for running submitted Python without inherited import paths."""
    env = _scrubbed_environ()
    env.pop("PYTHONPATH", None)
    env["PYTHONSAFEPATH"] = "1"
    return env


def _agent_name() -> str:
    return os.environ.get(_AGENT_USER_ENV) or _AGENT_USER


def _agent_id_from_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise PolicyWorkerError(f"{name} must be an integer") from exc
    if value <= 0:
        raise PolicyWorkerError(f"{name} must identify a non-root account")
    return value


def _agent_identity() -> tuple[int, int, str, str]:
    """Resolve the unprivileged account required for root-run policies."""
    name = _agent_name()
    uid = _agent_id_from_env(_AGENT_UID_ENV)
    gid = _agent_id_from_env(_AGENT_GID_ENV)
    if (uid is None) != (gid is None):
        raise PolicyWorkerError(
            f"{_AGENT_UID_ENV} and {_AGENT_GID_ENV} must be set together"
        )
    if uid is not None and gid is not None:
        try:
            account = pwd.getpwuid(uid)
            home = account.pw_dir
            login = account.pw_name
        except KeyError:
            home = f"/home/{name}"
            login = name
        home = os.environ.get(_AGENT_HOME_ENV) or home
        login = os.environ.get(_AGENT_USER_ENV) or login
        return uid, gid, home, login
    try:
        account = pwd.getpwnam(name)
    except KeyError as exc:
        raise PolicyWorkerError(
            f"cannot drop privileges: user {name!r} not found; set "
            f"{_AGENT_USER_ENV} or {_AGENT_UID_ENV}/{_AGENT_GID_ENV}"
        ) from exc
    if account.pw_uid <= 0 or account.pw_gid <= 0:
        raise PolicyWorkerError(f"cannot drop privileges to root account {name!r}")
    return account.pw_uid, account.pw_gid, account.pw_dir, account.pw_name


def _agent_preexec(
    uid: int, gid: int, ipc_status_fd: int | None = None
) -> Callable[[], None]:
    """``preexec_fn`` that unshares the IPC namespace WHILE STILL ROOT, then
    drops to the agent account. ``unshare(CLONE_NEWIPC)`` needs CAP_SYS_ADMIN so
    it must run before ``setuid``; the unshare is best-effort -- a runtime
    without CAP_SYS_ADMIN must still grade honest policies, so a failure is
    never raised.

    The unshare result is surfaced (not silently dropped): when
    ``ipc_status_fd`` is given, a single status byte is written to it -- ``0``
    when isolation took, otherwise the failing ``errno`` (capped at 255). This
    runs in the forked child between fork and exec where normal logging is
    unsafe, so we only do a raw ``os.write``; the parent reads the byte and logs
    a warning when isolation did not take (mirrors ML_Envs' "report, never
    abort, parent logs a warning").
    """

    def _preexec() -> None:
        status = 0
        if _LIBC is not None and _CLONE_NEWIPC_ARG is not None:
            ctypes.set_errno(0)
            if _LIBC.unshare(_CLONE_NEWIPC_ARG) != 0:  # best-effort; never raise
                status = ctypes.get_errno() or errno.EPERM
        else:
            status = errno.ENOSYS
        if ipc_status_fd is not None:
            try:
                os.write(ipc_status_fd, bytes([min(status, 255)]))
            except OSError:
                pass
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)

    return _preexec


def _agent_drop_kwargs(
    *, unshare_ipc: bool = False, ipc_status_fd: int | None = None
) -> dict[str, Any]:
    """Popen kwargs dropping a child to the unprivileged agent account.

    When the grader is root, submitted code must not inherit root. Missing or
    invalid drop identity fails closed before the policy worker starts. When
    ``unshare_ipc`` is set (the default for submitted-policy workers), the drop
    runs in a ``preexec_fn`` so the worker can unshare its IPC namespace before
    ``setuid``; otherwise the faster C-level ``Popen(user=...)`` drop is used.

    ``ipc_status_fd`` (only meaningful with ``unshare_ipc``) is the write end of
    a pipe the preexec hook reports the unshare result on, so the parent can log
    a warning when isolation did not take.
    """
    env = _isolated_python_environ()
    kwargs: dict[str, Any] = {"env": env}
    if os.geteuid() != 0:
        return kwargs
    uid, gid, home, name = _agent_identity()
    env["HOME"] = home
    env["USER"] = env["LOGNAME"] = name
    if unshare_ipc:
        kwargs["preexec_fn"] = _agent_preexec(uid, gid, ipc_status_fd)
    else:
        kwargs.update(user=uid, group=gid, extra_groups=[])
    return kwargs


def _agent_user_site_dirs() -> list[str]:
    """User-site dirs of the sandboxed agent account, so a path-mode policy can
    import sibling modules the agent pip-installed --user during the rollout.

    The grading worker inherits the root grader's HOME, so Python never
    auto-discovers the agent's user-site; we add it explicitly (post-drop, via a
    plain ``sys.path.insert`` that does NOT execute any agent ``.pth`` file).
    """
    import glob

    home = os.environ.get(_AGENT_HOME_OVERRIDE_ENV)
    if not home:
        uid = None
        raw = os.environ.get(_AGENT_UID_ENV)
        if raw:
            try:
                uid = int(raw)
            except ValueError:
                uid = None
        try:
            home = (
                pwd.getpwuid(uid).pw_dir if uid else pwd.getpwnam(_agent_name()).pw_dir
            )
        except KeyError:
            return []
    pattern = os.path.join(home, ".local", "lib", "python*", "site-packages")
    return [d for d in sorted(glob.glob(pattern)) if os.path.isdir(d)]


_WORKER_SOURCE = r"""
import sys

# Defense-in-depth import isolation. MUST run before any non-builtin import: the
# worker is launched via ``python -c``, which puts the (agent-writable) cwd on
# sys.path as ''. Drop the cwd entries (and caller-supplied scrub paths) up front
# so a submitted policy cannot plant json.py / os.py to shadow the modules the
# worker uses to serialize its response across the privilege boundary.
_unsafe = {p for p in sys.argv[2:] if p}
sys.path[:] = [p for p in sys.path if p not in ("", ".") and p not in _unsafe]

import contextlib
import dataclasses
import datetime
import enum
import importlib.util
import msgpack
import os
import struct
import tempfile
import traceback
import types
from pathlib import Path

_FRAME_HEADER = struct.Struct(">I")
_MAX_FRAME_BYTES = 1 << 30
_EXT_NDARRAY = 1
_EXT_GYM_SPACE = 2
_EXT_TORCH_TENSOR = 3
_EXT_JAX_ARRAY = 4


def _ndarray_payload(arr, np_module):
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np_module.ascontiguousarray(arr)
    return msgpack.packb(
        [str(arr.dtype), list(arr.shape), arr.tobytes(order="C")],
        use_bin_type=True,
    )


def _ndarray_to_ext(arr, np_module):
    return msgpack.ExtType(_EXT_NDARRAY, _ndarray_payload(arr, np_module))


def _payload_to_ndarray(payload, np_module):
    dtype_str, shape, raw = msgpack.unpackb(payload, raw=False)
    return np_module.frombuffer(raw, dtype=dtype_str).reshape(shape).copy()


def _default(obj):
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        if isinstance(obj, np.ndarray):
            return _ndarray_to_ext(obj, np)
        if isinstance(obj, np.generic):
            return obj.item()

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and np is not None and isinstance(obj, torch.Tensor):
        return msgpack.ExtType(
            _EXT_TORCH_TENSOR, _ndarray_payload(obj.detach().cpu().numpy(), np)
        )

    try:
        import jax
    except ImportError:
        jax = None
    if jax is not None and np is not None and isinstance(obj, jax.Array):
        return msgpack.ExtType(_EXT_JAX_ARRAY, _ndarray_payload(np.asarray(obj), np))

    try:
        import gymnasium as _gym
    except ImportError:
        try:
            import gym as _gym
        except ImportError:
            _gym = None
    if _gym is not None and np is not None:
        spaces = getattr(_gym, "spaces", None)
        if spaces is not None:
            payload = None
            if isinstance(obj, spaces.Box):
                payload = {
                    "_kind": "Box",
                    "low": _ndarray_to_ext(obj.low, np),
                    "high": _ndarray_to_ext(obj.high, np),
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }
            elif isinstance(obj, spaces.Discrete):
                payload = {
                    "_kind": "Discrete",
                    "n": int(obj.n),
                    "start": int(getattr(obj, "start", 0)),
                }
            elif isinstance(obj, spaces.MultiDiscrete):
                payload = {
                    "_kind": "MultiDiscrete",
                    "nvec": _ndarray_to_ext(np.asarray(obj.nvec), np),
                }
            elif isinstance(obj, spaces.MultiBinary):
                payload = {
                    "_kind": "MultiBinary",
                    "n": int(obj.n) if isinstance(obj.n, (int, np.integer)) else list(obj.n),
                }
            elif isinstance(obj, spaces.Dict):
                payload = {"_kind": "Dict", "spaces": dict(obj.spaces)}
            elif isinstance(obj, spaces.Tuple):
                payload = {"_kind": "Tuple", "spaces": list(obj.spaces)}
            if payload is not None:
                return msgpack.ExtType(
                    _EXT_GYM_SPACE,
                    msgpack.packb(payload, default=_default, use_bin_type=True),
                )

    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()

    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()

    try:
        from pydantic import BaseModel as _PydBaseModel
    except ImportError:
        _PydBaseModel = None
    if _PydBaseModel is not None and isinstance(obj, _PydBaseModel):
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            try:
                return obj.model_dump()
            except Exception:
                pass
        if hasattr(obj, "dict") and callable(obj.dict):
            try:
                return obj.dict()
            except Exception:
                pass

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if isinstance(obj, (bytearray, memoryview)):
        return bytes(obj)

    raise TypeError(
        "object of type %r is not msgpack-serializable" % type(obj).__name__
    )


def _raw_ndarray_payload(data):
    dtype_str, shape, raw = msgpack.unpackb(data, raw=False)
    return {"_type": "ndarray", "dtype": dtype_str, "shape": list(shape), "raw": raw}


def _ext_hook(code, data):
    if code == _EXT_NDARRAY:
        try:
            import numpy as np
        except ImportError:
            return _raw_ndarray_payload(data)
        return _payload_to_ndarray(data, np)
    if code == _EXT_TORCH_TENSOR:
        try:
            import numpy as np
        except ImportError:
            return _raw_ndarray_payload(data)
        arr = _payload_to_ndarray(data, np)
        try:
            import torch
        except ImportError:
            return arr
        return torch.from_numpy(arr)
    if code == _EXT_JAX_ARRAY:
        try:
            import numpy as np
        except ImportError:
            return _raw_ndarray_payload(data)
        arr = _payload_to_ndarray(data, np)
        try:
            import jax.numpy as jnp
        except ImportError:
            return arr
        return jnp.asarray(arr)
    if code == _EXT_GYM_SPACE:
        spec = msgpack.unpackb(data, raw=False, ext_hook=_ext_hook)
        kind = spec.pop("_kind", None)
        try:
            import gymnasium as _gym
        except ImportError:
            try:
                import gym as _gym
            except ImportError:
                _gym = None
        if _gym is None:
            spec["_kind"] = kind
            spec["_type"] = "gym.spaces.%s" % kind if kind else "gym.spaces.Unknown"
            return spec
        spaces_mod = _gym.spaces
        try:
            import numpy as np
        except ImportError:
            np = None
        if kind == "Box":
            kwargs = {
                "low": spec["low"],
                "high": spec["high"],
                "shape": tuple(spec["shape"]),
            }
            if np is not None:
                kwargs["dtype"] = np.dtype(spec["dtype"])
            return spaces_mod.Box(**kwargs)
        if kind == "Discrete":
            start_value = spec.get("start", 0)
            try:
                return spaces_mod.Discrete(spec["n"], start=start_value)
            except TypeError:
                return spaces_mod.Discrete(spec["n"])
        if kind == "MultiDiscrete":
            return spaces_mod.MultiDiscrete(spec["nvec"])
        if kind == "MultiBinary":
            return spaces_mod.MultiBinary(spec["n"])
        if kind == "Dict":
            return spaces_mod.Dict(spec["spaces"])
        if kind == "Tuple":
            return spaces_mod.Tuple(spec["spaces"])
        spec["_kind"] = kind
        return spec
    return msgpack.ExtType(code, data)


def _pack(obj):
    return msgpack.packb(obj, default=_default, use_bin_type=True)


def _unpack(data):
    return msgpack.unpackb(data, raw=False, ext_hook=_ext_hook)


def _frame(body):
    if len(body) > _MAX_FRAME_BYTES:
        raise ValueError("frame body exceeds %d bytes (%d)" % (_MAX_FRAME_BYTES, len(body)))
    return _FRAME_HEADER.pack(len(body)) + body

_PROTO_FD = int(sys.argv[1])
_proto = os.fdopen(_PROTO_FD, "wb", buffering=0)
_stdin = sys.stdin.buffer

OUTPUT_MODEL_XML = None


def _read_frame(stream):
    header = b""
    while len(header) < _FRAME_HEADER.size:
        chunk = stream.read(_FRAME_HEADER.size - len(header))
        if not chunk:
            return None
        header += chunk
    (length,) = _FRAME_HEADER.unpack(header)
    body = b""
    while len(body) < length:
        chunk = stream.read(length - len(body))
        if not chunk:
            return None
        body += chunk
    return body


def _send(obj):
    try:
        payload = _pack(obj)
    except Exception as exc:
        payload = _pack({
            "ok": False,
            "error": "unencodable response: %s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(limit=8),
        })
    _proto.write(_frame(payload))


def _auto_factory(module):
    if hasattr(module, "act"):
        return module
    if hasattr(module, "Policy"):
        return module.Policy()
    return module


def _load_root(cfg):
    mode = cfg.get("mode", "path")
    factory_name = cfg.get("factory")
    for d in reversed([str(x) for x in (cfg.get("sys_path_dirs") or [])]):
        sys.path.insert(0, d)
    if mode == "source":
        filename = cfg.get("path") or "<policy-source>"
        code = compile(cfg.get("source"), filename, "exec")
        module = types.ModuleType("submitted_policy")
        module.__file__ = filename
        sys.modules["submitted_policy"] = module
        with contextlib.redirect_stdout(sys.stderr):
            exec(code, module.__dict__)
    else:
        filename = cfg["path"]
        spec = importlib.util.spec_from_file_location("submitted_policy", filename)
        if spec is None or spec.loader is None:
            raise ImportError("cannot import policy from %s" % filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules["submitted_policy"] = module
        with contextlib.redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
    if factory_name:
        factory = getattr(module, factory_name, None)
        if not callable(factory):
            raise AttributeError(
                "%s must define a callable %s()" % (filename, factory_name)
            )
        return factory()
    return _auto_factory(module)


def _install_output_model_xml(xml_text):
    out = Path("/tmp/output")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError:
        out = Path(tempfile.mkdtemp(prefix="policy-worker-output-")) / "output"
        out.mkdir(parents=True, exist_ok=True)
    p = out / "model.xml"
    p.write_text(xml_text)
    return p


def _resolve(root, path):
    obj = root
    for key in path:
        obj = obj[key]
    return obj


_init = _read_frame(_stdin)
if _init is None:
    sys.exit(0)
# Two-phase handshake. Emit a pre-agent ack AFTER trusted worker setup but
# BEFORE the submitted module is exec'd. The parent uses its presence to decide
# blame: no ack means the worker died in trusted setup (env_internal_failure ->
# discard); ack-then-death means the agent killed its own loader -- including a
# no-frame os._exit/SIGKILL/segfault -- and earns a kept 0.0 (AgentFault).
_cfg = _unpack(_init)
_send({"phase": "ready"})
try:
    root = _load_root(_cfg)
    _send({"ok": True})
except BaseException as exc:
    _send({
        "ok": False,
        "error": "%s: %s" % (type(exc).__name__, exc),
        "traceback": traceback.format_exc(),
    })
    sys.exit(0)


while True:
    body = _read_frame(_stdin)
    if body is None:
        break
    try:
        req = _unpack(body)
        op = req.get("op", "call")
        path = tuple(req.get("path") or ())
        if op == "init_model_xml":
            OUTPUT_MODEL_XML = str(req.get("xml"))
            _send({"ok": True, "result": True})
            continue
        target = _resolve(root, path)
        if op == "hasattr":
            _send({"ok": True, "result": hasattr(target, req.get("name"))})
            continue
        if op == "keys":
            if isinstance(target, dict):
                _send({"ok": True, "result": list(target.keys()), "kind": "dict"})
            elif isinstance(target, (list, tuple)):
                _send({"ok": True, "result": list(range(len(target))), "kind": "seq"})
            else:
                _send({"ok": True, "result": None, "kind": "scalar"})
            continue
        method = req.get("method", "act")
        args = req.get("args") or []
        kwargs = req.get("kwargs") or {}
        if not isinstance(method, str) or not method:
            raise ValueError("request.method must be a non-empty string")
        fn = getattr(target, method)
        if not callable(fn):
            raise TypeError("%r on target is not callable" % method)
        with contextlib.redirect_stdout(sys.stderr):
            if OUTPUT_MODEL_XML is None:
                result = fn(*args, **kwargs)
            else:
                model_path = _install_output_model_xml(OUTPUT_MODEL_XML)
                old_exists = os.path.exists
                old_open = Path.open
                old_cwd = os.getcwd()

                def _exists(p, _o=old_exists):
                    if str(p) == "/tmp/output/model.xml":
                        return True
                    return _o(p)

                def _open(self, *a, _o=old_open, _m=model_path, **k):
                    if str(self) == "/tmp/output/model.xml":
                        return _o(_m, *a, **k)
                    return _o(self, *a, **k)

                os.path.exists = _exists
                Path.open = _open
                try:
                    os.chdir(model_path.parent)
                    result = fn(*args, **kwargs)
                finally:
                    os.chdir(old_cwd)
                    os.path.exists = old_exists
                    Path.open = old_open
        _send({"ok": True, "result": result})
    except Exception:
        _send({"ok": False, "error": traceback.format_exc(limit=8)})
"""


# Generous floor (seconds) for the first call / init handshake when the caller
# does not pin ``first_call_timeout_s``. Loading the policy module (and heavy
# deps like MuJoCo / torch) happens during the init handshake, so a tight
# per-step ``timeout_s`` must not race that one-time startup cost.
_FIRST_CALL_FLOOR_S = 30.0


class PolicyWorker:
    """Call a submitted ``policy.py`` through a narrow msgpack observation/action
    API in a privilege-dropped child process."""

    def __init__(
        self,
        policy_path: Path,
        *,
        timeout_s: float = 5.0,
        first_call_timeout_s: float | None = None,
        cwd: Path | None = None,
        max_stderr_chars: int = 8000,
        drop_privileges: bool = True,
        unshare_ipc: bool = True,
        factory_name: str | None = None,
        source: str | bytes | None = None,
        sys_path_dirs: list[str | Path] | None = None,
    ) -> None:
        self.policy_path = Path(policy_path)
        self.timeout_s = timeout_s
        self.first_call_timeout_s = first_call_timeout_s
        self._first_call_done = False
        self.cwd = Path(cwd) if cwd is not None else None
        self.max_stderr_chars = max_stderr_chars
        self.drop_privileges = drop_privileges
        self.unshare_ipc = unshare_ipc
        # Named factory (e.g. "load_policy"); None keeps the legacy auto-detect
        # (module-level ``act`` or a ``Policy`` class).
        self.factory_name = factory_name
        # Source mode: a grader-supplied literal wrapper (str/bytes) executed in
        # the worker instead of reading ``policy_path``.
        self.source = source
        self.sys_path_dirs = sys_path_dirs
        self._proc: subprocess.Popen[bytes] | None = None
        self._responses: queue.Queue[bytes | None] = queue.Queue()
        self._stderr_parts: list[str] = []
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._proto_stream: Any = None

    def __enter__(self) -> "PolicyWorker":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __call__(self, obs: Any) -> Any:
        return self.act(obs)

    @staticmethod
    def _warn_if_ipc_unisolated(status_fd: int) -> None:
        """Read the preexec IPC-unshare status byte and warn if isolation did
        not take. Best-effort: never raises, so an un-isolated runtime still
        grades honest policies (it just logs that the defense is off)."""
        data = b""
        try:
            # The byte is written before exec, hence before Popen returned; a
            # short select guards against a hang if the child died pre-write.
            if select.select([status_fd], [], [], 5.0)[0]:
                data = os.read(status_fd, 1)
        except OSError:
            data = b""
        finally:
            try:
                os.close(status_fd)
            except OSError:
                pass
        if not data:
            logger.warning(
                "[GRADING] could not determine policy worker IPC isolation "
                "status; proceeding without verifying CLONE_NEWIPC"
            )
            return
        status = data[0]
        if status != 0:
            logger.warning(
                "[GRADING] policy worker IPC namespace NOT isolated: "
                "unshare(CLONE_NEWIPC) failed (errno=%d %s); proceeding without "
                "per-worker IPC isolation",
                status,
                os.strerror(status),
            )

    def start(self) -> None:
        if self._proc is not None:
            return
        if self.source is None and not self.policy_path.exists():
            raise FileNotFoundError(f"missing policy file: {self.policy_path}")
        self._responses = queue.Queue()
        self._stderr_parts = []
        self._first_call_done = False

        # IPC-isolation status pipe: when the drop runs through the preexec hook
        # (root + unshare_ipc), the hook reports the unshare(CLONE_NEWIPC) result
        # on the write end so we can warn if isolation did not take. Only created
        # when the unshare actually runs (preexec path); skipped on the C-level
        # drop and on non-root, where no unshare is attempted.
        ipc_status_r: int | None = None
        ipc_status_w: int | None = None
        if self.drop_privileges and self.unshare_ipc and os.geteuid() == 0:
            ipc_status_r, ipc_status_w = os.pipe()

        popen_kwargs = (
            _agent_drop_kwargs(
                unshare_ipc=self.unshare_ipc, ipc_status_fd=ipc_status_w
            )
            if self.drop_privileges
            else {"env": _isolated_python_environ()}
        )
        unsafe_sys_paths = self._unsafe_sys_path_args()

        # Dedicated pipe for the msgpack protocol so C-level output (e.g. MuJoCo
        # warnings) on stdout cannot corrupt or forge the response stream.
        proto_read_fd, proto_write_fd = os.pipe()
        pass_fds = (
            (proto_write_fd,)
            if ipc_status_w is None
            else (proto_write_fd, ipc_status_w)
        )
        try:
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    "-P",
                    "-u",
                    "-c",
                    _WORKER_SOURCE,
                    str(proto_write_fd),
                    *unsafe_sys_paths,
                ],
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                pass_fds=pass_fds,
                **popen_kwargs,
            )
        except BaseException:
            os.close(proto_read_fd)
            os.close(proto_write_fd)
            if ipc_status_r is not None:
                os.close(ipc_status_r)
            if ipc_status_w is not None:
                os.close(ipc_status_w)
            raise

        os.close(proto_write_fd)
        if ipc_status_w is not None:
            os.close(ipc_status_w)
        if ipc_status_r is not None:
            # The preexec hook writes the status byte before exec, so it is
            # available by the time Popen returns. Read it now (before the
            # handshake / any agent code) and surface a warning if needed.
            self._warn_if_ipc_unisolated(ipc_status_r)
        self._proto_stream = os.fdopen(proto_read_fd, "rb", buffering=0)

        assert self._proc.stdout is not None
        self._reader_thread = threading.Thread(
            target=self._drain_responses, args=(self._proto_stream,), daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self._proc.stdout,), daemon=True
        )
        self._reader_thread.start()
        self._stderr_thread.start()

        try:
            self._handshake()
        except BaseException:
            # A failed handshake means the worker exited (or never replied). Tear
            # it down and clear _proc so a later start() respawns instead of
            # returning early on a dead child.
            self.kill()
            raise

    def _handshake(self) -> None:
        """Send the load config and wait for the worker's init reply within the
        first-call budget (module import / heavy deps load here)."""
        cfg: dict[str, Any] = {
            "mode": "source" if self.source is not None else "path",
            "path": str(self.policy_path),
            "factory": self.factory_name,
            "sys_path_dirs": self._resolved_sys_path_dirs(),
        }
        if self.source is not None:
            cfg["source"] = (
                self.source
                if isinstance(self.source, (str, bytes))
                else str(self.source)
            )
        self._write_frame(cfg)
        # Phase 1: pre-agent ack. The worker sends this after trusted setup but
        # before it execs the submitted module, so its absence means the worker
        # died in trusted setup -> genuine infra failure (discarded rollout).
        try:
            ack = self._read_reply(
                self._effective_timeout(), what="init handshake (pre-agent ack)"
            )
        except (PolicyWorkerError, TimeoutError) as exc:
            raise PolicyWorkerError(
                self._error_context(
                    "policy worker died during setup, before running the "
                    "submitted module"
                )
            ) from exc
        if not isinstance(ack, dict) or ack.get("phase") != "ready":
            raise PolicyWorkerError(
                self._error_context(f"unexpected pre-agent handshake frame: {ack!r}")
            )
        # Phase 2: load result, emitted AFTER the submitted module was entered.
        # Any failure now is the agent's doing -> AgentFault (kept 0.0): a missing
        # frame from a hard os._exit/SIGKILL/segfault, a post-ack timeout (the
        # agent hanging its own import), or an ok=False frame with a traceback.
        try:
            reply = self._read_reply(
                self._effective_timeout(), what="init handshake (load result)"
            )
        except (PolicyWorkerError, TimeoutError) as exc:
            raise AgentFault(
                "policy failed to load: the submitted module killed or hung its "
                "loader before returning a result "
                "(e.g. os._exit/SIGKILL/segfault/hang at import)"
            ) from exc
        self._first_call_done = True
        if not isinstance(reply, dict) or not reply.get("ok"):
            err = reply.get("error", "unknown") if isinstance(reply, dict) else reply
            tb = reply.get("traceback") if isinstance(reply, dict) else None
            message = f"policy failed to load: {err}" + (f"\n{tb}" if tb else "")
            raise AgentFault(message)

    def _resolved_sys_path_dirs(self) -> list[str]:
        if self.sys_path_dirs is not None:
            return [str(p) for p in self.sys_path_dirs]
        if self.source is not None:
            # Source mode: do NOT add the agent-writable dir to sys.path -- the
            # grader's wrapper decides what is importable.
            return []
        # Path mode: the agent's policy.py legitimately imports its own siblings
        # plus deps it pip-installed --user during the rollout.
        return [str(self.policy_path.parent), *_agent_user_site_dirs()]

    def _unsafe_sys_path_args(self) -> list[str]:
        if self.cwd is None:
            return []
        paths = {str(self.cwd)}
        try:
            paths.add(str(self.cwd.resolve()))
        except OSError:
            pass
        return sorted(paths)

    def act(self, obs: Any) -> Any:
        return self.call("act", obs)

    def init_model_xml(self, xml_text: str) -> None:
        self.start()
        self._rpc({"op": "init_model_xml", "xml": str(xml_text)})

    def keys(self, _path: tuple = ()) -> Any:
        self.start()
        reply = self._rpc({"op": "keys", "path": list(_path)})
        return reply

    def has(self, name: str, _path: tuple = ()) -> bool:
        self.start()
        reply = self._rpc({"op": "hasattr", "path": list(_path), "name": name})
        return bool(reply.get("result"))

    def call(self, method: str, *args: Any, _path: tuple = (), **kwargs: Any) -> Any:
        self.start()
        reply = self._rpc(
            {
                "op": "call",
                "path": list(_path),
                "method": method,
                "args": list(args),
                "kwargs": dict(kwargs),
            }
        )
        return reply.get("result")

    def _rpc(self, request: dict) -> dict:
        proc = self._require_process()
        if proc.stdin is None:
            raise PolicyWorkerError("policy worker stdin is closed")
        try:
            self._write_frame(request)
        except BrokenPipeError as exc:
            raise PolicyWorkerError(
                self._error_context("policy worker exited")
            ) from exc
        method = request.get("method") or request.get("op")
        reply = self._read_reply(self._effective_timeout(), what=str(method))
        self._first_call_done = True
        if not isinstance(reply, dict) or not reply.get("ok"):
            err = (
                reply.get("error") if isinstance(reply, dict) else None
            ) or "policy worker error"
            raise PolicyWorkerError(str(err))
        return reply

    def _write_frame(self, obj: Any) -> None:
        proc = self._require_process()
        assert proc.stdin is not None
        proc.stdin.write(_frame(_pack(obj)))
        proc.stdin.flush()

    def _read_reply(self, timeout_s: float, *, what: str) -> Any:
        try:
            frame = self._responses.get(timeout=timeout_s)
        except queue.Empty as exc:
            self.kill()
            raise PolicyTimeoutError(
                f"policy.{what} timed out after {timeout_s:.3f}s"
            ) from exc
        if frame is None:
            raise PolicyWorkerError(self._error_context("policy worker exited"))
        return _unpack(frame)

    def _effective_timeout(self) -> float:
        if self._first_call_done:
            return self.timeout_s
        if self.first_call_timeout_s is not None:
            return self.first_call_timeout_s
        return max(self.timeout_s, _FIRST_CALL_FLOOR_S)

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.kill()
        self._close_proto_stream()
        self._proc = None

    def kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=1.0)
        self._close_proto_stream()
        self._proc = None

    def _close_proto_stream(self) -> None:
        stream = self._proto_stream
        if stream is None:
            return
        try:
            stream.close()
        except OSError:
            pass
        self._proto_stream = None

    def stderr(self) -> str:
        text = "".join(self._stderr_parts)
        if len(text) <= self.max_stderr_chars:
            return text
        return text[-self.max_stderr_chars :]

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._proc is None:
            raise PolicyWorkerError("policy worker is not started")
        return self._proc

    def _error_context(self, message: str) -> str:
        stderr = self.stderr().strip()
        return f"{message}: {stderr}" if stderr else message

    def _drain_responses(self, stream: Any) -> None:
        try:
            while True:
                header = stream.read(_FRAME_HEADER.size)
                if not header or len(header) < _FRAME_HEADER.size:
                    break
                (length,) = _FRAME_HEADER.unpack(header)
                body = b""
                while len(body) < length:
                    chunk = stream.read(length - len(body))
                    if not chunk:
                        break
                    body += chunk
                if len(body) < length:
                    break
                self._responses.put(body)
        finally:
            self._responses.put(None)

    def _drain_stderr(self, stream: Any) -> None:
        for chunk in stream:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", "replace")
            self._stderr_parts.append(chunk)


class PolicyHandle:
    """Path-addressed handle to a (possibly nested) submitted object held by a
    shared :class:`PolicyWorker`.

    ``handle.method(*args, **kwargs)`` forwards an RPC at this handle's path;
    ``handle["grid"]`` returns a child handle; ``hasattr(handle, name)`` round
    trips once and is cached. Used by :func:`load_submitted_policy` for
    multi-policy (dict / list) factories.
    """

    __slots__ = ("_worker", "_path", "_attr_cache")

    def __init__(self, worker: PolicyWorker, path: tuple = ()) -> None:
        self._worker = worker
        self._path = path
        self._attr_cache: dict[str, bool] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        cache = self._attr_cache
        if name not in cache:
            cache[name] = self._worker.has(name, _path=self._path)
        if not cache[name]:
            raise AttributeError(f"submitted policy has no attribute {name!r}")
        path = self._path

        def _bound(*args: Any, **kwargs: Any) -> Any:
            return self._worker.call(name, *args, _path=path, **kwargs)

        return _bound

    def __getitem__(self, key: Any) -> "PolicyHandle":
        return PolicyHandle(self._worker, self._path + (key,))

    def close(self) -> None:
        self._worker.close()


DEFAULT_POLICY_PATH = Path("/tmp/output/policy.py")
DEFAULT_FACTORY_NAME = "load_policy"


def load_submitted_policy(
    path: str | Path = DEFAULT_POLICY_PATH,
    factory_name: str = DEFAULT_FACTORY_NAME,
    *,
    source: str | bytes | None = None,
    sys_path_dirs: list[str | Path] | None = None,
    timeout_s: float = 5.0,
    first_call_timeout_s: float | None = None,
    cwd: str | Path | None = None,
    drop_privileges: bool = True,
    unshare_ipc: bool = True,
) -> Any:
    """Load a submitted policy in a sandboxed worker and return proxy handle(s).

    ML_Envs-style loader. The factory (``load_policy`` by default) may return a
    single object, a dict, or a list:

    * single object -> a :class:`PolicyHandle`,
    * dict          -> ``{key: PolicyHandle}``,
    * list / tuple  -> ``[PolicyHandle, ...]``.

    Path mode (default) reads ``path`` (the agent's file). Source mode (pass a
    LITERAL ``source`` string -- never f-string agent data into it) runs a
    grader-supplied wrapper, e.g. to deserialize an agent-shipped model artifact;
    the wrapper runs in the same privilege-dropped worker.

    Raises :class:`grading.faults.AgentFault` for agent-caused load failures (a
    missing / non-regular / oversized policy file, or the module raising at
    load); the runner records a kept 0.0.
    """
    if source is None:
        p = Path(path)
        if not p.exists():
            raise AgentFault(f"missing submitted policy at {p}")
        st = os.lstat(p)
        if not stat.S_ISREG(st.st_mode):
            raise AgentFault(
                f"submitted policy at {p} is not a regular file "
                f"(mode={stat.filemode(st.st_mode)}); expected a plain file"
            )
        if st.st_size > _MAX_POLICY_BYTES:
            raise AgentFault(
                f"submitted policy at {p} is {st.st_size} bytes, over the "
                f"{_MAX_POLICY_BYTES}-byte limit"
            )

    worker = PolicyWorker(
        Path(path),
        timeout_s=timeout_s,
        first_call_timeout_s=first_call_timeout_s,
        cwd=Path(cwd) if cwd is not None else None,
        drop_privileges=drop_privileges,
        unshare_ipc=unshare_ipc,
        factory_name=factory_name,
        source=source,
        sys_path_dirs=sys_path_dirs,
    )
    worker.start()  # raises AgentFault / PolicyWorkerError on load failure
    kind_reply = worker.keys()
    kind = kind_reply.get("kind")
    if kind == "dict":
        return {k: PolicyHandle(worker, (k,)) for k in kind_reply["result"]}
    if kind == "seq":
        return [PolicyHandle(worker, (i,)) for i in kind_reply["result"]]
    return PolicyHandle(worker)
