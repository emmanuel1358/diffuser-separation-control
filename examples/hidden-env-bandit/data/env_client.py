"""Agent-facing client for the hidden-env bandit task (baked to /data/env_client.py).

Import this from your solution to talk to the hidden bandit over /tmp/env.sock::

    from env_client import BanditEnv

    with BanditEnv() as env:
        n = env.n_arms()
        r = env.pull(0)

This is a trimmed copy of the template's canonical ``env_server/env_client.py``
with a task-specific ``BanditEnv`` wrapper added. Only ``msgpack`` (preinstalled)
plus the stdlib is required. Do NOT change the wire constants.
"""

from __future__ import annotations

import socket
import struct
from typing import Any

import msgpack

SOCKET_PATH = "/tmp/env.sock"
_RECV_CHUNK_BYTES = 65536
_FRAME_HEADER = struct.Struct(">I")
_MAX_FRAME_BYTES = 1 << 30
_EXT_NDARRAY = 1


def _default(obj: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None:
        if isinstance(obj, np.ndarray):
            arr = obj if obj.flags["C_CONTIGUOUS"] else np.ascontiguousarray(obj)
            return msgpack.ExtType(
                _EXT_NDARRAY,
                msgpack.packb(
                    [str(arr.dtype), list(arr.shape), arr.tobytes(order="C")],
                    use_bin_type=True,
                ),
            )
        if isinstance(obj, np.generic):
            return obj.item()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"{type(obj).__name__!r} cannot be sent over the env wire")


def _ext_hook(code: int, data: bytes) -> Any:
    if code == _EXT_NDARRAY:
        import numpy as np

        dtype_str, shape, raw = msgpack.unpackb(data, raw=False)
        return np.frombuffer(raw, dtype=dtype_str).reshape(shape).copy()
    return msgpack.ExtType(code, data)


def _pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_default, use_bin_type=True)


def _unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False, ext_hook=_ext_hook)


class EnvError(Exception):
    def __init__(self, message: str, *, type: str = "Error") -> None:
        super().__init__(message)
        self.type = type


class Env:
    """Generic proxy for one env instance on the server."""

    def __init__(self, instance_id: int = 0, env_kwargs: dict | None = None) -> None:
        self.instance_id = instance_id
        self._env_kwargs = dict(env_kwargs or {})
        self._sock: socket.socket | None = None
        self._buf = b""
        self._created = False
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(SOCKET_PATH)
        sock.settimeout(None)
        self._sock = sock

    def __enter__(self) -> "Env":
        self.call("__create__", env_kwargs=self._env_kwargs)
        self._created = True
        return self

    def __exit__(self, *_exc) -> None:
        try:
            if self._created:
                self.call("__destroy__")
        except Exception:
            pass
        finally:
            if self._sock is not None:
                self._sock.close()
                self._sock = None

    def call(self, method: str, **args) -> Any:
        if self._sock is None:
            raise EnvError("client is closed")
        body = _pack({"method": method, "instance_id": self.instance_id, "args": args})
        self._sock.sendall(_FRAME_HEADER.pack(len(body)) + body)
        resp = _unpack(self._read_frame())
        if not resp.get("ok"):
            err = resp.get("error") or {}
            raise EnvError(err.get("message", ""), type=err.get("type", "Error"))
        return resp["result"]

    def _read_frame(self) -> bytes:
        assert self._sock is not None
        while len(self._buf) < _FRAME_HEADER.size:
            chunk = self._sock.recv(_RECV_CHUNK_BYTES)
            if not chunk:
                raise EnvError("server closed the connection")
            self._buf += chunk
        (length,) = _FRAME_HEADER.unpack(self._buf[: _FRAME_HEADER.size])
        if length > _MAX_FRAME_BYTES:
            raise EnvError(f"oversized frame {length}")
        self._buf = self._buf[_FRAME_HEADER.size :]
        while len(self._buf) < length:
            chunk = self._sock.recv(_RECV_CHUNK_BYTES)
            if not chunk:
                raise EnvError("server closed the connection mid-frame")
            self._buf += chunk
        body = self._buf[:length]
        self._buf = self._buf[length:]
        return body


class BanditEnv(Env):
    """Ergonomic wrapper over the hidden bandit env."""

    def n_arms(self) -> int:
        return self.call("n_arms")

    def reset(self, seed: int | None = None) -> dict:
        return self.call("reset", seed=seed)

    def pull(self, arm: int) -> float:
        return self.call("pull", arm=int(arm))


def estimate_best_arm(pulls_per_arm: int = 200) -> int:
    """Pull every arm ``pulls_per_arm`` times and return the empirical argmax."""
    with BanditEnv() as env:
        n = env.n_arms()
        totals = [0.0] * n
        for arm in range(n):
            for _ in range(pulls_per_arm):
                totals[arm] += env.pull(arm)
    means = [t / pulls_per_arm for t in totals]
    return max(range(n), key=lambda a: means[a])
