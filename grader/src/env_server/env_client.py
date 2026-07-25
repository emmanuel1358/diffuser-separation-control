"""Canonical agent-facing client for the hidden-environment RPC server.

COPY THIS FILE into your task's ``data/env_client.py`` (it is baked to
``/data/env_client.py``, which the agent can read and import). The agent talks
to the env over ``/tmp/env.sock``; it can never read the env source under
``/mcp_server/data/`` (root-only 0700).

This generic ``Env`` works with ANY env your task ships -- ``env.call("reset",
seed=0)`` dispatches any method. For ergonomics, add a thin task-specific
subclass that hard-codes ``env_kwargs`` and wraps your env's methods (see the
example at the bottom). Do NOT change the wire constants (socket path, framing,
``__create__`` / ``__destroy__``); everything else is yours.

Only ``msgpack`` (in the base image) plus the stdlib is required. numpy / torch
/ jax / gymnasium decode automatically when installed, and degrade to plain
arrays / tagged dicts when not.
"""

from __future__ import annotations

import os
import socket
import struct
from typing import Any

import msgpack

SOCKET_PATH = "/tmp/env.sock"
_RECV_CHUNK_BYTES = 65536
_FRAME_HEADER = struct.Struct(">I")
_MAX_FRAME_BYTES = 1 << 30  # 1 GiB cap; mirrors the server's outgoing cap.

# MessagePack extension-type codes. MUST match env_server.protocol.
_EXT_NDARRAY = 1
_EXT_GYM_SPACE = 2
_EXT_TORCH_TENSOR = 3
_EXT_JAX_ARRAY = 4


def _default(obj: Any) -> Any:
    """Pack hook for requests: numpy -> ext 1, scalars -> python, sets -> list."""
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None:
        if isinstance(obj, np.ndarray):
            arr = obj if obj.flags["C_CONTIGUOUS"] else np.ascontiguousarray(obj)
            payload = msgpack.packb(
                [str(arr.dtype), list(arr.shape), arr.tobytes(order="C")],
                use_bin_type=True,
            )
            return msgpack.ExtType(_EXT_NDARRAY, payload)
        if isinstance(obj, np.generic):
            return obj.item()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, (bytearray, memoryview)):
        return bytes(obj)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    raise TypeError(
        f"object of type {type(obj).__name__!r} cannot be sent over the env wire"
    )


def _payload_to_ndarray(payload: bytes):
    import numpy as np

    dtype_str, shape, raw = msgpack.unpackb(payload, raw=False)
    return np.frombuffer(raw, dtype=dtype_str).reshape(shape).copy()


def _ext_hook(code: int, data: bytes) -> Any:
    """Unpack hook for responses: ndarray / torch / jax / gym Space, each with a
    graceful fallback when the agent lacks the library."""
    if code == _EXT_NDARRAY:
        return _payload_to_ndarray(data)
    if code == _EXT_TORCH_TENSOR:
        arr = _payload_to_ndarray(data)
        try:
            import torch
        except ImportError:
            return arr
        return torch.from_numpy(arr)
    if code == _EXT_JAX_ARRAY:
        arr = _payload_to_ndarray(data)
        try:
            import jax.numpy as jnp  # type: ignore[import-not-found]
        except ImportError:
            return arr
        return jnp.asarray(arr)
    if code == _EXT_GYM_SPACE:
        spec = msgpack.unpackb(data, raw=False, ext_hook=_ext_hook)
        kind = spec.pop("_kind", None)
        try:
            import gymnasium as _gym  # type: ignore[import-not-found]
        except ImportError:
            try:
                import gym as _gym  # type: ignore[import-not-found]
            except ImportError:
                _gym = None  # type: ignore[assignment]
        if _gym is None:
            spec["_kind"] = kind
            spec["_type"] = f"gym.spaces.{kind}" if kind else "gym.spaces.Unknown"
            return spec
        spaces_mod = _gym.spaces
        try:
            import numpy as np
        except ImportError:
            np = None  # type: ignore[assignment]
        if kind == "Box":
            kwargs: dict[str, Any] = {
                "low": spec["low"],
                "high": spec["high"],
                "shape": tuple(spec["shape"]),
            }
            if np is not None:
                kwargs["dtype"] = np.dtype(spec["dtype"])
            return spaces_mod.Box(**kwargs)
        if kind == "Discrete":
            try:
                return spaces_mod.Discrete(spec["n"], start=spec.get("start", 0))
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


def _pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_default, use_bin_type=True)


def _unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False, ext_hook=_ext_hook)


class EnvError(Exception):
    """Raised on any server-side failure or a dropped connection."""

    def __init__(self, message: str, *, type: str = "Error") -> None:
        super().__init__(message)
        self.type = type


class Env:
    """Proxy for one env instance on the server.

    Usage (generic form, works with any env kind)::

        with Env(instance_id=0, env_kwargs={"split": "train"}) as env:
            obs, info = env.call("reset", seed=0)
            obs, reward, term, trunc, info = env.call("step", action=2)

    Wrap this in a task-specific subclass for a cleaner API (see bottom).
    """

    def __init__(
        self,
        instance_id: int = 0,
        env_kwargs: dict | None = None,
        socket_path: str | None = None,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self.instance_id = instance_id
        self._env_kwargs = dict(env_kwargs or {})
        self._socket_path = socket_path if socket_path is not None else SOCKET_PATH
        self._connect_timeout_s = connect_timeout_s
        self._sock: socket.socket | None = None
        self._buf = b""
        self._created = False
        # Per-process server-generation token from the latest response. A change
        # means the env_server restarted and prior instance_ids are gone.
        self.server_generation: str | None = None
        self._connect()

    def _connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout_s)
        sock.connect(self._socket_path)
        sock.settimeout(None)
        self._sock = sock

    def __enter__(self) -> "Env":
        self.create(**self._env_kwargs)
        return self

    def __exit__(self, *_exc) -> None:
        try:
            self.destroy()
        except Exception:
            pass

    def create(self, **env_kwargs) -> None:
        """Instantiate the env on the server via ``make_env(**env_kwargs)``."""
        merged = {**self._env_kwargs, **env_kwargs}
        self.call("__create__", env_kwargs=merged)
        self._created = True

    def destroy(self) -> None:
        """Remove the server-side instance and close the socket (idempotent)."""
        try:
            if self._created:
                self.call("__destroy__")
        finally:
            self._created = False
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def call(self, method: str, **args) -> Any:
        """Dispatch ``method`` on the server-side instance. The only method that
        touches the wire; convenience wrappers bottom out here."""
        if self._sock is None:
            raise EnvError("client is closed; create a new Env")
        req = {"method": method, "instance_id": self.instance_id, "args": args}
        body = _pack(req)
        self._sock.sendall(_FRAME_HEADER.pack(len(body)) + body)
        resp = _unpack(self._read_frame())
        gen = resp.get("gen")
        if isinstance(gen, str):
            self.server_generation = gen
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
            raise EnvError(
                f"env server sent oversized frame length {length} > {_MAX_FRAME_BYTES}"
            )
        self._buf = self._buf[_FRAME_HEADER.size :]
        while len(self._buf) < length:
            chunk = self._sock.recv(_RECV_CHUNK_BYTES)
            if not chunk:
                raise EnvError("server closed the connection mid-frame")
            self._buf += chunk
        body = self._buf[:length]
        self._buf = self._buf[length:]
        return body


# --------------------------------------------------------------------------
# Example task-specific subclass (delete / replace for your task).
#
# class MyEnv(Env):
#     def __init__(self, instance_id: int = 0, split: str = "train", **kw):
#         super().__init__(instance_id=instance_id, env_kwargs={"split": split}, **kw)
#
#     def reset(self, seed: int | None = None):
#         return self.call("reset", seed=seed)
#
#     def step(self, action):
#         return self.call("step", action=action)
# --------------------------------------------------------------------------

__version__ = "1.0-generic"
