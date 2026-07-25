"""Wire protocol for the env server.

Length-prefixed MessagePack over a Unix stream socket. One request per frame,
one response per frame. Connections are keep-alive.

Framing::

    <4-byte big-endian length> <msgpack body>

The length covers the body only, not itself. The body is an msgpack-encoded
map; request / response shapes are documented in ``env_server/server.py``.

Type coercion is handled automatically on the way out (via msgpack's
``default=`` hook) and in (via ``ext_hook=``):

  * numpy ndarrays (any dtype/shape)      -> ext type 1 (zero-copy on recv)
  * numpy scalars                         -> Python int/float via .item()
  * torch.Tensor                          -> ext type 3 (numpy on slim recv)
  * jax.Array                             -> ext type 4 (numpy on slim recv)
  * gymnasium / gym Space objects         -> ext type 2 (tagged dict on slim recv)
  * pandas DataFrame / Series / Timestamp -> records / dict / ISO string
  * pydantic models (v1 and v2)           -> .dict() / .model_dump()
  * dataclass instances                   -> dict via dataclasses.asdict
  * datetime / date / time                -> ISO-8601 string
  * timedelta                             -> total_seconds() (float)
  * sets and frozensets                   -> list
  * pathlib.Path and other PathLike       -> str
  * bytes / bytearray / memoryview        -> msgpack bin (native)
  * complex                               -> [real, imag] (LOSSY)
  * enum.Enum                             -> .value (LOSSY; loses class)

``msgpack`` is the only hard dependency; numpy / torch / jax / gym / pandas /
pydantic are all lazily imported so slim images degrade gracefully. This module
is the single serialization story shared by the env RPC server and the
sandboxed policy worker (see ``grading.policy_runner``).
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import os
import struct
from typing import Any

import msgpack

# 4-byte big-endian frame length, covering the msgpack body only.
_FRAME_HEADER = struct.Struct(">I")
_MAX_FRAME_BYTES = 1 << 30  # 1 GiB; defense against bogus length headers.

# MessagePack extension-type codes. Code 0 is reserved; 1..127 are user-defined.
_EXT_NDARRAY = 1       # numpy ndarrays (any dtype/shape); zero-copy on recv
_EXT_GYM_SPACE = 2     # gymnasium / gym Space objects
_EXT_TORCH_TENSOR = 3  # torch.Tensor; falls back to ndarray on slim recv
_EXT_JAX_ARRAY = 4     # jax.Array; falls back to ndarray on slim recv


def _ndarray_payload(arr: Any, np_module: Any) -> bytes:
    """Pack the (dtype, shape, raw-bytes) payload for an ndarray, shared across
    the numpy / torch / JAX ext encoders."""
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np_module.ascontiguousarray(arr)
    return msgpack.packb(
        [str(arr.dtype), list(arr.shape), arr.tobytes(order="C")],
        use_bin_type=True,
    )


def _ndarray_to_ext(arr: Any, np_module: Any) -> msgpack.ExtType:
    """Encode a numpy ndarray as ext type 1."""
    return msgpack.ExtType(_EXT_NDARRAY, _ndarray_payload(arr, np_module))


def _payload_to_ndarray(payload: bytes, np_module: Any) -> Any:
    """Decode a (dtype, shape, raw-bytes) payload into a numpy ndarray."""
    dtype_str, shape, raw = msgpack.unpackb(payload, raw=False)
    return np_module.frombuffer(raw, dtype=dtype_str).reshape(shape).copy()


def _default(obj: Any) -> Any:
    """msgpack ``default`` hook: convert non-native objects. Called once per
    non-native object; nested non-native types inside returned containers ARE
    recursed into. Unknown types raise TypeError; the caller turns that into a
    structured error reply naming the class.
    """
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return _ndarray_to_ext(obj, np)
        if isinstance(obj, np.generic):
            return obj.item()

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and np is not None and isinstance(obj, torch.Tensor):
        return msgpack.ExtType(
            _EXT_TORCH_TENSOR, _ndarray_payload(obj.detach().cpu().numpy(), np)
        )

    try:
        import jax  # type: ignore[import-not-found]
    except ImportError:
        jax = None  # type: ignore[assignment]
    if jax is not None and np is not None and isinstance(obj, jax.Array):
        return msgpack.ExtType(_EXT_JAX_ARRAY, _ndarray_payload(np.asarray(obj), np))

    try:
        import gymnasium as _gym  # type: ignore[import-not-found]
    except ImportError:
        try:
            import gym as _gym  # type: ignore[import-not-found]
        except ImportError:
            _gym = None  # type: ignore[assignment]
    if _gym is not None and np is not None:
        spaces = getattr(_gym, "spaces", None)
        if spaces is not None:
            payload: dict | None = None
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
                    "n": int(obj.n)
                    if isinstance(obj.n, (int, np.integer))
                    else list(obj.n),
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
        pd = None  # type: ignore[assignment]
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
        _PydBaseModel = None  # type: ignore[assignment]
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
        f"object of type {type(obj).__name__!r} is not msgpack-serializable; "
        "extend _default() in env_server/protocol.py if your env needs it. For "
        "framework-bound types (torch / JAX / mujoco), prefer extracting raw "
        "numbers (qpos, qvel, observation arrays) on the env side rather than "
        "serializing the framework object whole."
    )


def _raw_ndarray_payload(data: bytes) -> dict[str, Any]:
    """Slim-receiver fallback for ndarray-shaped ext payloads when numpy is not
    installed on the receiver."""
    dtype_str, shape, raw = msgpack.unpackb(data, raw=False)
    return {"_type": "ndarray", "dtype": dtype_str, "shape": list(shape), "raw": raw}


def _ext_hook(code: int, data: bytes) -> Any:
    """msgpack ``ext_hook``: decode ext-type payloads back to Python, degrading
    to a tagged dict / ndarray when the relevant library is absent."""
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


def _pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_default, use_bin_type=True)


def _unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False, ext_hook=_ext_hook)


def _frame(body: bytes) -> bytes:
    """Prepend the 4-byte big-endian length header to a msgpack body."""
    if len(body) > _MAX_FRAME_BYTES:
        raise ValueError(f"frame body exceeds {_MAX_FRAME_BYTES} bytes ({len(body)})")
    return _FRAME_HEADER.pack(len(body)) + body
