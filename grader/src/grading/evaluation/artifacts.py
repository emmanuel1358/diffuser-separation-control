"""Declarative, bounded artifact and trusted-fixture loaders."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from grading.faults import AgentFault, GraderFault
from grading.numeric import NumericContractError, finite_number

DEFAULT_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_JSON_DEPTH = 64
DEFAULT_MAX_JSON_NODES = 100_000


class ArtifactSpec(Protocol):
    """One agent artifact loaded by :class:`RubricTask`."""

    path: str

    def load(self, workspace: Path) -> Any: ...

    def spec_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SubmittedFile:
    """A validated regular artifact path and immutable pre-read metadata."""

    path: Path
    size: int


@dataclass(frozen=True)
class NumericField:
    """One dotted JSON field converted to a bounded finite float."""

    path: str
    minimum: float | None = None
    maximum: float | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.path or any(not part for part in self.path.split(".")):
            raise ValueError("numeric field path must be a non-empty dotted path")

    def spec_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "required": self.required,
        }


def _relative_artifact_path(raw: str) -> Path:
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact paths must be non-empty and workspace-relative")
    return path


def _read_regular_bytes(
    path: Path,
    *,
    max_bytes: int,
    label: str,
    fault_type: type[Exception],
) -> bytes:
    """Atomically reject symlink/FIFO/device and bound bytes before parsing."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except Exception as exc:
        raise fault_type(
            f"{label} could not be opened as a regular file: {exc}"
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise fault_type(
                f"{label} is not a regular file (mode={stat.filemode(info.st_mode)})"
            )
        if info.st_size == 0:
            raise fault_type(f"{label} is empty")
        if info.st_size > max_bytes:
            raise fault_type(
                f"{label} is {info.st_size} bytes, over the {max_bytes}-byte limit"
            )
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) & ~os.O_NONBLOCK)
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise fault_type(f"{label} exceeds the {max_bytes}-byte read limit")
        return data
    except fault_type:
        raise
    except Exception as exc:
        raise fault_type(
            f"{label} could not be read: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        os.close(fd)


def _bounded_json_shape(
    value: Any,
    *,
    max_depth: int,
    max_nodes: int,
    label: str,
    fault_type: type[Exception],
) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise fault_type(f"{label} exceeds the {max_nodes}-node JSON limit")
        if depth > max_depth:
            raise fault_type(f"{label} exceeds the {max_depth}-level JSON depth limit")
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def _resolve_field(document: dict[str, Any], dotted: str) -> tuple[dict[str, Any], str]:
    parts = dotted.split(".")
    parent: Any = document
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            raise KeyError(dotted)
        parent = parent[part]
    if not isinstance(parent, dict) or parts[-1] not in parent:
        raise KeyError(dotted)
    return parent, parts[-1]


@dataclass(frozen=True)
class JsonArtifact:
    """A JSON submission with framework-owned I/O, shape, and numeric checks."""

    path: str
    required_keys: tuple[str, ...] = ()
    numeric_fields: tuple[NumericField, ...] = ()
    allow_extra_keys: bool = True
    require_object: bool = True
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES
    max_depth: int = DEFAULT_MAX_JSON_DEPTH
    max_nodes: int = DEFAULT_MAX_JSON_NODES

    def __post_init__(self) -> None:
        _relative_artifact_path(self.path)
        if self.max_bytes <= 0 or self.max_depth < 1 or self.max_nodes < 1:
            raise ValueError("JSON artifact limits must be positive")
        if len(self.required_keys) != len(set(self.required_keys)):
            raise ValueError("JSON required_keys must be unique")

    def load(self, workspace: Path) -> Any:
        path = Path(workspace) / _relative_artifact_path(self.path)
        data = _read_regular_bytes(
            path,
            max_bytes=self.max_bytes,
            label=self.path,
            fault_type=AgentFault,
        )
        try:
            text = data.decode("utf-8", errors="strict")
            document = json.loads(text)
        except Exception as exc:
            raise AgentFault(
                f"{self.path} is not valid UTF-8 JSON: {type(exc).__name__}: {exc}"
            ) from exc
        _bounded_json_shape(
            document,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
            label=self.path,
            fault_type=AgentFault,
        )
        if self.require_object and not isinstance(document, dict):
            raise AgentFault(f"{self.path} must contain a JSON object")
        if not isinstance(document, dict):
            return document

        missing = [key for key in self.required_keys if key not in document]
        if missing:
            raise AgentFault(f"{self.path} is missing required key(s): {missing}")
        if not self.allow_extra_keys:
            allowed = set(self.required_keys)
            allowed.update(field.path.split(".", 1)[0] for field in self.numeric_fields)
            extra = sorted(set(document) - allowed)
            if extra:
                raise AgentFault(f"{self.path} has unexpected key(s): {extra}")

        for field in self.numeric_fields:
            try:
                parent, key = _resolve_field(document, field.path)
            except KeyError as exc:
                if field.required:
                    raise AgentFault(
                        f"{self.path} is missing numeric field {field.path!r}"
                    ) from exc
                continue
            try:
                parent[key] = finite_number(
                    parent[key],
                    label=f"{self.path}:{field.path}",
                    minimum=field.minimum,
                    maximum=field.maximum,
                )
            except NumericContractError as exc:
                raise AgentFault(str(exc)) from exc
        return document

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "json-artifact.v1",
            "path": self.path,
            "required_keys": list(self.required_keys),
            "numeric_fields": [field.spec_dict() for field in self.numeric_fields],
            "allow_extra_keys": self.allow_extra_keys,
            "require_object": self.require_object,
            "max_bytes": self.max_bytes,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
        }


@dataclass(frozen=True)
class TextArtifact:
    path: str
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES

    def __post_init__(self) -> None:
        _relative_artifact_path(self.path)

    def load(self, workspace: Path) -> str:
        data = _read_regular_bytes(
            Path(workspace) / _relative_artifact_path(self.path),
            max_bytes=self.max_bytes,
            label=self.path,
            fault_type=AgentFault,
        )
        try:
            return data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise AgentFault(f"{self.path} is not valid UTF-8 text: {exc}") from exc

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "text-artifact.v1",
            "path": self.path,
            "max_bytes": self.max_bytes,
        }


@dataclass(frozen=True)
class RegularFileArtifact:
    path: str
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES

    def __post_init__(self) -> None:
        _relative_artifact_path(self.path)

    def load(self, workspace: Path) -> SubmittedFile:
        path = Path(workspace) / _relative_artifact_path(self.path)
        # Read once to make all malformed path/size outcomes typed. The policy or
        # domain library may reopen the regular file after pre-grade quiescence.
        data = _read_regular_bytes(
            path,
            max_bytes=self.max_bytes,
            label=self.path,
            fault_type=AgentFault,
        )
        return SubmittedFile(path=path, size=len(data))

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "regular-file-artifact.v1",
            "path": self.path,
            "max_bytes": self.max_bytes,
        }


@dataclass(frozen=True)
class TrustedJson:
    """A root-only JSON fixture loaded before author callbacks run."""

    filename: str
    require_object: bool = True
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES

    def __post_init__(self) -> None:
        _relative_artifact_path(self.filename)

    def load(self, private: Path) -> Any:
        data = _read_regular_bytes(
            Path(private) / _relative_artifact_path(self.filename),
            max_bytes=self.max_bytes,
            label=f"trusted fixture {self.filename}",
            fault_type=GraderFault,
        )
        try:
            document = json.loads(data.decode("utf-8", errors="strict"))
        except Exception as exc:
            raise GraderFault(
                f"trusted fixture {self.filename} is invalid: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if self.require_object and not isinstance(document, dict):
            raise GraderFault(f"trusted fixture {self.filename} must be a JSON object")
        return document

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "trusted-json.v1",
            "filename": self.filename,
            "require_object": self.require_object,
            "max_bytes": self.max_bytes,
        }


def trusted_fixture_specs(
    fixtures: Mapping[str, TrustedJson],
) -> dict[str, dict[str, Any]]:
    return {name: fixture.spec_dict() for name, fixture in sorted(fixtures.items())}


__all__ = [
    "ArtifactSpec",
    "DEFAULT_MAX_ARTIFACT_BYTES",
    "JsonArtifact",
    "NumericField",
    "RegularFileArtifact",
    "SubmittedFile",
    "TextArtifact",
    "TrustedJson",
    "trusted_fixture_specs",
]
