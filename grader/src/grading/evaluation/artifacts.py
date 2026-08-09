"""Declarative, bounded artifact and trusted-fixture loaders."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Protocol

from grading.faults import AgentFault, GraderFault
from grading.numeric import NumericContractError, finite_number
from grading.secure_io import (
    open_directory_fd,
    persistent_regular_file_snapshot,
    read_regular_bytes,
)

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
    """An immutable grader-owned artifact plus its original submission path."""

    path: Path
    original_path: Path
    size: int


@dataclass(frozen=True)
class SubmittedWorkspace:
    """An immutable grader-owned master plus its untrusted source path."""

    path: Path
    original_path: Path
    file_count: int
    total_bytes: int
    root_device: int
    root_inode: int
    _descriptor: WorkspaceArtifact = dataclass_field(repr=False, compare=False)

    @property
    def snapshot_path(self) -> Path:
        """Explicit alias for the backwards-compatible ``path`` attribute."""
        return self.path

    @contextmanager
    def execution_cwd(self, cwd: str | Path | None = None) -> Iterator[int]:
        """Yield a pinned cwd in a fresh disposable candidate-owned clone."""
        with self._descriptor._execution_cwd(self, cwd) as cwd_fd:
            yield cwd_fd


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
    """Reject every symlink component and read one descriptor-pinned file."""
    try:
        return read_regular_bytes(path, max_bytes=max_bytes)
    except Exception as exc:
        raise fault_type(
            f"{label} could not be read as a stable regular file: {exc}"
        ) from exc


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
        try:
            snapshot = persistent_regular_file_snapshot(
                path,
                max_bytes=self.max_bytes,
            )
        except OSError as exc:
            raise AgentFault(
                f"{self.path} could not be captured as a stable regular file: {exc}"
            ) from exc
        return SubmittedFile(
            path=snapshot,
            original_path=path,
            size=snapshot.stat().st_size,
        )

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "regular-file-artifact.v1",
            "path": self.path,
            "max_bytes": self.max_bytes,
        }


_NATIVE_PAYLOAD_MAGICS = (
    b"\x7fELF",
    b"MZ",
    b"!<arch>\n",
    b"\x00asm",
    b"\xcf\xfa\xed\xfe",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xfe\xed\xfa\xce",
)
_MAX_WORKSPACE_ENTRIES = 20_000
_MAX_WORKSPACE_DEPTH = 128
_STABLE_FILE_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
_STABLE_DIRECTORY_FIELDS = ("st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns")
_PERSISTENT_WORKSPACE_DIRS: set[Path] = set()


def _remove_workspace_tree(path: Path, *, ignore_errors: bool = False) -> None:
    def make_removable(function, failed_path, _exception) -> None:
        failed = Path(failed_path)
        with contextlib.suppress(OSError):
            os.chmod(failed.parent, 0o700)
        with contextlib.suppress(OSError):
            if stat.S_ISDIR(os.lstat(failed).st_mode):
                os.chmod(failed, 0o700)
        function(failed_path)

    try:
        shutil.rmtree(path, onexc=make_removable)
    except FileNotFoundError:
        return
    except OSError:
        if not ignore_errors:
            raise


def _cleanup_persistent_workspaces() -> None:
    for snapshot_dir in tuple(_PERSISTENT_WORKSPACE_DIRS):
        _remove_workspace_tree(snapshot_dir, ignore_errors=True)
        _PERSISTENT_WORKSPACE_DIRS.discard(snapshot_dir)


atexit.register(_cleanup_persistent_workspaces)


def _workspace_directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    missing = [name for name in required if not hasattr(os, name)]
    if missing:
        raise AgentFault(
            "secure workspace traversal is unavailable; missing " + ", ".join(missing)
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _workspace_file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise AgentFault(
            "secure workspace file inspection is unavailable; missing O_NOFOLLOW"
        )
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


def _candidate_workspace_owner() -> tuple[int, int]:
    if os.geteuid() != 0:
        return os.geteuid(), os.getegid()
    from grading.helpers import resolve_submitted_process_identity

    uid, gid, _home, _name = resolve_submitted_process_identity()
    return uid, gid


def _create_workspace_tree(
    root_name: str,
    *,
    prefix: str,
    parent_mode: int,
) -> tuple[Path, Path, int]:
    snapshot_parent = Path(os.path.realpath("/tmp"))
    snapshot_fd = -1
    try:
        snapshot_dir = Path(
            tempfile.mkdtemp(prefix=prefix, dir=snapshot_parent)
        ).resolve(strict=True)
        os.chmod(snapshot_dir, parent_mode)
        snapshot_root = snapshot_dir / root_name
        snapshot_root.mkdir(mode=0o700)
        snapshot_fd = open_directory_fd(snapshot_root)
        return snapshot_dir, snapshot_root, snapshot_fd
    except BaseException:
        if snapshot_fd >= 0:
            os.close(snapshot_fd)
        if "snapshot_dir" in locals():
            _remove_workspace_tree(snapshot_dir, ignore_errors=True)
        raise


def _finish_snapshot_entry(
    fd: int,
    *,
    mode: int,
    mutable: bool,
    uid: int | None,
    gid: int | None,
) -> None:
    executable = bool(stat.S_IMODE(mode) & 0o111)
    if stat.S_ISDIR(mode):
        target_mode = 0o700 if mutable else 0o500
    else:
        target_mode = (
            (0o700 if executable else 0o600)
            if mutable
            else (0o500 if executable else 0o400)
        )
    os.fchmod(fd, target_mode)
    if mutable and os.geteuid() == 0:
        assert uid is not None and gid is not None
        os.fchown(fd, uid, gid)


def _write_snapshot_file(
    directory_fd: int,
    name: str,
    data: bytes,
    *,
    mode: int,
    mutable: bool,
    uid: int | None,
    gid: int | None,
    relative: str,
) -> None:
    destination_fd = -1
    try:
        destination_fd = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        written = 0
        while written < len(data):
            count = os.write(destination_fd, data[written:])
            if count <= 0:
                raise OSError("snapshot write made no progress")
            written += count
        _finish_snapshot_entry(
            destination_fd,
            mode=mode,
            mutable=mutable,
            uid=uid,
            gid=gid,
        )
    except OSError as exc:
        raise GraderFault(
            f"could not snapshot workspace file {relative!r}: {exc}"
        ) from exc
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)


def _same_stat_fields(
    before: os.stat_result,
    after: os.stat_result,
    fields: tuple[str, ...],
) -> bool:
    return all(getattr(before, field) == getattr(after, field) for field in fields)


def _workspace_entry_names(
    directory_fd: int,
    *,
    label: str,
) -> list[str]:
    before = os.fstat(directory_fd)
    names: list[str] = []
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                try:
                    entry.name.encode("utf-8", errors="strict")
                except UnicodeEncodeError as exc:
                    raise AgentFault(
                        f"{label} contains a filename that is not valid UTF-8"
                    ) from exc
                names.append(entry.name)
                if len(names) > _MAX_WORKSPACE_ENTRIES:
                    raise AgentFault(
                        f"{label} has more than {_MAX_WORKSPACE_ENTRIES} entries"
                    )
    except AgentFault:
        raise
    except OSError as exc:
        raise AgentFault(f"{label} is unreadable: {exc}") from exc
    after = os.fstat(directory_fd)
    if not _same_stat_fields(before, after, _STABLE_DIRECTORY_FIELDS):
        raise AgentFault(f"{label} changed while it was being inspected")
    names.sort()
    return names


def _open_relative_workspace_directory(
    directory_fd: int,
    relative: Path,
    *,
    label: str,
) -> int:
    current_fd = os.dup(directory_fd)
    try:
        for component in relative.parts:
            next_fd = os.open(
                component,
                _workspace_directory_flags(),
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except OSError as exc:
        os.close(current_fd)
        raise AgentFault(
            f"{label} must be a stable, real workspace directory: {exc}"
        ) from exc
    except BaseException:
        os.close(current_fd)
        raise


def _read_workspace_file(
    directory_fd: int,
    name: str,
    *,
    relative: str,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    try:
        fd = os.open(name, _workspace_file_flags(), dir_fd=directory_fd)
    except OSError as exc:
        raise AgentFault(f"workspace file {relative!r} changed: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise AgentFault(
                f"workspace contains non-directory or non-regular entry {relative!r}"
            )
        if before.st_size > max_bytes:
            raise AgentFault(f"workspace file {relative!r} exceeds {max_bytes} bytes")

        remaining = max_bytes + 1
        chunks: list[bytes] = []
        while remaining > 0:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        if len(data) > max_bytes:
            raise AgentFault(f"workspace file {relative!r} exceeds {max_bytes} bytes")
        if (
            not _same_stat_fields(before, after, _STABLE_FILE_FIELDS)
            or len(data) != after.st_size
        ):
            raise AgentFault(
                f"workspace file {relative!r} changed while it was being read"
            )
        return data, after
    except AgentFault:
        raise
    except OSError as exc:
        raise AgentFault(
            f"workspace file {relative!r} could not be read: {exc}"
        ) from exc
    finally:
        os.close(fd)


@dataclass(frozen=True)
class WorkspaceArtifact:
    """A bounded, source-only directory submission.

    Declared build caches are removed before validation into an immutable
    grader-owned master. Candidate calls execute only in disposable clones.
    Traversal pins every directory and file descriptor, rejects links and
    special files, and optionally blocks bundled native or delegated payloads.
    """

    path: str
    max_files: int = 10_000
    max_total_bytes: int = 512 * 1024 * 1024
    max_file_bytes: int = 64 * 1024 * 1024
    clean_paths: tuple[str, ...] = ()
    forbidden_names: tuple[str, ...] = ()
    forbidden_suffixes: tuple[str, ...] = ()
    forbidden_text_patterns: tuple[str, ...] = ()
    text_suffixes: tuple[str, ...] = ()
    reject_native_payloads: bool = True
    allow_extra_workspace_entries: bool = False

    def __post_init__(self) -> None:
        artifact_path = _relative_artifact_path(self.path)
        if not self.allow_extra_workspace_entries and len(artifact_path.parts) != 1:
            raise ValueError(
                "exclusive workspace artifacts must use a top-level directory"
            )
        if self.max_files < 1 or self.max_total_bytes < 1 or self.max_file_bytes < 1:
            raise ValueError("workspace artifact limits must be positive")
        for raw in self.clean_paths:
            path = _relative_artifact_path(raw)
            if len(path.parts) != 1:
                raise ValueError("workspace clean paths must be top-level entries")
        if any(not value for value in self.forbidden_text_patterns):
            raise ValueError("workspace forbidden text patterns must be non-empty")

    def _clean(self, root_fd: int) -> None:
        if not shutil.rmtree.avoids_symlink_attacks:
            raise AgentFault("secure workspace cache cleanup is unavailable")
        for raw in self.clean_paths:
            name = _relative_artifact_path(raw).name
            try:
                info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise AgentFault(
                    f"workspace clean path {raw!r} is unreadable: {exc}"
                ) from exc
            try:
                if stat.S_ISDIR(info.st_mode):
                    shutil.rmtree(name, dir_fd=root_fd)
                else:
                    os.unlink(name, dir_fd=root_fd)
            except OSError as exc:
                raise AgentFault(
                    f"workspace clean path {raw!r} could not be removed: {exc}"
                ) from exc

    def _validate_directory(
        self,
        directory_fd: int,
        *,
        destination_fd: int,
        snapshot_mutable: bool,
        snapshot_uid: int | None,
        snapshot_gid: int | None,
        prefix: str,
        depth: int,
        totals: list[int],
        visited: set[tuple[int, int]],
        forbidden_names: set[str],
        forbidden_suffixes: tuple[str, ...],
        text_suffixes: tuple[str, ...],
    ) -> None:
        before_directory = os.fstat(directory_fd)
        names = _workspace_entry_names(
            directory_fd,
            label=f"workspace directory {prefix or self.path!r}",
        )
        totals[2] += len(names)
        if totals[2] > _MAX_WORKSPACE_ENTRIES:
            raise AgentFault(
                f"workspace has more than {_MAX_WORKSPACE_ENTRIES} entries"
            )

        for name in names:
            relative = f"{prefix}/{name}" if prefix else name
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise AgentFault(
                    f"workspace entry {relative!r} could not be inspected: {exc}"
                ) from exc

            if stat.S_ISDIR(info.st_mode):
                if depth + 1 > _MAX_WORKSPACE_DEPTH:
                    raise AgentFault(
                        f"workspace exceeds maximum depth {_MAX_WORKSPACE_DEPTH}"
                    )
                try:
                    child_fd = os.open(
                        name,
                        _workspace_directory_flags(),
                        dir_fd=directory_fd,
                    )
                except OSError as exc:
                    raise AgentFault(
                        f"workspace directory {relative!r} changed: {exc}"
                    ) from exc
                destination_child_fd = -1
                try:
                    opened = os.fstat(child_fd)
                    if not os.path.samestat(info, opened):
                        raise AgentFault(f"workspace directory {relative!r} changed")
                    identity = (opened.st_dev, opened.st_ino)
                    if identity in visited:
                        raise AgentFault(f"workspace directory cycle at {relative!r}")
                    visited.add(identity)
                    try:
                        os.mkdir(name, mode=0o700, dir_fd=destination_fd)
                        destination_child_fd = os.open(
                            name,
                            _workspace_directory_flags(),
                            dir_fd=destination_fd,
                        )
                    except OSError as exc:
                        raise GraderFault(
                            f"could not create snapshot directory {relative!r}: {exc}"
                        ) from exc
                    self._validate_directory(
                        child_fd,
                        destination_fd=destination_child_fd,
                        snapshot_mutable=snapshot_mutable,
                        snapshot_uid=snapshot_uid,
                        snapshot_gid=snapshot_gid,
                        prefix=relative,
                        depth=depth + 1,
                        totals=totals,
                        visited=visited,
                        forbidden_names=forbidden_names,
                        forbidden_suffixes=forbidden_suffixes,
                        text_suffixes=text_suffixes,
                    )
                    try:
                        _finish_snapshot_entry(
                            destination_child_fd,
                            mode=opened.st_mode,
                            mutable=snapshot_mutable,
                            uid=snapshot_uid,
                            gid=snapshot_gid,
                        )
                    except OSError as exc:
                        raise GraderFault(
                            f"could not finalize snapshot directory {relative!r}: {exc}"
                        ) from exc
                finally:
                    if destination_child_fd >= 0:
                        os.close(destination_child_fd)
                    os.close(child_fd)
                continue

            if not stat.S_ISREG(info.st_mode):
                raise AgentFault(
                    f"workspace contains non-directory or non-regular entry "
                    f"{relative!r}"
                )

            folded_name = name.casefold()
            folded_suffix = Path(name).suffix.casefold()
            if folded_name in forbidden_names:
                raise AgentFault(f"workspace contains forbidden file {relative!r}")
            if forbidden_suffixes and folded_suffix in forbidden_suffixes:
                raise AgentFault(f"workspace contains forbidden suffix in {relative!r}")

            data, read_info = _read_workspace_file(
                directory_fd,
                name,
                relative=relative,
                max_bytes=self.max_file_bytes,
            )
            try:
                current_info = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise AgentFault(f"workspace file {relative!r} changed: {exc}") from exc
            if not _same_stat_fields(
                read_info,
                current_info,
                _STABLE_FILE_FIELDS,
            ):
                raise AgentFault(f"workspace file {relative!r} changed")

            totals[0] += 1
            totals[1] += len(data)
            if totals[0] > self.max_files:
                raise AgentFault(f"workspace has more than {self.max_files} files")
            if totals[1] > self.max_total_bytes:
                raise AgentFault(
                    f"workspace exceeds {self.max_total_bytes} total bytes"
                )

            if self.reject_native_payloads and any(
                data.startswith(magic) for magic in _NATIVE_PAYLOAD_MAGICS
            ):
                raise AgentFault(
                    f"workspace contains bundled native payload {relative!r}"
                )
            if text_suffixes and folded_suffix in text_suffixes:
                try:
                    text = data.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise AgentFault(
                        f"workspace source {relative!r} is not UTF-8"
                    ) from exc
                # Collapse whitespace so spaced tokens (e.g. ``Command :: new``)
                # cannot evade literal forbidden patterns.
                compact = "".join(text.split())
                for pattern in self.forbidden_text_patterns:
                    if "".join(pattern.split()) in compact:
                        raise AgentFault(
                            f"workspace source {relative!r} contains "
                            f"forbidden pattern {pattern!r}"
                        )
            _write_snapshot_file(
                destination_fd,
                name,
                data,
                mode=read_info.st_mode,
                mutable=snapshot_mutable,
                uid=snapshot_uid,
                gid=snapshot_gid,
                relative=relative,
            )

        after_directory = os.fstat(directory_fd)
        if not _same_stat_fields(
            before_directory,
            after_directory,
            _STABLE_DIRECTORY_FIELDS,
        ):
            raise AgentFault(
                f"workspace directory {prefix or self.path!r} changed "
                "while it was being inspected"
            )

    @contextmanager
    def _execution_cwd(
        self,
        submitted: SubmittedWorkspace,
        cwd: str | Path | None,
    ) -> Iterator[int]:
        if not shutil.rmtree.avoids_symlink_attacks:
            raise GraderFault("secure execution workspace cleanup is unavailable")

        requested = Path(cwd) if cwd is not None else Path()
        if requested.is_absolute():
            relative: Path | None = None
            for base in (submitted.path, submitted.original_path):
                try:
                    relative = requested.relative_to(base)
                    break
                except ValueError:
                    continue
            if relative is None:
                raise GraderFault(
                    "candidate process cwd must stay within the declared workspace"
                )
        else:
            relative = requested
        if ".." in relative.parts:
            raise GraderFault(
                "candidate process cwd must stay within the declared workspace"
            )

        master_fd = -1
        clone_fd = -1
        cwd_fd = -1
        clone_dir: Path | None = None
        try:
            try:
                master_fd = open_directory_fd(submitted.path)
            except OSError as exc:
                raise GraderFault(f"committed workspace is unavailable: {exc}") from exc
            master_info = os.fstat(master_fd)
            if (master_info.st_dev, master_info.st_ino) != (
                submitted.root_device,
                submitted.root_inode,
            ):
                raise GraderFault("committed workspace root was replaced")

            clone_dir, _clone_root, clone_fd = _create_workspace_tree(
                Path(self.path).name,
                prefix="lbx-workspace-run-",
                parent_mode=0o711,
            )
            clone_uid, clone_gid = _candidate_workspace_owner()
            totals = [0, 0, 0]
            try:
                self._validate_directory(
                    master_fd,
                    destination_fd=clone_fd,
                    snapshot_mutable=True,
                    snapshot_uid=clone_uid,
                    snapshot_gid=clone_gid,
                    prefix="",
                    depth=0,
                    totals=totals,
                    visited={(master_info.st_dev, master_info.st_ino)},
                    forbidden_names={name.casefold() for name in self.forbidden_names},
                    forbidden_suffixes=tuple(
                        suffix.casefold() for suffix in self.forbidden_suffixes
                    ),
                    text_suffixes=tuple(
                        suffix.casefold() for suffix in self.text_suffixes
                    ),
                )
            except AgentFault as exc:
                raise GraderFault(
                    f"committed workspace could not be cloned safely: {exc}"
                ) from exc
            if totals[:2] != [submitted.file_count, submitted.total_bytes]:
                raise GraderFault(
                    "committed workspace contents changed after validation"
                )
            try:
                _finish_snapshot_entry(
                    clone_fd,
                    mode=master_info.st_mode,
                    mutable=True,
                    uid=clone_uid,
                    gid=clone_gid,
                )
            except OSError as exc:
                raise GraderFault(
                    f"could not finalize execution workspace: {exc}"
                ) from exc

            cwd_fd = _open_relative_workspace_directory(
                clone_fd,
                relative,
                label="candidate process cwd",
            )
            yield cwd_fd
        finally:
            if cwd_fd >= 0:
                os.close(cwd_fd)
            if clone_fd >= 0:
                os.close(clone_fd)
            if master_fd >= 0:
                os.close(master_fd)
            if clone_dir is not None:
                try:
                    _remove_workspace_tree(clone_dir)
                except OSError as exc:
                    raise GraderFault(
                        f"execution workspace could not be removed securely: {exc}"
                    ) from exc

    def load(self, workspace: Path) -> SubmittedWorkspace:
        workspace = Path(workspace)
        artifact_path = _relative_artifact_path(self.path)
        try:
            workspace_fd = open_directory_fd(workspace)
        except OSError as exc:
            raise AgentFault(f"workspace directory is unreadable: {exc}") from exc

        root_fd = -1
        snapshot_fd = -1
        snapshot_dir: Path | None = None
        snapshot_root: Path | None = None
        snapshot_retained = False
        try:
            if not self.allow_extra_workspace_entries:
                extras = [
                    name
                    for name in _workspace_entry_names(
                        workspace_fd,
                        label="workspace directory",
                    )
                    if name != artifact_path.name
                ]
                if extras:
                    raise AgentFault(
                        "workspace contains undeclared entries outside "
                        f"{self.path!r}: {extras}"
                    )

            root_fd = _open_relative_workspace_directory(
                workspace_fd,
                artifact_path,
                label=self.path,
            )
            self._clean(root_fd)

            root_info = os.fstat(root_fd)
            snapshot_dir, snapshot_root, snapshot_fd = _create_workspace_tree(
                artifact_path.name,
                prefix="lbx-workspace-master-",
                parent_mode=0o700,
            )
            totals = [0, 0, 0]  # file count, total bytes, all entries
            self._validate_directory(
                root_fd,
                destination_fd=snapshot_fd,
                snapshot_mutable=False,
                snapshot_uid=None,
                snapshot_gid=None,
                prefix="",
                depth=0,
                totals=totals,
                visited={(root_info.st_dev, root_info.st_ino)},
                forbidden_names={name.casefold() for name in self.forbidden_names},
                forbidden_suffixes=tuple(
                    suffix.casefold() for suffix in self.forbidden_suffixes
                ),
                text_suffixes=tuple(suffix.casefold() for suffix in self.text_suffixes),
            )
            if totals[0] == 0:
                raise AgentFault(f"{self.path} contains no source files")
            try:
                _finish_snapshot_entry(
                    snapshot_fd,
                    mode=root_info.st_mode,
                    mutable=False,
                    uid=None,
                    gid=None,
                )
            except OSError as exc:
                raise GraderFault(
                    f"could not finalize committed workspace: {exc}"
                ) from exc
            snapshot_info = os.fstat(snapshot_fd)

            reopened_fd = _open_relative_workspace_directory(
                workspace_fd,
                artifact_path,
                label=self.path,
            )
            try:
                if not os.path.samestat(root_info, os.fstat(reopened_fd)):
                    raise AgentFault(
                        f"{self.path} changed while it was being validated"
                    )
            finally:
                os.close(reopened_fd)

            if not self.allow_extra_workspace_entries:
                extras = [
                    name
                    for name in _workspace_entry_names(
                        workspace_fd,
                        label="workspace directory",
                    )
                    if name != artifact_path.name
                ]
                if extras:
                    raise AgentFault(
                        "workspace contains undeclared entries outside "
                        f"{self.path!r}: {extras}"
                    )

            assert snapshot_dir is not None and snapshot_root is not None
            _PERSISTENT_WORKSPACE_DIRS.add(snapshot_dir)
            snapshot_retained = True
            return SubmittedWorkspace(
                path=snapshot_root,
                original_path=workspace / artifact_path,
                file_count=totals[0],
                total_bytes=totals[1],
                root_device=snapshot_info.st_dev,
                root_inode=snapshot_info.st_ino,
                _descriptor=self,
            )
        finally:
            if snapshot_fd >= 0:
                os.close(snapshot_fd)
            if root_fd >= 0:
                os.close(root_fd)
            os.close(workspace_fd)
            if snapshot_dir is not None and not snapshot_retained:
                _remove_workspace_tree(snapshot_dir, ignore_errors=True)

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "workspace-artifact.v1",
            "path": self.path,
            "max_files": self.max_files,
            "max_total_bytes": self.max_total_bytes,
            "max_file_bytes": self.max_file_bytes,
            "clean_paths": list(self.clean_paths),
            "forbidden_names": list(self.forbidden_names),
            "forbidden_suffixes": list(self.forbidden_suffixes),
            "forbidden_text_patterns": list(self.forbidden_text_patterns),
            "text_suffixes": list(self.text_suffixes),
            "reject_native_payloads": self.reject_native_payloads,
            "allow_extra_workspace_entries": self.allow_extra_workspace_entries,
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
    "DEFAULT_MAX_ARTIFACT_BYTES",
    "ArtifactSpec",
    "JsonArtifact",
    "NumericField",
    "RegularFileArtifact",
    "SubmittedFile",
    "SubmittedWorkspace",
    "TextArtifact",
    "TrustedJson",
    "WorkspaceArtifact",
    "trusted_fixture_specs",
]
