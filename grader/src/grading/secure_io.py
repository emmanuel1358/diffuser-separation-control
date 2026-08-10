"""Descriptor-pinned I/O for files beneath agent-writable directories.

``O_NOFOLLOW`` protects only the final component of a path.  These helpers walk
every parent directory through ``dir_fd`` descriptors opened with
``O_DIRECTORY | O_NOFOLLOW`` and then open the leaf relative to the pinned
parent.  Renaming or replacing any component after it is opened cannot redirect
the resulting file descriptor.
"""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO


class SecureFileError(OSError):
    """A path could not be consumed as a stable regular file."""


_TRUSTED_SYSTEM_ALIASES = {
    # macOS exposes root-owned system entries through /tmp -> /private/tmp and
    # /var -> /private/var. Resolve only these immutable entries, never an
    # agent-owned descendant such as /tmp/output.
    "/tmp": os.path.realpath("/tmp"),
    "/var": os.path.realpath("/var"),
}


def _path_components(path: str | os.PathLike[str]) -> tuple[str, ...]:
    raw = os.fspath(path)
    if not raw:
        raise SecureFileError("path must not be empty")
    if "\x00" in raw:
        raise SecureFileError("path must not contain NUL")

    lexical = Path(raw)
    if ".." in lexical.parts:
        raise SecureFileError(f"path must not contain '..': {raw}")

    absolute = os.path.abspath(raw)
    for alias, target in _TRUSTED_SYSTEM_ALIASES.items():
        if absolute == alias or absolute.startswith(alias + os.sep):
            absolute = target + absolute[len(alias) :]
            break
    components = tuple(part for part in absolute.split(os.sep) if part and part != ".")
    if not components:
        raise SecureFileError(f"path must name a file or directory: {raw}")
    return components


def _directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    missing = [name for name in required if not hasattr(os, name)]
    if missing:
        raise SecureFileError(
            "secure component walking is unavailable; missing " + ", ".join(missing)
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_parent_directory(
    path: str | os.PathLike[str],
) -> tuple[int, str]:
    components = _path_components(path)
    flags = _directory_flags()
    directory_fd = os.open(os.sep, flags)
    try:
        for component in components[:-1]:
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd, components[-1]
    except BaseException:
        os.close(directory_fd)
        raise


def open_directory_fd(path: str | os.PathLike[str]) -> int:
    """Open ``path`` as a directory without following any symlink component."""

    components = _path_components(path)
    flags = _directory_flags()
    directory_fd = os.open(os.sep, flags)
    try:
        for component in components:
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def open_regular_fd(
    path: str | os.PathLike[str],
    *,
    max_bytes: int | None,
    allow_empty: bool = False,
) -> tuple[int, os.stat_result]:
    """Open and validate a regular file, returning its pinned descriptor."""

    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be non-negative or None")

    directory_fd, leaf = _open_parent_directory(path)
    fd = -1
    try:
        flags = (
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
        )
        fd = os.open(leaf, flags, dir_fd=directory_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SecureFileError(
                f"{path} is not a regular file (mode={stat.filemode(info.st_mode)})"
            )
        if not allow_empty and info.st_size == 0:
            raise SecureFileError(f"{path} is empty")
        if max_bytes is not None and info.st_size > max_bytes:
            raise SecureFileError(
                f"{path} is {info.st_size} bytes, over the {max_bytes}-byte limit"
            )
        fcntl.fcntl(
            fd,
            fcntl.F_SETFL,
            fcntl.fcntl(fd, fcntl.F_GETFL) & ~os.O_NONBLOCK,
        )
        return fd, info
    except BaseException:
        if fd >= 0:
            os.close(fd)
        raise
    finally:
        os.close(directory_fd)


@contextlib.contextmanager
def open_regular_file(
    path: str | os.PathLike[str],
    *,
    max_bytes: int | None,
    allow_empty: bool = False,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Yield a binary stream and metadata for one descriptor-pinned file."""

    fd, info = open_regular_fd(
        path,
        max_bytes=max_bytes,
        allow_empty=allow_empty,
    )
    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = -1
            yield handle, info
    finally:
        if fd >= 0:
            os.close(fd)


def _same_file_state(before: os.stat_result, after: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    return all(getattr(before, field) == getattr(after, field) for field in fields)


def read_regular_bytes(
    path: str | os.PathLike[str],
    *,
    max_bytes: int,
    allow_empty: bool = False,
) -> bytes:
    """Read one stable regular file through the descriptor used for validation."""

    with open_regular_file(
        path,
        max_bytes=max_bytes,
        allow_empty=allow_empty,
    ) as (handle, before):
        remaining = max_bytes + 1
        chunks: list[bytes] = []
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(handle.fileno())

    if len(data) > max_bytes:
        raise SecureFileError(f"{path} exceeds the {max_bytes}-byte read limit")
    if not allow_empty and not data:
        raise SecureFileError(f"{path} is empty")
    if not _same_file_state(before, after) or len(data) != after.st_size:
        raise SecureFileError(f"{path} changed while it was being read")
    return data


def _create_regular_file_snapshot(
    path: str | os.PathLike[str],
    *,
    max_bytes: int,
    allow_empty: bool = False,
) -> tuple[Path, Path]:
    # Canonicalize this grader-created directory once. macOS exposes its trusted
    # temporary root through /var -> /private/var; submission paths themselves
    # are never canonicalized because doing so would follow attacker symlinks.
    snapshot_parent = Path(_TRUSTED_SYSTEM_ALIASES["/tmp"])
    snapshot_dir = Path(
        tempfile.mkdtemp(prefix="lbx-submission-", dir=snapshot_parent)
    ).resolve(strict=True)
    original_name = Path(os.fspath(path)).name
    snapshot_name = (
        original_name if original_name not in {"", ".", ".."} else "artifact"
    )
    snapshot_path = snapshot_dir / snapshot_name
    try:
        os.chmod(snapshot_dir, 0o711)
        with open_regular_file(
            path,
            max_bytes=max_bytes,
            allow_empty=allow_empty,
        ) as (source, before):
            destination_fd = os.open(
                snapshot_path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o400,
            )
            total = 0
            try:
                with os.fdopen(destination_fd, "wb", closefd=True) as destination:
                    destination_fd = -1
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            raise SecureFileError(
                                f"{path} exceeds the {max_bytes}-byte read limit"
                            )
                        destination.write(chunk)
                    destination.flush()
                after = os.fstat(source.fileno())
            finally:
                if destination_fd >= 0:
                    os.close(destination_fd)

        if not allow_empty and total == 0:
            raise SecureFileError(f"{path} is empty")
        if not _same_file_state(before, after) or total != after.st_size:
            raise SecureFileError(f"{path} changed while it was being snapshotted")
        os.chmod(snapshot_path, 0o444)
        return snapshot_dir, snapshot_path
    except BaseException:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        raise


@contextlib.contextmanager
def regular_file_snapshot(
    path: str | os.PathLike[str],
    *,
    max_bytes: int,
    allow_empty: bool = False,
) -> Iterator[Path]:
    """Yield a root-owned immutable snapshot readable by a dropped worker."""

    snapshot_dir, snapshot_path = _create_regular_file_snapshot(
        path,
        max_bytes=max_bytes,
        allow_empty=allow_empty,
    )
    try:
        yield snapshot_path
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)


_PERSISTENT_SNAPSHOT_DIRS: set[Path] = set()


def _cleanup_persistent_snapshots() -> None:
    for snapshot_dir in tuple(_PERSISTENT_SNAPSHOT_DIRS):
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        _PERSISTENT_SNAPSHOT_DIRS.discard(snapshot_dir)


atexit.register(_cleanup_persistent_snapshots)


def persistent_regular_file_snapshot(
    path: str | os.PathLike[str],
    *,
    max_bytes: int,
    allow_empty: bool = False,
) -> Path:
    """Return an immutable snapshot retained until the grader process exits."""

    snapshot_dir, snapshot_path = _create_regular_file_snapshot(
        path,
        max_bytes=max_bytes,
        allow_empty=allow_empty,
    )
    _PERSISTENT_SNAPSHOT_DIRS.add(snapshot_dir)
    return snapshot_path


__all__ = [
    "SecureFileError",
    "open_directory_fd",
    "open_regular_fd",
    "open_regular_file",
    "persistent_regular_file_snapshot",
    "read_regular_bytes",
    "regular_file_snapshot",
]
