"""Private, reproducible context for one evaluation attempt."""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from grading.secure_io import open_directory_fd

EVALUATION_NONCE_ENV = "LBX_EVALUATION_NONCE"
EVALUATION_PLAN_ATTESTED_ENV = "LBX_EVALUATION_PLAN_ATTESTED"
MAX_COMMITTED_FILES = 10_000
MAX_COMMITTED_ENTRIES = 20_000
MAX_COMMITTED_DEPTH = 128
MAX_COMMITTED_BYTES = 1024 * 1024 * 1024


def _framed(value: bytes) -> bytes:
    return len(value).to_bytes(8, byteorder="big") + value


def _update_array_digest(digest: Any, value: Any) -> None:
    import numpy as np

    array = np.asarray(value)
    digest.update(_framed(str(array.dtype).encode("ascii", errors="replace")))
    digest.update(
        _framed(repr(tuple(int(part) for part in array.shape)).encode("ascii"))
    )
    digest.update(_framed(np.ascontiguousarray(array).tobytes()))


def artifact_digest(raw_arrays: Mapping[str, tuple[Any, Any]]) -> str:
    """Digest candidate predictions without including private truth values."""
    digest = hashlib.sha256()
    for name in sorted(raw_arrays):
        digest.update(_framed(name.encode("utf-8")))
        prediction, _truth = raw_arrays[name]
        _update_array_digest(digest, prediction)
    return digest.hexdigest()


def _stable_file_state(before: os.stat_result, after: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    return all(getattr(before, field) == getattr(after, field) for field in fields)


def _workspace_file_digests(
    directory_fd: int,
    *,
    prefix: str = "",
    totals: list[int] | None = None,
    depth: int = 0,
    visited: set[tuple[int, int]] | None = None,
) -> list[tuple[str, int, bytes]]:
    if totals is None:
        totals = [0, 0, 0]  # file count, committed bytes, all entries
    if visited is None:
        info = os.fstat(directory_fd)
        visited = {(info.st_dev, info.st_ino)}
    before_directory = os.fstat(directory_fd)
    records: list[tuple[str, int, bytes]] = []
    names: list[str] = []
    with os.scandir(directory_fd) as iterator:
        for entry in iterator:
            totals[2] += 1
            if totals[2] > MAX_COMMITTED_ENTRIES:
                raise ValueError(
                    f"artifact workspace has more than "
                    f"{MAX_COMMITTED_ENTRIES} entries"
                )
            names.append(entry.name)
    names.sort()

    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    )
    for name in names:
        if depth + 1 > MAX_COMMITTED_DEPTH:
            raise ValueError(
                f"artifact workspace exceeds maximum depth {MAX_COMMITTED_DEPTH}"
            )
        relative = f"{prefix}/{name}" if prefix else name
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(
                f"artifact workspace entry {relative!r} could not be inspected: {exc}"
            ) from exc
        if stat.S_ISDIR(info.st_mode):
            try:
                child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise ValueError(
                    f"artifact workspace directory {relative!r} changed: {exc}"
                ) from exc
            try:
                opened_directory = os.fstat(child_fd)
                identity = (opened_directory.st_dev, opened_directory.st_ino)
                if identity in visited:
                    raise ValueError(
                        f"artifact workspace directory cycle at {relative!r}"
                    )
                visited.add(identity)
                records.extend(
                    _workspace_file_digests(
                        child_fd,
                        prefix=relative,
                        totals=totals,
                        depth=depth + 1,
                        visited=visited,
                    )
                )
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                f"artifact workspace contains non-regular entry {relative!r}"
            )

        try:
            fd = os.open(name, file_flags, dir_fd=directory_fd)
        except OSError as exc:
            raise ValueError(
                f"artifact workspace file {relative!r} changed: {exc}"
            ) from exc
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError(
                    f"artifact workspace contains non-regular entry {relative!r}"
                )
            totals[0] += 1
            if totals[0] > MAX_COMMITTED_FILES:
                raise ValueError(
                    f"artifact workspace has more than {MAX_COMMITTED_FILES} files"
                )
            if totals[1] + opened.st_size > MAX_COMMITTED_BYTES:
                raise ValueError(
                    f"artifact workspace exceeds {MAX_COMMITTED_BYTES} "
                    "committed bytes"
                )
            file_digest = hashlib.sha256()
            bytes_read = 0
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > MAX_COMMITTED_BYTES:
                    raise ValueError(
                        f"artifact workspace exceeds {MAX_COMMITTED_BYTES} "
                        "committed bytes"
                    )
                file_digest.update(chunk)
            after = os.fstat(fd)
            if not _stable_file_state(opened, after) or bytes_read != after.st_size:
                raise ValueError(
                    f"artifact workspace file {relative!r} changed while hashing"
                )
            totals[1] += bytes_read
            records.append((relative, bytes_read, file_digest.digest()))
        finally:
            os.close(fd)
    after_directory = os.fstat(directory_fd)
    directory_fields = ("st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(before_directory, field) != getattr(after_directory, field)
        for field in directory_fields
    ):
        raise ValueError(
            f"artifact workspace directory {prefix or '.'!r} changed while hashing"
        )
    return records


def workspace_artifact_digest(root: Path) -> str:
    """Commit every regular file through a pinned, symlink-free directory tree."""

    root = Path(root)
    try:
        root_fd = open_directory_fd(root)
    except OSError as exc:
        raise ValueError(
            f"artifact workspace is not a stable directory: {root}: {exc}"
        ) from exc
    try:
        records = sorted(_workspace_file_digests(root_fd))
    finally:
        os.close(root_fd)

    if len(records) > MAX_COMMITTED_FILES:
        raise ValueError(
            f"artifact workspace has {len(records)} files, over limit "
            f"{MAX_COMMITTED_FILES}"
        )
    total_bytes = sum(size for _relative, size, _file_digest in records)
    if total_bytes > MAX_COMMITTED_BYTES:
        raise ValueError(
            f"artifact workspace exceeds {MAX_COMMITTED_BYTES} committed bytes"
        )

    digest = hashlib.sha256()
    for relative, size, file_digest in records:
        digest.update(_framed(relative.encode("utf-8")))
        digest.update(size.to_bytes(8, byteorder="big"))
        digest.update(file_digest)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvaluationContext:
    """Private-nonce-derived deterministic seeds plus a public commitment."""

    task_digest: str
    artifact_digest: str
    nonce: str
    seed_material: bytes
    commitment: str
    attested: bool

    @classmethod
    def create(
        cls,
        *,
        task_digest: str,
        raw_arrays: Mapping[str, tuple[Any, Any]],
    ) -> "EvaluationContext":
        return cls.create_from_artifact_digest(
            task_digest=task_digest,
            candidate_digest=artifact_digest(raw_arrays),
        )

    @classmethod
    def create_from_artifact_digest(
        cls,
        *,
        task_digest: str,
        candidate_digest: str,
    ) -> "EvaluationContext":
        nonce = os.environ.get(EVALUATION_NONCE_ENV)
        attested = bool(nonce and os.environ.get(EVALUATION_PLAN_ATTESTED_ENV) == "1")
        return cls._create_with_nonce(
            task_digest=task_digest,
            candidate_digest=candidate_digest,
            nonce=nonce,
            attested=attested,
        )

    @classmethod
    def replay(
        cls,
        *,
        task_digest: str,
        candidate_digest: str,
        replay: Mapping[str, Any],
    ) -> EvaluationContext:
        """Rebuild an exact private context from a root-side trace record.

        Replay is refused when the committed artifact digest differs from the
        trace. A replay reproduces the seed commitment but is intentionally not
        itself marked as a fresh attested evaluation.
        """
        recorded_digest = replay.get("artifact_digest")
        nonce = replay.get("nonce")
        if recorded_digest != candidate_digest:
            raise ValueError(
                "replay artifact digest does not match the committed candidate"
            )
        if not isinstance(nonce, str) or not nonce:
            raise ValueError("replay record is missing a non-empty nonce")
        return cls._create_with_nonce(
            task_digest=task_digest,
            candidate_digest=candidate_digest,
            nonce=nonce,
            attested=False,
        )

    @classmethod
    def _create_with_nonce(
        cls,
        *,
        task_digest: str,
        candidate_digest: str,
        nonce: str | None,
        attested: bool,
    ) -> EvaluationContext:
        # A fresh private nonce is generated only after the artifact has been
        # committed. It therefore provides challenge unpredictability without a
        # long-lived shared secret. Local fallback remains deterministic and is
        # never presented as attested evidence.
        secret_bytes = (
            nonce.encode("utf-8")
            if nonce
            else hashlib.sha256(f"local:{task_digest}".encode()).digest()
        )
        effective_nonce = nonce or candidate_digest
        message = (
            f"lbx-evaluation.v1\0{task_digest}\0{candidate_digest}\0"
            f"{effective_nonce}"
        ).encode("utf-8")
        seed_material = hmac.new(secret_bytes, message, hashlib.sha256).digest()
        commitment = hashlib.sha256(seed_material).hexdigest()
        return cls(
            task_digest=task_digest,
            artifact_digest=candidate_digest,
            nonce=effective_nonce,
            seed_material=seed_material,
            commitment=commitment,
            attested=attested,
        )

    def seed(self, label: str) -> int:
        """Nonce-bound seed for permutation evidence and other fresh probes."""
        material = hmac.new(
            self.seed_material,
            label.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return int.from_bytes(material[:8], byteorder="big", signed=False)

    def selection_seed(self, label: str) -> int:
        """Task-bound seed for stable hidden challenge selection.

        Candidate bytes and the fresh evaluation nonce deliberately do not enter
        this seed. Calibration, production, and exact replays therefore measure
        the same hidden rows, while ``seed()`` remains nonce-bound for
        unpredictable permutation evidence.
        """
        material = hashlib.sha256(
            (f"lbx-challenge-selection.v1\0{self.task_digest}\0{label}").encode()
        ).digest()
        return int.from_bytes(material[:8], byteorder="big", signed=False)


__all__ = [
    "EVALUATION_NONCE_ENV",
    "EVALUATION_PLAN_ATTESTED_ENV",
    "EvaluationContext",
    "artifact_digest",
    "workspace_artifact_digest",
]
