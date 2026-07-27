"""Regression tests for component-safe submission-file reads."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from grading.evaluation import context as context_module
from grading.secure_io import (
    SecureFileError,
    read_regular_bytes,
    regular_file_snapshot,
)


def test_read_regular_bytes_accepts_regular_file(tmp_path: Path) -> None:
    artifact = tmp_path / "submission.bin"
    artifact.write_bytes(b"candidate")

    assert read_regular_bytes(artifact, max_bytes=32) == b"candidate"


def test_read_regular_bytes_accepts_root_owned_temp_alias() -> None:
    fd, raw_path = tempfile.mkstemp(prefix="secure-io-")
    path = Path(raw_path)
    try:
        os.write(fd, b"candidate")
        os.close(fd)
        fd = -1
        assert read_regular_bytes(path, max_bytes=32) == b"candidate"
    finally:
        if fd >= 0:
            os.close(fd)
        path.unlink(missing_ok=True)


def test_read_regular_bytes_rejects_leaf_symlink(tmp_path: Path) -> None:
    truth = tmp_path / "truth.bin"
    truth.write_bytes(b"private")
    artifact = tmp_path / "submission.bin"
    os.symlink(truth, artifact)

    with pytest.raises(OSError):
        read_regular_bytes(artifact, max_bytes=32)


def test_read_regular_bytes_rejects_parent_symlink(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    (private / "truth.bin").write_bytes(b"private")
    workspace = tmp_path / "output"
    workspace.mkdir()
    os.symlink(private, workspace / "nested")

    with pytest.raises(OSError):
        read_regular_bytes(workspace / "nested" / "truth.bin", max_bytes=32)


def test_read_regular_bytes_rejects_parent_traversal(tmp_path: Path) -> None:
    artifact = tmp_path / "submission.bin"
    artifact.write_bytes(b"candidate")

    with pytest.raises(SecureFileError, match="must not contain"):
        read_regular_bytes(
            tmp_path / "child" / ".." / "submission.bin",
            max_bytes=32,
        )


def test_regular_file_snapshot_is_stable_after_source_replacement(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "submission.bin"
    artifact.write_bytes(b"candidate")

    with regular_file_snapshot(artifact, max_bytes=32) as snapshot:
        artifact.unlink()
        artifact.write_bytes(b"replacement")
        assert snapshot.read_bytes() == b"candidate"
        assert snapshot.stat().st_mode & 0o222 == 0


def test_workspace_digest_has_stable_golden_format(tmp_path: Path) -> None:
    workspace = tmp_path / "output"
    (workspace / "nested").mkdir(parents=True)
    (workspace / "a.txt").write_bytes(b"alpha")
    (workspace / "nested" / "b.bin").write_bytes(b"\x00\x01")

    assert context_module.workspace_artifact_digest(workspace) == (
        "13b7c79c165bd5f1f60c61546b79fd5a18c7bf9863b3e1c38e2ca706eaae6ad5"
    )


def test_workspace_digest_bounds_directory_entries_and_depth(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "output"
    (workspace / "one" / "two").mkdir(parents=True)
    monkeypatch.setattr(context_module, "MAX_COMMITTED_DEPTH", 1)
    with pytest.raises(ValueError, match="maximum depth"):
        context_module.workspace_artifact_digest(workspace)

    monkeypatch.setattr(context_module, "MAX_COMMITTED_DEPTH", 128)
    monkeypatch.setattr(context_module, "MAX_COMMITTED_ENTRIES", 1)
    with pytest.raises(ValueError, match="entries"):
        context_module.workspace_artifact_digest(workspace)
