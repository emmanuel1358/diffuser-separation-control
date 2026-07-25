"""Tests for shared runtime hardening helpers."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from grading import runtime_hardening
from grading.runtime_hardening import (
    _holds_nvidia_fd,
    kill_nvproxy_fd_holders,
    prepare_grader_cache,
    lock_down_grader_private,
    pre_grade_cleanup,
    scrub_escaping_symlinks,
    scrub_nonregular_files,
)


def test_scrub_escaping_symlinks_removes_private_redirect(tmp_path: Path) -> None:
    output = tmp_path / "output"
    private = tmp_path / "private"
    output.mkdir()
    private.mkdir()
    secret = private / "truth.csv"
    secret.write_text("answer")
    link = output / "submission.csv"
    os.symlink(secret, link)

    assert scrub_escaping_symlinks(output) == 1
    assert not link.exists()


def test_scrub_escaping_symlinks_keeps_internal_links(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    real = output / "real.csv"
    real.write_text("ok")
    link = output / "submission.csv"
    os.symlink(real, link)

    assert scrub_escaping_symlinks(output) == 0
    assert link.exists()


def test_scrub_nonregular_files_removes_fifo(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    fifo = output / "submission.csv"
    os.mkfifo(fifo)

    assert scrub_nonregular_files(output) == 1
    assert not fifo.exists()


def test_pre_grade_cleanup_runs_scrubs(tmp_path: Path) -> None:
    output = tmp_path / "output"
    private = tmp_path / "private"
    output.mkdir()
    private.mkdir()
    secret = private / "truth.csv"
    secret.write_text("answer")
    os.symlink(secret, output / "submission.csv")

    result = pre_grade_cleanup(output)

    assert result["removed_symlinks"] == 1
    assert result["removed_nonregular"] == 0
    assert result["killed_processes"] >= 0


def test_lock_down_grader_private_removes_group_other_bits(tmp_path: Path) -> None:
    private = tmp_path / "grader"
    private.mkdir()
    secret = private / "compute_score.py"
    secret.write_text("x = 1\n")
    os.chmod(private, 0o755)
    os.chmod(secret, 0o644)

    lock_down_grader_private((private,))

    assert stat.S_IMODE(private.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(secret.stat().st_mode) & 0o077 == 0


def _make_proc_with_fd(root: Path, pid: int, targets: list[str]) -> None:
    """Build a fake /proc/<pid>/fd/ tree whose fds symlink to ``targets``."""
    fd_dir = root / str(pid) / "fd"
    fd_dir.mkdir(parents=True)
    for idx, target in enumerate(targets):
        os.symlink(target, fd_dir / str(idx))


def test_holds_nvidia_fd_detects_device_symlink(tmp_path: Path) -> None:
    _make_proc_with_fd(tmp_path, 1234, ["/dev/nvidia0", "/dev/null"])
    assert _holds_nvidia_fd(str(tmp_path / "1234")) is True


def test_holds_nvidia_fd_false_without_device(tmp_path: Path) -> None:
    _make_proc_with_fd(tmp_path, 1234, ["/dev/null", "/tmp/output/result.json"])
    assert _holds_nvidia_fd(str(tmp_path / "1234")) is False


def test_holds_nvidia_fd_missing_fd_dir_is_false(tmp_path: Path) -> None:
    (tmp_path / "1234").mkdir()
    assert _holds_nvidia_fd(str(tmp_path / "1234")) is False


def _patch_fake_proc(
    monkeypatch,
    *,
    self_pid: int,
    parent_pid: int,
    entries: list[str],
    nvidia_holders: set[int],
    zombies: set[int] | None = None,
):
    zombies = zombies or set()
    killed: list[int] = []

    monkeypatch.setattr(runtime_hardening.os, "getpid", lambda: self_pid)
    monkeypatch.setattr(runtime_hardening.os, "getppid", lambda: parent_pid)
    monkeypatch.setattr(
        runtime_hardening.os,
        "listdir",
        lambda path: list(entries) if path == "/proc" else [],
    )
    monkeypatch.setattr(
        runtime_hardening,
        "_read_proc_state",
        lambda proc_path: "Z" if int(proc_path.rsplit("/", 1)[-1]) in zombies else "R",
    )
    monkeypatch.setattr(
        runtime_hardening,
        "_holds_nvidia_fd",
        lambda proc_path: int(proc_path.rsplit("/", 1)[-1]) in nvidia_holders,
    )

    def _fake_kill(pid: int, sig: int) -> None:
        killed.append(pid)

    monkeypatch.setattr(runtime_hardening.os, "kill", _fake_kill)
    return killed


def test_kill_nvproxy_fd_holders_kills_only_eligible(monkeypatch) -> None:
    killed = _patch_fake_proc(
        monkeypatch,
        self_pid=100,
        parent_pid=200,
        entries=["1", "100", "200", "300", "400", "not-a-pid"],
        nvidia_holders={1, 100, 200, 300},
        zombies={400},
    )
    n = kill_nvproxy_fd_holders()
    assert n == 1
    assert killed == [300]


def test_kill_nvproxy_fd_holders_never_kills_protected(monkeypatch) -> None:
    killed = _patch_fake_proc(
        monkeypatch,
        self_pid=100,
        parent_pid=200,
        entries=["1", "100", "200"],
        nvidia_holders={1, 100, 200},
    )
    assert kill_nvproxy_fd_holders() == 0
    assert killed == []


def test_kill_nvproxy_fd_holders_clean_list_is_noop(monkeypatch) -> None:
    killed = _patch_fake_proc(
        monkeypatch,
        self_pid=100,
        parent_pid=200,
        entries=["100", "200", "300", "400"],
        nvidia_holders=set(),
    )
    assert kill_nvproxy_fd_holders() == 0
    assert killed == []


def test_prepare_grader_cache_omits_cache_env_when_all_roots_fail(
    monkeypatch,
) -> None:
    class FailingPath(type(Path())):
        def mkdir(self, *args, **kwargs):
            raise OSError("no writable cache")

    monkeypatch.setattr("grading.runtime_hardening.Path", FailingPath)

    env = prepare_grader_cache("/nope/.grader_cache")

    assert env["PYTHONSAFEPATH"] == "1"
    assert "PYTHONPATH" not in env
    for name in (
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "NUMBA_CACHE_DIR",
        "TORCH_HOME",
        "HF_HOME",
    ):
        assert name not in env
