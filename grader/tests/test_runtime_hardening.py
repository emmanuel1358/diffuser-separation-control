"""Tests for shared runtime hardening helpers."""

from __future__ import annotations

import errno
import os
import signal
import stat
from pathlib import Path

import pytest
from grading import runtime_hardening
from grading.runtime_hardening import (
    _holds_nvidia_fd,
    classify_failure,
    ensure_agent_output_directory,
    kill_nvproxy_fd_holders,
    lock_down_grader_private,
    lock_down_public_readonly,
    pre_grade_cleanup,
    prepare_grader_cache,
    scrub_escaping_symlinks,
    scrub_nonregular_files,
)


def test_classify_failure_distinguishes_noninfra_and_signal_failures() -> None:
    ordinary = classify_failure(1, "Traceback: OverflowError")
    assert ordinary.is_infra is False

    killed = classify_failure(137, "")
    assert killed.is_infra is True

    oom = classify_failure(1, "MemoryError: out of memory")
    assert oom.is_infra is True


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


def test_ensure_agent_output_directory_replaces_symlink(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    secret = private / "truth.csv"
    secret.write_text("answer")
    output = tmp_path / "output"
    os.symlink(private, output)

    assert ensure_agent_output_directory(output) is True
    assert output.is_dir()
    assert not output.is_symlink()
    assert secret.read_text() == "answer"


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


def test_filesystem_exhaustion_checks_blocks_and_inodes(
    monkeypatch, tmp_path: Path
) -> None:
    def statvfs(*, free_bytes: int, files: int, free_files: int):
        size = 4096
        return type(
            "Fs",
            (),
            {
                "f_bfree": free_bytes // size,
                "f_frsize": size,
                "f_files": files,
                "f_ffree": free_files,
            },
        )()

    monkeypatch.setattr(
        runtime_hardening.os,
        "statvfs",
        lambda _path: statvfs(free_bytes=0, files=1000, free_files=1000),
    )
    assert runtime_hardening.filesystem_exhausted(tmp_path) is True

    monkeypatch.setattr(
        runtime_hardening.os,
        "statvfs",
        lambda _path: statvfs(free_bytes=10**9, files=1000, free_files=0),
    )
    assert runtime_hardening.filesystem_exhausted(tmp_path) is True


def test_shared_memory_exhaustion_requires_pressure_and_high_shmem() -> None:
    assert runtime_hardening.shared_memory_exhausted(
        {"MemTotal": 16_000_000, "MemAvailable": 100_000, "Shmem": 15_000_000}
    )
    assert not runtime_hardening.shared_memory_exhausted(
        {"MemTotal": 16_000_000, "MemAvailable": 2_000_000, "Shmem": 15_000_000}
    )


def test_child_memory_limit_leaves_container_headroom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RUBRIC_AGENT_MEMORY_LIMIT_BYTES", raising=False)
    memory_max = tmp_path / "memory.max"
    memory_max.write_text(str(64 * 1024**3))

    limit = runtime_hardening.child_memory_limit_bytes(cgroup_paths=(memory_max,))

    assert limit == 48 * 1024**3


def test_child_memory_limit_honors_explicit_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RUBRIC_AGENT_MEMORY_LIMIT_BYTES", str(12 * 1024**3))

    limit = runtime_hardening.child_memory_limit_bytes(
        cgroup_paths=(tmp_path / "missing",)
    )

    assert limit == 12 * 1024**3


def test_apply_address_space_limit_sets_inherited_hard_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(
        runtime_hardening.resource,
        "getrlimit",
        lambda _kind: (
            runtime_hardening.resource.RLIM_INFINITY,
            runtime_hardening.resource.RLIM_INFINITY,
        ),
    )
    monkeypatch.setattr(
        runtime_hardening.resource,
        "setrlimit",
        lambda kind, limits: applied.append((kind, limits)),
    )

    runtime_hardening.apply_address_space_limit(123_456)

    assert applied == [(runtime_hardening.resource.RLIMIT_AS, (123_456, 123_456))]


def test_exhaustion_without_an_exit_status_trusts_meminfo_alone() -> None:
    """A timeout has no exit code to corroborate, so meminfo has to stand alone.

    Boreal used to check only shared memory here and let a disk-full hang void
    the episode as infrastructure, handing the agent a free retry for filling
    the box. Both resources are charged now.
    """
    exhaustion = runtime_hardening.ResourceExhaustion(
        disk=False, shmem=True, meminfo_kib={}
    )
    assert exhaustion.agent_fault_kind() == "shared_memory_exhausted"

    disk = runtime_hardening.ResourceExhaustion(disk=True, shmem=False, meminfo_kib={})
    assert disk.agent_fault_kind() == "disk_exhausted"


def test_shmem_with_an_exit_status_needs_a_kill_or_a_pressure_marker() -> None:
    """A grader may exit non-zero for reasons unrelated to a full box."""
    exhaustion = runtime_hardening.ResourceExhaustion(
        disk=False, shmem=True, meminfo_kib={}
    )
    assert exhaustion.agent_fault_kind(returncode=1, stderr="assertion failed") is None
    assert (
        exhaustion.agent_fault_kind(returncode=-signal.SIGKILL)
        == "shared_memory_exhausted"
    )
    assert (
        exhaustion.agent_fault_kind(returncode=1, stderr="Out of memory")
        == "shared_memory_exhausted"
    )


def test_sampling_reads_the_box_before_cleanup_can_free_it(monkeypatch) -> None:
    """The snapshot must capture state at call time, not at decision time.

    pre_grade_cleanup reaps the agent's /dev/shm entries, so a verdict computed
    after cleanup sees a healthy box and voids the episode instead of charging
    the agent. Sampling into a value is what keeps the two apart.
    """
    live = {"Shmem": 15_000_000, "MemTotal": 16_000_000, "MemAvailable": 100_000}
    monkeypatch.setattr(runtime_hardening, "meminfo_kib", lambda *a, **k: dict(live))
    monkeypatch.setattr(
        runtime_hardening, "filesystem_exhausted", lambda *a, **k: False
    )

    exhaustion = runtime_hardening.sample_resource_exhaustion()
    live.update({"Shmem": 0, "MemAvailable": 15_000_000})  # cleanup frees it

    assert exhaustion.shmem is True
    assert exhaustion.agent_fault_kind() == "shared_memory_exhausted"


def test_cleanup_agent_tmpfs_is_owner_scoped_and_bounded(tmp_path: Path) -> None:
    root = tmp_path / "shm"
    root.mkdir()
    for index in range(10):
        (root / f"entry-{index}").write_text("")

    removed, flooded = runtime_hardening.cleanup_agent_tmpfs(
        os.getuid(), roots=(root,), max_entries=3, max_seconds=0
    )

    assert flooded is True
    assert removed < 10


def test_protect_current_process_from_oom_is_best_effort(tmp_path: Path) -> None:
    target = tmp_path / "oom_score_adj"
    target.write_text("0")
    runtime_hardening.protect_current_process_from_oom(str(target))
    assert target.read_text() == "-1000"
    runtime_hardening.protect_current_process_from_oom(
        str(tmp_path / "missing" / "oom_score_adj")
    )


def test_quiesce_nonconvergence_fails_closed() -> None:
    with pytest.raises(
        runtime_hardening.AgentProcessQuiesceError,
        match="did not converge",
    ):
        runtime_hardening.kill_pre_grade_agent_processes(max_passes=0)


def test_quiesce_proc_enumeration_failure_fails_closed(monkeypatch) -> None:
    def fail_listdir(_path: str):
        raise PermissionError("denied")

    monkeypatch.setattr(runtime_hardening.os, "listdir", fail_listdir)
    with pytest.raises(
        runtime_hardening.ProcessQuiesceError,
        match="could not enumerate",
    ):
        runtime_hardening.kill_pre_grade_agent_processes(max_passes=1)


def test_root_cleanup_without_proc_fails_closed(monkeypatch, tmp_path: Path) -> None:
    class MissingProc:
        @staticmethod
        def is_dir() -> bool:
            return False

    monkeypatch.setattr(runtime_hardening.os, "geteuid", lambda: 0)
    monkeypatch.setattr(runtime_hardening, "Path", lambda _path: MissingProc())

    with pytest.raises(runtime_hardening.ProcessQuiesceError, match="unavailable"):
        runtime_hardening.pre_grade_cleanup(tmp_path)


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


def test_lock_down_grader_private_rejects_symlink(tmp_path: Path) -> None:
    private = tmp_path / "grader"
    private.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("truth")
    os.symlink(secret, private / "redirect")

    with pytest.raises(
        runtime_hardening.InfrastructureFault,
        match="contains symlink",
    ):
        lock_down_grader_private((private,))


def test_lock_down_grader_private_allows_optional_missing_path(
    tmp_path: Path,
) -> None:
    lock_down_grader_private(
        (tmp_path / "optional",),
        missing_ok=True,
    )


def _simulate_readonly_mount(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Make `root` look and behave like a read-only bind mount."""

    def readonly(path, *_args, **_kwargs):  # noqa: ANN001 - os shim
        raise OSError(errno.EROFS, "Read-only file system", str(path))

    class _Statvfs:
        f_flag = os.ST_RDONLY

    monkeypatch.setattr(runtime_hardening.os, "chmod", readonly)
    monkeypatch.setattr(runtime_hardening.os, "chown", readonly)
    monkeypatch.setattr(
        runtime_hardening.os,
        "statvfs",
        lambda path: _Statvfs() if str(path) == str(root) else os.statvfs(path),
    )


def test_lock_down_grader_private_tolerates_readonly_code_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grader = tmp_path / "grader"
    grader.mkdir()
    (grader / "compute_score.py").write_text("x = 1\n")
    os.chmod(grader / "compute_score.py", 0o644)
    _simulate_readonly_mount(monkeypatch, grader)

    lock_down_grader_private((grader,), readonly_mount_ok=(grader,))


def test_lock_down_grader_private_tolerates_readonly_held_out_data_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Held-out truth delivered as a Taiga is_read_only squashfs mount is
    # genuinely read-only on firecracker (the CPU-QA lane), so the ownership
    # reset fails with EROFS. When the caller whitelists the path AND the
    # filesystem really is read-only, that degrades to a warning rather than a
    # fatal fault -- the read-only mount is tamper-proof by construction.
    private = tmp_path / "data"
    private.mkdir()
    (private / "truth.json").write_text("{}\n")
    os.chmod(private / "truth.json", 0o644)
    _simulate_readonly_mount(monkeypatch, private)

    lock_down_grader_private((private,), readonly_mount_ok=(private,))


def test_lock_down_grader_private_still_fails_on_writable_held_out_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The relaxation is gated on the filesystem really being read-only. A
    # whitelisted path whose chmod/chown fails on a WRITABLE filesystem is the
    # real tamper case and must still fail hard, even when whitelisted.
    private = tmp_path / "data"
    private.mkdir()
    (private / "truth.json").write_text("{}\n")
    os.chmod(private / "truth.json", 0o644)

    def readonly(path, *_args, **_kwargs):  # noqa: ANN001 - os shim
        raise OSError(errno.EROFS, "Read-only file system", str(path))

    class _WritableStatvfs:
        f_flag = 0

    monkeypatch.setattr(runtime_hardening.os, "chmod", readonly)
    monkeypatch.setattr(runtime_hardening.os, "chown", readonly)
    monkeypatch.setattr(
        runtime_hardening.os, "statvfs", lambda _path: _WritableStatvfs()
    )

    with pytest.raises(runtime_hardening.InfrastructureFault):
        lock_down_grader_private((private,), readonly_mount_ok=(private,))


def test_lock_down_public_readonly_preserves_agent_reads(tmp_path: Path) -> None:
    public = tmp_path / "data"
    nested = public / "nested"
    nested.mkdir(parents=True)
    train = nested / "train.parquet"
    train.write_bytes(b"data")
    os.chmod(public, 0o777)
    os.chmod(nested, 0o777)
    os.chmod(train, 0o666)

    changed = lock_down_public_readonly(public)

    assert changed == 3
    assert stat.S_IMODE(public.stat().st_mode) == 0o555
    assert stat.S_IMODE(nested.stat().st_mode) == 0o555
    assert stat.S_IMODE(train.stat().st_mode) == 0o444


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
