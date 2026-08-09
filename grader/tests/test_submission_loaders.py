from __future__ import annotations

import errno
import os
import stat
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from grading import helpers, policy_runner, runtime_hardening, score_kfold_cv
from grading import kfold as kfold_module
from grading.faults import AgentFault, GraderFault
from grading.policy_runner import PolicyWorker


def _write_script(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# ── run_submitted_executable (capture + streaming, configured identity) ──


@pytest.fixture
def local_submitted_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise process mechanics without pretending a local verifier is root."""
    identity = (os.geteuid(), os.getegid(), str(Path.home()), "local-test-agent")
    monkeypatch.setattr(
        helpers,
        "resolve_submitted_process_identity",
        lambda: identity,
    )

    def local_spawn_config(_identity, *, cwd_fd, ipc_status_fd):
        def enter_cwd() -> None:
            try:
                os.write(ipc_status_fd, b"\0")
            finally:
                os.close(ipc_status_fd)
            if cwd_fd is not None:
                os.fchdir(cwd_fd)
                os.close(cwd_fd)

        pass_fds = tuple(
            descriptor
            for descriptor in (cwd_fd, ipc_status_fd)
            if descriptor is not None
        )
        return {"preexec_fn": enter_cwd, "pass_fds": pass_fds}

    monkeypatch.setattr(
        helpers,
        "_submitted_process_spawn_config",
        local_spawn_config,
    )


def test_executable_capture_returns_stdout_and_returncode(
    tmp_path: Path,
    local_submitted_execution: None,
) -> None:
    exe = _write_script(tmp_path / "run.sh", "#!/bin/sh\necho hello\nexit 0\n")
    proc = helpers.run_submitted_executable([str(exe)], timeout_s=10)
    assert proc.returncode == 0
    assert b"hello" in proc.stdout


def test_executable_passes_args_and_returncode(
    tmp_path: Path,
    local_submitted_execution: None,
) -> None:
    exe = _write_script(tmp_path / "run.sh", '#!/bin/sh\necho "$1"\nexit 3\n')
    proc = helpers.run_submitted_executable([str(exe), "payload"], timeout_s=10)
    assert proc.returncode == 3
    assert proc.stdout.strip() == b"payload"


def test_executable_stdin_bytes_piped_to_child(
    tmp_path: Path,
    local_submitted_execution: None,
) -> None:
    exe = _write_script(tmp_path / "cat.sh", "#!/bin/sh\ncat\n")
    proc = helpers.run_submitted_executable(
        [str(exe)], stdin_bytes=b"ping", timeout_s=10
    )
    assert proc.stdout == b"ping"


def test_executable_sanitized_env_hides_grader_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_submitted_execution: None,
) -> None:
    # Default (no explicit env / passthrough): a grading-server secret is not
    # visible to the agent binary.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    exe = _write_script(
        tmp_path / "env.sh", '#!/bin/sh\necho "${ANTHROPIC_API_KEY:-MISSING}"\n'
    )
    proc = helpers.run_submitted_executable([str(exe)], timeout_s=10)
    assert proc.stdout.strip() == b"MISSING"


def test_executable_env_passthrough_exposes_parent_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_submitted_execution: None,
) -> None:
    monkeypatch.setenv("MY_TASK_CONFIG", "visible")
    exe = _write_script(
        tmp_path / "env.sh", '#!/bin/sh\necho "${MY_TASK_CONFIG:-MISSING}"\n'
    )
    proc = helpers.run_submitted_executable(
        [str(exe)], env_passthrough=True, timeout_s=10
    )
    assert proc.stdout.strip() == b"visible"


def test_executable_timeout_raises_agent_fault(
    tmp_path: Path,
    local_submitted_execution: None,
) -> None:
    # A capture-mode timeout is an agent-caused failure: it must raise AgentFault
    # (kept 0.0), NOT the builtin TimeoutExpired, which would escape compute_score
    # as env_internal_failure and DISCARD the rollout (a free veto).
    exe = _write_script(tmp_path / "slow.sh", "#!/bin/sh\nsleep 5\n")
    with pytest.raises(AgentFault):
        helpers.run_submitted_executable([str(exe)], timeout_s=0.3)


def test_executable_stdout_flood_raises_agent_fault(
    tmp_path: Path,
    local_submitted_execution: None,
) -> None:
    # A stdout flood past max_output_bytes is killed and raised as AgentFault
    # (kept 0.0), so the child cannot OOM the grader (MemoryError would be a
    # non-AgentFault -> DISCARDED free veto).
    exe = _write_script(
        tmp_path / "flood.sh",
        "#!/bin/sh\nyes AAAAAAAAAAAAAAAA\n",  # unbounded stdout
    )
    with pytest.raises(AgentFault):
        helpers.run_submitted_executable(
            [str(exe)], timeout_s=10, max_output_bytes=64 * 1024
        )


def test_executable_capture_discards_stderr(
    tmp_path: Path,
    local_submitted_execution: None,
) -> None:
    # Capture mode discards stderr at the kernel (memory safety); stdout still
    # returned. The returned stderr is empty.
    exe = _write_script(
        tmp_path / "err.sh", "#!/bin/sh\necho out\necho oops 1>&2\nexit 0\n"
    )
    proc = helpers.run_submitted_executable([str(exe)], timeout_s=10)
    assert proc.returncode == 0
    assert b"out" in proc.stdout
    assert proc.stderr == b""


def test_executable_streaming_rejects_stdin_and_timeout(tmp_path: Path) -> None:
    exe = _write_script(tmp_path / "run.sh", "#!/bin/sh\necho hi\n")
    with pytest.raises(ValueError):
        helpers.run_submitted_executable([str(exe)], streaming=True, stdin_bytes=b"x")
    with pytest.raises(ValueError):
        helpers.run_submitted_executable([str(exe)], streaming=True, timeout_s=5)


def test_executable_streaming_filters_rubric_score(
    tmp_path: Path,
    capsys,
    local_submitted_execution: None,
) -> None:
    # Streaming pumps to sys.stderr with RUBRIC_SCORE= dropped, so the child
    # cannot reach or forge the score-parsed stdout.
    exe = _write_script(
        tmp_path / "run.sh",
        "#!/bin/sh\necho hello\necho 'RUBRIC_SCORE=1.0'\necho world\n",
    )
    proc = helpers.run_submitted_executable([str(exe)], streaming=True)
    assert proc.returncode == 0
    assert proc.stdout == b""
    err = capsys.readouterr().err
    assert "hello" in err and "world" in err
    assert "RUBRIC_SCORE=" not in err


def test_executable_fails_closed_when_verifier_is_not_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        helpers,
        "resolve_submitted_process_identity",
        lambda: (1234, 1235, "/home/candidate", "candidate"),
    )
    monkeypatch.setattr(helpers.os, "geteuid", lambda: 501)
    with pytest.raises(GraderFault, match="verifier must run as root"):
        helpers.run_submitted_executable(["/bin/true"], timeout_s=1)


def test_executable_isolates_ipc_before_using_configured_uid_gid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUBRIC_AGENT_USER", "configured-agent")
    monkeypatch.setenv("RUBRIC_AGENT_UID", "2345")
    monkeypatch.setenv("RUBRIC_AGENT_GID", "2346")
    identity = helpers.resolve_submitted_process_identity()
    assert identity[:2] == (2345, 2346)

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(helpers.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        helpers.os, "setgroups", lambda groups: calls.append(("groups", groups))
    )
    monkeypatch.setattr(helpers.os, "setgid", lambda gid: calls.append(("gid", gid)))
    monkeypatch.setattr(helpers.os, "setuid", lambda uid: calls.append(("uid", uid)))
    monkeypatch.setattr(
        policy_runner,
        "_LIBC",
        type(
            "FakeLibc",
            (),
            {
                "unshare": staticmethod(
                    lambda _arg: calls.append(("unshare", None)) or 0
                )
            },
        )(),
    )

    ipc_status_r, ipc_status_w = os.pipe()
    try:
        config = helpers._submitted_process_spawn_config(
            identity,
            cwd_fd=None,
            ipc_status_fd=ipc_status_w,
        )
        assert config["pass_fds"] == (ipc_status_w,)
        config["preexec_fn"]()
        ipc_status_w = -1
        assert os.read(ipc_status_r, 1) == b"\0"
    finally:
        os.close(ipc_status_r)
        if ipc_status_w >= 0:
            os.close(ipc_status_w)

    assert calls == [
        ("unshare", None),
        ("groups", []),
        ("gid", 2346),
        ("uid", 2345),
    ]


@pytest.mark.parametrize("failure_mode", ["timeout", "overflow"])
def test_executable_kills_descendant_process_group(
    tmp_path: Path,
    local_submitted_execution: None,
    failure_mode: str,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    program = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen(['/bin/sleep', '30'])\n"
        "with open(sys.argv[1], 'w') as handle:\n"
        "    handle.write(str(child.pid))\n"
        "if sys.argv[2] == 'overflow':\n"
        "    while True:\n"
        "        os.write(1, b'A' * 65536)\n"
        "else:\n"
        "    time.sleep(30)\n"
    )
    kwargs = (
        {"timeout_s": 10, "max_output_bytes": 4096}
        if failure_mode == "overflow"
        else {"timeout_s": 1.0}
    )
    with pytest.raises(AgentFault):
        helpers.run_submitted_executable(
            [sys.executable, "-c", program, str(child_pid_path), failure_mode],
            **kwargs,
        )

    child_pid = int(child_pid_path.read_text().strip())
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"descendant process {child_pid} survived {failure_mode}")


def test_executable_kills_descendants_after_successful_leader_exit(
    tmp_path: Path,
    local_submitted_execution: None,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    program = (
        "import subprocess, sys\n"
        "child = subprocess.Popen(['/bin/sleep', '30'])\n"
        "with open(sys.argv[1], 'w') as handle:\n"
        "    handle.write(str(child.pid))\n"
        "print('done')\n"
    )

    result = helpers.run_submitted_executable(
        [sys.executable, "-c", program, str(child_pid_path)],
        timeout_s=5,
    )

    assert result.returncode == 0
    assert result.stdout == b"done\n"
    child_pid = int(child_pid_path.read_text().strip())
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"descendant process {child_pid} survived successful leader exit")


# ── load_submission_h5_or_fault ───────────────────────────────────────────


def test_h5_reads_regular_dataset(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    sub = tmp_path / "submission.h5"
    with h5py.File(sub, "w") as f:
        f.create_dataset("preds", data=np.arange(4, dtype=float))
    out = helpers.load_submission_h5_or_fault(sub, datasets=["preds"])
    assert np.array_equal(out["preds"], np.arange(4, dtype=float))


def test_h5_rejects_parent_symlink(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    private = tmp_path / "private"
    private.mkdir()
    with h5py.File(private / "submission.h5", "w") as handle:
        handle.create_dataset("preds", data=np.arange(4, dtype=float))
    workspace = tmp_path / "output"
    workspace.mkdir()
    os.symlink(private, workspace / "nested")

    with pytest.raises(AgentFault):
        helpers.load_submission_h5_or_fault(
            workspace / "nested" / "submission.h5",
            datasets=["preds"],
        )


def test_h5_rejects_external_link(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    secret = tmp_path / "truth.h5"
    with h5py.File(secret, "w") as f:
        f.create_dataset("secret", data=np.ones(4))
    sub = tmp_path / "submission.h5"
    with h5py.File(sub, "w") as f:
        f["leak"] = h5py.ExternalLink("truth.h5", "secret")
    with pytest.raises(AgentFault):
        helpers.load_submission_h5_or_fault(sub, datasets=["leak"])


def test_h5_rejects_virtual_dataset(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    layout = h5py.VirtualLayout(shape=(4,), dtype="f")
    layout[:] = h5py.VirtualSource("source.h5", "data", shape=(4,))
    sub = tmp_path / "submission.h5"
    with h5py.File(sub, "w") as f:
        f.create_virtual_dataset("vds", layout)
    with pytest.raises(AgentFault):
        helpers.load_submission_h5_or_fault(sub, datasets=["vds"])


def test_h5_missing_file_raises_agent_fault(tmp_path: Path) -> None:
    with pytest.raises(AgentFault):
        helpers.load_submission_h5_or_fault(tmp_path / "nope.h5", datasets=["x"])


def test_h5_reads_all_top_level_datasets_when_unspecified(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    sub = tmp_path / "submission.h5"
    with h5py.File(sub, "w") as f:
        f.create_dataset("a", data=np.arange(3, dtype=float))
        f.create_dataset("b", data=np.ones(2, dtype="int64"))
    out = helpers.load_submission_h5_or_fault(sub)
    assert set(out) == {"a", "b"}
    assert np.array_equal(out["a"], np.arange(3, dtype=float))
    assert np.array_equal(out["b"], np.ones(2, dtype="int64"))


def test_h5_enforces_dataset_count_and_logical_size_limits(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    sub = tmp_path / "submission.h5"
    with h5py.File(sub, "w") as handle:
        handle.create_dataset("a", data=np.arange(4, dtype="float32"))
        handle.create_dataset("b", data=np.arange(4, dtype="float32"))

    with pytest.raises(AgentFault, match="over limit"):
        helpers.load_submission_h5_or_fault(sub, max_datasets=1)
    with pytest.raises(AgentFault, match="per-dataset limit"):
        helpers.load_submission_h5_or_fault(
            sub,
            datasets=["a"],
            max_dataset_bytes=8,
        )
    with pytest.raises(AgentFault, match="aggregate limit"):
        helpers.load_submission_h5_or_fault(
            sub,
            datasets=["a", "b"],
            max_total_bytes=24,
        )


def test_h5_reads_large_multidim_dataset_via_file_transfer(tmp_path: Path) -> None:
    """Datasets cross the privilege boundary as .npy files, not a single RPC
    frame, so a read is not bounded by the policy protocol's 1 GiB wire limit.
    A few-MiB multi-dim array round-trips exactly (dtype + shape preserved)."""
    h5py = pytest.importorskip("h5py")
    sub = tmp_path / "submission.h5"
    arr = np.arange(256 * 1024, dtype="float64").reshape(512, 512)
    with h5py.File(sub, "w") as f:
        f.create_dataset("big", data=arr)
    out = helpers.load_submission_h5_or_fault(sub, datasets=["big"])
    assert out["big"].dtype == np.dtype("float64")
    assert out["big"].shape == (512, 512)
    assert np.array_equal(out["big"], arr)


def test_h5_corrupt_file_raises_agent_fault_not_crash(tmp_path: Path) -> None:
    """A crafted/corrupt .h5 must surface as an AgentFault in the parent, not a
    crash: the libhdf5 parse runs in the worker, so a malformed file that kills
    the parse is attributed to the agent (kept 0.0) and the grader survives."""
    pytest.importorskip("h5py")
    sub = tmp_path / "submission.h5"
    # HDF5 magic prefix then garbage -> libhdf5 errors while parsing the body.
    sub.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00\xff" * 4096)
    with pytest.raises(AgentFault):
        helpers.load_submission_h5_or_fault(sub, datasets=["preds"])


@pytest.mark.skipif(
    os.geteuid() != 0, reason="privilege drop only activates when grader is root"
)
def test_h5_read_runs_unprivileged_not_as_root(tmp_path: Path) -> None:
    """When the grader is root, the libhdf5 parse runs as the unprivileged agent
    account, not in the root process. Proof: a valid .h5 readable ONLY by root
    (0600 root:root) is unreadable to the uid-1000 worker, so the read fails as
    an AgentFault -- whereas an in-process root read would have succeeded.
    Mirrors test_policy_runner_sandbox.test_root_policy_cannot_read_root_only_*.
    """
    h5py = pytest.importorskip("h5py")
    sub = tmp_path / "root_only.h5"
    with h5py.File(sub, "w") as f:
        f.create_dataset("preds", data=np.arange(4, dtype=float))
    os.chmod(sub, 0o600)  # root:root 0600 -- unreadable to the agent uid
    with pytest.raises(AgentFault):
        helpers.load_submission_h5_or_fault(sub, datasets=["preds"])


# ── load_submission_h5ad_or_fault ─────────────────────────────────────────


def test_h5ad_whole_object_handoff_is_disabled(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="whole-object H5AD loading is disabled"):
        helpers.load_submission_h5ad_or_fault(
            tmp_path / "submission.h5ad",
            out_path=tmp_path / "out.h5ad",
        )


# ── score_kfold_cv (per-fold disk isolation) ─────────────────────
# The full cross-fold run wipes the agent roots between folds, so it is
# exercised by integration tasks. These cover only the pre-flight validation
# guards, which raise before any policy runs or any root is wiped.


def test_kfold_quiesce_failure_does_not_continue(monkeypatch) -> None:
    def fail_quiesce():
        raise runtime_hardening.AgentProcessQuiesceError("respawning")

    monkeypatch.setattr(
        runtime_hardening,
        "kill_pre_grade_agent_processes",
        fail_quiesce,
    )
    monkeypatch.setattr(kfold_module.os, "geteuid", lambda: 0)
    with pytest.raises(AgentFault, match="respawning"):
        kfold_module._quiesce_agent_processes_between_folds()


def test_kfold_tree_walk_error_fails_closed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "wipe-root"
    root.mkdir()
    pristine = tmp_path / "pristine"
    pristine.mkdir()
    real_walk = os.walk
    calls = 0

    def fail_walk(*_args, onerror, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            onerror(OSError(errno.ENAMETOOLONG, "path exceeds PATH_MAX"))
            return ()
        return real_walk(*_args, onerror=onerror, **_kwargs)

    monkeypatch.setattr(kfold_module.os, "walk", fail_walk)
    with pytest.raises(AgentFault, match="PATH_MAX"):
        kfold_module._capture_agent_tree([str(root)], str(pristine))


def test_kfold_pathmax_cache_is_quarantined(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "wipe-root"
    root.mkdir()
    pristine = tmp_path / "pristine"
    pristine.mkdir()
    monkeypatch.setattr(kfold_module, "_AGENT_UID", os.getuid())

    directory_fds = [os.open(root, os.O_RDONLY | os.O_DIRECTORY)]
    names: list[str] = []
    try:
        for index in range(30):
            name = f"{index:04d}-" + ("x" * 180)
            names.append(name)
            os.mkdir(name, dir_fd=directory_fds[-1])
            next_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY,
                dir_fd=directory_fds[-1],
            )
            directory_fds.append(next_fd)
        cache_fd = os.open(
            "labels.cache",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fds[-1],
        )
        os.write(cache_fd, b"held-out labels")
        os.close(cache_fd)
        with pytest.raises(AgentFault, match="securely traverse"):
            kfold_module._capture_agent_tree([str(root)], str(pristine))
        quarantined = list(root.iterdir())
        assert len(quarantined) == 1
        assert stat.S_IMODE(quarantined[0].lstat().st_mode) == 0
        if os.geteuid() == 0:
            assert quarantined[0].lstat().st_uid == 0
    finally:
        if len(directory_fds) > 1:
            os.fchmod(directory_fds[1], 0o700)
        try:
            os.unlink("labels.cache", dir_fd=directory_fds[-1])
        except FileNotFoundError:
            pass
        for index in range(len(names) - 1, -1, -1):
            os.rmdir(names[index], dir_fd=directory_fds[index])
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def _kfold_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(10),
            "f0": np.arange(10, dtype=float),
            "y": np.zeros(10, dtype=float),
            "fold": [0, 1, 2, 3, 4] * 2,
        }
    )


def test_kfold_missing_columns_raises_runtime_error() -> None:
    df = _kfold_df().drop(columns=["y"])
    with pytest.raises(RuntimeError, match="missing required columns"):
        score_kfold_cv(
            df,
            id_col="id",
            target_col="y",
            fold_col="fold",
            feature_cols=["f0"],
            metric=lambda yt, yp: 0.0,
            n_folds=5,
        )


def test_kfold_too_few_folds_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="n_folds must be >= 2"):
        score_kfold_cv(
            _kfold_df(),
            id_col="id",
            target_col="y",
            fold_col="fold",
            feature_cols=["f0"],
            metric=lambda yt, yp: 0.0,
            n_folds=1,
        )


def test_kfold_fold_values_must_match_n_folds() -> None:
    # fold ids {0..4} but n_folds=3 -> the fold_col guard rejects the mismatch.
    with pytest.raises(RuntimeError, match="must be exactly"):
        score_kfold_cv(
            _kfold_df(),
            id_col="id",
            target_col="y",
            fold_col="fold",
            feature_cols=["f0"],
            metric=lambda yt, yp: 0.0,
            n_folds=3,
        )


# ── PolicyWorker unshare_ipc opt-in ───────────────────────────────────────


def test_policy_worker_accepts_unshare_ipc(tmp_path: Path) -> None:
    policy = tmp_path / "policy.py"
    policy.write_text("def act(obs):\n    return obs\n")
    with PolicyWorker(policy, unshare_ipc=True, timeout_s=15) as worker:
        assert worker.act(7) == 7


# ── load_submission_npz_or_fault (symlink-exfil / non-regular guard) ───────


def test_npz_round_trips(tmp_path: Path) -> None:
    np.savez(tmp_path / "s.npz", y=np.arange(5), z=np.ones(3))
    out = helpers.load_submission_npz_or_fault(tmp_path / "s.npz")
    assert np.array_equal(out["y"], np.arange(5))
    np.save(tmp_path / "s.npy", np.arange(4))
    assert np.array_equal(
        helpers.load_submission_npz_or_fault(tmp_path / "s.npy"), np.arange(4)
    )


def test_npz_symlink_to_truth_is_agent_fault(tmp_path: Path) -> None:
    # A symlink to the held-out truth must NOT be followed (O_NOFOLLOW), or the
    # grader would score the truth as the agent's predictions.
    truth = tmp_path / "test_target.npz"
    np.savez(truth, y=np.arange(99))
    link = tmp_path / "predictions.npz"
    os.symlink(truth, link)
    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(link)


def test_npz_parent_symlink_to_truth_is_agent_fault(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    np.savez(private / "predictions.npz", y=np.arange(99))
    workspace = tmp_path / "output"
    workspace.mkdir()
    os.symlink(private, workspace / "nested")

    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(workspace / "nested" / "predictions.npz")


def test_npz_fifo_does_not_hang(tmp_path: Path) -> None:
    # O_NONBLOCK: a writerless FIFO opens immediately and is rejected instead of
    # blocking the grader forever.
    fifo = tmp_path / "p.npz"
    os.mkfifo(fifo)
    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(fifo)


def test_h5_readback_symlink_to_truth_is_agent_fault(tmp_path: Path) -> None:
    # The root h5 readback loads worker-written .npy files back with
    # _np_load_regular_nofollow. A worker-planted symlink to the held-out truth
    # must fail the open (O_NOFOLLOW), not be dereferenced and scored as the
    # prediction.
    truth = tmp_path / "y_true.npy"
    np.save(truth, np.arange(7))
    link = tmp_path / "0.npy"
    os.symlink(truth, link)
    with pytest.raises(AgentFault):
        helpers._np_load_regular_nofollow(str(link))


def test_h5_readback_malformed_npy_is_agent_fault(tmp_path: Path) -> None:
    malformed = tmp_path / "0.npy"
    malformed.write_bytes(b"not a numpy file")

    with pytest.raises(AgentFault, match="parsed safely"):
        helpers._np_load_regular_nofollow(str(malformed))


def test_h5_readback_fifo_does_not_hang(tmp_path: Path) -> None:
    # A worker-planted writerless FIFO in the readback dir must be rejected
    # (O_NONBLOCK + S_ISREG), not block the ROOT loader forever: this readback
    # runs in the root parent with no timeout, so a hang is itself a free veto.
    # Run in a thread so a regression (a blocking open) fails fast here instead of
    # hanging the suite.
    fifo = tmp_path / "0.npy"
    os.mkfifo(fifo)
    result: dict = {}

    def _run() -> None:
        try:
            helpers._np_load_regular_nofollow(str(fifo))
        except Exception as exc:  # noqa: BLE001  capture for the assertion below
            result["exc"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "h5 readback FIFO open hung (missing O_NONBLOCK?)"
    assert isinstance(result.get("exc"), AgentFault)


def test_npz_directory_and_oversize_and_missing_are_agent_faults(
    tmp_path: Path,
) -> None:
    os.mkdir(tmp_path / "d.npz")
    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(tmp_path / "d.npz")
    np.savez(tmp_path / "big.npz", y=np.arange(1000))
    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(tmp_path / "big.npz", max_bytes=10)
    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(tmp_path / "nope.npz")


def test_npz_decompression_bomb_is_agent_fault(tmp_path: Path) -> None:
    # A highly compressible .npz is tiny on disk (under the compressed cap) but
    # expands to MiB on member access. Its UNCOMPRESSED size must be bounded and
    # raise AgentFault (kept 0.0), not OOM the grader (MemoryError -> DISCARDED).
    bomb = tmp_path / "bomb.npz"
    np.savez_compressed(bomb, y=np.zeros(2_000_000, dtype=np.float64))  # ~16 MB flat
    assert bomb.stat().st_size < 1_000_000  # compresses to well under 1 MB on disk
    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(bomb, max_uncompressed_bytes=1_000_000)
    # Under a generous cap the same archive loads fine (no false positive).
    out = helpers.load_submission_npz_or_fault(
        bomb, max_uncompressed_bytes=64 * 1024 * 1024
    )
    assert out["y"].shape == (2_000_000,)


def test_npz_rejects_pickle_object_array_by_default(tmp_path: Path) -> None:
    np.save(tmp_path / "obj.npy", np.array({"a": 1}, dtype=object))
    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(tmp_path / "obj.npy")  # allow_pickle=False


def test_npz_rejects_allow_pickle_opt_out(tmp_path: Path) -> None:
    np.save(tmp_path / "values.npy", np.arange(3))

    with pytest.raises(ValueError, match="allow_pickle=True is forbidden"):
        helpers.load_submission_npz_or_fault(
            tmp_path / "values.npy",
            allow_pickle=True,
        )


def test_npz_rejects_object_array_member(tmp_path: Path) -> None:
    # A .npz whose MEMBER is an object array must be rejected at load (AgentFault),
    # not lazily at data[name] access -- members are materialized eagerly, so the
    # allow_pickle=False failure is a kept 0.0, not a bare ValueError the runtime
    # would discard as env_internal_failure.
    np.savez(tmp_path / "m.npz", predictions=np.array([{"a": 1}], dtype=object))
    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(tmp_path / "m.npz")


# ── require_regular_file (the pre-read guard for hand-rolled reads) ─────────


def test_require_regular_file_returns_immutable_snapshot(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("ok")
    snapshot = helpers.require_regular_file(f)
    assert snapshot != f
    assert snapshot.read_text() == "ok"

    f.write_text("changed")
    assert snapshot.read_text() == "ok"


def test_require_regular_file_rejects_symlink_fifo_dir_oversize_missing(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real.txt"
    real.write_text("x")
    link = tmp_path / "link.txt"
    os.symlink(real, link)
    with pytest.raises(AgentFault):  # symlink (os.lstat does not follow)
        helpers.require_regular_file(link)
    fifo = tmp_path / "f"
    os.mkfifo(fifo)
    with pytest.raises(AgentFault):
        helpers.require_regular_file(fifo)
    os.mkdir(tmp_path / "d")
    with pytest.raises(AgentFault):
        helpers.require_regular_file(tmp_path / "d")
    with pytest.raises(AgentFault):
        helpers.require_regular_file(real, max_bytes=0)
    with pytest.raises(AgentFault):
        helpers.require_regular_file(tmp_path / "nope")
