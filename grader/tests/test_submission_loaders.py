from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grading import helpers, score_kfold_cv
from grading.faults import AgentFault
from grading.policy_runner import PolicyWorker


def _write_script(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# ── run_submitted_executable (capture + streaming, uid-1000 drop) ──
# The privilege drop no-ops as a non-root test process, so these run without setuid.


def test_executable_capture_returns_stdout_and_returncode(tmp_path: Path) -> None:
    exe = _write_script(tmp_path / "run.sh", "#!/bin/sh\necho hello\nexit 0\n")
    proc = helpers.run_submitted_executable([str(exe)], timeout_s=10)
    assert proc.returncode == 0
    assert b"hello" in proc.stdout


def test_executable_passes_args_and_returncode(tmp_path: Path) -> None:
    exe = _write_script(tmp_path / "run.sh", '#!/bin/sh\necho "$1"\nexit 3\n')
    proc = helpers.run_submitted_executable([str(exe), "payload"], timeout_s=10)
    assert proc.returncode == 3
    assert proc.stdout.strip() == b"payload"


def test_executable_stdin_bytes_piped_to_child(tmp_path: Path) -> None:
    exe = _write_script(tmp_path / "cat.sh", "#!/bin/sh\ncat\n")
    proc = helpers.run_submitted_executable([str(exe)], stdin_bytes=b"ping", timeout_s=10)
    assert proc.stdout == b"ping"


def test_executable_sanitized_env_hides_grader_secrets(
    tmp_path: Path, monkeypatch
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
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MY_TASK_CONFIG", "visible")
    exe = _write_script(
        tmp_path / "env.sh", '#!/bin/sh\necho "${MY_TASK_CONFIG:-MISSING}"\n'
    )
    proc = helpers.run_submitted_executable(
        [str(exe)], env_passthrough=True, timeout_s=10
    )
    assert proc.stdout.strip() == b"visible"


def test_executable_timeout_raises_agent_fault(tmp_path: Path) -> None:
    # A capture-mode timeout is an agent-caused failure: it must raise AgentFault
    # (kept 0.0), NOT the builtin TimeoutExpired, which would escape compute_score
    # as env_internal_failure and DISCARD the rollout (a free veto).
    exe = _write_script(tmp_path / "slow.sh", "#!/bin/sh\nsleep 5\n")
    with pytest.raises(AgentFault):
        helpers.run_submitted_executable([str(exe)], timeout_s=0.3)


def test_executable_stdout_flood_raises_agent_fault(tmp_path: Path) -> None:
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


def test_executable_capture_discards_stderr(tmp_path: Path) -> None:
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
        helpers.run_submitted_executable(
            [str(exe)], streaming=True, stdin_bytes=b"x"
        )
    with pytest.raises(ValueError):
        helpers.run_submitted_executable([str(exe)], streaming=True, timeout_s=5)


def test_executable_streaming_filters_rubric_score(tmp_path: Path, capsys) -> None:
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


# ── load_submission_h5_or_fault ───────────────────────────────────────────


def test_h5_reads_regular_dataset(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    sub = tmp_path / "submission.h5"
    with h5py.File(sub, "w") as f:
        f.create_dataset("preds", data=np.arange(4, dtype=float))
    out = helpers.load_submission_h5_or_fault(sub, datasets=["preds"])
    assert np.array_equal(out["preds"], np.arange(4, dtype=float))


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


def test_h5ad_sanitizer_accepts_valid_anndata(tmp_path: Path) -> None:
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("h5py")
    src = tmp_path / "submission.h5ad"
    anndata.AnnData(X=np.eye(3, dtype="float32")).write_h5ad(src)
    out = tmp_path / "clean.h5ad"
    result = helpers.load_submission_h5ad_or_fault(src, out_path=out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    # The sanitized copy round-trips back through anndata with the same X.
    assert np.array_equal(anndata.read_h5ad(out).X, np.eye(3, dtype="float32"))


def test_h5ad_sanitizer_rejects_malformed(tmp_path: Path) -> None:
    pytest.importorskip("anndata")
    src = tmp_path / "broken.h5ad"
    src.write_bytes(b"this is not an HDF5 / AnnData file at all")
    with pytest.raises(AgentFault):
        helpers.load_submission_h5ad_or_fault(src, out_path=tmp_path / "out.h5ad")


def test_h5ad_sanitizer_missing_file_raises_agent_fault(tmp_path: Path) -> None:
    with pytest.raises(AgentFault):
        helpers.load_submission_h5ad_or_fault(
            tmp_path / "nope.h5ad", out_path=tmp_path / "out.h5ad"
        )


# ── score_kfold_cv (per-fold disk isolation) ─────────────────────
# The full cross-fold run wipes the agent roots between folds, so it is
# exercised by integration tasks. These cover only the pre-flight validation
# guards, which raise before any policy runs or any root is wiped.


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
    assert np.array_equal(helpers.load_submission_npz_or_fault(tmp_path / "s.npy"), np.arange(4))


def test_npz_symlink_to_truth_is_agent_fault(tmp_path: Path) -> None:
    # A symlink to the held-out truth must NOT be followed (O_NOFOLLOW), or the
    # grader would score the truth as the agent's predictions.
    truth = tmp_path / "test_target.npz"
    np.savez(truth, y=np.arange(99))
    link = tmp_path / "predictions.npz"
    os.symlink(truth, link)
    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(link)


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


def test_npz_directory_and_oversize_and_missing_are_agent_faults(tmp_path: Path) -> None:
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
    out = helpers.load_submission_npz_or_fault(bomb, max_uncompressed_bytes=64 * 1024 * 1024)
    assert out["y"].shape == (2_000_000,)


def test_npz_rejects_pickle_object_array_by_default(tmp_path: Path) -> None:
    np.save(tmp_path / "obj.npy", np.array({"a": 1}, dtype=object))
    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(tmp_path / "obj.npy")  # allow_pickle=False


def test_npz_rejects_object_array_member(tmp_path: Path) -> None:
    # A .npz whose MEMBER is an object array must be rejected at load (AgentFault),
    # not lazily at data[name] access -- members are materialized eagerly, so the
    # allow_pickle=False failure is a kept 0.0, not a bare ValueError the runtime
    # would discard as env_internal_failure.
    np.savez(tmp_path / "m.npz", predictions=np.array([{"a": 1}], dtype=object))
    with pytest.raises(AgentFault):
        helpers.load_submission_npz_or_fault(tmp_path / "m.npz")


# ── require_regular_file (the pre-read guard for hand-rolled reads) ─────────


def test_require_regular_file_accepts_regular_returns_path(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("ok")
    assert helpers.require_regular_file(f) == f


def test_require_regular_file_rejects_symlink_fifo_dir_oversize_missing(tmp_path: Path) -> None:
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
