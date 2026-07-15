"""Leak-safe k-fold CV for dataset graders that call a submitted policy's
``fit``/``predict`` once per fold.

In K-fold CV every sample's target sits in K-1 folds' train sets, so a policy
could recover a fold's held-out targets by reading another fold's staged train
file or caching an (id -> target) map across folds. ``score_kfold_cv`` closes
the cross-fold channels: it stages only the current fold at fixed overwritten
paths; between folds it quiesces surviving uid>=1000 processes (RAM channel),
rebuilds the agent-writable area from a root-owned pristine snapshot (agent-file
channel), and restores root-owned-but-world-writable files (e.g. /tmp/uv-*.lock)
(non-agent-file channel); the kernel IPC channel is closed by each fold's worker
getting a private IPC namespace. The quiesce + reset also run after the final
fold so no target survives on disk and no daemon blocks interpreter exit.

Prediction faults and any raising ``fit``/``predict`` surface as ``AgentFault``
(kept 0.0).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from env_server.policy_loader import load_submitted_policy

from grading.faults import AgentFault
from grading.helpers import load_submission_or_fault

if TYPE_CHECKING:
    # pandas is imported lazily inside score_kfold_cv so that importing
    # `grading` does not require the scientific stack in lightweight QA venvs.
    import pandas as pd

DEFAULT_POLICY_PATH = Path("/tmp/output/policy.py")
DEFAULT_INPUTS_DIR = Path("/tmp/grader_inputs")
DEFAULT_PREDICTIONS_PATH = Path("/tmp/output/predictions.csv")

# Owners >= this uid are the agent / sandboxed policy; the grading server is
# root (uid 0) and its staged inputs are never matched by the wipe.
_AGENT_UID = 1000

# Agent-writable persistence in the task image: the agent's two homes
# (/home/model, /home/claude), the WORKDIR, /mnt/skills, and the conventional
# world-writable scratch dirs. Capture / reset skip any absent root or one with
# no uid>=1000 entry; the uid-1000 passwd home is also resolved dynamically in
# _agent_wipe_roots. Pass wipe_roots to override.
DEFAULT_WIPE_ROOTS: tuple[str, ...] = (
    "/tmp",
    "/var/tmp",
    "/dev/shm",
    "/run/lock",
    "/home/claude",
    "/home/model",
    "/mnt/skills",
    "/workdir",
    # /sys/fs/cgroup is a world-writable tmpfs (1777) here, so a uid-1000 policy
    # can cache the other folds' targets there.
    "/sys/fs/cgroup",
)


def _world_writable_mounts(mounts_path: str = "/proc/self/mounts") -> list[str]:
    """Every world-writable (o+w) mount point from /proc/self/mounts -- an
    agent-writable cross-fold cache location the reset must cover even when it is
    absent from the static wipe-root list. Safe: the reset deletes only
    uid>=_AGENT_UID entries, so root-owned mount contents are untouched."""
    out: list[str] = []
    try:
        with open(mounts_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return out
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        # /proc mount lines octal-escape space (\040) and tab (\011) in the path.
        mp = parts[1].replace("\\040", " ").replace("\\011", "\t")
        try:
            st = os.lstat(mp)
        except OSError:
            continue
        if stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode) and (st.st_mode & stat.S_IWOTH):
            out.append(mp)
    return out


def _agent_wipe_roots(roots: Sequence[str]) -> list[str]:
    """De-duplicated wipe roots, plus the agent uid's passwd home (resolved so a
    moved uid-1000 home stays covered) and every world-writable mount discovered
    at run time. Order preserved, duplicates dropped."""
    out = list(roots)
    try:
        import pwd  # noqa: PLC0415
        home = pwd.getpwuid(_AGENT_UID).pw_dir
        if home:
            out.append(home)
    except (KeyError, OSError):
        pass
    out.extend(_world_writable_mounts())
    return list(dict.fromkeys(out))

# Root-only (0700) base for the per-run pristine copy of the agent tree; not
# agent-writable and not a wipe root.
DEFAULT_PRISTINE_ROOT = "/mcp_server"


def _pristine_path(pristine_dir: str, p: str) -> str:
    """Root-only pristine location for a captured path, named by sha256(path).
    A fixed-length name (not ``pristine_dir + p``) keeps a deep agent path from
    exceeding PATH_MAX and silently skipping capture / restore."""
    return os.path.join(pristine_dir, hashlib.sha256(p.encode("utf-8", "surrogateescape")).hexdigest())


def _get_dir_xattrs(p: str) -> dict:
    out: dict = {}
    try:
        for n in os.listxattr(p, follow_symlinks=False):
            try:
                out[n] = os.getxattr(p, n, follow_symlinks=False)
            except OSError:
                pass
    except (OSError, AttributeError):
        pass
    return out


def _clear_xattrs(p: str) -> None:
    try:
        for n in os.listxattr(p, follow_symlinks=False):
            try:
                os.removexattr(p, n, follow_symlinks=False)
            except OSError:
                pass
    except (OSError, AttributeError):
        pass


def _capture_entry(seen: dict, pristine_dir: str, p: str, st) -> None:
    """Snapshot one entry into ``seen``; for a regular file also copy it into
    the pristine store."""
    if stat.S_ISLNK(st.st_mode):
        try:
            seen[p] = ("l", st.st_uid, st.st_gid, os.readlink(p))
        except OSError:
            pass
    elif stat.S_ISDIR(st.st_mode):
        seen[p] = (
            "d", st.st_uid, st.st_gid,
            (st.st_mtime_ns, st.st_atime_ns, stat.S_IMODE(st.st_mode), _get_dir_xattrs(p)),
        )
    elif stat.S_ISREG(st.st_mode):
        try:
            shutil.copy2(p, _pristine_path(pristine_dir, p), follow_symlinks=False)
        except OSError:
            return
        seen[p] = ("f", st.st_uid, st.st_gid, None)


def _capture_agent_tree(roots: Sequence[str], pristine_dir: str) -> dict:
    """Snapshot every uid>=_AGENT_UID-owned entry (plus each agent-owned wipe-root
    dir itself) into a root-only pristine store keyed by sha256(path), for the
    matching reset to rebuild from. Returns ``{path: (kind, uid, gid, extra)}``
    with kind 'f' | 'l' | 'd' (extra = readlink target for 'l',
    (mtime_ns, atime_ns, mode, xattrs) for 'd', None for 'f')."""
    seen: dict = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        # os.walk yields CHILDREN only; capture the root's own metadata so a
        # policy cannot stash a cache in its mtime/xattrs.
        try:
            rst = os.lstat(root)
            if rst.st_uid >= _AGENT_UID and stat.S_ISDIR(rst.st_mode) and not stat.S_ISLNK(rst.st_mode):
                _capture_entry(seen, pristine_dir, root, rst)
        except OSError:
            pass
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in dirnames + filenames:
                p = os.path.join(dirpath, name)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                if st.st_uid < _AGENT_UID:
                    continue
                _capture_entry(seen, pristine_dir, p, st)
    return seen


def _reset_agent_tree(initial: dict, pristine_dir: str, roots: Sequence[str]) -> None:
    """Reset the agent-writable area to its grading-start state between folds:
    DELETE every current uid>=_AGENT_UID entry (symlink-safely) then REBUILD the
    captured tree from the pristine snapshot. Deleting first closes the
    in-place-restore holes (a parent swapped for a symlink, a dir<->file
    type-swap, planted xattrs/mtime). Best-effort; unrecoverable paths skipped."""
    # 1. Remove every current agent-owned entry (bottom-up, symlink-safe), never
    #    following a symlink and never removing a wipe root itself.
    roots_set = set(roots)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
            for name in filenames + dirnames:
                p = os.path.join(dirpath, name)
                if p in roots_set:
                    continue
                try:
                    st = os.lstat(p)
                    if st.st_uid < _AGENT_UID:
                        continue
                    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                        os.unlink(p)
                    else:
                        try:
                            os.rmdir(p)
                        except OSError:
                            pass  # non-empty (root-owned children remain)
                except OSError:
                    continue
    # 2. Recreate captured entries, parents first. Step 1 removed every agent
    #    symlink, so no path component can redirect a write.
    for p, (kind, uid, gid, extra) in sorted(initial.items(), key=lambda kv: kv[0].count(os.sep)):
        try:
            parent = os.path.dirname(p)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            if kind == "d":
                os.makedirs(p, exist_ok=True)
            elif kind == "l":
                if os.path.lexists(p):
                    try:
                        if os.path.islink(p) or not os.path.isdir(p):
                            os.unlink(p)
                        else:
                            shutil.rmtree(p, ignore_errors=True)
                    except OSError:
                        pass
                os.symlink(extra, p)
                try:
                    os.lchown(p, uid, gid)
                except OSError:
                    pass
            elif kind == "f":
                if os.path.lexists(p):
                    try:
                        if os.path.isdir(p) and not os.path.islink(p):
                            shutil.rmtree(p, ignore_errors=True)
                        else:
                            os.unlink(p)
                    except OSError:
                        pass
                shutil.copy2(_pristine_path(pristine_dir, p), p, follow_symlinks=False)
                try:
                    os.chown(p, uid, gid)
                except OSError:
                    pass
        except OSError:
            continue
    # 3. Re-stamp captured-directory metadata LAST, deepest first (re-creating a
    #    child bumps its parent's mtime). Clear policy xattrs, then restore
    #    captured xattrs / mode / mtime / atime.
    for p, (kind, uid, gid, extra) in sorted(initial.items(), key=lambda kv: kv[0].count(os.sep), reverse=True):
        if kind != "d":
            continue
        if not (os.path.isdir(p) and not os.path.islink(p)):
            continue
        mtime_ns, atime_ns, mode, xattrs = extra
        try:
            _clear_xattrs(p)
            for n, v in xattrs.items():
                try:
                    os.setxattr(p, n, v, follow_symlinks=False)
                except OSError:
                    pass
            try:
                os.chown(p, uid, gid)
            except OSError:
                pass
            os.chmod(p, mode)
            os.utime(p, ns=(atime_ns, mtime_ns))
        except OSError:
            continue


def _capture_writable_nonagent_files(roots: Sequence[str], pristine_dir: str) -> dict[str, tuple[int, int, int]]:
    """Capture root-owned (uid < _AGENT_UID) but world-writable regular files.

    A root-owned yet world-writable file (e.g. /tmp/uv-*.lock) is a cross-fold
    channel the agent-owned reset never covers: the policy can write an
    (id -> target) map into one and read it back in a later fold. The agent can
    neither create nor widen such a file, so the set is fixed at image build and
    captured once. Copies each into ``pristine_dir``. Returns {path: (uid, gid,
    mode)}.
    """
    out: dict[str, tuple[int, int, int]] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
            for name in filenames:
                p = os.path.join(dirpath, name)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue  # symlink / socket / device: not a content channel
                if st.st_uid >= _AGENT_UID:
                    continue  # agent-owned: handled by _capture_agent_tree
                if not (st.st_mode & stat.S_IWOTH):
                    continue  # root-owned but not agent-writable: not a channel
                dst = pristine_dir + p
                try:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(p, dst, follow_symlinks=False)
                except OSError:
                    continue
                out[p] = (st.st_uid, st.st_gid, stat.S_IMODE(st.st_mode))
    return out


def _restore_writable_nonagent_files(captured: dict[str, tuple[int, int, int]], pristine_dir: str) -> None:
    """Revert each captured root-owned world-writable file to its start content /
    owner / mode between folds, wiping any cache the policy wrote. Best-effort."""
    for p, (uid, gid, mode) in captured.items():
        src = pristine_dir + p
        try:
            if not os.path.isfile(src):
                continue
            shutil.copy2(src, p, follow_symlinks=False)
            try:
                os.chown(p, uid, gid)
            except OSError:
                pass
            try:
                os.chmod(p, mode)
            except OSError:
                pass
        except OSError:
            continue


# Per-RPC timeout for the policy's fit/predict during CV. PolicyWorker defaults
# to 5s (too short for a real fit/predict), so kfold raises the ceiling.
DEFAULT_KFOLD_RPC_TIMEOUT_S = 300.0


def _load_policy_or_fault(policy_path: Path, timeout_s: float):
    try:
        return load_submitted_policy(
            policy_path, timeout_s=timeout_s, first_call_timeout_s=timeout_s
        )
    except FileNotFoundError as exc:
        raise AgentFault(f"missing submitted policy at {policy_path}") from exc


def _quiesce_agent_processes_between_folds() -> None:
    """SIGKILL every surviving uid>=1000 process between folds.

    Closes the RAM channel: a daemon spawned during ``fit`` (which read K-1/K of
    the targets) can re-serve the (id -> target) map with nothing left on disk.
    After the per-fold ``policy.close()`` the only uid>=1000 processes are
    agent-spawned, so this is safe; reuses the runner's pre-grade quiesce.
    Best-effort: import or kill failure never aborts grading.
    """
    try:
        from grading.runtime_hardening import (  # noqa: PLC0415
            kill_pre_grade_agent_processes,
        )
        kill_pre_grade_agent_processes()
    except Exception:
        pass


def _ensure_root_owned_dir(d: Path) -> None:
    """Guarantee ``d`` is a real, root-owned directory before staging fold inputs.

    An agent that pre-creates it as a symlink to ``/mcp_server`` would otherwise
    have the chmod below FOLLOW the link and expose the root-only tree, so a
    symlink / non-owned dir is removed and recreated as root. /tmp is sticky, so
    once this root-owned dir exists a uid-1000 agent cannot replace it."""
    try:
        st = os.lstat(d)
    except FileNotFoundError:
        st = None
    if st is not None:
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            os.unlink(d)
        elif st.st_uid != 0:
            shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    if os.path.islink(d):
        raise RuntimeError(f"refusing to chmod a symlinked grader inputs dir: {d}")
    os.chmod(d, 0o755)


def score_kfold_cv(
    full_df: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
    fold_col: str,
    feature_cols: Sequence[str],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_folds: int,
    policy_path: Path = DEFAULT_POLICY_PATH,
    inputs_dir: Path = DEFAULT_INPUTS_DIR,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    prediction_col: str = "prediction",
    wipe_roots: Sequence[str] = DEFAULT_WIPE_ROOTS,
    pristine_root: str = DEFAULT_PRISTINE_ROOT,
    rpc_timeout_s: float | None = None,
) -> float:
    """Run ``n_folds`` CV with per-fold disk isolation; return the mean metric.

    ``full_df`` must hold ``id_col``, ``target_col``, ``fold_col`` and every
    ``feature_cols`` column. Per fold the policy is handed a train parquet
    (``id_col`` + ``feature_cols`` + ``target_col``) via ``fit(path)`` and a
    test parquet (``id_col`` + ``feature_cols``, target stripped) via
    ``predict(path)``, and must write ``predictions_path`` with columns
    ``id_col`` and ``prediction_col``. ``metric(y_true, y_pred) -> float`` is
    averaged over folds. See the module docstring for the leak model.

    Raises ``RuntimeError`` for an author/infra fault (missing dataset
    columns); ``AgentFault`` for any submission-attributable fault.
    """
    # pandas is provisioned in the task image at grade time, which is the only
    # place this scoring helper runs.
    import pandas as pd

    inputs_dir = Path(inputs_dir)
    predictions_path = Path(predictions_path)
    feature_cols = list(feature_cols)
    wipe_roots = _agent_wipe_roots(wipe_roots)

    missing = [c for c in (id_col, target_col, fold_col, *feature_cols) if c not in full_df.columns]
    if missing:
        raise RuntimeError(f"dataset missing required columns {missing}")

    if n_folds < 2:
        raise RuntimeError(f"n_folds must be >= 2 for cross-validation; got {n_folds}")
    try:
        fold_values = {int(v) for v in pd.unique(full_df[fold_col])}
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{fold_col!r} must hold integer fold ids: {exc}") from exc
    if fold_values != set(range(n_folds)):
        raise RuntimeError(
            f"{fold_col!r} values {sorted(fold_values)} must be exactly "
            f"0..{n_folds - 1}: a value outside that range is never held out, "
            f"and a missing fold has no test split"
        )

    # Per-RPC budget for each fold's fit/predict; None -> the 300s default
    # (PolicyWorker's 5s would spuriously AgentFault a slow fit/predict).
    rpc_timeout_s = (
        DEFAULT_KFOLD_RPC_TIMEOUT_S if rpc_timeout_s is None else float(rpc_timeout_s)
    )

    train_path = inputs_dir / "fold_train.parquet"
    test_path = inputs_dir / "fold_test_inputs.parquet"

    pristine_dir = os.path.join(pristine_root, f".grader_kfold_pristine.{os.getpid()}")
    shutil.rmtree(pristine_dir, ignore_errors=True)
    os.makedirs(pristine_dir, exist_ok=True)
    initial = _capture_agent_tree(wipe_roots, pristine_dir)
    writable_nonagent = _capture_writable_nonagent_files(wipe_roots, pristine_dir)
    fold_metrics: list[float] = []
    try:
        for fold in range(n_folds):
            if fold > 0:
                # Kill any agent daemon BEFORE wiping the disk so a dying process
                # cannot re-plant a cache mid-reset, then reset the disk.
                _quiesce_agent_processes_between_folds()
                _reset_agent_tree(initial, pristine_dir, wipe_roots)
                _restore_writable_nonagent_files(writable_nonagent, pristine_dir)

            _ensure_root_owned_dir(inputs_dir)

            train_df = full_df.loc[
                full_df[fold_col] != fold, [id_col, *feature_cols, target_col]
            ].reset_index(drop=True)
            test_df = full_df.loc[
                full_df[fold_col] == fold, [id_col, *feature_cols]
            ].reset_index(drop=True)
            labels = full_df.loc[
                full_df[fold_col] == fold, [id_col, target_col]
            ].reset_index(drop=True)

            for p in (train_path, test_path):
                if p.exists():
                    p.unlink()
            train_df.to_parquet(train_path, index=False)
            test_df.to_parquet(test_path, index=False)
            os.chmod(train_path, 0o644)
            os.chmod(test_path, 0o644)
            if predictions_path.exists():
                predictions_path.unlink()

            policy = _load_policy_or_fault(policy_path, rpc_timeout_s)
            try:
                policy.fit(str(train_path))
                policy.predict(str(test_path))
            except Exception as exc:
                raise AgentFault(
                    f"fold {fold}: submitted policy failed during fit/predict: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            finally:
                # Tear down this fold's sandbox worker; without this they
                # accumulate (daemon=True) and memory / process count grow.
                try:
                    policy.close()
                except Exception:
                    pass

            preds = load_submission_or_fault(
                predictions_path,
                required_columns=[id_col, prediction_col],
                numeric_columns=[prediction_col],
                unique_key_column=id_col,
            )
            merged = labels.merge(
                preds[[id_col, prediction_col]], on=id_col, how="inner", validate="1:1"
            )
            if len(merged) != len(labels):
                raise AgentFault(
                    f"fold {fold}: predictions cover {len(merged)}/{len(labels)} "
                    "test ids on a 1:1 id join"
                )
            y_true = merged[target_col].to_numpy(dtype=float)
            y_pred = merged[prediction_col].to_numpy(dtype=float)
            fold_metrics.append(float(metric(y_true, y_pred)))

        return float(np.mean(fold_metrics))
    finally:
        # The between-fold quiesce + reset never run after the LAST fold, leaving
        # two channels open: a daemon the last fold forked can block the grader at
        # interpreter exit, and the last fold's fit/predict scratch (with targets
        # it saw) stays on disk for a re-grade or checkpoint to capture. So
        # quiesce + reset once more before dropping the pristine store.
        try:
            _quiesce_agent_processes_between_folds()
        except Exception:
            pass
        try:
            _reset_agent_tree(initial, pristine_dir, wipe_roots)
            _restore_writable_nonagent_files(writable_nonagent, pristine_dir)
        except Exception:
            pass
        shutil.rmtree(pristine_dir, ignore_errors=True)
