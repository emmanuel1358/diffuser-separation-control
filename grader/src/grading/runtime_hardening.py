"""Runtime hardening shared by Boreal and Harbor grading entry points."""

from __future__ import annotations

import logging
import os
import signal
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_NON_SYSTEM_UID_THRESHOLD = 1000
_QUIESCE_MAX_PASSES = 32
_QUIESCE_CONSECUTIVE_ZEROS = 2
_AGENT_PERM_BITS = (
    stat.S_IRGRP
    | stat.S_IWGRP
    | stat.S_IXGRP
    | stat.S_IROTH
    | stat.S_IWOTH
    | stat.S_IXOTH
)

_INFRA_RETURNCODES: frozenset[int] = frozenset(
    {
        137,  # 128 + SIGKILL, usually OOM
        139,  # 128 + SIGSEGV
        134,  # 128 + SIGABRT
        -signal.SIGKILL,
        -signal.SIGSEGV,
        -signal.SIGABRT,
    }
)

_INFRA_PATTERNS: tuple[str, ...] = (
    "memoryerror",
    "outofmemoryerror",
    "out of memory",
    "[errno 28]",
    "[errno 12]",
    "no space left on device",
    "cuda error",
    "cudnn error",
    "devicelost",
)
_CACHE_ENV_KEYS = (
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "TORCH_HOME",
    "HF_HOME",
)


@dataclass(frozen=True)
class FailureClassification:
    """Best-effort classification for a failed grader subprocess."""

    is_infra: bool
    reason: str


def classify_failure(returncode: int, stderr_tail: str) -> FailureClassification:
    """Classify a failed grader subprocess for operator metadata."""
    if returncode in _INFRA_RETURNCODES:
        return FailureClassification(
            True, f"grader subprocess exited from signal-like returncode {returncode}"
        )
    lowered = stderr_tail.lower()
    for pattern in _INFRA_PATTERNS:
        if pattern in lowered:
            return FailureClassification(
                True, f"grader stderr matched infra pattern {pattern!r}"
            )
    return FailureClassification(
        False, f"grader subprocess failed with exit {returncode}"
    )


def isolated_grader_environ(*, cache_root: str | Path | None = None) -> dict[str, str]:
    """Return an environment suitable for root-side grader subprocesses."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONSAFEPATH"] = "1"
    if cache_root is not None:
        root = str(cache_root)
        env.update(
            XDG_CACHE_HOME=root,
            MPLCONFIGDIR=f"{root}/matplotlib",
            NUMBA_CACHE_DIR=f"{root}/numba",
            TORCH_HOME=f"{root}/torch",
            HF_HOME=f"{root}/huggingface",
        )
    else:
        for key in _CACHE_ENV_KEYS:
            env.pop(key, None)
    return env


def prepare_grader_cache(
    cache_root: str | Path = "/mcp_server/.grader_cache",
) -> dict[str, str]:
    """Create an agent-unwritable cache root and return env vars for it."""
    root = Path(cache_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o700)
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "lbx-grader-cache"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            os.chmod(fallback, 0o700)
            logger.warning(
                "[GRADING] could not prepare private grader cache %s: %s; using %s",
                root,
                exc,
                fallback,
            )
            root = fallback
        except OSError:
            logger.warning(
                "[GRADING] could not prepare private grader cache %s: %s; "
                "omitting grader cache env vars",
                root,
                exc,
            )
            return isolated_grader_environ(cache_root=None)
    return isolated_grader_environ(cache_root=root)


def _read_proc_state(proc_dir: str) -> str:
    try:
        with open(f"{proc_dir}/stat", "rb") as handle:
            stat_line = handle.read()
    except OSError:
        return ""
    rparen = stat_line.rfind(b")")
    if rparen == -1 or rparen + 2 >= len(stat_line):
        return ""
    return stat_line[rparen + 2 : rparen + 3].decode("ascii", errors="replace")


def kill_pre_grade_agent_processes(
    *,
    min_uid: int = _NON_SYSTEM_UID_THRESHOLD,
    max_passes: int = _QUIESCE_MAX_PASSES,
    required_zero_passes: int = _QUIESCE_CONSECUTIVE_ZEROS,
) -> int:
    """SIGKILL uid>=1000 processes before root-side grading reads outputs."""
    self_pid = os.getpid()
    parent_pid = os.getppid()
    protected = {self_pid, parent_pid, 1}
    total_killed = 0
    consecutive_zeros = 0

    for pass_idx in range(max_passes):
        try:
            proc_entries = os.listdir("/proc")
        except OSError as exc:
            logger.warning(
                "[GRADING] could not enumerate /proc during quiesce: %s", exc
            )
            return total_killed

        eligible: list[tuple[int, int, str]] = []
        for entry in proc_entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid in protected:
                continue
            proc_path = f"/proc/{entry}"
            try:
                uid = os.stat(proc_path).st_uid
            except (FileNotFoundError, OSError):
                continue
            if uid < min_uid or _read_proc_state(proc_path) == "Z":
                continue
            cmd = "<unknown>"
            try:
                with open(f"{proc_path}/cmdline", "rb") as handle:
                    raw = handle.read()
                cmd = (
                    raw.replace(b"\x00", b" ").strip().decode(errors="replace")
                    or "<empty>"
                )
            except OSError:
                pass
            eligible.append((pid, uid, cmd))

        if not eligible:
            consecutive_zeros += 1
            if consecutive_zeros >= required_zero_passes:
                return total_killed
            continue

        consecutive_zeros = 0
        for pid, uid, cmd in eligible:
            try:
                os.kill(pid, signal.SIGKILL)
                total_killed += 1
                logger.warning(
                    "[GRADING] pre-grade SIGKILL pid=%d uid=%d cmd=%r", pid, uid, cmd
                )
            except ProcessLookupError:
                continue
            except (PermissionError, OSError) as exc:
                logger.warning(
                    "[GRADING] could not SIGKILL pid=%d uid=%d cmd=%r: %s",
                    pid,
                    uid,
                    cmd,
                    exc,
                )

    logger.error(
        "[GRADING] pre-grade quiesce did not converge after %d passes; killed %d process(es)",
        max_passes,
        total_killed,
    )
    return total_killed


_NVIDIA_DEVICE_PREFIX = "/dev/nvidia"


def _holds_nvidia_fd(proc_path: str) -> bool:
    """True if any open fd of ``proc_path`` resolves to a ``/dev/nvidia*`` node.

    Pure ``/proc`` inspection so it is unit-testable against a fake proc tree:
    point ``proc_path`` at a directory whose ``fd/`` holds symlinks to
    arbitrary targets. Any unreadable fd is skipped, never raised.
    """
    fd_dir = f"{proc_path}/fd"
    try:
        fd_entries = os.listdir(fd_dir)
    except OSError:
        return False
    for fd_name in fd_entries:
        try:
            target = os.readlink(f"{fd_dir}/{fd_name}")
        except OSError:
            continue
        if target.startswith(_NVIDIA_DEVICE_PREFIX):
            return True
    return False


def kill_nvproxy_fd_holders() -> int:
    """SIGKILL processes holding ``/dev/nvidia*`` fds so the post-grade checkpoint can save.

    gVisor/nvproxy refuses to checkpoint a container while any process still
    holds an open NVIDIA device fd. The agent may leave such a holder behind
    (e.g. a training/inference job in tmux). The grade score is already written
    before this runs, so this is strictly best-effort cleanup: it must never
    raise, never change the score, and never touch self/parent/init. Returns the
    number of processes killed. Safe (zero kills) on CPU-only tasks.
    """
    self_pid = os.getpid()
    parent_pid = os.getppid()
    protected = {self_pid, parent_pid, 1}
    killed = 0

    try:
        proc_entries = os.listdir("/proc")
    except OSError as exc:
        logger.warning("[GRADING] could not enumerate /proc for nvproxy sweep: %s", exc)
        return 0

    for entry in proc_entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in protected:
            continue
        proc_path = f"/proc/{entry}"
        try:
            if _read_proc_state(proc_path) == "Z":
                continue
            if not _holds_nvidia_fd(proc_path):
                continue
        except OSError:
            continue
        cmd = "<unknown>"
        try:
            with open(f"{proc_path}/cmdline", "rb") as handle:
                raw = handle.read()
            cmd = raw.replace(b"\x00", b" ").strip().decode(errors="replace") or "<empty>"
        except OSError:
            pass
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
            logger.warning(
                "[GRADING] nvproxy-sweep SIGKILL pid=%d cmd=%r (held /dev/nvidia* fd)",
                pid,
                cmd,
            )
        except ProcessLookupError:
            continue
        except (PermissionError, OSError) as exc:
            logger.warning(
                "[GRADING] could not SIGKILL nvproxy holder pid=%d cmd=%r: %s",
                pid,
                cmd,
                exc,
            )

    if killed:
        logger.warning("[GRADING] nvproxy sweep killed %d process(es)", killed)
    return killed


def scrub_escaping_symlinks(output_dir: str | Path) -> int:
    """Remove symlinks under ``output_dir`` that resolve outside it."""
    output = os.fspath(output_dir)
    if os.path.islink(output):
        try:
            target = os.readlink(output)
        except OSError:
            target = "<unresolved>"
        try:
            os.unlink(output)
            logger.warning(
                "[GRADING] removed top-level output symlink %s -> %s", output, target
            )
            return 1
        except OSError as exc:
            logger.warning(
                "[GRADING] failed to remove top-level output symlink %s -> %s: %s",
                output,
                target,
                exc,
            )
            return 0

    real_output = os.path.realpath(output).rstrip(os.sep)
    if not os.path.isdir(real_output):
        return 0
    output_prefix = real_output + os.sep
    removed = 0

    for dirpath, dirnames, filenames in os.walk(real_output, followlinks=False):
        for name in dirnames + filenames:
            entry = os.path.join(dirpath, name)
            try:
                if not os.path.islink(entry):
                    continue
                target = os.path.realpath(entry)
            except OSError:
                continue
            if target == real_output or target.startswith(output_prefix):
                continue
            try:
                os.unlink(entry)
                removed += 1
                logger.warning(
                    "[GRADING] removed escaping symlink %s -> %s", entry, target
                )
            except OSError as exc:
                logger.warning(
                    "[GRADING] failed to remove escaping symlink %s -> %s: %s",
                    entry,
                    target,
                    exc,
                )
    return removed


def scrub_nonregular_files(output_dir: str | Path) -> int:
    """Remove FIFOs, sockets, and device nodes under ``output_dir``."""
    output = os.fspath(output_dir)
    if os.path.islink(output) or not os.path.isdir(output):
        return 0
    removed = 0
    for dirpath, _dirnames, filenames in os.walk(output, followlinks=False):
        for name in filenames:
            entry = os.path.join(dirpath, name)
            try:
                mode = os.lstat(entry).st_mode
            except OSError:
                continue
            if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                continue
            try:
                os.unlink(entry)
                removed += 1
                logger.warning(
                    "[GRADING] removed non-regular submission entry %s (mode=%s)",
                    entry,
                    stat.filemode(mode),
                )
            except OSError as exc:
                logger.warning(
                    "[GRADING] failed to remove non-regular entry %s: %s", entry, exc
                )
    return removed


def pre_grade_cleanup(output_dir: str | Path) -> dict[str, int]:
    """Quiesce agent processes, then scrub dangerous output entries."""
    killed = 0
    if os.name == "posix" and os.geteuid() == 0 and Path("/proc").exists():
        killed = kill_pre_grade_agent_processes()
    symlinks = scrub_escaping_symlinks(output_dir)
    nonregular = scrub_nonregular_files(output_dir)
    return {
        "killed_processes": killed,
        "removed_symlinks": symlinks,
        "removed_nonregular": nonregular,
    }


def lock_down_grader_private(paths: tuple[str | Path, ...]) -> None:
    """Best-effort root ownership and no group/other bits for private trees."""
    targets: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() and not path.is_symlink():
            continue
        targets.append(path)
        if path.is_dir() and not path.is_symlink():
            for dirpath, dirnames, filenames in os.walk(path):
                targets.extend(Path(dirpath) / name for name in (*dirnames, *filenames))

    for path in targets:
        try:
            info = os.lstat(path)
        except OSError as exc:
            logger.warning(
                "[SETUP_GUARD] could not stat grader-private path %s: %s", path, exc
            )
            continue
        if stat.S_ISLNK(info.st_mode):
            continue
        if info.st_uid != 0 or info.st_gid != 0:
            try:
                os.chown(path, 0, 0)
                logger.warning("[SETUP_GUARD] reset root ownership on %s", path)
            except OSError as exc:
                logger.warning(
                    "[SETUP_GUARD] could not reset root ownership on %s: %s", path, exc
                )
        if info.st_mode & _AGENT_PERM_BITS:
            try:
                os.chmod(path, info.st_mode & ~_AGENT_PERM_BITS)
                logger.warning("[SETUP_GUARD] tightened group/other perms on %s", path)
            except OSError as exc:
                logger.warning(
                    "[SETUP_GUARD] could not tighten perms on %s: %s", path, exc
                )
