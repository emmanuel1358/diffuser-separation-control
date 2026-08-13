"""Runtime hardening shared by Boreal and Harbor grading entry points."""

from __future__ import annotations

import errno
import logging
import os
import pwd
import resource
import signal
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from grading.faults import AgentFault, InfrastructureFault
from grading.secure_io import open_directory_fd

logger = logging.getLogger(__name__)

SHARED_RUNTIME_SECURITY_REVISION = "2026-08-12.1"
_NON_SYSTEM_UID_THRESHOLD = 1000
_QUIESCE_MAX_PASSES = 32
_QUIESCE_CONSECUTIVE_ZEROS = 2


class ProcessQuiesceError(InfrastructureFault):
    """The runtime could not prove that all agent processes stopped."""


class AgentProcessQuiesceError(AgentFault):
    """Agent-owned processes kept respawning and prevented quiescence."""


class AgentTmpfsFloodError(AgentProcessQuiesceError):
    """Agent-owned shared-memory entries exceeded the cleanup budget."""


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
_AGENT_TMPFS_ROOTS: tuple[Path, ...] = (Path("/dev/shm"),)
_AGENT_TMPFS_MAX_ENTRIES = 200_000
_AGENT_TMPFS_MAX_WALK_S = 20.0
_FS_MIN_FREE_BYTES = 4 * 1024 * 1024
_FS_MIN_FREE_INODES = 256
_SHMEM_EXHAUSTED_MIN_KIB = 512 * 1024
_SHMEM_EXHAUSTED_MIN_FRACTION = 0.50
_MEMAVAILABLE_EXHAUSTED_MAX_KIB = 512 * 1024
_DISK_FULL_MARKERS = (
    f"[errno {errno.ENOSPC}]",
    f"[errno {errno.EDQUOT}]",
    "no space left on device",
    "disk quota exceeded",
)
_MEMORY_PRESSURE_MARKERS = (
    "killed",
    "out of memory",
    "cannot allocate memory",
    "oom-kill",
    "oom killed",
)
_DEFAULT_AGENT_MEMORY_LIMIT_BYTES = 56 * 1024**3
_AGENT_MEMORY_LIMIT_FRACTION = 0.75
_CGROUP_MEMORY_LIMIT_PATHS = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)
_UNBOUNDED_CGROUP_MEMORY_BYTES = 1 << 60


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


def filesystem_exhausted(path: str | Path) -> bool:
    """Conservatively detect exhausted blocks or inodes on a known filesystem."""
    try:
        fs = os.statvfs(path)
    except OSError:
        return False
    free_bytes = fs.f_bfree * fs.f_frsize
    inodes_exhausted = fs.f_files > 0 and fs.f_ffree < _FS_MIN_FREE_INODES
    return free_bytes < _FS_MIN_FREE_BYTES or inodes_exhausted


def stderr_shows_disk_full(stderr_tail: str | None) -> bool:
    lowered = (stderr_tail or "").lower()
    return any(marker in lowered for marker in _DISK_FULL_MARKERS)


def stderr_shows_memory_pressure(stderr_tail: str | None) -> bool:
    lowered = (stderr_tail or "").lower()
    return any(marker in lowered for marker in _MEMORY_PRESSURE_MARKERS)


def meminfo_kib(path: str | Path = "/proc/meminfo") -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    try:
                        values[parts[0][:-1]] = int(parts[1])
                    except ValueError:
                        pass
    except OSError:
        pass
    return values


def _read_cgroup_memory_limit(
    paths: tuple[Path, ...] = _CGROUP_MEMORY_LIMIT_PATHS,
) -> int | None:
    """Return the finite container memory limit, if cgroups expose one."""
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if 0 < value < _UNBOUNDED_CGROUP_MEMORY_BYTES:
            return value
    return None


def child_memory_limit_bytes(
    *,
    explicit_bytes: int | None = None,
    env_keys: tuple[str, ...] = ("RUBRIC_AGENT_MEMORY_LIMIT_BYTES",),
    cgroup_paths: tuple[Path, ...] = _CGROUP_MEMORY_LIMIT_PATHS,
) -> int:
    """Resolve the hard address-space cap for untrusted child processes.

    An explicit API value wins, followed by the first configured environment
    key. Otherwise the cap is 75% of the container limit (leaving room for the
    root MCP/grader), bounded by a conservative 56 GiB fallback for runtimes
    that do not expose cgroup limits.
    """
    if explicit_bytes is not None:
        if explicit_bytes <= 0:
            raise ValueError("child memory limit must be positive")
        return int(explicit_bytes)

    for key in env_keys:
        raw = os.environ.get(key)
        if raw in (None, ""):
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} must be a positive integer byte count") from exc
        if value <= 0:
            raise ValueError(f"{key} must be a positive integer byte count")
        return value

    cgroup_limit = _read_cgroup_memory_limit(cgroup_paths)
    if cgroup_limit is None:
        return _DEFAULT_AGENT_MEMORY_LIMIT_BYTES
    return min(
        _DEFAULT_AGENT_MEMORY_LIMIT_BYTES,
        max(1, int(cgroup_limit * _AGENT_MEMORY_LIMIT_FRACTION)),
    )


def apply_address_space_limit(limit_bytes: int) -> None:
    """Apply an inherited hard RLIMIT_AS to the current child process."""
    if limit_bytes <= 0:
        raise ValueError("address-space limit must be positive")
    current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
    target = int(limit_bytes)
    if current_hard != resource.RLIM_INFINITY:
        target = min(target, current_hard)
    if current_soft != resource.RLIM_INFINITY:
        target = min(target, current_soft)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (target, target))
    except (OSError, ValueError):
        # Darwin reports RLIMIT_AS but rejects lowering it below the process's
        # enormous reserved VM map ("current limit exceeds maximum limit").
        # Production grading is Linux; keep local authoring/tests usable.
        if sys.platform == "darwin":
            return
        raise


def shared_memory_exhausted(mem: dict[str, int] | None = None) -> bool:
    values = mem if mem is not None else meminfo_kib()
    total = values.get("MemTotal", 0)
    shmem = values.get("Shmem", 0)
    available = values.get("MemAvailable")
    return bool(
        total > 0
        and shmem > 0
        and available is not None
        and available < _MEMAVAILABLE_EXHAUSTED_MAX_KIB
        and shmem
        >= max(_SHMEM_EXHAUSTED_MIN_KIB, int(total * _SHMEM_EXHAUSTED_MIN_FRACTION))
    )


@dataclass(frozen=True)
class ResourceExhaustion:
    """Whether the agent exhausted disk or shared memory, sampled at one instant.

    This is a value rather than a pair of predicates because the sample has to
    be taken *before* cleanup runs. ``pre_grade_cleanup`` reaps the agent's
    ``/dev/shm`` entries and tears down the staging tree, which is precisely the
    evidence these checks read, so a post-cleanup sample reports a healthy box
    and the episode is voided as infrastructure instead of being kept as the
    agent's zero. Both grading paths have made that mistake; holding the verdict
    in a value makes the ordering explicit at the call site.
    """

    disk: bool
    shmem: bool
    meminfo_kib: dict[str, int]

    def agent_fault_kind(
        self, *, returncode: int | None = None, stderr: str | None = None
    ) -> str | None:
        """The agent-fault kind to charge, or None when resources were fine.

        ``returncode=None`` means the grader was still running when we gave up
        on it (a timeout), so there is no exit status to corroborate memory
        pressure and the meminfo verdict stands alone. With an exit status we
        additionally require a SIGKILL or an explicit pressure marker, since a
        grader can exit non-zero for reasons that have nothing to do with the
        box being full.
        """
        if self.disk:
            return "disk_exhausted"
        if not self.shmem:
            return None
        if returncode is None:
            return "shared_memory_exhausted"
        killed = returncode in {-signal.SIGKILL, 128 + signal.SIGKILL}
        if killed or stderr_shows_memory_pressure(stderr):
            return "shared_memory_exhausted"
        return None


def sample_resource_exhaustion(
    stderr: str | None = None, *, tmpdir: str | Path | None = None
) -> ResourceExhaustion:
    """Snapshot disk/shared-memory exhaustion. Call before any cleanup."""
    mem = meminfo_kib()
    return ResourceExhaustion(
        disk=filesystem_exhausted(tempfile.gettempdir() if tmpdir is None else tmpdir)
        or stderr_shows_disk_full(stderr),
        shmem=shared_memory_exhausted(mem),
        meminfo_kib=mem,
    )


def _agent_uid() -> int | None:
    raw = os.environ.get("RUBRIC_AGENT_UID")
    if raw:
        try:
            uid = int(raw)
        except ValueError:
            return None
        return uid if uid > 0 else None
    try:
        return pwd.getpwnam(os.environ.get("RUBRIC_AGENT_USER", "agent")).pw_uid
    except KeyError:
        return None


def _remove_owned_path(path: Path, uid: int) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if info.st_uid != uid:
        return False
    try:
        os.rmdir(path) if stat.S_ISDIR(info.st_mode) else os.unlink(path)
        return True
    except OSError:
        return False


def cleanup_agent_tmpfs(
    agent_uid: int | None = None,
    *,
    roots: tuple[Path, ...] = _AGENT_TMPFS_ROOTS,
    max_entries: int = _AGENT_TMPFS_MAX_ENTRIES,
    max_seconds: float = _AGENT_TMPFS_MAX_WALK_S,
) -> tuple[int, bool]:
    """Remove agent-owned tmpfs entries without following links or walking forever."""
    uid = _agent_uid() if agent_uid is None else agent_uid
    if uid is None:
        return 0, False
    removed = 0
    visited = 0
    deadline = time.monotonic() + max_seconds if max_seconds > 0 else None

    def over_budget() -> bool:
        return (max_entries > 0 and visited >= max_entries) or (
            deadline is not None and time.monotonic() >= deadline
        )

    for root in roots:
        try:
            if not stat.S_ISDIR(os.lstat(root).st_mode):
                continue
        except OSError:
            continue
        stack = [root]
        directories: list[Path] = []
        while stack:
            if over_budget():
                return removed, True
            current = stack.pop()
            try:
                scanner = os.scandir(current)
            except OSError:
                continue
            with scanner:
                while True:
                    if over_budget():
                        return removed, True
                    try:
                        entry = next(scanner)
                    except StopIteration:
                        break
                    except OSError:
                        break
                    visited += 1
                    path = Path(entry.path)
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        is_dir = False
                    if is_dir:
                        stack.append(path)
                        directories.append(path)
                    elif _remove_owned_path(path, uid):
                        removed += 1
        for directory in reversed(directories):
            if over_budget():
                return removed, True
            if _remove_owned_path(directory, uid):
                removed += 1
    return removed, False


def protect_current_process_from_oom(path: str = "/proc/self/oom_score_adj") -> None:
    try:
        with open(path, "w") as handle:
            handle.write("-1000")
    except OSError:
        pass


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
                "[GRADING] could not prepare private grader cache %s: %s; omitting grader cache env vars",
                root,
                exc,
            )
            return isolated_grader_environ(cache_root=None)
    return isolated_grader_environ(cache_root=root)


def _read_proc_identity(proc_dir: str) -> tuple[str, int | None]:
    """Return ``(state, starttime)`` from one ``/proc/<pid>/stat`` entry."""
    try:
        with open(f"{proc_dir}/stat", "rb") as handle:
            stat_line = handle.read()
    except OSError:
        return "", None
    rparen = stat_line.rfind(b")")
    if rparen == -1 or rparen + 2 >= len(stat_line):
        return "", None
    fields = stat_line[rparen + 2 :].split()
    if not fields:
        return "", None
    state = fields[0].decode("ascii", errors="replace")
    # The suffix begins at proc field 3 (state); starttime is field 22.
    try:
        starttime = int(fields[19])
    except (IndexError, ValueError):
        starttime = None
    return state, starttime


def _read_proc_state(proc_dir: str) -> str:
    return _read_proc_identity(proc_dir)[0]


def kill_pre_grade_agent_processes(
    *,
    min_uid: int = _NON_SYSTEM_UID_THRESHOLD,
    max_passes: int = _QUIESCE_MAX_PASSES,
    required_zero_passes: int = _QUIESCE_CONSECUTIVE_ZEROS,
    inter_pass_delay_s: float = 0.05,
) -> int:
    """SIGKILL uid>=1000 processes before root-side grading reads outputs.

    Raises :class:`ProcessQuiesceError` when runtime inspection/termination
    fails, or :class:`AgentProcessQuiesceError` when confirmed agent processes
    keep respawning. Grading never proceeds through either state.
    """
    self_pid = os.getpid()
    parent_pid = os.getppid()
    protected = {self_pid, parent_pid, 1}
    total_killed = 0
    consecutive_zeros = 0
    killed_identities: set[tuple[int, int | None]] = set()
    saw_replacement = False

    if inter_pass_delay_s < 0.0:
        raise ValueError("inter_pass_delay_s must be non-negative")

    for pass_idx in range(max_passes):
        try:
            proc_entries = os.listdir("/proc")
        except OSError as exc:
            raise ProcessQuiesceError(
                f"could not enumerate /proc during pre-grade quiesce: {exc}"
            ) from exc

        eligible: list[tuple[int, int, str, tuple[int, int | None]]] = []
        for entry in proc_entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid in protected:
                continue
            proc_path = f"/proc/{entry}"
            try:
                uid = os.stat(proc_path).st_uid
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ProcessQuiesceError(
                    f"could not inspect {proc_path} during pre-grade quiesce: {exc}"
                ) from exc
            state, starttime = _read_proc_identity(proc_path)
            if uid < min_uid or state == "Z":
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
            eligible.append((pid, uid, cmd, (pid, starttime)))

        if not eligible:
            consecutive_zeros += 1
            if consecutive_zeros >= required_zero_passes:
                return total_killed
            continue

        consecutive_zeros = 0
        previously_killed = set(killed_identities)
        for pid, uid, cmd, identity in eligible:
            if identity in killed_identities:
                # SIGKILL is asynchronous. Give the same process identity time
                # to leave kernel teardown instead of counting it repeatedly as
                # a respawner.
                continue
            if previously_killed:
                saw_replacement = True
            try:
                os.kill(pid, signal.SIGKILL)
                total_killed += 1
                killed_identities.add(identity)
                logger.warning(
                    "[GRADING] pre-grade SIGKILL pid=%d uid=%d cmd=%r", pid, uid, cmd
                )
            except ProcessLookupError:
                continue
            except (PermissionError, OSError) as exc:
                raise ProcessQuiesceError(
                    f"could not SIGKILL pid={pid} uid={uid} cmd={cmd!r}: {exc}"
                ) from exc
        if inter_pass_delay_s:
            time.sleep(inter_pass_delay_s)

    message = (
        f"pre-grade quiesce did not converge after {max_passes} passes; "
        f"killed {total_killed} process(es)"
    )
    if saw_replacement or not killed_identities:
        raise AgentProcessQuiesceError(message)
    raise ProcessQuiesceError(
        f"{message}; already-killed process identities remained in kernel teardown"
    )


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
            cmd = (
                raw.replace(b"\x00", b" ").strip().decode(errors="replace") or "<empty>"
            )
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


def ensure_agent_output_directory(output_dir: str | Path) -> bool:
    """Restore and root-pin one agent-writable output directory.

    Call only after process quiescence. A missing, symlink, or non-directory
    leaf is replaced with a root-owned mode-0777 directory so later root-side
    publication cannot be redirected and future agent runs can still write
    inside it. Returns whether the leaf had to be replaced.
    """

    output = Path(output_dir)
    if not output.name:
        raise ProcessQuiesceError(f"invalid agent output directory: {output}")
    try:
        parent_fd = open_directory_fd(output.parent)
    except OSError as exc:
        raise ProcessQuiesceError(
            f"could not pin output parent {output.parent}: {exc}"
        ) from exc

    directory_fd = -1
    restored = False
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        try:
            info = os.stat(
                output.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            os.mkdir(output.name, mode=0o777, dir_fd=parent_fd)
            restored = True
        except OSError as exc:
            try:
                os.close(parent_fd)
            finally:
                parent_fd = -1
            raise ProcessQuiesceError(
                f"could not inspect output directory {output}: {exc}"
            ) from exc
        else:
            if not stat.S_ISDIR(info.st_mode):
                try:
                    os.unlink(output.name, dir_fd=parent_fd)
                    os.mkdir(output.name, mode=0o777, dir_fd=parent_fd)
                except OSError as exc:
                    raise ProcessQuiesceError(
                        f"could not replace unsafe output leaf {output}: {exc}"
                    ) from exc
                restored = True

        try:
            expected = os.stat(
                output.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(expected.st_mode):
                raise ProcessQuiesceError(
                    f"output leaf is not a directory after restoration: {output}"
                )
            directory_fd = os.open(output.name, flags, dir_fd=parent_fd)
            opened = os.fstat(directory_fd)
        except ProcessQuiesceError:
            raise
        except OSError as exc:
            raise ProcessQuiesceError(
                f"could not securely open output directory {output}: {exc}"
            ) from exc
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise ProcessQuiesceError(
                f"output directory changed while being opened: {output}"
            )

        if os.geteuid() == 0:
            os.fchown(directory_fd, 0, 0)
        os.fchmod(directory_fd, 0o777)
        return restored
    except OSError as exc:
        raise ProcessQuiesceError(
            f"could not restore output directory {output}: {exc}"
        ) from exc
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def pre_grade_cleanup(output_dir: str | Path) -> dict[str, int]:
    """Quiesce agent processes, then scrub dangerous output entries."""
    killed = 0
    if os.name == "posix" and os.geteuid() == 0:
        if not Path("/proc").is_dir():
            raise ProcessQuiesceError(
                "cannot prove pre-grade process quiescence because /proc is unavailable"
            )
        killed = kill_pre_grade_agent_processes()
    removed_tmpfs, tmpfs_flood = cleanup_agent_tmpfs()
    if tmpfs_flood:
        raise AgentTmpfsFloodError(
            "agent flooded /dev/shm beyond the bounded cleanup budget"
        )
    symlinks = scrub_escaping_symlinks(output_dir)
    restored = ensure_agent_output_directory(output_dir)
    nonregular = scrub_nonregular_files(output_dir)
    return {
        "killed_processes": killed,
        "removed_tmpfs_entries": removed_tmpfs,
        "restored_output_directory": int(restored),
        "removed_symlinks": symlinks,
        "removed_nonregular": nonregular,
    }


def _is_readonly_mount(path: Path) -> bool:
    try:
        return bool(os.statvfs(path).f_flag & os.ST_RDONLY)
    except OSError:
        return False


def lock_down_grader_private(
    paths: tuple[str | Path, ...],
    *,
    missing_ok: bool = False,
    readonly_mount_ok: tuple[str | Path, ...] = (),
) -> None:
    """Enforce root ownership and no group/other bits for private trees.

    ``readonly_mount_ok`` names roots that may legitimately arrive on a
    read-only mount. Two cases hit this:

    * grader CODE -- the harness mounts the host ``scorer/`` over
      ``/mcp_server/grader:ro`` so an edit rescores without an image rebuild.
    * held-out truth (``/mcp_server/data``) delivered as a Taiga
      ``is_read_only`` squashfs mount. On firecracker (the CPU-QA lane) that
      mount is genuinely read-only, so the ownership-reset chown fails with
      EROFS; on gVisor the writable overlay lets the same chown succeed.

    In both cases the read-only mount already delivers the tamper-proofing this
    lockdown exists for: the bytes cannot be modified, and the baked
    ``0700 root /mcp_server`` parent stops the uid-1000 agent from reading in.
    So a tolerated root degrades to a ``[SETUP_GUARD]`` warning, **but only when
    the filesystem really is read-only** (``_is_readonly_mount``). A writable,
    non-root private tree is never tolerated and still fails hard.
    """
    require_root_owner = os.geteuid() == 0
    tolerated_roots = {str(Path(raw)) for raw in readonly_mount_ok}
    targets: list[tuple[Path, bool]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() and not path.is_symlink():
            if missing_ok:
                continue
            raise InfrastructureFault(f"grader-private path is missing: {path}")
        readonly_ok = str(path) in tolerated_roots and _is_readonly_mount(path)
        targets.append((path, readonly_ok))
        if path.is_dir() and not path.is_symlink():

            def fail_walk(exc: OSError) -> None:
                raise InfrastructureFault(
                    f"could not traverse grader-private path {path}: {exc}"
                ) from exc

            for dirpath, dirnames, filenames in os.walk(path, onerror=fail_walk):
                targets.extend(
                    (Path(dirpath) / name, readonly_ok)
                    for name in (*dirnames, *filenames)
                )

    for path, readonly_ok in targets:
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise InfrastructureFault(
                f"could not inspect grader-private path {path}: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise InfrastructureFault(f"grader-private tree contains symlink: {path}")
        if require_root_owner and (info.st_uid != 0 or info.st_gid != 0):
            try:
                os.chown(path, 0, 0)
                logger.warning("[SETUP_GUARD] reset root ownership on %s", path)
            except OSError as exc:
                if not (readonly_ok and exc.errno in {errno.EROFS, errno.EPERM}):
                    raise InfrastructureFault(
                        f"could not reset root ownership on {path}: {exc}"
                    ) from exc
                logger.warning(
                    "[SETUP_GUARD] leaving %s owned by uid=%d gid=%d: "
                    "read-only grader-private mount (%s)",
                    path,
                    info.st_uid,
                    info.st_gid,
                    exc,
                )
        if info.st_mode & _AGENT_PERM_BITS:
            try:
                os.chmod(path, info.st_mode & ~_AGENT_PERM_BITS)
                logger.warning("[SETUP_GUARD] tightened group/other perms on %s", path)
            except OSError as exc:
                if not (readonly_ok and exc.errno in {errno.EROFS, errno.EPERM}):
                    raise InfrastructureFault(
                        f"could not tighten grader-private permissions on {path}: {exc}"
                    ) from exc
                logger.warning(
                    "[SETUP_GUARD] leaving mode %o on %s: "
                    "read-only grader-private mount (%s)",
                    stat.S_IMODE(info.st_mode),
                    path,
                    exc,
                )
        try:
            verified = os.lstat(path)
        except OSError as exc:
            raise InfrastructureFault(
                f"could not verify grader-private path {path}: {exc}"
            ) from exc
        if (
            require_root_owner and (verified.st_uid != 0 or verified.st_gid != 0)
        ) or verified.st_mode & _AGENT_PERM_BITS:
            if not readonly_ok:
                raise InfrastructureFault(
                    f"grader-private path remains accessible after lockdown: {path}"
                )


def lock_down_public_readonly(path: str | Path = "/data") -> int:
    """Make regular public inputs immutable while keeping them agent-readable.

    Small QA-visible inputs arrive as regular Taiga preloaded files because
    read-only mounts require squashfs images. The root setup process restores
    the original `/data` contract (0555 directories, 0444 files) before the
    agent starts. EROFS/EPERM is expected for an already read-only squashfs.
    """
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        return 0
    targets = [root]
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        targets.extend(Path(dirpath) / name for name in (*dirnames, *filenames))

    changed = 0
    for target in targets:
        try:
            info = os.lstat(target)
        except OSError as exc:
            logger.warning(
                "[SETUP_GUARD] could not stat public path %s: %s", target, exc
            )
            continue
        if stat.S_ISLNK(info.st_mode):
            continue
        mode = 0o555 if stat.S_ISDIR(info.st_mode) else 0o444
        if os.geteuid() == 0 and (info.st_uid != 0 or info.st_gid != 0):
            try:
                os.chown(target, 0, 0)
            except OSError as exc:
                if exc.errno not in {errno.EROFS, errno.EPERM}:
                    logger.warning(
                        "[SETUP_GUARD] could not reset public ownership on %s: %s",
                        target,
                        exc,
                    )
        if stat.S_IMODE(info.st_mode) == mode:
            continue
        try:
            os.chmod(target, mode)
            changed += 1
        except OSError as exc:
            if exc.errno not in {errno.EROFS, errno.EPERM}:
                logger.warning(
                    "[SETUP_GUARD] could not make public path read-only %s: %s",
                    target,
                    exc,
                )
    return changed
