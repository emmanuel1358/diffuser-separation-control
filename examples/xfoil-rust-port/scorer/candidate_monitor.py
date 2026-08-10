"""Root-owned ptrace monitor for one hidden candidate execution.

The monitor allows the initial candidate exec and CLONE_THREAD descendants.
It stops the traced process tree before any fork/vfork/process-clone child or
subsequent exec can execute user code. Candidate stdout is captured separately
and returned inside one trusted JSON envelope, so it cannot forge policy state.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import json
import os
import platform
import pwd
import signal
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = "xfoil-ptrace-monitor/v1"
MAX_STDOUT_BYTES = 32 * 1024 * 1024
MAX_STDERR_BYTES = 1024 * 1024
SUPPORTED_MACHINE = "x86_64"
ROSETTA_BOOTSTRAP = "/tmp/rstub"

PTRACE_TRACEME = 0
PTRACE_CONT = 7
PTRACE_SETOPTIONS = 0x4200
PTRACE_GETEVENTMSG = 0x4201

PTRACE_O_TRACEFORK = 1 << 1
PTRACE_O_TRACEVFORK = 1 << 2
PTRACE_O_TRACECLONE = 1 << 3
PTRACE_O_TRACEEXEC = 1 << 4
PTRACE_O_EXITKILL = 1 << 20

PTRACE_EVENT_FORK = 1
PTRACE_EVENT_VFORK = 2
PTRACE_EVENT_CLONE = 3
PTRACE_EVENT_EXEC = 4

WAIT_ALL = 0x40000000
TRACE_OPTIONS = (
    PTRACE_O_TRACEFORK
    | PTRACE_O_TRACEVFORK
    | PTRACE_O_TRACECLONE
    | PTRACE_O_TRACEEXEC
    | PTRACE_O_EXITKILL
)

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.ptrace.argtypes = [
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_LIBC.ptrace.restype = ctypes.c_long


class MonitorError(RuntimeError):
    """The trusted monitor could not establish or preserve its boundary."""


def _ptrace(request: int, pid: int, data: int = 0) -> int:
    ctypes.set_errno(0)
    result = _LIBC.ptrace(
        request,
        pid,
        ctypes.c_void_p(),
        ctypes.c_void_p(data),
    )
    error = ctypes.get_errno()
    if result == -1 and error:
        raise OSError(error, os.strerror(error))
    return int(result)


def _event_pid(pid: int) -> int:
    value = ctypes.c_ulong()
    ctypes.set_errno(0)
    result = _LIBC.ptrace(
        PTRACE_GETEVENTMSG,
        pid,
        ctypes.c_void_p(),
        ctypes.cast(ctypes.byref(value), ctypes.c_void_p),
    )
    error = ctypes.get_errno()
    if result == -1 and error:
        raise OSError(error, os.strerror(error))
    return int(value.value)


def _task_group_id(pid: int) -> int:
    status_path = Path(f"/proc/{pid}/status")
    for _ in range(50):
        try:
            for line in status_path.read_text().splitlines():
                if line.startswith("Tgid:"):
                    return int(line.split(":", 1)[1].strip())
        except FileNotFoundError:
            pass
        time.sleep(0.001)
    raise MonitorError(f"could not resolve traced task group for pid {pid}")


def _requests_payload(path: Path) -> bytes:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value:
        raise MonitorError("hidden request file is not a non-empty JSON array")
    return (
        "\n".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value
        )
        + "\n"
    ).encode("utf-8")


def _input_fd(payload: bytes) -> int:
    if not hasattr(os, "memfd_create"):
        raise MonitorError("memfd_create is required for private candidate stdin")
    fd = os.memfd_create("xfoil-hidden-input", os.MFD_CLOEXEC)
    offset = 0
    while offset < len(payload):
        offset += os.write(fd, payload[offset:])
    os.lseek(fd, 0, os.SEEK_SET)
    return fd


def _binary_fd(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(fd)
        raise MonitorError("candidate executable is not a regular file")
    if metadata.st_uid != 0 or metadata.st_mode & 0o022:
        os.close(fd)
        raise MonitorError("candidate executable is not sealed by root")
    if not metadata.st_mode & 0o111:
        os.close(fd)
        raise MonitorError("candidate executable has no execute bit")
    return fd


def _kill_tracees(leader: int, tracees: set[int]) -> None:
    try:
        os.killpg(leader, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid in tuple(tracees):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _reap_tracees(tracees: set[int]) -> None:
    deadline = time.monotonic() + 5.0
    while tracees and time.monotonic() < deadline:
        try:
            pid, _status = os.waitpid(-1, WAIT_ALL | os.WNOHANG)
        except ChildProcessError:
            return
        if pid > 0:
            tracees.discard(pid)
        else:
            time.sleep(0.002)


def _child(
    *,
    binary: Path,
    error_fd: int,
    input_fd: int,
    output_fd: int,
    stderr_fd: int,
    cwd: Path,
    uid: int,
    gid: int,
) -> None:
    try:
        os.setsid()
        os.chdir(cwd)
        os.dup2(input_fd, sys.stdin.fileno())
        os.dup2(output_fd, sys.stdout.fileno())
        os.dup2(stderr_fd, sys.stderr.fileno())
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
        if os.getuid() != 1000 or os.geteuid() != 1000:
            os._exit(125)
        _ptrace(PTRACE_TRACEME, 0)
        os.kill(os.getpid(), signal.SIGSTOP)
        argv = [str(binary)]
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "LC_ALL": "C",
        }
        os.execve(binary, argv, env)
    except BaseException as exc:
        try:
            detail = f"{type(exc).__name__}: {exc}".encode("utf-8", errors="replace")[
                :4096
            ]
            os.write(error_fd, detail)
        except BaseException:
            pass
        os._exit(126)


def _trace_candidate(
    *,
    binary: Path,
    cwd: Path,
    requests: Path,
    timeout_s: float,
) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise MonitorError("candidate monitor must run as root")
    machine = platform.machine().lower()
    if machine != SUPPORTED_MACHINE:
        raise MonitorError(
            f"candidate monitor requires {SUPPORTED_MACHINE}, received {machine}"
        )
    account = pwd.getpwnam("agent")
    if account.pw_uid != 1000 or account.pw_gid != 1000:
        raise MonitorError("the agent account must be uid/gid 1000")

    cwd = cwd.resolve(strict=True)
    binary = binary.resolve(strict=True)
    binary_parent = binary.parent.stat()
    if binary_parent.st_uid != 0 or binary_parent.st_mode & 0o022:
        raise MonitorError("candidate executable directory is not sealed by root")
    binary_fd = _binary_fd(binary)
    os.close(binary_fd)
    input_fd = _input_fd(_requests_payload(requests))
    error_read, error_write = os.pipe()
    output_read, output_write = os.pipe()
    stderr_read, stderr_write = os.pipe()
    os.set_inheritable(error_read, False)
    os.set_inheritable(error_write, False)
    os.set_inheritable(output_read, False)
    os.set_inheritable(output_write, False)
    os.set_inheritable(stderr_read, False)
    os.set_inheritable(stderr_write, False)

    stdout = bytearray()
    stderr = bytearray()
    output_exceeded = threading.Event()

    def read_stdout() -> None:
        while True:
            chunk = os.read(output_read, 64 * 1024)
            if not chunk:
                return
            remaining = MAX_STDOUT_BYTES + 1 - len(stdout)
            if remaining > 0:
                stdout.extend(chunk[:remaining])
            if len(stdout) > MAX_STDOUT_BYTES:
                output_exceeded.set()

    def read_stderr() -> None:
        while len(stderr) < MAX_STDERR_BYTES:
            chunk = os.read(stderr_read, min(64 * 1024, MAX_STDERR_BYTES - len(stderr)))
            if not chunk:
                return
            stderr.extend(chunk)

    leader = os.fork()
    if leader == 0:
        os.close(output_read)
        _child(
            binary=binary,
            error_fd=error_write,
            input_fd=input_fd,
            output_fd=output_write,
            stderr_fd=stderr_write,
            cwd=cwd,
            uid=account.pw_uid,
            gid=account.pw_gid,
        )
        os._exit(126)

    os.close(error_write)
    os.close(input_fd)
    os.close(output_write)
    os.close(stderr_write)
    reader = threading.Thread(target=read_stdout, name="candidate-stdout", daemon=True)
    stderr_reader = threading.Thread(
        target=read_stderr,
        name="candidate-stderr",
        daemon=True,
    )
    reader.start()
    stderr_reader.start()

    tracees = {leader}
    exec_events = 0
    emulator_exec_events = 0
    exec_paths: list[str] = []
    thread_clones = 0
    policy_violation: str | None = None
    leader_returncode: int | None = None
    deadline = time.monotonic() + timeout_s
    timed_out = False

    try:
        stopped_pid, status = os.waitpid(leader, 0)
        if stopped_pid != leader or not os.WIFSTOPPED(status):
            raise MonitorError("candidate did not enter its initial ptrace stop")
        _ptrace(PTRACE_SETOPTIONS, leader, TRACE_OPTIONS)
        _ptrace(PTRACE_CONT, leader)

        while tracees:
            if output_exceeded.is_set():
                policy_violation = "candidate stdout exceeded the trusted limit"
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            try:
                pid, status = os.waitpid(-1, WAIT_ALL | os.WNOHANG)
            except ChildProcessError:
                tracees.clear()
                break
            if pid == 0:
                time.sleep(0.002)
                continue
            if os.WIFEXITED(status):
                tracees.discard(pid)
                if pid == leader:
                    leader_returncode = os.WEXITSTATUS(status)
                continue
            if os.WIFSIGNALED(status):
                tracees.discard(pid)
                if pid == leader:
                    leader_returncode = 128 + os.WTERMSIG(status)
                continue
            if not os.WIFSTOPPED(status):
                continue

            event = status >> 16
            stop_signal = os.WSTOPSIG(status)
            if event == PTRACE_EVENT_EXEC:
                try:
                    exec_path = os.readlink(f"/proc/{pid}/exe")
                except OSError:
                    exec_path = "<unavailable>"
                exec_paths.append(exec_path)
                if (
                    pid == leader
                    and exec_events == 0
                    and emulator_exec_events == 0
                    and exec_path == ROSETTA_BOOTSTRAP
                ):
                    emulator_exec_events += 1
                elif pid == leader and exec_path == str(binary):
                    exec_events += 1
                    if exec_events != 1:
                        policy_violation = "candidate attempted a subsequent exec"
                        break
                else:
                    policy_violation = "candidate attempted a subsequent exec"
                    break
            elif event in (PTRACE_EVENT_FORK, PTRACE_EVENT_VFORK):
                child_pid = _event_pid(pid)
                tracees.add(child_pid)
                operation = "fork" if event == PTRACE_EVENT_FORK else "vfork"
                policy_violation = f"candidate attempted {operation}"
                break
            elif event == PTRACE_EVENT_CLONE:
                child_pid = _event_pid(pid)
                tracees.add(child_pid)
                if _task_group_id(child_pid) != leader:
                    policy_violation = "candidate attempted a process clone"
                    break
                thread_clones += 1

            forwarded_signal = 0
            if not event and stop_signal not in (signal.SIGSTOP, signal.SIGTRAP):
                forwarded_signal = stop_signal
            try:
                _ptrace(PTRACE_CONT, pid, forwarded_signal)
            except OSError as exc:
                if exc.errno != errno.ESRCH:
                    raise
    finally:
        if policy_violation is not None or timed_out or tracees:
            _kill_tracees(leader, tracees)
            _reap_tracees(tracees)
        reader.join(timeout=5.0)
        if reader.is_alive():
            os.close(output_read)
            reader.join(timeout=1.0)
        else:
            os.close(output_read)
        stderr_reader.join(timeout=5.0)
        if stderr_reader.is_alive():
            os.close(stderr_read)
            stderr_reader.join(timeout=1.0)
        else:
            os.close(stderr_read)

    if reader.is_alive() or stderr_reader.is_alive():
        raise MonitorError("candidate output reader did not terminate")
    setup_error = os.read(error_read, 4096).decode("utf-8", errors="replace")
    os.close(error_read)
    status_name = "ok"
    if policy_violation is not None:
        status_name = "policy_violation"
    elif timed_out:
        status_name = "candidate_timeout"
    elif leader_returncode is None:
        raise MonitorError("candidate leader produced no exit status")
    elif leader_returncode != 0:
        status_name = "candidate_failure"

    return {
        "schema": SCHEMA,
        "status": status_name,
        "returncode": leader_returncode,
        "timed_out": timed_out,
        "violation": policy_violation,
        "exec_events": exec_events,
        "emulator_exec_events": emulator_exec_events,
        "exec_paths": exec_paths,
        "thread_clones": thread_clones,
        "setup_error": setup_error or None,
        "stderr_b64": base64.b64encode(bytes(stderr)).decode("ascii"),
        "stdout_b64": base64.b64encode(bytes(stdout)).decode("ascii"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _trace_candidate(
            binary=args.binary,
            cwd=args.cwd,
            requests=args.requests,
            timeout_s=args.timeout,
        )
    except BaseException as exc:
        result = {
            "schema": SCHEMA,
            "status": "monitor_failure",
            "error": f"{type(exc).__name__}: {exc}",
        }
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
