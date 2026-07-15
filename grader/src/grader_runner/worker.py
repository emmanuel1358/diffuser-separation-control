"""Isolated child process for Harbor-side grading."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import inspect
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

from grading import Grade, normalize_compute_score_return
from grading.faults import AgentFault
from grading.rubric_builder import RubricBuilder


def _enter_pid_namespace() -> None:
    """Best-effort PID namespace isolation for root-side grading."""
    if (
        os.name != "posix"
        or os.geteuid() != 0
        or os.environ.get("GRADING_DISABLE_PID_NS_ISOLATION") == "1"
    ):
        return

    newns = 0x00020000
    newpid = 0x20000000
    ms_rec = 0x4000
    ms_private = 1 << 18
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.unshare.argtypes = [ctypes.c_int]
    libc.mount.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]

    def check(ret: int, what: str) -> None:
        if ret != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"{what}: {os.strerror(err)}")

    try:
        check(libc.unshare(newpid), "unshare(CLONE_NEWPID)")
    except OSError as exc:
        print(
            "WARNING: CLONE_NEWPID unavailable (%s); grading without "
            "PID-namespace isolation." % exc,
            file=sys.stderr,
        )
        return

    try:
        kid = os.fork()
    except OSError as exc:
        print(f"RUBRIC_ERROR: fork into PID namespace failed ({exc})", file=sys.stderr)
        sys.exit(1)

    if kid != 0:
        _pid, status = os.waitpid(kid, 0)
        if os.WIFSIGNALED(status):
            sig = os.WTERMSIG(status)
            try:
                signal.signal(sig, signal.SIG_DFL)
                os.kill(os.getpid(), sig)
            except Exception:
                pass
            os._exit(128 + sig)
        os._exit(os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1)

    try:
        if libc.prctl(1, 9) != 0:  # PR_SET_PDEATHSIG, SIGKILL
            print(
                "WARNING: PR_SET_PDEATHSIG failed; grading child may outlive parent timeout.",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"WARNING: PR_SET_PDEATHSIG could not be set ({exc}).", file=sys.stderr)

    try:
        check(libc.unshare(newns), "unshare(CLONE_NEWNS)")
        check(
            libc.mount(b"none", b"/", None, ms_rec | ms_private, None), "make-rprivate"
        )
        check(libc.mount(b"proc", b"/proc", b"proc", 0, None), "mount /proc")
    except OSError as exc:
        print(
            "WARNING: could not remount /proc in PID namespace (%s); "
            "/proc survivor channels may remain." % exc,
            file=sys.stderr,
        )


def _harden_import_path(workspace: Path) -> None:
    try:
        ws = str(workspace.resolve())
    except OSError:
        ws = None
    cleaned: list[str] = []
    for entry in sys.path:
        if entry in ("", "."):
            continue
        try:
            resolved = str(Path(entry).resolve())
        except OSError:
            resolved = entry
        if ws is not None and (resolved == ws or resolved.startswith(ws + "/")):
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def _resolve_grader_path(grader_dir: Path) -> Path:
    """Locate the grader source (always baked as ``compute_score.py``) in ``grader_dir``."""
    candidate = grader_dir / "compute_score.py"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"no grader source (compute_score.py) found in {grader_dir}"
    )


def _prepend_grading_deps() -> None:
    """Put the root-only grader-only site-packages (``/mcp_server/grading_deps``)
    on ``sys.path`` so ``compute_score`` can import a task's ``grading_dependencies``.

    This augments ONLY this (root) grader worker's path -- never the uid-1000
    submission workers (fresh-exec'd with PYTHONPATH scrubbed + PYTHONSAFEPATH),
    and the 0700 ``/mcp_server`` blocks the agent at the filesystem regardless, so
    a grader-only dep is exactly as isolated as the held-out truth. Idempotent;
    a no-op when the dir is absent (every task without grading_dependencies, and
    any unprivileged context that cannot traverse ``/mcp_server``). Called before
    ``exec_module`` so a top-level ``import <grading_dep>`` in the grader resolves.
    """
    from env_server.config import GRADING_DEPS_DIR

    deps = str(GRADING_DEPS_DIR)
    if deps not in sys.path and GRADING_DEPS_DIR.is_dir():
        sys.path.insert(0, deps)


def _import_grader(grader_path: Path):
    if not grader_path.exists():
        raise FileNotFoundError(f"grader source not found at {grader_path}")
    spec = importlib.util.spec_from_file_location("task_compute_score", grader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {grader_path}")
    module = importlib.util.module_from_spec(spec)
    grader_dir = str(grader_path.parent)
    if grader_dir not in sys.path:
        sys.path.insert(0, grader_dir)
    # Grader-only deps must be importable BEFORE the grader's top-level imports run.
    _prepend_grading_deps()
    spec.loader.exec_module(module)
    return module


def _read_transcript(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _call_compute_score(
    fn,
    *,
    workspace: Path,
    trajectory: list[dict] | None,
    private: Path,
    transcript: str,
) -> object:
    sig = inspect.signature(fn)
    # A no-arg ``compute_score()`` reads the fixed baked paths (/tmp/output,
    # /mcp_server/data) directly; call it with no arguments.
    if not sig.parameters:
        return fn()
    kwargs: dict[str, object] = {}
    if "workspace" in sig.parameters:
        kwargs["workspace"] = workspace
    if "trajectory" in sig.parameters:
        kwargs["trajectory"] = trajectory
    if "private" in sig.parameters:
        kwargs["private"] = private
    if "transcript" in sig.parameters:
        kwargs["transcript"] = transcript
    if not kwargs:
        return fn(workspace, trajectory, private)
    return fn(**kwargs)


def _wire_transcript_into_rubric(value: object, transcript: str) -> object:
    if isinstance(value, RubricBuilder):
        value.transcript = transcript
        return value.grade()
    return value


def _agent_fault_grade(message: str) -> Grade:
    return Grade(
        subscores={"agent_fault": 0.0},
        weights={"agent_fault": 1.0},
        headline_score_override=0.0,
        metadata={"return_shape": "agent_fault", "agent_fault": message},
        criterion_logs={
            "agent_fault": {
                "grading_type": "agent_fault",
                "error_type": "agent_fault",
                "error_message": message,
                "passed": False,
                "reasoning": message,
            }
        },
        env_internal_failure=False,
    )


def _failure_grade(error_type: str, message: str, tb: str = "") -> Grade:
    metadata: dict[str, Any] = {"return_shape": "error"}
    if tb:
        metadata["traceback"] = tb
    return Grade(
        subscores={"run_grader": 0.0},
        weights={"run_grader": 1.0},
        headline_score_override=0.0,
        metadata=metadata,
        criterion_logs={
            "run_grader": {
                "grading_type": "runner",
                "error_type": error_type,
                "error_message": message,
                "passed": False,
                "reasoning": message,
            }
        },
        env_internal_failure=True,
        env_internal_failure_logs=[message],
    )


def _write_result(result_path: Path, payload: dict[str, Any]) -> None:
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run compute_score in an isolated worker."
    )
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--grader-dir", required=True, type=Path)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--result-path", required=True, type=Path)
    parser.add_argument("--transcript", type=Path, default=None)
    args = parser.parse_args(argv)

    _enter_pid_namespace()
    _harden_import_path(args.workspace)

    try:
        grader_path = _resolve_grader_path(args.grader_dir)
        module = _import_grader(grader_path)
    except Exception as exc:
        _write_result(
            args.result_path,
            _failure_grade(
                "grader_import_failed",
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            ).to_dict(),
        )
        return 1

    fn = getattr(module, "compute_score", None)
    if not callable(fn):
        _write_result(
            args.result_path,
            _failure_grade(
                "missing_compute_score",
                f"{grader_path} does not define a callable compute_score().",
            ).to_dict(),
        )
        return 1

    transcript = _read_transcript(args.transcript)
    try:
        raw = _call_compute_score(
            fn,
            workspace=args.workspace,
            trajectory=None,
            private=args.private_dir,
            transcript=transcript,
        )
        raw = _wire_transcript_into_rubric(raw, transcript)
    except AgentFault as exc:
        _write_result(args.result_path, _agent_fault_grade(str(exc)).to_dict())
        return 0
    except Exception:
        _write_result(
            args.result_path,
            _failure_grade(
                "grader_runtime_error",
                traceback.format_exc().splitlines()[-1],
                traceback.format_exc(),
            ).to_dict(),
        )
        return 1

    try:
        grade = normalize_compute_score_return(raw)
    except Exception as exc:
        _write_result(
            args.result_path,
            _failure_grade(
                "grader_return_unsupported",
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            ).to_dict(),
        )
        return 1

    _write_result(args.result_path, grade.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
