"""Generic MCP runtime for Alignerr RL tasks.

This intentionally mirrors the small subset of the ML_Envs rubric server that
Boreal needs, without vendoring Anthropic-specific `taiga-core`.
"""

import dataclasses
import errno
import hashlib
import json
import math
import os
import pwd
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

from grading.faults import InfrastructureFault
from grading.runtime_hardening import (
    AgentProcessQuiesceError,
    ProcessQuiesceError,
    classify_failure,
    cleanup_agent_tmpfs,
    kill_nvproxy_fd_holders,
    lock_down_grader_private,
    lock_down_public_readonly,
    meminfo_kib,
    pre_grade_cleanup,
    prepare_grader_cache,
    sample_resource_exhaustion,
)
from grading.secure_io import open_directory_fd, open_regular_file
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from rubric.service_runtime import (
    ServiceRuntimeAgentError,
    TaskServiceRuntime,
)
from rubric.tool_runtime import ToolRequestError

mcp = FastMCP("alignerr-rl-tasks")

WORKDIR = Path("/workdir")
OUTPUT_DIR = Path("/tmp/output")

_RESULT_PATH_ENV = "RUBRIC_RESULT_PATH"
_EVALUATE_TIMEOUT_ENV = "RUBRIC_EVALUATE_TIMEOUT_S"
_DEFAULT_EVALUATE_TIMEOUT_S = 600.0
_RESULT_DIR_PREFIX = "lbx-rubric-result-"
_TRACE_DIR_PREFIX = "lbx-evaluation-trace-"
_TRACE_OUTPUT_DIRNAME = ".lbx-evaluation"
_TRACE_FILENAME = "evaluation-details.json"
_TRACE_PATH_ENV = "LBX_EVALUATION_TRACE_PATH"
_GRADING_TMPDIR = tempfile.gettempdir()
_TASK_TOML_ENV = "RUBRIC_TASK_TOML_PATH"
_SERVICE_SNAPSHOT_ENV = "LBX_SERVICE_ARTIFACT_SNAPSHOT"
_SERVICE_MANIFEST_ENV = "LBX_SERVICE_ARTIFACT_MANIFEST"
_SERVICE_RUNTIME_UNSET = object()
_SERVICE_RUNTIME: TaskServiceRuntime | None | object = _SERVICE_RUNTIME_UNSET
_SERVICE_RUNTIME_LOCK = threading.RLock()


def _task_service_runtime() -> TaskServiceRuntime | None:
    """Lazily load optional nested-service capabilities exactly once."""
    global _SERVICE_RUNTIME
    with _SERVICE_RUNTIME_LOCK:
        if _SERVICE_RUNTIME is _SERVICE_RUNTIME_UNSET:
            task_toml = Path(os.environ.get(_TASK_TOML_ENV, "/task/task.toml"))
            _SERVICE_RUNTIME = TaskServiceRuntime.from_task_toml(task_toml)
        return _SERVICE_RUNTIME  # type: ignore[return-value]


def _shutdown_task_service_runtime() -> None:
    """Best-effort nested-runtime teardown for normal MCP shutdown."""
    with _SERVICE_RUNTIME_LOCK:
        runtime = (
            _SERVICE_RUNTIME if _SERVICE_RUNTIME is not _SERVICE_RUNTIME_UNSET else None
        )
    if runtime is not None:
        try:
            runtime.cleanup()
        except Exception as exc:  # noqa: BLE001 - process shutdown must continue
            print(
                f"[SERVICE_RUNTIME] cleanup failed during shutdown: {exc}",
                file=sys.stderr,
                flush=True,
            )


@dataclasses.dataclass(kw_only=True)
class ToolResult:
    """Simple MCP tool result."""

    output: str | None = None
    error: str | None = None
    system: str | None = None


@dataclasses.dataclass(kw_only=True)
class Grade:
    """Grade returned from grade_problem."""

    subscores: dict[str, float]
    weights: dict[str, float]
    metadata: dict[str, Any] | None = None
    env_internal_failure: bool | None = None
    env_internal_failure_logs: list[str] | None = None


def _extra(extra_fields: Any) -> dict[str, Any]:
    if extra_fields is None:
        return {}
    if isinstance(extra_fields, dict):
        return extra_fields
    if hasattr(extra_fields, "model_dump"):
        return extra_fields.model_dump()
    if hasattr(extra_fields, "__dict__"):
        return dict(extra_fields.__dict__)
    return {}


def _verify_calibration(fields: dict[str, Any]) -> bool:
    evidence = fields.get("calibration")
    if not isinstance(evidence, dict):
        return False
    expected = evidence.get("lock_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError("calibration evidence is missing a full lock SHA-256")
    path = Path("/mcp_server/calibration/calibration.lock.json")
    if not path.is_file():
        raise RuntimeError(f"promoted calibration lock is missing at {path}")
    if (
        evidence.get("requires_trusted_mount")
        and (path.parent / ".author-source").exists()
    ):
        raise RuntimeError(
            "Taiga calibration requires the trusted-CI promoted mount; "
            "the container is still using its baked author fallback"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"promoted calibration lock digest mismatch: expected {expected}, got {actual}"
        )
    expected_plan = evidence.get("evaluation_plan_sha256")
    if not isinstance(expected_plan, str) or len(expected_plan) != 64:
        raise RuntimeError("calibration evidence is missing evaluation plan identity")
    try:
        lock_payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not parse promoted calibration lock: {exc}") from exc
    if lock_payload.get("evaluation_plan_sha256") != expected_plan:
        raise RuntimeError("promoted calibration evaluation plan digest mismatch")
    expected_tier = evidence.get("security_tier")
    actual_tier = (lock_payload.get("evaluation_plan") or {}).get("security_tier")
    if expected_tier and actual_tier != expected_tier:
        raise RuntimeError("promoted calibration security tier mismatch")
    return True


def _verify_evaluation_plan(fields: dict[str, Any]) -> bool:
    evidence = fields.get("evaluation_plan")
    if not isinstance(evidence, dict):
        return False
    expected = evidence.get("sha256")
    filename = evidence.get("path")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or not isinstance(filename, str)
        or Path(filename).name != filename
    ):
        raise RuntimeError("evaluation plan evidence is malformed")
    path = Path("/mcp_server/grader") / filename
    if not path.is_file():
        raise RuntimeError(f"sealed evaluation plan is missing at {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"sealed evaluation plan digest mismatch: expected {expected}, got {actual}"
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not parse sealed evaluation plan: {exc}") from exc
    try:
        from grading.evaluation.plan import validate_serialized_plan

        plan_sha = validate_serialized_plan(payload)
    except (ImportError, ValueError) as exc:
        raise RuntimeError(f"sealed evaluation plan is invalid: {exc}") from exc
    if plan_sha != evidence.get("plan_sha256"):
        raise RuntimeError("sealed evaluation plan identity mismatch")
    if payload.get("security_tier") != evidence.get("security_tier"):
        raise RuntimeError("sealed evaluation plan security tier mismatch")
    return True


def _verify_continuous_evaluation(fields: dict[str, Any]) -> bool:
    """Verify continuous or declarative-rubric evidence for this exact call."""
    os.environ.pop("LBX_EVALUATION_PLAN_ATTESTED", None)
    calibration_verified = _verify_calibration(fields)
    plan_verified = _verify_evaluation_plan(fields)
    verified = calibration_verified or plan_verified

    policies = [
        policy
        for policy in (
            fields.get("continuous_evaluation"),
            fields.get("rubric_evaluation"),
        )
        if isinstance(policy, dict)
    ]
    required = any(policy.get("required") is True for policy in policies)
    attestation_required = any(
        policy.get("attestation_required") is True for policy in policies
    )
    if (required or attestation_required) and not verified:
        raise RuntimeError(
            "protected evaluation requires trusted calibration or plan evidence"
        )
    if verified and attestation_required:
        os.environ["LBX_EVALUATION_PLAN_ATTESTED"] = "1"
    return verified


# Grader-private trees sealed (root:root, no group/other bits) before the agent
# starts. Held-out truth and calibration are delivered as read-only squashfs
# mounts on the CPU-QA lane, so this same set is passed as readonly_mount_ok in
# setup_problem -- see the note there.
_SETUP_PRIVATE_ROOTS = (
    "/mcp_server/data",
    "/mcp_server/calibration",
    "/mcp_server/grader",
    "/mcp_server/grading",
    "/mcp_server/src/rubric",
    "/runtime/grading",
)


@mcp.tool()
async def setup_problem(
    problem_id: str = Field(description="The id of the problem to solve"),
    extra_fields: dict | None = None,
    use_hinted_problem: bool = True,
) -> str:
    """Return the task prompt."""
    _ = use_hinted_problem
    # Every sealed root may legitimately arrive on a read-only mount: grader
    # CODE via the scorer:ro bind mount, and held-out truth (/mcp_server/data) /
    # calibration (/mcp_server/calibration) via Taiga is_read_only squashfs
    # mounts. On firecracker (the CPU-QA lane) those squashfs mounts are
    # genuinely read-only, so the ownership-reset chown fails with EROFS and
    # this setup tool would abort the whole episode; gVisor (deploy lane) masks
    # it because its writable overlay lets the chown succeed. A read-only mount
    # already delivers the tamper-proofing this lockdown exists for (the bytes
    # cannot be modified, and the baked 0700 root /mcp_server parent blocks the
    # uid-1000 agent from reading in), so the tolerated set is the SAME as the
    # sealed set -- keeping them identical prevents a sealed-but-not-tolerated
    # root from re-introducing the CPU-QA setup failure. lock_down_grader_private
    # still only relaxes when the filesystem really is read-only, so a writable
    # non-root private tree fails hard regardless.
    lock_down_grader_private(
        _SETUP_PRIVATE_ROOTS,
        missing_ok=True,
        readonly_mount_ok=_SETUP_PRIVATE_ROOTS,
    )
    lock_down_public_readonly("/data")
    lock_down_public_readonly("/lbx-public-files")
    fields = _extra(extra_fields)
    _verify_continuous_evaluation(fields)
    service_runtime = _task_service_runtime()
    if service_runtime is not None:
        # Service startup errors inherit InfrastructureFault.  Let the MCP
        # boundary fail the setup call instead of giving the agent a rollout
        # against a partially initialized stack.
        service_runtime.start()
    return str(
        fields.get("task_prompt")
        or fields.get("prompt")
        or f"Solve problem {problem_id}."
    )


_AGENT_USER = "agent"
_AGENT_USER_ENV = "RUBRIC_AGENT_USER"
_AGENT_UID_ENV = "RUBRIC_AGENT_UID"
_AGENT_GID_ENV = "RUBRIC_AGENT_GID"
_AGENT_HOME_ENV = "RUBRIC_AGENT_HOME"

# Secrets provisioned for the grading-side LLM judge (the Anthropic proxy
# credentials) must never reach agent-authored code — the bash/editor tools or
# a submitted policy. The grader parent keeps its own os.environ; only the env
# handed to a privilege-dropped child is scrubbed. (Mirrors the scrub in the
# grading library's policy_runner so both subprocess boundaries match.)
_SECRET_ENV_PREFIXES = ("ANTHROPIC_",)
_SECRET_ENV_SUBSTRINGS = (
    "API_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
)


def _scrubbed_environ() -> dict[str, str]:
    """Copy of os.environ with grading-side secrets removed."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_SECRET_ENV_PREFIXES)
        and not any(token in key.upper() for token in _SECRET_ENV_SUBSTRINGS)
    }


def _isolated_python_environ(*, scrub_secrets: bool) -> dict[str, str]:
    """Subprocess env without inherited Python import-path injection."""
    env = _scrubbed_environ() if scrub_secrets else dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONSAFEPATH"] = "1"
    return env


def _agent_name() -> str:
    return os.environ.get(_AGENT_USER_ENV) or _AGENT_USER


def _agent_id_from_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must identify a non-root account")
    return value


def _agent_identity() -> tuple[int, int, str, str]:
    """Resolve the unprivileged account required for root-run agent tools."""
    name = _agent_name()
    uid = _agent_id_from_env(_AGENT_UID_ENV)
    gid = _agent_id_from_env(_AGENT_GID_ENV)
    if (uid is None) != (gid is None):
        raise RuntimeError(
            f"{_AGENT_UID_ENV} and {_AGENT_GID_ENV} must be set together"
        )
    if uid is not None and gid is not None:
        try:
            account = pwd.getpwuid(uid)
            home = account.pw_dir
            login = account.pw_name
        except KeyError:
            home = f"/home/{name}"
            login = name
        home = os.environ.get(_AGENT_HOME_ENV) or home
        login = os.environ.get(_AGENT_USER_ENV) or login
        return uid, gid, home, login
    try:
        account = pwd.getpwnam(name)
    except KeyError as exc:
        raise RuntimeError(
            f"cannot drop privileges: user {name!r} not found; set "
            f"{_AGENT_USER_ENV} or {_AGENT_UID_ENV}/{_AGENT_GID_ENV}"
        ) from exc
    if account.pw_uid <= 0 or account.pw_gid <= 0:
        raise RuntimeError(f"cannot drop privileges to root account {name!r}")
    return account.pw_uid, account.pw_gid, account.pw_dir, account.pw_name


def _agent_subprocess_kwargs(*, isolate_python: bool = False) -> dict[str, Any]:
    """Drop agent-facing subprocesses to the unprivileged account when root.

    The rubric server runs as root so `grade_problem` can read the hidden
    /mcp_server/{data,grader} fixtures, but agent-facing tools (bash, the
    editor) must not inherit that privilege. Root execution fails closed if the
    configured unprivileged identity cannot be resolved. HOME is set explicitly
    because Popen(user=...) does not read /etc/passwd to populate it. The Python
    editor shim additionally gets import-path isolation.
    """
    env = (
        _isolated_python_environ(scrub_secrets=True)
        if isolate_python
        else _scrubbed_environ()
    )
    kwargs: dict[str, Any] = {"env": env}
    if os.geteuid() != 0:
        return kwargs
    uid, gid, home, name = _agent_identity()
    env["HOME"] = home
    env["USER"] = env["LOGNAME"] = name
    kwargs.update(user=uid, group=gid, extra_groups=[])
    return kwargs


_AGENT_OOM_SCORE_ADJ = 500


def _bias_agent_child_toward_oom(pid: int, *, proc_root: str = "/proc") -> None:
    """Prefer killing an agent-facing child before the root grader on OOM."""
    try:
        with open(f"{proc_root}/{pid}/oom_score_adj", "w") as handle:
            handle.write(str(_AGENT_OOM_SCORE_ADJ))
    except OSError:
        pass


@mcp.tool()
async def bash(command: str = "", restart: bool = False) -> ToolResult:
    """Run a shell command in the agent workdir."""
    try:
        service_runtime = _task_service_runtime()
    except InfrastructureFault as exc:
        return ToolResult(error=str(exc))
    if service_runtime is not None:
        try:
            if restart:
                service_runtime.restart_main()
            if not command:
                return ToolResult(output="")
            result = service_runtime.exec_main(command)
        except InfrastructureFault as exc:
            return ToolResult(error=str(exc))
        if result.timed_out:
            return ToolResult(
                output=result.stdout,
                error=(
                    result.stderr
                    or f"command timed out with status {result.returncode}"
                ),
            )
        return ToolResult(
            output=result.stdout,
            error=(
                result.stderr or f"command exited with status {result.returncode}"
                if result.returncode
                else None
            ),
        )

    _ = restart
    if not command:
        return ToolResult(output="")
    try:
        popen_kwargs = _agent_subprocess_kwargs()
    except RuntimeError as exc:
        return ToolResult(error=str(exc))
    proc = subprocess.Popen(  # noqa: ASYNC220 - legacy streaming tool contract
        command,
        shell=True,
        cwd=WORKDIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    _bias_agent_child_toward_oom(proc.pid)
    stdout, stderr = proc.communicate()
    return ToolResult(output=stdout, error=stderr if proc.returncode else None)


@mcp.tool()
async def task_mcp_list_tools(
    server: str = Field(description="Declared task-local MCP server name"),
) -> dict:
    """List tools exposed by one ready declared task-local MCP server."""
    service_runtime = _task_service_runtime()
    if service_runtime is None:
        raise ToolRequestError("this task does not declare task-local MCP servers")
    return await service_runtime.task_mcp_list_tools(server)


@mcp.tool()
async def task_mcp_call(
    server: str = Field(description="Declared task-local MCP server name"),
    tool_name: str = Field(description="Tool name returned by task_mcp_list_tools"),
    arguments: dict | None = None,
) -> dict:
    """Call one tool on a ready declared task-local MCP server."""
    service_runtime = _task_service_runtime()
    if service_runtime is None:
        raise ToolRequestError("this task does not declare task-local MCP servers")
    return await service_runtime.task_mcp_call(
        server,
        tool_name,
        arguments,
    )


# Resolved at module load — before the agent ever gets control — so a later
# `rm -rf /tmp/output && ln -s /mcp_server/data /tmp/output` from agent bash
# (uid 1000 owns /tmp/output, /tmp is sticky-but-world-writable so the swap
# is allowed) cannot shift the trust anchor under us.
_AGENT_PATH_ROOTS: tuple[Path, ...] = tuple(
    root.resolve(strict=False) for root in (WORKDIR, OUTPUT_DIR)
)


def _resolve_agent_path(raw_path: str) -> Path:
    """Resolve `raw_path` to a real path under an agent-writable root.

    Defense in depth in front of the privilege-dropped editor worker: we
    canonicalize the path with symlinks resolved and require the result to
    live under one of the roots captured at startup, giving clear errors and
    confining the editor to the agent area. The worker's unprivileged uid is
    the real boundary — it cannot read the 0700 fixtures even if a symlink is
    swapped in after this check — but resolving against cached roots still
    defeats the obvious escapes (`/mcp_server/data/x`, `ln -s /mcp_server
    /workdir/m`, or the agent re-pointing a root such as /tmp/output).
    """
    target = Path(raw_path)
    if not target.is_absolute():
        target = WORKDIR / target
    resolved = target.resolve(strict=False)
    for root in _AGENT_PATH_ROOTS:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    allowed = ", ".join(str(root) for root in _AGENT_PATH_ROOTS)
    raise PermissionError(
        f"path {raw_path!r} resolves to {resolved} which is outside the agent-writable roots ({allowed})"
    )


# Runs in the privilege-dropped editor subprocess: reads a request JSON on
# stdin, performs the file op on the already-resolved path, and writes a
# {"output", "error"} JSON to stdout. Because this executes as the agent uid,
# the kernel's 0700 fixture perms — not just the path check above — enforce
# confinement, and any file it creates is agent-owned.
_EDITOR_WORKER = textwrap.dedent("""
    import sys
    sys.path[:] = [p for p in sys.path if p not in ("", ".")]
    import json
    from pathlib import Path

    req = json.load(sys.stdin)
    command = req["command"]
    target = Path(req["path"])
    file_text = req.get("file_text") or ""
    old_str = req.get("old_str") or ""
    new_str = req.get("new_str") or ""
    insert_line = req.get("insert_line") or 0
    insert_text = req.get("insert_text") or ""


    def emit(output=None, error=None):
        json.dump({"output": output, "error": error}, sys.stdout)


    try:
        if command == "create":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file_text)
            emit(output=f"created {target}")
        elif command == "view":
            emit(output=target.read_text())
        elif command == "str_replace":
            if not old_str:
                emit(error="old_str and new_str are required")
            else:
                text = target.read_text()
                if old_str not in text:
                    emit(error="old_str not found")
                else:
                    target.write_text(text.replace(old_str, new_str, 1))
                    emit(output=f"updated {target}")
        elif command == "insert":
            text = target.read_text()
            lines = text.splitlines()
            lines.insert(max(0, int(insert_line)), insert_text)
            target.write_text("\\n".join(lines) + "\\n")
            emit(output=f"updated {target}")
        else:
            emit(error=f"unsupported command: {command}")
    except Exception as exc:
        emit(error=f"{type(exc).__name__}: {exc}")
    """)


@mcp.tool(name="str_replace_editor")
async def str_replace_editor(
    *,
    command: str,
    path: str,
    file_text: str = "",
    old_str: str = "",
    new_str: str = "",
    insert_line: int = 0,
    insert_text: str = "",
    view_range: list | None = None,
) -> ToolResult:
    """Minimal file editor compatible with common str_replace_editor calls.

    Path confinement happens here (as root, resolving symlinks), but the file
    I/O runs in a subprocess dropped to the unprivileged agent account via
    `_agent_subprocess_kwargs()`. That makes the kernel's 0700 fixture perms —
    not just the path check — the real boundary, so a symlink swapped in
    between resolve and open cannot trick a privileged reader, and created
    files come out agent-owned with no chown fix-up needed.
    """
    _ = view_range
    try:
        service_runtime = _task_service_runtime()
    except InfrastructureFault as exc:
        return ToolResult(error=str(exc))
    if service_runtime is not None:
        try:
            output = service_runtime.edit_main_file(
                command=command,
                path=path,
                file_text=file_text,
                old_str=old_str,
                new_str=new_str,
                insert_line=insert_line,
                insert_text=insert_text,
            )
        except (InfrastructureFault, OSError, ValueError) as exc:
            return ToolResult(error=f"{type(exc).__name__}: {exc}")
        return ToolResult(output=output)

    try:
        target = _resolve_agent_path(path)
    except Exception as exc:  # noqa: BLE001 - untrusted tool argument boundary
        return ToolResult(error=f"{type(exc).__name__}: {exc}")

    request = {
        "command": command,
        "path": str(target),
        "file_text": file_text,
        "old_str": old_str,
        "new_str": new_str,
        "insert_line": insert_line,
        "insert_text": insert_text,
    }
    try:
        popen_kwargs = _agent_subprocess_kwargs(isolate_python=True)
    except RuntimeError as exc:
        return ToolResult(error=str(exc))
    editor_args = [sys.executable, "-P", "-c", _EDITOR_WORKER]
    editor = subprocess.Popen(  # noqa: ASYNC220 - isolated editor worker
        editor_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=WORKDIR,
        text=True,
        **popen_kwargs,
    )
    _bias_agent_child_toward_oom(editor.pid)
    stdout, stderr = editor.communicate(json.dumps(request))
    proc = subprocess.CompletedProcess(editor_args, editor.returncode, stdout, stderr)
    if proc.returncode != 0 or not proc.stdout:
        detail = (
            proc.stderr.strip()
            or proc.stdout.strip()
            or f"editor worker exited with {proc.returncode}"
        )
        return ToolResult(error=detail)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return ToolResult(error=proc.stderr.strip() or "invalid editor worker output")
    return ToolResult(output=result.get("output"), error=result.get("error"))


_RUNNER = textwrap.dedent("""
    import sys
    sys.path[:] = [p for p in sys.path if p not in ("", ".")]
    import inspect, json, math, os, traceback
    from pathlib import Path

    # Grader-only deps (/mcp_server/grading_deps) must be importable BEFORE the
    # grader's top-level imports run, so a task's ``grading_dependencies`` resolve
    # on the platform (Boreal) grade exactly as they do on the local/harness lane
    # (grader_runner.worker._prepend_grading_deps). Defensive import with a literal
    # fallback -- mirrors the AgentFault import below -- since this isolated runner
    # may not have env_server importable. Idempotent; a no-op when the dir is
    # absent (every task without grading_dependencies) and unreachable by the
    # uid-1000 agent (0700 /mcp_server), so it augments only this root grader path.
    try:
        from env_server.config import GRADING_DEPS_DIR as _grading_deps_dir
        _grading_deps = str(_grading_deps_dir)
    except Exception:
        _grading_deps = "/mcp_server/grading_deps"
    if os.path.isdir(_grading_deps) and _grading_deps not in sys.path:
        sys.path.insert(0, _grading_deps)

    try:
        from grader_runner.worker import _enter_pid_namespace
        _enter_pid_namespace()
    except Exception as _pid_exc:
        print(
            "WARNING: PID-namespace isolation setup failed (%s); grading will "
            "continue without that defense." % _pid_exc,
            file=sys.stderr,
        )

    def _clamp_score(value):
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"non-finite score {score!r}")
        return max(0.0, min(1.0, score))

    def _scalar_payload(score, metadata=None):
        score = _clamp_score(score)
        return {
            "score": score,
            "subscores": {"score": score},
            "weights": {"score": 1.0},
            "metadata": dict(metadata or {}),
        }

    def _fallback_payload(result, error):
        metadata = {"grading_normalization_error": error}
        if isinstance(result, dict):
            return _scalar_payload(result.get("score", 0.0), metadata)
        return _scalar_payload(result, metadata)

    def _normalized_payload(result, transcript=""):
        from grader_runner.evaluate import normalize_result_payload

        return normalize_result_payload(result, transcript=transcript)

    def _write_payload(path, payload):
        if not path:
            raise RuntimeError("RUBRIC_RESULT_PATH is not set")
        with open(path, "w") as _result_file:
            json.dump(payload, _result_file, sort_keys=True, default=str)
            _result_file.write("\\n")

    def _main():
        result_path = os.environ.pop("RUBRIC_RESULT_PATH", "")
        os.environ["LBX_EVALUATION_PRODUCTION"] = "1"
        source = sys.stdin.read()
        # The agent transcript (if any) is staged by the server in a root-owned
        # 0600 file and its path handed over via env, so the grader/test_file can
        # inspect what the agent actually did (e.g. reject submissions that read
        # grader-private artifacts). Exposed as a TRANSCRIPT string global plus its
        # TRANSCRIPT_PATH; both are empty when no transcript was provided.
        _transcript = ""
        _transcript_path = os.environ.get("LBX_AGENT_TRANSCRIPT_PATH") or ""
        if _transcript_path:
            try:
                with open(_transcript_path) as _tf:
                    _transcript = _tf.read()
            except OSError:
                _transcript = ""
        _service_root = Path(
            os.environ.get("LBX_SERVICE_ARTIFACT_SNAPSHOT") or "/tmp/output"
        )
        if not _service_root.is_dir():
            _service_root = Path("/tmp/output")
        namespace = {
            "__name__": "agent_test_module",
            "__file__": "/mcp_server/grader/compute_score.py",
            "__builtins__": __builtins__,
            "TRANSCRIPT": _transcript,
            "TRANSCRIPT_PATH": _transcript_path,
            "SERVICE_ARTIFACT_ROOT": _service_root,
            "SERVICE_ARTIFACT_MANIFEST": Path(
                os.environ.get("LBX_SERVICE_ARTIFACT_MANIFEST")
                or "/tmp/output/manifest.json"
            ),
        }
        namespace["WORKSPACE"] = namespace["SERVICE_ARTIFACT_ROOT"]
        try:
            from grading.faults import AgentFault, GraderFault, InfrastructureFault
        except Exception:
            class AgentFault(Exception):
                pass
            class GraderFault(Exception):
                pass
            class InfrastructureFault(Exception):
                pass
        try:
            exec(source, namespace)
            compute = namespace.get("compute_score")
            registered_task = namespace.get("TASK")
            try:
                from grading.evaluation import RubricTask
            except Exception:
                RubricTask = ()
            workspace = namespace["SERVICE_ARTIFACT_ROOT"]
            private = Path("/mcp_server/data")
            if isinstance(registered_task, RubricTask):
                compute = lambda: registered_task.grade(
                    workspace=workspace,
                    trajectory=_transcript,
                    private=private,
                )
            elif callable(compute):
                original_compute = compute
                signature = inspect.signature(original_compute)
                try:
                    signature.bind(workspace, _transcript, private)
                except TypeError:
                    try:
                        signature.bind()
                    except TypeError as _signature_exc:
                        raise RuntimeError(
                            "compute_score must accept either no arguments or "
                            "(workspace, trajectory, private)"
                        ) from _signature_exc
                    compute = lambda: original_compute()
                else:
                    compute = lambda: original_compute(
                        workspace,
                        _transcript,
                        private,
                    )
            if not callable(compute):
                raise RuntimeError(
                    "test_file defines neither TASK=RubricTask(...) nor compute_score()"
                )
            try:
                previous_cwd = os.getcwd()
                os.chdir(workspace)
                try:
                    result = compute()
                finally:
                    os.chdir(previous_cwd)
            except AgentFault as exc:
                payload = _scalar_payload(
                    0.0,
                    {"return_shape": "agent_fault", "agent_fault": str(exc)},
                )
                payload["env_internal_failure"] = False
                payload["env_internal_failure_logs"] = None
                _write_payload(result_path, payload)
                return
            except (GraderFault, InfrastructureFault) as exc:
                message = f"{type(exc).__name__}: {exc}"
                payload = _scalar_payload(
                    0.0,
                    {"return_shape": "typed_grader_failure", "error": message},
                )
                payload["env_internal_failure"] = True
                payload["env_internal_failure_logs"] = [message]
                _write_payload(result_path, payload)
                return
            except Exception as exc:
                # Candidate evaluation must never gain a free episode/group veto
                # from an untyped Python exception. Keep zero, surface a critical
                # operator alert, and let trusted CI's adversarial probes block it.
                message = f"{type(exc).__name__}: {exc}"
                payload = _scalar_payload(
                    0.0,
                    {
                        "return_shape": "unclassified_grader_crash",
                        "error": message,
                        "critical_operator_alert": True,
                    },
                )
                payload["env_internal_failure"] = False
                payload["env_internal_failure_logs"] = None
                _write_payload(result_path, payload)
                return
            try:
                payload = _normalized_payload(result, _transcript)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                payload = _scalar_payload(
                    0.0,
                    {"return_shape": "normalization_failure", "error": message},
                )
                payload["env_internal_failure"] = True
                payload["env_internal_failure_logs"] = [message]
            _write_payload(result_path, payload)
        except Exception as exc:
            traceback.print_exc()
            message = f"{type(exc).__name__}: {exc}"
            payload = _scalar_payload(
                0.0,
                {
                    "return_shape": "grader_import_failure",
                    "error": message,
                    "traceback": traceback.format_exc(),
                },
            )
            payload["env_internal_failure"] = True
            payload["env_internal_failure_logs"] = [message]
            _write_payload(result_path, payload)

    _main()
    """)


def _clamp_score(value: Any) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"non-finite score {score!r}")
    return max(0.0, min(1.0, score))


def _normalize_weights(
    subscores: dict[str, float], raw_weights: Any
) -> dict[str, float]:
    if not subscores:
        return {"score": 1.0}

    weights: dict[str, float] = {}
    if isinstance(raw_weights, dict):
        weights = {
            key: max(0.0, float(raw_weights.get(key, 0.0)))
            for key in subscores
            if _is_number(raw_weights.get(key, 0.0))
        }

    if not weights or sum(weights.values()) <= 0.0:
        even = 1.0 / len(subscores)
        return {key: even for key in subscores}

    total = sum(weights.values())
    normalized = {key: value / total for key, value in weights.items()}
    missing = [key for key in subscores if key not in normalized]
    if missing:
        # Preserve explicit weights when possible, then distribute any tiny
        # leftover caused by malformed/incomplete maps across missing keys.
        leftover = max(0.0, 1.0 - sum(normalized.values()))
        share = leftover / len(missing) if missing else 0.0
        normalized.update({key: share for key in missing})

    drift = 1.0 - sum(normalized.values())
    if normalized:
        last_key = list(normalized)[-1]
        normalized[last_key] += drift
    return normalized


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _criterion_identity(entry: dict[str, Any]) -> str:
    for key in ("criterion_id", "id", "criterion"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("name", "label", "description"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _display_label(entry: dict[str, Any], fallback: str) -> str:
    for key in ("description", "label", "name", "criterion"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _unique_labels(labels_by_id: dict[str, str]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for label in labels_by_id.values():
        counts[label] = counts.get(label, 0) + 1

    used: set[str] = set()
    unique: dict[str, str] = {}
    for criterion_id, base in labels_by_id.items():
        if counts[base] <= 1 and base not in used:
            label = base
        else:
            id_suffix = f"{base} [{criterion_id}]"
            if id_suffix not in used:
                label = id_suffix
            else:
                idx = 2
                label = f"{base} ({idx})"
                while label in used:
                    idx += 1
                    label = f"{base} ({idx})"
        used.add(label)
        unique[criterion_id] = label
    return unique


def _weight_for(
    raw_weights: Any,
    *,
    criterion_id: str,
    label: str,
    entry: dict[str, Any] | None = None,
) -> Any:
    if entry is not None and entry.get("weight") is not None:
        return entry.get("weight")
    if isinstance(raw_weights, dict):
        for key in (label, criterion_id):
            if key in raw_weights:
                return raw_weights[key]
    return 0.0


def _canonicalize_breakdown(
    raw_breakdown: Any, labels_by_id: dict[str, str]
) -> list[dict[str, Any]]:
    if not isinstance(raw_breakdown, list):
        return []

    out: list[dict[str, Any]] = []
    for entry in raw_breakdown:
        if not isinstance(entry, dict):
            continue
        criterion_id = _criterion_identity(entry)
        if not criterion_id:
            continue
        label = labels_by_id.get(criterion_id) or _display_label(entry, criterion_id)
        description = str(entry.get("description") or label).strip()
        canonical = dict(entry)
        canonical["id"] = criterion_id
        canonical["criterion_id"] = criterion_id
        canonical["label"] = label
        canonical["description"] = description
        out.append(canonical)
    return out


def _headline_subscores(
    subscores: dict[str, float],
    weights: dict[str, float],
    headline_score: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (subscores, weights) whose weighted sum equals ``headline_score``.

    The platform records ``sum(subscore * weight)`` from this MCP ``Grade``
    reply as the run reward. A grader headline that applies caps, gates, a
    binary collapse, or a calibrated/anchor-mapped curve is deliberately NOT a
    weighted average of the per-criterion subscores (see GRADING.md: a score
    dict's ``score`` is used verbatim, never recomputed). Emitting the raw
    per-criterion rows as the scored quantity therefore silently discards that
    headline and strands it in metadata.

    When the headline already equals the weighted subscore sum (bare float /
    plain weighted rubric with no override), the rows are returned unchanged.
    Otherwise the per-criterion rows are demoted to zero weight — kept for
    display and credit assignment — and a single row carries the headline at
    weight 1.0 so caps/gates/calibration actually reach the reward. The
    per-criterion view still survives in metadata (structured_subscores /
    rubric_breakdown / serialized_grade), which is what the Boreal UI reads.
    """
    if not subscores:
        return {"score": headline_score}, {"score": 1.0}
    weighted_sum = _clamp_score(
        sum(subscores[key] * weights.get(key, 0.0) for key in subscores)
    )
    if abs(weighted_sum - headline_score) <= 1e-9:
        return dict(subscores), dict(weights)
    headline_key = "score" if "score" not in subscores else "__headline_score__"
    reward_subscores = {**subscores, headline_key: headline_score}
    reward_weights = {key: 0.0 for key in subscores}
    reward_weights[headline_key] = 1.0
    return reward_subscores, reward_weights


def _grade_from_payload(payload: dict[str, Any]) -> Grade:
    headline_score = _clamp_score(payload.get("score", 0.0))
    structured = payload.get("structured_subscores")
    if isinstance(structured, list) and structured:
        raw_structured = [entry for entry in structured if isinstance(entry, dict)]
        base_labels = {
            _criterion_identity(entry): _display_label(
                entry, _criterion_identity(entry)
            )
            for entry in raw_structured
            if _criterion_identity(entry)
        }
        labels_by_id = _unique_labels(base_labels)

        subscores = {}
        raw_weights = {}
        canonical_structured = []
        for entry in structured:
            if not isinstance(entry, dict):
                continue
            criterion_id = _criterion_identity(entry)
            if not criterion_id:
                continue
            label = labels_by_id[criterion_id]
            subscores[label] = _clamp_score(entry.get("score", 0.0))
            raw_weights[label] = _weight_for(
                payload.get("weights"),
                criterion_id=criterion_id,
                label=label,
                entry=entry,
            )
            description = str(entry.get("description") or label).strip()
            canonical_entry = dict(entry)
            canonical_entry["name"] = label
            canonical_entry["label"] = label
            canonical_entry["id"] = criterion_id
            canonical_entry["criterion_id"] = criterion_id
            canonical_entry["description"] = description
            canonical_structured.append(canonical_entry)
        if not subscores:
            subscores = {"score": headline_score}
            raw_weights = {"score": 1.0}
            canonical_structured = []
        weights = _normalize_weights(subscores, raw_weights)
    else:
        raw_subscores = payload.get("subscores")
        if not isinstance(raw_subscores, dict) or not raw_subscores:
            raw_subscores = {"score": headline_score}

        metadata = dict(payload.get("metadata") or {})
        raw_breakdown = metadata.get("rubric_breakdown")
        if isinstance(raw_breakdown, list) and raw_breakdown:
            breakdown_by_id = {
                _criterion_identity(entry): entry
                for entry in raw_breakdown
                if isinstance(entry, dict) and _criterion_identity(entry)
            }
            base_labels = {
                str(key): _display_label(breakdown_by_id.get(str(key), {}), str(key))
                for key in raw_subscores
            }
            labels_by_id = _unique_labels(base_labels)
            subscores = {
                labels_by_id[str(key)]: _clamp_score(value)
                for key, value in raw_subscores.items()
            }
            raw_weights = {
                labels_by_id[str(key)]: _weight_for(
                    payload.get("weights"),
                    criterion_id=str(key),
                    label=labels_by_id[str(key)],
                )
                for key in raw_subscores
            }
        else:
            labels_by_id = {}
            subscores = {
                str(key): _clamp_score(value) for key, value in raw_subscores.items()
            }
            raw_weights = payload.get("weights")
        weights = _normalize_weights(subscores, raw_weights)
        canonical_structured = []
    metadata = dict(payload.get("metadata") or {})
    if canonical_structured:
        metadata["structured_subscores"] = canonical_structured
    breakdown = _canonicalize_breakdown(
        metadata.get("rubric_breakdown"),
        labels_by_id if "labels_by_id" in locals() else {},
    )
    if breakdown:
        metadata["rubric_breakdown"] = breakdown
    if payload.get("scoring_mode") is not None:
        metadata["scoring_mode"] = payload["scoring_mode"]
    if payload.get("penalties") is not None:
        metadata["penalties"] = payload["penalties"]
    metadata["score"] = headline_score
    metadata["headline_score"] = headline_score
    metadata["reported_final_score"] = headline_score
    metadata["weighted_subscore_total"] = _clamp_score(
        sum(subscores[key] * weights.get(key, 0.0) for key in subscores)
    )
    metadata.setdefault("weighted_total", metadata["weighted_subscore_total"])
    metadata["rubric_weights"] = weights
    metadata.setdefault("return_shape", "rubric_grade" if structured else "score_dict")
    metadata["serialized_grade"] = {
        "score": headline_score,
        "subscores": subscores,
        "weights": weights,
        "structured_subscores": canonical_structured,
        "scoring_mode": payload.get("scoring_mode"),
        "penalties": payload.get("penalties"),
    }
    reward_subscores, reward_weights = _headline_subscores(
        subscores, weights, headline_score
    )
    return Grade(
        subscores=reward_subscores,
        weights=reward_weights,
        metadata=metadata,
        env_internal_failure=payload.get("env_internal_failure"),
        env_internal_failure_logs=payload.get("env_internal_failure_logs"),
    )


def _stage_transcript(transcript: str) -> str | None:
    """Write the transcript to a root-owned 0600 file; return its path.

    The runner reads it via ``LBX_AGENT_TRANSCRIPT_PATH``. We pass the path
    rather than the content so large transcripts never hit env-size limits,
    and 0600 keeps it unreadable by the unprivileged agent. Returns ``None``
    when there is nothing to stage or staging fails.
    """
    if not transcript:
        return None
    try:
        fd, path = tempfile.mkstemp(prefix="lbx-transcript-", suffix=".txt")
        with os.fdopen(fd, "w") as handle:
            handle.write(transcript)
        os.chmod(path, 0o600)
        return path
    except OSError:
        return None


def _stage_result_file() -> tuple[str | None, bool]:
    """Create the authoritative result file and report ENOSPC/EDQUOT."""
    result_dir: str | None = None
    result_path: str | None = None
    fd: int | None = None
    try:
        result_dir = tempfile.mkdtemp(prefix=_RESULT_DIR_PREFIX, dir=_GRADING_TMPDIR)
        os.chmod(result_dir, 0o700)
        fd, result_path = tempfile.mkstemp(
            prefix="result-", suffix=".json", dir=result_dir
        )
        os.close(fd)
        fd = None
        os.chmod(result_path, 0o600)
        return result_path, False
    except OSError as exc:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        _unlink_if_present(result_path)
        if result_dir:
            try:
                os.rmdir(result_dir)
            except OSError:
                pass
        return None, exc.errno in (errno.ENOSPC, errno.EDQUOT)


def _unlink_if_present(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _cleanup_result_file(path: str | None) -> None:
    if not path:
        return
    result_dir = Path(path).parent
    _unlink_if_present(path)
    if result_dir.name.startswith(_RESULT_DIR_PREFIX):
        try:
            result_dir.rmdir()
        except OSError:
            pass


def _stage_evaluation_trace() -> str | None:
    """Return a non-existent path inside a root-owned private directory."""
    try:
        trace_dir = Path(tempfile.mkdtemp(prefix=_TRACE_DIR_PREFIX)).resolve(
            strict=True
        )
        trace_dir.chmod(0o700)
        return str(trace_dir / _TRACE_FILENAME)
    except OSError:
        return None


def _cleanup_evaluation_trace(path: str | None) -> None:
    if not path:
        return
    trace_dir = Path(path).parent
    if trace_dir.name.startswith(_TRACE_DIR_PREFIX):
        shutil.rmtree(trace_dir, ignore_errors=True)


def _clear_persisted_evaluation_trace() -> None:
    destination_dir = OUTPUT_DIR / _TRACE_OUTPUT_DIRNAME
    if destination_dir.is_symlink() or (
        destination_dir.exists() and not destination_dir.is_dir()
    ):
        destination_dir.unlink()
    elif destination_dir.exists():
        shutil.rmtree(destination_dir)


def _persist_evaluation_trace(path: str) -> Path:
    """Copy a completed private trace into Taiga's extracted output tree."""
    source = Path(path)
    source_context = open_regular_file(source, max_bytes=None)
    try:
        source_handle, _source_info = source_context.__enter__()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "sealed evaluation did not produce a private replay trace"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"private replay trace is not a regular file: {exc}"
        ) from exc

    destination_dir = OUTPUT_DIR / _TRACE_OUTPUT_DIRNAME
    root_fd = -1
    destination_fd = -1
    file_fd = -1
    try:
        root_fd = open_directory_fd(OUTPUT_DIR)
        os.mkdir(_TRACE_OUTPUT_DIRNAME, mode=0o700, dir_fd=root_fd)
        destination_fd = os.open(
            _TRACE_OUTPUT_DIRNAME,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=root_fd,
        )
        destination = destination_dir / _TRACE_FILENAME
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        file_fd = os.open(_TRACE_FILENAME, flags, 0o600, dir_fd=destination_fd)
        with os.fdopen(file_fd, "wb", closefd=True) as destination_handle:
            file_fd = -1
            while chunk := source_handle.read(1024 * 1024):
                destination_handle.write(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        return destination
    except OSError as exc:
        raise RuntimeError(
            f"could not persist private evaluation trace: {exc}"
        ) from exc
    finally:
        source_context.__exit__(None, None, None)
        if file_fd >= 0:
            os.close(file_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _write_agent_fault_trace(path: str, *, nonce: str) -> None:
    """Record replay identity when grading stops before protocol evaluation."""
    payload = {
        "schema_version": "continuous-evaluation-trace.v1",
        "protocol": "agent-fault.v1",
        "replay": {"nonce": nonce},
        "targets": {},
    }
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", closefd=True) as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _coerce_timeout(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _evaluate_timeout(timeout_s: float | None) -> float:
    return (
        _coerce_timeout(timeout_s)
        or _coerce_timeout(os.environ.get(_EVALUATE_TIMEOUT_ENV))
        or _DEFAULT_EVALUATE_TIMEOUT_S
    )


def _failure_grade(
    metadata: dict[str, Any],
    error: str | None = None,
    *,
    env_internal_failure: bool | None = True,
    env_internal_failure_logs: list[str] | None = None,
) -> Grade:
    if error:
        metadata["error"] = error
    return Grade(
        subscores={"score": 0.0},
        weights={"score": 1.0},
        metadata={**metadata, "score": 0.0, "headline_score": 0.0},
        env_internal_failure=env_internal_failure,
        env_internal_failure_logs=env_internal_failure_logs,
    )


def _agent_uid_for_cleanup() -> int | None:
    try:
        return _agent_identity()[0]
    except RuntimeError:
        return None


def _pre_grade_resource_cleanup() -> tuple[int | None, dict[str, Any], bool]:
    agent_uid = _agent_uid_for_cleanup()
    removed, flooded = cleanup_agent_tmpfs(agent_uid)
    metadata: dict[str, Any] = {}
    if removed:
        metadata["pre_grade_agent_tmpfs_entries_removed"] = removed
    if flooded:
        metadata["pre_grade_agent_tmpfs_flood"] = True
    return agent_uid, metadata, flooded


def _memory_snapshot(mem: dict[str, int] | None = None) -> dict[str, int]:
    mem = meminfo_kib() if mem is None else mem
    return {
        key: mem[key]
        for key in ("MemTotal", "MemAvailable", "Shmem", "SwapTotal", "SwapFree")
        if key in mem
    }


_RESOURCE_LABELS = {
    "disk_exhausted": "the grading filesystem",
    "shared_memory_exhausted": "shared memory",
}


def _agent_resource_grade(
    kind: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
    *,
    meminfo: dict[str, int] | None = None,
) -> Grade:
    """Record an authoritative agent zero for exhausting a shared resource.

    ``meminfo`` should be the reading taken when the failure happened. Reading
    it here instead would describe the box after cleanup has already freed the
    agent's /dev/shm entries, so a disputed zero would be filed alongside
    evidence of a perfectly healthy machine.
    """
    details = {**(metadata or {}), "agent_fault": kind}
    if kind in {"shared_memory_exhausted", "agent_tmpfs_flood"}:
        snapshot = _memory_snapshot(meminfo)
        if snapshot:
            details["meminfo_kib_at_failure"] = snapshot
    return _failure_grade(details, reason, env_internal_failure=False)


def _cleanup_grade_processes(
    metadata: dict[str, Any],
    *,
    phase: str,
) -> Grade | None:
    try:
        metadata[f"{phase}_grade_cleanup"] = pre_grade_cleanup(OUTPUT_DIR)
    except AgentProcessQuiesceError as exc:
        message = f"{phase}-grade agent process quiesce failed: {exc}"
        metadata["agent_fault"] = str(exc)
        return _failure_grade(
            metadata,
            message,
            env_internal_failure=False,
        )
    except ProcessQuiesceError as exc:
        message = f"{phase}-grade process quiesce failed: {exc}"
        return _failure_grade(
            metadata,
            message,
            env_internal_failure=True,
            env_internal_failure_logs=[message],
        )
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _mirror_runner_output(stdout: Any, stderr: Any, metadata: dict[str, Any]) -> None:
    stdout_text = _as_text(stdout)
    stderr_text = _as_text(stderr)
    if stdout_text:
        sys.stderr.write(stdout_text)
    if stderr_text:
        sys.stderr.write(stderr_text)
        metadata["stderr"] = stderr_text[-2000:]


def _kill_runner_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def _read_result_payload(
    result_path: str, metadata: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        raw = Path(result_path).read_text()
    except OSError as exc:
        metadata["error"] = f"could not read rubric result: {exc}"
        return None
    if not raw.strip():
        metadata["error"] = "missing rubric result"
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        metadata["error"] = f"invalid rubric result JSON: {exc}"
        return None
    if not isinstance(payload, dict):
        metadata["error"] = "rubric result must be a JSON object"
        return None
    return payload


def _evaluate(
    test_file_source: str,
    transcript: str = "",
    timeout_s: float | None = None,
    *,
    trace_required: bool = False,
) -> Grade:
    agent_uid, resource_metadata, tmpfs_flood = _pre_grade_resource_cleanup()
    if tmpfs_flood:
        return _agent_resource_grade(
            "agent_tmpfs_flood",
            "agent flooded /dev/shm with entries before grading could start",
            resource_metadata,
        )
    pre_grade = sample_resource_exhaustion()
    if pre_grade.shmem:
        cleanup_agent_tmpfs(agent_uid)
        return _agent_resource_grade(
            "shared_memory_exhausted",
            "agent exhausted shared memory before grading could start",
            resource_metadata,
            meminfo=pre_grade.meminfo_kib,
        )
    runner_env = prepare_grader_cache()
    attested = runner_env.get("LBX_EVALUATION_PLAN_ATTESTED") == "1"
    if attested:
        runner_env.setdefault("LBX_EVALUATION_NONCE", secrets.token_hex(16))
    transcript_path = _stage_transcript(transcript)
    result_path, staging_disk_full = _stage_result_file()
    trace_path = _stage_evaluation_trace() if attested else None
    timeout = _evaluate_timeout(timeout_s)
    if transcript_path:
        runner_env["LBX_AGENT_TRANSCRIPT_PATH"] = transcript_path
    if not result_path:
        _unlink_if_present(transcript_path)
        _cleanup_evaluation_trace(trace_path)
        exhaustion = sample_resource_exhaustion()
        if staging_disk_full or exhaustion.disk:
            return _agent_resource_grade(
                "disk_exhausted",
                "agent exhausted the grading filesystem before the private result file could be staged",
                resource_metadata,
            )
        if exhaustion.shmem:
            return _agent_resource_grade(
                "shared_memory_exhausted",
                "agent exhausted shared memory before grading could start",
                resource_metadata,
                meminfo=exhaustion.meminfo_kib,
            )
        return _failure_grade(
            resource_metadata, "could not create private rubric result file"
        )
    if trace_required and not trace_path:
        _unlink_if_present(transcript_path)
        _cleanup_result_file(result_path)
        return _failure_grade({}, "could not create private evaluation trace sink")
    runner_env[_RESULT_PATH_ENV] = result_path
    if trace_path:
        runner_env[_TRACE_PATH_ENV] = trace_path
    metadata: dict[str, Any] = dict(resource_metadata)
    try:
        cleanup_failure = _cleanup_grade_processes(metadata, phase="pre")
        if cleanup_failure is not None:
            return cleanup_failure
        _clear_persisted_evaluation_trace()
        # Cut the agent's grade-time access to the live hidden env before the
        # grader subprocess runs (closes free reset()-seed fingerprinting, a
        # twin-env oracle, private-method probes). No-op for static tasks.
        try:
            from env_server.supervisor import stop_env_server

            stop_env_server()
        except Exception as exc:  # noqa: BLE001 - never block grading
            print(
                f"[ENV_SERVER] stop_env_server failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        proc = subprocess.Popen(
            [sys.executable, "-P", "-c", _RUNNER],
            cwd="/",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=runner_env,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(test_file_source, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_runner_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                stdout, stderr = exc.stdout, exc.stderr
            # Sample before cleanup: the post-grade cleanup reaps the agent's
            # /dev/shm entries, which is the very evidence of the exhaustion
            # that hung the grader.
            exhaustion = sample_resource_exhaustion(stderr)
            cleanup_failure = _cleanup_grade_processes(metadata, phase="post")
            if cleanup_failure is not None:
                return cleanup_failure
            _clear_persisted_evaluation_trace()
            _mirror_runner_output(stdout, stderr, metadata)
            # No exit status to corroborate with: the grader never finished.
            kind = exhaustion.agent_fault_kind()
            if kind:
                cleanup_agent_tmpfs(agent_uid)
                return _agent_resource_grade(
                    kind,
                    f"agent exhausted {_RESOURCE_LABELS[kind]} during grading",
                    metadata,
                    meminfo=exhaustion.meminfo_kib,
                )
            message = f"test_file subprocess timed out after {timeout:.3f}s"
            return _failure_grade(
                metadata,
                message,
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
        # Sampled before cleanup for the same reason as the timeout path above.
        exhaustion = sample_resource_exhaustion(stderr)
        cleanup_failure = _cleanup_grade_processes(metadata, phase="post")
        if cleanup_failure is not None:
            return cleanup_failure
        _clear_persisted_evaluation_trace()
        _mirror_runner_output(stdout, stderr, metadata)
        if proc.returncode:
            kind = exhaustion.agent_fault_kind(
                returncode=proc.returncode, stderr=stderr
            )
            if kind:
                if kind == "shared_memory_exhausted":
                    cleanup_agent_tmpfs(agent_uid)
                return _agent_resource_grade(
                    kind,
                    f"agent exhausted {_RESOURCE_LABELS[kind]} during grading",
                    metadata,
                    meminfo=exhaustion.meminfo_kib,
                )
            classification = classify_failure(
                proc.returncode, metadata.get("stderr", "")
            )
            message = classification.reason
            payload = _read_result_payload(result_path, metadata)
            if payload is not None:
                try:
                    grade = _grade_from_payload(payload)
                except Exception:  # noqa: BLE001 - untrusted result boundary
                    grade = None
                if grade is not None:
                    grade.metadata = {**metadata, **(grade.metadata or {})}
                    return grade
            if not classification.is_infra:
                metadata["critical_operator_alert"] = True
                metadata["failure_classification"] = "unclassified_non_infra_exit"
            return _failure_grade(
                metadata,
                message,
                env_internal_failure=classification.is_infra,
                env_internal_failure_logs=(
                    [message] if classification.is_infra else None
                ),
            )
        payload = _read_result_payload(result_path, metadata)
        if payload is None:
            message = metadata.get("error", "missing rubric result")
            if exhaustion.disk:
                return _agent_resource_grade(
                    "disk_exhausted",
                    "agent exhausted the grading filesystem while publishing its result",
                    metadata,
                )
            if exhaustion.shmem:
                return _agent_resource_grade(
                    "shared_memory_exhausted",
                    "agent exhausted shared memory before the rubric result could be read",
                    metadata,
                    meminfo=exhaustion.meminfo_kib,
                )
            return _failure_grade(
                metadata,
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
        payload_metadata = payload.get("metadata")
        if (
            trace_required
            and trace_path
            and not Path(trace_path).exists()
            and isinstance(payload_metadata, dict)
            and payload_metadata.get("return_shape") == "agent_fault"
        ):
            try:
                _write_agent_fault_trace(
                    trace_path,
                    nonce=runner_env["LBX_EVALUATION_NONCE"],
                )
            except OSError as exc:
                message = f"could not record agent-fault replay trace: {exc}"
                return _failure_grade(
                    metadata,
                    message,
                    env_internal_failure=True,
                    env_internal_failure_logs=[message],
                )
        if trace_path and (trace_required or Path(trace_path).is_file()):
            try:
                _persist_evaluation_trace(trace_path)
            except RuntimeError as exc:
                message = str(exc)
                return _failure_grade(
                    metadata,
                    message,
                    env_internal_failure=True,
                    env_internal_failure_logs=[message],
                )
        try:
            grade = _grade_from_payload(payload)
        except Exception as exc:  # noqa: BLE001 - untrusted result boundary
            message = f"could not normalize rubric result: {type(exc).__name__}: {exc}"
            return _failure_grade(
                metadata,
                message,
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
        grade.metadata = {**metadata, **(grade.metadata or {})}
        if grade.env_internal_failure and exhaustion.disk:
            return _agent_resource_grade(
                "disk_exhausted",
                "grader reported an internal failure while the grading filesystem was exhausted",
                grade.metadata,
            )
        if grade.env_internal_failure and exhaustion.shmem:
            cleanup_agent_tmpfs(agent_uid)
            return _agent_resource_grade(
                "shared_memory_exhausted",
                "grader reported an internal failure while shared memory was exhausted",
                grade.metadata,
            )
        return grade
    finally:
        _unlink_if_present(transcript_path)
        _cleanup_result_file(result_path)
        _cleanup_evaluation_trace(trace_path)
        cleanup_agent_tmpfs(agent_uid)
        # Best-effort: free any leftover /dev/nvidia* fd holders so the
        # end-of-container checkpoint can save. Runs after the grade is computed
        # and must never change the score or fail grading.
        try:
            kill_nvproxy_fd_holders()
        except Exception as exc:  # noqa: BLE001 - cleanup must never fail grading
            print(
                f"[GRADING] nvproxy sweep failed (ignored): {exc}",
                file=sys.stderr,
                flush=True,
            )


@mcp.tool()
async def grade_problem(
    problem_id: str,
    transcript: str = Field(description="The full transcript produced by the model"),
    extra_fields: dict | None = None,
) -> Grade:
    """Grade by executing the image-baked grader through the test_file shim."""
    _ = problem_id
    fields = _extra(extra_fields)
    _verify_continuous_evaluation(fields)
    os.environ.pop(_SERVICE_SNAPSHOT_ENV, None)
    os.environ.pop(_SERVICE_MANIFEST_ENV, None)
    try:
        service_runtime = _task_service_runtime()
        service_result = (
            service_runtime.finalize_and_verify()
            if service_runtime is not None
            else None
        )
        service_handoff = (
            service_runtime.grader_handoff() if service_runtime is not None else None
        )
        if service_handoff is not None:
            os.environ[_SERVICE_SNAPSHOT_ENV] = str(service_handoff.workspace)
            os.environ[_SERVICE_MANIFEST_ENV] = str(service_handoff.manifest)
    except ServiceRuntimeAgentError as exc:
        return _failure_grade(
            {
                "failure_classification": "service_runtime_agent_fault",
                "agent_fault": str(exc),
            },
            str(exc),
            env_internal_failure=False,
        )
    except InfrastructureFault as exc:
        message = f"{type(exc).__name__}: {exc}"
        return _failure_grade(
            {"failure_classification": "service_runtime_infrastructure"},
            message,
            env_internal_failure=True,
            env_internal_failure_logs=[message],
        )
    if service_result is not None:
        try:
            grade = _grade_from_payload(service_result.payload)
        except Exception as exc:  # noqa: BLE001 - untrusted result boundary
            message = (
                "could not normalize nested verifier result: "
                f"{type(exc).__name__}: {exc}"
            )
            return _failure_grade(
                {"failure_classification": "service_verifier_result"},
                message,
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
        grade.metadata = {
            "service_runtime": "nested-docker",
            "service_verifier_result_dir": str(service_result.result_dir),
            **(grade.metadata or {}),
        }
        return grade
    test_file = fields.get("test_file")
    if not test_file:
        # A missing test_file is a caller/configuration fault, never an agent
        # fault. Reporting it as a plain 0.0 makes regrade and monitoring lanes
        # record a false zero that is indistinguishable from a failed attempt.
        message = "grade_problem called without extra_fields.test_file"
        return _failure_grade(
            {"failure_classification": "missing_test_file"},
            message,
            env_internal_failure=True,
            env_internal_failure_logs=[message],
        )
    timeout_s = _coerce_timeout(fields.get("grading_timeout_seconds"))
    policy = fields.get("continuous_evaluation") or fields.get("rubric_evaluation")
    trace_required = isinstance(policy, dict) and policy.get("trace_required") is True
    return _evaluate(
        str(test_file),
        transcript=transcript or "",
        timeout_s=timeout_s,
        trace_required=trace_required,
    )


_SHMEM_REAP_POLL_S = 0.5
_SHMEM_REAP_MIN_KIB = 2 * 1024 * 1024
_SHMEM_REAP_FRACTION = 0.30


def _shmem_over_reap_threshold(mem: dict[str, int]) -> bool:
    total = mem.get("MemTotal", 0)
    shmem = mem.get("Shmem", 0)
    return total > 0 and shmem >= max(
        _SHMEM_REAP_MIN_KIB, int(total * _SHMEM_REAP_FRACTION)
    )


def _shmem_reaper_loop(agent_uid: int, poll_s: float) -> None:
    while True:
        try:
            if _shmem_over_reap_threshold(meminfo_kib()):
                cleanup_agent_tmpfs(agent_uid)
        except Exception:  # noqa: BLE001, S110 - best-effort daemon
            pass
        time.sleep(poll_s)


def start_shmem_reaper(poll_s: float = _SHMEM_REAP_POLL_S) -> None:
    """Start a best-effort daemon that bounds agent-owned /dev/shm usage."""
    try:
        if os.geteuid() != 0:
            return
        agent_uid = _agent_uid_for_cleanup()
        if agent_uid is None:
            return
        threading.Thread(
            target=_shmem_reaper_loop,
            args=(agent_uid, poll_s),
            daemon=True,
            name="shmem-reaper",
        ).start()
    except Exception:  # noqa: BLE001, S110 - best-effort startup
        pass


def _protect_grader_from_oom(path: str = "/proc/self/oom_score_adj") -> None:
    """Best-effort protection for the root grading/control process."""
    try:
        with open(path, "w") as handle:
            handle.write("-1000")
    except OSError:
        pass


def main() -> None:
    """Run MCP server."""
    _protect_grader_from_oom()
    try:
        _pre_grade_resource_cleanup()
    except Exception:  # noqa: BLE001, S110 - best-effort startup
        pass
    WORKDIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORKDIR)
    start_shmem_reaper()
    # Hidden-environment tasks ([environment].hidden_env = env/hybrid) spawn an
    # env_server subprocess the agent reaches over /tmp/env.sock. A no-op for
    # static tasks. Best-effort: a failure to import/launch must never block a
    # normal task's MCP server from starting.
    try:
        from env_server.supervisor import supervise_if_enabled

        supervise_if_enabled()
    except Exception as exc:  # noqa: BLE001 - env server is opt-in; never block startup
        # stderr only: stdio transport owns fd 1 for JSON-RPC framing.
        print(
            f"[ENV_SERVER] supervisor failed to start; continuing: {exc}",
            file=sys.stderr,
            flush=True,
        )
    try:
        mcp.run(transport="stdio")
    finally:
        _shutdown_task_service_runtime()
