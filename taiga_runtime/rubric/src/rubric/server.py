"""Generic MCP runtime for Alignerr RL tasks.

This intentionally mirrors the small subset of the ML_Envs rubric server that
Boreal needs, without vendoring Anthropic-specific `taiga-core`.
"""

import dataclasses
import json
import math
import os
import pwd
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from grading.runtime_hardening import (
    classify_failure,
    kill_nvproxy_fd_holders,
    lock_down_grader_private,
    pre_grade_cleanup,
    prepare_grader_cache,
)
from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("alignerr-rl-tasks")

WORKDIR = Path("/workdir")
OUTPUT_DIR = Path("/tmp/output")

_RESULT_PATH_ENV = "RUBRIC_RESULT_PATH"
_EVALUATE_TIMEOUT_ENV = "RUBRIC_EVALUATE_TIMEOUT_S"
_DEFAULT_EVALUATE_TIMEOUT_S = 600.0
_RESULT_DIR_PREFIX = "lbx-rubric-result-"


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


@mcp.tool()
async def setup_problem(
    problem_id: str = Field(description="The id of the problem to solve"),
    extra_fields: dict = None,
    use_hinted_problem: bool = True,
) -> str:
    """Return the task prompt."""
    _ = use_hinted_problem
    lock_down_grader_private(
        (
            "/mcp_server/data",
            "/mcp_server/grader",
            "/mcp_server/grading",
            "/mcp_server/src/rubric",
            "/runtime/grading",
        )
    )
    fields = _extra(extra_fields)
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


@mcp.tool()
async def bash(command: str = "", restart: bool = False) -> ToolResult:
    """Run a shell command in the agent workdir."""
    _ = restart
    if not command:
        return ToolResult(output="")
    try:
        popen_kwargs = _agent_subprocess_kwargs()
    except RuntimeError as exc:
        return ToolResult(error=str(exc))
    proc = subprocess.run(
        command,
        shell=True,
        cwd=WORKDIR,
        text=True,
        capture_output=True,
        **popen_kwargs,
    )
    return ToolResult(
        output=proc.stdout, error=proc.stderr if proc.returncode else None
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
        f"path {raw_path!r} resolves to {resolved} which is outside the "
        f"agent-writable roots ({allowed})"
    )


# Runs in the privilege-dropped editor subprocess: reads a request JSON on
# stdin, performs the file op on the already-resolved path, and writes a
# {"output", "error"} JSON to stdout. Because this executes as the agent uid,
# the kernel's 0700 fixture perms — not just the path check above — enforce
# confinement, and any file it creates is agent-owned.
_EDITOR_WORKER = textwrap.dedent(
    """
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
    """
)


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
    view_range: list = None,
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
        target = _resolve_agent_path(path)
    except Exception as exc:
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
    proc = subprocess.run(
        [sys.executable, "-P", "-c", _EDITOR_WORKER],
        input=json.dumps(request),
        cwd=WORKDIR,
        text=True,
        capture_output=True,
        **popen_kwargs,
    )
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


_RUNNER = textwrap.dedent(
    """
    import sys
    sys.path[:] = [p for p in sys.path if p not in ("", ".")]
    import json, math, os, traceback

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

    def _normalized_payload(result):
        from grading import normalize_compute_score_return

        grade = normalize_compute_score_return(result)
        serialized = grade.to_dict() if hasattr(grade, "to_dict") else {}
        metadata = dict(getattr(grade, "metadata", None) or {})
        metadata.update(dict(serialized.get("metadata") or {}))
        if getattr(grade, "metadata", None):
            metadata.update(
                {
                    key: value
                    for key, value in dict(grade.metadata).items()
                    if key not in metadata
                }
            )
        metadata.setdefault("serialized_grade", serialized)
        return {
            "score": _clamp_score(
                grade.score() if hasattr(grade, "score") else serialized.get("score", 0.0)
            ),
            "subscores": dict(serialized.get("subscores") or getattr(grade, "subscores", None) or {}),
            "weights": dict(serialized.get("weights") or getattr(grade, "weights", None) or {}),
            "structured_subscores": serialized.get("structured_subscores") or [],
            "scoring_mode": serialized.get("scoring_mode"),
            "penalties": serialized.get("penalties"),
            "env_internal_failure": serialized.get("env_internal_failure"),
            "env_internal_failure_logs": serialized.get("env_internal_failure_logs"),
            "metadata": metadata,
        }

    def _write_payload(path, payload):
        if not path:
            raise RuntimeError("RUBRIC_RESULT_PATH is not set")
        with open(path, "w") as _result_file:
            json.dump(payload, _result_file, sort_keys=True, default=str)
            _result_file.write("\\n")

    def _main():
        result_path = os.environ.pop("RUBRIC_RESULT_PATH", "")
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
        namespace = {
            "__name__": "agent_test_module",
            "__file__": "/mcp_server/grader/compute_score.py",
            "__builtins__": __builtins__,
            "TRANSCRIPT": _transcript,
            "TRANSCRIPT_PATH": _transcript_path,
        }
        try:
            from grading.faults import AgentFault
        except Exception:
            class AgentFault(Exception):
                pass
        try:
            exec(source, namespace)
            compute = namespace.get("compute_score")
            if not callable(compute):
                raise RuntimeError("test_file does not define compute_score()")
            try:
                result = compute()
            except AgentFault as exc:
                payload = _scalar_payload(
                    0.0,
                    {"return_shape": "agent_fault", "agent_fault": str(exc)},
                )
                payload["env_internal_failure"] = False
                payload["env_internal_failure_logs"] = None
                _write_payload(result_path, payload)
                return
            payload = _normalized_payload(result)
            _write_payload(result_path, payload)
        except Exception:
            traceback.print_exc()
            sys.exit(1)

    _main()
    """
)


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


def _stage_result_file() -> str | None:
    """Create the private file used for the runner's authoritative grade."""
    result_dir: str | None = None
    result_path: str | None = None
    fd: int | None = None
    try:
        result_dir = tempfile.mkdtemp(prefix=_RESULT_DIR_PREFIX)
        os.chmod(result_dir, 0o700)
        fd, result_path = tempfile.mkstemp(
            prefix="result-", suffix=".json", dir=result_dir
        )
        os.close(fd)
        fd = None
        os.chmod(result_path, 0o600)
        return result_path
    except OSError:
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
        return None


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
    test_file_source: str, transcript: str = "", timeout_s: float | None = None
) -> Grade:
    runner_env = prepare_grader_cache()
    transcript_path = _stage_transcript(transcript)
    result_path = _stage_result_file()
    timeout = _evaluate_timeout(timeout_s)
    if transcript_path:
        runner_env["LBX_AGENT_TRANSCRIPT_PATH"] = transcript_path
    if not result_path:
        _unlink_if_present(transcript_path)
        return _failure_grade({}, "could not create private rubric result file")
    runner_env[_RESULT_PATH_ENV] = result_path
    metadata: dict[str, Any] = {}
    try:
        metadata["pre_grade_cleanup"] = pre_grade_cleanup(OUTPUT_DIR)
        # Cut the agent's grade-time access to the live hidden env before the
        # grader subprocess runs (closes free reset()-seed fingerprinting, a
        # twin-env oracle, private-method probes). No-op for static tasks.
        try:
            from env_server.supervisor import stop_env_server

            stop_env_server()
        except Exception as exc:  # noqa: BLE001 - never block grading
            print(f"[ENV_SERVER] stop_env_server failed: {exc}",
                  file=sys.stderr, flush=True)
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
            _mirror_runner_output(stdout, stderr, metadata)
            message = f"test_file subprocess timed out after {timeout:.3f}s"
            return _failure_grade(
                metadata,
                message,
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
        _mirror_runner_output(stdout, stderr, metadata)
        if proc.returncode:
            classification = classify_failure(
                proc.returncode, metadata.get("stderr", "")
            )
            message = classification.reason
            return _failure_grade(
                metadata,
                message,
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
        payload = _read_result_payload(result_path, metadata)
        if payload is None:
            message = metadata.get("error", "missing rubric result")
            return _failure_grade(
                metadata,
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
        try:
            grade = _grade_from_payload(payload)
        except Exception as exc:
            message = f"could not normalize rubric result: {type(exc).__name__}: {exc}"
            return _failure_grade(
                metadata,
                message,
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
        grade.metadata = {**metadata, **(grade.metadata or {})}
        return grade
    finally:
        _unlink_if_present(transcript_path)
        _cleanup_result_file(result_path)
        # Best-effort: free any leftover /dev/nvidia* fd holders so the
        # end-of-container checkpoint can save. Runs after the grade is computed
        # and must never change the score or fail grading.
        try:
            kill_nvproxy_fd_holders()
        except Exception as exc:  # noqa: BLE001 - cleanup must never fail grading
            print(f"[GRADING] nvproxy sweep failed (ignored): {exc}",
                  file=sys.stderr, flush=True)


@mcp.tool()
async def grade_problem(
    problem_id: str,
    transcript: str = Field(description="The full transcript produced by the model"),
    extra_fields: dict = None,
) -> Grade:
    """Grade by executing the image-baked grader through the test_file shim."""
    _ = problem_id
    fields = _extra(extra_fields)
    test_file = fields.get("test_file")
    if not test_file:
        return Grade(
            subscores={"score": 0.0},
            weights={"score": 1.0},
            metadata={"error": "missing test_file"},
        )
    timeout_s = _coerce_timeout(fields.get("grading_timeout_seconds"))
    return _evaluate(str(test_file), transcript=transcript or "", timeout_s=timeout_s)


def main() -> None:
    """Run MCP server."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORKDIR)
    # Hidden-environment tasks ([environment].hidden_env = env/hybrid) spawn an
    # env_server subprocess the agent reaches over /tmp/env.sock. A no-op for
    # static tasks. Best-effort: a failure to import/launch must never block a
    # normal task's MCP server from starting.
    try:
        from env_server.supervisor import supervise_if_enabled

        supervise_if_enabled()
    except Exception as exc:  # noqa: BLE001 - env server is opt-in; never block startup
        # stderr only: stdio transport owns fd 1 for JSON-RPC framing.
        print(f"[ENV_SERVER] supervisor failed to start; continuing: {exc}",
              file=sys.stderr, flush=True)
    mcp.run(transport="stdio")
