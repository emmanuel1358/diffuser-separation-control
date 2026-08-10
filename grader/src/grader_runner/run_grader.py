"""Harbor-side grader runner.

Run with::

    /runtime/run_grader.py --workspace /app --grader-dir /grader \
        --output-dir /logs/verifier [--transcript /logs/agent/transcript.txt]

Imports the image-baked ``compute_score`` from ``<grader-dir>/compute_score.py``,
calls it with ``(workspace, trajectory, private)``, normalizes the return value
into a canonical :class:`grading.Grade`, and writes:

  * ``<output-dir>/reward.json``           Harbor canonical headline
  * ``<output-dir>/reward.txt``            single float (Harbor fallback)
  * ``<output-dir>/reward-details.json``   full ``Grade.to_dict()`` (per-criterion)
  * ``<output-dir>/evaluation-details.json`` root-only private evidence trace
    (only for protected continuous evaluations)

Customer flow (RL training)::

    harbor run -p ./my-task -a my-agent -m my-model
    # then in your trainer:
    reward = json.loads("logs/verifier/reward.json")["score"]

Usable outside Harbor too::

    docker run --rm \
        -v "$PWD/agent_workspace:/app" \
        -v "$PWD/logs:/logs" \
        my-task-image \
        /runtime/run_grader.py --workspace /app --grader-dir /grader \
            --output-dir /logs/verifier
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from grading import Grade
from grading.runtime_hardening import (
    AgentProcessQuiesceError,
    ProcessQuiesceError,
    classify_failure,
    kill_nvproxy_fd_holders,
    lock_down_grader_private,
    pre_grade_cleanup,
    prepare_grader_cache,
    protect_current_process_from_oom,
    sample_resource_exhaustion,
    scrub_escaping_symlinks,
    scrub_nonregular_files,
)
from grading.secure_io import open_directory_fd, open_regular_file

# Some graders read the submission from the canonical /tmp/output regardless of
# --workspace, so the symlink / non-regular scrub must cover it too.
_AGENT_OUTPUT_DIR = Path("/tmp/output")
_GRADING_TMPDIR = tempfile.gettempdir()
_RESOURCE_LABELS = {
    "disk_exhausted": "the grading filesystem",
    "shared_memory_exhausted": "shared memory",
}

logger = logging.getLogger("run_grader")


def _grade_to_payloads(grade: Grade) -> tuple[dict, dict, str]:
    """Build (reward.json, reward-details.json, reward.txt) payloads.

    `reward.json` carries the Harbor canonical headline ``{"score": <float>}``
    plus, when criteria exist, flat per-criterion entries
    ``{<criterion_id>: <subscore>, ...}`` so multi-reward consumers
    (rewardkit-style) work too.

    `reward-details.json` is the full :func:`Grade.to_dict` for rich UI rendering.

    `reward.txt` is the single float Harbor reads as the primary fallback.
    """
    details = grade.to_dict()
    headline = float(details["score"])

    reward: dict[str, float] = {"score": headline}
    for sub in details.get("structured_subscores", []) or []:
        name = sub.get("name")
        if isinstance(name, str) and name and name != "score":
            reward[name] = float(sub.get("score", 0.0))

    return reward, details, f"{headline:.6f}\n"


def _write_outputs(output_dir: Path, reward: dict, details: dict, txt: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reward.json").write_text(json.dumps(reward, indent=2) + "\n")
    (output_dir / "reward-details.json").write_text(
        json.dumps(details, indent=2, default=str) + "\n"
    )
    (output_dir / "reward.txt").write_text(txt)


def _persist_evaluation_trace(source: Path, output_dir: Path) -> None:
    directory_fd = open_directory_fd(output_dir)
    destination_fd = -1
    try:
        try:
            os.unlink("evaluation-details.json", dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        with open_regular_file(source, max_bytes=None) as (source_handle, _info):
            destination_fd = os.open(
                "evaluation-details.json",
                flags,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(destination_fd, "wb", closefd=True) as destination:
                destination_fd = -1
                while chunk := source_handle.read(1024 * 1024):
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(directory_fd)


def _clear_evaluation_trace(output_dir: Path) -> None:
    directory_fd = open_directory_fd(output_dir)
    try:
        try:
            os.unlink("evaluation-details.json", dir_fd=directory_fd)
        except FileNotFoundError:
            pass
    finally:
        os.close(directory_fd)


def _emit_failure(
    output_dir: Path,
    *,
    error_type: str,
    message: str,
    tb: str = "",
    env_internal_failure: bool | None = True,
    env_internal_failure_logs: list[str] | None = None,
    metadata: dict | None = None,
) -> None:
    """Write a zero-score reward + details payload describing the failure.

    Populates `criterion_logs` (not just metadata) so `Grade.to_dict()`'s
    canonical ``metadata.grading_errors`` collector picks the failure up
    and surfaces it in both the Boreal UI and Harbor's reward-details.json.

    The message also goes to stderr: failures raised before the worker starts
    produce no subprocess output, so without this the only signal a caller sees
    is a bare non-zero exit and an empty stderr tail.
    """
    print(f"[grader failure] {error_type}: {message}", file=sys.stderr, flush=True)
    grade = Grade(
        subscores={"run_grader": 0.0},
        weights={"run_grader": 1.0},
        scoring_mode="weighted",
        headline_score_override=0.0,
        criterion_logs={
            "run_grader": {
                "grading_type": "runner",
                "error_type": error_type,
                "error_message": message,
                "passed": False,
                "reasoning": message,
            }
        },
        metadata={"return_shape": "error", **(metadata or {})},
        env_internal_failure=env_internal_failure,
        env_internal_failure_logs=env_internal_failure_logs,
    )
    reward, details, txt = _grade_to_payloads(grade)
    if tb:
        details["metadata"]["traceback"] = tb
    _write_outputs(output_dir, reward, details, txt)


def _emit_resource_failure(output_dir: Path, kind: str, message: str) -> int:
    _emit_failure(
        output_dir,
        error_type=kind,
        message=message,
        env_internal_failure=False,
        metadata={"agent_fault": kind},
    )
    return 0


def _emit_quiesce_failure(
    output_dir: Path,
    *,
    phase: str,
    exc: AgentProcessQuiesceError | ProcessQuiesceError,
) -> int:
    agent_fault = isinstance(exc, AgentProcessQuiesceError)
    message = f"{phase}-grade process quiesce failed: {exc}"
    _emit_failure(
        output_dir,
        error_type=(
            "agent_process_quiesce_failed"
            if agent_fault
            else "process_quiesce_infrastructure_failure"
        ),
        message=message,
        env_internal_failure=not agent_fault,
        env_internal_failure_logs=None if agent_fault else [message],
    )
    return 0 if agent_fault else 1


def _reward_from_details(details: dict) -> tuple[dict, dict, str]:
    headline = float(details["score"])
    reward: dict[str, float] = {"score": headline}
    for sub in details.get("structured_subscores", []) or []:
        name = sub.get("name")
        if isinstance(name, str) and name and name != "score":
            reward[name] = float(sub.get("score", 0.0))
    return reward, details, f"{headline:.6f}\n"


def _run_worker(
    *,
    workspace: Path,
    grader_dir: Path,
    private: Path,
    output_dir: Path,
    transcript: Path | None,
    timeout_s: float | None,
) -> int:
    protect_current_process_from_oom()
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="lbx-harbor-grade-")).resolve(
        strict=True
    )
    os.chmod(staging_dir, 0o700)
    evaluation_trace_path = staging_dir / "evaluation-details.json"
    fd, raw_result_path = tempfile.mkstemp(
        prefix="result-", suffix=".json", dir=staging_dir
    )
    os.close(fd)
    result_path = Path(raw_result_path)
    try:
        try:
            if os.name == "posix" and os.geteuid() == 0:
                # grader_dir holds CODE and may be a read-only bind mount of the
                # host scorer/. private holds held-out truth and is often
                # delivered as a read-only squashfs mount (Taiga is_read_only
                # preloaded file) -- on firecracker (the CPU-QA lane) that mount
                # is genuinely read-only, so the root ownership reset chown fails
                # with EROFS. A read-only mount already provides the
                # tamper-proofing this lockdown exists for (nobody can write it,
                # and 0700 root /mcp_server blocks the uid-1000 agent from
                # reading it), so both roots may sit on a read-only mount. The
                # relaxation is gated on the filesystem really being read-only,
                # so a writable non-root private tree still fails hard.
                lock_down_grader_private(
                    (grader_dir, private),
                    readonly_mount_ok=(grader_dir, private),
                )
            elif os.name == "posix":
                # Non-root: the private-dir lockdown AND the submitted-worker
                # privilege drop / PID-namespace sandbox all no-op (they require
                # root). Held-out truth is not hardened and agent code is not
                # sandboxed -- expected only for local dev; production grades as
                # root. Surface it so a local run is not mistaken for the real
                # isolation posture.
                logger.warning(
                    "grader running as NON-ROOT (euid=%d): private-dir lockdown and "
                    "the submitted-worker sandbox are INACTIVE (root-only). Held-out "
                    "truth is unhardened and agent code is not sandboxed -- local-dev "
                    "posture, not production.",
                    os.geteuid(),
                )
            cleanup = pre_grade_cleanup(workspace)
            # Also scrub /tmp/output when it is a distinct tree, so an agent
            # cannot exfil truth via a symlink a grader reads with plain open().
            if (
                _AGENT_OUTPUT_DIR.exists()
                and _AGENT_OUTPUT_DIR.resolve() != Path(workspace).resolve()
            ):
                out_symlinks = scrub_escaping_symlinks(_AGENT_OUTPUT_DIR)
                out_nonregular = scrub_nonregular_files(_AGENT_OUTPUT_DIR)
                cleanup["removed_symlinks"] = (
                    cleanup.get("removed_symlinks", 0) + out_symlinks
                )
                cleanup["removed_nonregular"] = (
                    cleanup.get("removed_nonregular", 0) + out_nonregular
                )
            try:
                _clear_evaluation_trace(output_dir)
            except OSError as exc:
                logger.error("refusing insecure grader output directory: %s", exc)
                return 1
            # Cut the agent's grade-time access to the hidden env before the
            # grader worker runs. No-op for static tasks.
            try:
                from env_server.supervisor import stop_env_server

                stop_env_server()
            except Exception as exc:  # noqa: BLE001 - never block grading
                print(f"[ENV_SERVER] stop_env_server failed: {exc}", file=sys.stderr)
            env = prepare_grader_cache()
            if env.get("LBX_EVALUATION_PLAN_ATTESTED") == "1":
                env.setdefault("LBX_EVALUATION_NONCE", secrets.token_hex(16))
            cmd = [
                sys.executable,
                "-P",
                "-m",
                "grader_runner.worker",
                "--workspace",
                str(workspace),
                "--grader-dir",
                str(grader_dir),
                "--private-dir",
                str(private),
                "--result-path",
                str(result_path),
                "--evaluation-trace",
                str(evaluation_trace_path),
            ]
            if transcript is not None:
                cmd.extend(["--transcript", str(transcript)])
            proc = subprocess.run(
                cmd,
                cwd="/",
                text=True,
                capture_output=True,
                env=env,
                timeout=timeout_s,
            )
        except AgentProcessQuiesceError as exc:
            return _emit_quiesce_failure(output_dir, phase="pre", exc=exc)
        except ProcessQuiesceError as exc:
            return _emit_quiesce_failure(output_dir, phase="pre", exc=exc)
        except subprocess.TimeoutExpired as exc:
            # Sampled before cleanup, which reaps the /dev/shm entries that
            # are the evidence of the exhaustion.
            exhaustion = sample_resource_exhaustion(
                exc.stderr if isinstance(exc.stderr, str) else None,
                tmpdir=_GRADING_TMPDIR,
            )
            try:
                pre_grade_cleanup(workspace)
            except AgentProcessQuiesceError as cleanup_exc:
                return _emit_quiesce_failure(
                    output_dir,
                    phase="post",
                    exc=cleanup_exc,
                )
            except ProcessQuiesceError as cleanup_exc:
                return _emit_quiesce_failure(
                    output_dir,
                    phase="post",
                    exc=cleanup_exc,
                )
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            message = f"grader subprocess timed out after {timeout_s:.3f}s"
            kind = exhaustion.agent_fault_kind()
            if kind:
                return _emit_resource_failure(
                    output_dir,
                    kind,
                    f"agent exhausted {_RESOURCE_LABELS[kind]} during grading",
                )
            _emit_failure(
                output_dir,
                error_type="grader_timeout",
                message=message,
                tb=(stdout + "\n" + stderr).strip(),
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
            return 1
        except Exception as exc:
            message = f"could not launch grader subprocess: {type(exc).__name__}: {exc}"
            _emit_failure(
                output_dir,
                error_type="grader_launch_failed",
                message=message,
                tb=traceback.format_exc(),
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
            return 1

        exhaustion = sample_resource_exhaustion(proc.stderr, tmpdir=_GRADING_TMPDIR)
        try:
            post_cleanup = pre_grade_cleanup(workspace)
        except AgentProcessQuiesceError as exc:
            return _emit_quiesce_failure(output_dir, phase="post", exc=exc)
        except ProcessQuiesceError as exc:
            return _emit_quiesce_failure(output_dir, phase="post", exc=exc)

        try:
            _clear_evaluation_trace(output_dir)
        except OSError as exc:
            logger.error(
                "could not securely publish grader outputs; refusing path: %s",
                exc,
            )
            return 1
        if evaluation_trace_path.is_file():
            try:
                _persist_evaluation_trace(evaluation_trace_path, output_dir)
            except OSError as exc:
                message = f"could not persist private evaluation trace: {exc}"
                _emit_failure(
                    output_dir,
                    error_type="evaluation_trace_persist_failed",
                    message=message,
                    env_internal_failure=True,
                    env_internal_failure_logs=[message],
                )
                return 1

        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)

        if proc.returncode:
            kind = exhaustion.agent_fault_kind(
                returncode=proc.returncode, stderr=proc.stderr
            )
            if kind:
                return _emit_resource_failure(
                    output_dir,
                    kind,
                    f"agent exhausted {_RESOURCE_LABELS[kind]} during grading",
                )
            classification = classify_failure(proc.returncode, proc.stderr[-4000:])
            message = classification.reason
            if classification.is_infra:
                _emit_failure(
                    output_dir,
                    error_type="grader_infra_failure",
                    message=message,
                    tb=proc.stderr[-4000:],
                    env_internal_failure=True,
                    env_internal_failure_logs=[message],
                )
                return proc.returncode
            try:
                details = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                details = None
            if isinstance(details, dict) and "score" in details:
                metadata = details.setdefault("metadata", {})
                metadata.setdefault("pre_grade_cleanup", cleanup)
                metadata.setdefault("post_grade_cleanup", post_cleanup)
                reward, details, txt = _reward_from_details(details)
                _write_outputs(output_dir, reward, details, txt)
                return proc.returncode
            _emit_failure(
                output_dir,
                error_type="grader_runtime_error",
                message=message,
                tb=proc.stderr[-4000:],
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
            return proc.returncode

        try:
            details = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            # An agent that fills the box can stop the grader from publishing
            # its result at all. Without this the episode is voided as
            # infrastructure and retried, which rewards the sabotage.
            kind = exhaustion.agent_fault_kind()
            if kind:
                return _emit_resource_failure(
                    output_dir,
                    kind,
                    f"agent exhausted {_RESOURCE_LABELS[kind]} before the grader "
                    "could publish its result",
                )
            message = f"could not read grader result: {type(exc).__name__}: {exc}"
            _emit_failure(
                output_dir,
                error_type="grader_result_missing",
                message=message,
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
            return 1

        metadata = details.setdefault("metadata", {})
        if details.get("env_internal_failure"):
            kind = exhaustion.agent_fault_kind()
            if kind:
                return _emit_resource_failure(
                    output_dir,
                    kind,
                    f"grader reported an internal failure while "
                    f"{_RESOURCE_LABELS[kind]} was exhausted",
                )
        metadata.setdefault("pre_grade_cleanup", cleanup)
        metadata.setdefault("post_grade_cleanup", post_cleanup)
        reward, details, txt = _reward_from_details(details)
        _write_outputs(output_dir, reward, details, txt)
        logger.info("wrote reward score=%.4f to %s", reward["score"], output_dir)
        return 0
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        # Best-effort: free leftover /dev/nvidia* fd holders so the checkpoint
        # can save. Runs after the score is written; never changes the reward.
        try:
            kill_nvproxy_fd_holders()
        except Exception as exc:  # noqa: BLE001 - cleanup must never fail the grade
            logger.warning("nvproxy sweep failed (ignored): %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the image-baked grader.")
    parser.add_argument(
        "--workspace",
        required=True,
        type=Path,
        help="Agent's workspace directory (Harbor: /app).",
    )
    parser.add_argument(
        "--grader-dir",
        required=True,
        type=Path,
        help="Directory containing compute_score.py.",
    )
    parser.add_argument(
        "--private-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding the grader's private data (passed to "
            "compute_score(..., private=...)). Defaults to <grader-dir>/data."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to write reward.json / reward-details.json / reward.txt.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Optional path to the agent's transcript (for LLM-judge criteria).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional grader subprocess timeout in seconds.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    grader_dir: Path = args.grader_dir.resolve(strict=False)
    workspace: Path = args.workspace.resolve(strict=False)
    output_dir: Path = args.output_dir.resolve(strict=False)
    private: Path = (
        args.private_dir if args.private_dir is not None else grader_dir / "data"
    ).resolve(strict=False)
    timeout_s = args.timeout
    if timeout_s is None:
        raw_timeout = os.environ.get("RUBRIC_EVALUATE_TIMEOUT_S")
        if raw_timeout:
            try:
                timeout_s = float(raw_timeout)
            except ValueError:
                timeout_s = None

    return _run_worker(
        workspace=workspace,
        grader_dir=grader_dir,
        private=private,
        output_dir=output_dir,
        transcript=args.transcript.resolve(strict=False) if args.transcript else None,
        timeout_s=timeout_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
