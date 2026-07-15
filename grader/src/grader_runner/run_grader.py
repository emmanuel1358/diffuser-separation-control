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
import subprocess
import sys
import tempfile
from pathlib import Path

from grading import Grade
from grading.runtime_hardening import (
    classify_failure,
    kill_nvproxy_fd_holders,
    lock_down_grader_private,
    pre_grade_cleanup,
    prepare_grader_cache,
    scrub_escaping_symlinks,
    scrub_nonregular_files,
)

# Some graders read the submission from the canonical /tmp/output regardless of
# --workspace, so the symlink / non-regular scrub must cover it too.
_MLENVS_OUTPUT_DIR = Path("/tmp/output")

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


def _emit_failure(
    output_dir: Path,
    *,
    error_type: str,
    message: str,
    tb: str = "",
    env_internal_failure: bool | None = True,
    env_internal_failure_logs: list[str] | None = None,
) -> None:
    """Write a zero-score reward + details payload describing the failure.

    Populates `criterion_logs` (not just metadata) so `Grade.to_dict()`'s
    canonical ``metadata.grading_errors`` collector picks the failure up
    and surfaces it in both the Boreal UI and Harbor's reward-details.json.
    """
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
        metadata={"return_shape": "error"},
        env_internal_failure=env_internal_failure,
        env_internal_failure_logs=env_internal_failure_logs,
    )
    reward, details, txt = _grade_to_payloads(grade)
    if tb:
        details["metadata"]["traceback"] = tb
    _write_outputs(output_dir, reward, details, txt)


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
    output_dir.mkdir(parents=True, exist_ok=True)
    fd, raw_result_path = tempfile.mkstemp(
        prefix="lbx-harbor-grader-", suffix=".json", dir=output_dir
    )
    os.close(fd)
    result_path = Path(raw_result_path)
    try:
        try:
            if os.name == "posix" and os.geteuid() == 0:
                lock_down_grader_private((grader_dir, private))
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
                _MLENVS_OUTPUT_DIR.exists()
                and _MLENVS_OUTPUT_DIR.resolve() != Path(workspace).resolve()
            ):
                out_symlinks = scrub_escaping_symlinks(_MLENVS_OUTPUT_DIR)
                out_nonregular = scrub_nonregular_files(_MLENVS_OUTPUT_DIR)
                cleanup["removed_symlinks"] = (
                    cleanup.get("removed_symlinks", 0) + out_symlinks
                )
                cleanup["removed_nonregular"] = (
                    cleanup.get("removed_nonregular", 0) + out_nonregular
                )
            # Cut the agent's grade-time access to the hidden env before the
            # grader worker runs. No-op for static tasks.
            try:
                from env_server.supervisor import stop_env_server

                stop_env_server()
            except Exception as exc:  # noqa: BLE001 - never block grading
                print(f"[ENV_SERVER] stop_env_server failed: {exc}", file=sys.stderr)
            env = prepare_grader_cache()
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
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            message = f"grader subprocess timed out after {timeout_s:.3f}s"
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
                env_internal_failure=True,
                env_internal_failure_logs=[message],
            )
            return 1

        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)

        if proc.returncode:
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
        metadata.setdefault("pre_grade_cleanup", cleanup)
        reward, details, txt = _reward_from_details(details)
        _write_outputs(output_dir, reward, details, txt)
        logger.info("wrote reward score=%.4f to %s", reward["score"], output_dir)
        return 0
    finally:
        try:
            result_path.unlink()
        except OSError:
            pass
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
