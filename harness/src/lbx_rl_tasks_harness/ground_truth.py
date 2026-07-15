from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from alignerr_plugin.ground_truth import (
    SUBMISSION_ARTIFACT_DIR,
    VIDEO_SUFFIXES,
    artifact_metadata,
    expected_ground_truth_score,
    failed_criteria,
    render_expected,
    validate_video_file,
)

from lbx_rl_tasks_harness.env import harness_subprocess_env
from lbx_rl_tasks_harness.models import HarnessProblem, OutputSpec


def require_expected_ground_truth(
    grade_payload: dict,
    *,
    score: float,
    reward_type: str,
    deterministic_epsilon: float,
    continuous_epsilon: float,
) -> None:
    """Require the oracle/reference solution to match its reward-type target."""
    expectation = expected_ground_truth_score(
        reward_type,
        deterministic_epsilon=deterministic_epsilon,
        continuous_epsilon=continuous_epsilon,
    )
    if expectation.passed(score):
        return

    details = failed_criteria(grade_payload)
    detail_text = "; ".join(details[:10])
    if len(details) > 10:
        detail_text += f"; {len(details) - 10} more failed criteria"
    if detail_text:
        detail_text = f" Failed criteria: {detail_text}."
    raise RuntimeError(
        "ground truth solution for reward_type "
        f"{expectation.reward_type!r} must score {expectation.description}, "
        f"got {score:.6f}.{detail_text}"
    )


def require_zero_anchored_noop(
    noop_score: float,
    *,
    reward_type: str,
    zero_anchor_epsilon: float,
) -> None:
    """Require continuous scoring functions to map "no attempt" to a score of 0.

    Only ``continuous_scoring_function`` tasks are held to the strict zero
    anchor; rubric tasks keep the ``max_trivial_score`` ceiling enforced by the
    task validator.
    """
    if reward_type != "continuous_scoring_function":
        return
    if noop_score <= zero_anchor_epsilon + 1e-9:
        return
    raise RuntimeError(
        "continuous_scoring_function tasks must anchor an empty submission to "
        f"0; a no-op submission scores {noop_score:.6f}, above the zero-anchor "
        f"tolerance ({zero_anchor_epsilon:.3f}). Set the FLOOR calibration "
        "anchor / AgentFault path so no attempt yields 0, or adjust "
        "[ground_truth].zero_anchor_epsilon only with justification."
    )


def validate_render_contract(problem: HarnessProblem) -> list[str]:
    """Validate that a task declares the required ground-truth video artifact."""
    issues: list[str] = []
    if not render_expected(
        problem.metadata.get("difficulty", {}).get("task_type"),
        problem.ground_truth.render_outputs,
    ):
        return issues
    spec = problem.ground_truth
    if not spec.render_command.strip():
        issues.append("task.toml [ground_truth].render_command is required")
    if not spec.render_outputs:
        issues.append(
            "task.toml [ground_truth].render_outputs must declare at least one video"
        )
    for output in spec.render_outputs:
        issues.extend(_validate_video_output_spec(output))
    return issues


def run_ground_truth_render(
    problem: HarnessProblem,
    *,
    workspace: Path,
    run_dir: Path,
    transcript_path: Path,
) -> list[dict[str, Any]]:
    """Run the task-owned render hook and copy reviewer videos into the run dir."""
    issues = validate_render_contract(problem)
    if issues:
        raise RuntimeError("; ".join(issues))
    if problem.source_problem_dir is None:
        raise ValueError("ground-truth rendering requires a source problem directory")

    env = harness_subprocess_env({"LBT_OUTPUT_DIR": str(workspace)})
    timeout_sec = problem.metadata.get("agent", {}).get("timeout_sec") or 3600
    proc = subprocess.run(
        ["bash", "-lc", problem.ground_truth.render_command],
        cwd=problem.source_problem_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    with transcript_path.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    "[ground truth render stdout]",
                    proc.stdout,
                    "[ground truth render stderr]",
                    proc.stderr,
                    "",
                ]
            )
        )
    if proc.returncode != 0:
        raise RuntimeError(
            render_failure_details(proc.stdout, proc.stderr, proc.returncode)
        )

    return commit_render_outputs(problem, workspace=workspace, run_dir=run_dir)


def commit_render_outputs(
    problem: HarnessProblem, *, workspace: Path, run_dir: Path
) -> list[dict[str, Any]]:
    """Validate + copy already-rendered reviewer videos from ``workspace`` into
    the run's review dir and the committed ``.alignerr/ground_truth`` dir.

    Shared by the host render path and the in-container ground-truth path (where
    the video was produced inside the task image and copied back to ``workspace``).
    """
    if problem.source_problem_dir is None:
        raise ValueError(
            "committing render outputs requires a source problem directory"
        )
    review_dir = run_dir / "review"
    submission_dir = problem.source_problem_dir / SUBMISSION_ARTIFACT_DIR
    copied: list[dict[str, Any]] = []
    for output in problem.ground_truth.render_outputs:
        source = workspace / output.relative_path
        validate_video_file(source, output.path)
        run_destination = review_dir / output.relative_path
        run_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, run_destination)

        submission_destination = submission_dir / output.relative_path
        submission_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, submission_destination)
        copied.append(
            artifact_metadata(
                problem.source_problem_dir,
                submission_destination,
                logical_path=output.path,
            )
        )
    return copied


def render_failure_details(
    stdout: str,
    stderr: str,
    returncode: int | None = None,
) -> str:
    prefix = (
        f"ground-truth render command exited with status {returncode}"
        if returncode is not None
        else "ground-truth render command failed"
    )
    return "\n".join(
        [
            prefix,
            "",
            "stdout:",
            stdout.rstrip() or "<empty>",
            "",
            "stderr:",
            stderr.rstrip() or "<empty>",
        ]
    )


def write_ground_truth_artifacts(
    run_dir: Path, artifacts: list[dict[str, Any]]
) -> None:
    payload = {"review_artifacts": artifacts}
    (run_dir / "ground_truth.json").write_text(json.dumps(payload, indent=2) + "\n")


def _validate_video_output_spec(output: OutputSpec) -> list[str]:
    issues: list[str] = []
    try:
        relative = output.relative_path
    except ValueError as exc:
        issues.append(str(exc))
        return issues
    if relative.suffix.lower() not in VIDEO_SUFFIXES:
        issues.append(
            f"ground truth render output must be a video file ({', '.join(sorted(VIDEO_SUFFIXES))}): "
            f"{output.path}"
        )
    if relative.name != "rendering.mp4":
        issues.append(
            "ground truth render output must be named /tmp/output/rendering.mp4"
        )
    if not output.required:
        issues.append(f"ground truth render output must be required: {output.path}")
    return issues
