from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from alignerr_plugin.proof import PROOF_PATH, update_build_proof_auto_qa
from alignerr_plugin.proof import update_build_proof_result
from alignerr_plugin.utils import read_json, write_json
from alignerr_plugin.validators.task.validator import TaskValidator

from lbx_rl_tasks_harness.auto_qa import run_auto_qa_check
from lbx_rl_tasks_harness.grading import grade_workspace
from lbx_rl_tasks_harness.ground_truth import (
    commit_render_outputs,
    require_expected_ground_truth,
    require_zero_anchored_noop,
    run_ground_truth_render,
    write_ground_truth_artifacts,
)
from alignerr_plugin.ground_truth import render_expected
from lbx_rl_tasks_harness.models import HarnessProblem, HarnessResult, RuntimeName
from lbx_rl_tasks_harness.rubric_quality import run_rubric_quality_check
from lbx_rl_tasks_harness.reference_config import resolve_reference_execution
from lbx_rl_tasks_harness.runtimes.claude_code import (
    effective_model_name as effective_claude_code_model,
    run_claude_code,
)
from lbx_rl_tasks_harness.runtimes.deepagents import run_deepagents
from lbx_rl_tasks_harness.runtimes.reference import (
    ReferenceRunOptions,
    grade_workspace_in_container,
    run_reference,
)
from lbx_rl_tasks_harness.trajectory import (
    print_auto_qa_review,
    print_rubric_quality_review,
    StreamingAutoQaRenderer,
)


def _prepare_run_dir(base: Path, problem: HarnessProblem) -> Path:
    run_dir = (
        base / f"{problem.id}-{problem.source_format}-{int(time.time())}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_manifest(
    run_dir: Path,
    problem: HarnessProblem,
    runtime: RuntimeName,
    score: float | None,
    model: str | None = None,
    review_artifacts: list[dict[str, Any]] | None = None,
    flavor_requested: str = "auto",
) -> None:
    manifest = {
        "problem_id": problem.id,
        "source_format": problem.source_format,
        "runtime": runtime,
        "score": score,
        "model": model,
        "image": problem.image,
        "required_tools": problem.required_tools,
        "required_resources": problem.required_resources,
        "review_artifacts": review_artifacts or [],
        "metadata": problem.metadata,
    }
    manifest.update(_flavor_manifest_fields(problem, flavor_requested))
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )


def _flavor_manifest_fields(
    problem: HarnessProblem, flavor_requested: str
) -> dict[str, Any]:
    """Flavor telemetry from the build sidecar; falls back to the request alone."""
    from lbx_rl_tasks_harness.docker import read_flavor_sidecar

    fields: dict[str, Any] = {
        "flavor_requested": flavor_requested,
        "image_flavor": "heavy",
        "flavor_fallback": False,
    }
    if problem.source_problem_dir is not None:
        recorded = read_flavor_sidecar(problem.source_problem_dir.resolve())
        if recorded:
            fields["image_flavor"] = recorded.get("image_flavor", "heavy")
            fields["flavor_fallback"] = bool(recorded.get("flavor_fallback", False))
    return fields


def _score_from_mcp_grade(payload: dict) -> float:
    subscores = payload.get("subscores") if isinstance(payload, dict) else None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if isinstance(subscores, dict) and "score" in subscores:
        return float(subscores["score"])
    if isinstance(metadata, dict) and "score" in metadata:
        return float(metadata["score"])
    if isinstance(payload, dict) and "score" in payload:
        return float(payload["score"])
    return 0.0


def _write_reward_outputs(verifier_dir: Path, payload: dict, score: float) -> None:
    verifier_dir.mkdir(parents=True, exist_ok=True)
    reward = {"score": score}
    subscores = payload.get("subscores") if isinstance(payload, dict) else None
    if isinstance(subscores, dict):
        for key, value in subscores.items():
            if key != "score":
                reward[str(key)] = float(value)
    (verifier_dir / "reward.json").write_text(json.dumps(reward, indent=2) + "\n")
    (verifier_dir / "reward-details.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    (verifier_dir / "reward.txt").write_text(f"{score:.6f}\n")


def _read_grade_payload(details_path: Path, reward_path: Path, score: float) -> dict:
    for path in (details_path, reward_path):
        if path.exists():
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                return payload
    return {"score": score}


def _ensure_current_build_proof(problem: HarnessProblem) -> None:
    """Refresh Docker build proof before writing source-dir proof results."""
    if problem.source_problem_dir is None:
        return
    if not (problem.source_problem_dir / "task.toml").exists():
        return

    result = TaskValidator()._local_build_proof(problem.source_problem_dir)
    if not result.passed:
        raise RuntimeError("; ".join(result.issues) or "could not refresh build proof")


def _skip_ground_truth_render() -> bool:
    return os.environ.get("LBX_RL_SKIP_GROUND_TRUTH_RENDER", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _existing_ground_truth_review_artifacts(
    problem: HarnessProblem,
) -> list[dict[str, Any]]:
    if problem.source_problem_dir is None:
        return []
    proof_path = problem.source_problem_dir / PROOF_PATH
    if not proof_path.exists():
        return []
    proof = read_json(proof_path)
    result = proof.get("ground_truth_result")
    if not isinstance(result, dict):
        return []
    artifacts = result.get("review_artifacts")
    if not isinstance(artifacts, list):
        return []
    return [artifact for artifact in artifacts if isinstance(artifact, dict)]


def _update_build_proof_result(
    problem: HarnessProblem,
    *,
    runtime: RuntimeName,
    grade_payload: dict,
    run_dir: Path,
    reward_path: Path,
    details_path: Path,
    rubric_quality: dict | None = None,
    review_artifacts: list[dict[str, Any]] | None = None,
    result_key: str = "harness_result",
    trivial_baseline_score: float | None = None,
) -> None:
    if problem.source_problem_dir is None:
        return
    update_build_proof_result(
        problem.source_problem_dir,
        runtime=runtime,
        grade_payload=grade_payload,
        run_dir=run_dir,
        reward_path=reward_path,
        details_path=details_path,
        rubric_quality=rubric_quality,
        review_artifacts=review_artifacts,
        result_key=result_key,
        trivial_baseline_score=trivial_baseline_score,
    )


def _grade_noop_baseline_in_container(problem: HarnessProblem, run_dir: Path) -> float:
    """Grade an EMPTY workspace inside the task image (zero-anchor probe).

    In-container / ML_Envs graders cannot be no-op-probed on the host by the
    task validator, so the ground-truth run grades an empty submission here and
    records the result as ``trivial_baseline_score`` in the build proof, where
    the validator enforces the anti-reward-hacking ceiling (and, for continuous
    scoring functions, the strict zero anchor).
    """
    noop_dir = run_dir / "noop"
    noop_workspace = noop_dir / "workspace"
    noop_verifier = noop_dir / "verifier"
    noop_transcript = noop_dir / "transcript.txt"
    noop_workspace.mkdir(parents=True)
    return grade_workspace_in_container(
        problem, noop_workspace, noop_verifier, noop_transcript
    )


def _run_rubric_quality_check(
    problem: HarnessProblem, *, grade_payload: dict, model: str | None = None
) -> dict:
    return asyncio.run(
        run_rubric_quality_check(problem, grade_payload=grade_payload, model_name=model)
    )


def _grade_payload_from_harness_result(result: dict) -> dict:
    if "score" not in result:
        raise ValueError("build proof harness_result is missing score")
    grade_payload: dict[str, object] = {"score": float(result["score"])}
    for key in ("subscores", "weights", "structured_subscores", "metadata"):
        if key in result:
            grade_payload[key] = result[key]
    return grade_payload


def run_auto_qa_only(
    problem: HarnessProblem,
    *,
    proof_path: Path | None = None,
    output_proof_path: Path | None = None,
    output_json: Path | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    require: bool = False,
) -> dict:
    if problem.source_problem_dir is None:
        raise ValueError("Auto QA requires an authoring problem directory")

    source_proof_path = proof_path or problem.source_problem_dir / PROOF_PATH
    if not source_proof_path.exists():
        raise ValueError(f"build proof not found at {source_proof_path}")

    proof = read_json(source_proof_path)
    harness_result = proof.get("harness_result")
    if not isinstance(harness_result, dict):
        raise ValueError("build proof is missing harness_result")
    rubric_quality = harness_result.get("rubric_quality")
    if not isinstance(rubric_quality, dict):
        raise ValueError("build proof is missing harness_result.rubric_quality")

    grade_payload = _grade_payload_from_harness_result(harness_result)
    renderer = StreamingAutoQaRenderer()
    auto_qa = asyncio.run(
        run_auto_qa_check(
            problem,
            proof=proof,
            grade_payload=grade_payload,
            rubric_quality=rubric_quality,
            model_name=model,
            reasoning_effort=reasoning_effort,
            stream_callback=renderer.print_chunk,
        )
    )
    renderer.finish()
    print_auto_qa_review(auto_qa)

    destination = update_build_proof_auto_qa(
        problem.source_problem_dir,
        auto_qa=auto_qa,
        proof_path=source_proof_path,
        output_path=output_proof_path,
    )
    if destination is None:
        raise ValueError(f"build proof not found at {source_proof_path}")

    if output_json is not None:
        write_json(output_json, auto_qa)

    # --require ensures Auto QA actually RAN. Distinguish an INFRA failure
    # (status="error": empty / non-parseable model output, provider hiccup, or a
    # soft refusal) from a genuine inability to run. An infra error is ADVISORY —
    # it must not red-wall the task, or a single model outage/refusal fails every
    # PR (see AUTOQA-JSON); the error reason + raw excerpt are still surfaced in
    # the review and uploaded for diagnosis. A "skipped" result (missing API key
    # or no source dir) is a real misconfiguration and stays blocking. A completed
    # review — including overall_assessment="fail" — is advisory here as before.
    if require and auto_qa.get("status") == "skipped":
        reason = auto_qa.get("reason") or "Auto QA was skipped"
        raise RuntimeError(str(reason))

    return auto_qa


def run_reference_harness(
    problem: HarnessProblem,
    *,
    run_dir_base: Path,
    skip_solve: bool = False,
    no_grade: bool = False,
    clean_cache: bool = False,
    solution_dir: str = "solution",
) -> HarnessResult:
    """Author iteration loop: run reference solve/grade with optional cache."""
    run_dir = _prepare_run_dir(run_dir_base, problem)
    workspace = run_dir / "workspace"
    verifier_dir = run_dir / "verifier"
    transcript = run_dir / "transcript.txt"
    workspace.mkdir(parents=True)

    ref_result = run_reference(
        problem,
        workspace,
        verifier_dir,
        transcript,
        options=ReferenceRunOptions(
            mode="iterate",
            skip_solve=skip_solve,
            no_grade=no_grade,
            clean_cache=clean_cache,
            solution_dir=solution_dir,
        ),
        run_dir=run_dir,
    )
    score = ref_result.score
    _write_manifest(run_dir, problem, "solution", score)
    return HarnessResult(
        problem_id=problem.id,
        source_format=problem.source_format,
        runtime="solution",
        run_dir=run_dir,
        workspace=workspace,
        reward_path=verifier_dir / "reward.json",
        details_path=verifier_dir / "reward-details.json",
        score=score,
    )


def run_harness(
    problem: HarnessProblem,
    *,
    runtime: RuntimeName,
    run_dir_base: Path,
    model: str | None = None,
    max_steps: int | None = None,
    flavor: str = "auto",
) -> HarnessResult:
    run_dir = _prepare_run_dir(run_dir_base, problem)
    workspace = run_dir / ("app" if problem.source_format == "harbor" else "workspace")
    verifier_dir = run_dir / (
        "logs/verifier" if problem.source_format == "harbor" else "verifier"
    )
    transcript = run_dir / "transcript.txt"
    workspace.mkdir(parents=True)

    in_container_gt = (
        runtime == "solution"
        and resolve_reference_execution(problem, mode="prove") == "container"
    )
    existing_review_artifacts = (
        _existing_ground_truth_review_artifacts(problem)
        if runtime == "solution"
        else []
    )
    if runtime == "solution":
        _ensure_current_build_proof(problem)

    if runtime == "solution":
        render_required = render_expected(
            problem.metadata.get("difficulty", {}).get("task_type"),
            problem.ground_truth.render_outputs,
        )
        ref_result = run_reference(
            problem,
            workspace,
            verifier_dir,
            transcript,
            options=ReferenceRunOptions(
                mode="prove",
                render=render_required and bool(problem.ground_truth.render_command),
                write_manifest=False,
            ),
            run_dir=run_dir,
        )
        if ref_result.score is None:
            raise RuntimeError("ground-truth reference run did not produce a score")
        score = ref_result.score
        grade_payload = ref_result.grade_payload or {"score": score}
        reward_path = verifier_dir / "reward.json"
        details_path = verifier_dir / "reward-details.json"
        if not reward_path.exists():
            _write_reward_outputs(verifier_dir, grade_payload, score)
    elif runtime == "noop":
        transcript.write_text("[noop runtime]\n")
    elif runtime in {"claude-code", "deepagents"}:
        agent_runtime = run_claude_code if runtime == "claude-code" else run_deepagents
        manifest_model = (
            effective_claude_code_model(model) if runtime == "claude-code" else model
        )
        rubric_model = None if runtime == "claude-code" else model
        grade_payload = asyncio.run(
            agent_runtime(
                problem,
                workspace,
                transcript,
                model_name=model,
                max_steps=max_steps,
                flavor=flavor,
            )
        )
        score = _score_from_mcp_grade(grade_payload)
        _write_reward_outputs(verifier_dir, grade_payload, score)
        _write_manifest(
            run_dir, problem, runtime, score, model=manifest_model, flavor_requested=flavor
        )
        rubric_quality = _run_rubric_quality_check(
            problem, grade_payload=grade_payload, model=rubric_model
        )
        print_rubric_quality_review(rubric_quality)
        _update_build_proof_result(
            problem,
            runtime=runtime,
            grade_payload=grade_payload,
            run_dir=run_dir,
            reward_path=verifier_dir / "reward.json",
            details_path=verifier_dir / "reward-details.json",
            rubric_quality=rubric_quality,
        )
        return HarnessResult(
            problem_id=problem.id,
            source_format=problem.source_format,
            runtime=runtime,
            run_dir=run_dir,
            workspace=workspace,
            reward_path=verifier_dir / "reward.json",
            details_path=verifier_dir / "reward-details.json",
            score=score,
        )
    else:
        raise ValueError(f"unsupported runtime: {runtime}")

    if runtime == "solution":
        review_artifacts: list[dict[str, Any]] = []
        difficulty = problem.metadata.get("difficulty", {})
        require_expected_ground_truth(
            grade_payload,
            score=score,
            reward_type=str(difficulty.get("reward_type") or ""),
            deterministic_epsilon=problem.ground_truth.score_epsilon,
            continuous_epsilon=problem.ground_truth.continuous_score_epsilon,
        )
        # Zero-anchoring probe: in-container / ML_Envs graders cannot be
        # no-op-probed on the host, so grade an empty submission in the task
        # image here, enforce the anchor for continuous scoring functions, and
        # record the score in the build proof for the task validator.
        trivial_baseline_score: float | None = None
        if in_container_gt:
            trivial_baseline_score = _grade_noop_baseline_in_container(
                problem, run_dir
            )
            require_zero_anchored_noop(
                trivial_baseline_score,
                reward_type=str(difficulty.get("reward_type") or ""),
                zero_anchor_epsilon=problem.ground_truth.zero_anchor_epsilon,
            )
        if render_required:
            if in_container_gt:
                review_artifacts = commit_render_outputs(
                    problem, workspace=workspace, run_dir=run_dir
                )
                write_ground_truth_artifacts(run_dir, review_artifacts)
            elif _skip_ground_truth_render():
                review_artifacts = existing_review_artifacts
                if not review_artifacts:
                    raise RuntimeError(
                        "LBX_RL_SKIP_GROUND_TRUTH_RENDER is set, but the existing "
                        "build proof has no ground_truth_result.review_artifacts to preserve"
                    )
            else:
                review_artifacts = run_ground_truth_render(
                    problem,
                    workspace=workspace,
                    run_dir=run_dir,
                    transcript_path=transcript,
                )
                write_ground_truth_artifacts(run_dir, review_artifacts)
        _write_manifest(
            run_dir,
            problem,
            runtime,
            score,
            model=model,
            review_artifacts=review_artifacts,
            flavor_requested=flavor,
        )
        _update_build_proof_result(
            problem,
            runtime=runtime,
            grade_payload=grade_payload,
            run_dir=run_dir,
            reward_path=verifier_dir / "reward.json",
            details_path=verifier_dir / "reward-details.json",
            review_artifacts=review_artifacts,
            result_key="ground_truth_result",
            trivial_baseline_score=trivial_baseline_score,
        )
        return HarnessResult(
            problem_id=problem.id,
            source_format=problem.source_format,
            runtime=runtime,
            run_dir=run_dir,
            workspace=workspace,
            reward_path=verifier_dir / "reward.json",
            details_path=verifier_dir / "reward-details.json",
            score=score,
            review_artifacts=review_artifacts,
        )

    # Grade in the task image when the task resolves to container execution
    # (ml / accelerator / in_container) so the scorer has the task's own image
    # deps and the root-only held-out truth, mirroring ground-truth and Taiga.
    # Host-resolved tasks keep grading in-process on the host.
    if resolve_reference_execution(problem) == "container":
        score = grade_workspace_in_container(
            problem, workspace, verifier_dir, transcript
        )
    else:
        score = grade_workspace(problem, workspace, verifier_dir, transcript)
    reward_path = verifier_dir / "reward.json"
    details_path = verifier_dir / "reward-details.json"
    grade_payload = _read_grade_payload(details_path, reward_path, score)
    rubric_quality = _run_rubric_quality_check(
        problem, grade_payload=grade_payload, model=model
    )
    print_rubric_quality_review(rubric_quality)
    _write_manifest(
        run_dir, problem, runtime, score, model=model, flavor_requested=flavor
    )
    _update_build_proof_result(
        problem,
        runtime=runtime,
        grade_payload=grade_payload,
        run_dir=run_dir,
        reward_path=reward_path,
        details_path=details_path,
        rubric_quality=rubric_quality,
    )
    return HarnessResult(
        problem_id=problem.id,
        source_format=problem.source_format,
        runtime=runtime,
        run_dir=run_dir,
        workspace=workspace,
        reward_path=reward_path,
        details_path=details_path,
        score=score,
    )
