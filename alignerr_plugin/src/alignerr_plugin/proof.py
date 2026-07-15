"""Local Docker build proof helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alignerr_plugin.utils import grading_inputs_sha256, read_json, write_json

PROOF_PATH = Path(".alignerr/build_proof.json")


def write_build_proof(
    problem_dir: Path,
    *,
    image_digest: str,
    base_image_ref: str,
    platform: str,
    alignerr_cli_version: str,
    duration_seconds: float,
) -> Path:
    """Persist proof that the current task source built locally.

    Preserves an existing ``ground_truth_result`` / ``harness_result`` across a
    rebuild only when the rebuild is identical — same ``task_dir_sha256`` **and**
    same ``image_digest`` **and** same ``base_image_ref`` — so a re-run (or a run
    that fails after the build) does not drop previously committed grading results
    / reviewer artifacts. If the task files, the task image, or the base image
    changed, the prior results were graded against a different image and are
    intentionally dropped (so validation can never pass on stale oracle results).

    The ``task_dir_sha256`` field holds the *grading-inputs* hash
    (``grading_inputs_sha256``): only ``task.toml`` + ``solution/`` + ``scorer/``
    + ``data_generation/`` + ``environment/`` count, so doc-only edits do not drop
    a still-valid result.
    """
    path = problem_dir / PROOF_PATH
    task_hash = grading_inputs_sha256(problem_dir)
    preserved: dict[str, Any] = {}
    if path.exists():
        try:
            existing = read_json(path)
        except Exception:  # noqa: BLE001
            existing = {}
        identical_build = (
            existing.get("task_dir_sha256") == task_hash
            and existing.get("image_digest") == image_digest
            and existing.get("base_image_ref") == base_image_ref
        )
        if identical_build:
            for key in ("ground_truth_result", "harness_result"):
                if key in existing:
                    preserved[key] = existing[key]
    proof = {
        "schema_version": "1.0",
        "task_dir_sha256": task_hash,
        "image_digest": image_digest,
        "base_image_ref": base_image_ref,
        "platform": platform,
        "built_at": datetime.now(UTC).isoformat(),
        "alignerr_cli_version": alignerr_cli_version,
        "duration_seconds": duration_seconds,
        **preserved,
    }
    write_json(path, proof)
    return path


def update_build_proof_result(
    problem_dir: Path,
    *,
    runtime: str,
    grade_payload: dict[str, Any],
    run_dir: Path,
    reward_path: Path,
    details_path: Path,
    rubric_quality: dict[str, Any] | None = None,
    review_artifacts: list[dict[str, Any]] | None = None,
    result_key: str = "harness_result",
    trivial_baseline_score: float | None = None,
) -> Path | None:
    """Attach the latest local harness score to an existing build proof."""
    proof_path = problem_dir / PROOF_PATH
    if not proof_path.exists():
        return None

    proof = read_json(proof_path)
    proof[result_key] = {
        "runtime": runtime,
        "graded_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "reward_path": str(reward_path),
        "details_path": str(details_path),
        **grade_summary(grade_payload),
    }
    if rubric_quality is not None:
        proof[result_key]["rubric_quality"] = rubric_quality
    if review_artifacts:
        proof[result_key]["review_artifacts"] = review_artifacts
    if trivial_baseline_score is not None:
        proof[result_key]["trivial_baseline_score"] = float(trivial_baseline_score)
    write_json(proof_path, proof)
    return proof_path


def update_build_proof_auto_qa(
    problem_dir: Path,
    *,
    auto_qa: dict[str, Any],
    proof_path: Path | None = None,
    output_path: Path | None = None,
) -> Path | None:
    """Attach Auto QA to an existing proof without recomputing other result fields."""
    source_path = proof_path or problem_dir / PROOF_PATH
    if not source_path.exists():
        return None

    proof = read_json(source_path)
    result = proof.get("harness_result")
    if not isinstance(result, dict):
        raise ValueError("build proof is missing harness_result")
    result["auto_qa"] = auto_qa
    destination = output_path or source_path
    write_json(destination, proof)
    return destination


def grade_summary(grade_payload: dict[str, Any]) -> dict[str, Any]:
    """Extract stable headline and rubric details from a grade payload."""
    score = _score_from_grade_payload(grade_payload)
    summary: dict[str, Any] = {"score": score}

    subscores = grade_payload.get("subscores")
    if isinstance(subscores, dict):
        summary["subscores"] = {
            str(key): float(value) for key, value in subscores.items()
        }

    weights = grade_payload.get("weights")
    if isinstance(weights, dict):
        summary["weights"] = {str(key): float(value) for key, value in weights.items()}

    structured = grade_payload.get("structured_subscores")
    if isinstance(structured, list):
        summary["structured_subscores"] = structured

    metadata = grade_payload.get("metadata")
    if isinstance(metadata, dict):
        summary["metadata"] = metadata

    return summary


def _score_from_grade_payload(grade_payload: dict[str, Any]) -> float:
    subscores = grade_payload.get("subscores")
    metadata = grade_payload.get("metadata")
    if isinstance(subscores, dict) and "score" in subscores:
        return float(subscores["score"])
    if isinstance(metadata, dict):
        for key in ("headline_score", "reported_final_score", "score"):
            if key in metadata:
                return float(metadata[key])
    if "score" in grade_payload:
        return float(grade_payload["score"])
    return 0.0


def verify_build_proof(
    problem_dir: Path,
    *,
    max_age_days: int = 7,
    min_cli_version: str | None = None,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    """Verify the local build proof for a task directory."""
    proof_path = problem_dir / PROOF_PATH
    if not proof_path.exists():
        return False, [f"missing build proof at {proof_path}"], None

    proof = read_json(proof_path)
    errors: list[str] = []

    current_hash = grading_inputs_sha256(problem_dir)
    if proof.get("task_dir_sha256") != current_hash:
        errors.append(
            "build proof is stale: grading inputs (task.toml, solution/, scorer/, "
            "data_generation/, environment/) changed after the last successful local build"
        )

    built_at_raw = proof.get("built_at")
    if not isinstance(built_at_raw, str):
        errors.append("build proof is missing built_at")
    else:
        built_at = datetime.fromisoformat(built_at_raw)
        if datetime.now(UTC) - built_at > timedelta(days=max_age_days):
            errors.append(f"build proof is older than {max_age_days} days")

    if min_cli_version and str(proof.get("alignerr_cli_version", "")) < min_cli_version:
        errors.append(
            f"alignerr_cli_version is older than required minimum {min_cli_version}"
        )

    return not errors, errors, proof
