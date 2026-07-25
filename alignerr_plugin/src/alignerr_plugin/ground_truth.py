from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

from alignerr_plugin.task_metadata import (
    DOMAINS_BY_TASK_TYPE,
    LICENSES,
    REWARD_TYPES,
    TASK_TYPES,
    expected_ground_truth_score,
    metadata_validation_issues,
    normalize_enum_value,
    validate_domain,
    validate_license,
    validate_reward_type,
    validate_task_type,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}
SUBMISSION_ARTIFACT_DIR = Path(".alignerr") / "ground_truth"
REQUIRED_VIDEO_WIDTH = 1280
REQUIRED_VIDEO_HEIGHT = 720
REQUIRED_VIDEO_CODEC = "h264"
SUPPORTED_TASK_TYPES = set(TASK_TYPES)
RENDER_REQUIRED_TASK_TYPES = {"mujoco"}

__all__ = [
    "DOMAINS_BY_TASK_TYPE",
    "LICENSES",
    "REWARD_TYPES",
    "SUPPORTED_TASK_TYPES",
    "TASK_TYPES",
    "expected_ground_truth_score",
    "failed_criteria",
    "metadata_validation_issues",
    "normalize_enum_value",
    "normalize_task_type",
    "render_expected",
    "sha256_file",
    "task_type_requires_render",
    "validate_domain",
    "validate_license",
    "validate_reward_type",
    "validate_task_type",
    "validate_video_file",
]


def normalize_task_type(value: object) -> str:
    return normalize_enum_value(value)


def task_type_requires_render(value: object) -> bool:
    return normalize_task_type(value) in RENDER_REQUIRED_TASK_TYPES


def render_expected(task_type: object, render_outputs: object) -> bool:
    """Render is expected when the task type mandates it (mujoco) OR the task
    explicitly declares ``[ground_truth].render_outputs``. This generalizes the
    reviewer-artifact pipeline to non-MuJoCo domains (CFD, EM, EDA, ...)."""
    return task_type_requires_render(task_type) or bool(render_outputs)


def failed_criteria(grade_payload: dict[str, Any]) -> list[str]:
    """Return human-readable failed criteria from a normalized grade payload."""
    failed: list[str] = []
    structured = grade_payload.get("structured_subscores")
    if isinstance(structured, list):
        for entry in structured:
            if not isinstance(entry, dict):
                continue
            score = _float_or_none(entry.get("score"))
            max_score = _float_or_none(entry.get("max_score")) or 1.0
            if score is not None and score < max_score:
                name = entry.get("name") or entry.get("id") or "criterion"
                failed.append(f"{name}: {score:.3f}/{max_score:.3f}")
    return failed


def video_dimensions(path: Path) -> tuple[int, int]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError(
            "ffprobe is required to validate ground-truth video dimensions"
        )
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for ground-truth video {path}: {completed.stderr.strip()}"
        )
    raw = completed.stdout.strip()
    try:
        width_raw, height_raw = raw.split("x", 1)
        return int(width_raw), int(height_raw)
    except ValueError as exc:
        raise RuntimeError(
            f"could not parse video dimensions for {path}: {raw!r}"
        ) from exc


def video_codec(path: Path) -> str:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate ground-truth video codec")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for ground-truth video {path}: {completed.stderr.strip()}"
        )
    codec = completed.stdout.strip()
    if not codec:
        raise RuntimeError(f"could not determine video codec for {path}")
    return codec


def validate_video_file(path: Path, logical_path: str) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"ground-truth render did not create required video: {logical_path}"
        )
    if not path.is_file():
        raise RuntimeError(f"ground-truth render output must be a file: {logical_path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"ground-truth render output is empty: {logical_path}")
    width, height = video_dimensions(path)
    if (width, height) != (REQUIRED_VIDEO_WIDTH, REQUIRED_VIDEO_HEIGHT):
        raise RuntimeError(
            f"ground-truth render output must be {REQUIRED_VIDEO_WIDTH}x{REQUIRED_VIDEO_HEIGHT}, "
            f"got {width}x{height}: {logical_path}"
        )
    codec = video_codec(path)
    if codec != REQUIRED_VIDEO_CODEC:
        raise RuntimeError(
            f"ground-truth render output must use {REQUIRED_VIDEO_CODEC} video codec for browser playback, "
            f"got {codec}: {logical_path}"
        )
    return width, height


def artifact_metadata(
    problem_dir: Path,
    submission_path: Path,
    *,
    logical_path: str,
) -> dict[str, Any]:
    width, height = video_dimensions(submission_path)
    return {
        "path": str(submission_path.relative_to(problem_dir)),
        "logical_path": logical_path,
        "sha256": sha256_file(submission_path),
        "bytes": submission_path.stat().st_size,
        "width": width,
        "height": height,
    }


def artifact_matches_proof(problem_dir: Path, artifact: Any) -> bool:
    if not isinstance(artifact, dict):
        return False
    rel_path = artifact.get("path")
    expected_sha = artifact.get("sha256")
    expected_bytes = artifact.get("bytes")
    expected_width = artifact.get("width")
    expected_height = artifact.get("height")
    if not isinstance(rel_path, str) or not rel_path.startswith(
        f"{SUBMISSION_ARTIFACT_DIR.as_posix()}/"
    ):
        return False
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        return False
    path = problem_dir / rel_path
    if not path.is_file():
        return False
    try:
        size = int(expected_bytes)
        width = int(expected_width)
        height = int(expected_height)
    except (TypeError, ValueError):
        return False
    if (width, height) != (REQUIRED_VIDEO_WIDTH, REQUIRED_VIDEO_HEIGHT):
        return False
    return path.stat().st_size == size and sha256_file(path) == expected_sha


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
