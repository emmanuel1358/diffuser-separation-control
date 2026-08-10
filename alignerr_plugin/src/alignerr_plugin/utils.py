"""Utility helpers for task discovery, hashing, and config loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import tomllib

from alignerr_plugin.schemas import ProblemMetadata, TaskToml

IGNORED_HASH_PARTS = {".git", ".alignerr", "__pycache__", ".taiga_submit.json"}

# Task subtrees/files that can change the built image or the deterministic
# oracle/reference score. Everything else (README.md, instruction.md, other
# prose, NOTICE/LICENSE) never reaches the scorer or the built image, so editing
# it must NOT stale the build proof (which would force a pointless full
# rebuild+regrade that yields the identical score).
GRADING_INPUT_DIRS = (
    "solution",
    "scorer",
    "data",
    "data_generation",
    "data-generation",
    "baselines",
    "environment",
)
# calibration.lock.json is generated, but it is baked into the image and its
# anchors drive continuous scores, so a hand-edited lock has to stale the proof.
# update_build_proof_result refreshes task_dir_sha256 after writing a new lock,
# so including it here does not make a fresh ground-truth run look stale.
GRADING_INPUT_FILES = ("task.toml", "calibration.lock.json")


class LegacyTaskLayoutError(RuntimeError):
    """Raised for a legacy ML_Envs (metadata-mode) task directory.

    The metadata-mode contract (``metadata.json`` with ``ml_task_type`` +
    ``prompt.md`` + ``test_file.py``, no ``task.toml``) was removed. Failing
    loudly here beats misclassifying the directory as a malformed native task.
    """


def _declares_ml_task_type(metadata_path: Path) -> bool:
    """True for a metadata-mode ``metadata.json`` (i.e. one carrying ``ml_task_type``).

    Native tasks ship a ``metadata.json`` too -- the Taiga envelope -- so the file
    existing is not itself a legacy marker; only the removed ``ml_task_type`` key is.
    """
    try:
        # utf-8-sig, not utf-8: a BOM would otherwise make json.loads raise and
        # silently downgrade a legacy tree to a bare "no task.toml" failure,
        # losing the migration guidance below. Plain UTF-8 decodes identically.
        data = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and "ml_task_type" in data


def reject_legacy_layout(problem_dir: Path) -> None:
    """Fail with migration guidance when a task uses the removed ML_Envs layout."""
    if (problem_dir / "task.toml").is_file():
        return
    markers = [
        name
        for name in ("test_file.py", "prompt.md", "reference_solution")
        if (problem_dir / name).exists()
    ]
    # A partly-converted tree can have renamed prompt.md/test_file.py already and
    # still carry the metadata-mode config, so this is often the only marker left.
    metadata_path = problem_dir / "metadata.json"
    if metadata_path.is_file() and _declares_ml_task_type(metadata_path):
        markers.append("metadata.json with ml_task_type")
    if not markers:
        return
    raise LegacyTaskLayoutError(
        f"{problem_dir} uses the removed ML_Envs (metadata-mode) task layout "
        f"(found {', '.join(sorted(markers))}, no task.toml). Every task must now "
        "use the native layout: task.toml, instruction.md, scorer/compute_score.py, "
        "environment/Dockerfile, solution/. See docs/LEGACY_ML_LAYOUT.md for the "
        "conversion steps."
    )


def load_task_toml(problem_dir: Path) -> TaskToml:
    """Load and validate a native task.toml."""
    reject_legacy_layout(problem_dir)
    with (problem_dir / "task.toml").open("rb") as handle:
        return TaskToml.model_validate(tomllib.load(handle))


def load_metadata(problem_dir: Path) -> ProblemMetadata:
    """Load the Alignerr metadata.json envelope."""
    reject_legacy_layout(problem_dir)
    return ProblemMetadata.model_validate(
        json.loads((problem_dir / "metadata.json").read_text(encoding="utf-8-sig"))
    )


def read_prompt(problem_dir: Path) -> str:
    """Read the task prompt (``instruction.md``)."""
    reject_legacy_layout(problem_dir)
    return (problem_dir / "instruction.md").read_text()


def task_id(problem_dir: Path) -> str:
    """Return the problem instance id from metadata.json."""
    metadata = load_metadata(problem_dir)
    instance_id = metadata.problem_data.get("instance_id") or metadata.problem_data.get(
        "id"
    )
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError(
            "metadata.json problem_data must include a non-empty instance_id"
        )
    return instance_id


def task_dir_sha256(problem_dir: Path) -> str:
    """Compute a deterministic hash over task files, excluding local state."""
    digest = hashlib.sha256()
    for path in sorted(problem_dir.rglob("*")):
        rel = path.relative_to(problem_dir)
        if any(part in IGNORED_HASH_PARTS for part in rel.parts):
            continue
        if path.is_dir():
            continue
        digest.update(str(rel).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _is_grading_input(
    rel: Path,
    *,
    dirs: tuple[str, ...] = GRADING_INPUT_DIRS,
    files: tuple[str, ...] = GRADING_INPUT_FILES,
) -> bool:
    parts = rel.parts
    if parts and parts[0] in dirs:
        return True
    return rel.as_posix() in files


def grading_inputs_sha256(problem_dir: Path) -> str:
    """Deterministic hash over only grading-/image-affecting task files.

    Scope: ``task.toml`` + ``solution/`` / ``scorer/`` / ``data/`` /
    ``data_generation/`` / ``baselines/`` / ``environment/``; docs and local
    state are excluded. ``environment/`` is in scope because
    ``verify_build_proof`` does not separately check ``image_digest``, so a
    Dockerfile edit must stale the proof. ``data/`` is in scope because the
    public tree is baked into the image and trains the reference solution, so
    swapping it changes the oracle score.
    """
    dirs, files = GRADING_INPUT_DIRS, GRADING_INPUT_FILES
    digest = hashlib.sha256()
    for path in sorted(problem_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(problem_dir)
        if any(part in IGNORED_HASH_PARTS for part in rel.parts):
            continue
        if not _is_grading_input(rel, dirs=dirs, files=files):
            continue
        digest.update(rel.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON object with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
