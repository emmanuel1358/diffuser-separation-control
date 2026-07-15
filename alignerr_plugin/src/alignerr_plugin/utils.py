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
GRADING_INPUT_DIRS = ("solution", "scorer", "data_generation", "environment")
GRADING_INPUT_FILES = ("task.toml",)

# ML_Envs-mode grading-/image-affecting inputs (no task.toml / scorer/).
MLENVS_GRADING_INPUT_DIRS = ("data", "reference_solution")
MLENVS_GRADING_INPUT_FILES = ("metadata.json", "test_file.py")


def load_task_toml(problem_dir: Path) -> TaskToml:
    """Load and validate task.toml (synthesized in memory for ML_Envs-mode tasks)."""
    from alignerr_plugin import mlenvs

    if mlenvs.is_mlenvs_task(problem_dir):
        return mlenvs.synthesize_task_toml(problem_dir)
    with (problem_dir / "task.toml").open("rb") as handle:
        return TaskToml.model_validate(tomllib.load(handle))


def load_metadata(problem_dir: Path) -> ProblemMetadata:
    """Load the Alignerr metadata.json envelope (synthesized for ML_Envs-mode tasks)."""
    from alignerr_plugin import mlenvs

    if mlenvs.is_mlenvs_task(problem_dir):
        return mlenvs.synthesize_problem_metadata(problem_dir)
    return ProblemMetadata.model_validate(
        json.loads((problem_dir / "metadata.json").read_text())
    )


def read_prompt(problem_dir: Path) -> str:
    """Read the task prompt: ``prompt.md`` for ML_Envs-mode tasks, else
    ``instruction.md`` for native tasks."""
    from alignerr_plugin import mlenvs

    name = "prompt.md" if mlenvs.is_mlenvs_task(problem_dir) else "instruction.md"
    return (problem_dir / name).read_text()


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

    Scope (native): ``task.toml`` + ``solution/`` / ``scorer/`` /
    ``data_generation/`` / ``environment/``; docs and local state are excluded.
    ``environment/`` is in scope because ``verify_build_proof`` does not separately
    check ``image_digest``, so a Dockerfile edit must stale the proof. ML_Envs-mode
    tasks scope to ``metadata.json`` + ``test_file.py`` + ``data/`` +
    ``reference_solution/`` (see ``MLENVS_GRADING_INPUT_*``).
    """
    from alignerr_plugin import mlenvs

    if mlenvs.is_mlenvs_task(problem_dir):
        dirs, files = MLENVS_GRADING_INPUT_DIRS, MLENVS_GRADING_INPUT_FILES
    else:
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
