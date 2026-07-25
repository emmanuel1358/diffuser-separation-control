from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lbx_rl_tasks_harness.formats.problem_dir import load_problem_dir
from lbx_rl_tasks_harness.models import HarnessProblem, OutputSpec


def _problem_set(payload: dict[str, Any]) -> dict[str, Any]:
    if "problems_metadata" in payload:
        payload = payload["problems_metadata"]
    if "problem_set" not in payload:
        raise ValueError(
            "Boreal payload must contain problem_set or problems_metadata.problem_set"
        )
    return payload["problem_set"]


def _outputs_from_problem(problem: dict[str, Any]) -> list[OutputSpec]:
    raw_outputs = problem.get("outputs")
    if not isinstance(raw_outputs, list):
        task_metadata = problem.get("extra_fields", {}).get("task_metadata", {})
        raw_outputs = task_metadata.get("outputs", [])
    outputs: list[OutputSpec] = []
    for item in raw_outputs:
        if isinstance(item, dict) and "path" in item:
            outputs.append(
                OutputSpec(
                    path=str(item["path"]),
                    required=bool(item.get("required", True)),
                    description=str(item.get("description", "")),
                )
            )
    return outputs


def load_taiga_metadata(
    metadata_path: Path, source_problem_dir: Path | None = None
) -> HarnessProblem:
    metadata_path = metadata_path.resolve()
    payload = json.loads(metadata_path.read_text())
    problem_set = _problem_set(payload)
    problems = problem_set.get("problems") or []
    if len(problems) != 1:
        raise ValueError(
            f"local harness expects exactly one Boreal problem, found {len(problems)}"
        )
    problem = problems[0]

    source_problem = (
        load_problem_dir(source_problem_dir) if source_problem_dir else None
    )
    return HarnessProblem(
        id=str(problem["id"]),
        source_format="taiga",
        prompt=str(problem.get("task_prompt", "")),
        outputs=(
            source_problem.outputs if source_problem else _outputs_from_problem(problem)
        ),
        task_dir=metadata_path.parent,
        source_problem_dir=(
            source_problem.source_problem_dir if source_problem else None
        ),
        grader_dir=source_problem.grader_dir if source_problem else None,
        private_dir=source_problem.private_dir if source_problem else None,
        image=problem.get("image"),
        required_tools=list(problem.get("required_tools") or []),
        required_resources=problem.get("required_resources")
        or problem_set.get("required_resources"),
        metadata={
            "metadata_path": str(metadata_path),
            "startup_command": problem.get("startup_command"),
            "grading_strategy": problem.get("grading_strategy"),
            "job_fields": {
                key: value
                for key, value in payload.items()
                if key not in {"problems_metadata", "problem_set"}
            },
        },
        taiga_problem=problem,
    )
