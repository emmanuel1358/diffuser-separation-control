from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from alignerr_plugin.capabilities import is_capability_task
from alignerr_plugin.exporters.taiga import build_job_payload
from alignerr_plugin.schemas import TaskToml
from alignerr_plugin.utils import load_metadata, load_task_toml, read_prompt, task_id

from lbx_rl_tasks_harness.models import (
    GroundTruthSpec,
    HarnessProblem,
    OutputSpec,
    ReferenceSpec,
)


class LazyTaigaProblem(Mapping[str, Any]):
    """Build optional Taiga metadata only when a Taiga consumer requests it."""

    def __init__(
        self, problem_dir: Path, factory: Callable[[], dict[str, Any]]
    ) -> None:
        self._problem_dir = problem_dir
        self._factory = factory
        self._value: dict[str, Any] | None = None
        self._error: ValueError | None = None

    def _load(self) -> dict[str, Any]:
        if self._value is not None:
            return self._value
        if self._error is not None:
            raise self._error
        try:
            self._value = self._factory()
        except (OSError, TypeError, ValueError) as exc:
            self._error = ValueError(
                "Taiga metadata projection is unavailable for "
                f"{self._problem_dir}: {exc}"
            )
            raise self._error from exc
        return self._value

    def __getitem__(self, key: str) -> Any:
        return self._load()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())

    def __repr__(self) -> str:
        state = "loaded" if self._value is not None else "pending"
        if self._error is not None:
            state = "unavailable"
        return f"LazyTaigaProblem({self._problem_dir!s}, {state})"


def task_toml_metadata(task_toml: TaskToml) -> dict:
    """Project every runtime-relevant task.toml section into harness metadata."""
    return {
        "task": task_toml.task.model_dump(),
        "agent": task_toml.agent.model_dump(),
        "verifier": task_toml.verifier.model_dump(),
        "environment": task_toml.environment.model_dump(),
        "runner": task_toml.runner.model_dump(),
        "difficulty": task_toml.difficulty.model_dump(),
        "delivery": task_toml.delivery.model_dump(),
    }


def load_problem_dir(problem_dir: Path) -> HarnessProblem:
    problem_dir = problem_dir.resolve()
    metadata = load_metadata(problem_dir)
    task_toml = load_task_toml(problem_dir)
    prompt = read_prompt(problem_dir)
    scorer_dir = problem_dir / "scorer"
    grader_dir = scorer_dir
    private_dir = scorer_dir / "data"

    def build_taiga_problem() -> dict[str, Any]:
        taiga_problem = build_job_payload(
            problem_dir,
            image_ref="LOCAL_IMAGE",
            image_is_outer_capsule=is_capability_task(task_toml),
        )["problems_metadata"]["problem_set"]["problems"][0]
        _apply_prompt_to_taiga_problem(taiga_problem, prompt)
        return taiga_problem

    return HarnessProblem(
        id=task_id(problem_dir),
        source_format="problem-dir",
        prompt=prompt,
        outputs=[
            OutputSpec(
                path=out.path, required=out.required, description=out.description
            )
            for out in task_toml.outputs
        ],
        ground_truth=GroundTruthSpec(
            render_command=task_toml.ground_truth.render_command,
            render_outputs=[
                OutputSpec(
                    path=out.path,
                    required=out.required,
                    description=out.description,
                )
                for out in task_toml.ground_truth.render_outputs
            ],
            score_epsilon=task_toml.ground_truth.score_epsilon,
            continuous_score_epsilon=task_toml.ground_truth.continuous_score_epsilon,
            in_container=task_toml.ground_truth.in_container,
            zero_anchor_epsilon=task_toml.ground_truth.zero_anchor_epsilon,
        ),
        reference=ReferenceSpec(
            execution=task_toml.reference.execution,
            cache_dir=task_toml.reference.cache_dir,
            entrypoint=task_toml.reference.entrypoint,
            proof_mode=task_toml.reference.proof_mode,
        ),
        task_dir=problem_dir,
        source_problem_dir=problem_dir,
        grader_dir=grader_dir,
        private_dir=private_dir,
        required_resources=task_toml.environment.required_resources,
        required_tools=list(task_toml.runner.required_tools),
        metadata={"benchmark": metadata.benchmark, **task_toml_metadata(task_toml)},
        taiga_problem=LazyTaigaProblem(problem_dir, build_taiga_problem),
    )


def _apply_prompt_to_taiga_problem(taiga_problem: dict, prompt: str) -> None:
    taiga_problem["prompt"] = prompt
    extra_fields = taiga_problem.get("extra_fields")
    if isinstance(extra_fields, dict):
        extra_fields["task_prompt"] = prompt
    else:
        taiga_problem["extra_fields"] = {"task_prompt": prompt}
