from __future__ import annotations

from pathlib import Path

from alignerr_plugin.utils import load_task_toml

from lbx_rl_tasks_harness.formats.problem_dir import load_problem_dir, task_toml_metadata
from lbx_rl_tasks_harness.models import (
    GroundTruthSpec,
    HarnessProblem,
    OutputSpec,
    ReferenceSpec,
)


def load_harbor_dir(
    task_dir: Path, source_problem_dir: Path | None = None
) -> HarnessProblem:
    task_dir = task_dir.resolve()
    task_toml = load_task_toml(task_dir)
    prompt = (task_dir / "instruction.md").read_text()
    source_problem = (
        load_problem_dir(source_problem_dir) if source_problem_dir else None
    )
    exported_grader = task_dir / "environment" / "scorer"
    grader_dir = (
        source_problem.grader_dir
        if source_problem
        else exported_grader if exported_grader.is_dir() else None
    )
    private_dir = (
        source_problem.private_dir
        if source_problem
        else grader_dir / "data" if grader_dir else None
    )
    ground_truth = source_problem.ground_truth if source_problem else GroundTruthSpec(
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
    )
    reference = source_problem.reference if source_problem else ReferenceSpec(
        execution=task_toml.reference.execution,
        cache_dir=task_toml.reference.cache_dir,
        entrypoint=task_toml.reference.entrypoint,
        proof_mode=task_toml.reference.proof_mode,
    )

    return HarnessProblem(
        id=source_problem.id if source_problem else task_toml.task.name.split("/")[-1],
        source_format="harbor",
        prompt=prompt,
        outputs=[
            OutputSpec(
                path=out.path, required=out.required, description=out.description
            )
            for out in task_toml.outputs
        ],
        task_dir=task_dir,
        source_problem_dir=(
            source_problem.source_problem_dir if source_problem else None
        ),
        grader_dir=grader_dir,
        private_dir=private_dir,
        image=getattr(task_toml.environment, "docker_image", None),
        required_resources=(
            source_problem.required_resources
            if source_problem
            else task_toml.environment.required_resources
        ),
        required_tools=list(task_toml.runner.required_tools),
        ground_truth=ground_truth,
        reference=reference,
        metadata={
            **task_toml_metadata(task_toml),
            "harbor_task_dir": str(task_dir),
            "tests": str(task_dir / "tests" / "test.sh"),
            "runtime_notices": task_toml.metadata.get("runtime_notices", []),
        },
        taiga_problem=source_problem.taiga_problem if source_problem else None,
    )
