from __future__ import annotations

from pathlib import Path

from alignerr_plugin.utils import load_task_toml

from lbx_rl_tasks_harness.formats.problem_dir import load_problem_dir
from lbx_rl_tasks_harness.models import HarnessProblem, OutputSpec


def load_harbor_dir(
    task_dir: Path, source_problem_dir: Path | None = None
) -> HarnessProblem:
    task_dir = task_dir.resolve()
    task_toml = load_task_toml(task_dir)
    prompt = (task_dir / "instruction.md").read_text()
    source_problem = (
        load_problem_dir(source_problem_dir) if source_problem_dir else None
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
        grader_dir=source_problem.grader_dir if source_problem else None,
        private_dir=source_problem.private_dir if source_problem else None,
        image=getattr(task_toml.environment, "docker_image", None),
        required_tools=list(task_toml.runner.required_tools),
        metadata={
            "harbor_task_dir": str(task_dir),
            "tests": str(task_dir / "tests" / "test.sh"),
            "task": task_toml.task.model_dump(),
            "environment": task_toml.environment.model_dump(),
            "runtime_notices": task_toml.metadata.get("runtime_notices", []),
        },
        taiga_problem=source_problem.taiga_problem if source_problem else None,
    )
