"""Task scaffolding helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich.console import Console

console = Console()


class TaskCreator:
    """Create a task from one of the starter templates."""

    def collect_inputs(self) -> dict[str, str]:
        """Collect task creation inputs interactively."""
        name = typer.prompt("Task name, e.g. labelbox/reacher-control", type=str)
        template = typer.prompt(
            "Starter template (ml | mujoco | cfd | structures | prometheus | "
            "prometheus-cfd | prometheus-structures | prometheus-eval-cfd | "
            "prometheus-eval-structures)",
            type=str,
            default="ml",
        )
        return {"name": name, "template": template}

    def create_structure(self, output_dir: Path, inputs: dict[str, str]) -> Path:
        """Copy a starter template into the output directory.

        Templates live INSIDE this package (at
        ``alignerr_plugin/starter_templates/<name>/``) and are shipped with
        the wheel. Alignerrs don't need any external repo to scaffold a
        new task — `pip install lbx-rl-tasks-alignerr-plugin` puts the
        templates on disk under the installed package.
        """
        name = inputs["name"]
        template = inputs.get("template", "ml")
        task_id = name.split("/")[-1]

        # Templates ship under <plugin_pkg>/starter_templates/<template>/.
        # File layout: <pkg_root>/validators/task/creator.py -> parents[2] is <pkg_root>.
        package_root = Path(__file__).resolve().parents[2]
        template_dir = package_root / "starter_templates" / template
        if not template_dir.exists():
            raise FileNotFoundError(f"unknown starter template: {template}")

        problem_dir = output_dir / task_id
        shutil.copytree(template_dir, problem_dir, dirs_exist_ok=True)
        console.print(f"[green]Created task scaffold:[/green] {problem_dir}")
        return problem_dir
