"""Typer commands exposed by the task plugin."""

from pathlib import Path

import typer
from rich.console import Console

from alignerr_plugin.exporters.harbor import export_harbor as export_harbor_impl
from alignerr_plugin.exporters.taiga import export_taiga as export_taiga_impl
from alignerr_plugin.validators.task.creator import TaskCreator
from alignerr_plugin.validators.task.validator import TaskValidator

console = Console()


def create_problem(
    name: str = typer.Argument(..., help="Task name, e.g. labelbox/reacher-control"),
    template: str = typer.Option(
        "ml",
        "--template",
        help=(
            "Starter template: ml | mujoco | cfd | structures | "
            "software-engineering | prometheus | prometheus-cfd | "
            "prometheus-structures | prometheus-eval-cfd | "
            "prometheus-eval-structures"
        ),
    ),
    output_dir: Path = typer.Option(Path("problems"), "--output-dir", "-o"),
) -> None:
    """Create a task from a starter template."""
    creator = TaskCreator()
    creator.create_structure(output_dir, {"name": name, "template": template})


def evaluate(
    config_path: Path = typer.Option(..., "--config-path", "-c"),
    problem_dir: Path = typer.Option(..., "--problem-dir", "-d"),
) -> None:
    """Placeholder for full agent evaluation."""
    console.print(
        f"[yellow]Evaluation runner not implemented yet.[/yellow] Config={config_path} Problem={problem_dir}"
    )


def score(problem_dir: Path = typer.Option(..., "--problem-dir", "-d")) -> None:
    """Run validator scoring checks for a task."""
    result = TaskValidator().validate(
        problem_dir, problem_dir / ".alignerr" / "validations", Path.cwd()
    )
    console.print(result.model_dump_json(indent=2))


def export_taiga(
    problem_dir: Path = typer.Option(..., "--problem-dir", "-d"),
    output: Path = typer.Option(Path("problems-metadata.json"), "--out", "-o"),
    image: str = typer.Option("PLACEHOLDER", "--image"),
    outer_capsule: bool = typer.Option(
        False,
        "--outer-capsule",
        help=(
            "Trusted assertion that --image is the built outer capsule. Required "
            "for capability tasks; production images must be digest-pinned "
            "(LOCAL_IMAGE is allowed for local validation)."
        ),
    ),
) -> None:
    """Export task metadata for Boreal submission."""
    sidecar = export_taiga_impl(
        problem_dir,
        output,
        image_ref=image,
        image_is_outer_capsule=outer_capsule,
    )
    console.print(f"[green]Wrote Boreal metadata:[/green] {output}")
    console.print(sidecar)


def export_harbor(
    problem_dir: Path = typer.Option(..., "--problem-dir", "-d"),
    output_dir: Path = typer.Option(Path("harbor-export"), "--out", "-o"),
    image: str | None = typer.Option(
        None,
        "--image",
        help=(
            "Digest-pinned image to stamp for capability/separate-service tasks; "
            "ignored for legacy self-contained exports."
        ),
    ),
    runtime_notices: bool = typer.Option(
        True,
        "--runtime-notices/--no-runtime-notices",
        help="Include generated non-prompt runtime notices such as GPU/TPU availability.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Atomically replace an existing exporter-owned directory.",
    ),
) -> None:
    """Export a task directory in Harbor layout."""
    path = export_harbor_impl(
        problem_dir,
        output_dir,
        image_ref=image,
        include_runtime_notices=runtime_notices,
        force=force,
    )
    console.print(f"[green]Wrote Harbor export:[/green] {path}")
