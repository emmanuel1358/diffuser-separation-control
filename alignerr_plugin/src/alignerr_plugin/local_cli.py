from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from alignerr_plugin.exporters.harbor import export_harbor as export_harbor_impl
from alignerr_plugin.exporters.taiga import export_taiga as export_taiga_impl
from alignerr_plugin.validators.task.creator import TaskCreator
from alignerr_plugin.validators.task.validator import TaskValidator, reward_hack_lint

app = typer.Typer(
    help="Self-contained task validation and export helpers for lbx-rl tasks.",
    no_args_is_help=True,
)
console = Console()

STARTER_TEMPLATES = (
    "ml",
    "mujoco",
    "cfd",
    "structures",
    "prometheus",
    "prometheus-cfd",
    "prometheus-structures",
    "prometheus-eval-cfd",
    "prometheus-eval-structures",
)

# Human-friendly, actionable hints surfaced by `check` when a gate fails. Keyed
# by the validator stage name (see TaskValidator.validate).
STAGE_HINTS: dict[str, str] = {
    "schema": (
        "task.toml / metadata.json shape problem. Confirm required files exist; "
        "ml tasks must set allow_internet = false (locked off for ml)."
    ),
    "prompt_runtime_references": (
        "instruction.md references runtime details it shouldn't pin (e.g. exact "
        "GPU/TPU). Let the platform inject availability notices instead."
    ),
    "prompt_quality": (
        "instruction.md tripped the prompt-quality gate: keep it ASCII, no "
        "emoji, and free of AI-artifact boilerplate."
    ),
    "grader_import": "scorer/compute_score.py failed to import. Fix the syntax/import error.",
    "grader_sandbox": "scorer crashed under the sandbox. Make grading deterministic and self-contained.",
    "agent_fault": "Agent-fault handling is misconfigured; a crashing submission must score 0.0, not error.",
    "scorer_determinism": "compute_score is non-deterministic: same input must yield the same score.",
    "outputs": "Declared [[outputs]] don't match what the grader reads. Align paths in task.toml.",
    "ground_truth": (
        "solution/solve.sh must score the reward_type target (1.0 for rubrics, "
        "0.5\u00b10.05 for continuous). mujoco tasks also need a 1280x720 render."
    ),
    "private_data_layout": "Private/held-out truth must live under scorer/ (never readable by the agent).",
    "solution_answer_key_leak": "The reference solution leaks the answer key into agent-visible files. Move it under scorer/.",
    "hidden_env": "hidden_env is enabled but scorer/data/env.py (make_env) or data/env_client.py is missing/invalid.",
    "compute_score_return": "compute_score must return the expected score shape for this reward_type.",
    "local_build_proof": "Commit a fresh .alignerr/build_proof.json (run the ground-truth harness first).",
    "conditional": "A task-type-specific contract failed; read the stage issues above for the exact rule.",
}


@app.command()
def validate(
    problem_dir: Path = typer.Option(
        ..., "--problem-dir", "-d", exists=True, file_okay=False
    ),
) -> None:
    """Run template task validation without the external Alignerr CLI."""
    result = TaskValidator().validate(
        problem_dir,
        problem_dir / ".alignerr" / "validations",
        Path.cwd(),
    )
    console.print(result.model_dump_json(indent=2))
    warnings = [
        warning for stage in result.stages.values() for warning in stage.warnings
    ]
    if warnings:
        console.print(
            f"\n[yellow]{len(warnings)} advisory warning(s) (non-blocking):[/yellow]"
        )
        for warning in warnings:
            console.print(f"- {warning}")
    if result.status != "valid":
        raise typer.Exit(1)


@app.command()
def new(
    name: str = typer.Option(
        None, "--name", help="Task name, e.g. labelbox/my-task (prompted if omitted)"
    ),
    template: str = typer.Option(
        "ml",
        "--template",
        "-t",
        help=(
            "Starter: ml | mujoco | cfd | structures | prometheus | "
            "prometheus-cfd | prometheus-structures | prometheus-eval-cfd | "
            "prometheus-eval-structures"
        ),
    ),
    output_dir: Path = typer.Option(
        Path("problems"), "--out", "-o", help="Directory to create the task under"
    ),
) -> None:
    """Scaffold a new task from a starter template.

    Pre-fills every shim file (task.toml, instruction.md, environment/Dockerfile,
    solution/solve.sh, tests/test.sh, scorer/compute_score.py, ...) so authors
    edit working defaults instead of writing boilerplate from scratch.
    """
    creator = TaskCreator()
    if template not in STARTER_TEMPLATES:
        console.print(
            f"[red]unknown template {template!r}; choose one of "
            f"{', '.join(STARTER_TEMPLATES)}[/red]"
        )
        raise typer.Exit(1)
    if name is None:
        name = typer.prompt("Task name, e.g. labelbox/my-task")
    inputs = {"name": name, "template": template}
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_dir = creator.create_structure(output_dir, inputs)
    console.print(
        "\n[bold]Next steps[/bold]:\n"
        f"  1. Edit {problem_dir}/task.toml, instruction.md, and scorer/compute_score.py\n"
        f"  2. Implement solution/solve.sh (the oracle) and tests/test.sh\n"
        f"  3. Run: [cyan]lbx-rl-template check --problem-dir {problem_dir}[/cyan]"
    )


@app.command()
def check(
    problem_dir: Path = typer.Option(
        ..., "--problem-dir", "-d", exists=True, file_okay=False
    ),
) -> None:
    """One-shot "am I ready to submit?" preflight.

    Runs every blocking validation gate plus the advisory reward-hack lint and
    prints a friendly, actionable summary. Exits non-zero if anything blocking
    fails, so it doubles as a local pre-submit safety net (no H100/Taiga needed).
    """
    result = TaskValidator().validate(
        problem_dir,
        problem_dir / ".alignerr" / "validations",
        Path.cwd(),
    )

    failed = {name: stage for name, stage in result.stages.items() if not stage.passed}
    passed_count = len(result.stages) - len(failed)
    console.print(
        f"[bold]Preflight for {result.problem_id}[/bold] "
        f"({passed_count}/{len(result.stages)} gates passed)\n"
    )
    for name, stage in result.stages.items():
        if stage.passed:
            console.print(f"  [green]\u2713[/green] {name}")
            continue
        console.print(f"  [red]\u2717 {name}[/red]")
        hint = STAGE_HINTS.get(name)
        if hint:
            console.print(f"      [yellow]\u2192 {hint}[/yellow]")
        for issue in stage.issues:
            console.print(f"      - {issue}")

    warnings = reward_hack_lint(problem_dir)
    if warnings:
        console.print(
            f"\n[yellow]Advisory (non-blocking) reward-hack lint: "
            f"{len(warnings)} warning(s)[/yellow]"
        )
        for warning in warnings:
            console.print(f"  - {warning}")

    if result.status != "valid":
        console.print(
            f"\n[red]NOT ready to submit: {len(failed)} blocking gate(s) failed.[/red]"
        )
        raise typer.Exit(1)
    console.print("\n[green]Ready to submit: all blocking gates passed.[/green]")


@app.command("lint-reward-hacks")
def lint_reward_hacks(
    problem_dir: Path = typer.Option(
        ..., "--problem-dir", "-d", exists=True, file_okay=False
    ),
    output_json: Path = typer.Option(
        None, "--output-json", help="Optional path to write the warnings as JSON"
    ),
) -> None:
    """Advisory (non-blocking) reward-hacking lint for a task's grader.

    Never exits non-zero: these are heuristic warnings surfaced for the author,
    not a gate. The blocking guarantees live in `validate`.
    """
    warnings = reward_hack_lint(problem_dir)
    payload = {"problem_dir": str(problem_dir), "warnings": warnings}
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n")
    if warnings:
        console.print(
            f"[yellow]reward-hack lint: {len(warnings)} advisory warning(s)[/yellow]"
        )
        for warning in warnings:
            console.print(f"- {warning}")
    else:
        console.print("[green]reward-hack lint: no advisory warnings[/green]")


@app.command("export-taiga")
def export_taiga(
    problem_dir: Path = typer.Option(
        ..., "--problem-dir", "-d", exists=True, file_okay=False
    ),
    output: Path = typer.Option(Path("problems-metadata.json"), "--out", "-o"),
    image: str = typer.Option("PLACEHOLDER", "--image"),
) -> None:
    """Export Boreal/Taiga metadata without the external Alignerr CLI."""
    sidecar = export_taiga_impl(problem_dir, output, image_ref=image)
    console.print(f"[green]Wrote Boreal metadata:[/green] {output}")
    console.print(sidecar)


@app.command("export-harbor")
def export_harbor(
    problem_dir: Path = typer.Option(
        ..., "--problem-dir", "-d", exists=True, file_okay=False
    ),
    output_dir: Path = typer.Option(Path("harbor-export"), "--out", "-o"),
    image: str | None = typer.Option(
        None, "--image", help="Digest-pinned Docker image to write into task.toml"
    ),
    runtime_notices: bool = typer.Option(
        True,
        "--runtime-notices/--no-runtime-notices",
        help="Include generated non-prompt runtime notices such as GPU/TPU availability.",
    ),
) -> None:
    """Export Harbor task format without the external Alignerr CLI."""
    path = export_harbor_impl(
        problem_dir,
        output_dir,
        image_ref=image,
        include_runtime_notices=runtime_notices,
    )
    console.print(f"[green]Wrote Harbor export:[/green] {path}")


if __name__ == "__main__":
    app()
