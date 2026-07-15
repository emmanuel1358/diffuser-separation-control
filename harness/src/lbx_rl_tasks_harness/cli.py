from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from alignerr_plugin.local_runtime import VALID_FLAVORS

from lbx_rl_tasks_harness import __version__
from lbx_rl_tasks_harness.env import load_env_file
from lbx_rl_tasks_harness.formats.harbor import load_harbor_dir
from lbx_rl_tasks_harness.formats.problem_dir import load_problem_dir
from lbx_rl_tasks_harness.formats.taiga import load_taiga_metadata
from lbx_rl_tasks_harness.models import HarnessProblem, RuntimeName
from lbx_rl_tasks_harness.runner import (
    run_auto_qa_only,
    run_harness,
    run_reference_harness,
)

app = typer.Typer(
    help="Local test harness for lbx RL tasks.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
DEFAULT_RUN_DIR = Path(".harness-runs")


def _version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit(0)


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        help="Print version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    pass


def _emit(result) -> None:
    console.print(f"[green]runtime:[/green] {result.runtime}")
    if result.score is None:
        console.print("[green]score:[/green] (not graded)")
    else:
        console.print(f"[green]score:[/green] {result.score:.6f}")
    console.print(f"[green]run_dir:[/green] {result.run_dir}")
    console.print(f"[green]reward:[/green] {result.reward_path}")
    for artifact in getattr(result, "review_artifacts", []):
        path = artifact.get("path") if isinstance(artifact, dict) else artifact
        console.print(f"[green]review_artifact:[/green] {path}")


def _run(
    problem: HarnessProblem,
    runtime: RuntimeName,
    run_dir: Path,
    model: str | None,
    max_steps: int | None = None,
    flavor: str = "auto",
) -> None:
    load_env_file()
    try:
        results = []
        for concrete_runtime in _expand_runtime(runtime):
            results.append(
                run_harness(
                    problem,
                    runtime=concrete_runtime,
                    run_dir_base=run_dir,
                    model=model,
                    max_steps=max_steps,
                    flavor=flavor,
                )
            )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        console.print(f"[red]Harness failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    for result in results:
        _emit(result)


def _expand_runtime(runtime: RuntimeName) -> list[RuntimeName]:
    if runtime == "ground-truth":
        return ["solution"]
    if runtime == "rubric-quality":
        return ["noop"]
    return [runtime]


@app.command()
def reference(
    problem_dir: Path = typer.Option(
        ..., "--problem-dir", "-d", exists=True, file_okay=False
    ),
    run_dir: Path = typer.Option(DEFAULT_RUN_DIR, "--run-dir"),
    solution_dir: str = typer.Option(
        "solution",
        "--solution-dir",
        help=(
            "Directory under the task root holding the reference to run. "
            "ML_Envs tasks run <solution-dir>/solution.py (default reference_solution; "
            "e.g. --solution-dir baselines/naive to run a baseline instead). "
            "Native tasks run <solution-dir>/solve.sh (default solution)."
        ),
    ),
    skip_solve: bool = typer.Option(
        False,
        "--skip-solve",
        help="Grade the cached /tmp/output artifacts without rerunning the reference.",
    ),
    no_grade: bool = typer.Option(
        False,
        "--no-grade",
        help="Run the reference solve only; skip grading.",
    ),
    clean_cache: bool = typer.Option(
        False,
        "--clean-cache",
        help="Wipe the persistent reference cache before running.",
    ),
) -> None:
    """Run the reference solution iteration loop (train/solve, optional grade)."""
    load_env_file()
    try:
        result = run_reference_harness(
            load_problem_dir(problem_dir),
            run_dir_base=run_dir,
            skip_solve=skip_solve,
            no_grade=no_grade,
            clean_cache=clean_cache,
            solution_dir=solution_dir,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        console.print(f"[red]Reference run failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _emit(result)
    manifest = problem_dir / ".alignerr" / "reference_run" / "latest" / "manifest.json"
    if manifest.exists():
        console.print(f"[green]reference_manifest:[/green] {manifest}")


@app.command()
def run(
    problem_dir: Path = typer.Option(
        ..., "--problem-dir", "-d", exists=True, file_okay=False
    ),
    runtime: RuntimeName = typer.Option(
        "claude-code",
        "--runtime",
        help=(
            "What to run. Default 'claude-code' uses the local Claude Code CLI "
            "login. Use 'deepagents', 'ground-truth', or 'rubric-quality' for "
            "focused checks."
        ),
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help=(
            "Model for the selected agent runtime. DeepAgents supports provider "
            "prefixes like openai:gpt-5.1 or google:gemini-3-pro; Claude Code "
            "uses Claude model names."
        ),
    ),
    run_dir: Path = typer.Option(DEFAULT_RUN_DIR, "--run-dir"),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        help=(
            "Local-only cap on agent steps. Claude Code defaults to 300 turns "
            "when unset; DeepAgents stops after any pending tool call resolves "
            "once the cap is reached."
        ),
    ),
    flavor: str = typer.Option(
        "auto",
        "--flavor",
        help=(
            "Compute base flavor for ML_Envs tasks. 'auto' (default): build the "
            "heavy production-equivalent base and, on a build-time OOM, fall back "
            "to the local-only slim base (agent pip-installs ML wheels at runtime; "
            "turn budget bumped to compensate). 'heavy': force the production base "
            "(required for graphics/TPU tasks). 'slim': force the stripped base "
            "for low-RAM hosts (ML_Envs compute tasks only)."
        ),
    ),
) -> None:
    """Run a local authoring-format task directory."""
    if flavor not in VALID_FLAVORS:
        raise typer.BadParameter(
            f"--flavor must be one of {list(VALID_FLAVORS)}; got {flavor!r}"
        )
    _run(
        load_problem_dir(problem_dir),
        runtime,
        run_dir,
        model,
        max_steps=max_steps,
        flavor=flavor,
    )


@app.command("verify-ground-truth")
def verify_ground_truth(
    problem_dir: Path = typer.Option(
        ..., "--problem-dir", "-d", exists=True, file_okay=False
    ),
    run_dir: Path = typer.Option(DEFAULT_RUN_DIR, "--run-dir"),
) -> None:
    """Verify that the task's oracle solution scores 1.0 and renders review video."""
    _run(load_problem_dir(problem_dir), "solution", run_dir, None)


@app.command("autoqa")
def autoqa(
    problem_dir: Path = typer.Option(
        ..., "--problem-dir", "-d", exists=True, file_okay=False
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help=(
            "Auto QA model. Defaults to LBX_RL_AUTOQA_MODEL or openai:gpt-5.5. "
            "Does not use LBX_RL_HARNESS_MODEL."
        ),
    ),
    reasoning_effort: str | None = typer.Option(
        None,
        "--reasoning-effort",
        help=(
            "OpenAI reasoning effort for Auto QA only. Defaults to "
            "LBX_RL_AUTOQA_REASONING_EFFORT or xhigh."
        ),
    ),
    proof_path: Path | None = typer.Option(
        None,
        "--proof-path",
        exists=True,
        dir_okay=False,
        help="Existing build proof path. Defaults to <problem-dir>/.alignerr/build_proof.json.",
    ),
    output_proof_path: Path | None = typer.Option(
        None,
        "--output-proof-path",
        dir_okay=False,
        help="Write a proof copy with harness_result.auto_qa at this path.",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        dir_okay=False,
        help="Write the normalized Auto QA result JSON at this path.",
    ),
    require: bool = typer.Option(
        False,
        "--require",
        help=(
            "Exit non-zero if Auto QA is skipped (missing API key / no source "
            "dir). Model/parse infra errors are advisory (non-blocking)."
        ),
    ),
) -> None:
    """Run only Auto QA using the submitted build proof context."""
    load_env_file()
    try:
        run_auto_qa_only(
            load_problem_dir(problem_dir),
            proof_path=proof_path,
            output_proof_path=output_proof_path,
            output_json=output_json,
            model=model,
            reasoning_effort=reasoning_effort,
            require=require,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        console.print(f"[red]Auto QA failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command("run-taiga")
def run_taiga(
    metadata: Path = typer.Option(..., "--metadata", "-m", exists=True, dir_okay=False),
    source_problem_dir: Path | None = typer.Option(
        None,
        "--source-problem-dir",
        help="Original authoring task dir; required for local grading.",
        exists=True,
        file_okay=False,
    ),
    runtime: RuntimeName = typer.Option("claude-code", "--runtime"),
    model: str | None = typer.Option(None, "--model"),
    run_dir: Path = typer.Option(DEFAULT_RUN_DIR, "--run-dir"),
) -> None:
    """Run a local Boreal metadata or job payload."""
    _run(load_taiga_metadata(metadata, source_problem_dir), runtime, run_dir, model)


@app.command("run-harbor")
def run_harbor(
    task_dir: Path = typer.Option(
        ..., "--task-dir", "-d", exists=True, file_okay=False
    ),
    source_problem_dir: Path | None = typer.Option(
        None,
        "--source-problem-dir",
        help="Original authoring task dir; required for local grading.",
        exists=True,
        file_okay=False,
    ),
    runtime: RuntimeName = typer.Option("claude-code", "--runtime"),
    model: str | None = typer.Option(None, "--model"),
    run_dir: Path = typer.Option(DEFAULT_RUN_DIR, "--run-dir"),
) -> None:
    """Run a local Harbor export directory."""
    _run(load_harbor_dir(task_dir, source_problem_dir), runtime, run_dir, model)


if __name__ == "__main__":
    app()
