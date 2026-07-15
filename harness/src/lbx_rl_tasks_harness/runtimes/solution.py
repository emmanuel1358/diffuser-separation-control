from __future__ import annotations

import shutil
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path

from lbx_rl_tasks_harness.env import harness_subprocess_env
from lbx_rl_tasks_harness.models import HarnessProblem

TAIGA_PLATFORM = "linux/amd64"


@contextmanager
def _isolated_tmp_output() -> Path:
    """Provide a fresh host `/tmp/output` for reference solutions.

    The checked-in example solutions intentionally write to the production
    path `/tmp/output`. To run them locally without Docker, briefly move any
    existing host path aside and restore it after the run.
    """
    output = Path("/tmp/output")
    backup = Path(f"/tmp/output.lbx-harness-backup-{uuid.uuid4().hex}")
    had_existing = output.exists()
    if had_existing:
        output.rename(backup)
    output.mkdir(parents=True, exist_ok=True)
    try:
        yield output
    finally:
        if output.exists():
            shutil.rmtree(output)
        if had_existing:
            backup.rename(output)


def _copy_outputs(tmp_output: Path, workspace: Path, problem: HarnessProblem) -> None:
    for spec in problem.outputs:
        source = tmp_output / spec.relative_path
        destination = workspace / spec.relative_path
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        elif spec.required:
            raise FileNotFoundError(
                f"solution did not create required output: {spec.path}"
            )


def run_solution(
    problem: HarnessProblem, workspace: Path, transcript_path: Path
) -> None:
    if problem.source_problem_dir is None:
        raise ValueError("solution runtime requires a source problem directory")
    solve = problem.source_problem_dir / "solution" / "solve.sh"
    if not solve.exists():
        raise FileNotFoundError(f"solution script not found: {solve}")

    env = harness_subprocess_env({"LBT_OUTPUT_DIR": "/tmp/output"})
    with _isolated_tmp_output() as tmp_output:
        proc = subprocess.run(
            ["bash", str(solve)],
            cwd=problem.source_problem_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=problem.metadata.get("agent", {}).get("timeout_sec", 3600),
            check=False,
        )
        transcript_path.write_text(
            "\n".join(
                [
                    "[solution stdout]",
                    proc.stdout,
                    "[solution stderr]",
                    proc.stderr,
                ]
            )
        )
        if proc.returncode != 0:
            raise RuntimeError(f"solution exited with status {proc.returncode}")
        _copy_outputs(tmp_output, workspace, problem)


def run_solution_in_container(
    problem: HarnessProblem,
    workspace: Path,
    verifier_dir: Path,
    transcript_path: Path,
    *,
    render: bool,
) -> float:
    """Run the oracle solve + grader (+ optional reviewer render) INSIDE the
    built task image, where base-image-only engines (OpenFOAM/SU2/Meep/OpenROAD)
    are available. Copies /tmp/output (incl. rendering.mp4) and the verifier
    reward.json back to the host. Returns the oracle score.
    """
    # Imported here to avoid a hard docker dependency for host-only runtimes.
    from lbx_rl_tasks_harness.docker import build_task_image

    if problem.source_problem_dir is None:
        raise ValueError(
            "in-container ground truth requires a source problem directory"
        )
    src = problem.source_problem_dir.resolve()
    image_tag = build_task_image(problem).image_tag

    container_out = workspace.parent / "container_out"
    (container_out / "output").mkdir(parents=True, exist_ok=True)
    (container_out / "verifier").mkdir(parents=True, exist_ok=True)
    verifier_dir.mkdir(parents=True, exist_ok=True)

    # Use the task-configured render hook (same as the host path), not a
    # hard-coded script, so a custom [ground_truth].render_command is honored.
    render_cmd = problem.ground_truth.render_command.strip()
    render_line = render_cmd if (render and render_cmd) else "true"
    # `cp` runs under `set -e` (no silent `|| true`) so a failed copy fails the
    # container step instead of leaving the host workspace without artifacts.
    script = f"""
set -e
export LBT_OUTPUT_DIR=/tmp/output
mkdir -p /tmp/output /tmp/verifier
cd /host_task
bash solution/solve.sh
/opt/lbx-runtime/.venv/bin/python /runtime/run_grader.py \
  --workspace /tmp/output --grader-dir /mcp_server/grader \
  --private-dir /mcp_server/data --output-dir /tmp/verifier \
  --transcript /tmp/verifier/transcript.txt
{render_line}
cp -a /tmp/output/. /host_out/output/
cp -a /tmp/verifier/. /host_out/verifier/
"""
    # The single docker window covers oracle solve + grading + render. Budget
    # them separately: grading ~ verifier.timeout_sec, render ~ agent.timeout_sec,
    # plus margin for the (small) solve and container overhead.
    agent_timeout = int(problem.metadata.get("agent", {}).get("timeout_sec") or 3600)
    verifier_timeout = int(
        problem.metadata.get("verifier", {}).get("timeout_sec") or 600
    )
    timeout = agent_timeout + verifier_timeout + 300
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            TAIGA_PLATFORM,
            "-v",
            f"{src}:/host_task:ro",
            "-v",
            f"{container_out}:/host_out",
            "-e",
            "LBT_OUTPUT_DIR=/tmp/output",
            image_tag,
            "bash",
            "-lc",
            script,
        ],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    transcript_path.write_text(
        "\n".join(
            [
                "[in-container ground truth stdout]",
                proc.stdout,
                "[in-container ground truth stderr]",
                proc.stderr,
            ]
        )
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"in-container ground truth failed (status {proc.returncode}). "
            f"stderr tail:\n{proc.stderr[-1500:]}"
        )

    # Copy artifacts back to the host workspace + verifier dir.
    out_src = container_out / "output"
    for item in out_src.iterdir():
        dest = workspace / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    for name in ("reward.json", "reward-details.json", "reward.txt"):
        s = container_out / "verifier" / name
        if s.exists():
            shutil.copy2(s, verifier_dir / name)

    import json

    reward = json.loads((verifier_dir / "reward.json").read_text())
    return float(reward["score"])
