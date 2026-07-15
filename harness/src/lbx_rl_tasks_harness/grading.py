from __future__ import annotations

import json
from pathlib import Path

from grader_runner.run_grader import main as run_grader_main

from lbx_rl_tasks_harness.models import HarnessProblem


def grade_workspace(
    problem: HarnessProblem, workspace: Path, output_dir: Path, transcript: Path
) -> float:
    grader_dir, private_dir = problem.require_grader()
    exit_code = run_grader_main(
        [
            "--workspace",
            str(workspace),
            "--grader-dir",
            str(grader_dir),
            "--private-dir",
            str(private_dir),
            "--output-dir",
            str(output_dir),
            "--transcript",
            str(transcript),
        ]
    )
    if exit_code != 0:
        raise RuntimeError(f"run_grader exited with status {exit_code}")
    reward_path = output_dir / "reward.json"
    reward = json.loads(reward_path.read_text())
    return float(reward["score"])
