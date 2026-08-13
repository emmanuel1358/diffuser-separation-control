#!/usr/bin/env python3
"""Write the canonical evaluation plan for a task's registered TASK object."""

from __future__ import annotations

import argparse
from pathlib import Path

from grading.evaluation.plan import refresh_evaluation_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate scorer/evaluation.plan.json from a registered TASK. "
            "Prefer letting `lbx-rl-harness reference` / `ground-truth` do this; "
            "Trusted CI seals via an explicit refresh step before validate."
        )
    )
    parser.add_argument("problem_dir", type=Path)
    args = parser.parse_args()
    result = refresh_evaluation_plan(args.problem_dir.resolve())
    if result.status in {"not_rubric", "unavailable"}:
        raise SystemExit(result.message or f"could not write plan ({result.status})")
    print(result.path)
    if result.message:
        print(result.message)


if __name__ == "__main__":
    main()
