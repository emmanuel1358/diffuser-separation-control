"""Isolated raw-metric worker used by calibration and future sealed evaluation."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from grader_runner.worker import (
    _enter_pid_namespace,
    _harden_import_path,
    _import_grader,
    _resolve_grader_path,
)
from grading.evaluation.author import measure_task_module, task_from_module

RAW_METRICS_SCHEMA = "raw-continuous-metrics.v1"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure a calibrated TASK without loading or applying its curve."
    )
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--grader-dir", required=True, type=Path)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--result-path", required=True, type=Path)
    args = parser.parse_args(argv)

    _enter_pid_namespace()
    _harden_import_path(args.workspace)
    try:
        grader_path = _resolve_grader_path(args.grader_dir)
        module = _import_grader(grader_path)
        task = task_from_module(module)
        if task is None:
            raise RuntimeError(
                f"{grader_path} does not define a grading.evaluation ContinuousTask as TASK"
            )
        metrics = measure_task_module(
            module,
            workspace=args.workspace,
            private=args.private_dir,
        )
        _write(
            args.result_path,
            {
                "schema_version": RAW_METRICS_SCHEMA,
                "task_spec_sha256": task.spec_sha256,
                "metrics": metrics,
            },
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - process boundary
        _write(
            args.result_path,
            {
                "schema_version": RAW_METRICS_SCHEMA,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
