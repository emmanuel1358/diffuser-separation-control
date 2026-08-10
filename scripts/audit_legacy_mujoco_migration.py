#!/usr/bin/env python3
"""Exercise legacy MuJoCo projects through native, Harbor, and Taiga adapters."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from alignerr_plugin.exporters.harbor import export_harbor
from alignerr_plugin.exporters.taiga import export_taiga
from alignerr_plugin.migrations.mujoco import migrate_legacy_mujoco_task
from lbx_rl_tasks_harness.formats.harbor import load_harbor_dir
from lbx_rl_tasks_harness.formats.problem_dir import load_problem_dir

_RETIRED_RUNTIME_VENV = b"/mcp_server/.venv"
_WORLD_MCP_CHMOD = re.compile(
    r"\bchmod\s+0?755\s+[^;&\n]*?(?<!\S)/mcp_server(?=\s|\\|$)"
)


def file_contains(path: Path, needle: bytes) -> bool:
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            content = overlap + chunk
            if needle in content:
                return True
            overlap = content[-(len(needle) - 1) :] if len(needle) > 1 else b""
    return False


def path_contract_issues(task: Path) -> list[str]:
    issues: list[str] = []
    for path in sorted(task.rglob("*")):
        if (
            path.is_file()
            and not path.is_symlink()
            and file_contains(path, _RETIRED_RUNTIME_VENV)
        ):
            try:
                path.read_text()
            except UnicodeDecodeError:
                continue
            issues.append(
                f"{path.relative_to(task).as_posix()} retains /mcp_server/.venv"
            )
    harbor_test = task / "tests" / "test.sh"
    if harbor_test.is_file() and "--workspace /app" in harbor_test.read_text():
        issues.append("tests/test.sh retains Harbor workspace /app")
    dockerfile = task / "environment" / "Dockerfile"
    if dockerfile.is_file():
        dockerfile_text = dockerfile.read_text()
        if _WORLD_MCP_CHMOD.search(dockerfile_text):
            issues.append("environment/Dockerfile grants traversal on /mcp_server")
        if "chmod 0700 /mcp_server" not in dockerfile_text:
            issues.append("environment/Dockerfile does not make /mcp_server root-only")
        if "/tmp/output" not in dockerfile_text:
            issues.append("environment/Dockerfile does not provision /tmp/output")
    return issues


def discover(root: Path) -> list[Path]:
    candidates = [
        *root.glob("problems/*/task.toml"),
        *root.glob("examples/*/task.toml"),
        *root.glob("harbor_tasks/*/task.toml"),
        root / "alignerr_plugin" / "src" / "alignerr_plugin" / "starter_templates" / "mujoco" / "task.toml",
    ]
    tasks: list[Path] = []
    for task_toml in sorted({path.resolve() for path in candidates if path.is_file()}):
        with task_toml.open("rb") as handle:
            data = tomllib.load(handle)
        task_type = str((data.get("difficulty") or {}).get("task_type") or "")
        if task_type.strip().lower() == "mujoco":
            tasks.append(task_toml.parent)
    return tasks


def audit_task(source: Path, destination: Path) -> dict[str, Any]:
    destination = destination.resolve()
    entry: dict[str, Any] = {"source": str(source)}
    try:
        migration = migrate_legacy_mujoco_task(source, destination)
        problem = load_problem_dir(destination)
        path_issues = path_contract_issues(destination)
        if path_issues:
            raise ValueError("; ".join(path_issues))
        entry.update(
            {
                "migration": "pass",
                "source_layout": migration.source_layout,
                "required_resources": problem.required_resources,
                "outputs": [item.path for item in problem.outputs],
                "blockers": list(migration.blockers),
            }
        )
    except Exception as exc:  # noqa: BLE001 - audit must retain the whole matrix
        entry.update({"migration": "fail", "migration_error": f"{type(exc).__name__}: {exc}"})
        return entry

    harbor_dir = (destination.parent.parent / "harbor" / destination.name).resolve()
    try:
        export_harbor(destination, harbor_dir)
        harbor = load_harbor_dir(harbor_dir)
        entry.update(
            {
                "harbor": "pass",
                "harbor_grader": str(harbor.grader_dir.relative_to(harbor_dir)),
                "harbor_required_resources": harbor.required_resources,
            }
        )
    except Exception as exc:  # noqa: BLE001 - audit must retain the whole matrix
        entry.update({"harbor": "fail", "harbor_error": f"{type(exc).__name__}: {exc}"})

    taiga_path = destination.parent.parent / "taiga" / f"{destination.name}.json"
    taiga_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_taiga(destination, taiga_path, image_ref="local:migration-audit")
        entry["taiga_export_probe"] = "pass"
        entry["taiga"] = "pass"
    except Exception as exc:  # noqa: BLE001 - expected for an unsealed legacy scorer
        entry.update(
            {
                "taiga": ("blocked_by_migration_gates" if migration.blockers else "fail"),
                "taiga_export_probe": "fail",
                "taiga_error": f"{type(exc).__name__}: {exc}",
            }
        )
    return entry


def run(roots: list[Path], work_dir: Path) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for root_index, root in enumerate(roots):
        root = root.resolve()
        for source in discover(root):
            relative = source.relative_to(root).as_posix().replace("/", "__")
            label = f"{root_index}-{root.name}-{relative}"
            rows[label] = audit_task(source, work_dir / "native" / label)
    values = list(rows.values())
    return {
        "summary": {
            "total": len(values),
            "migration_passed": sum(row.get("migration") == "pass" for row in values),
            "harbor_passed": sum(row.get("harbor") == "pass" for row in values),
            "taiga_passed": sum(row.get("taiga") == "pass" for row in values),
            "taiga_blocked_by_migration_gates": sum(row.get("taiga") == "blocked_by_migration_gates" for row in values),
        },
        "tasks": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--require-taiga", action="store_true")
    args = parser.parse_args()

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="mujoco-migration-audit-")
        work_dir = Path(temporary.name)
    else:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    report = run(args.roots, work_dir)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    print(rendered, end="")

    summary = report["summary"]
    passed = (
        summary["total"] > 0
        and summary["migration_passed"] == summary["total"]
        and summary["harbor_passed"] == summary["total"]
    )
    if args.require_taiga:
        passed = passed and summary["taiga_passed"] == summary["total"]
    if temporary is not None:
        temporary.cleanup()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
