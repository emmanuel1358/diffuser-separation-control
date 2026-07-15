from __future__ import annotations

from pathlib import Path
from typing import Any


def repo_root(problem_dir: Path) -> Path:
    for parent in [problem_dir, *problem_dir.parents]:
        if (parent / "pyproject.toml").exists() and (
            parent / "docs" / "RUBRIC_GUIDANCE.md"
        ).exists():
            return parent
    return problem_dir.parent.parent


def read_limited(path: Path, limit: int) -> str:
    if not path.exists():
        return f"[missing: {path}]"
    text = path.read_text(errors="replace")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def normalize_review_checks(
    raw_checks: Any,
    expected_checks: list[str],
    *,
    missing_recommendation: str,
) -> list[dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    if isinstance(raw_checks, list):
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            check = str(item.get("check") or item.get("name") or "").strip()
            if not check:
                continue
            status = str(item.get("status") or "warning").strip().lower()
            if status not in {"pass", "warning", "fail"}:
                status = "warning"
            by_name[check] = {
                "check": check,
                "status": status,
                "reason": str(item.get("reason") or ""),
                "recommendation": str(item.get("recommendation") or ""),
            }

    normalized = []
    for check in expected_checks:
        normalized.append(
            by_name.get(
                check,
                {
                    "check": check,
                    "status": "warning",
                    "reason": "Model response did not include this check.",
                    "recommendation": missing_recommendation,
                },
            )
        )
    return normalized
