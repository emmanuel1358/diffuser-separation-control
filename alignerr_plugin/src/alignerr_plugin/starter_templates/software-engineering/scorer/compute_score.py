"""Declarative repository grader for the software engineering starter."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

from grading.evaluation import (
    CandidateCommandSpec,
    RubricCriterion,
    RubricTask,
    TrustedJson,
    WorkspaceArtifact,
)

MAX_DRIVER_BYTES = 128 * 1024
MAX_RESULT_BYTES = 4096


def _load_driver(path: Path) -> str:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError("hidden candidate driver must be a regular file")
    if not 1 <= info.st_size <= MAX_DRIVER_BYTES:
        raise ValueError("hidden candidate driver has an invalid size")
    return path.read_bytes().decode("utf-8", errors="strict")


def _encode_cases(value: Any) -> tuple[bytes, int]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "cases"}
        or value.get("schema_version") != "slug-normalizer-cases.v1"
    ):
        raise ValueError("hidden cases use the wrong schema")
    cases = value.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 64:
        raise ValueError("hidden cases must contain 1..64 entries")
    for case in cases:
        if (
            not isinstance(case, dict)
            or set(case) != {"input", "expected"}
            or not isinstance(case["input"], str)
            or not isinstance(case["expected"], str)
        ):
            raise ValueError("hidden cases contain an invalid entry")
    data = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return data, len(cases)


def _validate_result(value: Any, expected_total: int) -> dict[str, int | str]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "passed",
        "total",
    }:
        raise ValueError("candidate driver returned the wrong result shape")
    if value["schema_version"] != "slug-normalizer-result.v1":
        raise ValueError("candidate driver returned the wrong result schema")
    if (
        type(value["passed"]) is not int
        or type(value["total"]) is not int
        or value["total"] != expected_total
        or not 0 <= value["passed"] <= value["total"]
    ):
        raise ValueError("candidate driver returned invalid result counts")
    return value


def evaluate(context):
    fixture_bytes, case_count = context.trusted_operation(
        "hidden case validation",
        _encode_cases,
        context.fixture("cases"),
    )
    driver_source = context.trusted_operation(
        "hidden candidate driver loading",
        _load_driver,
        context.private / "candidate_driver.py",
    )
    suite = context.run_candidate_suite(
        CandidateCommandSpec(
            argv=("python3", "-I", "-B", "-c", driver_source),
            env=(
                ("HOME", "/tmp"),
                ("LANG", "C.UTF-8"),
                ("LC_ALL", "C.UTF-8"),
                ("PATH", "/usr/local/bin:/usr/bin:/bin"),
                ("PYTHONDONTWRITEBYTECODE", "1"),
            ),
            stdin_bytes=fixture_bytes,
            timeout_s=60.0,
            max_output_bytes=MAX_RESULT_BYTES,
            repeats=3,
            deterministic_stdout=True,
        )
    )
    raw_result = suite.attempts[0].parse_json(
        context,
        max_bytes=MAX_RESULT_BYTES,
        max_depth=8,
        max_nodes=32,
        require_object=True,
    )
    result = context.candidate_operation(
        "candidate driver result validation",
        _validate_result,
        raw_result,
        case_count,
    )
    return {
        "hidden_behavior": result["passed"] == result["total"],
        "repeatability": (
            suite.metadata.attempt_count == 3 and suite.metadata.deterministic_stdout
        ),
    }


TASK = RubricTask(
    artifact=WorkspaceArtifact(
        "repo",
        max_files=64,
        max_total_bytes=2 * 1024 * 1024,
        max_file_bytes=256 * 1024,
        clean_paths=(".git", ".pytest_cache", "__pycache__"),
        forbidden_suffixes=(
            ".a",
            ".class",
            ".dll",
            ".dylib",
            ".exe",
            ".jar",
            ".o",
            ".pyc",
            ".pyo",
            ".so",
            ".wasm",
            ".zip",
        ),
        text_suffixes=(".py", ".md", ".txt", ".toml"),
        reject_native_payloads=True,
    ),
    fixtures={"cases": TrustedJson("hidden_cases.json")},
    criteria=(
        RubricCriterion(
            id="hidden_behavior",
            weight=0.9,
            description="Slug normalization satisfies all hidden behavioral cases",
            required=True,
            pass_threshold=1.0,
        ),
        RubricCriterion(
            id="repeatability",
            weight=0.1,
            description="Repeated candidate-suite executions produce identical output",
            required=True,
            pass_threshold=1.0,
        ),
    ),
    evaluate=evaluate,
    security_tier="sealed_rescore",
)
