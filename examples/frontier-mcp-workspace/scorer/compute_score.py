from __future__ import annotations

import json
from typing import Any

from grading.evaluation import (
    CandidateCommandSpec,
    RubricCriterion,
    RubricTask,
    TrustedJson,
    WorkspaceArtifact,
)

MAX_RESULT_BYTES = 4096
DRIVER_SOURCE = r"""
import json
import runpy
import sys

request = json.loads(sys.stdin.buffer.read())
if not isinstance(request, dict) or set(request) != {"input"}:
    raise SystemExit(2)
module = runpy.run_path("processor.py")
adjudicate = module["adjudicate"]
observed = adjudicate(request["input"])
print(json.dumps(
    {
        "schema_version": "claim-observation.v1",
        "observed": observed,
    },
    sort_keys=True,
    separators=(",", ":"),
))
"""


def _validate_cases(value: Any) -> tuple[dict[str, Any], ...]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "cases"}
        or value.get("schema_version") != "claim-cases.v1"
    ):
        raise ValueError("claim fixture has the wrong schema")
    cases = value.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError("claim fixture must contain 1..32 cases")
    for case in cases:
        if (
            not isinstance(case, dict)
            or set(case) != {"claim", "expected"}
            or not isinstance(case["claim"], dict)
            or not isinstance(case["expected"], dict)
        ):
            raise ValueError("claim fixture contains an invalid case")
    return tuple(cases)


def _encode_input(claim: dict[str, Any]) -> bytes:
    return json.dumps(
        {"input": claim},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _validate_observation(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "observed"}
        or value.get("schema_version") != "claim-observation.v1"
        or not isinstance(value.get("observed"), dict)
    ):
        raise ValueError("candidate driver returned an invalid observation")
    return value["observed"]


def evaluate(context):
    cases = context.trusted_operation(
        "claim fixture validation",
        _validate_cases,
        context.fixture("cases"),
    )
    passed = 0
    deterministic = True
    for index, case in enumerate(cases, start=1):
        request_bytes = context.trusted_operation(
            f"claim case {index} request encoding",
            _encode_input,
            case["claim"],
        )
        suite = context.run_candidate_suite(
            CandidateCommandSpec(
                argv=("python3", "-I", "-B", "-c", DRIVER_SOURCE),
                env=(
                    ("HOME", "/tmp"),
                    ("LANG", "C.UTF-8"),
                    ("LC_ALL", "C.UTF-8"),
                    ("PATH", "/usr/local/bin:/usr/bin:/bin"),
                    ("PYTHONDONTWRITEBYTECODE", "1"),
                ),
                stdin_bytes=request_bytes,
                timeout_s=15,
                max_output_bytes=MAX_RESULT_BYTES,
                repeats=2,
                deterministic_stdout=True,
            )
        )
        parsed = suite.attempts[0].parse_json(
            context,
            max_bytes=MAX_RESULT_BYTES,
            max_depth=8,
            max_nodes=32,
            require_object=True,
        )
        observed = context.candidate_operation(
            f"claim case {index} observation validation",
            _validate_observation,
            parsed,
        )
        passed += int(observed == case["expected"])
        deterministic = deterministic and suite.metadata.deterministic_stdout

    return {
        "hidden_behavior": context.ratio(
            passed,
            len(cases),
            label="hidden claim cases",
            zero="grader_fault",
        ),
        "repeatability": float(deterministic),
    }


TASK = RubricTask(
    artifact=WorkspaceArtifact(
        "repo",
        max_files=64,
        max_total_bytes=2 * 1024 * 1024,
        max_file_bytes=256 * 1024,
        clean_paths=(".git", ".pytest_cache", "__pycache__"),
        forbidden_suffixes=(".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe"),
        text_suffixes=(".py", ".md", ".txt"),
        reject_native_payloads=True,
        allow_extra_workspace_entries=True,
    ),
    fixtures={"cases": TrustedJson("claim_cases.json")},
    criteria=(
        RubricCriterion(
            id="hidden_behavior",
            weight=0.9,
            description="Every synthetic hidden claim follows the deterministic policy",
            required=True,
            pass_threshold=1.0,
        ),
        RubricCriterion(
            id="repeatability",
            weight=0.1,
            description="Repeated candidate executions return identical output",
            required=True,
            pass_threshold=1.0,
        ),
    ),
    evaluate=evaluate,
    security_tier="sealed_rescore",
)
