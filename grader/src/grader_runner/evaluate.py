"""Shared result semantics for Harbor, Boreal, and local grader lanes."""

from __future__ import annotations

from typing import Any

from grading import Grade, normalize_compute_score_return
from grading.rubric_builder import RubricBuilder


def wire_transcript(value: object, transcript: str) -> object:
    if isinstance(value, RubricBuilder):
        value.transcript = transcript
        return value.grade()
    return value


def agent_fault_grade(message: str) -> Grade:
    return Grade(
        subscores={"agent_fault": 0.0},
        weights={"agent_fault": 1.0},
        headline_score_override=0.0,
        metadata={"return_shape": "agent_fault", "agent_fault": message},
        criterion_logs={
            "agent_fault": {
                "grading_type": "agent_fault",
                "error_type": "agent_fault",
                "error_message": message,
                "passed": False,
                "reasoning": message,
            }
        },
        env_internal_failure=False,
    )


def failure_grade(error_type: str, message: str, traceback_text: str = "") -> Grade:
    metadata: dict[str, Any] = {"return_shape": "error"}
    if traceback_text:
        metadata["traceback"] = traceback_text
    return Grade(
        subscores={"run_grader": 0.0},
        weights={"run_grader": 1.0},
        headline_score_override=0.0,
        metadata=metadata,
        criterion_logs={
            "run_grader": {
                "grading_type": "runner",
                "error_type": error_type,
                "error_message": message,
                "passed": False,
                "reasoning": message,
            }
        },
        env_internal_failure=True,
        env_internal_failure_logs=[message],
    )


def unclassified_failure_grade(message: str, _traceback_text: str = "") -> Grade:
    """Keep an untyped candidate-evaluation crash as zero and alert operators."""
    grade = agent_fault_grade(message)
    grade.metadata = {
        "return_shape": "unclassified_grader_crash",
        "critical_operator_alert": True,
        "error": message,
    }
    grade.criterion_logs = {
        "unclassified_grader_crash": {
            "grading_type": "runtime_backstop",
            "error_type": "unclassified_grader_crash",
            "error_message": message,
            "passed": False,
            "reasoning": message,
        }
    }
    grade.subscores = {"unclassified_grader_crash": 0.0}
    grade.weights = {"unclassified_grader_crash": 1.0}
    return grade


def normalize_result_payload(value: object, *, transcript: str = "") -> dict[str, Any]:
    wired = wire_transcript(value, transcript)
    return normalize_compute_score_return(wired).to_dict()


__all__ = [
    "agent_fault_grade",
    "failure_grade",
    "normalize_result_payload",
    "unclassified_failure_grade",
    "wire_transcript",
]
