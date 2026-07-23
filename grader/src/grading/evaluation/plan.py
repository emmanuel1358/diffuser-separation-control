"""Versioned, hash-bound evaluation-plan identity."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

EVALUATION_PLAN_SCHEMA = "continuous-evaluation-plan.v1"
GENERIC_EVALUATION_PLAN_SCHEMA = "evaluation-plan.v2"
EVALUATION_PLAN_FILENAME = "evaluation.plan.json"

PlanSyncStatus = Literal[
    "written",
    "unchanged",
    "missing",
    "stale",
    "not_rubric",
    "unavailable",
]


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_plan_sha256(payload: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "plan_sha256"}
    return hashlib.sha256(_canonical(canonical)).hexdigest()


def validate_serialized_plan(payload: Mapping[str, Any]) -> str:
    expected = canonical_plan_sha256(payload)
    if payload.get("plan_sha256") != expected:
        raise ValueError(
            "evaluation plan digest mismatch: "
            f"expected {expected}, got {payload.get('plan_sha256')}"
        )
    tier = payload.get("security_tier")
    if tier not in {"sealed_rescore", "sealed_challenge", "custom_reviewed"}:
        raise ValueError(f"invalid evaluation security tier {tier!r}")
    schema = payload.get("schema_version")
    if schema not in {EVALUATION_PLAN_SCHEMA, GENERIC_EVALUATION_PLAN_SCHEMA}:
        raise ValueError(f"invalid evaluation plan schema {schema!r}")
    if schema == GENERIC_EVALUATION_PLAN_SCHEMA:
        protocol = payload.get("protocol")
        decisions = payload.get("decision_ids")
        if not isinstance(protocol, str) or not protocol:
            raise ValueError("evaluation-plan.v2 requires a protocol")
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("evaluation-plan.v2 requires decision_ids")
    return expected


@dataclass(frozen=True)
class EvaluationPlan:
    """Public, immutable identity of the protocol that must run."""

    task_spec_sha256: str
    security_tier: str
    evidence: Mapping[str, Any]
    metric_ids: tuple[str, ...]
    protocol: str = "continuous"
    decision_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        if self.protocol != "continuous" or self.decision_ids:
            return {
                "schema_version": GENERIC_EVALUATION_PLAN_SCHEMA,
                "protocol": self.protocol,
                "task_spec_sha256": self.task_spec_sha256,
                "security_tier": self.security_tier,
                "evidence": dict(self.evidence),
                "metric_ids": list(self.metric_ids),
                "decision_ids": list(self.decision_ids),
            }
        return {
            "schema_version": EVALUATION_PLAN_SCHEMA,
            "task_spec_sha256": self.task_spec_sha256,
            "security_tier": self.security_tier,
            "evidence": dict(self.evidence),
            "metric_ids": list(self.metric_ids),
        }

    @property
    def sha256(self) -> str:
        return canonical_plan_sha256(self.to_dict())


@dataclass(frozen=True)
class EvaluationPlanSyncResult:
    """Outcome of ensuring ``scorer/evaluation.plan.json`` matches ``TASK``."""

    path: Path
    status: PlanSyncStatus
    plan_sha256: str | None = None
    message: str = ""

    @property
    def wrote(self) -> bool:
        return self.status == "written"


def evaluation_plan_path(problem_dir: Path) -> Path:
    return Path(problem_dir) / "scorer" / EVALUATION_PLAN_FILENAME


def serialized_evaluation_plan(plan: EvaluationPlan) -> dict[str, Any]:
    return {**plan.to_dict(), "plan_sha256": plan.sha256}


def write_evaluation_plan_atomic(path: Path, plan: EvaluationPlan) -> None:
    """Atomically replace a sealed evaluation plan after canonical serialization."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialized_evaluation_plan(plan)
    data = _canonical(payload)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_rubric_evaluation_plan(problem_dir: Path) -> EvaluationPlan | None:
    """Load ``TASK.evaluation_plan`` when the scorer declares a ``RubricTask``."""

    scorer = Path(problem_dir) / "scorer" / "compute_score.py"
    if not scorer.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        f"lbx_rubric_plan_{hashlib.sha256(str(scorer).encode()).hexdigest()[:12]}",
        scorer,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {scorer}")
    module = importlib.util.module_from_spec(spec)
    inserted: list[str] = []
    for path in (str(scorer.parent),):
        if path not in sys.path:
            sys.path.insert(0, path)
            inserted.append(path)
    try:
        spec.loader.exec_module(module)
    finally:
        for path in inserted:
            if path in sys.path:
                sys.path.remove(path)
    registration = getattr(module, "TASK", None)
    try:
        from grading.evaluation.rubric import RubricTask
    except Exception:
        return None
    if not isinstance(registration, RubricTask):
        return None
    return registration.evaluation_plan


def _sync_evaluation_plan(
    problem_dir: Path, *, write: bool
) -> EvaluationPlanSyncResult:
    """Internal sync helper. Callers must choose refresh vs check explicitly."""

    problem_dir = Path(problem_dir).resolve()
    path = evaluation_plan_path(problem_dir)
    try:
        plan = load_rubric_evaluation_plan(problem_dir)
    except Exception as exc:
        return EvaluationPlanSyncResult(
            path=path,
            status="unavailable",
            message=f"could not load RubricTask evaluation plan: {exc}",
        )
    if plan is None:
        return EvaluationPlanSyncResult(
            path=path,
            status="not_rubric",
            message="scorer does not declare TASK = RubricTask(...)",
        )

    expected = serialized_evaluation_plan(plan)
    expected_sha = plan.sha256
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            stale_reason = f"unreadable ({exc})"
        else:
            try:
                current_sha = validate_serialized_plan(current)
            except Exception as exc:
                stale_reason = f"invalid ({exc})"
            else:
                if current_sha == expected_sha and current == expected:
                    return EvaluationPlanSyncResult(
                        path=path,
                        status="unchanged",
                        plan_sha256=expected_sha,
                        message="scorer/evaluation.plan.json matches TASK",
                    )
                stale_reason = "digest or payload does not match TASK.evaluation_plan"
        if write:
            write_evaluation_plan_atomic(path, plan)
            return EvaluationPlanSyncResult(
                path=path,
                status="written",
                plan_sha256=expected_sha,
                message=(
                    "rewrote scorer/evaluation.plan.json from TASK "
                    f"({stale_reason}); commit this generated file"
                ),
            )
        return EvaluationPlanSyncResult(
            path=path,
            status="stale",
            plan_sha256=expected_sha,
            message=(
                "scorer/evaluation.plan.json is stale "
                f"({stale_reason}); regenerate with "
                "`uv run lbx-rl-harness reference --problem-dir <task>` "
                "or `uv run python scripts/write_evaluation_plan.py <task>`"
            ),
        )

    if write:
        write_evaluation_plan_atomic(path, plan)
        return EvaluationPlanSyncResult(
            path=path,
            status="written",
            plan_sha256=expected_sha,
            message=(
                "wrote scorer/evaluation.plan.json from TASK; "
                "commit this generated file"
            ),
        )
    return EvaluationPlanSyncResult(
        path=path,
        status="missing",
        plan_sha256=expected_sha,
        message=(
            "declarative rubric grader must ship scorer/evaluation.plan.json; "
            "generate it with "
            "`uv run lbx-rl-harness reference --problem-dir <task>` "
            "or `uv run python scripts/write_evaluation_plan.py <task>`"
        ),
    )


def refresh_evaluation_plan(problem_dir: Path) -> EvaluationPlanSyncResult:
    """Write ``scorer/evaluation.plan.json`` from ``TASK`` (authoring / CI seal)."""

    return _sync_evaluation_plan(problem_dir, write=True)


def check_evaluation_plan(problem_dir: Path) -> EvaluationPlanSyncResult:
    """Verify the sealed plan matches ``TASK`` without mutating the tree."""

    return _sync_evaluation_plan(problem_dir, write=False)


__all__ = [
    "EVALUATION_PLAN_FILENAME",
    "EVALUATION_PLAN_SCHEMA",
    "GENERIC_EVALUATION_PLAN_SCHEMA",
    "EvaluationPlan",
    "EvaluationPlanSyncResult",
    "canonical_plan_sha256",
    "check_evaluation_plan",
    "evaluation_plan_path",
    "load_rubric_evaluation_plan",
    "refresh_evaluation_plan",
    "serialized_evaluation_plan",
    "validate_serialized_plan",
    "write_evaluation_plan_atomic",
]
