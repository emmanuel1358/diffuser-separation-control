"""Mandatory declarative evaluation protocol for deterministic rubric tasks."""

from __future__ import annotations

import hashlib
import inspect
import json
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from grading.evaluation.artifacts import (
    ArtifactSpec,
    TrustedJson,
    trusted_fixture_specs,
)
from grading.evaluation.context import EvaluationContext, workspace_artifact_digest
from grading.evaluation.plan import EvaluationPlan
from grading.evaluation.result import write_private_trace
from grading.faults import AgentFault, GraderFault, InfrastructureFault
from grading.grade import Grade
from grading.numeric import (
    NumericContractError,
    finite_number,
    normalized_weights,
    safe_mean,
    safe_ratio,
    score01,
)

RUBRIC_PROTOCOL = "declarative-rubric.v1"
RubricEvaluator = Callable[["RubricContext"], "RubricEvaluation | Mapping[str, Any]"]


def _evaluator_digest(evaluate: RubricEvaluator) -> str:
    """Bind the plan to every Python source file in the scorer module directory."""
    source_path = inspect.getsourcefile(evaluate)
    if source_path:
        try:
            root = Path(source_path).parent
            digest = hashlib.sha256()
            for path in sorted(root.glob("*.py"), key=lambda item: item.name):
                digest.update(path.name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
            return digest.hexdigest()
        except OSError:
            pass
    try:
        source = inspect.getsource(evaluate)
    except (OSError, TypeError):
        source = f"{evaluate.__module__}:{evaluate.__qualname__}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RubricCriterion:
    id: str
    weight: float = 1.0
    description: str = ""
    required: bool = False
    pass_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid rubric criterion id {self.id!r}")
        finite_number(self.weight, label=f"criterion {self.id!r} weight", minimum=0.0)
        if self.weight <= 0.0:
            raise ValueError(f"criterion {self.id!r} weight must be positive")
        finite_number(
            self.pass_threshold,
            label=f"criterion {self.id!r} pass_threshold",
            minimum=0.0,
            maximum=1.0,
        )

    def spec_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "weight": self.weight,
            "description": self.description or self.id,
            "required": self.required,
            "pass_threshold": self.pass_threshold,
        }


@dataclass(frozen=True)
class RubricEvaluation:
    """Pure domain callback output; aggregation remains framework-owned."""

    subscores: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RubricContext:
    """Typed values and safe shared operations supplied to rubric callbacks."""

    candidate: Any
    fixtures: Mapping[str, Any]
    workspace: Path
    private: Path
    trajectory: Any
    evaluation: EvaluationContext

    def fixture(self, name: str) -> Any:
        if name not in self.fixtures:
            raise GraderFault(f"rubric requested undeclared trusted fixture {name!r}")
        return self.fixtures[name]

    def reject_candidate(self, message: str) -> None:
        raise AgentFault(message)

    def grader_failure(self, message: str) -> None:
        raise GraderFault(message)

    def candidate_operation(
        self, label: str, operation: Callable[..., Any], *args, **kwargs
    ):
        """Run candidate-dependent domain parsing under the shared AgentFault boundary."""
        try:
            return operation(*args, **kwargs)
        except (GraderFault, InfrastructureFault):
            raise
        except Exception as exc:  # noqa: BLE001 - candidate parser boundary
            raise AgentFault(f"{label} failed: {type(exc).__name__}: {exc}") from exc

    def trusted_operation(
        self, label: str, operation: Callable[..., Any], *args, **kwargs
    ):
        """Run trusted domain logic under the shared GraderFault boundary."""
        try:
            return operation(*args, **kwargs)
        except (AgentFault, InfrastructureFault, GraderFault):
            raise
        except Exception as exc:  # noqa: BLE001 - trusted callback boundary
            raise GraderFault(f"{label} failed: {type(exc).__name__}: {exc}") from exc

    def number(
        self,
        value: Any,
        *,
        label: str,
        source: Literal["candidate", "trusted"] = "candidate",
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        try:
            return finite_number(
                value,
                label=label,
                minimum=minimum,
                maximum=maximum,
            )
        except NumericContractError as exc:
            fault = AgentFault if source == "candidate" else GraderFault
            raise fault(str(exc)) from exc

    def ratio(
        self,
        numerator: Any,
        denominator: Any,
        *,
        label: str,
        zero: Literal["zero", "one", "agent_fault", "grader_fault"] = "grader_fault",
    ) -> float:
        if zero in {"zero", "one"}:
            policy: Literal["zero", "one", "error"] = zero
        else:
            policy = "error"
        try:
            return safe_ratio(
                numerator,
                denominator,
                label=label,
                zero=policy,
            )
        except NumericContractError as exc:
            fault = AgentFault if zero == "agent_fault" else GraderFault
            raise fault(str(exc)) from exc

    def mean(
        self,
        values: list[Any] | tuple[Any, ...],
        *,
        label: str,
        empty: Literal["zero", "agent_fault", "grader_fault"] = "grader_fault",
    ) -> float:
        policy: Literal["zero", "error"] = "zero" if empty == "zero" else "error"
        try:
            return safe_mean(values, label=label, empty=policy)
        except NumericContractError as exc:
            fault = AgentFault if empty == "agent_fault" else GraderFault
            raise fault(str(exc)) from exc

    def run_solver(
        self,
        cmd: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float = 120.0,
        max_output_bytes: int = 16 * 1024 * 1024,
    ):
        """Run a trusted domain solver through the shared bounded process API."""
        from grading.helpers import run_trusted_solver

        return run_trusted_solver(
            cmd,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
        )

    def policy(
        self,
        *,
        timeout_s: float = 5.0,
        first_call_timeout_s: float | None = None,
        cwd: str | Path | None = None,
    ):
        """Open the declared regular-file artifact in a sandboxed policy worker."""
        from grading.evaluation.artifacts import SubmittedFile
        from grading.helpers import run_policy

        if not isinstance(self.candidate, SubmittedFile):
            raise GraderFault(
                "context.policy() requires a RegularFileArtifact declaration"
            )
        return run_policy(
            self.candidate.original_path,
            timeout_s=timeout_s,
            first_call_timeout_s=first_call_timeout_s,
            cwd=cwd,
            submitted_snapshot=self.candidate.path,
        )


@dataclass(frozen=True)
class RubricTask:
    """One hash-bound rubric registration invoked directly by the grader worker."""

    artifact: ArtifactSpec
    criteria: tuple[RubricCriterion, ...]
    evaluate: RubricEvaluator
    fixtures: Mapping[str, TrustedJson] = field(default_factory=dict)
    scoring_mode: Literal["weighted", "binary"] = "weighted"
    security_tier: str = "sealed_rescore"

    def __post_init__(self) -> None:
        if not callable(self.evaluate):
            raise TypeError("RubricTask evaluate must be callable")
        if not self.criteria:
            raise ValueError("RubricTask must declare at least one criterion")
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError(f"RubricTask criterion ids must be unique: {ids}")
        if self.scoring_mode not in {"weighted", "binary"}:
            raise ValueError("RubricTask scoring_mode must be 'weighted' or 'binary'")
        if self.security_tier not in {
            "sealed_rescore",
            "sealed_challenge",
            "custom_reviewed",
        }:
            raise ValueError(f"invalid rubric security tier {self.security_tier!r}")
        normalized_weights(
            {criterion.id: criterion.weight for criterion in self.criteria}
        )
        if any(not name for name in self.fixtures):
            raise ValueError("RubricTask fixture names must be non-empty")

    def spec_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rubric-task.v1",
            "protocol": RUBRIC_PROTOCOL,
            "artifact": self.artifact.spec_dict(),
            "criteria": [criterion.spec_dict() for criterion in self.criteria],
            "fixtures": trusted_fixture_specs(self.fixtures),
            "evaluator_sha256": _evaluator_digest(self.evaluate),
            "scoring_mode": self.scoring_mode,
            "security_tier": self.security_tier,
        }

    @property
    def spec_sha256(self) -> str:
        payload = json.dumps(
            self.spec_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def evaluation_plan(self) -> EvaluationPlan:
        return EvaluationPlan(
            task_spec_sha256=self.spec_sha256,
            security_tier=self.security_tier,
            evidence={
                "type": RUBRIC_PROTOCOL,
                "artifact": self.artifact.spec_dict(),
                "aggregation": self.scoring_mode,
            },
            metric_ids=(),
            protocol=RUBRIC_PROTOCOL,
            decision_ids=tuple(criterion.id for criterion in self.criteria),
        )

    def compute_score(
        self,
        workspace: Path,
        trajectory: Any = None,
        private: Path = Path("/mcp_server/data"),
    ) -> Grade:
        return self.grade(
            workspace=workspace,
            trajectory=trajectory,
            private=private,
        )

    def grade(
        self,
        *,
        workspace: Path,
        trajectory: Any = None,
        private: Path = Path("/mcp_server/data"),
    ) -> Grade:
        workspace = Path(workspace)
        private = Path(private)
        try:
            candidate_digest = workspace_artifact_digest(workspace)
        except (OSError, ValueError) as exc:
            raise AgentFault(f"could not commit submitted artifact: {exc}") from exc

        candidate = self.artifact.load(workspace)
        fixtures = {
            name: fixture.load(private) for name, fixture in self.fixtures.items()
        }
        evaluation_context = EvaluationContext.create_from_artifact_digest(
            task_digest=self.spec_sha256,
            candidate_digest=candidate_digest,
        )
        context = RubricContext(
            candidate=candidate,
            fixtures=fixtures,
            workspace=workspace,
            private=private,
            trajectory=trajectory,
            evaluation=evaluation_context,
        )

        try:
            raw = self.evaluate(context)
        except (AgentFault, GraderFault, InfrastructureFault):
            raise
        except Exception as exc:  # noqa: BLE001 - production no-free-veto backstop
            return self._unclassified_failure(exc, evaluation_context)

        if isinstance(raw, RubricEvaluation):
            raw_subscores = dict(raw.subscores)
            author_metadata = dict(raw.metadata)
        elif isinstance(raw, Mapping):
            raw_subscores = dict(raw)
            author_metadata = {}
        else:
            raise GraderFault(
                "RubricTask evaluate must return RubricEvaluation or a criterion mapping"
            )

        expected = {criterion.id for criterion in self.criteria}
        actual = set(raw_subscores)
        if actual != expected:
            raise GraderFault(
                "RubricTask evaluator criterion mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

        try:
            subscores = {
                criterion.id: score01(
                    raw_subscores[criterion.id],
                    label=f"criterion {criterion.id!r}",
                )
                for criterion in self.criteria
            }
        except NumericContractError as exc:
            return self._unclassified_failure(exc, evaluation_context)

        weights = normalized_weights(
            {criterion.id: criterion.weight for criterion in self.criteria}
        )
        criterion_logs = {
            criterion.id: {
                "grading_type": "deterministic",
                "criterion_id": criterion.id,
                "criterion": criterion.description or criterion.id,
                "description": criterion.description or criterion.id,
                "label": criterion.description or criterion.id,
                "actual": subscores[criterion.id],
                "passed": subscores[criterion.id] >= criterion.pass_threshold,
                "reasoning": criterion.description or criterion.id,
            }
            for criterion in self.criteria
        }
        gate_failed = any(
            criterion.required and subscores[criterion.id] < criterion.pass_threshold
            for criterion in self.criteria
        )
        receipt = {
            "schema_version": "rubric-evaluation-receipt.v1",
            "protocol": RUBRIC_PROTOCOL,
            "task_spec_sha256": self.spec_sha256,
            "plan_sha256": self.evaluation_plan.sha256,
            "seed_commitment": evaluation_context.commitment,
            "attested": evaluation_context.attested,
            "decisions": {
                criterion.id: {
                    "accepted": subscores[criterion.id] >= criterion.pass_threshold,
                    "reason": (
                        "criterion_passed"
                        if subscores[criterion.id] >= criterion.pass_threshold
                        else "criterion_failed"
                    ),
                }
                for criterion in self.criteria
            },
        }
        metadata = {
            **author_metadata,
            "return_shape": "declarative_rubric",
            "security_tier": self.security_tier,
            "task_spec_sha256": self.spec_sha256,
            "evaluation": receipt,
        }
        try:
            write_private_trace(
                protocol=RUBRIC_PROTOCOL,
                plan_sha256=self.evaluation_plan.sha256,
                seed_commitment=evaluation_context.commitment,
                targets={
                    criterion.id: {
                        "score": subscores[criterion.id],
                        "weight": weights[criterion.id],
                        "required": criterion.required,
                        "passed": criterion_logs[criterion.id]["passed"],
                    }
                    for criterion in self.criteria
                },
                replay={
                    "nonce": evaluation_context.nonce,
                    "artifact_digest": candidate_digest,
                },
            )
        except (OSError, RuntimeError) as exc:
            raise GraderFault(
                f"could not write rubric evaluation trace: {exc}"
            ) from exc

        return Grade(
            subscores=subscores,
            weights=weights,
            scoring_mode=self.scoring_mode,
            criterion_logs=criterion_logs,
            metadata=metadata,
            headline_score_override=0.0 if gate_failed else None,
        )

    def _unclassified_failure(
        self,
        exc: Exception,
        context: EvaluationContext,
    ) -> Grade:
        message = f"{type(exc).__name__}: {exc}"
        private_traceback = traceback.format_exc()
        try:
            write_private_trace(
                protocol=RUBRIC_PROTOCOL,
                plan_sha256=self.evaluation_plan.sha256,
                seed_commitment=context.commitment,
                targets={
                    "unclassified_grader_crash": {
                        "error": message,
                        "traceback": private_traceback,
                    }
                },
                replay={
                    "nonce": context.nonce,
                    "artifact_digest": context.artifact_digest,
                },
            )
        except (OSError, RuntimeError):
            pass
        return Grade(
            subscores={"unclassified_grader_crash": 0.0},
            weights={"unclassified_grader_crash": 1.0},
            headline_score_override=0.0,
            criterion_logs={
                "unclassified_grader_crash": {
                    "grading_type": "runtime_backstop",
                    "criterion_id": "unclassified_grader_crash",
                    "description": "Unclassified rubric evaluator failure",
                    "error_type": "unclassified_grader_crash",
                    "error_message": message,
                    "passed": False,
                    "reasoning": message,
                }
            },
            metadata={
                "return_shape": "declarative_rubric_failure",
                "task_spec_sha256": self.spec_sha256,
                "critical_operator_alert": True,
                "evaluation": {
                    "schema_version": "rubric-evaluation-receipt.v1",
                    "protocol": RUBRIC_PROTOCOL,
                    "plan_sha256": self.evaluation_plan.sha256,
                    "seed_commitment": context.commitment,
                    "attested": context.attested,
                    "decisions": {},
                },
            },
            env_internal_failure=False,
        )


__all__ = [
    "RUBRIC_PROTOCOL",
    "RubricContext",
    "RubricCriterion",
    "RubricEvaluation",
    "RubricTask",
]
