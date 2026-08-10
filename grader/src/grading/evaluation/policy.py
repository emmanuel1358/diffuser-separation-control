"""Post-commit paired policy challenges for continuous control tasks."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from grading.calibration import PiecewiseLinearCurve
from grading.evaluation.context import EvaluationContext, workspace_artifact_digest
from grading.evaluation.plan import EvaluationPlan
from grading.evaluation.result import (
    PublicEvaluationReceipt,
    TargetDecision,
    write_private_trace,
)
from grading.faults import AgentFault
from grading.policy_runner import PolicyWorkerError, load_submitted_policy

POLICY_CHALLENGE_PROTOCOL = "paired-policy-challenge.v1"
PolicyRollout = Callable[[Any, int], float]
ControlRollout = Callable[[int], float]


def _binomial_upper_tail(*, wins: int, trials: int) -> float:
    """P[Binomial(trials, .5) >= wins], exact for paired sign evidence."""
    if not 0 <= wins <= trials:
        raise ValueError("wins must lie in [0, trials]")
    numerator = sum(math.comb(trials, value) for value in range(wins, trials + 1))
    return numerator / (2**trials)


def _bounded_quality(value: Any, *, source: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{source} returned non-finite policy quality")
    return max(0.0, min(1.0, number))


@dataclass(frozen=True)
class PolicyEvaluationTask:
    """One continuous policy task evaluated on secret post-commit scenarios."""

    policy_path: str = "policy.py"
    factory_name: str = "load_policy"
    scenarios: int = 32
    alpha: float = 0.01
    reference_quality: float = 0.95
    call_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        relative = Path(self.policy_path)
        if (
            not self.policy_path
            or relative.is_absolute()
            or ".." in relative.parts
            or not self.factory_name
        ):
            raise ValueError("invalid policy artifact descriptor")
        if self.scenarios < 8:
            raise ValueError("policy challenge needs at least eight scenarios")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("policy challenge alpha must lie in (0, 1)")
        if not 0.0 < self.reference_quality < 1.0:
            raise ValueError("reference_quality must lie in (0, 1)")
        if self.call_timeout_s <= 0.0:
            raise ValueError("call timeout must be positive")

    def spec_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "policy-evaluation-task.v1",
            "policy_path": self.policy_path,
            "factory_name": self.factory_name,
            "scenarios": self.scenarios,
            "alpha": self.alpha,
            "reference_quality": self.reference_quality,
            "call_timeout_s": self.call_timeout_s,
            "protocol": POLICY_CHALLENGE_PROTOCOL,
        }

    @property
    def spec_sha256(self) -> str:
        import json

        payload = json.dumps(
            self.spec_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def challenge_sha256(self) -> str:
        """Challenge selection identity, intentionally independent of anchors."""
        import json

        payload = json.dumps(
            {
                "protocol": POLICY_CHALLENGE_PROTOCOL,
                "policy_path": self.policy_path,
                "factory_name": self.factory_name,
                "scenarios": self.scenarios,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def evaluation_plan(self) -> EvaluationPlan:
        return EvaluationPlan(
            task_spec_sha256=self.spec_sha256,
            security_tier="sealed_challenge",
            evidence={
                "type": POLICY_CHALLENGE_PROTOCOL,
                "scenarios": self.scenarios,
                "alpha": self.alpha,
                "controls": "author-registered-trusted",
            },
            metric_ids=("paired.normalized_return.v1",),
        )

    def grade(
        self,
        *,
        workspace: Path,
        rollout: PolicyRollout,
        controls: Mapping[str, ControlRollout],
    ) -> dict[str, Any]:
        if not controls:
            raise RuntimeError("policy challenge requires at least one trusted control")
        artifact = workspace / self.policy_path
        try:
            committed_digest = workspace_artifact_digest(workspace)
        except (OSError, ValueError) as exc:
            raise AgentFault(f"could not commit submitted policy: {exc}") from exc
        context = EvaluationContext.create_from_artifact_digest(
            task_digest=self.challenge_sha256,
            candidate_digest=committed_digest,
        )
        seeds = [
            context.seed(f"scenario:{index}") % (2**31 - 1)
            for index in range(self.scenarios)
        ]
        policy = load_submitted_policy(
            artifact,
            factory_name=self.factory_name,
            timeout_s=self.call_timeout_s,
        )
        candidate_scores: list[float] = []
        try:
            for seed in seeds:
                try:
                    score = _bounded_quality(
                        rollout(policy, seed),
                        source="submitted policy rollout",
                    )
                except PolicyWorkerError as exc:
                    raise AgentFault(f"submitted policy failed: {exc}") from exc
                candidate_scores.append(score)
        finally:
            policy.close()

        control_scores = {
            name: [
                _bounded_quality(control(seed), source=f"trusted control {name!r}")
                for seed in seeds
            ]
            for name, control in controls.items()
        }
        decisions: dict[str, TargetDecision] = {}
        trace: dict[str, dict[str, Any]] = {}
        accepted = True
        for name, values in sorted(control_scores.items()):
            differences = [
                candidate - baseline
                for candidate, baseline in zip(candidate_scores, values)
            ]
            wins = sum(value > 0.0 for value in differences)
            losses = sum(value < 0.0 for value in differences)
            informative_trials = wins + losses
            p_value = (
                _binomial_upper_tail(wins=wins, trials=informative_trials)
                if informative_trials
                else 1.0
            )
            mean_difference = sum(differences) / len(differences)
            passed = p_value <= self.alpha and mean_difference > 0.0
            decisions[name] = TargetDecision(
                passed,
                (
                    "paired_return_certificate_passed"
                    if passed
                    else "insufficient_paired_return"
                ),
            )
            accepted = accepted and passed
            trace[name] = {
                "p_value": p_value,
                "mean_difference": mean_difference,
                "wins": wins,
                "losses": losses,
                "ties": len(differences) - informative_trials,
                "candidate_scores": candidate_scores,
                "control_scores": values,
            }

        quality = sum(candidate_scores) / len(candidate_scores)
        score = (
            PiecewiseLinearCurve.from_reference(self.reference_quality).score(quality)
            if accepted
            else 0.0
        )
        receipt = PublicEvaluationReceipt(
            protocol=POLICY_CHALLENGE_PROTOCOL,
            plan_sha256=self.evaluation_plan.sha256,
            seed_commitment=context.commitment,
            attested=context.attested,
            challenge_count=self.scenarios,
            family_alpha=self.alpha,
            decisions=decisions,
        )
        write_private_trace(
            protocol=receipt.protocol,
            plan_sha256=receipt.plan_sha256,
            seed_commitment=receipt.seed_commitment,
            targets=trace,
            replay={
                "nonce": context.nonce,
                "artifact_digest": context.artifact_digest,
                "scenario_seeds": seeds,
            },
        )
        return {
            "score": score,
            "subscores": {"policy_quality": quality if accepted else 0.0},
            "weights": {"policy_quality": 1.0},
            "metadata": {
                "return_shape": "calibrated_continuous",
                "security_tier": "sealed_challenge",
                "task_spec_sha256": self.spec_sha256,
                "evaluation": receipt.to_dict(),
            },
        }


__all__ = [
    "POLICY_CHALLENGE_PROTOCOL",
    "ControlRollout",
    "PolicyEvaluationTask",
    "PolicyRollout",
]
