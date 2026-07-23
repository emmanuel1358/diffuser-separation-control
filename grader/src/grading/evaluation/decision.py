"""Information-evidence decisions kept separate from quality calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from grading.evaluation.context import EvaluationContext
from grading.evaluation.metrics import MetricTarget
from grading.evaluation.permutation import K_PERMUTATIONS, permutation_null
from grading.evaluation.result import TargetDecision

IID_EVIDENCE_PROTOCOL = "iid-permutation-holm.v1"


@dataclass(frozen=True)
class IIDPermutationEvidence:
    """Tier-B IID evidence with task-level family-wise error control."""

    family_alpha: float = 0.01
    permutations: int = K_PERMUTATIONS
    min_units: int = 32

    def __post_init__(self) -> None:
        if not 0.0 < self.family_alpha < 1.0:
            raise ValueError("family_alpha must lie strictly between 0 and 1")
        if self.permutations < 1:
            raise ValueError("permutations must be positive")
        if self.min_units < 2:
            raise ValueError("min_units must be at least 2")

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": IID_EVIDENCE_PROTOCOL,
            "family_alpha": self.family_alpha,
            "permutations": self.permutations,
            "min_units": self.min_units,
            "multiple_testing": "holm-bonferroni",
            "ties": "against-candidate",
        }


@dataclass(frozen=True)
class EvidenceBundle:
    protocol: IIDPermutationEvidence
    decisions: Mapping[str, TargetDecision]
    private_trace: Mapping[str, Mapping[str, Any]]
    challenge_count: int


def _is_exact_constant(value: Any) -> bool:
    import numpy as np

    array = np.asarray(value)
    if array.ndim < 1 or array.shape[0] == 0:
        return True
    return bool(np.array_equal(array, np.repeat(array[:1], array.shape[0], axis=0)))


def evaluate_iid_evidence(
    *,
    targets: tuple[MetricTarget, ...],
    raw_arrays: Mapping[str, tuple[Any, Any]],
    context: EvaluationContext,
    protocol: IIDPermutationEvidence,
) -> EvidenceBundle:
    """Evaluate every target, then apply Holm-Bonferroni across the family."""
    import numpy as np

    p_values: dict[str, float] = {}
    traces: dict[str, dict[str, Any]] = {}
    exact_constants: set[str] = set()
    challenge_count = 0

    for target in targets:
        raw = raw_arrays.get(target.name)
        if raw is None:
            raise RuntimeError(
                f"protected target {target.name!r} has no raw evidence arrays"
            )
        prediction, truth = raw
        pred = np.asarray(prediction)
        actual = np.asarray(truth)
        if pred.shape != actual.shape:
            raise RuntimeError(
                f"target {target.name!r} evidence shape mismatch: "
                f"prediction {pred.shape}, truth {actual.shape}"
            )
        if pred.ndim < 1 or pred.shape[0] < protocol.min_units:
            raise RuntimeError(
                f"target {target.name!r} has {pred.shape[0] if pred.ndim else 0} "
                f"independent units; protocol requires at least {protocol.min_units}"
            )
        if not (
            np.isfinite(np.asarray(pred, dtype=float)).all()
            and np.isfinite(np.asarray(actual, dtype=float)).all()
        ):
            raise RuntimeError(f"target {target.name!r} evidence arrays are non-finite")

        if _is_exact_constant(pred):
            exact_constants.add(target.name)
            p_values[target.name] = 1.0
            traces[target.name] = {
                "exact_constant": True,
                "p_value": 1.0,
                "n_units": int(pred.shape[0]),
            }
            continue

        result = permutation_null(
            target,
            pred,
            actual,
            k=protocol.permutations,
            alpha=protocol.family_alpha,
            seed=context.seed(f"target:{target.name}"),
        )
        if not math.isfinite(result.p_value):
            raise RuntimeError(
                f"target {target.name!r} produced non-finite evidence p-value"
            )
        challenge_count += result.k
        p_values[target.name] = result.p_value
        traces[target.name] = {
            **result.to_metadata(),
            "exact_constant": False,
            "n_units": int(pred.shape[0]),
        }

    # Holm step-down rejection of the no-information null. Once a sorted
    # p-value misses its threshold, it and all larger p-values fail evidence.
    accepted: set[str] = set()
    holm_thresholds: dict[str, float] = {}
    still_rejecting = True
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        threshold = protocol.family_alpha / (family_size - index)
        holm_thresholds[name] = threshold
        if still_rejecting and p_value <= threshold:
            accepted.add(name)
        else:
            still_rejecting = False

    decisions: dict[str, TargetDecision] = {}
    for target in targets:
        name = target.name
        traces[name]["holm_threshold"] = holm_thresholds[name]
        traces[name]["accepted"] = name in accepted
        if name in exact_constants:
            decisions[name] = TargetDecision(False, "exact_constant")
        elif name in accepted:
            decisions[name] = TargetDecision(True, "information_certificate_passed")
        else:
            decisions[name] = TargetDecision(False, "insufficient_information")

    return EvidenceBundle(
        protocol=protocol,
        decisions=decisions,
        private_trace=traces,
        challenge_count=challenge_count,
    )


__all__ = [
    "IID_EVIDENCE_PROTOCOL",
    "EvidenceBundle",
    "IIDPermutationEvidence",
    "evaluate_iid_evidence",
]
