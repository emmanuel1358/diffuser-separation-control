"""Versioned metric definitions and reviewable anchor semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

RATIONALE_KINDS = frozenset(
    {
        "theoretical",
        "metric_bound",
        "domain_review",
        "reviewed_exception",
    }
)


def _finite(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metric {name!r} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"metric {name!r} is non-finite: {number!r}")
    return number


@dataclass(frozen=True)
class AnchorRationale:
    """Human-reviewable justification for a semantic calibration anchor."""

    kind: str
    summary: str
    source: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in RATIONALE_KINDS:
            raise ValueError(
                f"anchor rationale kind must be one of {sorted(RATIONALE_KINDS)}"
            )
        if len(self.summary.strip()) < 20:
            raise ValueError("anchor rationale summary must be at least 20 characters")
        if self.source is not None and not self.source.strip():
            raise ValueError("anchor rationale source must be non-empty when provided")

    def spec_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "summary": self.summary.strip(),
        }
        if self.source is not None:
            payload["source"] = self.source.strip()
        return payload


@dataclass(frozen=True)
class FloorAnchor:
    """A floor chosen from metric semantics, never fitted to one baseline."""

    value: float
    rationale: AnchorRationale

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("floor anchor value must be finite")

    def spec_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "rationale": self.rationale.spec_dict(),
        }


@dataclass(frozen=True)
class RegisteredMetric:
    """Exact versioned formula and input contract for a platform metric."""

    id: str
    formula: str
    input_contract: str

    def __post_init__(self) -> None:
        if ".v" not in self.id:
            raise ValueError("registered metric id must include an explicit version")
        if not self.formula.strip() or not self.input_contract.strip():
            raise ValueError(
                "registered metric formula and input contract are required"
            )

    def spec_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "formula": self.formula.strip(),
            "input_contract": self.input_contract.strip(),
        }


POPULATION_SRE = RegisteredMetric(
    id="sre.rmse_over_population_std.v1",
    formula="sqrt(mean((prediction - truth)^2)) / std(truth, ddof=0)",
    input_contract=(
        "Finite numeric prediction and truth arrays with identical shape; "
        "when truth std is zero, use unstandardized RMSE."
    ),
)

BINARY_F1_THRESHOLD_0_5 = RegisteredMetric(
    id="f1.binary_threshold_0_5.v1",
    formula="2*TP / (2*TP + FP + FN), prediction>=0.5 and truth>=0.5",
    input_contract=(
        "Finite numeric prediction and truth arrays with identical shape; "
        "binary positive class only, no macro/micro averaging."
    ),
)


@dataclass(frozen=True)
class MetricTarget:
    """One registered raw metric and its stable progress semantics."""

    name: str
    metric: RegisteredMetric
    direction: str
    weight: float
    floor: FloorAnchor
    perfect: float
    prediction_column: str
    truth_column: str
    kernel: Callable[[Any, Any], float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric target name must be non-empty")
        if self.direction not in {"lower", "higher"}:
            raise ValueError(f"invalid metric direction {self.direction!r}")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError(f"metric {self.name!r} weight must be positive and finite")
        if not math.isfinite(self.perfect):
            raise ValueError(f"metric {self.name!r} perfect anchor must be finite")
        if self.direction == "lower" and self.floor.value <= self.perfect:
            raise ValueError(
                f"lower-is-better metric {self.name!r} requires floor > perfect"
            )
        if self.direction == "higher" and self.floor.value >= self.perfect:
            raise ValueError(
                f"higher-is-better metric {self.name!r} requires floor < perfect"
            )

    @property
    def metric_id(self) -> str:
        return self.metric.id

    def measure(self, prediction: Any, truth: Any) -> float:
        return _finite(self.kernel(prediction, truth), name=self.name)

    def progress(self, value: float) -> float:
        value = _finite(value, name=self.name)
        if self.direction == "lower":
            raw = (self.floor.value - value) / (self.floor.value - self.perfect)
        else:
            raw = (value - self.floor.value) / (self.perfect - self.floor.value)
        return max(0.0, min(1.0, raw))

    def spec_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric.spec_dict(),
            "kernel": registered_kernel_identity(self),
            "direction": self.direction,
            "weight": self.weight,
            "floor": self.floor.spec_dict(),
            "perfect": self.perfect,
            "prediction_column": self.prediction_column,
            "truth_column": self.truth_column,
        }


def is_classification_target(target: MetricTarget) -> bool:
    """Classification vs regression, derived from the registered metric kernel."""
    return target.metric_id == "f1.binary_threshold_0_5.v1"


def no_info_ceiling(target: MetricTarget, degenerate_raw: Iterable[float]) -> float:
    """Best raw metric any feature-blind (no-information) strategy reaches.

    "Best" is direction-aware: the minimum raw value for a lower-is-better
    metric, the maximum for a higher-is-better one.
    """
    values = [_finite(value, name=target.name) for value in degenerate_raw]
    if not values:
        raise ValueError(
            f"no degenerate strategy metrics provided for target {target.name!r}"
        )
    return min(values) if target.direction == "lower" else max(values)


def effective_floor(target: MetricTarget, ceiling: float) -> float:
    """Author floor bounded by the no-information ceiling, direction-aware."""
    if target.direction == "lower":
        return min(target.floor.value, ceiling)
    return max(target.floor.value, ceiling)


def _population_sre(prediction: Any, truth: Any) -> float:
    import numpy as np

    pred = np.asarray(prediction, dtype=float)
    actual = np.asarray(truth, dtype=float)
    if pred.shape != actual.shape:
        raise ValueError(
            f"SRE shape mismatch: prediction {pred.shape}, truth {actual.shape}"
        )
    magnitude = max(
        1.0,
        float(np.max(np.abs(pred), initial=0.0)),
        float(np.max(np.abs(actual), initial=0.0)),
    )
    scaled_error = pred / magnitude - actual / magnitude
    rmse = magnitude * float(np.sqrt(np.mean(scaled_error**2)))
    truth_magnitude = max(1.0, float(np.max(np.abs(actual), initial=0.0)))
    denominator = truth_magnitude * float(np.std(actual / truth_magnitude, ddof=0))
    return rmse / denominator if denominator > 0.0 else rmse


def _binary_f1_threshold_0_5(prediction: Any, truth: Any) -> float:
    import numpy as np

    pred = np.asarray(prediction, dtype=float)
    actual = np.asarray(truth, dtype=float)
    if pred.shape != actual.shape:
        raise ValueError(
            f"binary F1 shape mismatch: prediction {pred.shape}, truth {actual.shape}"
        )
    pred_label = pred >= 0.5
    true_label = actual >= 0.5
    true_positive = int(np.sum(pred_label & true_label))
    false_positive = int(np.sum(pred_label & ~true_label))
    false_negative = int(np.sum(~pred_label & true_label))
    denominator = 2 * true_positive + false_positive + false_negative
    return (2.0 * true_positive / denominator) if denominator else 0.0


_REGISTERED_KERNELS: dict[str, tuple[RegisteredMetric, Callable[[Any, Any], float]]] = {
    POPULATION_SRE.id: (POPULATION_SRE, _population_sre),
    BINARY_F1_THRESHOLD_0_5.id: (
        BINARY_F1_THRESHOLD_0_5,
        _binary_f1_threshold_0_5,
    ),
}


def resolve_registered_kernel(
    metric_id: str,
) -> tuple[RegisteredMetric, Callable[[Any, Any], float]]:
    """Resolve a platform-owned metric implementation by versioned id."""
    try:
        return _REGISTERED_KERNELS[metric_id]
    except KeyError as exc:
        raise ValueError(f"unknown platform metric id {metric_id!r}") from exc


def registered_kernel_identity(target: MetricTarget) -> str:
    """Stable identity, rejecting descriptive-id/callable mismatches."""
    registered = _REGISTERED_KERNELS.get(target.metric_id)
    if registered is None:
        return (
            f"custom:{getattr(target.kernel, '__module__', '<unknown>')}."
            f"{getattr(target.kernel, '__qualname__', '<callable>')}"
        )
    metric, kernel = registered
    if target.metric != metric or target.kernel is not kernel:
        raise ValueError(
            f"metric target {target.name!r} claims registered id "
            f"{target.metric_id!r} but does not use its platform kernel"
        )
    return f"platform:{target.metric_id}"


def is_platform_registered_target(target: MetricTarget) -> bool:
    try:
        return registered_kernel_identity(target).startswith("platform:")
    except ValueError:
        return False


class PopulationSRETarget:
    """Factory for the explicit population-standardized RMSE convention."""

    @staticmethod
    def lower(
        name: str,
        *,
        weight: float,
        floor: FloorAnchor,
        perfect: float = 0.0,
        prediction_column: str | None = None,
        truth_column: str | None = None,
    ) -> MetricTarget:
        return MetricTarget(
            name=name,
            metric=POPULATION_SRE,
            direction="lower",
            weight=float(weight),
            floor=floor,
            perfect=float(perfect),
            prediction_column=prediction_column or name,
            truth_column=truth_column or name,
            kernel=_population_sre,
        )


class BinaryF1Target:
    """Factory for binary F1 with an explicit 0.5 positive threshold."""

    @staticmethod
    def higher(
        name: str,
        *,
        weight: float,
        floor: FloorAnchor,
        perfect: float = 1.0,
        prediction_column: str | None = None,
        truth_column: str | None = None,
    ) -> MetricTarget:
        return MetricTarget(
            name=name,
            metric=BINARY_F1_THRESHOLD_0_5,
            direction="higher",
            weight=float(weight),
            floor=floor,
            perfect=float(perfect),
            prediction_column=prediction_column or name,
            truth_column=truth_column or name,
            kernel=_binary_f1_threshold_0_5,
        )


# Compatibility alias for the pre-merge v1 API. New code should use the exact
# convention name so an author cannot mistake this for another SRE definition.
SRETarget = PopulationSRETarget


def measure_registered_targets(
    targets: tuple[MetricTarget, ...],
    *,
    submission: Any,
    truth: Any,
) -> dict[str, float]:
    """Measure every target from dataframe-like submission and truth objects."""

    measured: dict[str, float] = {}
    for target in targets:
        try:
            prediction = submission[target.prediction_column]
        except Exception as exc:
            raise ValueError(
                f"submission is missing target column {target.prediction_column!r}"
            ) from exc
        try:
            actual = truth[target.truth_column]
        except Exception as exc:
            raise RuntimeError(
                f"private truth is missing target column {target.truth_column!r}"
            ) from exc
        measured[target.name] = target.measure(prediction, actual)
    return measured


def normalize_weights(targets: tuple[MetricTarget, ...]) -> dict[str, float]:
    total = sum(target.weight for target in targets)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("metric target weights must have a positive finite sum")
    return {target.name: target.weight / total for target in targets}


def validate_metric_vector(
    targets: tuple[MetricTarget, ...], metrics: Mapping[str, Any]
) -> dict[str, float]:
    expected = {target.name for target in targets}
    actual = set(metrics)
    if actual != expected:
        raise ValueError(
            f"metric vector keys must be exactly {sorted(expected)}, got {sorted(actual)}"
        )
    return {
        target.name: _finite(metrics[target.name], name=target.name)
        for target in targets
    }


__all__ = [
    "AnchorRationale",
    "BINARY_F1_THRESHOLD_0_5",
    "BinaryF1Target",
    "FloorAnchor",
    "MetricTarget",
    "POPULATION_SRE",
    "PopulationSRETarget",
    "RATIONALE_KINDS",
    "RegisteredMetric",
    "SRETarget",
    "effective_floor",
    "is_classification_target",
    "is_platform_registered_target",
    "measure_registered_targets",
    "no_info_ceiling",
    "normalize_weights",
    "registered_kernel_identity",
    "resolve_registered_kernel",
    "validate_metric_vector",
]
