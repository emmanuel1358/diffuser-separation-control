"""Declarative author surface for calibrated continuous tasks."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from grading.calibration import PiecewiseLinearCurve
from grading.evaluation.context import EvaluationContext, workspace_artifact_digest
from grading.evaluation.decision import (
    IIDPermutationEvidence,
    evaluate_iid_evidence,
)
from grading.evaluation.lock import (
    DEFAULT_LOCK_FILENAME,
    CalibrationLock,
    build_calibration_lock,
    canonical_json_bytes,
    load_calibration_lock,
)
from grading.evaluation.metrics import (
    AnchorRationale,
    MetricTarget,
    measure_registered_targets,
    normalize_weights,
    validate_metric_vector,
)
from grading.evaluation.plan import EvaluationPlan
from grading.evaluation.result import (
    PublicEvaluationReceipt,
    write_private_trace,
)
from grading.faults import AgentFault
from grading.helpers import load_submission_or_fault
from grading.policy_runner import load_submitted_policy

# Thread-parallel inference (the common shape for tree ensembles) sums float
# contributions in whatever order the threads finish, so a correct predictor can
# disagree with itself in the last bits. Compare floats within this relative
# tolerance and everything else exactly, so real nondeterminism still fails.
REPEAT_CALL_RTOL = 1e-9
DEFAULT_PREDICT_TIMEOUT_S = 60.0
DEFAULT_FIRST_CALL_TIMEOUT_S = 120.0
DEFAULT_MAX_PREDICT_ROWS = 100_000
DEFAULT_MAX_PREDICT_REPLY_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_UNACKNOWLEDGED_NAIVE_SCORE_GAP = 0.05


def _repeats_within_tolerance(first: Any, second: Any) -> bool:
    import numpy as np
    from pandas.api.types import is_float_dtype

    if list(first.columns) != list(second.columns) or len(first) != len(second):
        return False
    for column in first.columns:
        left = first[column]
        right = second[column]
        if is_float_dtype(left.dtype) and is_float_dtype(right.dtype):
            if not np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=REPEAT_CALL_RTOL,
                atol=0.0,
                equal_nan=True,
            ):
                return False
        elif not left.equals(right):
            return False
    return True


def _normalize_value_domains(
    value_domains: Mapping[str, list[Any] | tuple[Any, ...]] | None,
) -> dict[str, tuple[Any, ...]]:
    import numpy as np

    normalized: dict[str, tuple[Any, ...]] = {}
    for raw_column, raw_values in (value_domains or {}).items():
        column = str(raw_column)
        values = tuple(raw_values)
        if not column or not values:
            raise ValueError("value domains require non-empty columns and values")
        for value in values:
            if not isinstance(value, (str, bool, int, float)) or (
                isinstance(value, (float, np.floating))
                and not math.isfinite(float(value))
            ):
                raise ValueError(
                    "value domain entries must be finite JSON scalar values"
                )
        normalized[column] = values
    return dict(sorted(normalized.items()))


@dataclass(frozen=True)
class OneHot:
    """Exactly one binary output is active in every prediction row."""

    columns: tuple[str, ...]
    tolerance: float = 1e-8

    def __init__(
        self,
        columns: list[str] | tuple[str, ...],
        *,
        tolerance: float = 1e-8,
    ) -> None:
        normalized = tuple(str(column) for column in columns)
        if len(normalized) < 2 or len(set(normalized)) != len(normalized):
            raise ValueError("OneHot requires at least two unique columns")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("OneHot tolerance must be positive and finite")
        object.__setattr__(self, "columns", normalized)
        object.__setattr__(self, "tolerance", float(tolerance))

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "one_hot.v1",
            "columns": list(self.columns),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class Simplex:
    """Non-negative outputs whose values sum to one in every row."""

    columns: tuple[str, ...]
    tolerance: float = 1e-6

    def __init__(
        self,
        columns: list[str] | tuple[str, ...],
        *,
        tolerance: float = 1e-6,
    ) -> None:
        normalized = tuple(str(column) for column in columns)
        if len(normalized) < 2 or len(set(normalized)) != len(normalized):
            raise ValueError("Simplex requires at least two unique columns")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Simplex tolerance must be positive and finite")
        object.__setattr__(self, "columns", normalized)
        object.__setattr__(self, "tolerance", float(tolerance))

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "simplex.v1",
            "columns": list(self.columns),
            "tolerance": self.tolerance,
        }


PredictionConstraint = OneHot | Simplex


def _validate_prediction_contract(
    frame: Any,
    *,
    value_domains: Mapping[str, tuple[Any, ...]],
    constraints: tuple[PredictionConstraint, ...],
) -> None:
    import numpy as np

    for column, allowed in value_domains.items():
        if column not in frame.columns:
            raise AgentFault(
                f"prediction result is missing constrained column {column!r}"
            )
        invalid = ~frame[column].isin(allowed)
        if bool(invalid.any()):
            examples = frame.loc[invalid, column].head(3).tolist()
            raise AgentFault(
                f"prediction column {column!r} contains values outside its "
                f"declared domain {list(allowed)!r}: {examples!r}"
            )

    for constraint in constraints:
        missing = [column for column in constraint.columns if column not in frame]
        if missing:
            raise AgentFault(
                f"prediction result is missing constrained columns: {missing}"
            )
        try:
            values = frame[list(constraint.columns)].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise AgentFault(
                f"prediction constraint columns must be numeric: {exc}"
            ) from exc
        if not np.isfinite(values).all():
            raise AgentFault("prediction constraint columns contain NaN or infinity")
        if isinstance(constraint, OneHot):
            binary = np.logical_or(
                np.isclose(values, 0.0, atol=constraint.tolerance, rtol=0.0),
                np.isclose(values, 1.0, atol=constraint.tolerance, rtol=0.0),
            )
            valid = binary.all(axis=1) & np.isclose(
                values.sum(axis=1),
                1.0,
                atol=constraint.tolerance,
                rtol=0.0,
            )
            kind = "one-hot"
        else:
            valid = (
                (values >= -constraint.tolerance).all(axis=1)
                & (values <= 1.0 + constraint.tolerance).all(axis=1)
                & np.isclose(
                    values.sum(axis=1),
                    1.0,
                    atol=constraint.tolerance,
                    rtol=0.0,
                )
            )
            kind = "simplex"
        if not bool(valid.all()):
            raise AgentFault(
                f"prediction violates {kind} constraint over "
                f"{list(constraint.columns)!r} in {int((~valid).sum())} row(s)"
            )


@dataclass(frozen=True)
class CsvRows:
    path: str
    columns: tuple[str, ...]
    extra_columns: Literal["reject", "drop", "preserve"] = "reject"
    value_domains: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    constraints: tuple[PredictionConstraint, ...] = ()

    def __init__(
        self,
        path: str,
        *,
        columns: list[str] | tuple[str, ...],
        extra_columns: Literal["reject", "drop", "preserve"] | None = None,
        allow_extra_columns: bool | None = None,
        value_domains: Mapping[str, list[Any] | tuple[Any, ...]] | None = None,
        constraints: list[PredictionConstraint] | tuple[PredictionConstraint, ...] = (),
    ) -> None:
        normalized = tuple(str(column) for column in columns)
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(
                "CSV artifact path must be non-empty and workspace-relative"
            )
        if not normalized or any(not column for column in normalized):
            raise ValueError("CSV artifact columns must be non-empty")
        if allow_extra_columns is not None:
            if extra_columns is not None:
                raise ValueError(
                    "pass either extra_columns or allow_extra_columns, not both"
                )
            extra_policy = "preserve" if allow_extra_columns else "reject"
        else:
            extra_policy = extra_columns or "reject"
        if extra_policy not in {"reject", "drop", "preserve"}:
            raise ValueError("extra_columns must be 'reject', 'drop', or 'preserve'")
        domains = _normalize_value_domains(value_domains)
        normalized_constraints = tuple(constraints)
        constrained_columns = set(domains)
        for constraint in normalized_constraints:
            if not isinstance(constraint, (OneHot, Simplex)):
                raise TypeError("unsupported prediction constraint")
            constrained_columns.update(constraint.columns)
        undeclared = sorted(constrained_columns - set(normalized))
        if undeclared:
            raise ValueError(
                f"prediction constraints reference undeclared CSV columns: {undeclared}"
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "columns", normalized)
        object.__setattr__(self, "extra_columns", extra_policy)
        object.__setattr__(self, "value_domains", domains)
        object.__setattr__(self, "constraints", normalized_constraints)

    @property
    def allow_extra_columns(self) -> bool:
        """Compatibility view for legacy callers."""
        return self.extra_columns == "preserve"

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "csv_rows.v2",
            "path": self.path,
            "columns": list(self.columns),
            "extra_columns": self.extra_columns,
            "value_domains": {
                column: list(values)
                for column, values in sorted(self.value_domains.items())
            },
            "constraints": [constraint.spec_dict() for constraint in self.constraints],
        }

    def legacy_spec_dict(self) -> dict[str, Any] | None:
        """V1 identity for default-only descriptors during lock migration."""
        if self.extra_columns not in {"reject", "preserve"}:
            return None
        if self.value_domains or self.constraints:
            return None
        return {
            "type": "csv_rows.v1",
            "path": self.path,
            "columns": list(self.columns),
            "allow_extra_columns": self.extra_columns == "preserve",
        }


@dataclass(frozen=True)
class PythonPredictor:
    """Queryable Python artifact evaluated only on post-commit private rows."""

    path: str = "predictor.py"
    factory_name: str = "load_predictor"
    method: str = "predict"
    predict_timeout_s: float = DEFAULT_PREDICT_TIMEOUT_S
    first_call_timeout_s: float = DEFAULT_FIRST_CALL_TIMEOUT_S
    max_rows: int = DEFAULT_MAX_PREDICT_ROWS
    max_reply_bytes: int = DEFAULT_MAX_PREDICT_REPLY_BYTES
    prediction_scope: Literal["batch_allowed", "row_independent"] = "batch_allowed"
    row_independence_partitions: int = 3
    value_domains: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    constraints: tuple[PredictionConstraint, ...] = ()

    def __post_init__(self) -> None:
        relative = Path(self.path)
        if (
            not self.path
            or relative.is_absolute()
            or ".." in relative.parts
            or not self.factory_name
            or not self.method
        ):
            raise ValueError("invalid Python predictor artifact descriptor")
        if (
            not math.isfinite(self.predict_timeout_s)
            or self.predict_timeout_s <= 0
            or not math.isfinite(self.first_call_timeout_s)
            or self.first_call_timeout_s <= 0
        ):
            raise ValueError("predictor timeouts must be positive and finite")
        if self.max_rows <= 0 or self.max_reply_bytes <= 0:
            raise ValueError("predictor row and reply budgets must be positive")
        if self.prediction_scope not in {"batch_allowed", "row_independent"}:
            raise ValueError(
                "prediction_scope must be 'batch_allowed' or 'row_independent'"
            )
        if self.row_independence_partitions < 2:
            raise ValueError("row_independence_partitions must be at least two")
        domains = _normalize_value_domains(self.value_domains)
        constraints = tuple(self.constraints)
        if any(not isinstance(item, (OneHot, Simplex)) for item in constraints):
            raise TypeError("unsupported prediction constraint")
        object.__setattr__(self, "predict_timeout_s", float(self.predict_timeout_s))
        object.__setattr__(
            self, "first_call_timeout_s", float(self.first_call_timeout_s)
        )
        object.__setattr__(self, "max_rows", int(self.max_rows))
        object.__setattr__(self, "max_reply_bytes", int(self.max_reply_bytes))
        object.__setattr__(self, "value_domains", domains)
        object.__setattr__(self, "constraints", constraints)

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "python_predictor.v2",
            "path": self.path,
            "factory_name": self.factory_name,
            "method": self.method,
            "predict_timeout_s": self.predict_timeout_s,
            "first_call_timeout_s": self.first_call_timeout_s,
            "max_rows": self.max_rows,
            "max_reply_bytes": self.max_reply_bytes,
            "prediction_scope": self.prediction_scope,
            "row_independence_partitions": self.row_independence_partitions,
            "value_domains": {
                column: list(values)
                for column, values in sorted(self.value_domains.items())
            },
            "constraints": [constraint.spec_dict() for constraint in self.constraints],
        }

    def legacy_spec_dict(self) -> dict[str, Any] | None:
        """V1 identity when every newly declared contract keeps its default."""
        if (
            self.predict_timeout_s != DEFAULT_PREDICT_TIMEOUT_S
            or self.first_call_timeout_s != DEFAULT_FIRST_CALL_TIMEOUT_S
            or self.max_rows != DEFAULT_MAX_PREDICT_ROWS
            or self.max_reply_bytes != DEFAULT_MAX_PREDICT_REPLY_BYTES
            or self.prediction_scope != "batch_allowed"
            or self.row_independence_partitions != 3
            or self.value_domains
            or self.constraints
        ):
            return None
        return {
            "type": "python_predictor.v1",
            "path": self.path,
            "factory_name": self.factory_name,
            "method": self.method,
        }


@dataclass(frozen=True)
class PrivateTableChallenge:
    """Root-only challenge bank sampled after artifact commitment."""

    filename: str
    feature_columns: tuple[str, ...]
    sample_size: int | None
    selection_policy: Literal["artifact_digest", "full_bank", "stable_subset"]

    def __init__(
        self,
        filename: str,
        *,
        feature_columns: list[str] | tuple[str, ...],
        sample_size: int | None = None,
        selection_policy: (
            Literal["artifact_digest", "full_bank", "stable_subset"] | None
        ) = None,
    ) -> None:
        relative = Path(filename)
        columns = tuple(str(column) for column in feature_columns)
        if (
            not filename
            or relative.is_absolute()
            or ".." in relative.parts
            or not columns
            or any(not column for column in columns)
        ):
            raise ValueError("invalid private table challenge descriptor")
        policy = selection_policy or (
            "full_bank" if sample_size is None else "artifact_digest"
        )
        if policy not in {"artifact_digest", "full_bank", "stable_subset"}:
            raise ValueError(
                "selection_policy must be 'artifact_digest', 'full_bank', "
                "or 'stable_subset'"
            )
        if policy == "full_bank" and sample_size is not None:
            raise ValueError("full_bank selection must not declare sample_size")
        if policy in {"artifact_digest", "stable_subset"} and (
            sample_size is None or sample_size < 32
        ):
            raise ValueError("subset selection requires sample_size of at least 32")
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "feature_columns", columns)
        object.__setattr__(
            self, "sample_size", int(sample_size) if sample_size is not None else None
        )
        object.__setattr__(self, "selection_policy", policy)

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "private_table.v2",
            "filename": self.filename,
            "feature_columns": list(self.feature_columns),
            "sample_size": self.sample_size,
            "selection_policy": self.selection_policy,
        }

    def legacy_spec_dict(self) -> dict[str, Any] | None:
        """V1 identity for an explicitly sized challenge during migration."""
        if self.sample_size is None or self.selection_policy != "artifact_digest":
            return None
        return {
            "type": "private_table.v1",
            "filename": self.filename,
            "feature_columns": list(self.feature_columns),
            "sample_size": self.sample_size,
        }


_PROBE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class WorkspaceProbe:
    """One committed ready-to-measure no-information workspace."""

    name: str
    path: str
    rationale: str

    def __post_init__(self) -> None:
        relative = Path(self.path)
        if not _PROBE_NAME_RE.fullmatch(self.name):
            raise ValueError(
                "workspace probe name must use letters, numbers, dot, underscore, "
                "or hyphen"
            )
        if (
            not self.path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:2] != ("baselines", "degenerate")
            or len(relative.parts) < 3
        ):
            raise ValueError(
                "workspace probe path must be task-relative under "
                "baselines/degenerate/<probe>"
            )
        if len(self.rationale.strip()) < 20:
            raise ValueError("workspace probe rationale must be at least 20 characters")

    def spec_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": self.path,
            "rationale": self.rationale.strip(),
        }


@dataclass(frozen=True)
class WorkspaceDegenerateProbes:
    """Explicit no-information artifacts for opaque/non-tabular graders."""

    probes: tuple[WorkspaceProbe, ...]

    def __init__(
        self, probes: list[WorkspaceProbe] | tuple[WorkspaceProbe, ...]
    ) -> None:
        normalized = tuple(probes)
        if not normalized:
            raise ValueError("workspace degenerate probes must be non-empty")
        names = [probe.name for probe in normalized]
        paths = [probe.path for probe in normalized]
        if len(names) != len(set(names)):
            raise ValueError("workspace probe names must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("workspace probe paths must be unique")
        object.__setattr__(self, "probes", normalized)

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "workspace_degenerate_probes.v1",
            "probes": [
                probe.spec_dict() for probe in sorted(self.probes, key=lambda p: p.name)
            ],
        }


@dataclass(frozen=True)
class GeneratedCalibration:
    filename: str = DEFAULT_LOCK_FILENAME
    degenerate_probes: WorkspaceDegenerateProbes | None = None
    max_unacknowledged_naive_score_gap: float = (
        DEFAULT_MAX_UNACKNOWLEDGED_NAIVE_SCORE_GAP
    )
    naive_semantic_gap_acknowledgement: AnchorRationale | None = None

    def __post_init__(self) -> None:
        if self.filename != DEFAULT_LOCK_FILENAME:
            raise ValueError(
                f"generated calibration filename is fixed to {DEFAULT_LOCK_FILENAME!r}"
            )
        if (
            not math.isfinite(self.max_unacknowledged_naive_score_gap)
            or not 0 <= self.max_unacknowledged_naive_score_gap < 0.5
        ):
            raise ValueError(
                "max_unacknowledged_naive_score_gap must be finite in [0, 0.5)"
            )
        acknowledgement = self.naive_semantic_gap_acknowledgement
        if acknowledgement is not None and acknowledgement.kind != "reviewed_exception":
            raise ValueError(
                "naive semantic gap acknowledgement must use "
                "kind='reviewed_exception'"
            )

    def spec_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "generated_lock.v2",
            "filename": self.filename,
            "max_unacknowledged_naive_score_gap": (
                self.max_unacknowledged_naive_score_gap
            ),
        }
        if self.degenerate_probes is not None:
            payload["degenerate_probes"] = self.degenerate_probes.spec_dict()
        if self.naive_semantic_gap_acknowledgement is not None:
            payload["naive_semantic_gap_acknowledgement"] = (
                self.naive_semantic_gap_acknowledgement.spec_dict()
            )
        return payload

    def legacy_spec_dict(self) -> dict[str, Any] | None:
        """V1 identity when the new semantic-gap policy keeps its defaults."""
        if (
            self.max_unacknowledged_naive_score_gap
            != DEFAULT_MAX_UNACKNOWLEDGED_NAIVE_SCORE_GAP
            or self.naive_semantic_gap_acknowledgement is not None
        ):
            return None
        payload: dict[str, Any] = {
            "type": "generated_lock.v1",
            "filename": self.filename,
        }
        if self.degenerate_probes is not None:
            payload["degenerate_probes"] = self.degenerate_probes.spec_dict()
        return payload


RawEvaluator = Callable[[Any, Any], Mapping[str, Any]]
PRODUCTION_EVALUATION_ENV = "LBX_EVALUATION_PRODUCTION"
CALIBRATION_SEED_ENV = "LBX_CALIBRATION_SEED"


@dataclass(frozen=True)
class CalibrationMeasureContext:
    """Shared deterministic case-selection context for calibration strategies."""

    seed: int = 0

    def derive_seed(self, label: str) -> int:
        payload = f"continuous-calibration.v1\0{self.seed}\0{label}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class ContinuousTask:
    """One continuous task registration shared by calibration and grading."""

    artifact: CsvRows | PythonPredictor | None
    challenge: PrivateTableChallenge | None
    targets: tuple[MetricTarget, ...]
    calibration: GeneratedCalibration
    naive: str
    truth_filename: str | None
    security_tier: str
    raw_evaluator: RawEvaluator | None
    evidence: IIDPermutationEvidence | None
    naive_score_min: float
    naive_score_max: float
    naive_at_floor: AnchorRationale | None

    @classmethod
    def calibrated(
        cls,
        *,
        targets: list[MetricTarget] | tuple[MetricTarget, ...],
        calibration: GeneratedCalibration | None = None,
        naive: str = "baselines/naive",
        security_tier: str = "sealed_rescore",
        evidence: IIDPermutationEvidence | None = None,
        naive_score_min: float = 1e-6,
        naive_score_max: float = 0.10,
        naive_at_floor: AnchorRationale | None = None,
    ) -> "ContinuousTask":
        """Compose generated calibration with fully hand-authored evaluation."""
        return cls._create(
            artifact=None,
            challenge=None,
            targets=targets,
            calibration=calibration,
            naive=naive,
            truth_filename=None,
            security_tier=security_tier,
            raw_evaluator=None,
            evidence=evidence or IIDPermutationEvidence(),
            naive_score_min=naive_score_min,
            naive_score_max=naive_score_max,
            naive_at_floor=naive_at_floor,
        )

    @classmethod
    def static(
        cls,
        *,
        artifact: CsvRows | None,
        targets: list[MetricTarget] | tuple[MetricTarget, ...],
        calibration: GeneratedCalibration | None = None,
        naive: str = "baselines/naive",
        truth_filename: str = "test_target.parquet",
        security_tier: str = "sealed_rescore",
        evidence: IIDPermutationEvidence | None = None,
        naive_score_min: float = 1e-6,
        naive_score_max: float = 0.10,
        naive_at_floor: AnchorRationale | None = None,
    ) -> "ContinuousTask":
        return cls._create(
            artifact=artifact,
            challenge=None,
            targets=targets,
            calibration=calibration,
            naive=naive,
            truth_filename=truth_filename,
            security_tier=security_tier,
            raw_evaluator=None,
            evidence=evidence or IIDPermutationEvidence(),
            naive_score_min=naive_score_min,
            naive_score_max=naive_score_max,
            naive_at_floor=naive_at_floor,
        )

    @classmethod
    def custom_static(
        cls,
        *,
        artifact: CsvRows | None,
        targets: list[MetricTarget] | tuple[MetricTarget, ...],
        evaluate: RawEvaluator,
        calibration: GeneratedCalibration | None = None,
        naive: str = "baselines/naive",
        truth_filename: str = "test_target.parquet",
        security_tier: str = "custom_reviewed",
        naive_score_min: float = 1e-6,
        naive_score_max: float = 0.10,
        naive_at_floor: AnchorRationale | None = None,
    ) -> "ContinuousTask":
        if not callable(evaluate):
            raise TypeError("custom continuous evaluator must be callable")
        return cls._create(
            artifact=artifact,
            challenge=None,
            targets=targets,
            calibration=calibration,
            naive=naive,
            truth_filename=truth_filename,
            security_tier=security_tier,
            raw_evaluator=evaluate,
            evidence=None,
            naive_score_min=naive_score_min,
            naive_score_max=naive_score_max,
            naive_at_floor=naive_at_floor,
        )

    @classmethod
    def model(
        cls,
        *,
        artifact: PythonPredictor,
        challenge: PrivateTableChallenge,
        targets: list[MetricTarget] | tuple[MetricTarget, ...],
        calibration: GeneratedCalibration | None = None,
        naive: str = "baselines/naive",
        evidence: IIDPermutationEvidence | None = None,
        naive_score_min: float = 1e-6,
        naive_score_max: float = 0.10,
        naive_at_floor: AnchorRationale | None = None,
    ) -> "ContinuousTask":
        """Create a queryable model evaluated on a private post-commit bank."""
        return cls._create(
            artifact=artifact,
            challenge=challenge,
            targets=targets,
            calibration=calibration,
            naive=naive,
            truth_filename=challenge.filename,
            security_tier="sealed_challenge",
            raw_evaluator=None,
            evidence=evidence or IIDPermutationEvidence(),
            naive_score_min=naive_score_min,
            naive_score_max=naive_score_max,
            naive_at_floor=naive_at_floor,
        )

    @classmethod
    def _create(
        cls,
        *,
        artifact: CsvRows | PythonPredictor | None,
        challenge: PrivateTableChallenge | None,
        targets: list[MetricTarget] | tuple[MetricTarget, ...],
        calibration: GeneratedCalibration | None,
        naive: str,
        truth_filename: str | None,
        security_tier: str,
        raw_evaluator: RawEvaluator | None,
        evidence: IIDPermutationEvidence | None,
        naive_score_min: float,
        naive_score_max: float,
        naive_at_floor: AnchorRationale | None,
    ) -> "ContinuousTask":
        normalized_targets = tuple(targets)
        if not normalized_targets:
            raise ValueError("continuous TASK must declare at least one target")
        names = [target.name for target in normalized_targets]
        if len(names) != len(set(names)):
            raise ValueError(f"continuous TASK target names must be unique: {names}")
        if challenge is not None:
            leaked = sorted(
                set(challenge.feature_columns)
                & {target.truth_column for target in normalized_targets}
            )
            if leaked:
                raise ValueError(
                    "private challenge features must not include target truth "
                    f"columns: {leaked}"
                )
        if isinstance(artifact, PythonPredictor):
            prediction_columns = {
                target.prediction_column for target in normalized_targets
            }
            constrained_columns = set(artifact.value_domains)
            for constraint in artifact.constraints:
                constrained_columns.update(constraint.columns)
            undeclared = sorted(constrained_columns - prediction_columns)
            if undeclared:
                raise ValueError(
                    "predictor constraints reference columns that are not target "
                    f"predictions: {undeclared}"
                )
            if (
                challenge is not None
                and challenge.sample_size is not None
                and challenge.sample_size > artifact.max_rows
            ):
                raise ValueError(
                    "private challenge sample_size exceeds predictor max_rows"
                )
        normalize_weights(normalized_targets)
        if not naive or Path(naive).is_absolute() or ".." in Path(naive).parts:
            raise ValueError("naive strategy path must be task-relative")
        if truth_filename is not None and (
            not truth_filename
            or Path(truth_filename).is_absolute()
            or ".." in Path(truth_filename).parts
        ):
            raise ValueError("truth filename must be private-directory-relative")
        if security_tier not in {
            "sealed_rescore",
            "sealed_challenge",
            "custom_reviewed",
        }:
            raise ValueError(f"unsupported continuous security tier {security_tier!r}")
        if security_tier == "sealed_challenge" and not (
            isinstance(artifact, PythonPredictor)
            and isinstance(challenge, PrivateTableChallenge)
        ):
            raise ValueError(
                "sealed_challenge is derived from a compiled queryable adapter; "
                "authors cannot self-declare it"
            )
        if security_tier == "sealed_rescore" and evidence is None:
            raise ValueError("sealed_rescore tasks require an evidence protocol")
        if security_tier == "custom_reviewed" and evidence is not None:
            raise ValueError("custom_reviewed tasks cannot claim sealed evidence")
        if not 0.0 <= naive_score_min < naive_score_max < 0.5:
            raise ValueError(
                "naive qualification range must satisfy "
                "0 <= min < max < reference score 0.5"
            )
        if naive_at_floor is not None:
            if naive_at_floor.kind != "reviewed_exception":
                raise ValueError(
                    "naive_at_floor must use an AnchorRationale with "
                    "kind='reviewed_exception'"
                )
            if naive_score_min != 0.0:
                raise ValueError(
                    "naive_at_floor requires naive_score_min=0 so only the null "
                    "boundary becomes inclusive"
                )
        return cls(
            artifact=artifact,
            challenge=challenge,
            targets=normalized_targets,
            calibration=calibration or GeneratedCalibration(),
            naive=naive,
            truth_filename=truth_filename,
            security_tier=security_tier,
            raw_evaluator=raw_evaluator,
            evidence=evidence,
            naive_score_min=float(naive_score_min),
            naive_score_max=float(naive_score_max),
            naive_at_floor=naive_at_floor,
        )

    def spec_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "continuous-task.v3",
            "artifact": self.artifact.spec_dict() if self.artifact else None,
            "challenge": self.challenge.spec_dict() if self.challenge else None,
            "targets": [target.spec_dict() for target in self.targets],
            "calibration": self.calibration.spec_dict(),
            "naive": self.naive,
            "truth_filename": self.truth_filename,
            "security_tier": self.security_tier,
            "evidence": self.evidence.spec_dict() if self.evidence else None,
            "custom_evaluator": self.raw_evaluator is not None,
            "measurement_mode": (
                "standard_adapter" if self.artifact is not None else "module_callback"
            ),
            "naive_score_range": (
                {
                    "inclusive_min": self.naive_score_min,
                    "inclusive_max": self.naive_score_max,
                    "rationale": self.naive_at_floor.spec_dict(),
                }
                if self.naive_at_floor is not None
                else {
                    "exclusive_min": self.naive_score_min,
                    "inclusive_max": self.naive_score_max,
                }
            ),
        }

    @property
    def spec_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.spec_dict())).hexdigest()

    def legacy_spec_dict(self) -> dict[str, Any] | None:
        """Return the pre-v3 task identity when only default contracts changed."""
        if self.artifact is None:
            artifact_spec = None
        else:
            artifact_spec = self.artifact.legacy_spec_dict()
            if artifact_spec is None:
                return None
        if self.challenge is None:
            challenge_spec = None
        else:
            challenge_spec = self.challenge.legacy_spec_dict()
            if challenge_spec is None:
                return None
        calibration_spec = self.calibration.legacy_spec_dict()
        if calibration_spec is None:
            return None

        payload = self.spec_dict()
        payload.update(
            schema_version="continuous-task.v2",
            artifact=artifact_spec,
            challenge=challenge_spec,
            calibration=calibration_spec,
        )
        return payload

    @property
    def legacy_spec_sha256(self) -> str | None:
        payload = self.legacy_spec_dict()
        if payload is None:
            return None
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    @property
    def evaluation_plan(self) -> EvaluationPlan:
        return EvaluationPlan(
            task_spec_sha256=self.spec_sha256,
            security_tier=self.security_tier,
            evidence=self.evidence.spec_dict() if self.evidence else {"type": "custom"},
            metric_ids=tuple(target.metric_id for target in self.targets),
        )

    @property
    def legacy_evaluation_plan(self) -> EvaluationPlan | None:
        legacy_digest = self.legacy_spec_sha256
        if legacy_digest is None:
            return None
        return EvaluationPlan(
            task_spec_sha256=legacy_digest,
            security_tier=self.security_tier,
            evidence=self.evidence.spec_dict() if self.evidence else {"type": "custom"},
            metric_ids=tuple(target.metric_id for target in self.targets),
        )

    @property
    def challenge_sha256(self) -> str:
        """Challenge selection identity, independent of quality anchors."""
        payload = {
            "artifact": self.artifact.spec_dict() if self.artifact else None,
            "challenge": self.challenge.spec_dict() if self.challenge else None,
            "targets": [
                {
                    "name": target.name,
                    "metric_id": target.metric_id,
                    "prediction_column": target.prediction_column,
                    "truth_column": target.truth_column,
                }
                for target in self.targets
            ],
            "evidence": self.evidence.spec_dict() if self.evidence else None,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    @property
    def legacy_challenge_sha256(self) -> str | None:
        if self.artifact is None:
            artifact_spec = None
        else:
            artifact_spec = self.artifact.legacy_spec_dict()
            if artifact_spec is None:
                return None
        if self.challenge is None:
            challenge_spec = None
        else:
            challenge_spec = self.challenge.legacy_spec_dict()
            if challenge_spec is None:
                return None
        payload = {
            "artifact": artifact_spec,
            "challenge": challenge_spec,
            "targets": [
                {
                    "name": target.name,
                    "metric_id": target.metric_id,
                    "prediction_column": target.prediction_column,
                    "truth_column": target.truth_column,
                }
                for target in self.targets
            ],
            "evidence": self.evidence.spec_dict() if self.evidence else None,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def _load_model_challenge(
        self,
        *,
        workspace: Path,
        private: Path,
    ) -> tuple[Any, Any, EvaluationContext]:
        import pandas as pd

        if not isinstance(self.artifact, PythonPredictor) or self.challenge is None:
            raise RuntimeError("TASK does not declare a queryable model challenge")
        challenge_path = private / self.challenge.filename
        if challenge_path.suffix == ".parquet":
            bank = pd.read_parquet(challenge_path)
        elif challenge_path.suffix == ".csv":
            bank = pd.read_csv(challenge_path)
        else:
            raise RuntimeError(
                f"unsupported private challenge format: {challenge_path}"
            )
        required = {
            *self.challenge.feature_columns,
            *(target.truth_column for target in self.targets),
        }
        missing = sorted(required - set(bank.columns))
        if missing:
            raise RuntimeError(f"private challenge is missing columns: {missing}")
        if (
            self.challenge.sample_size is not None
            and len(bank) < self.challenge.sample_size
        ):
            raise RuntimeError(
                f"private challenge has {len(bank)} rows, needs "
                f"{self.challenge.sample_size}"
            )

        artifact_path = workspace / self.artifact.path
        try:
            committed_digest = workspace_artifact_digest(workspace)
        except (OSError, ValueError) as exc:
            raise AgentFault(f"could not commit submitted artifact: {exc}") from exc
        challenge_digest = (
            self.legacy_challenge_sha256
            if self.challenge.selection_policy == "artifact_digest"
            and self.legacy_challenge_sha256 is not None
            else self.challenge_sha256
        )
        context = EvaluationContext.create_from_artifact_digest(
            task_digest=challenge_digest,
            candidate_digest=committed_digest,
        )
        import numpy as np

        if self.challenge.selection_policy == "full_bank":
            selected = bank.reset_index(drop=True)
        else:
            assert self.challenge.sample_size is not None
            seed = (
                context.selection_seed("private-table-selection")
                if self.challenge.selection_policy == "stable_subset"
                else context.seed("private-table-selection")
            )
            rng = np.random.default_rng(seed)
            indices = rng.choice(
                len(bank),
                size=self.challenge.sample_size,
                replace=False,
            )
            selected = bank.iloc[indices].reset_index(drop=True)
        if len(selected) > self.artifact.max_rows:
            raise RuntimeError(
                f"private challenge selected {len(selected)} rows, exceeding "
                f"predictor max_rows={self.artifact.max_rows}"
            )
        features = selected[list(self.challenge.feature_columns)]

        def _prediction_frame(value: Any):
            if isinstance(value, list) and value and isinstance(value[0], Mapping):
                return pd.DataFrame(value)
            if isinstance(value, Mapping):
                return pd.DataFrame(dict(value))
            raise TypeError(
                "predict() must return a mapping of target columns to values "
                "or a list of row mappings"
            )

        def _load_predictor():
            return load_submitted_policy(
                artifact_path,
                factory_name=self.artifact.factory_name,
                timeout_s=self.artifact.predict_timeout_s,
                first_call_timeout_s=self.artifact.first_call_timeout_s,
                max_reply_bytes=self.artifact.max_reply_bytes,
            )

        records = features.to_dict(orient="records")
        started = time.monotonic()
        predictor = None
        try:
            predictor = _load_predictor()
            method = getattr(predictor, self.artifact.method)
            raw = method(records)
            repeated_raw = method(records)
            submission = _prediction_frame(raw)
            repeated = _prediction_frame(repeated_raw)
        except AgentFault:
            raise
        except Exception as exc:
            elapsed = time.monotonic() - started
            raise AgentFault(
                "submitted predictor failed "
                f"(artifact={committed_digest}, elapsed={elapsed:.3f}s): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if predictor is not None:
                predictor.close()
        if not _repeats_within_tolerance(submission, repeated):
            raise AgentFault(
                "predict() is not deterministic for an identical challenge batch"
            )
        prediction_columns = [target.prediction_column for target in self.targets]
        missing_predictions = sorted(set(prediction_columns) - set(submission.columns))
        if missing_predictions:
            raise AgentFault(
                f"predict() result is missing target columns: {missing_predictions}"
            )
        if len(submission) != len(selected):
            raise AgentFault(
                f"predict() returned {len(submission)} rows, expected {len(selected)}"
            )
        for column in prediction_columns:
            try:
                values = submission[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise AgentFault(
                    f"predict() target {column!r} is not numeric: {exc}"
                ) from exc
            if not np.isfinite(values).all():
                raise AgentFault(
                    f"predict() target {column!r} contains NaN or infinity"
                )

        _validate_prediction_contract(
            submission,
            value_domains=self.artifact.value_domains,
            constraints=self.artifact.constraints,
        )

        if self.artifact.prediction_scope == "row_independent":
            permutation = np.random.default_rng(
                context.selection_seed("row-independence-partitions")
            ).permutation(len(features))
            chunks = [
                chunk
                for chunk in np.array_split(
                    permutation,
                    min(self.artifact.row_independence_partitions, len(features)),
                )
                if len(chunk)
            ]
            partition_frames = []
            for partition_index, positions in enumerate(chunks):
                partition_records = features.iloc[positions].to_dict(orient="records")
                partition_started = time.monotonic()
                partition_predictor = None
                try:
                    partition_predictor = _load_predictor()
                    partition_method = getattr(
                        partition_predictor, self.artifact.method
                    )
                    partition = _prediction_frame(partition_method(partition_records))
                except AgentFault:
                    raise
                except Exception as exc:
                    elapsed = time.monotonic() - partition_started
                    raise AgentFault(
                        "submitted predictor failed row-independence probe "
                        f"{partition_index} (artifact={committed_digest}, "
                        f"elapsed={elapsed:.3f}s): {type(exc).__name__}: {exc}"
                    ) from exc
                finally:
                    if partition_predictor is not None:
                        partition_predictor.close()
                if len(partition) != len(positions):
                    raise AgentFault(
                        "predict() returned the wrong row count for a "
                        "row-independence partition"
                    )
                missing_partition = sorted(
                    set(prediction_columns) - set(partition.columns)
                )
                if missing_partition:
                    raise AgentFault(
                        "predict() partition result is missing target columns: "
                        f"{missing_partition}"
                    )
                projected = partition[prediction_columns].copy()
                projected.index = positions
                partition_frames.append(projected)
            partitioned = (
                pd.concat(partition_frames).sort_index().reset_index(drop=True)
            )
            if not _repeats_within_tolerance(
                submission[prediction_columns].reset_index(drop=True),
                partitioned,
            ):
                raise AgentFault(
                    "predict() violates prediction_scope='row_independent': "
                    "outputs changed when rows were shuffled into fresh-worker "
                    "partitions"
                )
        truth = selected[[target.truth_column for target in self.targets]].copy()
        return submission[prediction_columns], truth, context

    def _load_submission_and_truth(
        self, *, workspace: Path, private: Path
    ) -> tuple[Any, Any]:
        import pandas as pd

        if isinstance(self.artifact, PythonPredictor):
            submission, truth, _context = self._load_model_challenge(
                workspace=workspace,
                private=private,
            )
            return submission, truth
        if self.artifact is None or self.truth_filename is None:
            raise RuntimeError(
                "this TASK uses hand-authored measurement; define a module-level "
                "measure_submission(workspace, private) callback"
            )
        truth_path = private / self.truth_filename
        if truth_path.suffix == ".parquet":
            truth = pd.read_parquet(truth_path)
        elif truth_path.suffix == ".csv":
            truth = pd.read_csv(truth_path)
        else:
            raise RuntimeError(
                f"unsupported private truth format for calibrated TASK: {truth_path}"
            )
        if not isinstance(self.artifact, CsvRows):
            raise RuntimeError("unsupported continuous artifact descriptor")
        submission = load_submission_or_fault(
            workspace / self.artifact.path,
            required_columns=self.artifact.columns,
            numeric_columns=self.artifact.columns,
            n_rows=len(truth),
            extra_columns=self.artifact.extra_columns,
        )
        _validate_prediction_contract(
            submission,
            value_domains=self.artifact.value_domains,
            constraints=self.artifact.constraints,
        )
        return submission, truth

    def measure(
        self,
        *,
        workspace: Path = Path("/tmp/output"),
        private: Path = Path("/mcp_server/data"),
    ) -> dict[str, float]:
        """Scalar metrics only. Safe to call across the calibration IPC
        boundary (`grader_runner/raw_worker.py`) -- the return value is a
        plain float dict the host can JSON-serialize."""
        submission, truth = self._load_submission_and_truth(
            workspace=workspace, private=private
        )
        if self.raw_evaluator is not None:
            measured = self.raw_evaluator(submission, truth)
            if not isinstance(measured, Mapping):
                raise RuntimeError("custom continuous evaluator must return a mapping")
            return validate_metric_vector(self.targets, measured)
        return measure_registered_targets(
            self.targets,
            submission=submission,
            truth=truth,
        )

    def measure_registered(self, submission: Any, truth: Any) -> dict[str, float]:
        """Optional convenience for custom loaders with registered metric kernels."""
        return measure_registered_targets(
            self.targets,
            submission=submission,
            truth=truth,
        )

    def measure_for_grading(
        self,
        *,
        workspace: Path = Path("/tmp/output"),
        private: Path = Path("/mcp_server/data"),
    ) -> tuple[dict[str, float], dict[str, tuple[Any, Any]]]:
        """Grade-time only: scalar metrics plus the per-target (prediction,
        truth) arrays the permutation null gate needs. Must never be called
        from the calibration IPC boundary -- calibration ships only scalars
        across the process boundary; raw arrays never leave this process.

        A `custom_static` `raw_evaluator` collapses straight to metrics with
        no per-target arrays, so those targets get no entry here and the
        permutation gate is skipped for them (see `_grade_time_floors`).
        """
        submission, truth = self._load_submission_and_truth(
            workspace=workspace, private=private
        )
        if self.raw_evaluator is not None:
            measured = self.raw_evaluator(submission, truth)
            if not isinstance(measured, Mapping):
                raise RuntimeError("custom continuous evaluator must return a mapping")
            return validate_metric_vector(self.targets, measured), {}
        metrics = measure_registered_targets(
            self.targets,
            submission=submission,
            truth=truth,
        )
        raw_arrays = {
            target.name: (
                submission[target.prediction_column],
                truth[target.truth_column],
            )
            for target in self.targets
        }
        return metrics, raw_arrays

    def build_lock(
        self,
        *,
        reference_metrics: Mapping[str, Any],
        naive_metrics: Mapping[str, Any],
        degenerate_metrics: Mapping[str, Mapping[str, Any]],
        input_digests: Mapping[str, str],
    ) -> CalibrationLock:
        return build_calibration_lock(
            task_spec_sha256=self.spec_sha256,
            evaluation_plan_sha256=self.evaluation_plan.sha256,
            evaluation_plan=self.evaluation_plan.to_dict(),
            targets=self.targets,
            reference_metrics=reference_metrics,
            naive_metrics=naive_metrics,
            degenerate_metrics=degenerate_metrics,
            input_digests=input_digests,
            naive_score_min=self.naive_score_min,
            naive_score_max=self.naive_score_max,
            naive_at_floor=(
                self.naive_at_floor.spec_dict()
                if self.naive_at_floor is not None
                else None
            ),
            max_unacknowledged_naive_score_gap=(
                self.calibration.max_unacknowledged_naive_score_gap
            ),
            naive_semantic_gap_acknowledgement=(
                self.calibration.naive_semantic_gap_acknowledgement.spec_dict()
                if self.calibration.naive_semantic_gap_acknowledgement is not None
                else None
            ),
        )

    def score_metrics(
        self,
        metrics: Mapping[str, Any],
        lock: CalibrationLock,
    ) -> tuple[float, dict[str, float], float]:
        accepted_task_digests = {self.spec_sha256}
        if self.legacy_spec_sha256 is not None:
            accepted_task_digests.add(self.legacy_spec_sha256)
        if lock.payload.get("task_spec_sha256") not in accepted_task_digests:
            raise RuntimeError("calibration lock does not match the TASK registration")
        accepted_plan_digests = {self.evaluation_plan.sha256}
        if self.legacy_evaluation_plan is not None:
            accepted_plan_digests.add(self.legacy_evaluation_plan.sha256)
        if lock.payload.get("evaluation_plan_sha256") not in accepted_plan_digests:
            raise RuntimeError(
                "calibration lock does not match the current evaluation plan"
            )
        expected_range = self.spec_dict()["naive_score_range"]
        actual_range = (lock.payload.get("qualification") or {}).get(
            "naive_score_range"
        )
        if actual_range != expected_range:
            raise RuntimeError(
                "calibration lock naive qualification range does not match the TASK"
            )
        normalized_weights = normalize_weights(self.targets)
        lock_targets = lock.payload.get("targets") or {}
        if set(lock_targets) != {target.name for target in self.targets}:
            raise RuntimeError("calibration lock targets do not match the TASK")
        for target in self.targets:
            actual = lock_targets[target.name]
            expected = target.spec_dict()
            for field, value in expected.items():
                if field == "weight":
                    # Locks persist normalized weights; author specs intentionally
                    # preserve the original positive relative weights.
                    continue
                if actual.get(field) != value:
                    raise RuntimeError(
                        f"calibration lock target {target.name!r} field "
                        f"{field!r} does not match the TASK"
                    )
            if not math.isclose(
                float(actual.get("weight")),
                normalized_weights[target.name],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"calibration lock target {target.name!r} weight is stale"
                )
        finite = validate_metric_vector(self.targets, metrics)
        weights = normalized_weights
        progress = {
            target.name: target.progress(finite[target.name]) for target in self.targets
        }
        aggregate = sum(weights[name] * progress[name] for name in progress)
        final = PiecewiseLinearCurve.from_reference(lock.x_ref).score(aggregate)
        return final, progress, aggregate

    def grade(
        self,
        submission: Any,
        truth: Any,
        *,
        context: EvaluationContext | None = None,
    ) -> dict[str, Any]:
        """Grade one loaded submission through quality and evidence channels."""
        if self.raw_evaluator is not None:
            measured = self.raw_evaluator(submission, truth)
            if not isinstance(measured, Mapping):
                raise RuntimeError("custom continuous evaluator must return a mapping")
            metrics = validate_metric_vector(self.targets, measured)
            raw_arrays: dict[str, tuple[Any, Any]] = {}
        else:
            try:
                metrics = measure_registered_targets(
                    self.targets, submission=submission, truth=truth
                )
            except ValueError as exc:
                raise AgentFault(f"submitted predictions are invalid: {exc}") from exc
            raw_arrays = {
                target.name: (
                    submission[target.prediction_column],
                    truth[target.truth_column],
                )
                for target in self.targets
            }
        return self._grade_from_metrics_and_arrays(
            metrics,
            raw_arrays,
            context=context,
        )

    def _grade_from_metrics_and_arrays(
        self,
        metrics: dict[str, Any],
        raw_arrays: dict[str, tuple[Any, Any]],
        *,
        context: EvaluationContext | None = None,
    ) -> dict[str, Any]:
        lock = load_calibration_lock(
            filename=self.calibration.filename,
            task_spec_sha256=self.spec_sha256,
            compatible_task_spec_sha256s=(
                (self.legacy_spec_sha256,)
                if self.legacy_spec_sha256 is not None
                else ()
            ),
        )
        _quality_score, quality_progress, _quality_aggregate = self.score_metrics(
            metrics, lock
        )

        if self.evidence is None:
            guarded_progress = dict(quality_progress)
            receipt = PublicEvaluationReceipt(
                protocol="custom-reviewed.v1",
                plan_sha256=self.evaluation_plan.sha256,
                seed_commitment="",
                attested=False,
                challenge_count=0,
                family_alpha=1.0,
                decisions={},
            )
        else:
            context = context or EvaluationContext.create(
                task_digest=self.spec_sha256, raw_arrays=raw_arrays
            )
            evidence = evaluate_iid_evidence(
                targets=self.targets,
                raw_arrays=raw_arrays,
                context=context,
                protocol=self.evidence,
            )
            guarded_progress = {
                name: (
                    quality_progress[name] if evidence.decisions[name].accepted else 0.0
                )
                for name in quality_progress
            }
            receipt = PublicEvaluationReceipt(
                protocol=evidence.protocol.spec_dict()["type"],
                plan_sha256=self.evaluation_plan.sha256,
                seed_commitment=context.commitment,
                attested=context.attested,
                challenge_count=evidence.challenge_count,
                family_alpha=evidence.protocol.family_alpha,
                decisions=evidence.decisions,
            )
            write_private_trace(
                protocol=receipt.protocol,
                plan_sha256=receipt.plan_sha256,
                seed_commitment=receipt.seed_commitment,
                targets={
                    name: {
                        **trace,
                        "raw_metric": float(metrics[name]),
                        "quality_progress": float(quality_progress[name]),
                        "guarded_progress": float(guarded_progress[name]),
                    }
                    for name, trace in evidence.private_trace.items()
                },
                replay={
                    "nonce": context.nonce,
                    "artifact_digest": context.artifact_digest,
                },
            )

        weights = normalize_weights(self.targets)
        aggregate = sum(
            weights[name] * guarded_progress[name] for name in guarded_progress
        )
        final = PiecewiseLinearCurve.from_reference(lock.x_ref).score(aggregate)
        return {
            "score": final,
            "subscores": {
                f"{name}_progress": value for name, value in guarded_progress.items()
            },
            "weights": {f"{name}_progress": weight for name, weight in weights.items()},
            "metadata": {
                "return_shape": "calibrated_continuous",
                "security_tier": self.security_tier,
                "calibration_lock_sha256": lock.sha256,
                "task_spec_sha256": self.spec_sha256,
                "evaluation": receipt.to_dict(),
                "aggregate_progress": aggregate,
            },
        }

    def compute_score(
        self,
        workspace: Path = Path("/tmp/output"),
        trajectory: Any = None,
        private: Path = Path("/mcp_server/data"),
        transcript: str = "",
    ) -> dict[str, Any]:
        del trajectory, transcript
        if isinstance(self.artifact, PythonPredictor):
            submission, truth, context = self._load_model_challenge(
                workspace=Path(workspace),
                private=Path(private),
            )
            return self.grade(submission, truth, context=context)
        submission, truth = self._load_submission_and_truth(
            workspace=Path(workspace), private=Path(private)
        )
        return self.grade(submission, truth)

    def score(
        self,
        metrics: Mapping[str, Any],
        *,
        calibration_only: bool = False,
    ) -> dict[str, Any]:
        """Calibration-only scalar scoring; protected production must call grade()."""
        if os.environ.get(PRODUCTION_EVALUATION_ENV) == "1" and not calibration_only:
            raise RuntimeError(
                "ContinuousTask.score(metrics) is calibration-only; production "
                "compute_score must call TASK.grade(submission, truth)"
            )
        lock = load_calibration_lock(
            filename=self.calibration.filename,
            task_spec_sha256=self.spec_sha256,
            compatible_task_spec_sha256s=(
                (self.legacy_spec_sha256,)
                if self.legacy_spec_sha256 is not None
                else ()
            ),
        )
        final, progress, aggregate = self.score_metrics(metrics, lock)
        return {
            "score": final,
            "subscores": {
                f"{name}_progress": value for name, value in progress.items()
            },
            "weights": {
                f"{name}_progress": weight
                for name, weight in normalize_weights(self.targets).items()
            },
            "metadata": {
                "return_shape": "calibrated_continuous",
                "security_tier": self.security_tier,
                "calibration_lock_sha256": lock.sha256,
                "task_spec_sha256": self.spec_sha256,
                "evaluation_protected": False,
                "aggregate_progress": aggregate,
            },
        }


def load_task_module(grader_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"lbx_continuous_task_{hashlib.sha256(str(grader_path).encode()).hexdigest()[:12]}",
        grader_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot import calibrated task registration from {grader_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_from_module(module: ModuleType) -> ContinuousTask | None:
    task = getattr(module, "TASK", None)
    return task if isinstance(task, ContinuousTask) else None


def _call_measurement_callback(
    callback: Callable[..., Mapping[str, Any]],
    *,
    workspace: Path,
    private: Path,
    context: CalibrationMeasureContext | None,
) -> Mapping[str, Any]:
    signature = inspect.signature(callback)
    if not signature.parameters:
        return callback()
    kwargs: dict[str, Any] = {}
    if "workspace" in signature.parameters:
        kwargs["workspace"] = workspace
    if "private" in signature.parameters:
        kwargs["private"] = private
    if "context" in signature.parameters:
        kwargs["context"] = context or CalibrationMeasureContext()
    if kwargs:
        return callback(**kwargs)
    return callback(workspace, private)


def measure_task_module(
    module: ModuleType,
    *,
    workspace: Path,
    private: Path,
    context: CalibrationMeasureContext | None = None,
) -> dict[str, float]:
    """Invoke custom module measurement or the optional standard adapter."""
    task = task_from_module(module)
    if task is None:
        raise RuntimeError("grader module does not define a ContinuousTask as TASK")
    callback = next(
        (
            candidate
            for name in ("measure_submission", "evaluate_submission")
            if callable(candidate := getattr(module, name, None))
        ),
        None,
    )
    if callback is None:
        return task.measure(workspace=workspace, private=private)
    measured = _call_measurement_callback(
        callback,
        workspace=workspace,
        private=private,
        context=context,
    )
    if not isinstance(measured, Mapping):
        raise RuntimeError("measurement callback must return a metric mapping")
    return validate_metric_vector(task.targets, measured)


def load_task_registration(grader_path: Path) -> ContinuousTask | None:
    return task_from_module(load_task_module(grader_path))


__all__ = [
    "CALIBRATION_SEED_ENV",
    "PRODUCTION_EVALUATION_ENV",
    "CalibrationMeasureContext",
    "ContinuousTask",
    "CsvRows",
    "GeneratedCalibration",
    "PrivateTableChallenge",
    "PythonPredictor",
    "WorkspaceDegenerateProbes",
    "WorkspaceProbe",
    "load_task_module",
    "load_task_registration",
    "measure_task_module",
    "task_from_module",
]
