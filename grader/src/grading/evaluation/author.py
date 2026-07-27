"""Declarative author surface for calibrated continuous tasks."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

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


@dataclass(frozen=True)
class CsvRows:
    path: str
    columns: tuple[str, ...]
    allow_extra_columns: bool = False

    def __init__(
        self,
        path: str,
        *,
        columns: list[str] | tuple[str, ...],
        allow_extra_columns: bool = False,
    ) -> None:
        normalized = tuple(str(column) for column in columns)
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(
                "CSV artifact path must be non-empty and workspace-relative"
            )
        if not normalized or any(not column for column in normalized):
            raise ValueError("CSV artifact columns must be non-empty")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "columns", normalized)
        object.__setattr__(self, "allow_extra_columns", bool(allow_extra_columns))

    def spec_dict(self) -> dict[str, Any]:
        return {
            "type": "csv_rows.v1",
            "path": self.path,
            "columns": list(self.columns),
            "allow_extra_columns": self.allow_extra_columns,
        }


@dataclass(frozen=True)
class PythonPredictor:
    """Queryable Python artifact evaluated only on post-commit private rows."""

    path: str = "predictor.py"
    factory_name: str = "load_predictor"
    method: str = "predict"

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

    def spec_dict(self) -> dict[str, Any]:
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
    sample_size: int

    def __init__(
        self,
        filename: str,
        *,
        feature_columns: list[str] | tuple[str, ...],
        sample_size: int,
    ) -> None:
        relative = Path(filename)
        columns = tuple(str(column) for column in feature_columns)
        if (
            not filename
            or relative.is_absolute()
            or ".." in relative.parts
            or not columns
            or any(not column for column in columns)
            or sample_size < 32
        ):
            raise ValueError("invalid private table challenge descriptor")
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "feature_columns", columns)
        object.__setattr__(self, "sample_size", int(sample_size))

    def spec_dict(self) -> dict[str, Any]:
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

    def __post_init__(self) -> None:
        if self.filename != DEFAULT_LOCK_FILENAME:
            raise ValueError(
                f"generated calibration filename is fixed to {DEFAULT_LOCK_FILENAME!r}"
            )

    def spec_dict(self) -> dict[str, Any]:
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
            "schema_version": "continuous-task.v2",
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

    @property
    def evaluation_plan(self) -> EvaluationPlan:
        return EvaluationPlan(
            task_spec_sha256=self.spec_sha256,
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
        if len(bank) < self.challenge.sample_size:
            raise RuntimeError(
                f"private challenge has {len(bank)} rows, needs "
                f"{self.challenge.sample_size}"
            )

        artifact_path = workspace / self.artifact.path
        try:
            committed_digest = workspace_artifact_digest(workspace)
        except (OSError, ValueError) as exc:
            raise AgentFault(f"could not commit submitted artifact: {exc}") from exc
        context = EvaluationContext.create_from_artifact_digest(
            task_digest=self.challenge_sha256,
            candidate_digest=committed_digest,
        )
        import numpy as np

        rng = np.random.default_rng(context.seed("private-table-selection"))
        indices = rng.choice(
            len(bank),
            size=self.challenge.sample_size,
            replace=False,
        )
        selected = bank.iloc[indices].reset_index(drop=True)
        features = selected[list(self.challenge.feature_columns)]

        predictor = load_submitted_policy(
            artifact_path,
            factory_name=self.artifact.factory_name,
            timeout_s=60.0,
        )
        try:
            method = getattr(predictor, self.artifact.method)
            raw = method(features.to_dict(orient="records"))
            repeated_raw = method(features.to_dict(orient="records"))
        except AgentFault:
            raise
        except Exception as exc:
            raise AgentFault(
                f"submitted predictor failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            predictor.close()

        def _prediction_frame(value: Any):
            if isinstance(value, list) and value and isinstance(value[0], Mapping):
                return pd.DataFrame(value)
            if isinstance(value, Mapping):
                return pd.DataFrame(dict(value))
            raise TypeError(
                "predict() must return a mapping of target columns to values "
                "or a list of row mappings"
            )

        try:
            submission = _prediction_frame(raw)
            repeated = _prediction_frame(repeated_raw)
        except Exception as exc:
            raise AgentFault(f"predict() returned an invalid table: {exc}") from exc
        if list(submission.columns) != list(repeated.columns) or not submission.equals(
            repeated
        ):
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
            allow_extra_columns=self.artifact.allow_extra_columns,
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
        )

    def score_metrics(
        self,
        metrics: Mapping[str, Any],
        lock: CalibrationLock,
    ) -> tuple[float, dict[str, float], float]:
        if lock.payload.get("task_spec_sha256") != self.spec_sha256:
            raise RuntimeError("calibration lock does not match the TASK registration")
        if lock.payload.get("evaluation_plan_sha256") != self.evaluation_plan.sha256:
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
