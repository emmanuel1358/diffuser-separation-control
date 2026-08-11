"""Ground-truth calibration orchestration for v2 continuous ML tasks."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from alignerr_plugin.ml_model_contract import (
    validate_committed_model_manifest,
    validate_ml_strategy_contract,
)
from alignerr_plugin.proof import PROOF_PATH
from alignerr_plugin.utils import read_json
from grading.evaluation import (
    CalibrationLock,
    ContinuousTask,
    PythonPredictor,
    is_classification_target,
    load_calibration_lock,
    load_task_registration,
    write_calibration_lock_atomic,
)
from grading.evaluation.context import (
    MAX_COMMITTED_BYTES,
    MAX_COMMITTED_FILES,
)
from grading.evaluation.lock import canonical_json_bytes
from grading.evaluation.metrics import validate_metric_vector

from lbx_rl_tasks_harness.models import HarnessProblem
from lbx_rl_tasks_harness.runtimes.reference import (
    ReferenceRunOptions,
    grade_workspace_in_container,
    measure_workspace_in_container,
    run_reference,
)

_IGNORED_STRATEGY_PARTS = {
    ".alignerr",
    ".cache",
    "__pycache__",
    "local_harness_run",
    "solution_run",
}
# results.txt is generated score noise. submission.csv is a Tier-B additive
# static artifact and MUST participate in strategy digests so edits invalidate
# calibration; it never substitutes for train.py / model / manifest.
_IGNORED_STRATEGY_FILES = {"results.txt", ".DS_Store"}
CALIBRATION_EVIDENCE_SCHEMA = "continuous-calibration-evidence.v1"
CALIBRATION_EVIDENCE_PATH = Path(".alignerr") / "calibration.evidence.json"
CALIBRATION_FRAMEWORK_REVISION_ENV = "LBX_CALIBRATION_FRAMEWORK_REVISION"


@dataclass(frozen=True)
class CalibrationGroundTruthResult:
    score: float
    grade_payload: dict[str, Any]
    trivial_baseline_score: float
    lock: CalibrationLock
    lock_path: Path
    evidence_path: Path
    cache_key: str
    input_digests: dict[str, str]
    reference_metrics: dict[str, float]
    naive_metrics: dict[str, float]


def _grader_source(problem: HarnessProblem) -> Path | None:
    if problem.source_problem_dir is None:
        return None
    native = problem.source_problem_dir / "scorer" / "compute_score.py"
    return native if native.is_file() else None


def load_continuous_task(problem: HarnessProblem) -> ContinuousTask | None:
    """Return the v2 TASK registration for an ML problem, if present."""
    difficulty = problem.metadata.get("difficulty") or {}
    if str(difficulty.get("task_type") or "").strip().lower() != "ml":
        return None
    source = _grader_source(problem)
    if source is None:
        return None
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    declares_task = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(
                isinstance(target, ast.Name) and target.id == "TASK"
                for target in node.targets
            )
            if isinstance(node, ast.Assign)
            else isinstance(node.target, ast.Name) and node.target.id == "TASK"
        )
        for node in tree.body
    )
    return load_task_registration(source) if declares_task else None


def _tree_sha256(
    path: Path,
    *,
    ignore_generated_strategy_outputs: bool = False,
) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    for file in files:
        relative = file.name if path.is_file() else file.relative_to(path).as_posix()
        parts = Path(relative).parts
        if any(part in _IGNORED_STRATEGY_PARTS for part in parts):
            continue
        if ignore_generated_strategy_outputs and file.name in _IGNORED_STRATEGY_FILES:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def calibration_input_digests(
    problem: HarnessProblem,
    task: ContinuousTask,
) -> dict[str, str]:
    if problem.source_problem_dir is None:
        raise ValueError("calibration requires a source problem directory")
    root = problem.source_problem_dir
    proof_path = root / PROOF_PATH
    proof = read_json(proof_path) if proof_path.is_file() else {}
    reference_dir = root / "solution"
    inputs = {
        "task_spec": task.spec_sha256,
        "task_config": _tree_sha256(root / "task.toml"),
        "environment": _tree_sha256(root / "environment"),
        "dockerfile": _tree_sha256(root / "environment" / "Dockerfile"),
        "grader": _tree_sha256(
            _grader_source(problem) or root / "scorer" / "compute_score.py"
        ),
        "data_generation": _tree_sha256(root / "data_generation"),
        "public_data": _tree_sha256(root / "data"),
        "private_data": _tree_sha256(root / "scorer" / "data"),
        "reference_strategy": _tree_sha256(
            reference_dir, ignore_generated_strategy_outputs=True
        ),
        "naive_strategy": _tree_sha256(
            root / task.naive, ignore_generated_strategy_outputs=True
        ),
        "image": str(proof.get("image_digest") or ""),
        "base_image": str(proof.get("base_image_ref") or ""),
    }
    provider = task.calibration.degenerate_probes
    if provider is not None:
        seen: dict[str, str] = {}
        for probe in provider.probes:
            probe_path = _validated_probe_source(root, probe.path)
            digest = _secure_probe_digest(probe_path)
            if digest in seen:
                raise ValueError(
                    f"degenerate probes {seen[digest]!r} and {probe.name!r} "
                    "have identical workspace contents"
                )
            seen[digest] = probe.name
            inputs[f"degenerate_probe:{probe.name}"] = digest
    return inputs


def calibration_cache_key(
    problem: HarnessProblem,
    task: ContinuousTask,
    *,
    framework_revision: str | None = None,
) -> str:
    """Digest semantic calibration inputs, excluding one concrete image build."""
    inputs = calibration_input_digests(problem, task)
    source_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in {"image", "base_image"}
    }
    payload = {
        "schema_version": "continuous-calibration-cache-key.v1",
        "framework_revision": (
            framework_revision
            if framework_revision is not None
            else os.environ.get(CALIBRATION_FRAMEWORK_REVISION_ENV, "local")
        ),
        "inputs": source_inputs,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def calibration_evidence_payload(
    *,
    task: ContinuousTask,
    lock: CalibrationLock,
    input_digests: dict[str, str],
    cache_key: str,
) -> dict[str, Any]:
    return {
        "schema_version": CALIBRATION_EVIDENCE_SCHEMA,
        "cache_key": cache_key,
        "lock_sha256": lock.sha256,
        "task_spec_sha256": task.spec_sha256,
        "evaluation_plan_sha256": task.evaluation_plan.sha256,
        "security_tier": task.security_tier,
        "inputs": dict(sorted(input_digests.items())),
        "qualification": lock.payload["qualification"],
    }


def write_calibration_evidence_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    if not source.exists():
        return
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _read_grade_payload(verifier_dir: Path, score: float) -> dict[str, Any]:
    for name in ("reward-details.json", "reward.json"):
        path = verifier_dir / name
        if path.is_file():
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                return payload
    return {"score": score}


_CALIBRATION_SEED = 0
_DEGENERATE_SEED = 0


def _measure_tabular_degenerate_family(
    problem: HarnessProblem,
    task: ContinuousTask,
    run_dir: Path,
) -> dict[str, dict[str, float]]:
    """Measure the no-information strategy family used to bound each floor.

    Every strategy is built on the host from public train data and the
    truth values stripped of row correspondence -- never from author code --
    and scored with the registered metric kernels (or the task's in-container
    measurement path). Queryable-model strategies use the same challenge row
    count as reference, naive, and production evaluation. See
    ``no_info_ceiling`` in ``grading.evaluation.metrics`` for how these raw
    metrics turn into a floor bound.
    """
    import numpy as np
    import pandas as pd

    source = problem.source_problem_dir
    public_dir = source / "data"
    train_path = public_dir / "train.parquet"
    if not train_path.is_file():
        train_path = public_dir / "train.csv"
    if not train_path.is_file():
        raise ValueError(
            "built-in tabular degenerate probes require data/train.parquet "
            "or train.csv; declare GeneratedCalibration(degenerate_probes="
            "WorkspaceDegenerateProbes(...)) for non-tabular tasks"
        )
    train = (
        pd.read_parquet(train_path)
        if train_path.suffix == ".parquet"
        else pd.read_csv(train_path)
    )
    # Hand-authored (calibrated) tasks own submission loading and do not declare
    # a truth filename on TASK, so fall back to the conventional private target.
    private_dir = source / "scorer" / "data"
    if task.truth_filename is not None:
        truth_path = private_dir / task.truth_filename
    else:
        truth_path = private_dir / "test_target.parquet"
        if not truth_path.is_file():
            truth_path = private_dir / "test_target.csv"
    if not truth_path.is_file():
        raise ValueError(
            "built-in tabular degenerate probes require a private truth table; "
            "declare workspace degenerate probes for non-tabular tasks"
        )
    truth = (
        pd.read_parquet(truth_path)
        if truth_path.suffix == ".parquet"
        else pd.read_csv(truth_path)
    )
    truth_columns = [target.truth_column for target in task.targets]
    truth = truth[truth_columns]
    if isinstance(task.artifact, PythonPredictor):
        if task.challenge is None:
            raise ValueError("PythonPredictor calibration requires a challenge")
        sample_size = task.challenge.sample_size
        if sample_size is not None and len(truth) < sample_size:
            raise ValueError(
                f"private challenge has {len(truth)} rows, needs {sample_size}"
            )
        if task.challenge.selection_policy == "stable_subset":
            from grading.evaluation.context import EvaluationContext

            assert sample_size is not None
            context = EvaluationContext.create_from_artifact_digest(
                task_digest=task.challenge_sha256,
                candidate_digest="calibration-degenerate-family",
            )
            rng = np.random.default_rng(
                context.selection_seed("private-table-selection")
            )
            indices = rng.choice(len(truth), size=sample_size, replace=False)
            truth = truth.iloc[indices].reset_index(drop=True)
        elif task.challenge.selection_policy == "artifact_digest":
            assert sample_size is not None
            rng = np.random.default_rng(_DEGENERATE_SEED)
            indices = rng.choice(len(truth), size=sample_size, replace=False)
            truth = truth.iloc[indices].reset_index(drop=True)
    is_classification = {
        target.truth_column: is_classification_target(target) for target in task.targets
    }
    prediction_columns = {
        target.truth_column: target.prediction_column for target in task.targets
    }

    def majority(column: "pd.Series") -> float:
        return float(column.mode().iloc[0])

    def constant_mean(train: "pd.DataFrame", truth: "pd.DataFrame") -> "pd.DataFrame":
        out = {}
        for column in truth.columns:
            value = (
                majority(train[column])
                if is_classification[column]
                else float(train[column].mean())
            )
            out[column] = np.full(len(truth), value)
        return pd.DataFrame(out)

    def constant_median(train: "pd.DataFrame", truth: "pd.DataFrame") -> "pd.DataFrame":
        out = {}
        for column in truth.columns:
            if is_classification[column]:
                value = float(round(float(train[column].median())))
            else:
                value = float(train[column].median())
            out[column] = np.full(len(truth), value)
        return pd.DataFrame(out)

    def jittered_constant(
        train: "pd.DataFrame", truth: "pd.DataFrame"
    ) -> "pd.DataFrame":
        rng = np.random.default_rng(_DEGENERATE_SEED)
        out = {}
        for column in truth.columns:
            if is_classification[column]:
                out[column] = np.full(len(truth), majority(train[column]))
            else:
                mean = float(train[column].mean())
                scale = max(1e-6, 1e-3 * float(train[column].std(ddof=0)))
                out[column] = mean + rng.normal(0.0, scale, size=len(truth))
        return pd.DataFrame(out)

    def shuffled_truth(train: "pd.DataFrame", truth: "pd.DataFrame") -> "pd.DataFrame":
        del train
        rng = np.random.default_rng(_DEGENERATE_SEED)
        out = {}
        for column in truth.columns:
            values = truth[column].to_numpy().copy()
            rng.shuffle(values)
            out[column] = values
        return pd.DataFrame(out)

    def row_index(train: "pd.DataFrame", truth: "pd.DataFrame") -> "pd.DataFrame":
        del train
        n = len(truth)
        norm = np.arange(n, dtype=float) / max(1, n - 1)
        out = {}
        for column in truth.columns:
            out[column] = (
                np.rint(norm).astype(int) if is_classification[column] else norm.copy()
            )
        return pd.DataFrame(out)

    def fixed_class(label: float):
        """Constant classification prediction, independent of train prevalence.

        Train-derived majority/median constants can miss the best feature-blind
        constant when train/test class prevalence differs. Trying both fixed
        classes directly closes that gap for binary targets.
        """

        def generate(train: "pd.DataFrame", truth: "pd.DataFrame") -> "pd.DataFrame":
            out = {}
            for column in truth.columns:
                value = (
                    label if is_classification[column] else float(train[column].mean())
                )
                out[column] = np.full(len(truth), value)
            return pd.DataFrame(out)

        return generate

    strategies = {
        "constant-mean": constant_mean,
        "constant-median": constant_median,
        "jittered-constant": jittered_constant,
        "shuffled-truth": shuffled_truth,
        "row-index": row_index,
        "constant-negative": fixed_class(0.0),
        "constant-positive": fixed_class(1.0),
    }

    if isinstance(task.artifact, PythonPredictor):
        return {
            name: task.measure_registered(
                generate(train, truth).rename(columns=prediction_columns),
                truth,
            )
            for name, generate in strategies.items()
        }

    # Calibrated tasks have no declared CSV artifact schema; the hand-authored
    # measurement reads a conventional submission.csv over the prediction columns.
    if task.artifact is not None:
        artifact_relpath = task.artifact.path
        artifact_columns = list(task.artifact.columns)
    else:
        artifact_relpath = "submission.csv"
        artifact_columns = [target.prediction_column for target in task.targets]

    measurements: dict[str, dict[str, float]] = {}
    for name, generate in strategies.items():
        frame = generate(train, truth).rename(columns=prediction_columns)
        workspace = run_dir / "calibration" / f"degenerate-{name}"
        artifact_path = workspace / artifact_relpath
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        frame[artifact_columns].to_csv(artifact_path, index=False)
        measured = measure_workspace_in_container(
            problem,
            workspace,
            run_dir / "calibration" / f"degenerate-{name}-measure",
            run_dir / "calibration" / f"degenerate-{name}-measure.txt",
        )
        measurements[name] = _validated_measurement_metrics(
            measured, task=task, label=f"degenerate probe {name!r}"
        )
    return measurements


def _validated_probe_source(task_root: Path, relative_path: str) -> Path:
    source = task_root / relative_path
    try:
        info = os.lstat(source)
    except FileNotFoundError as exc:
        raise ValueError(
            f"degenerate probe workspace is missing: {relative_path}"
        ) from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(
            "degenerate probe root must be a real directory, not a symlink or "
            f"special file: {relative_path}"
        )
    resolved = source.resolve(strict=True)
    try:
        resolved.relative_to(task_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(
            f"degenerate probe workspace escapes the task root: {relative_path}"
        ) from exc
    return resolved


def _secure_probe_digest(source: Path) -> str:
    """Hash a regular-file tree using no-follow file descriptors."""

    entries: list[tuple[str, Path, os.stat_result]] = []
    total_bytes = 0
    for directory, dirnames, filenames in os.walk(source, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            candidate = directory_path / dirname
            if not stat.S_ISDIR(os.lstat(candidate).st_mode):
                raise ValueError(
                    "degenerate probe workspace contains a non-directory entry: "
                    f"{candidate.relative_to(source).as_posix()}"
                )
        for filename in sorted(filenames):
            candidate = directory_path / filename
            info = os.lstat(candidate)
            relative = candidate.relative_to(source).as_posix()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(
                    "degenerate probe workspace contains a non-regular entry: "
                    f"{relative}"
                )
            total_bytes += int(info.st_size)
            if total_bytes > MAX_COMMITTED_BYTES:
                raise ValueError(
                    f"degenerate probe workspace exceeds {MAX_COMMITTED_BYTES} bytes"
                )
            entries.append((relative, candidate, info))
    if not entries:
        raise ValueError(f"degenerate probe workspace is empty: {source}")
    if len(entries) > MAX_COMMITTED_FILES:
        raise ValueError(
            f"degenerate probe workspace has {len(entries)} files, over limit "
            f"{MAX_COMMITTED_FILES}"
        )

    digest = hashlib.sha256()
    for relative, path, expected_info in sorted(entries):
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != expected_info.st_dev
                or opened.st_ino != expected_info.st_ino
                or opened.st_size != expected_info.st_size
            ):
                raise ValueError(f"degenerate probe changed while hashing: {relative}")
            file_digest = hashlib.sha256()
            with os.fdopen(fd, "rb", closefd=False) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            relative_bytes = relative.encode("utf-8")
            digest.update(len(relative_bytes).to_bytes(8, "big"))
            digest.update(relative_bytes)
            digest.update(int(opened.st_size).to_bytes(8, "big"))
            digest.update(file_digest.digest())
        finally:
            os.close(fd)
    return digest.hexdigest()


def _secure_copy_probe_workspace(source: Path, destination: Path) -> str:
    """Snapshot one committed probe without following non-regular entries."""

    digest = _secure_probe_digest(source)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        info = os.lstat(path)
        if stat.S_ISDIR(info.st_mode):
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                "degenerate probe workspace contains a non-regular entry: "
                f"{relative.as_posix()}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError(
                    "degenerate probe changed while being snapshotted: "
                    f"{relative.as_posix()}"
                )
            with os.fdopen(fd, "rb", closefd=False) as source_handle:
                with target.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle)
        finally:
            os.close(fd)
    if _secure_probe_digest(destination) != digest:
        raise ValueError(f"degenerate probe snapshot digest changed: {source}")
    return digest


def _validated_measurement_metrics(
    payload: dict[str, Any],
    *,
    task: ContinuousTask,
    label: str,
) -> dict[str, float]:
    if payload.get("schema_version") != "raw-continuous-metrics.v1":
        raise RuntimeError(f"{label} returned an unsupported raw-metric schema")
    if payload.get("task_spec_sha256") != task.spec_sha256:
        raise RuntimeError(f"{label} raw metrics were produced by a stale TASK")
    if int(payload.get("calibration_seed", -1)) != _CALIBRATION_SEED:
        raise RuntimeError(f"{label} did not use the shared calibration seed")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"{label} raw metrics are missing")
    return validate_metric_vector(task.targets, metrics)


def _measure_workspace_degenerate_family(
    problem: HarnessProblem,
    task: ContinuousTask,
    run_dir: Path,
) -> dict[str, dict[str, float]]:
    source_root = problem.source_problem_dir
    if source_root is None or task.calibration.degenerate_probes is None:
        raise ValueError("workspace degenerate probes are not configured")

    measurements: dict[str, dict[str, float]] = {}
    seen_digests: dict[str, str] = {}
    for probe in sorted(
        task.calibration.degenerate_probes.probes, key=lambda item: item.name
    ):
        source = _validated_probe_source(source_root, probe.path)
        first_workspace = run_dir / "calibration" / f"degenerate-{probe.name}-first"
        digest = _secure_copy_probe_workspace(source, first_workspace)
        if digest in seen_digests:
            raise ValueError(
                f"degenerate probes {seen_digests[digest]!r} and {probe.name!r} "
                "have identical workspace contents"
            )
        seen_digests[digest] = probe.name
        first = measure_workspace_in_container(
            problem,
            first_workspace,
            run_dir / "calibration" / f"degenerate-{probe.name}-first-measure",
            run_dir / "calibration" / f"degenerate-{probe.name}-first-measure.txt",
        )
        first_metrics = _validated_measurement_metrics(
            first, task=task, label=f"degenerate probe {probe.name!r}"
        )

        second_workspace = run_dir / "calibration" / f"degenerate-{probe.name}-second"
        _secure_copy_probe_workspace(source, second_workspace)
        second = measure_workspace_in_container(
            problem,
            second_workspace,
            run_dir / "calibration" / f"degenerate-{probe.name}-second-measure",
            run_dir / "calibration" / f"degenerate-{probe.name}-second-measure.txt",
        )
        second_metrics = _validated_measurement_metrics(
            second, task=task, label=f"degenerate probe {probe.name!r} replay"
        )
        if any(
            not math.isclose(
                first_metrics[name],
                second_metrics[name],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for name in first_metrics
        ):
            raise RuntimeError(
                f"degenerate probe {probe.name!r} is nondeterministic under the "
                "shared calibration context"
            )
        measurements[probe.name] = first_metrics
    return measurements


def _measure_degenerate_family(
    problem: HarnessProblem,
    task: ContinuousTask,
    run_dir: Path,
) -> dict[str, dict[str, float]]:
    if task.calibration.degenerate_probes is not None:
        return _measure_workspace_degenerate_family(problem, task, run_dir)
    try:
        return _measure_tabular_degenerate_family(problem, task, run_dir)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "built-in tabular degenerate probes could not construct numeric "
            "submission tables; declare WorkspaceDegenerateProbes for this task"
        ) from exc


def _run_ml_calibrated_ground_truth_impl(
    problem: HarnessProblem,
    *,
    run_dir: Path,
    workspace: Path,
    verifier_dir: Path,
    transcript_path: Path,
    check: bool = False,
) -> CalibrationGroundTruthResult:
    """Generate a lock from committed strategies, then replay the final scorer."""
    if problem.source_problem_dir is None:
        raise ValueError("calibrated ground truth requires a source problem directory")
    task = load_continuous_task(problem)
    if task is None:
        raise ValueError("problem does not define a v2 continuous TASK")
    source = problem.source_problem_dir
    reference_strategy = source / "solution"
    naive_strategy = source / task.naive
    # Fail closed before any container solve: require train+weights+manifest and
    # an inference-only entrypoint. Never invoke training_entrypoint.
    reference_contract = validate_ml_strategy_contract(
        reference_strategy, role="reference"
    )
    naive_contract = validate_ml_strategy_contract(naive_strategy, role="naive")

    reference_dir = "solution"
    reference_cache = (run_dir / "calibration" / "reference-cache").resolve()
    reference_problem = replace(
        problem,
        reference=replace(
            problem.reference,
            cache_dir=str(reference_cache),
            entrypoint=reference_contract.inference_entrypoint,
        ),
    )
    run_reference(
        reference_problem,
        workspace,
        verifier_dir,
        transcript_path,
        options=ReferenceRunOptions(
            mode="iterate" if check else "prove",
            no_grade=True,
            clean_cache=True,
            solution_dir=reference_dir,
            write_manifest=False,
        ),
        run_dir=run_dir,
    )
    reference_artifacts = run_dir / "calibration" / "reference"
    _copy_tree(reference_cache, reference_artifacts)
    reference_measurement = measure_workspace_in_container(
        problem,
        reference_artifacts,
        run_dir / "calibration" / "reference-measure",
        run_dir / "calibration" / "reference-measure.txt",
    )
    reference_metrics = _validated_measurement_metrics(
        reference_measurement,
        task=task,
        label="reference strategy",
    )

    naive_workspace = run_dir / "calibration" / "naive-workspace"
    naive_workspace.mkdir(parents=True, exist_ok=True)
    naive_cache = (run_dir / "calibration" / "naive-cache").resolve()
    naive_problem = replace(
        problem,
        reference=replace(
            problem.reference,
            cache_dir=str(naive_cache),
            entrypoint=naive_contract.inference_entrypoint,
        ),
    )
    run_reference(
        naive_problem,
        naive_workspace,
        run_dir / "calibration" / "naive-verifier",
        run_dir / "calibration" / "naive-inference.txt",
        options=ReferenceRunOptions(
            mode="iterate" if check else "prove",
            no_grade=True,
            clean_cache=True,
            solution_dir=task.naive,
            write_manifest=False,
        ),
        run_dir=run_dir,
    )
    naive_artifacts = run_dir / "calibration" / "naive"
    _copy_tree(naive_cache, naive_artifacts)
    naive_measurement = measure_workspace_in_container(
        problem,
        naive_artifacts,
        run_dir / "calibration" / "naive-measure",
        run_dir / "calibration" / "naive-measure.txt",
    )
    naive_metrics = _validated_measurement_metrics(
        naive_measurement,
        task=task,
        label="naive strategy",
    )

    degenerate_measurements = _measure_degenerate_family(problem, task, run_dir)

    input_digests = calibration_input_digests(problem, task)
    cache_key = calibration_cache_key(problem, task)
    lock = task.build_lock(
        reference_metrics=reference_metrics,
        naive_metrics=naive_metrics,
        degenerate_metrics=degenerate_measurements,
        input_digests=input_digests,
    )
    staged_lock = run_dir / "calibration" / "calibration.lock.json"
    write_calibration_lock_atomic(staged_lock, lock)

    score = grade_workspace_in_container(
        problem,
        reference_artifacts,
        verifier_dir,
        transcript_path,
        calibration_lock=staged_lock,
    )
    grade_payload = _read_grade_payload(verifier_dir, score)

    noop_workspace = run_dir / "calibration" / "noop"
    noop_verifier = run_dir / "calibration" / "noop-verifier"
    noop_workspace.mkdir(parents=True)
    trivial_score = grade_workspace_in_container(
        problem,
        noop_workspace,
        noop_verifier,
        run_dir / "calibration" / "noop-grade.txt",
        calibration_lock=staged_lock,
    )
    if abs(score - 0.5) > problem.ground_truth.continuous_score_epsilon:
        raise RuntimeError(
            "generated calibration failed reference replay: expected "
            f"0.5 +/- {problem.ground_truth.continuous_score_epsilon}, got {score}"
        )
    if trivial_score > problem.ground_truth.zero_anchor_epsilon:
        raise RuntimeError(
            "generated calibration failed zero-anchor replay: expected no-op <= "
            f"{problem.ground_truth.zero_anchor_epsilon}, got {trivial_score}"
        )
    if abs(float(lock.payload["qualification"]["oracle_score"]) - 1.0) > 1e-12:
        raise RuntimeError("generated calibration failed the oracle=1 contract")

    lock_path = source / task.calibration.filename
    evidence_path = source / CALIBRATION_EVIDENCE_PATH
    evidence_payload = calibration_evidence_payload(
        task=task,
        lock=lock,
        input_digests=input_digests,
        cache_key=cache_key,
    )
    if check:
        committed = load_calibration_lock(lock_path, task_spec_sha256=task.spec_sha256)
        if committed.payload != lock.payload:
            raise RuntimeError(
                "development calibration.lock.json is stale; rerun the ML "
                "ground-truth workflow"
            )
        evidence = read_json(evidence_path) if evidence_path.is_file() else {}
        if evidence != evidence_payload:
            raise RuntimeError(
                "development calibration evidence is stale; rerun the ML "
                "ground-truth workflow"
            )
    else:
        write_calibration_lock_atomic(lock_path, lock)
        write_calibration_evidence_atomic(evidence_path, evidence_payload)
    _copy_tree(reference_artifacts, workspace)
    return CalibrationGroundTruthResult(
        score=score,
        grade_payload=grade_payload,
        trivial_baseline_score=trivial_score,
        lock=lock,
        lock_path=lock_path,
        evidence_path=evidence_path,
        cache_key=cache_key,
        input_digests=input_digests,
        reference_metrics={
            str(key): float(value) for key, value in reference_metrics.items()
        },
        naive_metrics={str(key): float(value) for key, value in naive_metrics.items()},
    )


def _restore_file(path: Path, content: bytes | None) -> None:
    if content is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def run_ml_calibrated_ground_truth(
    problem: HarnessProblem,
    *,
    run_dir: Path,
    workspace: Path,
    verifier_dir: Path,
    transcript_path: Path,
    check: bool = False,
) -> CalibrationGroundTruthResult:
    """Run calibration transactionally, restoring committed evidence on failure."""
    if problem.source_problem_dir is None:
        raise ValueError("calibrated ground truth requires a source problem directory")
    task = load_continuous_task(problem)
    if task is None:
        raise ValueError("problem does not define a v2 continuous TASK")
    lock_path = problem.source_problem_dir / task.calibration.filename
    proof_path = problem.source_problem_dir / PROOF_PATH
    evidence_path = problem.source_problem_dir / CALIBRATION_EVIDENCE_PATH
    lock_before = lock_path.read_bytes() if lock_path.is_file() else None
    proof_before = proof_path.read_bytes() if proof_path.is_file() else None
    evidence_before = evidence_path.read_bytes() if evidence_path.is_file() else None
    try:
        return _run_ml_calibrated_ground_truth_impl(
            problem,
            run_dir=run_dir,
            workspace=workspace,
            verifier_dir=verifier_dir,
            transcript_path=transcript_path,
            check=check,
        )
    except Exception:
        _restore_file(lock_path, lock_before)
        _restore_file(proof_path, proof_before)
        _restore_file(evidence_path, evidence_before)
        raise


__all__ = [
    "CALIBRATION_EVIDENCE_PATH",
    "CALIBRATION_EVIDENCE_SCHEMA",
    "CalibrationGroundTruthResult",
    "calibration_cache_key",
    "calibration_evidence_payload",
    "calibration_input_digests",
    "load_continuous_task",
    "run_ml_calibrated_ground_truth",
    "validate_committed_model_manifest",
    "write_calibration_evidence_atomic",
]
