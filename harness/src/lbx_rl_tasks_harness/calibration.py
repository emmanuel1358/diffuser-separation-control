"""Ground-truth calibration orchestration for v2 continuous ML tasks."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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
from grading.evaluation.lock import canonical_json_bytes

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
_IGNORED_STRATEGY_FILES = {"results.txt", "submission.csv", ".DS_Store"}
MODEL_MANIFEST_FILENAME = "model.manifest.json"
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
    root = problem.source_problem_dir
    test_file = root / "test_file.py"
    if test_file.is_file():
        return test_file
    native = root / "scorer" / "compute_score.py"
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_committed_model_manifest(
    strategy_dir: Path, *, role: str
) -> dict[str, Any]:
    """Validate a committed trained-model manifest without loading the model."""
    manifest_path = strategy_dir / MODEL_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(
            f"{role} strategy is missing {MODEL_MANIFEST_FILENAME}: {strategy_dir}"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role} model manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0":
        raise ValueError(f"{role} model manifest must use schema_version '1.0'")
    if manifest.get("role") != role:
        raise ValueError(
            f"{role} model manifest role must be {role!r}, got {manifest.get('role')!r}"
        )
    for field in ("training_entrypoint", "inference_entrypoint"):
        raw = manifest.get(field)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{role} model manifest is missing {field}")
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{role} model manifest {field} must be strategy-relative")
        if not (strategy_dir / relative).is_file():
            raise ValueError(f"{role} model manifest {field} does not exist: {raw}")
    if not isinstance(manifest.get("seed"), int):
        raise ValueError(f"{role} model manifest seed must be an integer")
    training_data = manifest.get("public_training_data")
    if not isinstance(training_data, dict):
        raise ValueError(f"{role} model manifest is missing public_training_data")
    training_path_raw = training_data.get("path")
    training_sha = training_data.get("sha256")
    if not isinstance(training_path_raw, str) or not isinstance(training_sha, str):
        raise ValueError(
            f"{role} model manifest public_training_data needs path and sha256"
        )
    task_root = next(
        (
            parent
            for parent in (strategy_dir, *strategy_dir.parents)
            if (parent / "metadata.json").is_file() or (parent / "task.toml").is_file()
        ),
        None,
    )
    if task_root is None:
        raise ValueError(f"could not resolve task root for {role} model manifest")
    training_path = (strategy_dir / training_path_raw).resolve()
    try:
        training_path.relative_to(task_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{role} model training data escapes the task root: {training_path_raw}"
        ) from exc
    if not training_path.is_file():
        raise ValueError(f"{role} model training data is missing: {training_path_raw}")
    actual_training_sha = _file_sha256(training_path)
    if actual_training_sha != training_sha:
        raise ValueError(
            f"{role} model is stale for public training data {training_path_raw}: "
            f"expected {training_sha}, got {actual_training_sha}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{role} model manifest must declare trained artifacts")
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ValueError(f"{role} model manifest artifact entries must be objects")
        raw_path = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ValueError(f"{role} model manifest artifact needs path and sha256")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{role} model artifact path must be strategy-relative")
        artifact = strategy_dir / relative
        if not artifact.is_file():
            raise ValueError(f"{role} trained model artifact is missing: {raw_path}")
        actual = _file_sha256(artifact)
        if actual != expected:
            raise ValueError(
                f"{role} model artifact digest mismatch for {raw_path}: "
                f"expected {expected}, got {actual}"
            )
    return manifest


def calibration_input_digests(
    problem: HarnessProblem,
    task: ContinuousTask,
) -> dict[str, str]:
    if problem.source_problem_dir is None:
        raise ValueError("calibration requires a source problem directory")
    root = problem.source_problem_dir
    proof_path = root / PROOF_PATH
    proof = read_json(proof_path) if proof_path.is_file() else {}
    reference_dir = root / (
        "reference_solution" if (root / "reference_solution").is_dir() else "solution"
    )
    task_config = (
        root / "task.toml" if (root / "task.toml").is_file() else root / "metadata.json"
    )
    return {
        "task_spec": task.spec_sha256,
        "task_config": _tree_sha256(task_config),
        "environment": _tree_sha256(root / "environment"),
        "dockerfile": _tree_sha256(root / "Dockerfile"),
        "grader": _tree_sha256(_grader_source(problem) or root / "test_file.py"),
        "data_generation": _tree_sha256(root / "data-generation"),
        "public_data": _tree_sha256(root / "data" / "public"),
        "private_data": _tree_sha256(root / "data" / "private"),
        "reference_strategy": _tree_sha256(
            reference_dir, ignore_generated_strategy_outputs=True
        ),
        "naive_strategy": _tree_sha256(
            root / task.naive, ignore_generated_strategy_outputs=True
        ),
        "image": str(proof.get("image_digest") or ""),
        "base_image": str(proof.get("base_image_ref") or ""),
    }


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


_DEGENERATE_SEED = 0


def _measure_degenerate_family(
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
    public_dir = source / "data" / "public"
    train_path = public_dir / "train.parquet"
    if not train_path.is_file():
        train_path = public_dir / "train.csv"
    train = (
        pd.read_parquet(train_path)
        if train_path.suffix == ".parquet"
        else pd.read_csv(train_path)
    )
    # Hand-authored (calibrated) tasks own submission loading and do not declare
    # a truth filename on TASK, so fall back to the conventional private target.
    private_dir = source / "data" / "private"
    if task.truth_filename is not None:
        truth_path = private_dir / task.truth_filename
    else:
        truth_path = private_dir / "test_target.parquet"
        if not truth_path.is_file():
            truth_path = private_dir / "test_target.csv"
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
        if len(truth) < sample_size:
            raise ValueError(
                f"private challenge has {len(truth)} rows, needs {sample_size}"
            )
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
        measurements[name] = {
            str(key): float(value) for key, value in measured["metrics"].items()
        }
    return measurements


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
    reference_strategy = source / (
        "reference_solution" if (source / "reference_solution").is_dir() else "solution"
    )
    naive_strategy = source / task.naive
    validate_committed_model_manifest(reference_strategy, role="reference")
    validate_committed_model_manifest(naive_strategy, role="naive")

    # Reference inference. ML metadata-mode tasks default "solution" to
    # reference_solution; native tasks retain solution/solve.sh.
    reference_dir = (
        "reference_solution" if (source / "reference_solution").is_dir() else "solution"
    )
    reference_cache = (run_dir / "calibration" / "reference-cache").resolve()
    reference_problem = replace(
        problem,
        reference=replace(problem.reference, cache_dir=str(reference_cache)),
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

    naive_workspace = run_dir / "calibration" / "naive-workspace"
    naive_workspace.mkdir(parents=True, exist_ok=True)
    naive_cache = (run_dir / "calibration" / "naive-cache").resolve()
    naive_problem = replace(
        problem,
        reference=replace(problem.reference, cache_dir=str(naive_cache)),
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

    degenerate_measurements = _measure_degenerate_family(problem, task, run_dir)

    input_digests = calibration_input_digests(problem, task)
    cache_key = calibration_cache_key(problem, task)
    lock = task.build_lock(
        reference_metrics=reference_measurement["metrics"],
        naive_metrics=naive_measurement["metrics"],
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
            str(key): float(value)
            for key, value in reference_measurement["metrics"].items()
        },
        naive_metrics={
            str(key): float(value)
            for key, value in naive_measurement["metrics"].items()
        },
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
