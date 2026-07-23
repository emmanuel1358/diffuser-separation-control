from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from alignerr_plugin.proof import update_build_proof_result, write_build_proof
from alignerr_plugin.utils import grading_inputs_sha256
from lbx_rl_tasks_harness import calibration
from lbx_rl_tasks_harness.models import GroundTruthSpec, HarnessProblem, ReferenceSpec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_strategy(root: Path, *, role: str, training_data: Path) -> None:
    root.mkdir(parents=True)
    (root / "train.py").write_text("print('train')\n")
    (root / "solution.py").write_text("print('infer')\n")
    (root / "model.json").write_text(json.dumps({"role": role}) + "\n")
    (root / "model.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "role": role,
                "training_entrypoint": "train.py",
                "inference_entrypoint": "solution.py",
                "seed": 7,
                "public_training_data": {
                    "path": str(training_data.relative_to(root, walk_up=True)),
                    "sha256": _sha(training_data),
                },
                "artifacts": [
                    {"path": "model.json", "sha256": _sha(root / "model.json")}
                ],
            },
            indent=2,
        )
        + "\n"
    )


def _problem(tmp_path: Path) -> HarnessProblem:
    source = tmp_path / "problem"
    source.mkdir(parents=True)
    (source / "metadata.json").write_text("{}\n")
    (source / "data" / "public").mkdir(parents=True)
    (source / "data" / "private").mkdir(parents=True)
    (source / "data" / "public" / "train.csv").write_text(
        "x,y,value,label\n1,1,0.1,0\n2,2,0.3,0\n3,3,0.5,1\n4,4,0.7,1\n"
    )
    (source / "data" / "private" / "test_target.csv").write_text("value,label\n0,0\n")
    training_data = source / "data" / "public" / "train.csv"
    _write_strategy(
        source / "reference_solution",
        role="reference",
        training_data=training_data,
    )
    _write_strategy(
        source / "baselines" / "naive",
        role="naive",
        training_data=training_data,
    )
    (source / "test_file.py").write_text(
        "\n".join(
            [
                "from grading.evaluation import AnchorRationale, BinaryF1Target, ContinuousTask, FloorAnchor, GeneratedCalibration, SRETarget",
                "LOW = FloorAnchor(0.0, AnchorRationale('metric_bound', 'Binary F1 is bounded below by zero.'))",
                "HIGH = FloorAnchor(1.0, AnchorRationale('theoretical', 'Population-standardized RMSE has a no-skill value of one.'))",
                "TASK = ContinuousTask.calibrated(",
                "  targets=[SRETarget.lower('value', weight=.5, floor=HIGH), BinaryF1Target.higher('label', weight=.5, floor=LOW)],",
                "  calibration=GeneratedCalibration(),",
                ")",
                "def measure_submission(): return {'value': .2, 'label': .9}",
                "def compute_score(): return TASK.grade(None, None)",
                "",
            ]
        )
    )
    proof_dir = source / ".alignerr"
    proof_dir.mkdir()
    (proof_dir / "build_proof.json").write_text(
        json.dumps(
            {
                "image_digest": "sha256:image",
                "base_image_ref": "base:tag",
            }
        )
        + "\n"
    )
    return HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[],
        source_problem_dir=source,
        grader_dir=source,
        private_dir=source / "data" / "private",
        ground_truth=GroundTruthSpec(
            continuous_score_epsilon=0.05,
            zero_anchor_epsilon=0.01,
        ),
        reference=ReferenceSpec(),
        metadata={
            "difficulty": {
                "task_type": "ml",
                "reward_type": "continuous_scoring_function",
            }
        },
    )


def test_model_manifest_rejects_tampered_artifact(tmp_path) -> None:
    strategy = tmp_path / "reference"
    task_root = tmp_path
    (task_root / "metadata.json").write_text("{}\n")
    training_data = task_root / "train.csv"
    training_data.write_text("x,y\n1,1\n")
    _write_strategy(strategy, role="reference", training_data=training_data)
    (strategy / "model.json").write_text("{}\n")

    with pytest.raises(ValueError, match="digest mismatch"):
        calibration.validate_committed_model_manifest(strategy, role="reference")


def test_calibration_cache_key_tracks_semantic_inputs_not_image_rebuilds(
    tmp_path: Path,
) -> None:
    problem = _problem(tmp_path)
    task = calibration.load_continuous_task(problem)
    assert task is not None
    first = calibration.calibration_cache_key(
        problem,
        task,
        framework_revision="framework-a",
    )

    proof = problem.source_problem_dir / ".alignerr" / "build_proof.json"
    payload = json.loads(proof.read_text())
    payload["image_digest"] = "sha256:rebuilt"
    payload["base_image_ref"] = "base:rebuilt"
    proof.write_text(json.dumps(payload) + "\n")
    assert (
        calibration.calibration_cache_key(
            problem,
            task,
            framework_revision="framework-a",
        )
        == first
    )

    grader = problem.source_problem_dir / "test_file.py"
    grader.write_text(grader.read_text() + "# semantic grader revision\n")
    updated_task = calibration.load_continuous_task(problem)
    assert updated_task is not None
    assert (
        calibration.calibration_cache_key(
            problem,
            updated_task,
            framework_revision="framework-a",
        )
        != first
    )


def test_ml_ground_truth_generates_lock_and_replays(monkeypatch, tmp_path) -> None:
    problem = _problem(tmp_path)
    source = problem.source_problem_dir
    assert source is not None

    def fake_run_reference(
        _problem,
        _workspace,
        _verifier,
        _transcript,
        *,
        options,
        run_dir,
    ):
        del _workspace, _verifier, _transcript, run_dir
        cache = Path(_problem.reference.cache_dir)
        if cache.exists():
            import shutil

            shutil.rmtree(cache)
        cache.mkdir(parents=True)
        (cache / "submission.csv").write_text("value,label\n0,0\n")
        assert options.no_grade is True
        return None

    def fake_measure(_problem, workspace, _output, _transcript):
        del _problem, _output, _transcript
        if workspace.name == "reference":
            metrics = {"value": 0.2, "label": 0.9}
        elif workspace.name.startswith("degenerate-"):
            # No-info baseline is worse than (or equal to) the author floor,
            # so it must not move the floor in this fixture's assertions.
            metrics = {"value": 1.0, "label": 0.0}
        else:
            metrics = {"value": 0.98, "label": 0.0}
        return {
            "schema_version": "raw-continuous-metrics.v1",
            "task_spec_sha256": "fixture",
            "metrics": metrics,
        }

    def fake_grade(
        _problem,
        workspace,
        verifier,
        _transcript,
        *,
        calibration_lock=None,
    ):
        del _problem, _transcript
        assert calibration_lock is not None and calibration_lock.is_file()
        score = 0.0 if workspace.name == "noop" else 0.5
        verifier.mkdir(parents=True, exist_ok=True)
        (verifier / "reward.json").write_text(json.dumps({"score": score}) + "\n")
        (verifier / "reward-details.json").write_text(
            json.dumps({"score": score}) + "\n"
        )
        return score

    monkeypatch.setattr(calibration, "run_reference", fake_run_reference)
    monkeypatch.setattr(calibration, "measure_workspace_in_container", fake_measure)
    monkeypatch.setattr(calibration, "grade_workspace_in_container", fake_grade)
    run_dir = tmp_path / "run"
    workspace = run_dir / "workspace"
    verifier = run_dir / "verifier"
    workspace.mkdir(parents=True)

    result = calibration.run_ml_calibrated_ground_truth(
        problem,
        run_dir=run_dir,
        workspace=workspace,
        verifier_dir=verifier,
        transcript_path=run_dir / "transcript.txt",
    )

    assert result.score == pytest.approx(0.5)
    assert result.trivial_baseline_score == 0.0
    assert result.lock_path == source / "calibration.lock.json"
    assert result.lock_path.is_file()
    assert json.loads(result.lock_path.read_text())["task_spec_sha256"]
    assert result.evidence_path == source / ".alignerr" / "calibration.evidence.json"
    evidence = json.loads(result.evidence_path.read_text())
    assert evidence["schema_version"] == "continuous-calibration-evidence.v1"
    assert evidence["lock_sha256"] == result.lock.sha256
    assert evidence["cache_key"] == result.cache_key
    assert (workspace / "submission.csv").is_file()


def test_proof_hash_refreshes_after_generated_lock(tmp_path) -> None:
    problem = _problem(tmp_path)
    source = problem.source_problem_dir
    assert source is not None
    write_build_proof(
        source,
        image_digest="sha256:image",
        base_image_ref="base:tag",
        platform="linux/amd64",
        alignerr_cli_version="test",
        duration_seconds=0.0,
    )
    (source / "calibration.lock.json").write_text("{}\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    reward = run_dir / "reward.json"
    details = run_dir / "reward-details.json"
    reward.write_text('{"score": 0.5}\n')
    details.write_text('{"score": 0.5}\n')

    update_build_proof_result(
        source,
        runtime="solution",
        grade_payload={"score": 0.5},
        run_dir=run_dir,
        reward_path=reward,
        details_path=details,
        result_key="ground_truth_result",
        calibration={"lock_sha256": "a" * 64},
    )

    proof = json.loads((source / ".alignerr" / "build_proof.json").read_text())
    assert proof["task_dir_sha256"] == grading_inputs_sha256(source)


def test_failed_calibration_restores_committed_lock_and_proof(
    monkeypatch, tmp_path
) -> None:
    problem = _problem(tmp_path)
    source = problem.source_problem_dir
    assert source is not None
    lock_path = source / "calibration.lock.json"
    proof_path = source / ".alignerr" / "build_proof.json"
    evidence_path = source / ".alignerr" / "calibration.evidence.json"
    lock_path.write_text('{"old":"lock"}\n')
    evidence_path.write_text('{"old":"evidence"}\n')
    old_lock = lock_path.read_bytes()
    old_proof = proof_path.read_bytes()
    old_evidence = evidence_path.read_bytes()

    def fake_run_reference(*_args, **_kwargs):
        mocked_problem = _args[0]
        cache = Path(mocked_problem.reference.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "submission.csv").write_text("value,label\n0,0\n")
        proof_path.write_text('{"mutated":true}\n')

    monkeypatch.setattr(calibration, "run_reference", fake_run_reference)
    monkeypatch.setattr(
        calibration,
        "measure_workspace_in_container",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("measure failed")),
    )
    run_dir = tmp_path / "failed-run"
    (run_dir / "workspace").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="measure failed"):
        calibration.run_ml_calibrated_ground_truth(
            problem,
            run_dir=run_dir,
            workspace=run_dir / "workspace",
            verifier_dir=run_dir / "verifier",
            transcript_path=run_dir / "transcript.txt",
        )

    assert lock_path.read_bytes() == old_lock
    assert proof_path.read_bytes() == old_proof
    assert evidence_path.read_bytes() == old_evidence
