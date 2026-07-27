from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from alignerr_plugin.proof import update_build_proof_result, write_build_proof
from alignerr_plugin.utils import grading_inputs_sha256
from grading.evaluation import (
    GeneratedCalibration,
    WorkspaceDegenerateProbes,
    WorkspaceProbe,
)
from lbx_rl_tasks_harness import calibration
from lbx_rl_tasks_harness.models import GroundTruthSpec, HarnessProblem, ReferenceSpec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_strategy(root: Path, *, role: str, training_data: Path) -> None:
    root.mkdir(parents=True)
    (root / "train.py").write_text("print('train')\n")
    (root / "solution.py").write_text(
        "from pathlib import Path\n"
        "import shutil\n"
        "MODEL = Path(__file__).with_name('model.json')\n"
        "print('infer')\n"
        "out = Path('/tmp/output')\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "shutil.copy2(MODEL, out / 'model.json')\n"
    )
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
    (source / "task.toml").write_text("[task]\nname = \"labelbox/demo\"\n")
    (source / "data").mkdir(parents=True)
    (source / "scorer" / "data").mkdir(parents=True)
    (source / "data" / "train.csv").write_text(
        "x,y,value,label\n1,1,0.1,0\n2,2,0.3,0\n3,3,0.5,1\n4,4,0.7,1\n"
    )
    (source / "scorer" / "data" / "test_target.csv").write_text("value,label\n0,0\n")
    training_data = source / "data" / "train.csv"
    _write_strategy(
        source / "solution",
        role="reference",
        training_data=training_data,
    )
    _write_strategy(
        source / "baselines" / "naive",
        role="naive",
        training_data=training_data,
    )
    (source / "scorer" / "compute_score.py").write_text(
        "\n".join(
            [
                (
                    "from grading.evaluation import AnchorRationale, "
                    "BinaryF1Target, ContinuousTask, FloorAnchor, "
                    "GeneratedCalibration, SRETarget"
                ),
                "LOW = FloorAnchor(0.0, AnchorRationale('metric_bound', 'Binary F1 is bounded below by zero.'))",
                (
                    "HIGH = FloorAnchor(1.0, AnchorRationale('theoretical', "
                    "'Population-standardized RMSE has a no-skill value of one.'))"
                ),
                "TASK = ContinuousTask.calibrated(",
                (
                    "  targets=[SRETarget.lower('value', weight=.5, floor=HIGH), "
                    "BinaryF1Target.higher('label', weight=.5, floor=LOW)],"
                ),
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
        private_dir=source / "scorer" / "data",
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

    grader = problem.source_problem_dir / "scorer" / "compute_score.py"
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
    registered_task = calibration.load_continuous_task(problem)
    assert registered_task is not None
    source = problem.source_problem_dir
    assert source is not None
    invoked_entrypoints: list[str] = []

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
        entrypoint = _problem.reference.entrypoint or "solution.py"
        invoked_entrypoints.append(f"{options.solution_dir}/{entrypoint}")
        assert "train" not in Path(entrypoint).name.lower()
        assert options.solution_dir.endswith(
            ("solution", "baselines/naive", "naive")
        ) or options.solution_dir in {"solution", "baselines/naive"}
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
            "task_spec_sha256": registered_task.spec_sha256,
            "calibration_seed": 0,
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
    assert invoked_entrypoints
    assert all(path.endswith("solution.py") for path in invoked_entrypoints)
    assert not any("train.py" in path for path in invoked_entrypoints)


def test_ml_ground_truth_rejects_train_in_solution(tmp_path) -> None:
    from alignerr_plugin.ml_model_contract import validate_ml_strategy_contract

    problem = _problem(tmp_path)
    source = problem.source_problem_dir
    assert source is not None
    (source / "solution" / "solution.py").write_text(
        "from sklearn.linear_model import LogisticRegression\n"
        "LogisticRegression().fit([[0], [1]], [0, 1])\n"
    )

    with pytest.raises(ValueError, match="looks like a training"):
        validate_ml_strategy_contract(source / "solution", role="reference")


def test_tier_b_submission_csv_participates_in_strategy_digest(tmp_path) -> None:
    problem = _problem(tmp_path)
    task = calibration.load_continuous_task(problem)
    assert task is not None
    before = calibration.calibration_input_digests(problem, task)["reference_strategy"]
    (problem.source_problem_dir / "solution" / "submission.csv").write_text(
        "value,label\n0,0\n"
    )
    after = calibration.calibration_input_digests(problem, task)["reference_strategy"]
    assert before != after


def _task_with_workspace_probes(problem: HarnessProblem):
    task = calibration.load_continuous_task(problem)
    assert task is not None
    provider = WorkspaceDegenerateProbes(
        probes=[
            WorkspaceProbe(
                name="no-op",
                path="baselines/degenerate/no-op",
                rationale="A committed policy that always emits the neutral action.",
            ),
            WorkspaceProbe(
                name="seeded-random",
                path="baselines/degenerate/seeded-random",
                rationale="A committed seeded policy with no learned state or signal.",
            ),
        ]
    )
    return replace(
        task,
        calibration=GeneratedCalibration(degenerate_probes=provider),
    )


def test_workspace_degenerate_probes_are_measured_and_digest_bound(
    monkeypatch, tmp_path
) -> None:
    problem = _problem(tmp_path)
    source = problem.source_problem_dir
    assert source is not None
    for name, action in (("no-op", 0), ("seeded-random", 1)):
        workspace = source / "baselines" / "degenerate" / name
        workspace.mkdir(parents=True)
        (workspace / "policy.py").write_text(f"ACTION = {action}\n", encoding="utf-8")
    task = _task_with_workspace_probes(problem)
    calls: list[str] = []

    def fake_measure(_problem, workspace, _output, _transcript):
        del _problem, _output, _transcript
        calls.append(workspace.name)
        action = int(
            (workspace / "policy.py").read_text(encoding="utf-8").split("=")[1]
        )
        return {
            "schema_version": "raw-continuous-metrics.v1",
            "task_spec_sha256": task.spec_sha256,
            "calibration_seed": 0,
            "metrics": {
                "value": 1.0,
                "label": float(action),
            },
        }

    monkeypatch.setattr(calibration, "measure_workspace_in_container", fake_measure)
    measured = calibration._measure_degenerate_family(problem, task, tmp_path / "run")

    assert measured == {
        "no-op": {"value": 1.0, "label": 0.0},
        "seeded-random": {"value": 1.0, "label": 1.0},
    }
    assert len(calls) == 4
    inputs = calibration.calibration_input_digests(problem, task)
    assert set(key for key in inputs if key.startswith("degenerate_probe:")) == {
        "degenerate_probe:no-op",
        "degenerate_probe:seeded-random",
    }
    before = inputs["degenerate_probe:no-op"]
    (source / "baselines" / "degenerate" / "no-op" / "policy.py").write_text(
        "ACTION = 2\n", encoding="utf-8"
    )
    after = calibration.calibration_input_digests(problem, task)[
        "degenerate_probe:no-op"
    ]
    assert before != after


def test_workspace_degenerate_probe_rejects_nondeterminism(
    monkeypatch, tmp_path
) -> None:
    problem = _problem(tmp_path)
    source = problem.source_problem_dir
    assert source is not None
    for name in ("no-op", "seeded-random"):
        workspace = source / "baselines" / "degenerate" / name
        workspace.mkdir(parents=True)
        (workspace / "policy.py").write_text("ACTION = 0\n", encoding="utf-8")
    # Keep workspace digests distinct so nondeterminism is the first failure.
    (source / "baselines" / "degenerate" / "seeded-random" / "seed.txt").write_text(
        "7\n", encoding="utf-8"
    )
    task = _task_with_workspace_probes(problem)
    count = 0

    def fake_measure(_problem, workspace, _output, _transcript):
        nonlocal count
        del _problem, workspace, _output, _transcript
        count += 1
        return {
            "schema_version": "raw-continuous-metrics.v1",
            "task_spec_sha256": task.spec_sha256,
            "calibration_seed": 0,
            "metrics": {"value": 1.0 + count * 0.01, "label": 0.0},
        }

    monkeypatch.setattr(calibration, "measure_workspace_in_container", fake_measure)
    with pytest.raises(RuntimeError, match="nondeterministic"):
        calibration._measure_degenerate_family(problem, task, tmp_path / "run")


def test_workspace_degenerate_probe_rejects_symlink(tmp_path) -> None:
    problem = _problem(tmp_path)
    source = problem.source_problem_dir
    assert source is not None
    for name in ("no-op", "seeded-random"):
        workspace = source / "baselines" / "degenerate" / name
        workspace.mkdir(parents=True)
        (workspace / "policy.py").write_text("ACTION = 0\n", encoding="utf-8")
    (source / "baselines" / "degenerate" / "no-op" / "leak").symlink_to(
        source / "scorer" / "data" / "test_target.csv"
    )
    task = _task_with_workspace_probes(problem)

    with pytest.raises(ValueError, match="non-regular"):
        calibration._measure_degenerate_family(problem, task, tmp_path / "run")


def test_workspace_degenerate_probe_rejects_symlinked_root(tmp_path) -> None:
    problem = _problem(tmp_path)
    source = problem.source_problem_dir
    assert source is not None
    external = tmp_path / "external-probe"
    external.mkdir()
    (external / "policy.py").write_text("ACTION = 0\n", encoding="utf-8")
    degenerate = source / "baselines" / "degenerate"
    degenerate.mkdir(parents=True)
    (degenerate / "no-op").symlink_to(external, target_is_directory=True)
    seeded = degenerate / "seeded-random"
    seeded.mkdir()
    (seeded / "policy.py").write_text("ACTION = 1\n", encoding="utf-8")
    task = _task_with_workspace_probes(problem)

    with pytest.raises(ValueError, match="real directory"):
        calibration.calibration_input_digests(problem, task)


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
