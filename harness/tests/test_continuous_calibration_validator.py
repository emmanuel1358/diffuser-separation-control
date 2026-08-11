from __future__ import annotations

import hashlib
import json

import pytest

from alignerr_plugin.proof import PROOF_PATH
from alignerr_plugin.validators.task.validator import TaskValidator
from grading.evaluation import (
    AnchorRationale,
    BinaryF1Target,
    ContinuousTask,
    FloorAnchor,
    GeneratedCalibration,
    SRETarget,
    write_calibration_lock_atomic,
)
from grading.evaluation.lock import canonical_json_bytes, validate_calibration_lock

LOW = FloorAnchor(
    0.0, AnchorRationale("metric_bound", "Binary F1 is bounded below by zero.")
)
HIGH = FloorAnchor(
    1.0,
    AnchorRationale(
        "theoretical",
        "Population-standardized RMSE has a no-skill value of one.",
    ),
)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_strategy(root, *, role: str, training_data) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.py").write_text("print('train')\n")
    model = root / "model.json"
    model.write_text(json.dumps({"role": role}) + "\n")
    (root / "solution.py").write_text(
        "from pathlib import Path\n"
        "import shutil\n"
        "MODEL = Path(__file__).with_name('model.json')\n"
        "def main():\n"
        "    out = Path('/tmp/output')\n"
        "    out.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copy2(MODEL, out / 'model.json')\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
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
                "artifacts": [{"path": "model.json", "sha256": _sha(model)}],
            },
            indent=2,
        )
        + "\n"
    )


_TASK_TOML = """\
schema_version = "1.1"

[task]
name = "labelbox/demo-taiga"

[environment]
required_resources = "12vcpu+100gib+h100/2"

[difficulty]
task_type = "ml"
domain = "scientific_discovery_computational_science"
reward_type = "continuous_scoring_function"
license = "CC0-1.0"
license_source = "https://creativecommons.org/publicdomain/zero/1.0/"

[[outputs]]
path = "/tmp/output/model.json"
required = true
"""


def _write_v2_task(tmp_path):
    task_dir = tmp_path / "demo_taiga"
    (task_dir / "data").mkdir(parents=True)
    (task_dir / "scorer" / "data").mkdir(parents=True)
    (task_dir / "task.toml").write_text(_TASK_TOML)
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "taiga_task",
                "problem_data": {"instance_id": "demo-taiga"},
            }
        )
    )
    (task_dir / "instruction.md").write_text("x" * 250)
    source = "\n".join(
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
    (task_dir / "scorer" / "compute_score.py").write_text(source)
    training_data = task_dir / "data" / "train.csv"
    training_data.write_text("x,y\n1,2\n")
    _write_valid_strategy(
        task_dir / "solution",
        role="reference",
        training_data=training_data,
    )
    _write_valid_strategy(
        task_dir / "baselines" / "naive",
        role="naive",
        training_data=training_data,
    )

    task = ContinuousTask.calibrated(
        targets=[
            SRETarget.lower("value", weight=0.5, floor=HIGH),
            BinaryF1Target.higher("label", weight=0.5, floor=LOW),
        ],
        calibration=GeneratedCalibration(),
    )
    lock = task.build_lock(
        reference_metrics={"value": 0.2, "label": 0.9},
        naive_metrics={"value": 0.98, "label": 0.0},
        degenerate_metrics={"constant": {"value": 1.0, "label": 0.0}},
        input_digests={"fixture": "digest"},
    )
    write_calibration_lock_atomic(task_dir / "calibration.lock.json", lock)
    proof_path = task_dir / PROOF_PATH
    proof_path.parent.mkdir()
    proof_path.parent.joinpath("calibration.evidence.json").write_text(
        json.dumps(
            {
                "schema_version": "continuous-calibration-evidence.v1",
                "cache_key": "c" * 64,
                "lock_sha256": lock.sha256,
                "task_spec_sha256": task.spec_sha256,
                "evaluation_plan_sha256": task.evaluation_plan.sha256,
                "security_tier": task.security_tier,
                "inputs": lock.payload["inputs"],
                "qualification": lock.payload["qualification"],
            }
        )
        + "\n"
    )
    proof_path.write_text(
        json.dumps(
            {
                "ground_truth_result": {
                    "calibration": {
                        "lock_sha256": lock.sha256,
                        "task_spec_sha256": task.spec_sha256,
                        "evaluation_plan_sha256": task.evaluation_plan.sha256,
                        "inputs": lock.payload["inputs"],
                    }
                }
            }
        )
        + "\n"
    )
    return task_dir


def test_v2_calibration_stage_accepts_generated_lock_and_evidence(tmp_path) -> None:
    task_dir = _write_v2_task(tmp_path)

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is True
    assert result.issues == []


def test_calibration_stage_accepts_paired_legacy_lock_and_evidence(tmp_path) -> None:
    task_dir = _write_v2_task(tmp_path)
    task = ContinuousTask.calibrated(
        targets=[
            SRETarget.lower("value", weight=0.5, floor=HIGH),
            BinaryF1Target.higher("label", weight=0.5, floor=LOW),
        ],
        calibration=GeneratedCalibration(),
    )
    assert task.legacy_spec_sha256 is not None
    assert task.legacy_evaluation_plan is not None

    lock_path = task_dir / "calibration.lock.json"
    payload = json.loads(lock_path.read_text())
    payload["schema_version"] = "3.0"
    payload["task_spec_sha256"] = task.legacy_spec_sha256
    payload["evaluation_plan"] = task.legacy_evaluation_plan.to_dict()
    payload["evaluation_plan_sha256"] = task.legacy_evaluation_plan.sha256
    for field in (
        "qualification_naive_score",
        "runtime_naive_quality_score",
        "naive_semantic_gap",
        "max_unacknowledged_naive_score_gap",
        "naive_semantic_gap_acknowledgement",
    ):
        payload["qualification"].pop(field)
    legacy_lock = validate_calibration_lock(
        payload,
        task_spec_sha256=task.spec_sha256,
        compatible_task_spec_sha256s=(task.legacy_spec_sha256,),
    )
    lock_path.write_bytes(canonical_json_bytes(legacy_lock.payload))

    evidence_path = task_dir / ".alignerr" / "calibration.evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence.update(
        lock_sha256=legacy_lock.sha256,
        task_spec_sha256=task.legacy_spec_sha256,
        evaluation_plan_sha256=task.legacy_evaluation_plan.sha256,
        qualification=legacy_lock.payload["qualification"],
    )
    evidence_path.write_text(json.dumps(evidence) + "\n")

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is True
    assert result.issues == []
    assert not any("stale" in warning for warning in result.warnings)


def test_calibration_stage_accepts_trusted_ci_bundle(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir = _write_v2_task(tmp_path)
    trusted_dir = tmp_path / "trusted-calibration"
    trusted_dir.mkdir()
    local_lock = task_dir / "calibration.lock.json"
    local_evidence = task_dir / ".alignerr" / "calibration.evidence.json"
    trusted_dir.joinpath("calibration.lock.json").write_bytes(local_lock.read_bytes())
    trusted_dir.joinpath("calibration.evidence.json").write_bytes(
        local_evidence.read_bytes()
    )
    local_lock.unlink()
    local_evidence.unlink()
    monkeypatch.setenv("LBX_TRUSTED_CALIBRATION_DIR", str(trusted_dir))

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is True
    assert result.issues == []


def test_calibration_stage_hard_fails_missing_trusted_ci_bundle(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir = _write_v2_task(tmp_path)
    trusted_dir = tmp_path / "missing-trusted-calibration"
    trusted_dir.mkdir()
    monkeypatch.setenv("LBX_TRUSTED_CALIBRATION_DIR", str(trusted_dir))

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is False
    assert any("trusted-CI continuous calibration" in issue for issue in result.issues)


def test_calibration_stage_requires_bundle_in_trusted_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir = _write_v2_task(tmp_path)
    monkeypatch.setenv("LBX_REQUIRE_TRUSTED_CONTINUOUS_EVALUATION", "1")

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is False
    assert any("did not stage an authoritative" in issue for issue in result.issues)


def test_calibration_stage_warns_on_missing_development_evidence(tmp_path) -> None:
    task_dir = _write_v2_task(tmp_path)
    (task_dir / ".alignerr" / "calibration.evidence.json").unlink()

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is True
    assert any("calibration.evidence.json" in warning for warning in result.warnings)


def test_calibration_stage_allows_missing_local_cache(tmp_path) -> None:
    task_dir = _write_v2_task(tmp_path)
    (task_dir / "calibration.lock.json").unlink()
    (task_dir / ".alignerr" / "calibration.evidence.json").unlink()

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is True
    assert any("Trusted CI regenerates" in warning for warning in result.warnings)


def test_v2_calibration_stage_rejects_committed_score_artifacts(tmp_path) -> None:
    task_dir = _write_v2_task(tmp_path)
    (task_dir / "solution" / "results.txt").write_text("0.5\n")

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is False
    assert any("must not commit" in issue for issue in result.issues)


def test_v2_calibration_stage_rejects_noncanonical_lock(tmp_path) -> None:
    task_dir = _write_v2_task(tmp_path)
    lock_path = task_dir / "calibration.lock.json"
    payload = json.loads(lock_path.read_text())
    lock_path.write_text(json.dumps(payload) + "\n")

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is False
    assert any("canonically serialized" in issue for issue in result.issues)


def test_v2_calibration_stage_rejects_missing_floor_rationale(tmp_path) -> None:
    task_dir = _write_v2_task(tmp_path)
    lock_path = task_dir / "calibration.lock.json"
    payload = json.loads(lock_path.read_text())
    payload["targets"]["value"]["floor"]["rationale"]["summary"] = "baseline"
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is False
    assert any("invalid floor rationale" in issue for issue in result.issues)


def test_continuous_stage_rejects_scalar_only_production_bypass(tmp_path) -> None:
    task_dir = _write_v2_task(tmp_path)
    grader = task_dir / "scorer" / "compute_score.py"
    grader.write_text(
        grader.read_text().replace(
            "return TASK.grade(None, None)",
            "return TASK.score(measure_submission())",
        )
    )

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is False
    assert any("calibration-only" in issue for issue in result.issues)


def test_continuous_stage_hard_blocks_legacy_grader_without_task(tmp_path) -> None:
    task_dir = _write_v2_task(tmp_path)
    (task_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n    return 0.25\n"
    )

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is False
    assert any("legacy unprotected graders" in issue for issue in result.issues)


_DOMAINS = {
    "ml": "scientific_discovery_computational_science",
    "mujoco": "locomotion",
    "cfd": "aerodynamics",
    "structures": "structural_mechanics",
}


def _write_native_task(
    tmp_path,
    *,
    task_type: str,
    reward_type: str,
):
    task_dir = tmp_path / f"{task_type}-{reward_type}"
    scorer = task_dir / "scorer"
    scorer.mkdir(parents=True)
    ml_fields = (
        '\nlicense = "self_generated"\n' 'license_source = "synthetic test fixture"\n'
        if task_type == "ml"
        else ""
    )
    (task_dir / "task.toml").write_text(f"""
schema_version = "1.1"
[task]
name = "labelbox/{task_type}-fixture"
[environment]
required_resources = "2vcpu+6gib"
[difficulty]
task_type = "{task_type}"
domain = "{_DOMAINS[task_type]}"
reward_type = "{reward_type}"
{ml_fields}
""")
    (scorer / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n" "    return 0.25\n"
    )
    return task_dir


@pytest.mark.parametrize("task_type", sorted(_DOMAINS))
def test_continuous_stage_blocks_every_supported_task_type(tmp_path, task_type) -> None:
    task_dir = _write_native_task(
        tmp_path,
        task_type=task_type,
        reward_type="continuous_scoring_function",
    )

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is False
    assert any("legacy unprotected graders" in issue for issue in result.issues)


@pytest.mark.parametrize("task_type", sorted(_DOMAINS))
def test_non_continuous_reward_types_are_unchanged(tmp_path, task_type) -> None:
    task_dir = _write_native_task(
        tmp_path,
        task_type=task_type,
        reward_type="multi_deterministic_rubrics",
    )

    result = TaskValidator()._continuous_calibration(task_dir)

    assert result.passed is True
    assert result.issues == []
