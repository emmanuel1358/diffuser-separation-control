"""Unit tests for the continuous-ML committed-model hard contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from alignerr_plugin.ml_model_contract import (
    validate_committed_model_manifest,
    validate_ml_strategy_contract,
    validate_problem_ml_model_contracts,
)
from alignerr_plugin.validators.task.validator import TaskValidator
from grading.evaluation import (
    AnchorRationale,
    ContinuousTask,
    FloorAnchor,
    GeneratedCalibration,
    SRETarget,
    write_calibration_lock_atomic,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_strategy(
    root: Path,
    *,
    role: str,
    training_data: Path,
    inference_source: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.py").write_text("print('train')\n", encoding="utf-8")
    model = root / "model.json"
    model.write_text(json.dumps({"role": role}) + "\n", encoding="utf-8")
    if inference_source is None:
        inference_source = (
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
    (root / "solution.py").write_text(inference_source, encoding="utf-8")
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
        + "\n",
        encoding="utf-8",
    )


def _write_mlenvs_problem(
    tmp_path: Path, *, inference_source: str | None = None
) -> Path:
    task_dir = tmp_path / "demo_taiga"
    (task_dir / "data" / "public").mkdir(parents=True)
    (task_dir / "data" / "private").mkdir(parents=True)
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "ml_task_type": "dataset",
                "required_resources": "12vcpu+100gib+h100/2",
                "domain": "scientific_discovery_computational_science",
                "license": "CC0-1.0",
                "license_source": "https://creativecommons.org/publicdomain/zero/1.0/",
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "prompt.md").write_text("x" * 250, encoding="utf-8")
    (task_dir / "test_file.py").write_text(
        "\n".join(
            [
                "from grading.evaluation import AnchorRationale, ContinuousTask, FloorAnchor, GeneratedCalibration, SRETarget",
                "FLOOR = FloorAnchor(1.0, AnchorRationale('theoretical', 'Population-standardized RMSE has a no-skill value of one.'))",
                "TASK = ContinuousTask.calibrated(",
                "  targets=[SRETarget.lower('y', weight=1.0, floor=FLOOR)],",
                "  calibration=GeneratedCalibration(),",
                "  naive='baselines/naive',",
                ")",
                "def compute_score():",
                "    return TASK.grade(None, None)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    train_csv = task_dir / "data" / "public" / "train.csv"
    train_csv.write_text("x,y\n1,2\n", encoding="utf-8")
    _write_valid_strategy(
        task_dir / "reference_solution",
        role="reference",
        training_data=train_csv,
        inference_source=inference_source,
    )
    _write_valid_strategy(
        task_dir / "baselines" / "naive",
        role="naive",
        training_data=train_csv,
        inference_source=inference_source,
    )
    task = ContinuousTask.calibrated(
        targets=[
            SRETarget.lower(
                "y",
                weight=1.0,
                floor=FloorAnchor(
                    1.0,
                    AnchorRationale(
                        "theoretical",
                        "Population-standardized RMSE has a no-skill value of one.",
                    ),
                ),
            )
        ],
        calibration=GeneratedCalibration(),
    )
    lock = task.build_lock(
        reference_metrics={"y": 0.2},
        naive_metrics={"y": 0.98},
        degenerate_metrics={"constant": {"y": 1.0}},
        input_digests={"fixture": "digest"},
    )
    write_calibration_lock_atomic(task_dir / "calibration.lock.json", lock)
    evidence = task_dir / ".alignerr" / "calibration.evidence.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
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
        + "\n",
        encoding="utf-8",
    )
    return task_dir


def test_valid_strategy_contract_passes(tmp_path: Path) -> None:
    training = tmp_path / "train.csv"
    training.write_text("x,y\n1,1\n", encoding="utf-8")
    strategy = tmp_path / "reference_solution"
    _write_valid_strategy(strategy, role="reference", training_data=training)
    (tmp_path / "metadata.json").write_text("{}\n", encoding="utf-8")

    contract = validate_ml_strategy_contract(strategy, role="reference")
    assert contract.inference_entrypoint == "solution.py"
    assert contract.training_entrypoint == "train.py"


def test_manifest_rejects_digest_mismatch(tmp_path: Path) -> None:
    training = tmp_path / "train.csv"
    training.write_text("x,y\n1,1\n", encoding="utf-8")
    strategy = tmp_path / "reference_solution"
    _write_valid_strategy(strategy, role="reference", training_data=training)
    (tmp_path / "metadata.json").write_text("{}\n", encoding="utf-8")
    (strategy / "model.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        validate_committed_model_manifest(strategy, role="reference")


def test_missing_manifest_fails(tmp_path: Path) -> None:
    training = tmp_path / "train.csv"
    training.write_text("x,y\n1,1\n", encoding="utf-8")
    strategy = tmp_path / "reference_solution"
    _write_valid_strategy(strategy, role="reference", training_data=training)
    (tmp_path / "metadata.json").write_text("{}\n", encoding="utf-8")
    (strategy / "model.manifest.json").unlink()

    with pytest.raises(ValueError, match="missing model.manifest.json"):
        validate_ml_strategy_contract(strategy, role="reference")


def test_train_in_solution_fails(tmp_path: Path) -> None:
    training = tmp_path / "train.csv"
    training.write_text("x,y\n1,1\n", encoding="utf-8")
    strategy = tmp_path / "reference_solution"
    bad_inference = (
        "from pathlib import Path\n"
        "import json\n"
        "def main():\n"
        "    model = {'w': [1, 2, 3]}\n"
        "    for epoch in range(10):\n"
        "        model['w'] = [x + 1 for x in model['w']]\n"
        "    Path('model.json').write_text(json.dumps(model))\n"
        "    model.fit  # noqa: intentional training marker below\n"
        "if __name__ == '__main__':\n"
        "    from sklearn.linear_model import LogisticRegression\n"
        "    LogisticRegression().fit([[0], [1]], [0, 1])\n"
    )
    _write_valid_strategy(
        strategy,
        role="reference",
        training_data=training,
        inference_source=bad_inference,
    )
    (tmp_path / "metadata.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="looks like a training"):
        validate_ml_strategy_contract(strategy, role="reference")


def test_validator_stage_rejects_train_in_solution(tmp_path: Path) -> None:
    bad_inference = (
        "from sklearn.linear_model import LogisticRegression\n"
        "def main():\n"
        "    LogisticRegression().fit([[0], [1]], [0, 1])\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    task_dir = _write_mlenvs_problem(tmp_path, inference_source=bad_inference)
    result = TaskValidator()._continuous_ml_model_contract(task_dir)
    assert result.passed is False
    assert any("training" in issue.lower() for issue in result.issues)


def test_validator_stage_rejects_missing_manifest(tmp_path: Path) -> None:
    task_dir = _write_mlenvs_problem(tmp_path)
    (task_dir / "reference_solution" / "model.manifest.json").unlink()
    result = TaskValidator()._continuous_ml_model_contract(task_dir)
    assert result.passed is False
    assert any("manifest" in issue.lower() for issue in result.issues)


def test_validator_accepts_valid_contract(tmp_path: Path) -> None:
    task_dir = _write_mlenvs_problem(tmp_path)
    result = TaskValidator()._continuous_ml_model_contract(task_dir)
    assert result.passed is True
    assert result.issues == []
    contracts = validate_problem_ml_model_contracts(task_dir)
    assert {c.role for c in contracts} == {"reference", "naive"}


def test_example_tabular_passes_contract() -> None:
    example = (
        Path(__file__).resolve().parents[2] / "examples" / "mle-tabular-classification"
    )
    if not (example / "reference_solution").is_dir():
        pytest.skip("canonical mle-tabular-classification example not in this checkout")
    contracts = validate_problem_ml_model_contracts(
        example, naive_rel="baselines/naive"
    )
    assert {c.role for c in contracts} == {"reference", "naive"}
    stage = TaskValidator()._continuous_ml_model_contract(example)
    assert stage.passed is True, stage.issues


def test_pytorch_train_mode_toggle_is_allowed(tmp_path: Path) -> None:
    training = tmp_path / "train.csv"
    training.write_text("x,y\n1,1\n", encoding="utf-8")
    strategy = tmp_path / "reference_solution"
    inference = (
        "from pathlib import Path\n"
        "import torch\n"
        "MODEL = Path(__file__).with_name('model.json')\n"
        "def main():\n"
        "    net = torch.nn.Linear(1, 1)\n"
        "    net.train(False)\n"
        "    net.eval()\n"
        "    _ = MODEL.read_text()\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    _write_valid_strategy(
        strategy, role="reference", training_data=training, inference_source=inference
    )
    (tmp_path / "metadata.json").write_text("{}\n", encoding="utf-8")
    validate_ml_strategy_contract(strategy, role="reference")


def test_train_csv_reference_is_not_training_entrypoint(tmp_path: Path) -> None:
    training = tmp_path / "train.csv"
    training.write_text("x,y\n1,1\n", encoding="utf-8")
    strategy = tmp_path / "reference_solution"
    inference = (
        "from pathlib import Path\n"
        "import shutil\n"
        "MODEL = Path(__file__).with_name('model.json')\n"
        "DATA = Path(__file__).resolve().parents[1] / 'train.csv'\n"
        "def main():\n"
        "    assert DATA.name == 'train.csv'\n"
        "    out = Path('/tmp/output')\n"
        "    out.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copy2(MODEL, out / 'model.json')\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    _write_valid_strategy(
        strategy, role="reference", training_data=training, inference_source=inference
    )
    (tmp_path / "metadata.json").write_text("{}\n", encoding="utf-8")
    validate_ml_strategy_contract(strategy, role="reference")


def test_shell_comment_mentioning_train_py_is_allowed(tmp_path: Path) -> None:
    training = tmp_path / "train.csv"
    training.write_text("x,y\n1,1\n", encoding="utf-8")
    strategy = tmp_path / "reference_solution"
    _write_valid_strategy(strategy, role="reference", training_data=training)
    (tmp_path / "metadata.json").write_text("{}\n", encoding="utf-8")
    (strategy / "solve.sh").write_text(
        "#!/bin/bash\n"
        "# Authors: run train.py offline to refresh weights.\n"
        "cp model.json /tmp/output/model.json\n",
        encoding="utf-8",
    )
    manifest = json.loads((strategy / "model.manifest.json").read_text())
    manifest["inference_entrypoint"] = "solve.sh"
    (strategy / "model.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    validate_ml_strategy_contract(strategy, role="reference")


def test_shell_invoking_train_py_is_rejected(tmp_path: Path) -> None:
    training = tmp_path / "train.csv"
    training.write_text("x,y\n1,1\n", encoding="utf-8")
    strategy = tmp_path / "reference_solution"
    _write_valid_strategy(strategy, role="reference", training_data=training)
    (tmp_path / "metadata.json").write_text("{}\n", encoding="utf-8")
    (strategy / "solve.sh").write_text(
        "#!/bin/bash\npython train.py\ncp model.json /tmp/output/model.json\n",
        encoding="utf-8",
    )
    manifest = json.loads((strategy / "model.manifest.json").read_text())
    manifest["inference_entrypoint"] = "solve.sh"
    (strategy / "model.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must not invoke"):
        validate_ml_strategy_contract(strategy, role="reference")


def test_open_write_to_committed_model_is_rejected(tmp_path: Path) -> None:
    training = tmp_path / "train.csv"
    training.write_text("x,y\n1,1\n", encoding="utf-8")
    strategy = tmp_path / "reference_solution"
    inference = (
        "from pathlib import Path\n"
        "def main():\n"
        "    with open('model.json', 'w') as handle:\n"
        "        handle.write('{}')\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    _write_valid_strategy(
        strategy, role="reference", training_data=training, inference_source=inference
    )
    (tmp_path / "metadata.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite committed model artifact"):
        validate_ml_strategy_contract(strategy, role="reference")
