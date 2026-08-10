from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from grader_runner.run_grader import main


def _grader_dir(example: Path, tmp_path: Path) -> Path:
    grader = tmp_path / "grader"
    grader.mkdir()
    (grader / "compute_score.py").write_text(
        (example / "scorer" / "compute_score.py").read_text(), encoding="utf-8"
    )
    return grader


def _grade(example: Path, workspace: Path, output: Path, grader: Path) -> dict:
    rc = main(
        [
            "--workspace",
            str(workspace),
            "--grader-dir",
            str(grader),
            "--private-dir",
            str(example / "scorer" / "data"),
            "--output-dir",
            str(output),
            "--timeout",
            "120",
        ]
    )
    assert rc == 0
    return json.loads((output / "reward-details.json").read_text())


def test_tabular_reference_scores_half_with_redacted_receipt(
    template_examples: Path, tmp_path, monkeypatch
) -> None:
    example = template_examples / "mle-tabular-classification"
    workspace = tmp_path / "reference"
    output = tmp_path / "reference-verifier"
    workspace.mkdir()
    monkeypatch.setenv(
        "LBX_CALIBRATION_LOCK_PATH", str(example / "calibration.lock.json")
    )
    subprocess.run(
        [sys.executable, str(example / "solution" / "solution.py")],
        env={
            **os.environ,
            "LBT_MODEL_DIR": str(example / "solution"),
            "LBT_OUTPUT_DIR": str(workspace),
        },
        check=True,
    )

    details = _grade(example, workspace, output, _grader_dir(example, tmp_path))

    assert details["score"] == pytest.approx(0.5, abs=0.025)
    receipt_text = json.dumps(details["metadata"]["evaluation"])
    assert "p_value" not in receipt_text
    assert "raw_metrics" not in json.dumps(details["metadata"])
    private_trace = json.loads((output / "evaluation-details.json").read_text())
    assert "p_value" in private_trace["targets"]["t1"]


@pytest.mark.parametrize("jitter", [0.0, 1e-8, 1e-5, 1e-3, 1e-2])
def test_tabular_constant_perturbation_ladder_scores_zero(
    template_examples: Path, tmp_path, monkeypatch, jitter
) -> None:
    example = template_examples / "mle-tabular-classification"
    workspace = tmp_path / f"jitter-{jitter}"
    output = tmp_path / f"jitter-{jitter}-verifier"
    workspace.mkdir()
    monkeypatch.setenv(
        "LBX_CALIBRATION_LOCK_PATH", str(example / "calibration.lock.json")
    )
    (workspace / "predictor.py").write_text(
        "def load_predictor():\n"
        "    class Predictor:\n"
        "        def predict(self, rows):\n"
        f"            eps = {jitter!r}\n"
        "            return {\n"
        "                't1': [0.1 + eps*i for i, _ in enumerate(rows)],\n"
        "                't2': [-0.2 + eps*((i*17)%101) for i, _ in enumerate(rows)],\n"
        "                'label': [1.0 + eps*i for i, _ in enumerate(rows)],\n"
        "            }\n"
        "    return Predictor()\n",
        encoding="utf-8",
    )

    details = _grade(example, workspace, output, _grader_dir(example, tmp_path))

    assert details["score"] == pytest.approx(0.0)
    assert all(
        not decision["accepted"]
        for decision in details["metadata"]["evaluation"]["decisions"].values()
    )
