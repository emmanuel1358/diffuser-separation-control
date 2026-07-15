from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignerr_plugin.proof import (
    update_build_proof_result,
    verify_build_proof,
    write_build_proof,
)


def _make_task(problem_dir: Path) -> None:
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "scorer").mkdir(parents=True)
    (problem_dir / "environment").mkdir(parents=True)
    (problem_dir / "task.toml").write_text("[task]\nname = 'demo'\n")
    (problem_dir / "solution" / "solve.sh").write_text("echo solve\n")
    (problem_dir / "scorer" / "compute_score.py").write_text("score = 1.0\n")
    (problem_dir / "environment" / "Dockerfile").write_text("FROM base\n")
    (problem_dir / "README.md").write_text("# Demo\n")
    (problem_dir / "instruction.md").write_text("Do the thing.\n")


def _write_proof(problem_dir: Path) -> Path:
    return write_build_proof(
        problem_dir,
        image_digest="sha256:image",
        base_image_ref="lbx-tasks-base:runtime",
        platform="linux/amd64",
        alignerr_cli_version="0.1.0",
        duration_seconds=1.0,
    )


def test_update_build_proof_result_records_harness_score(tmp_path: Path) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "task.toml").write_text("[task]\nname = 'demo'\n")

    proof_path = write_build_proof(
        problem_dir,
        image_digest="sha256:image",
        base_image_ref="lbx-tasks-base:runtime",
        platform="linux/amd64",
        alignerr_cli_version="0.1.0",
        duration_seconds=1.25,
    )

    updated = update_build_proof_result(
        problem_dir,
        runtime="deepagents",
        grade_payload={
            "score": 0.290909,
            "subscores": {"criterion": 0.0},
            "weights": {"criterion": 1.0},
            "structured_subscores": [
                {"name": "criterion", "score": 0.0, "weight": 1.0}
            ],
        },
        run_dir=Path(".harness-runs/demo"),
        reward_path=Path(".harness-runs/demo/verifier/reward.json"),
        details_path=Path(".harness-runs/demo/verifier/reward-details.json"),
        rubric_quality={
            "status": "completed",
            "checks": [{"check": "Measurability", "status": "pass"}],
        },
    )

    assert updated == proof_path
    proof = json.loads(proof_path.read_text())
    assert proof["harness_result"]["runtime"] == "deepagents"
    assert proof["harness_result"]["score"] == 0.290909
    assert proof["harness_result"]["subscores"] == {"criterion": 0.0}
    assert proof["harness_result"]["weights"] == {"criterion": 1.0}
    assert proof["harness_result"]["structured_subscores"] == [
        {"name": "criterion", "score": 0.0, "weight": 1.0}
    ]
    assert proof["harness_result"]["rubric_quality"] == {
        "status": "completed",
        "checks": [{"check": "Measurability", "status": "pass"}],
    }


def test_update_build_proof_result_records_trivial_baseline_score(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "task.toml").write_text("[task]\nname = 'demo'\n")

    proof_path = write_build_proof(
        problem_dir,
        image_digest="sha256:image",
        base_image_ref="lbx-tasks-base:runtime",
        platform="linux/amd64",
        alignerr_cli_version="0.1.0",
        duration_seconds=1.25,
    )

    update_build_proof_result(
        problem_dir,
        runtime="solution",
        grade_payload={"score": 0.5},
        run_dir=Path(".harness-runs/demo"),
        reward_path=Path(".harness-runs/demo/verifier/reward.json"),
        details_path=Path(".harness-runs/demo/verifier/reward-details.json"),
        result_key="ground_truth_result",
        trivial_baseline_score=0.0,
    )

    proof = json.loads(proof_path.read_text())
    assert proof["ground_truth_result"]["score"] == 0.5
    assert proof["ground_truth_result"]["trivial_baseline_score"] == 0.0

    # Omitting the parameter keeps the field out of the recorded result.
    update_build_proof_result(
        problem_dir,
        runtime="solution",
        grade_payload={"score": 0.5},
        run_dir=Path(".harness-runs/demo"),
        reward_path=Path(".harness-runs/demo/verifier/reward.json"),
        details_path=Path(".harness-runs/demo/verifier/reward-details.json"),
        result_key="ground_truth_result",
    )
    proof = json.loads(proof_path.read_text())
    assert "trivial_baseline_score" not in proof["ground_truth_result"]


def test_doc_only_edit_does_not_stale_build_proof(tmp_path: Path) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    problem_dir.mkdir(parents=True)
    _make_task(problem_dir)
    _write_proof(problem_dir)

    # Edit prose that never reaches the scorer or the built image.
    (problem_dir / "README.md").write_text("# Demo (reworded)\n")
    (problem_dir / "instruction.md").write_text("Do the thing, described anew.\n")

    passed, errors, _ = verify_build_proof(problem_dir)
    assert passed, errors


@pytest.mark.parametrize(
    "rel,content",
    [
        ("task.toml", "[task]\nname = 'demo2'\n"),
        ("solution/solve.sh", "echo changed\n"),
        ("scorer/compute_score.py", "score = 0.5\n"),
        ("data_generation/gen.py", "x = 1\n"),
        ("environment/Dockerfile", "FROM other\n"),
    ],
)
def test_grading_input_edit_stales_build_proof(
    tmp_path: Path, rel: str, content: str
) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    problem_dir.mkdir(parents=True)
    _make_task(problem_dir)
    _write_proof(problem_dir)

    target = problem_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)

    passed, errors, _ = verify_build_proof(problem_dir)
    assert not passed
    assert any("stale" in error for error in errors)


def test_ground_truth_result_survives_doc_edit_but_drops_on_code_edit(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    problem_dir.mkdir(parents=True)
    _make_task(problem_dir)
    proof_path = _write_proof(problem_dir)

    proof = json.loads(proof_path.read_text())
    proof["ground_truth_result"] = {"score": 1.0}
    proof_path.write_text(json.dumps(proof) + "\n")

    # Doc-only edit + identical rebuild (same image/base) preserves the result.
    (problem_dir / "README.md").write_text("# changed\n")
    _write_proof(problem_dir)
    assert json.loads(proof_path.read_text()).get("ground_truth_result") == {
        "score": 1.0
    }

    # A scorer edit changes the grading hash, so the stale result is dropped.
    (problem_dir / "scorer" / "compute_score.py").write_text("score = 0.0\n")
    _write_proof(problem_dir)
    assert "ground_truth_result" not in json.loads(proof_path.read_text())
