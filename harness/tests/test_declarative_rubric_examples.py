from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from _fixture_guard import requires_examples

from alignerr_plugin.utils import load_task_toml
from alignerr_plugin.validators.task.validator import (
    TaskValidator,
    _declarative_rubric_api_issues,
)
from grading.evaluation import RubricTask
from grading.evaluation.plan import validate_serialized_plan

ROOT = Path(__file__).resolve().parents[2]


def _rubric_problem_dirs() -> list[Path]:
    roots = [
        ROOT / "examples",
        ROOT / "alignerr_plugin/src/alignerr_plugin/starter_templates",
    ]
    found: list[Path] = []
    for root in roots:
        for task_toml in sorted(root.glob("*/task.toml")):
            problem = task_toml.parent
            if (
                load_task_toml(problem).difficulty.reward_type
                == "multi_deterministic_rubrics"
            ):
                found.append(problem)
    return found


RUBRIC_PROBLEMS = _rubric_problem_dirs()


def _load_task(problem: Path) -> RubricTask:
    scorer = problem / "scorer" / "compute_score.py"
    module_name = "rubric_" + problem.name.replace("-", "_")
    sys.path.insert(0, str(scorer.parent))
    try:
        spec = importlib.util.spec_from_file_location(module_name, scorer)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scorer.parent))
    assert isinstance(module.TASK, RubricTask)
    assert not hasattr(module, "compute_score")
    return module.TASK


@pytest.mark.parametrize(
    "problem",
    RUBRIC_PROBLEMS,
    ids=lambda problem: problem.name,
)
def test_all_shipped_rubrics_use_current_declarative_protocol(problem: Path) -> None:
    task = _load_task(problem)
    plan_path = problem / "scorer" / "evaluation.plan.json"
    payload = json.loads(plan_path.read_text())

    assert validate_serialized_plan(payload) == task.evaluation_plan.sha256
    assert payload["task_spec_sha256"] == task.spec_sha256

    validator = TaskValidator()
    import_stage = validator._grader_import(problem)
    protocol_stage = validator._rubric_protocol(problem)
    assert import_stage.passed, import_stage.issues
    assert protocol_stage.passed, protocol_stage.issues


@requires_examples("opensees-base-isolation", "openfoam-hydrofoil-flap")
def test_canonical_json_rubrics_reject_reported_crash_vectors(tmp_path: Path) -> None:
    vectors = {
        "opensees-base-isolation": (
            b'{"isolation_system":{"Qd_kip":'
            + str(10**400).encode()
            + b',"Kd_kip_per_in":20,"Dy_in":0.6}}'
        ),
        "openfoam-hydrofoil-flap": (
            b'{"flap_deflection_deg":'
            + str(10**400).encode()
            + b',"hinge_gap_m":0.01,"flap_chord_fraction":0.2,'
            b'"blend_radius_m":0.01}'
        ),
    }
    from grading import AgentFault

    for name, payload in vectors.items():
        task = _load_task(ROOT / "examples" / name)
        for content in (payload, b"\xff\xfe{}", b"[" * 60_000 + b"]" * 60_000):
            workspace = tmp_path / name
            workspace.mkdir(exist_ok=True)
            artifact = workspace / task.artifact.path
            artifact.write_bytes(content)
            with pytest.raises(AgentFault):
                task.artifact.load(workspace)


def test_declarative_rubric_lint_blocks_task_owned_hardening() -> None:
    source = """
import json
import subprocess

def compute_score(workspace, trajectory, private):
    return 0.0

def evaluate(context):
    raw = json.loads((context.workspace / "design.json").read_text())
    subprocess.run(["solver"])
    return {"ratio": context.candidate["a"] / context.candidate["b"]}
"""
    issues = _declarative_rubric_api_issues("scorer/compute_score.py", source)

    assert any("must not define compute_score" in issue for issue in issues)
    assert any("artifact descriptor" in issue for issue in issues)
    assert any("context.run_solver" in issue for issue in issues)
    assert any("context.ratio" in issue for issue in issues)
