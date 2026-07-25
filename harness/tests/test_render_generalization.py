"""Tests for generalizing the reviewer-render contract to non-MuJoCo tasks
(0a) and the in-container ground-truth opt-in flag (0b)."""

from __future__ import annotations

from alignerr_plugin.ground_truth import render_expected, task_type_requires_render
from lbx_rl_tasks_harness import ground_truth as gt
from lbx_rl_tasks_harness.models import GroundTruthSpec, HarnessProblem, OutputSpec


def _problem(task_type: str, render_outputs, *, in_container: bool = False):
    return HarnessProblem(
        id="t",
        source_format="problem-dir",
        prompt="",
        outputs=[OutputSpec(path="/tmp/output/control.json")],
        ground_truth=GroundTruthSpec(
            render_command="bash solution/render.sh" if render_outputs else "",
            render_outputs=render_outputs,
            in_container=in_container,
        ),
        metadata={
            "difficulty": {
                "task_type": task_type,
                "domain": "model_environment_construction"
                if task_type == "mujoco"
                else "scientific_discovery_computational_science",
                "reward_type": "multi_deterministic_rubrics",
            }
        },
    )


def test_render_expected_truth_table():
    assert task_type_requires_render("mujoco") is True
    assert task_type_requires_render("ml") is False
    # mujoco always renders; ml renders only when it declares outputs
    assert render_expected("mujoco", []) is True
    assert render_expected("ml", []) is False
    assert render_expected("ml", [OutputSpec(path="/tmp/output/rendering.mp4")]) is True


def test_validate_render_contract_skips_plain_ml():
    problem = _problem("ml", [])
    assert gt.validate_render_contract(problem) == []


def test_validate_render_contract_enforced_for_declared_ml():
    # a declared-render ml task is now held to the same contract as mujoco
    ok = _problem("ml", [OutputSpec(path="/tmp/output/rendering.mp4", required=True)])
    assert gt.validate_render_contract(ok) == []

    misnamed = _problem("ml", [OutputSpec(path="/tmp/output/wake.mp4", required=True)])
    issues = gt.validate_render_contract(misnamed)
    assert any("rendering.mp4" in i for i in issues)

    not_required = _problem(
        "ml", [OutputSpec(path="/tmp/output/rendering.mp4", required=False)]
    )
    assert any(
        "must be required" in i for i in gt.validate_render_contract(not_required)
    )


def test_in_container_flag_defaults_false_and_roundtrips():
    assert GroundTruthSpec().in_container is False
    assert (
        _problem(
            "ml", [OutputSpec(path="/tmp/output/rendering.mp4")], in_container=True
        ).ground_truth.in_container
        is True
    )
