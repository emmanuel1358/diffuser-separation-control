from __future__ import annotations

import json
from pathlib import Path

from _fixture_guard import requires_examples
from alignerr_plugin.exporters.harbor import export_harbor
from alignerr_plugin.exporters.taiga import build_job_payload, export_taiga
from lbx_rl_tasks_harness.formats.harbor import load_harbor_dir
from lbx_rl_tasks_harness.formats.problem_dir import load_problem_dir
from lbx_rl_tasks_harness.formats.taiga import load_taiga_metadata
from lbx_rl_tasks_harness.prompts import task_type_hint
from lbx_rl_tasks_harness.runtimes.deepagents import _extra_fields_for_mcp

ROOT = Path(__file__).resolve().parents[2]
MUJOCO = ROOT / "examples" / "mujoco-pendulum"
OPENSEES = ROOT / "examples" / "opensees-base-isolation"
OPENFOAM = ROOT / "examples" / "openfoam-hydrofoil-flap"
TABULAR = ROOT / "examples" / "mle-tabular-classification"


@requires_examples("mujoco-pendulum")
def test_load_problem_dir() -> None:
    problem = load_problem_dir(MUJOCO)
    assert problem.id == "mujoco-pendulum"
    assert problem.source_format == "problem-dir"
    assert problem.outputs[0].path == "/tmp/output/model.xml"
    assert problem.grader_dir == MUJOCO / "scorer"
    assert problem.required_resources == "4vcpu+16gib"
    assert problem.taiga_problem is not None
    assert (
        problem.taiga_problem["startup_command"]
        == "/opt/lbx-runtime/.venv/bin/rubric mcp"
    )
    fields = _extra_fields_for_mcp(problem, "local:test")
    assert "task_prompt" in fields
    assert "test_file" in fields


@requires_examples("opensees-base-isolation")
def test_load_problem_dir_uses_structures_hint_not_prompt_prefix() -> None:
    problem = load_problem_dir(OPENSEES)
    fields = _extra_fields_for_mcp(problem, "local:test")

    assert not problem.prompt.startswith("## OpenSees Availability")
    assert fields["task_prompt"] == problem.prompt
    assert problem.taiga_problem is not None
    assert problem.taiga_problem["hints"][0] == {
        "message": task_type_hint("structures").strip(),
        "enabled": True,
    }


@requires_examples("opensees-base-isolation")
def test_export_taiga_uses_structures_hint_not_prompt_prefix(tmp_path: Path) -> None:
    metadata = tmp_path / "problems-metadata.json"
    export_taiga(OPENSEES, metadata, image_ref="local:test")

    data = json.loads(metadata.read_text())
    problem = data["problem_set"]["problems"][0]

    assert not problem["task_prompt"].startswith("## OpenSees Availability")
    assert problem["hints"][0]["message"].startswith("## OpenSees Availability")
    assert problem["hints"][0]["enabled"] is True


@requires_examples("openfoam-hydrofoil-flap")
def test_load_problem_dir_uses_cfd_hint_not_prompt_prefix() -> None:
    problem = load_problem_dir(OPENFOAM)
    fields = _extra_fields_for_mcp(problem, "local:test")

    assert not problem.prompt.startswith("## OpenFOAM Availability")
    assert fields["task_prompt"] == problem.prompt
    assert problem.taiga_problem is not None
    assert problem.taiga_problem["hints"][0]["message"].startswith(
        "## OpenFOAM Availability"
    )
    assert problem.taiga_problem["hints"][0]["enabled"] is True


@requires_examples("openfoam-hydrofoil-flap")
def test_export_taiga_uses_cfd_hint_not_prompt_prefix(tmp_path: Path) -> None:
    metadata = tmp_path / "problems-metadata.json"
    export_taiga(OPENFOAM, metadata, image_ref="local:test")

    data = json.loads(metadata.read_text())
    problem = data["problem_set"]["problems"][0]

    assert not problem["task_prompt"].startswith("## OpenFOAM Availability")
    assert problem["hints"][0]["message"].startswith("## OpenFOAM Availability")
    assert problem["hints"][0]["enabled"] is True


@requires_examples("mujoco-pendulum")
def test_load_problem_dir_does_not_add_solver_hint_for_mujoco() -> None:
    problem = load_problem_dir(MUJOCO)

    assert problem.taiga_problem is not None
    assert "hints" not in problem.taiga_problem


@requires_examples("mujoco-pendulum")
def test_load_taiga_metadata_with_source_problem(tmp_path: Path) -> None:
    metadata = tmp_path / "problems-metadata.json"
    export_taiga(MUJOCO, metadata, image_ref="local:test")

    problem = load_taiga_metadata(metadata, MUJOCO)

    assert problem.source_format == "taiga"
    assert problem.id == "mujoco-pendulum"
    assert problem.image == "local:test"
    assert problem.grader_dir == MUJOCO / "scorer"
    assert problem.required_resources == "4vcpu+16gib"
    assert problem.metadata["difficulty"]["task_type"] == "mujoco"
    assert problem.metadata["difficulty"]["reward_type"] == "multi_deterministic_rubrics"
    assert problem.metadata["runner"]["api_model_name"] == "claude-fable-5"
    assert problem.ground_truth.render_command == "bash solution/render.sh"
    assert problem.reference.execution == "auto"
    assert problem.metadata["grading_strategy"] == [{"type": "mcp", "weight": 1.0}]
    assert problem.taiga_problem is not None


@requires_examples("mujoco-pendulum")
def test_load_full_taiga_payload_recovers_standalone_runtime_contract(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "job.json"
    metadata.write_text(
        json.dumps(build_job_payload(MUJOCO, image_ref="local:test")) + "\n"
    )

    problem = load_taiga_metadata(metadata)

    assert problem.required_resources == "4vcpu+16gib"
    assert problem.metadata["difficulty"]["task_type"] == "mujoco"
    assert problem.metadata["difficulty"]["domain"] == "model_environment_construction"
    assert problem.metadata["difficulty"]["reward_type"] == "multi_deterministic_rubrics"
    assert problem.metadata["runner"]["api_model_name"] == "claude-fable-5"
    assert problem.metadata["agent"]["timeout_sec"] == 3600
    assert problem.metadata["verifier"]["timeout_sec"] == 600
    assert problem.taiga_problem["image"] == "local:test"


def test_load_taiga_metadata_recovers_exported_outputs(tmp_path: Path) -> None:
    metadata = tmp_path / "problems-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "problem_set": {
                    "owner": "labelbox",
                    "name": "test",
                    "description": "test",
                    "problems": [
                        {
                            "id": "output-fixture",
                            "image": "local:test",
                            "startup_command": "rubric mcp",
                            "required_tools": ["bash"],
                            "task_prompt": "Write /tmp/output/model.xml.",
                            "outputs": [
                                {
                                    "path": "/tmp/output/model.xml",
                                    "required": True,
                                    "description": "MJCF XML model",
                                }
                            ],
                        }
                    ],
                }
            }
        )
        + "\n"
    )

    problem = load_taiga_metadata(metadata)

    assert [output.path for output in problem.outputs] == ["/tmp/output/model.xml"]
    assert problem.outputs[0].required is True
    assert problem.outputs[0].description == "MJCF XML model"


@requires_examples("mujoco-pendulum")
def test_load_harbor_dir_with_source_problem(tmp_path: Path) -> None:
    harbor_dir = tmp_path / "harbor"
    export_harbor(MUJOCO, harbor_dir, image_ref="local:test")

    problem = load_harbor_dir(harbor_dir, MUJOCO)

    assert problem.source_format == "harbor"
    assert problem.id == "mujoco-pendulum"
    assert problem.grader_dir == MUJOCO / "scorer"
    assert problem.required_resources == "4vcpu+16gib"
    assert (harbor_dir / "tests" / "test.sh").exists()
    assert problem.taiga_problem is not None


@requires_examples("mujoco-pendulum")
def test_load_standalone_harbor_preserves_native_runtime_contract(
    tmp_path: Path,
) -> None:
    harbor_dir = tmp_path / "harbor"
    export_harbor(MUJOCO, harbor_dir)

    native = load_problem_dir(MUJOCO)
    harbor = load_harbor_dir(harbor_dir)

    assert harbor.required_resources == native.required_resources
    assert harbor.required_tools == native.required_tools
    assert harbor.metadata["agent"] == native.metadata["agent"]
    assert harbor.metadata["verifier"] == native.metadata["verifier"]
    assert harbor.metadata["environment"] == native.metadata["environment"]
    assert harbor.metadata["runner"] == native.metadata["runner"]
    assert harbor.metadata["difficulty"] == native.metadata["difficulty"]
    assert harbor.metadata["delivery"] == native.metadata["delivery"]
    assert harbor.ground_truth == native.ground_truth
    assert harbor.reference == native.reference


@requires_examples("mle-tabular-classification")
def test_load_harbor_dir_roundtrips_continuous_ml_task(tmp_path: Path) -> None:
    # A continuous ml task exports the same native envelope as every other task,
    # and load_harbor_dir must read it back without the source problem dir.
    harbor_dir = tmp_path / "harbor"
    export_harbor(TABULAR, harbor_dir)

    assert (harbor_dir / "task.toml").exists()
    assert (harbor_dir / "instruction.md").exists()

    problem = load_harbor_dir(harbor_dir)

    assert problem.source_format == "harbor"
    assert problem.prompt == (harbor_dir / "instruction.md").read_text()
    assert problem.grader_dir == harbor_dir / "environment" / "scorer"
    assert problem.private_dir == harbor_dir / "environment" / "scorer" / "data"
    assert problem.required_resources == "12vcpu+100gib+h100/2"
    assert problem.metadata["difficulty"]["reward_type"] == "continuous_scoring_function"
    assert problem.metadata["runner"]["api_model_name"] == "claude-fable-5"
    assert problem.metadata["agent"]["timeout_sec"] == 21600
    assert problem.ground_truth.score_epsilon == 1e-9
    assert problem.reference.execution == "auto"
