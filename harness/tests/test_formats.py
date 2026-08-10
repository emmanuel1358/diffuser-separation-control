from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

import pytest
import tomli_w
from _fixture_guard import requires_examples
from alignerr_plugin.exporters.harbor import export_harbor
from alignerr_plugin.exporters.taiga import build_job_payload, export_taiga
from alignerr_plugin.validators.task.creator import TaskCreator
from lbx_rl_tasks_harness.formats import problem_dir as problem_dir_format
from lbx_rl_tasks_harness.formats.harbor import load_harbor_dir
from lbx_rl_tasks_harness.formats.problem_dir import load_problem_dir
from lbx_rl_tasks_harness.formats.taiga import load_taiga_metadata
from lbx_rl_tasks_harness.prompts import task_type_hint
from lbx_rl_tasks_harness.runtimes.deepagents import _extra_fields_for_mcp
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
MUJOCO = ROOT / "examples" / "mujoco-pendulum"
OPENSEES = ROOT / "examples" / "opensees-base-isolation"
OPENFOAM = ROOT / "examples" / "openfoam-hydrofoil-flap"
TABULAR = ROOT / "examples" / "mle-tabular-classification"
WAL = ROOT / "examples" / "wal-recovery-ordering"


def _write_capability_problem(problem_dir: Path) -> Path:
    task: dict[str, Any] = {
        "schema_version": "1.2",
        "task": {
            "name": "labelbox/harbor-capability-loader",
            "description": "Exercise Harbor capability projection loading.",
        },
        "environment": {
            "required_resources": "4vcpu+16gib",
            "storage_mb": 10_000,
            "allow_internet": False,
        },
        "agent": {
            "timeout_sec": 1800,
            "user": "agent",
            "resources": {
                "cpus": 4,
                "memory_mb": 4096,
                "storage_mb": 8192,
                "gpus": 0,
                "network": "none",
            },
        },
        "verifier": {
            "timeout_sec": 600,
            "user": "root",
            "env": ["VERIFIER_TOKEN"],
            "resources": {
                "cpus": 2,
                "memory_mb": 2048,
                "storage_mb": 4096,
                "gpus": 0,
                "network": "none",
            },
        },
        "workspace": {
            "seed": "starter",
            "root": "/workdir",
            "agent_cwd": "/workdir",
        },
        "services": [
            {
                "name": "main",
                "role": "main",
                "build": {
                    "context": "environment/main",
                    "dockerfile": "Dockerfile",
                },
                "user": "agent",
            }
        ],
        "artifacts": [
            {
                "name": "result",
                "kind": "file",
                "source": "/tmp/output/result.txt",
                "destination": "result.txt",
                "service": "main",
            }
        ],
        "outputs": [
            {
                "path": "/tmp/output/result.txt",
                "required": True,
                "description": "Capability result.",
            }
        ],
        "difficulty": {
            "task_type": "software_engineering",
            "domain": "repo_debugging",
            "reward_type": "multi_deterministic_rubrics",
        },
        "metadata": {"taiga": {"image_contract": "outer_capsule"}},
    }
    problem_dir.mkdir(parents=True)
    (problem_dir / "task.toml").write_text(tomli_w.dumps(task))
    (problem_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "taiga_task",
                "problem_data": {
                    "instance_id": "harbor-capability-loader",
                    "description": "Capability loader fixture.",
                },
            }
        )
        + "\n"
    )
    (problem_dir / "instruction.md").write_text(
        "Repair the seeded repository and write /tmp/output/result.txt.\n"
    )
    main = problem_dir / "environment" / "main"
    main.mkdir(parents=True)
    (main / "Dockerfile").write_text("FROM python:3.13-slim\nWORKDIR /workdir\n")
    (problem_dir / "starter").mkdir()
    (problem_dir / "starter" / "app.py").write_text("print('seed')\n")
    scorer = problem_dir / "scorer"
    (scorer / "data").mkdir(parents=True)
    (scorer / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n"
        "    return {'score': 1.0}\n"
    )
    (problem_dir / "solution").mkdir()
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/bin/sh\nmkdir -p /tmp/output\necho ok > /tmp/output/result.txt\n"
    )
    return problem_dir


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
    assert (
        problem.metadata["difficulty"]["reward_type"] == "multi_deterministic_rubrics"
    )
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
    assert (
        problem.metadata["difficulty"]["reward_type"] == "multi_deterministic_rubrics"
    )
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
    export_harbor(
        MUJOCO,
        harbor_dir,
        image_ref=f"example.invalid/task@sha256:{'a' * 64}",
    )

    problem = load_harbor_dir(harbor_dir, MUJOCO)

    assert problem.source_format == "harbor"
    assert problem.id == "mujoco-pendulum"
    assert problem.grader_dir == MUJOCO / "scorer"
    assert problem.required_resources == "4vcpu+16gib"
    assert (harbor_dir / "tests" / "test.sh").exists()
    assert problem.metadata["verifier"]["env"] == []
    assert problem.metadata["harbor_task_toml"]["verifier"]["env"] == {}
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
    assert harbor.metadata["harbor_task_toml"]["verifier"]["env"] == {}
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
    assert (
        problem.metadata["difficulty"]["reward_type"] == "continuous_scoring_function"
    )
    assert problem.metadata["runner"]["api_model_name"] == "claude-fable-5"
    assert problem.metadata["agent"]["timeout_sec"] == 21600
    assert problem.ground_truth.score_epsilon == 1e-9
    assert problem.reference.execution == "auto"


@requires_examples("mujoco-pendulum")
def test_load_harbor_adapts_llm_and_phase_env_without_losing_raw_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "llm-source"
    shutil.copytree(MUJOCO, source)
    scorer = source / "scorer" / "compute_score.py"
    scorer.write_text(scorer.read_text() + "\n# llm_criterion\n")
    harbor_dir = tmp_path / "harbor"
    export_harbor(source, harbor_dir)

    exported = tomllib.loads((harbor_dir / "task.toml").read_text())
    exported["environment"]["docker_image"] = "local:harbor-loader"
    exported["environment"]["env"] = {"AGENT_TOKEN": "${AGENT_TOKEN}"}
    exported["verifier"]["environment"] = {"env": {"SEALED_TOKEN": "${SEALED_TOKEN}"}}
    exported["solution"] = {"env": {"ORACLE_TOKEN": "${ORACLE_TOKEN}"}}
    (harbor_dir / "task.toml").write_text(tomli_w.dumps(exported))

    problem = load_harbor_dir(harbor_dir)
    with_source = load_harbor_dir(harbor_dir, source)
    raw = problem.metadata["harbor_task_toml"]

    assert problem.image == "local:harbor-loader"
    assert problem.metadata["verifier"]["env"] == ["ANTHROPIC_API_KEY"]
    assert with_source.metadata["verifier"]["env"] == ["ANTHROPIC_API_KEY"]
    assert with_source.grader_dir == source / "scorer"
    assert raw["verifier"]["env"] == {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}
    assert raw["environment"]["env"] == {"AGENT_TOKEN": "${AGENT_TOKEN}"}
    assert raw["verifier"]["environment"]["env"] == {"SEALED_TOKEN": "${SEALED_TOKEN}"}
    assert raw["solution"]["env"] == {"ORACLE_TOKEN": "${ORACLE_TOKEN}"}


def test_load_software_starter_harbor_standalone_and_with_source(
    tmp_path: Path,
) -> None:
    source = TaskCreator().create_structure(
        tmp_path / "problems",
        {
            "name": "labelbox/software-loader-smoke",
            "template": "software-engineering",
        },
    )
    harbor_dir = tmp_path / "harbor"
    export_harbor(source, harbor_dir)

    standalone = load_harbor_dir(harbor_dir)
    with_source = load_harbor_dir(harbor_dir, source)

    assert standalone.id == "software-loader-smoke"
    assert standalone.grader_dir == harbor_dir / "environment" / "scorer"
    assert standalone.metadata["verifier"]["env"] == []
    assert standalone.metadata["harbor_task_toml"]["verifier"]["env"] == {}
    assert with_source.id == standalone.id
    assert with_source.grader_dir == source / "scorer"
    assert with_source.taiga_problem is not None


def test_problem_dir_and_harbor_loading_do_not_eagerly_project_taiga(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = TaskCreator().create_structure(
        tmp_path / "problems",
        {
            "name": "labelbox/lazy-taiga-loader",
            "template": "software-engineering",
        },
    )
    harbor_dir = tmp_path / "harbor"
    export_harbor(source, harbor_dir)
    calls = 0

    def unsupported_projection(*_args: object, **_kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise ValueError("capability projection needs a trusted outer capsule")

    monkeypatch.setattr(
        problem_dir_format,
        "build_job_payload",
        unsupported_projection,
    )

    native = load_problem_dir(source)
    harbor = load_harbor_dir(harbor_dir, source)

    assert native.id == "lazy-taiga-loader"
    assert harbor.id == native.id
    assert calls == 0
    assert native.taiga_problem is not None
    with pytest.raises(ValueError, match="Taiga metadata projection is unavailable"):
        dict(native.taiga_problem)
    assert calls == 1


def test_capability_problem_loads_before_trusted_taiga_projection(
    tmp_path: Path,
) -> None:
    source = _write_capability_problem(tmp_path / "capability")

    problem = load_problem_dir(source)

    assert problem.id == "harbor-capability-loader"
    assert problem.metadata["difficulty"]["task_type"] == "software_engineering"
    assert problem.taiga_problem is not None
    taiga_problem = dict(problem.taiga_problem)
    assert taiga_problem["image"] == "LOCAL_IMAGE"
    assert (
        taiga_problem["extra_fields"]["task_metadata"]["capability_summary"][
            "runtime"
        ]["outer_image_contract"]
        == "outer_capsule"
    )


@requires_examples("wal-recovery-ordering")
def test_load_wal_harbor_standalone_and_with_source(tmp_path: Path) -> None:
    harbor_dir = tmp_path / "harbor"
    export_harbor(WAL, harbor_dir)

    standalone = load_harbor_dir(harbor_dir)
    with_source = load_harbor_dir(harbor_dir, WAL)

    assert standalone.id == "wal-recovery-ordering"
    assert standalone.outputs[0].path == "/tmp/output/repo"
    assert standalone.metadata["verifier"]["env"] == []
    assert with_source.id == standalone.id
    assert with_source.grader_dir == WAL / "scorer"
    assert with_source.taiga_problem is not None


def test_load_capability_harbor_maps_resources_and_preserves_projection(
    tmp_path: Path,
) -> None:
    source = _write_capability_problem(tmp_path / "problem")
    harbor_dir = tmp_path / "harbor"
    export_harbor(source, harbor_dir)

    standalone = load_harbor_dir(harbor_dir)
    with_source = load_harbor_dir(harbor_dir, source)
    raw = standalone.metadata["harbor_task_toml"]

    assert standalone.id == "harbor-capability-loader"
    assert standalone.required_resources == "4vcpu+16gib"
    assert standalone.metadata["agent"]["resources"]["cpus"] == 4
    assert standalone.metadata["agent"]["resources"]["network"] == "none"
    assert standalone.metadata["agent"]["resources"]["storage_mb"] == 10_000
    assert standalone.metadata["environment"]["storage_mb"] == 10_000
    assert standalone.metadata["verifier"]["resources"]["memory_mb"] == 2048
    assert standalone.metadata["verifier"]["env"] == ["VERIFIER_TOKEN"]
    assert raw["environment"]["memory_mb"] == 4096
    assert raw["environment"]["network_mode"] == "no-network"
    assert raw["verifier"]["environment"]["memory_mb"] == 2048
    assert raw["verifier"]["env"] == {"VERIFIER_TOKEN": "${VERIFIER_TOKEN}"}
    assert raw["artifacts"][0]["source"] == "/tmp/output/result.txt"
    assert "name" not in raw["artifacts"][0]
    assert with_source.id == standalone.id
    assert with_source.grader_dir == source / "scorer"
    assert with_source.taiga_problem is not None


def test_load_harbor_keeps_native_validation_strict(tmp_path: Path) -> None:
    source = TaskCreator().create_structure(
        tmp_path / "problems",
        {
            "name": "labelbox/strict-loader-smoke",
            "template": "software-engineering",
        },
    )
    harbor_dir = tmp_path / "harbor"
    export_harbor(source, harbor_dir)
    exported = tomllib.loads((harbor_dir / "task.toml").read_text())
    exported["environment"]["unexpected_runtime_field"] = "invalid"
    (harbor_dir / "task.toml").write_text(tomli_w.dumps(exported))

    with pytest.raises(ValidationError, match="unexpected_runtime_field"):
        load_harbor_dir(harbor_dir)


def test_load_optional_harbor_task_config_export(tmp_path: Path) -> None:
    config_module = pytest.importorskip("harbor.models.task.config")
    sources = [
        (
            TaskCreator().create_structure(
                tmp_path / "problems",
                {
                    "name": "labelbox/harbor-018-loader-smoke",
                    "template": "software-engineering",
                },
            ),
            "harbor-018-loader-smoke",
        ),
        (
            _write_capability_problem(tmp_path / "capability"),
            "harbor-capability-loader",
        ),
    ]

    for index, (source, expected_id) in enumerate(sources):
        harbor_dir = tmp_path / f"harbor-{index}"
        export_harbor(source, harbor_dir)
        config_module.TaskConfig.model_validate_toml(
            (harbor_dir / "task.toml").read_text()
        )
        assert load_harbor_dir(harbor_dir).id == expected_id
