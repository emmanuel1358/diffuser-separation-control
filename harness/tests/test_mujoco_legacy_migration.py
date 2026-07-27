from __future__ import annotations

import json
import runpy
import shutil
import tomllib
from pathlib import Path

import pytest
import tomli_w
from _fixture_guard import requires_examples
from alignerr_plugin.migrations.mujoco import (
    legacy_resources_to_taiga,
    migrate_legacy_mujoco_task,
)
from alignerr_plugin.utils import load_task_toml
from alignerr_plugin.validators.task.validator import TaskValidator
from lbx_rl_tasks_harness.formats.problem_dir import load_problem_dir

ROOT = Path(__file__).resolve().parents[2]
MUJOCO = ROOT / "examples" / "mujoco-pendulum"


def _write_legacy_task(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "task.toml").write_text(
        """\
[task]
name = "labelbox/legacy-pusher"
description = "Train a contact-rich pushing policy."

[environment]
cpus = 4
memory_mb = 8192
storage_mb = 10000
gpus = 0
gpu_types = []
allow_internet = false

[policy]
protocol_version = 2
spec = "data/policy_spec.json"

[scorer]
path = "scorer/compute_score.py"

[difficulty]
task_type = "mujoco"
domain = "robotics"

[[outputs]]
path = "/tmp/output/policy.py"
required = true
"""
    )
    (root / "instruction.md").write_text("Write /tmp/output/policy.py using the public files in /data/.\n")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "taiga_task",
                "problem_data": {"instance_id": "legacy-pusher"},
            }
        )
    )
    (root / "environment").mkdir()
    (root / "environment" / "Dockerfile").write_text(
        """\
ARG PROBLEM_DIR=.
FROM lbx-tasks-base:local
RUN uv pip install --python /mcp_server/.venv/bin/python gymnasium
RUN mkdir -p /mcp_server/data /mcp_server/grader /tmp/output
COPY --chmod=0700 ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN rm -rf /mcp_server/grader/data
RUN chmod 0755 /data /mcp_server /task && chmod -R 0700 /mcp_server/data /mcp_server/grader
"""
    )
    (root / "scorer").mkdir()
    (root / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n    return {'score': 1.0}\n"
    )
    (root / "solution").mkdir()
    (root / "solution" / "solve.sh").write_text(
        "#!/bin/sh\n"
        "PYTHON_BIN=/mcp_server/.venv/bin/python\n"
        "mkdir -p /tmp/output\n"
        "touch /tmp/output/policy.py\n"
    )
    (root / "solution" / "render.sh").write_text(
        "#!/bin/sh\n/mcp_server/.venv/bin/python render.py\n"
    )


def test_legacy_resources_never_underprovision() -> None:
    assert legacy_resources_to_taiga({"cpus": 12, "memory_mb": 102400, "gpus": 0}) == "16vcpu+128gib"
    assert legacy_resources_to_taiga({"cpus": 4, "memory_mb": 8192, "gpus": 1}) == "24vcpu+200gib+h100/1"
    assert (
        legacy_resources_to_taiga(
            {"cpus": 4, "memory_mb": 8192},
            dockerfile_text="ARG BASE_IMAGE=lbx-tasks-base-gpu",
        )
        == "24vcpu+200gib+h100/1"
    )
    assert (
        legacy_resources_to_taiga(
            {"cpus": 4, "memory_mb": 8192, "gpus": 0},
            dockerfile_text="ARG BASE_IMAGE=lbx-tasks-base-gpu",
        )
        == "4vcpu+16gib"
    )
    with pytest.raises(ValueError, match="reviewed manual mapping"):
        legacy_resources_to_taiga({"cpus": 24, "memory_mb": 204800, "gpus": 0})
    with pytest.raises(ValueError, match="reviewed manual mapping"):
        legacy_resources_to_taiga({"cpus": 32, "memory_mb": 307200, "gpus": 1})


def test_migrate_native_paths_and_schema(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    output = tmp_path / "native"
    _write_legacy_task(source)

    result = migrate_legacy_mujoco_task(source, output)

    task = load_task_toml(output)
    assert result.source_layout == "native"
    assert task.environment.required_resources == "4vcpu+16gib"
    assert task.difficulty.domain == "contact_rich_manipulation"
    assert task.difficulty.reward_type == "multi_deterministic_rubrics"
    with (output / "task.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    assert "policy" not in raw
    assert "scorer" not in raw
    assert (output / "data" / ".gitkeep").exists()
    assert (output / "scorer" / "data" / ".gitkeep").exists()
    dockerfile = (output / "environment" / "Dockerfile").read_text()
    assert "chmod 0755 /data /task" in dockerfile
    assert "chmod 0755 /data /mcp_server /task" not in dockerfile
    assert "chmod 0700 /mcp_server" in dockerfile
    assert "/mcp_server/.venv" not in dockerfile
    assert "/opt/lbx-runtime/.venv/bin/python" in dockerfile
    assert "/mcp_server/.venv" not in (output / "solution" / "solve.sh").read_text()
    assert "/opt/lbx-runtime/.venv/bin/python" in (output / "solution" / "solve.sh").read_text()
    assert "/mcp_server/.venv" not in (output / "solution" / "render.sh").read_text()
    assert not result.ready_for_validation
    assert any("RubricTask" in blocker for blocker in result.blockers)


def test_migrate_native_in_place(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    _write_legacy_task(source)
    binary = source / "scorer" / "data" / "cached-policy.bin"
    binary.parent.mkdir()
    binary.write_bytes(b"\xff\x00/mcp_server/.venv\x00")

    result = migrate_legacy_mujoco_task(source, source)

    assert result.destination == source.resolve()
    assert load_task_toml(source).environment.required_resources == "4vcpu+16gib"
    assert binary.read_bytes() == b"\xff\x00/mcp_server/.venv\x00"
    assert not any("cached-policy.bin" in blocker for blocker in result.blockers)


def test_migration_preserves_current_domain_and_reward_type(tmp_path: Path) -> None:
    source = tmp_path / "current"
    _write_legacy_task(source)
    with (source / "task.toml").open("rb") as handle:
        config = tomllib.load(handle)
    config["difficulty"] = {
        "task_type": "mujoco",
        "domain": "swimming_aquatic_control",
        "reward_type": "continuous_scoring_function",
    }
    (source / "task.toml").write_text(tomli_w.dumps(config))

    result = migrate_legacy_mujoco_task(source, source)

    task = load_task_toml(source)
    assert task.difficulty.domain == "swimming_aquatic_control"
    assert task.difficulty.reward_type == "continuous_scoring_function"
    assert any("continuous_scoring_function requires" in item for item in result.blockers)


def test_migration_requires_each_sealed_evaluation_artifact(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric"
    _write_legacy_task(rubric)
    (rubric / "scorer" / "compute_score.py").write_text(
        "from grading.evaluation import RubricTask\nTASK = RubricTask(criteria=())\n"
    )

    rubric_result = migrate_legacy_mujoco_task(rubric, rubric)

    assert any("evaluation.plan.json" in item for item in rubric_result.blockers)
    (rubric / "scorer" / "evaluation.plan.json").write_text("{}\n")
    invalid_rubric_result = migrate_legacy_mujoco_task(rubric, rubric)
    assert any("file is invalid" in item for item in invalid_rubric_result.blockers)

    continuous = tmp_path / "continuous"
    _write_legacy_task(continuous)
    with (continuous / "task.toml").open("rb") as handle:
        config = tomllib.load(handle)
    config["difficulty"] = {
        "task_type": "mujoco",
        "domain": "controller_planner_authoring",
        "reward_type": "continuous_scoring_function",
    }
    (continuous / "task.toml").write_text(tomli_w.dumps(config))
    (continuous / "scorer" / "compute_score.py").write_text(
        "from grading.evaluation import ContinuousTask\nTASK = ContinuousTask()\n"
    )

    continuous_result = migrate_legacy_mujoco_task(continuous, continuous)

    assert any("calibration.lock.json" in item for item in continuous_result.blockers)
    (continuous / "calibration.lock.json").write_text("{}\n")
    invalid_continuous_result = migrate_legacy_mujoco_task(continuous, continuous)
    assert any("file is invalid" in item for item in invalid_continuous_result.blockers)

    (continuous / "scorer" / "compute_score.py").write_text(
        "from grading.evaluation import PolicyEvaluationTask\n"
        "TASK = PolicyEvaluationTask()\n"
    )

    policy_result = migrate_legacy_mujoco_task(continuous, continuous)

    assert any("PolicyEvaluationTask requires" in item for item in policy_result.blockers)


def test_migration_audit_counts_successful_taiga_export_as_pass(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "legacy"
    destination = tmp_path / "native"
    _write_legacy_task(source)

    def fake_export_taiga(_problem_dir, output_path, *, image_ref):
        assert image_ref == "local:migration-audit"
        output_path.write_text("{}\n")

    audit_namespace = runpy.run_path(
        str(ROOT / "scripts" / "audit_legacy_mujoco_migration.py")
    )
    audit_task = audit_namespace["audit_task"]
    monkeypatch.setitem(audit_task.__globals__, "export_taiga", fake_export_taiga)

    result = audit_task(source, destination)

    assert result["blockers"]
    assert result["taiga_export_probe"] == "pass"
    assert result["taiga"] == "pass"


@requires_examples("mujoco-pendulum")
def test_migrated_legacy_envelope_reaches_only_the_fresh_proof_gate(
    tmp_path: Path,
) -> None:
    task = tmp_path / "mujoco-pendulum"
    shutil.copytree(MUJOCO, task)
    with (task / "task.toml").open("rb") as handle:
        config = tomllib.load(handle)
    environment = config["environment"]
    config["environment"] = {
        "cpus": 4,
        "memory_mb": 8192,
        "storage_mb": environment["storage_mb"],
        "gpus": 0,
        "gpu_types": [],
        "allow_internet": environment["allow_internet"],
    }
    config["difficulty"] = {
        "task_type": "mujoco",
        "domain": "robotics",
    }
    config["policy"] = {
        "protocol_version": 2,
        "spec": "data/policy_spec.json",
    }
    (task / "task.toml").write_text(tomli_w.dumps(config))

    result = migrate_legacy_mujoco_task(task, task)
    problem = load_problem_dir(task)
    validation = TaskValidator().validate(task, task / ".alignerr" / "validations", ROOT)
    failed = {name for name, stage in validation.stages.items() if not stage.passed}

    assert not result.ready_for_validation
    assert not any("RubricTask" in blocker for blocker in result.blockers)
    assert any("build proof is stale" in blocker for blocker in result.blockers)
    assert problem.required_resources == "4vcpu+16gib"
    assert failed == {"local_build_proof"}


def test_migrate_harbor_to_native_paths(tmp_path: Path) -> None:
    native = tmp_path / "source-native"
    harbor = tmp_path / "harbor"
    output = tmp_path / "native-output"
    _write_legacy_task(native)
    harbor.mkdir()
    for name in ("task.toml", "instruction.md", "solution"):
        source = native / name
        destination = harbor / name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.write_bytes(source.read_bytes())
    environment = harbor / "environment"
    environment.mkdir()
    (harbor / "tests").mkdir()
    (harbor / "tests" / "test.sh").write_text(
        "/runtime/run_grader.py --workspace /app --grader-dir /mcp_server/grader\n"
    )
    shutil.copytree(native / "scorer", environment / "scorer")
    shutil.copytree(native / "environment", environment / "source_environment")
    (environment / "data").mkdir()
    (environment / "calibration.lock.json").write_text('{"schema_version": "3.0"}\n')

    result = migrate_legacy_mujoco_task(harbor, output)

    assert result.source_layout == "harbor"
    assert (output / "scorer" / "compute_score.py").is_file()
    assert (output / "environment" / "Dockerfile").is_file()
    assert (output / "metadata.json").is_file()
    assert (output / "calibration.lock.json").read_text() == '{"schema_version": "3.0"}\n'
    assert "--workspace /tmp/output" in (output / "tests" / "test.sh").read_text()
    assert load_task_toml(output).difficulty.task_type == "mujoco"
