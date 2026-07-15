from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from alignerr_plugin.schemas import EnvironmentSection
from alignerr_plugin.validators.task.validator import TaskValidator

_TASK_TOML = """
schema_version = "1.1"
[task]
name = "labelbox/hidden-env-test"
[environment]
required_resources = "12vcpu+100gib+h100/2"
hidden_env = "{mode}"
[difficulty]
task_type = "ml"
domain = "reinforcement_learning_agentic_systems"
reward_type = "continuous_scoring_function"
license = "self_generated"
license_source = "fully synthetic env generated in-repo; no external dataset"
[[outputs]]
path = "/tmp/output/policy.py"
required = true
description = "policy"
"""


def _make_task(tmp_path: Path, *, mode: str = "env", env_py: bool = True,
               client: bool = True, leak_env: bool = False) -> Path:
    d = tmp_path / "task"
    (d / "scorer" / "data").mkdir(parents=True, exist_ok=True)
    (d / "data").mkdir(parents=True, exist_ok=True)
    (d / "task.toml").write_text(_TASK_TOML.format(mode=mode))
    (d / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n    return 0.0\n"
    )
    if env_py:
        (d / "scorer" / "data" / "env.py").write_text(
            "def make_env(**k):\n    return object()\n"
        )
    if client:
        (d / "data" / "env_client.py").write_text("# client\n")
    if leak_env:
        (d / "data" / "env.py").write_text("def make_env(**k):\n    return 1\n")
    return d


# ---- schema ----


_RESOURCE = "12vcpu+100gib+h100/2"


@pytest.mark.parametrize("mode", ["", "env", "hybrid"])
def test_schema_accepts_valid_modes(mode: str) -> None:
    assert (
        EnvironmentSection(required_resources=_RESOURCE, hidden_env=mode).hidden_env
        == mode
    )


def test_schema_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError, match="hidden_env"):
        EnvironmentSection(required_resources=_RESOURCE, hidden_env="bogus")


def test_schema_normalizes_case() -> None:
    assert (
        EnvironmentSection(required_resources=_RESOURCE, hidden_env="ENV").hidden_env
        == "env"
    )


# ---- validator stage ----


def test_validator_passes_for_well_formed_env_task(tmp_path: Path) -> None:
    stage = TaskValidator()._hidden_env(_make_task(tmp_path))
    assert stage.passed, stage.issues


def test_validator_noop_for_static_task(tmp_path: Path) -> None:
    stage = TaskValidator()._hidden_env(_make_task(tmp_path, mode=""))
    assert stage.passed and stage.issues == []


def test_validator_fails_without_env_module(tmp_path: Path) -> None:
    stage = TaskValidator()._hidden_env(_make_task(tmp_path, env_py=False))
    assert not stage.passed
    assert any("env module" in i for i in stage.issues)


def test_validator_fails_without_client(tmp_path: Path) -> None:
    stage = TaskValidator()._hidden_env(_make_task(tmp_path, client=False))
    assert not stage.passed
    assert any("env_client.py" in i for i in stage.issues)


def test_validator_fails_when_env_source_is_public(tmp_path: Path) -> None:
    stage = TaskValidator()._hidden_env(_make_task(tmp_path, leak_env=True))
    assert not stage.passed
    assert any("agent-visible" in i for i in stage.issues)


def test_validator_fails_when_make_env_missing(tmp_path: Path) -> None:
    d = _make_task(tmp_path)
    (d / "scorer" / "data" / "env.py").write_text("X = 1\n")
    stage = TaskValidator()._hidden_env(d)
    assert not stage.passed
    assert any("make_env" in i for i in stage.issues)
