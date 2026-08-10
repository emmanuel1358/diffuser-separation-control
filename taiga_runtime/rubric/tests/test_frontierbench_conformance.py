from __future__ import annotations

import json
import tomllib
from pathlib import Path

from rubric.service_config import RuntimeOperatorRoots, load_task_service_config

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    ROOT / "harness" / "tests" / "fixtures" / "frontierbench_native_conformance.toml"
)
MATRIX_PATH = ROOT / "docs" / "frontierbench_capabilities.json"


def test_native_conformance_fixture_loads_runtime_primitives(
    tmp_path: Path,
) -> None:
    payload = tomllib.loads(FIXTURE_PATH.read_text())
    matrix = json.loads(MATRIX_PATH.read_text())
    declared = set(payload["metadata"]["frontierbench_conformance"]["primitive_ids"])

    assert declared == set(matrix["required_native_primitives"])
    roots = RuntimeOperatorRoots(
        capsule=tmp_path / "capsule",
        state=tmp_path / "state",
        sealed=tmp_path / "sealed",
    )
    config = load_task_service_config(FIXTURE_PATH, operator_roots=roots)

    assert config is not None
    assert config.main_service == "main"
    assert config.verifier_service == "verifier"
    assert config.agent_user == "agent"
    assert config.agent_workdir == "/workdir/src"
    assert config.workspace is not None
    assert config.workspace.seed == "starter"
    assert config.workspace.clean_paths == ("build", ".cache")
    assert config.workspace.checkpoint_restore is True
    assert {service.role for service in config.services} == {
        "init",
        "main",
        "sidecar",
        "verifier",
    }
    assert config.sidecar_services == (
        "database",
        "migrate",
        "browser",
        "browser-observer",
    )
    assert len(config.captures) == 1
    assert config.captures[0].service == "database"
    assert config.captures[0].atomic_destination == "/tmp/database.dump"
    assert config.captures[0].failure_policy == "infrastructure"
    assert len(config.artifacts) == 6
    assert {artifact.kind for artifact in config.artifacts} == {
        "binary",
        "file",
        "path_set",
        "service",
        "tree",
    }
    assert len(config.tools) == 1
    assert config.tools[0].name == "browser"
    assert config.tools[0].transport == "sse"
    assert config.tools[0].service == "browser"
    assert config.tools[0].readiness_kind == "http"
    assert config.verifier_result_paths == (
        "/tmp/output/grade.json",
        "/tmp/output/trace.json",
        "/tmp/output/reports/tests.json",
        "/tmp/output/reports/diagnostics.json",
    )
    assert config.primary_reward == "score"
    assert config.subscores_key == "subscores"
