from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alignerr_plugin.capabilities import (
    project_harbor_task_data,
    resolve_capabilities,
)
from alignerr_plugin.schemas import (
    BinaryArtifact,
    FileArtifact,
    PathSetArtifact,
    ServiceArtifact,
    TaskToml,
    TreeArtifact,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "docs" / "frontierbench_capabilities.json"
DOCS_PATH = ROOT / "docs" / "FRONTIERBENCH_CAPABILITIES.md"
FIXTURE_PATH = (
    ROOT / "harness" / "tests" / "fixtures" / "frontierbench_native_conformance.toml"
)
GENERATOR_PATH = ROOT / "scripts" / "generate_frontierbench_capabilities.py"
EXPECTED_TASK_COUNT = 74
EXPECTED_SLUGS_SHA256 = (
    "d90cdc3a869482daa0d6d04655da2da1b9a33b8817944addb5c6dfb1e76b44d5"
)


def _matrix() -> dict[str, Any]:
    return json.loads(MATRIX_PATH.read_text())


def _tasks_by_slug(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["slug"]: task for task in matrix["tasks"]}


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_frontierbench_capabilities",
        GENERATOR_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_has_exactly_the_pinned_74_frontier_slugs_once() -> None:
    matrix = _matrix()
    slugs = [task["slug"] for task in matrix["tasks"]]

    assert matrix["source"]["task_count"] == EXPECTED_TASK_COUNT
    assert len(slugs) == EXPECTED_TASK_COUNT
    assert slugs == sorted(slugs)
    assert len(slugs) == len(set(slugs))
    assert (
        hashlib.sha256("".join(f"{slug}\n" for slug in slugs).encode()).hexdigest()
        == EXPECTED_SLUGS_SHA256
        == matrix["source"]["slug_set_sha256"]
    )
    assert re.fullmatch(r"[0-9a-f]{40}", matrix["source"]["git_revision"])
    assert matrix["source"]["parsed_entrypoints"] == {
        "task_toml": 74,
        "compose": 12,
        "tests_test_sh": 74,
    }
    assert all("xfoil" not in slug.lower() for slug in slugs)


def test_representative_frontier_capabilities_are_measured() -> None:
    tasks = _tasks_by_slug(_matrix())

    wal = tasks["wal-recovery-ordering"]
    assert wal["artifact_contract"]["observed_shapes"] == {"tree": 1}
    assert wal["services"]["present"] is False
    assert "determinism.explicit" in wal["evaluation"]["patterns"]
    assert "security.anti_cheat" in wal["evaluation"]["patterns"]

    live = tasks["live-database-cutover"]
    assert live["services"]["service_count"] == 5
    assert set(live["services"]["services"]) == {
        "customer",
        "main",
        "mysql-db",
        "postgres-db",
        "redis",
    }
    assert live["collect_hooks"]["count"] == 5
    assert live["collect_hooks"]["atomic_publish_count"] >= 3
    assert live["artifact_contract"]["service_scoped_count"] == 4
    assert live["artifact_contract"]["sidecar_scoped_count"] == 3
    assert live["resources"]["verifier"]["explicit_resources"] is True

    medical = tasks["medical-claims-processing"]
    assert medical["services"]["service_count"] == 3
    assert medical["services"]["named_volumes"] == ["medical-shared"]
    assert medical["mcp"]["servers"] == [
        {
            "name": "playwright",
            "port": 3080,
            "service_host": "playwright-mcp",
            "transport": "sse",
        }
    ]
    assert medical["browser"]["required"] is True
    assert medical["artifact_contract"]["sidecar_scoped_count"] == 1

    jax = tasks["jax-speedrun-gpu"]
    assert jax["resources"]["agent"]["gpus"] == 1
    assert jax["resources"]["agent"]["gpu_types"] == ["H100"]
    assert jax["resources"]["verifier"]["gpus"] == 1
    assert jax["resources"]["verifier"]["gpu_types"] == ["H100"]
    assert jax["resources"]["verifier"]["explicit_resources"] is True

    assert "xfoil" not in tasks


def test_every_observed_field_and_pattern_has_a_mapping() -> None:
    matrix = _matrix()
    inventory = matrix["observed_inventory"]
    known = set(matrix["mapping_catalog"])

    assert inventory["unmapped_fields"] == []
    assert inventory["unmapped_patterns"] == []
    for group in ("task_toml_fields", "compose_fields", "test_patterns"):
        assert inventory[group]
        for row in inventory[group]:
            assert row["mapping"], row
            assert set(row["mapping"]) <= known
    for task in matrix["tasks"]:
        assert task["native_primitives"]
        assert set(task["native_primitives"]) <= known
        for artifact in task["artifact_contract"]["items"]:
            assert artifact["native_candidates"]
            assert set(artifact["native_candidates"]) <= known


def test_native_conformance_fixture_covers_schema_and_export() -> None:
    matrix = _matrix()
    data = tomllib.loads(FIXTURE_PATH.read_text())
    declared = set(data["metadata"]["frontierbench_conformance"]["primitive_ids"])
    required = set(matrix["required_native_primitives"])

    assert declared == required
    task = TaskToml.model_validate(data)
    assert {type(artifact) for artifact in task.artifacts} == {
        BinaryArtifact,
        FileArtifact,
        PathSetArtifact,
        ServiceArtifact,
        TreeArtifact,
    }
    assert task.workspace is not None
    assert task.workspace.checkpoint_restore is True
    assert {service.role for service in task.services} == {
        "init",
        "main",
        "sidecar",
        "verifier",
    }
    browser = next(service for service in task.services if service.name == "browser")
    assert browser.capabilities == ["SYS_PTRACE"]
    assert browser.shm_mb == 1024
    assert task.captures[0].atomic_destination == "/tmp/database.dump"
    assert task.mcp_servers[0].transport == "sse"
    assert {gate.kind for gate in task.gates} == {"behavioral", "determinism"}
    assert {report.format for report in task.reports} == {"ctrf", "json"}
    assert task.result is not None
    assert task.result.reports == ["tests", "diagnostics"]

    capabilities = resolve_capabilities(task)
    projected = project_harbor_task_data(data, capabilities)
    assert projected["schema_version"] == "1.4"
    assert projected["environment"]["mcp_servers"][0]["name"] == "browser"
    assert projected["verifier"]["environment_mode"] == "separate"
    assert projected["verifier"]["collect"][0]["service"] == "database"
    assert len(projected["artifacts"]) == 6
    assert {service.role for service in capabilities.services} == {
        "agent",
        "init",
        "sidecar",
        "verifier",
    }


def test_required_primitives_have_schema_export_and_runtime_fixtures() -> None:
    matrix = _matrix()
    required = set(matrix["required_native_primitives"])
    catalog = matrix["mapping_catalog"]

    assert required
    assert all(
        catalog[primitive]["kind"] == "native_primitive"
        and catalog[primitive]["conformance_required"] is True
        for primitive in required
    )
    assert set(matrix["conformance_fixtures"]) == {
        "schema",
        "export",
        "runtime",
    }
    for layer, fixture in matrix["conformance_fixtures"].items():
        assert set(fixture["covers"]) == required, layer
        assert (ROOT / fixture["fixture"]).is_file()
        test_path, test_name = fixture["test"].split("::", 1)
        test_source = (ROOT / test_path).read_text()
        assert f"def {test_name}(" in test_source


def test_matrix_contains_capability_facts_not_source_values() -> None:
    matrix = _matrix()
    assert matrix["privacy"] == {
        "credential_stores_read": False,
        "environment_or_command_values_serialized": False,
        "evidence_content": "metadata_and_capability_facts_only",
        "hidden_tests_or_fixtures_read": False,
        "solutions_read": False,
        "task_instructions_read": False,
    }
    for task in matrix["tasks"]:
        assert "description" not in task["metadata"]
        assert "authors" not in task["metadata"]
        assert "command" not in task["collect_hooks"]
        for service in task["services"]["environment_variable_names"].values():
            assert all("=" not in name for name in service)
        for server in task["mcp"]["servers"]:
            assert "url" not in server


def test_committed_outputs_match_source_when_checkout_is_available() -> None:
    source = ROOT.parent / "frontier-bench"
    if not (source / "tasks").is_dir():
        pytest.skip("sibling Frontier-Bench checkout is unavailable")

    generator = _generator()
    matrix = generator.build_matrix(source)

    assert generator._stable_json(matrix) == MATRIX_PATH.read_text()
    assert generator.render_docs(matrix) == DOCS_PATH.read_text()
