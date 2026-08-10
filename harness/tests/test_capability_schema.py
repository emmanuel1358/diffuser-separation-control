from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

import pytest
import tomli_w
from alignerr_plugin.capabilities import (
    is_capability_task,
    project_resource_fields,
    resolve_capabilities,
)
from alignerr_plugin.schemas import (
    BinaryArtifact,
    FileArtifact,
    PathSetArtifact,
    ServiceArtifact,
    SSEMCPServer,
    StdioMCPServer,
    TaskToml,
    TreeArtifact,
)
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]


def _pinned_image(repository: str, digit: str) -> str:
    return f"example.invalid/{repository}@sha256:{digit * 64}"


def _base_task() -> dict[str, Any]:
    return {
        "schema_version": "1.2",
        "task": {
            "name": "labelbox/native-capability-test",
            "description": "Native capability graph fixture.",
        },
        "environment": {
            "required_resources": "4vcpu+16gib",
            "storage_mb": 20_000,
            "allow_internet": False,
        },
        "difficulty": {
            "task_type": "mujoco",
            "domain": "model_environment_construction",
            "reward_type": "multi_deterministic_rubrics",
        },
    }


def _minimal_service_task() -> dict[str, Any]:
    data = _base_task()
    data["services"] = [
        {
            "name": "main",
            "role": "main",
            "image": _pinned_image("main", "1"),
        }
    ]
    return data


def _assert_toml_round_trip(task: TaskToml) -> None:
    dumped = task.model_dump(mode="json", exclude_none=True)
    encoded = tomli_w.dumps(dumped)
    reparsed = TaskToml.model_validate(tomllib.loads(encoded))
    assert reparsed == task


def test_service_without_resource_override_uses_default_isolated_network() -> None:
    task = TaskToml.model_validate(_minimal_service_task())

    service = resolve_capabilities(task).services[0]

    assert service.compose["networks"] == {"alignerr-isolated": {}}


def test_shared_network_namespace_survives_offline_resource_projection() -> None:
    data = _minimal_service_task()
    data["services"].append(
        {
            "name": "observer",
            "role": "sidecar",
            "image": _pinned_image("observer", "2"),
            "network_mode": "share",
            "network_share_target": "main",
            "resources": {"network": "none"},
        }
    )
    task = TaskToml.model_validate(data)

    observer = resolve_capabilities(task).services[1]

    assert observer.compose["network_mode"] == "service:main"
    assert "networks" not in observer.compose


def test_network_aliases_reject_non_bridge_modes() -> None:
    no_network = _minimal_service_task()
    no_network["services"][0].update(
        {"network_mode": "none", "network_aliases": ["main-alias"]}
    )
    with pytest.raises(ValidationError, match="aliases require bridge"):
        TaskToml.model_validate(no_network)

    shared = _minimal_service_task()
    shared["services"].append(
        {
            "name": "observer",
            "role": "sidecar",
            "image": _pinned_image("observer", "2"),
            "network_mode": "share",
            "network_share_target": "main",
            "network_aliases": ["observer-alias"],
        }
    )
    with pytest.raises(ValidationError, match="aliases require bridge"):
        TaskToml.model_validate(shared)


@pytest.mark.parametrize("share", ("1", "2", "8"))
def test_taiga_gpu_share_projects_as_one_harbor_device(share: str) -> None:
    resources = project_resource_fields(f"12vcpu+100gib+h100/{share}")

    assert resources["gpus"] == 1
    assert resources["gpu_types"] == ["H100"]


def test_live_database_capability_graph_round_trips() -> None:
    data = _base_task()
    data.update(
        {
            "workspace": {
                "seed": "environment/api",
                "root": "/app/api",
                "agent_cwd": "/app/api",
                "init_policy": "copy",
                "git_baseline": "a" * 40,
                "clean_paths": ["__pycache__", ".venv", ".pytest_cache"],
                "checkpoint_restore": True,
            },
            "agent": {
                "timeout_sec": 7200,
                "user": "agent",
                "resources": {
                    "cpus": 16,
                    "memory_mb": 16_384,
                    "storage_mb": 20_480,
                    "gpus": 0,
                    "platform": "linux/amd64",
                    "build_timeout_sec": 1800,
                    "runtime_timeout_sec": 7200,
                    "network": "isolated",
                },
            },
            "verifier": {
                "timeout_sec": 600,
                "user": "root",
                "resources": {
                    "cpus": 16,
                    "memory_mb": 16_384,
                    "storage_mb": 40_960,
                    "gpus": 0,
                    "platform": "linux/amd64",
                    "build_timeout_sec": 1800,
                    "runtime_timeout_sec": 600,
                    "network": "isolated",
                },
            },
            "volumes": [
                {"name": "postgres-data"},
                {"name": "redis-data"},
                {"name": "snapshots"},
            ],
            "services": [
                {
                    "name": "postgres-db",
                    "role": "sidecar",
                    "image": _pinned_image("postgres", "2"),
                    "env": {
                        "POSTGRES_USER": "postgres",
                        "POSTGRES_DB": "shop",
                    },
                    "ports": [{"container_port": 5432, "name": "postgres"}],
                    "healthcheck": {
                        "command": ["CMD-SHELL", "pg_isready -U postgres -d shop"],
                        "interval_sec": 3,
                        "timeout_sec": 5,
                        "retries": 30,
                        "start_period_sec": 10,
                    },
                    "volumes": [
                        {
                            "volume": "postgres-data",
                            "target": "/var/lib/postgresql/data",
                            "mode": "rw",
                        },
                        {
                            "volume": "snapshots",
                            "target": "/snapshots",
                            "mode": "rw",
                        },
                    ],
                },
                {
                    "name": "redis",
                    "role": "sidecar",
                    "image": _pinned_image("redis", "3"),
                    "command": [
                        "redis-server",
                        "--appendonly",
                        "no",
                        "--save",
                        "",
                    ],
                    "ports": [{"container_port": 6379}],
                    "healthcheck": {
                        "command": ["CMD", "redis-cli", "ping"],
                        "interval_sec": 2,
                        "timeout_sec": 3,
                        "retries": 30,
                    },
                    "volumes": [
                        {
                            "volume": "redis-data",
                            "target": "/data",
                            "mode": "rw",
                        }
                    ],
                },
                {
                    "name": "seed-db",
                    "role": "init",
                    "build": {
                        "context": "environment/seed",
                        "dockerfile": "Dockerfile",
                        "platform": "linux/amd64",
                    },
                    "depends_on": [
                        {
                            "service": "postgres-db",
                            "condition": "service_healthy",
                        }
                    ],
                },
                {
                    "name": "main",
                    "role": "main",
                    "build": {
                        "context": "environment/api",
                        "dockerfile": "Dockerfile",
                        "args": {"APP_ENV": "benchmark"},
                    },
                    "user": "agent",
                    "env": {
                        "POSTGRES_HOST": "postgres-db",
                        "REDIS_HOST": "redis",
                    },
                    "ports": [{"container_port": 8080, "name": "api"}],
                    "resources": {
                        "cpus": 8,
                        "memory_mb": 8192,
                        "storage_mb": 10_240,
                        "network": "isolated",
                    },
                    "depends_on": [
                        {"service": "postgres-db", "condition": "healthy"},
                        {"service": "redis", "condition": "healthy"},
                        {"service": "seed-db", "condition": "completed"},
                    ],
                    "healthcheck": {
                        "command": "curl -fsS http://127.0.0.1:8080/healthz",
                        "interval_sec": 2,
                        "timeout_sec": 2,
                        "retries": 60,
                        "start_period_sec": 30,
                    },
                    "volumes": [
                        {
                            "volume": "snapshots",
                            "target": "/snapshots",
                            "mode": "ro",
                        }
                    ],
                },
                {
                    "name": "customer",
                    "role": "sidecar",
                    "build": {
                        "context": "environment/customer",
                        "dockerfile": "Dockerfile",
                    },
                    "depends_on": [
                        {"service": "main", "condition": "healthy"},
                    ],
                    "healthcheck": {
                        "command": "curl -fsS http://127.0.0.1:9000/health",
                        "interval_sec": 2,
                        "timeout_sec": 5,
                        "retries": 30,
                    },
                },
                {
                    "name": "verifier",
                    "role": "verifier",
                    "image": _pinned_image("verifier", "4"),
                    "user": "root",
                    "resources": {
                        "cpus": 4,
                        "memory_mb": 4096,
                        "platform": "linux/amd64",
                        "network": "none",
                    },
                },
            ],
            "captures": [
                {
                    "name": "finalize-customer",
                    "service": "customer",
                    "command": "python /opt/finalize.py",
                    "timeout_sec": 180,
                    "accepted_exit_codes": [0],
                    "failure_policy": "infrastructure",
                },
                {
                    "name": "postgres-snapshot",
                    "service": "postgres-db",
                    "command": "pg_dump -U postgres -d shop -Fc -f /tmp/pg.dump.tmp",
                    "timeout_sec": 300,
                    "atomic_destination": "/tmp/pg.dump",
                },
                {
                    "name": "redis-snapshot",
                    "service": "redis",
                    "command": ["redis-cli", "--rdb", "/tmp/redis.rdb.tmp"],
                    "timeout_sec": 60,
                    "destination": "/tmp/redis.rdb",
                },
                {
                    "name": "agent-patch",
                    "service": "main",
                    "command": "git diff --binary HEAD > /tmp/agent.patch.tmp",
                    "timeout_sec": 60,
                    "atomic_destination": "/tmp/agent.patch",
                    "accepted_exit_codes": [0, 1],
                    "failure_policy": "agent",
                },
            ],
            "artifacts": [
                {
                    "name": "customer-results",
                    "kind": "file",
                    "source": "/tmp/results.json",
                    "destination": "results.json",
                    "service": "customer",
                    "limits": {"max_bytes": 1_000_000},
                },
                {
                    "name": "agent-tree",
                    "kind": "tree",
                    "source": "/app/api",
                    "destination": "api",
                    "service": "main",
                    "exclude": ["__pycache__", "*.pyc", ".venv"],
                    "limits": {
                        "max_bytes": 100_000_000,
                        "max_files": 10_000,
                        "max_depth": 30,
                    },
                },
                {
                    "name": "agent-sources",
                    "kind": "path_set",
                    "sources": ["/app/api/src", "/app/api/requirements.txt"],
                    "destination": "sources",
                    "service": "main",
                    "limits": {
                        "max_bytes": 20_000_000,
                        "max_files": 5000,
                    },
                },
                {
                    "name": "postgres-dump",
                    "kind": "binary",
                    "source": "/tmp/pg.dump",
                    "destination": "pg.dump",
                    "service": "postgres-db",
                    "preserve_mode": False,
                    "limits": {
                        "max_bytes": 2_000_000_000,
                        "max_depth": 4,
                    },
                },
                {
                    "name": "redis-state",
                    "kind": "service",
                    "source": "/tmp/redis.rdb",
                    "destination": "redis.rdb",
                    "service": "redis",
                    "required": True,
                    "limits": {"max_bytes": 500_000_000},
                },
            ],
            "evaluation": {
                "engine": "rubric_task",
                "entrypoint": "scorer/compute_score.py",
                "hidden_fixtures": ["scorer/data/cutover_cases.json"],
                "repetitions": 2,
                "metadata": {"preserves_existing_task": True},
            },
            "gates": [
                {
                    "name": "structural",
                    "kind": "structural",
                    "required": True,
                    "weight": 0.2,
                    "report": "pytest",
                },
                {
                    "name": "behavioral",
                    "kind": "behavioral",
                    "required": True,
                    "weight": 0.6,
                    "repetitions": 2,
                    "report": "pytest",
                },
                {
                    "name": "determinism",
                    "kind": "determinism",
                    "required": False,
                    "weight": 0.2,
                    "report": "diagnostics",
                },
            ],
            "reports": [
                {
                    "name": "pytest",
                    "format": "ctrf",
                    "path": "reports/pytest.json",
                    "required": True,
                    "max_bytes": 5_000_000,
                },
                {
                    "name": "diagnostics",
                    "format": "json",
                    "path": "reports/diagnostics.json",
                },
            ],
            "result": {
                "output_root": "/tmp/output/cutover",
                "reward_file": "grade.json",
                "reward_key": "score",
                "subscores_key": "subscores",
                "reports": ["pytest", "diagnostics"],
                "trace_file": "trace.json",
                "agent_fault_reward": 0.0,
                "infrastructure_fault": "discard",
            },
        }
    )

    task = TaskToml.model_validate(data)

    assert isinstance(task.artifacts[0], FileArtifact)
    assert isinstance(task.artifacts[1], TreeArtifact)
    assert isinstance(task.artifacts[2], PathSetArtifact)
    assert isinstance(task.artifacts[3], BinaryArtifact)
    assert isinstance(task.artifacts[4], ServiceArtifact)
    assert [capture.name for capture in task.captures] == [
        "finalize-customer",
        "postgres-snapshot",
        "redis-snapshot",
        "agent-patch",
    ]
    assert task.services[2].depends_on[0].condition == "healthy"
    assert task.agent.resources is not None
    assert task.agent.resources.cpus == 16
    assert task.verifier.resources is not None
    assert task.verifier.resources.storage_mb == 40_960
    assert task.verifier.resources.architecture == "amd64"
    assert task.services[-1].role == "verifier"
    assert task.workspace is not None
    assert task.workspace.git_baseline == "a" * 40
    assert task.workspace.clean_paths == [
        "__pycache__",
        ".venv",
        ".pytest_cache",
    ]
    dumped = task.model_dump(mode="json", exclude_none=True)
    assert dumped["artifacts"][0]["max_bytes"] == 1_000_000
    assert "limits" not in dumped["artifacts"][0]
    assert dumped["artifacts"][1]["max_depth"] == 30
    assert dumped["artifacts"][3]["preserve_mode"] is False
    assert dumped["captures"][1]["atomic_destination"] == "/tmp/pg.dump"
    assert dumped["captures"][2]["atomic_destination"] == "/tmp/redis.rdb"
    assert "destination" not in dumped["captures"][2]
    assert dumped["captures"][3]["failure_policy"] == "agent"
    _assert_toml_round_trip(task)


def test_medical_claims_mcp_and_shared_volume_graph_round_trips() -> None:
    data = _base_task()
    data.update(
        {
            "workspace": {
                "seed": "environment/workspace",
                "root": "/workspace",
                "agent_cwd": "/workspace",
                "init_policy": "overlay",
                "git_baseline": False,
                "clean_paths": [".cache"],
                "checkpoint_restore": False,
            },
            "volumes": [
                {"name": "medical-shared"},
                {"name": "claims-config"},
            ],
            "services": [
                {
                    "name": "main",
                    "role": "main",
                    "build": {
                        "context": "environment",
                        "dockerfile": "Dockerfile",
                    },
                    "depends_on": [
                        {"service": "playwright-mcp", "condition": "healthy"},
                        {"service": "workspace-web", "condition": "healthy"},
                    ],
                    "volumes": [
                        {
                            "volume": "medical-shared",
                            "target": "/shared",
                            "mode": "rw",
                        },
                        {
                            "volume": "claims-config",
                            "target": "/etc/claims",
                            "mode": "ro",
                        },
                    ],
                },
                {
                    "name": "playwright-mcp",
                    "role": "sidecar",
                    "build": {
                        "context": "environment/playwright-novnc",
                        "dockerfile": "Dockerfile",
                    },
                    "env": {
                        "MCP_PORT": "3080",
                        "BROWSER_URL": "http://workspace-web:18073",
                    },
                    "ports": [
                        {"container_port": 3080, "name": "mcp"},
                        {"container_port": 6080, "name": "novnc"},
                    ],
                    "shm_mb": 1024,
                    "capabilities": ["SYS_PTRACE"],
                    "healthcheck": {
                        "command": "curl -fsS http://localhost:3080/sse",
                        "interval_sec": 5,
                        "timeout_sec": 10,
                        "retries": 20,
                        "start_period_sec": 15,
                    },
                },
                {
                    "name": "workspace-web",
                    "role": "sidecar",
                    "build": {
                        "context": "environment/workspace",
                        "dockerfile": "Dockerfile",
                        "no_cache": True,
                    },
                    "ports": [{"container_port": 18073, "name": "web"}],
                    "healthcheck": {
                        "command": "curl -fsS http://localhost:18073/api/cases",
                        "interval_sec": 3,
                        "timeout_sec": 5,
                        "retries": 20,
                        "start_period_sec": 15,
                    },
                    "volumes": [
                        {
                            "volume": "medical-shared",
                            "target": "/shared",
                            "mode": "rw",
                        }
                    ],
                },
                {
                    "name": "metrics",
                    "role": "sidecar",
                    "image": _pinned_image("metrics", "5"),
                    "network_mode": "share",
                    "network_share_target": "workspace-web",
                },
            ],
            "artifacts": [
                {
                    "name": "shared-claims",
                    "kind": "service",
                    "source": "/shared",
                    "destination": "claims",
                    "service": "workspace-web",
                    "limits": {
                        "max_bytes": 100_000_000,
                        "max_files": 2000,
                        "max_depth": 20,
                    },
                }
            ],
            "runner": {
                "required_tools": ["bash", "str_replace_editor", "tmux"],
            },
            "mcp_servers": [
                {
                    "name": "playwright",
                    "transport": "sse",
                    "url": "http://playwright-mcp:3080/sse",
                    "service": "playwright-mcp",
                    "depends_on": ["workspace-web"],
                    "readiness": {
                        "kind": "http",
                        "url": "http://playwright-mcp:3080/sse",
                        "accepted_statuses": [200, 204],
                        "timeout_sec": 60,
                        "interval_sec": 2,
                    },
                    "access": "agent",
                },
                {
                    "name": "claims-db",
                    "transport": "stdio",
                    "command": ["python", "-m", "claims_mcp"],
                    "cwd": "/workspace",
                    "service": "main",
                    "readiness": {
                        "kind": "command",
                        "command": ["python", "-m", "claims_mcp", "--check"],
                        "service": "main",
                    },
                    "access": "both",
                },
            ],
        }
    )

    task = TaskToml.model_validate(data)

    assert isinstance(task.mcp_servers[0], SSEMCPServer)
    assert isinstance(task.mcp_servers[1], StdioMCPServer)
    assert task.services[0].volumes[0].mode == "rw"
    assert task.services[3].network_share_target == "workspace-web"
    _assert_toml_round_trip(task)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "services",
            [
                {
                    "name": "main",
                    "role": "main",
                    "image": _pinned_image("main", "1"),
                },
                {
                    "name": "main",
                    "role": "sidecar",
                    "image": _pinned_image("other", "2"),
                },
            ],
            "duplicate service name",
        ),
        (
            "volumes",
            [
                {"name": "shared"},
                {"name": "shared"},
            ],
            "duplicate volume name",
        ),
        (
            "mcp_servers",
            [
                {
                    "name": "browser",
                    "transport": "sse",
                    "url": "http://browser:3000/sse",
                },
                {
                    "name": "browser",
                    "transport": "stdio",
                    "command": ["browser-mcp"],
                },
            ],
            "duplicate MCP server name",
        ),
        (
            "mcp_servers",
            [
                {
                    "name": "bash",
                    "transport": "stdio",
                    "command": ["bash-mcp"],
                },
            ],
            "declared as both built-in and MCP",
        ),
    ],
)
def test_duplicate_capability_names_are_rejected(
    field: str, value: object, message: str
) -> None:
    data = _base_task()
    data[field] = value

    with pytest.raises(ValidationError, match=message):
        TaskToml.model_validate(data)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda data: data.update(
                {
                    "artifacts": [
                        {
                            "name": "snapshot",
                            "kind": "file",
                            "source": "/tmp/snapshot",
                            "service": "missing",
                        }
                    ]
                }
            ),
            "artifact 'snapshot' references unknown service 'missing'",
        ),
        (
            lambda data: data.update(
                {
                    "captures": [
                        {
                            "name": "snapshot",
                            "service": "missing",
                            "command": "snapshot",
                            "timeout_sec": 10,
                            "destination": "/tmp/snapshot",
                        }
                    ]
                }
            ),
            "capture 'snapshot' references unknown service 'missing'",
        ),
        (
            lambda data: data["services"][0].update(
                {
                    "volumes": [
                        {
                            "volume": "missing",
                            "target": "/workspace/data",
                        }
                    ]
                }
            ),
            "references unknown volume 'missing'",
        ),
        (
            lambda data: data["services"][0].update(
                {
                    "depends_on": [
                        {"service": "missing", "condition": "started"},
                    ]
                }
            ),
            "depends on unknown service 'missing'",
        ),
        (
            lambda data: data.update(
                {
                    "mcp_servers": [
                        {
                            "name": "browser",
                            "transport": "sse",
                            "url": "http://browser:3000/sse",
                            "service": "missing",
                        }
                    ]
                }
            ),
            "MCP server 'browser' references unknown service 'missing'",
        ),
    ],
)
def test_unknown_service_and_volume_references_are_rejected(
    mutator: Any, message: str
) -> None:
    data = _minimal_service_task()
    mutator(data)

    with pytest.raises(ValidationError, match=message):
        TaskToml.model_validate(data)


@pytest.mark.parametrize(
    ("services", "message"),
    [
        (
            [
                {
                    "name": "cache",
                    "role": "sidecar",
                    "image": _pinned_image("redis", "1"),
                }
            ],
            "exactly one role = 'main'",
        ),
        (
            [
                {
                    "name": "main-a",
                    "role": "main",
                    "image": _pinned_image("main-a", "1"),
                },
                {
                    "name": "main-b",
                    "role": "main",
                    "image": _pinned_image("main-b", "2"),
                },
            ],
            "exactly one role = 'main'",
        ),
        (
            [
                {
                    "name": "main",
                    "role": "main",
                    "image": "main:latest",
                    "build": {"context": "environment"},
                }
            ],
            "exactly one of image or build",
        ),
        (
            [{"name": "main", "role": "main"}],
            "exactly one of image or build",
        ),
    ],
)
def test_service_role_and_image_build_invariants(
    services: list[dict[str, Any]], message: str
) -> None:
    data = _base_task()
    data["services"] = services

    with pytest.raises(ValidationError, match=message):
        TaskToml.model_validate(data)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda data: data["services"][0].update(
                {
                    "build": {"context": "/Users/author/task"},
                    "image": None,
                }
            ),
            "absolute/host paths are forbidden",
        ),
        (
            lambda data: data["services"][0].update(
                {
                    "volumes": [
                        {
                            "volume": "/Users/author/database",
                            "target": "/data",
                        }
                    ]
                }
            ),
            "volume mount reference",
        ),
        (
            lambda data: data.update(
                {
                    "workspace": {
                        "root": "/Users/author/workspace",
                    }
                }
            ),
            "reserved path /Users",
        ),
        (
            lambda data: data.update(
                {
                    "workspace": {
                        "root": "/mcp_server/workspace",
                    }
                }
            ),
            "reserved path /mcp_server",
        ),
        (
            lambda data: data.update(
                {
                    "workspace": {
                        "root": "/task/capsule/workspace",
                    }
                }
            ),
            "reserved path /task/capsule",
        ),
        (
            lambda data: data.update({"workspace": {"root": "/"}}),
            "cannot be the container root",
        ),
        (
            lambda data: data.update(
                {
                    "result": {
                        "output_root": "/var/results",
                    }
                }
            ),
            "must be /tmp/output",
        ),
    ],
)
def test_host_binds_and_unsafe_roots_are_rejected(mutator: Any, message: str) -> None:
    data = _minimal_service_task()
    mutator(data)

    with pytest.raises(ValidationError, match=message):
        TaskToml.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("privileged", True),
        ("devices", ["/dev/kvm:/dev/kvm"]),
        ("volumes_from", ["docker-daemon"]),
    ],
)
def test_service_escape_hatches_are_not_authorable(field: str, value: object) -> None:
    data = _minimal_service_task()
    data["services"][0][field] = value

    with pytest.raises(ValidationError, match=field):
        TaskToml.model_validate(data)


def test_host_bind_and_runtime_socket_paths_are_rejected() -> None:
    data = _minimal_service_task()
    data["volumes"] = [{"name": "state"}]
    data["services"][0]["volumes"] = [
        {
            "volume": "state",
            "source": "/Users/author/state",
            "target": "/workspace/state",
        }
    ]
    with pytest.raises(ValidationError, match="source"):
        TaskToml.model_validate(data)

    socket_target = _minimal_service_task()
    socket_target["volumes"] = [{"name": "state"}]
    socket_target["services"][0]["volumes"] = [
        {
            "volume": "state",
            "target": "/var/run/docker.sock",
        }
    ]
    with pytest.raises(ValidationError, match="runtime socket"):
        TaskToml.model_validate(socket_target)

    protected_target = _minimal_service_task()
    protected_target["volumes"] = [{"name": "state"}]
    protected_target["services"][0]["volumes"] = [
        {
            "volume": "state",
            "target": "/proc/task-state",
        }
    ]
    with pytest.raises(ValidationError, match="protected path /proc"):
        TaskToml.model_validate(protected_target)

    indirect_socket = _minimal_service_task()
    indirect_socket["services"][0]["env"] = {
        "DOCKER_HOST": "unix:///var/run/docker.sock"
    }
    with pytest.raises(ValidationError, match="orchestration variable"):
        TaskToml.model_validate(indirect_socket)

    socket_artifact = _minimal_service_task()
    socket_artifact["artifacts"] = [
        {
            "name": "socket",
            "kind": "file",
            "source": "/var/run/docker.sock",
        }
    ]
    with pytest.raises(ValidationError, match="runtime socket"):
        TaskToml.model_validate(socket_artifact)


@pytest.mark.parametrize(
    "capability",
    ["SYS_ADMIN", "SYS_MODULE", "DAC_READ_SEARCH", "NET_ADMIN"],
)
def test_dangerous_or_unrecognized_capabilities_are_rejected(
    capability: str,
) -> None:
    data = _minimal_service_task()
    data["services"][0]["capabilities"] = [capability]

    with pytest.raises(ValidationError, match="forbidden capabilities"):
        TaskToml.model_validate(data)


def test_sys_ptrace_is_the_only_authorable_capability() -> None:
    data = _minimal_service_task()
    data["services"][0]["capabilities"] = ["sys_ptrace"]

    task = TaskToml.model_validate(data)
    assert task.services[0].capabilities == ["SYS_PTRACE"]


def test_verifier_can_request_only_sys_ptrace() -> None:
    data = _base_task()
    data["verifier"] = {"capabilities": ["sys_ptrace"]}

    task = TaskToml.model_validate(data)

    assert task.verifier.capabilities == ["SYS_PTRACE"]
    data["verifier"] = {"capabilities": ["SYS_ADMIN"]}
    with pytest.raises(ValidationError, match="forbidden capabilities"):
        TaskToml.model_validate(data)


def test_external_images_must_be_digest_pinned() -> None:
    data = _minimal_service_task()
    data["services"][0]["image"] = "example.invalid/main:latest"

    with pytest.raises(ValidationError, match="must be pinned"):
        TaskToml.model_validate(data)

    build_data = _minimal_service_task()
    build_data["services"][0].pop("image")
    build_data["services"][0]["build"] = {"context": "environment"}
    assert TaskToml.model_validate(build_data).services[0].build is not None


def test_explicit_verifier_service_round_trips_and_is_unique() -> None:
    data = _minimal_service_task()
    data["result"] = {
        "reward_file": "grade.json",
        "reward_key": "score",
    }
    data["services"].append(
        {
            "name": "verifier",
            "role": "verifier",
            "image": _pinned_image("verifier", "2"),
            "user": "root",
            "resources": {"platform": "linux/arm64"},
        }
    )

    task = TaskToml.model_validate(data)
    dumped = task.model_dump(mode="json", exclude_none=True)
    assert dumped["services"][1]["role"] == "verifier"
    assert dumped["services"][1]["resources"]["architecture"] == "arm64"
    _assert_toml_round_trip(task)

    duplicate = copy.deepcopy(data)
    duplicate["services"].append(
        {
            "name": "verifier-2",
            "role": "verifier",
            "build": {"context": "scorer"},
        }
    )
    with pytest.raises(ValidationError, match="at most one role = 'verifier'"):
        TaskToml.model_validate(duplicate)


@pytest.mark.parametrize(
    ("sharing_service", "target_service"),
    [("main", "verifier"), ("verifier", "main")],
)
def test_network_namespace_share_cannot_cross_verifier_boundary(
    sharing_service: str,
    target_service: str,
) -> None:
    data = _minimal_service_task()
    data["result"] = {
        "reward_file": "grade.json",
        "reward_key": "score",
    }
    data["services"].append(
        {
            "name": "verifier",
            "role": "verifier",
            "image": _pinned_image("verifier", "2"),
        }
    )
    sharing = next(
        service for service in data["services"] if service["name"] == sharing_service
    )
    sharing["network_mode"] = "share"
    sharing["network_share_target"] = target_service
    task = TaskToml.model_validate(data)

    with pytest.raises(ValueError, match="verifier trust boundary"):
        resolve_capabilities(task)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (None, r"\[result\] section with explicit reward_file and reward_key"),
        ({}, r"explicit \[result\] fields: reward_file, reward_key"),
        ({"reward_file": "grade.json"}, r"explicit \[result\] fields: reward_key"),
        ({"reward_key": "score"}, r"explicit \[result\] fields: reward_file"),
    ],
)
def test_verifier_service_requires_explicit_reward_contract(
    result: dict[str, Any] | None, message: str
) -> None:
    data = _minimal_service_task()
    data["services"].append(
        {
            "name": "verifier",
            "role": "verifier",
            "image": _pinned_image("verifier", "2"),
        }
    )
    if result is not None:
        data["result"] = result

    with pytest.raises(ValidationError, match=message):
        TaskToml.model_validate(data)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf")])
def test_result_agent_fault_reward_must_be_finite(reward: float) -> None:
    data = _base_task()
    data["result"] = {
        "reward_file": "grade.json",
        "reward_key": "score",
        "agent_fault_reward": reward,
    }

    with pytest.raises(ValidationError, match="agent_fault_reward must be finite"):
        TaskToml.model_validate(data)


@pytest.mark.parametrize(
    ("field", "declaration"),
    [
        (
            "workspace",
            {
                "seed": "environment/workspace",
                "root": "/workspace",
            },
        ),
        (
            "artifacts",
            [
                {
                    "name": "answer",
                    "kind": "file",
                    "source": "/tmp/output/answer.json",
                }
            ],
        ),
        (
            "evaluation",
            {
                "engine": "rubric_task",
                "entrypoint": "scorer/compute_score.py",
            },
        ),
    ],
)
def test_implicit_main_capability_sections_are_detected(
    field: str, declaration: object
) -> None:
    data = _base_task()
    data[field] = declaration

    task = TaskToml.model_validate(data)
    assert task.services == []
    assert is_capability_task(task)

    capabilities = resolve_capabilities(task)
    assert capabilities.services == ()
    if field == "artifacts":
        assert capabilities.artifacts[0]["service"] == "main"


def test_implicit_main_still_rejects_unknown_artifact_service() -> None:
    data = _base_task()
    data["artifacts"] = [
        {
            "name": "answer",
            "kind": "file",
            "source": "/tmp/output/answer.json",
            "service": "missing",
        }
    ]

    with pytest.raises(
        ValidationError,
        match="artifact 'answer' references unknown service 'missing'",
    ):
        TaskToml.model_validate(data)


def test_workspace_runtime_fields_normalize_and_round_trip() -> None:
    data = _minimal_service_task()
    data["workspace"] = {
        "seed": " environment//workspace ",
        "root": "/workspace",
        "agent_cwd": "/workspace/src",
        "init_policy": " Overlay ",
        "git_baseline": "b" * 40,
        "clean_paths": [" build//cache ", ".cache"],
        "checkpoint_restore": False,
    }

    task = TaskToml.model_validate(data)
    assert task.workspace is not None
    assert task.workspace.seed == "environment/workspace"
    assert task.workspace.init_policy == "overlay"
    assert task.workspace.clean_paths == ["build/cache", ".cache"]
    assert task.workspace.checkpoint_restore is False
    _assert_toml_round_trip(task)

    empty = _minimal_service_task()
    empty["workspace"] = {
        "root": "/workspace",
        "init_policy": "empty",
    }
    workspace = TaskToml.model_validate(empty).workspace
    assert workspace is not None
    assert workspace.seed is None
    assert workspace.agent_cwd == "/workspace"
    assert workspace.git_baseline is True
    assert workspace.checkpoint_restore is True


@pytest.mark.parametrize(
    ("workspace", "message"),
    [
        (
            {"root": "/workspace"},
            "seed is required unless init_policy is 'empty'",
        ),
        (
            {
                "root": "/workspace",
                "seed": "/Users/author/workspace",
            },
            "task-relative",
        ),
        (
            {
                "root": "/workspace",
                "seed": "../outside",
            },
            "task-relative",
        ),
        (
            {
                "root": "/workspace",
                "seed": "environment",
                "agent_cwd": "/other",
            },
            "must be inside workspace.root",
        ),
        (
            {
                "root": "/workspace",
                "seed": "environment",
                "init_policy": "snapshot",
            },
            "literal_error",
        ),
        (
            {
                "root": "/workspace",
                "seed": "environment",
                "git_baseline": "refs/tags/pristine",
            },
            "40-character lowercase commit ID",
        ),
        (
            {
                "root": "/workspace",
                "seed": "environment",
                "git_baseline": "A" * 40,
            },
            "40-character lowercase commit ID",
        ),
        (
            {
                "root": "/workspace",
                "seed": "environment",
                "clean_paths": ["build//cache", "build/cache"],
            },
            "clean_paths entries must be unique",
        ),
        (
            {
                "root": "/workspace",
                "seed": "environment",
                "clean_paths": ["../outside"],
            },
            "task-relative",
        ),
        (
            {
                "root": "/workspace",
                "seed": "environment",
                "checkpoint_restore": "true",
            },
            "valid boolean",
        ),
    ],
)
def test_workspace_runtime_field_validation(
    workspace: dict[str, Any], message: str
) -> None:
    data = _minimal_service_task()
    data["workspace"] = workspace

    with pytest.raises(ValidationError, match=message):
        TaskToml.model_validate(data)


@pytest.mark.parametrize(
    ("mutator", "field"),
    [
        (
            lambda data: data.update(
                {
                    "artifacts": [
                        {
                            "name": "tree",
                            "kind": "tree",
                            "source": "/tmp/tree",
                            "preserve_mode": True,
                        }
                    ]
                }
            ),
            "preserve_mode",
        ),
        (
            lambda data: data.update(
                {"volumes": [{"name": "state", "scope": "agent"}]}
            ),
            "scope",
        ),
        (
            lambda data: data.update({"capsule_path": "/task/capsule"}),
            "capsule_path",
        ),
    ],
)
def test_operator_owned_or_kind_specific_fields_are_rejected(
    mutator: Any, field: str
) -> None:
    data = _minimal_service_task()
    mutator(data)

    with pytest.raises(ValidationError, match=field):
        TaskToml.model_validate(data)


def test_capture_aliases_dump_canonical_fields() -> None:
    data = _minimal_service_task()
    data["captures"] = [
        {
            "name": "snapshot",
            "service": "main",
            "command": "snapshot --output /tmp/snapshot.tmp",
            "timeout_sec": 30,
            "destination": "/tmp/snapshot",
            "fault_policy": "agent",
        }
    ]

    task = TaskToml.model_validate(data)
    dumped = task.model_dump(mode="json", exclude_none=True)["captures"][0]
    assert dumped["atomic_destination"] == "/tmp/snapshot"
    assert dumped["failure_policy"] == "agent"
    assert "destination" not in dumped
    assert "fault_policy" not in dumped
    _assert_toml_round_trip(task)


@pytest.mark.parametrize(
    ("section", "destination", "message"),
    [
        ("artifact", "/tmp/output.bin", "task-relative"),
        ("capture", "snapshot.bin", "absolute path below /tmp"),
        ("capture", "/mcp_server/snapshot.bin", "protected path"),
    ],
)
def test_artifact_and_capture_destination_policies(
    section: str, destination: str, message: str
) -> None:
    data = _minimal_service_task()
    if section == "artifact":
        data["artifacts"] = [
            {
                "name": "snapshot",
                "kind": "file",
                "source": "/tmp/snapshot",
                "destination": destination,
            }
        ]
    else:
        data["captures"] = [
            {
                "name": "snapshot",
                "service": "main",
                "command": "snapshot",
                "timeout_sec": 10,
                "atomic_destination": destination,
            }
        ]

    with pytest.raises(ValidationError, match=message):
        TaskToml.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "artifacts",
            [
                {
                    "name": "bad",
                    "kind": "file",
                    "source": "/tmp/bad",
                    "limits": {"max_bytes": 0},
                }
            ],
            "greater than 0",
        ),
        (
            "artifacts",
            [
                {
                    "name": "boolean-depth",
                    "kind": "tree",
                    "source": "/tmp/bad",
                    "max_depth": True,
                }
            ],
            "must be an integer",
        ),
        (
            "artifacts",
            [
                {
                    "name": "deep",
                    "kind": "tree",
                    "source": "/tmp/bad",
                    "max_depth": 1025,
                }
            ],
            "less than or equal to 1024",
        ),
        (
            "artifacts",
            [
                {
                    "name": "bad-depth",
                    "kind": "tree",
                    "source": "/tmp/bad",
                    "limits": {"max_depth": 0},
                }
            ],
            "greater than 0",
        ),
        (
            "agent",
            {"resources": {"cpus": 0}},
            "greater than 0",
        ),
        (
            "captures",
            [
                {
                    "name": "bad",
                    "service": "main",
                    "command": "true",
                    "timeout_sec": 0,
                    "destination": "/tmp/bad",
                }
            ],
            "greater than 0",
        ),
    ],
)
def test_limits_must_be_positive(field: str, value: object, message: str) -> None:
    data = _base_task()
    data[field] = value

    with pytest.raises(ValidationError, match=message):
        TaskToml.model_validate(data)


def test_gpu_resources_require_types_and_valid_platform() -> None:
    data = _base_task()
    data["agent"] = {
        "resources": {
            "cpus": 8,
            "memory_mb": 32_768,
            "storage_mb": 1_024_000,
            "gpus": 1,
            "gpu_types": ["H100"],
            "platform": "linux/amd64",
            "architecture": "amd64",
            "build_timeout_sec": 7200,
            "network": "internet",
        }
    }

    task = TaskToml.model_validate(data)
    assert task.agent.resources is not None
    assert task.agent.resources.gpu_types == ["H100"]
    assert task.agent.resources.architecture == "amd64"

    missing_type = copy.deepcopy(data)
    missing_type["agent"]["resources"]["gpu_types"] = []
    with pytest.raises(ValidationError, match="gpu_types is required"):
        TaskToml.model_validate(missing_type)

    invalid_platform = copy.deepcopy(data)
    invalid_platform["agent"]["resources"]["platform"] = "amd64"
    with pytest.raises(ValidationError, match="OCI"):
        TaskToml.model_validate(invalid_platform)

    mismatched_architecture = copy.deepcopy(data)
    mismatched_architecture["agent"]["resources"]["architecture"] = "arm64"
    with pytest.raises(ValidationError, match="conflicts"):
        TaskToml.model_validate(mismatched_architecture)

    architecture_only = _minimal_service_task()
    architecture_only["services"][0].pop("image")
    architecture_only["services"][0]["build"] = {
        "context": "environment",
        "architecture": "arm64",
    }
    task = TaskToml.model_validate(architecture_only)
    assert task.services[0].build is not None
    assert task.services[0].build.platform == "linux/arm64"


@pytest.mark.parametrize(
    ("server", "message"),
    [
        (
            {
                "name": "browser",
                "transport": "sse",
                "url": "stdio://browser",
            },
            "absolute http:// or https:// URL",
        ),
        (
            {
                "name": "browser",
                "transport": "websocket",
                "url": "http://browser:3000",
            },
            "union_tag_invalid",
        ),
        (
            {
                "name": "browser",
                "transport": "stdio",
                "command": [],
            },
            "command argv cannot be empty",
        ),
    ],
)
def test_mcp_transport_and_endpoint_validation(
    server: dict[str, Any], message: str
) -> None:
    data = _base_task()
    data["mcp_servers"] = [server]

    with pytest.raises(ValidationError, match=message):
        TaskToml.model_validate(data)


def test_output_paths_reject_parent_traversal() -> None:
    data = _base_task()
    data["outputs"] = [{"path": "/tmp/output/../mcp_server/forged.json"}]

    with pytest.raises(ValidationError, match="under /tmp/output"):
        TaskToml.model_validate(data)


def test_every_checked_in_v11_task_toml_still_parses() -> None:
    roots = [
        ROOT / "examples",
        ROOT / "alignerr_plugin" / "src" / "alignerr_plugin" / "starter_templates",
    ]
    task_paths = sorted(
        path
        for root in roots
        for path in root.rglob("task.toml")
        if tomllib.loads(path.read_text()).get("schema_version") == "1.1"
    )

    assert task_paths
    for task_path in task_paths:
        parsed = TaskToml.model_validate(tomllib.loads(task_path.read_text()))
        assert parsed.schema_version == "1.1", task_path
        assert parsed.workspace is None, task_path
        assert parsed.services == [], task_path
        assert parsed.artifacts == [], task_path
        assert parsed.mcp_servers == [], task_path
