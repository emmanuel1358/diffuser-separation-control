from __future__ import annotations

import hashlib
import inspect
import io
import json
import subprocess
import tarfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import tomli_w
import yaml
from _fixture_guard import requires_examples
from alignerr_plugin.capsule import export_task_capsule
from alignerr_plugin.exporters import taiga as taiga_exporter
from alignerr_plugin.exporters.harbor import export_harbor
from alignerr_plugin.exporters.taiga import STARTUP_COMMAND, build_job_payload
from alignerr_plugin.local_cli import app as local_app
from alignerr_plugin.utils import load_task_toml
from typer.testing import CliRunner

from alignerr_plugin import commands as plugin_commands

ROOT = Path(__file__).resolve().parents[2]
_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_OUTER_IMAGE = f"example.invalid/task-capsule@{_DIGEST_A}"
_DATABASE_IMAGE = f"example.invalid/database@{_DIGEST_B}"
_SECRET_MARKERS = (
    "AGENT_SECRET_VALUE",
    "BUILD_SECRET_VALUE",
    "DATABASE_SECRET_VALUE",
    "CAPTURE_SECRET_VALUE",
    "GATE_SECRET_VALUE",
)


def _native_task(
    *,
    metadata_contract: bool = False,
    mcp_transport: str | None = None,
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "schema_version": "1.2",
        "task": {
            "name": "labelbox/taiga-capability-parity",
            "description": "Native multi-service Taiga export fixture.",
        },
        "environment": {
            "required_resources": "8vcpu+64gib",
            "storage_mb": 20_000,
            "allow_internet": False,
        },
        "agent": {
            "timeout_sec": 7200,
            "user": "agent",
            "resources": {
                "cpus": 4,
                "memory_mb": 8192,
                "storage_mb": 8192,
                "gpus": 0,
                "platform": "linux/amd64",
                "build_timeout_sec": 900,
                "runtime_timeout_sec": 7200,
            },
        },
        "verifier": {
            "timeout_sec": 5400,
            "user": "root",
            "env": [],
            "resources": {
                "cpus": 2,
                "memory_mb": 4096,
                "storage_mb": 4096,
                "gpus": 0,
                "platform": "linux/amd64",
                "build_timeout_sec": 300,
                "runtime_timeout_sec": 5400,
            },
        },
        "workspace": {
            "seed": "starter",
            "root": "/workdir",
            "agent_cwd": "/workdir/src",
            "init_policy": "copy",
            "git_baseline": True,
        },
        "volumes": [{"name": "database-data"}],
        "services": [
            {
                "name": "main",
                "role": "main",
                "build": {
                    "context": "environment/main",
                    "dockerfile": "Dockerfile",
                    "args": {"AUTH_TOKEN": "BUILD_SECRET_VALUE"},
                    "platform": "linux/amd64",
                },
                "user": "agent",
                "env": {"AGENT_SECRET": "AGENT_SECRET_VALUE"},
                "resources": {
                    "cpus": 4,
                    "memory_mb": 8192,
                    "storage_mb": 8192,
                    "gpus": 0,
                    "platform": "linux/amd64",
                    "build_timeout_sec": 1800,
                    "runtime_timeout_sec": 7200,
                },
                "depends_on": [
                    {"service": "database", "condition": "healthy"},
                ],
            },
            {
                "name": "database",
                "role": "sidecar",
                "image": _DATABASE_IMAGE,
                "env": {"DATABASE_PASSWORD": "DATABASE_SECRET_VALUE"},
                "resources": {
                    "cpus": 2,
                    "memory_mb": 2048,
                    "storage_mb": 1024,
                    "gpus": 0,
                    "platform": "linux/amd64",
                },
                "ports": [{"container_port": 5432, "name": "postgres"}],
                "healthcheck": {
                    "command": ["CMD", "pg_isready"],
                    "interval_sec": 2,
                    "timeout_sec": 3,
                    "retries": 5,
                },
                "volumes": [
                    {
                        "volume": "database-data",
                        "target": "/var/lib/postgresql/data",
                        "mode": "rw",
                    }
                ],
            },
        ],
        "captures": [
            {
                "name": "database-snapshot",
                "service": "database",
                "command": [
                    "sh",
                    "-c",
                    "CAPTURE_SECRET_VALUE > /tmp/database.sql.tmp",
                ],
                "timeout_sec": 45,
                "atomic_destination": "/tmp/database.sql",
            }
        ],
        "artifacts": [
            {
                "name": "database-dump",
                "kind": "service",
                "source": "/tmp/database.sql",
                "destination": "database.sql",
                "service": "database",
            },
            {
                "name": "agent-sources",
                "kind": "path_set",
                "sources": ["/workdir/src", "/workdir/pyproject.toml"],
                "destination": "sources",
                "service": "main",
            },
        ],
        "gates": [
            {
                "name": "behavioral",
                "kind": "behavioral",
                "command": "GATE_SECRET_VALUE",
                "required": True,
                "weight": 1.0,
                "report": "pytest",
            }
        ],
        "reports": [
            {
                "name": "pytest",
                "format": "ctrf",
                "path": "reports/pytest.json",
                "required": True,
            }
        ],
        "result": {
            "output_root": "/tmp/output",
            "reward_file": "grade.json",
            "reward_key": "score",
            "reports": ["pytest"],
        },
        "outputs": [
            {
                "path": "/tmp/output/grade.json",
                "required": True,
                "description": "Canonical verifier result.",
            }
        ],
        "runner": {
            "attempts": 1,
            "turn_limit": 1500,
            "max_ctx": 1_000_000,
            "context_mode": "memory",
            "api_model_name": "claude-fable-5",
            "required_tools": ["bash", "str_replace_editor", "tmux"],
            "timeouts": {
                "setup_sec": 600,
                "grading_sec": 1200,
                "tool_sec": 1800,
                "max_episode_sec": 3600,
            },
        },
        "difficulty": {
            "task_type": "software_engineering",
            "domain": "repo_debugging",
            "reward_type": "multi_deterministic_rubrics",
        },
    }
    if metadata_contract:
        task["metadata"] = {"taiga": {"image_contract": "outer_capsule"}}
    if mcp_transport == "sse":
        task["mcp_servers"] = [
            {
                "name": "database-tools",
                "transport": "sse",
                "url": "http://database:8080/sse",
                "headers": {"Authorization": "DATABASE_SECRET_VALUE"},
                "service": "database",
            }
        ]
    elif mcp_transport == "stdio":
        task["mcp_servers"] = [
            {
                "name": "database-tools",
                "transport": "stdio",
                "command": ["python", "-m", "database_mcp"],
                "cwd": "/workdir",
                "env": {"TOKEN": "DATABASE_SECRET_VALUE"},
                "service": "main",
            }
        ]
    return task


def _write_problem(problem_dir: Path, task: dict[str, Any]) -> Path:
    problem_dir.mkdir(parents=True)
    (problem_dir / "task.toml").write_text(tomli_w.dumps(task))
    (problem_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "taiga_task",
                "problem_data": {
                    "instance_id": "taiga-capability-parity",
                    "description": "Native capability parity fixture.",
                },
            }
        )
        + "\n"
    )
    (problem_dir / "instruction.md").write_text(
        "Repair the application from /workdir/src and publish verifier outputs.\n"
    )
    main = problem_dir / "environment" / "main"
    main.mkdir(parents=True)
    (main / "Dockerfile").write_text(
        "FROM python:3.13-slim\n"
        "ARG AUTH_TOKEN\n"
        "WORKDIR /workdir\n"
        'CMD ["sleep", "infinity"]\n'
    )
    starter = problem_dir / "starter" / "src"
    starter.mkdir(parents=True)
    (starter / "app.py").write_text("print('seed')\n")
    scorer_data = problem_dir / "scorer" / "data"
    scorer_data.mkdir(parents=True)
    (scorer_data / "private.json").write_text('{"expected": true}\n')
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n"
        "    return {'score': 1.0}\n"
    )
    (problem_dir / "data").mkdir()
    (problem_dir / "data" / "public.txt").write_text("public\n")
    return problem_dir


class FakeDocker:
    def __call__(
        self,
        command: list[str],
        *,
        cwd: str | None,
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, check, text, capture_output
        stdout = ""
        if command[1:3] == ["image", "inspect"]:
            image_ref = command[-1]
            image_id = _DIGEST_B if "database@" in image_ref else _DIGEST_A
            stdout = json.dumps(
                [
                    {
                        "Id": image_id,
                        "RepoDigests": [],
                        "Os": "linux",
                        "Architecture": "amd64",
                    }
                ]
            )
        elif command[1:3] == ["image", "save"]:
            output_path = Path(command[command.index("--output") + 1])
            refs = command[command.index("--output") + 2 :]
            payload = json.dumps(sorted(refs)).encode()
            with tarfile.open(output_path, "w") as archive:
                info = tarfile.TarInfo("manifest.json")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def _taiga_problem(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["problems_metadata"]["problem_set"]["problems"][0]


def test_native_semantics_match_taiga_harbor_and_capsule_manifests(
    tmp_path: Path,
) -> None:
    problem_dir = _write_problem(tmp_path / "problem", _native_task())
    taiga_payload = build_job_payload(
        problem_dir,
        image_ref="LOCAL_IMAGE",
        image_is_outer_capsule=True,
    )
    harbor_dir = tmp_path / "harbor"
    export_harbor(problem_dir, harbor_dir)
    capsule = export_task_capsule(
        problem_dir,
        tmp_path / "capsule",
        trusted_build=True,
        runner=FakeDocker(),
    )

    problem = _taiga_problem(taiga_payload)
    summary = problem["extra_fields"]["task_metadata"]["capability_summary"]
    harbor_task = tomllib.loads((harbor_dir / "task.toml").read_text())
    harbor_compose = yaml.safe_load(
        (harbor_dir / "environment" / "docker-compose.yaml").read_text()
    )
    capsule_manifest = json.loads(capsule.capsule_manifest.read_text())

    assert summary["schema_version"] == "alignerr.taiga.capability-summary.v1"
    assert summary["runtime"] == {
        "outer_image_contract": "outer_capsule",
        "contract_assertion": "exporter_flag",
        "isolation": "firecracker",
        "orchestration": "nested_docker",
    }
    assert problem["container_runtime"] == "firecracker"
    assert (
        taiga_payload["problems_metadata"]["problem_set"]["container_runtime"]
        == "firecracker"
    )
    assert problem["startup_command"] == STARTUP_COMMAND
    assert capsule_manifest["startup_command"] == STARTUP_COMMAND
    assert json.dumps(problem).count(STARTUP_COMMAND) == 1

    summary_services = [
        (service["name"], service["role"]) for service in summary["services"]
    ]
    capsule_services = [
        (service["service"], service["role"])
        for service in sorted(
            capsule_manifest["images"],
            key=lambda service: service["service"],
        )
    ]
    assert (
        summary_services
        == capsule_services
        == [
            ("database", "sidecar"),
            ("main", "main"),
        ]
    )
    assert set(harbor_compose["services"]) == {"main", "database"}
    assert [volume["name"] for volume in summary["volumes"]] == list(
        harbor_compose["volumes"]
    )

    assert summary["workspace"] == {
        "root": "/workdir",
        "agent_cwd": "/workdir/src",
        "output_root": "/tmp/output",
        "init_policy": "copy",
        "seed": "starter",
        "git_baseline": True,
    }
    assert harbor_task["workspace"]["root"] == summary["workspace"]["root"]
    assert harbor_task["workspace"]["agent_cwd"] == summary["workspace"]["agent_cwd"]
    assert problem["code_root"] == "/workdir/src"
    assert problem["output_directory"] == "/tmp/output"

    assert {(item["name"], item["service"]) for item in summary["artifacts"]} == {
        ("agent-sources", "main"),
        ("database-dump", "database"),
    }
    assert summary["captures"] == [
        {
            "order": 0,
            "name": "database-snapshot",
            "service": "database",
            "failure_policy": "infrastructure",
            "atomic_destination": "/tmp/database.sql",
        }
    ]
    assert [tool["name"] for tool in summary["tools"]] == [
        "bash",
        "str_replace_editor",
        "tmux",
    ]
    assert summary["gates"] == [
        {
            "order": 0,
            "name": "behavioral",
            "kind": "behavioral",
            "required": True,
            "report": "pytest",
        }
    ]

    resources = summary["resources"]
    assert resources["authored_required_resources"] == "8vcpu+64gib"
    assert resources["selected_required_resources"] == "8vcpu+64gib"
    assert resources["agent_service_peak"] == {
        "cpus": 6,
        "memory_mb": 10_240,
        "storage_mb": 20_000,
        "gpus": 0,
    }
    assert resources["verifier_phase"] == {
        "cpus": 2,
        "memory_mb": 4096,
        "storage_mb": 4096,
        "gpus": 0,
    }
    assert resources["outer_peak"] == resources["agent_service_peak"]
    assert resources["mapping"]["minimum_capacity_fit"] == "6vcpu+32gib"
    assert resources["mapping"]["capacity_validated"] is True
    assert resources["mapping"]["preflight_required"] is True
    assert resources["mapping"]["reasons"] == [
        "aggregate_peak_is_not_an_exact_taiga_enum",
        "capsule_image_and_daemon_overhead_not_encoded",
        "storage_and_named_volume_capacity_not_encoded",
    ]
    assert "built outer capsule peak" in resources["mapping"]["preflight_requirement"]

    assert problem["setup_timeout_seconds"] == 1800
    assert problem["tool_timeout_seconds"] == 7200
    assert problem["grading_timeout_seconds"] == 5400
    assert problem["extra_fields"]["grading_timeout_seconds"] == 5400
    assert taiga_payload["max_timeout_seconds"] == 7200

    redacted = json.dumps(summary, sort_keys=True)
    for marker in _SECRET_MARKERS:
        assert marker not in redacted


def test_capability_summary_uses_resource_platform_when_build_omits_it(
    tmp_path: Path,
) -> None:
    task = _native_task()
    main = next(service for service in task["services"] if service["name"] == "main")
    main["build"].pop("platform", None)
    main["resources"]["platform"] = "linux/arm64"
    problem_dir = _write_problem(tmp_path / "resource-platform", task)

    payload = build_job_payload(
        problem_dir,
        image_ref="LOCAL_IMAGE",
        image_is_outer_capsule=True,
    )

    services = _taiga_problem(payload)["extra_fields"]["task_metadata"][
        "capability_summary"
    ]["services"]
    main_summary = next(service for service in services if service["name"] == "main")
    assert main_summary["platform"] == "linux/arm64"


def test_taiga_summary_includes_implicit_main_service(tmp_path: Path) -> None:
    task = _native_task()
    for section in ("services", "volumes", "captures", "artifacts", "mcp_servers"):
        task.pop(section, None)
    task["evaluation"] = {"engine": "rubric_task"}
    problem_dir = _write_problem(tmp_path / "implicit-main", task)

    payload = build_job_payload(
        problem_dir,
        image_ref="LOCAL_IMAGE",
        image_is_outer_capsule=True,
    )

    services = _taiga_problem(payload)["extra_fields"]["task_metadata"][
        "capability_summary"
    ]["services"]
    capabilities = taiga_exporter._resolve_capabilities_for_problem(
        problem_dir,
        load_task_toml(problem_dir),
    )
    assert [(service["name"], service["role"]) for service in services] == [
        ("main", "main")
    ]
    assert services[0]["platform"] == "linux/amd64"
    assert capabilities.agent_service is not None
    assert capabilities.agent_service.compose["networks"] == {"alignerr-isolated": {}}


def test_taiga_implicit_main_preserves_none_network_policy(tmp_path: Path) -> None:
    task = _native_task()
    for section in ("services", "volumes", "captures", "artifacts", "mcp_servers"):
        task.pop(section, None)
    task["evaluation"] = {"engine": "rubric_task"}
    task["agent"]["resources"]["network"] = "none"
    problem_dir = _write_problem(tmp_path / "implicit-none", task)

    capabilities = taiga_exporter._resolve_capabilities_for_problem(
        problem_dir,
        load_task_toml(problem_dir),
    )

    assert capabilities.agent_service is not None
    assert capabilities.agent_service.compose["network_mode"] == "none"
    assert "networks" not in capabilities.agent_service.compose


def test_task_metadata_cannot_assert_outer_capsule_contract(tmp_path: Path) -> None:
    problem_dir = _write_problem(
        tmp_path / "metadata-contract",
        _native_task(metadata_contract=True),
    )

    with pytest.raises(ValueError, match="trusted export code"):
        build_job_payload(problem_dir, image_ref="LOCAL_IMAGE")


def test_capability_export_requires_outer_capsule_assertion(tmp_path: Path) -> None:
    problem_dir = _write_problem(tmp_path / "missing-contract", _native_task())

    with pytest.raises(ValueError, match="image_is_outer_capsule=True"):
        build_job_payload(problem_dir, image_ref="LOCAL_IMAGE")


def test_local_cli_requires_outer_capsule_flag_for_capability_export(
    tmp_path: Path,
) -> None:
    problem_dir = _write_problem(tmp_path / "cli-missing-contract", _native_task())
    output = tmp_path / "problems-metadata.json"

    result = CliRunner().invoke(
        local_app,
        [
            "export-taiga",
            "--problem-dir",
            str(problem_dir),
            "--out",
            str(output),
            "--image",
            "LOCAL_IMAGE",
        ],
    )

    assert result.exit_code != 0
    assert "trusted export code" in str(result.exception)
    assert not output.exists()


def test_local_cli_outer_capsule_flag_accepts_only_pinned_or_local_images(
    tmp_path: Path,
) -> None:
    problem_dir = _write_problem(tmp_path / "cli-outer-capsule", _native_task())
    runner = CliRunner()

    local_output = tmp_path / "local-metadata.json"
    local_result = runner.invoke(
        local_app,
        [
            "export-taiga",
            "--problem-dir",
            str(problem_dir),
            "--out",
            str(local_output),
            "--image",
            "LOCAL_IMAGE",
            "--outer-capsule",
        ],
    )
    assert local_result.exit_code == 0, str(local_result.exception)
    assert local_output.is_file()

    mutable_output = tmp_path / "mutable-metadata.json"
    mutable_result = runner.invoke(
        local_app,
        [
            "export-taiga",
            "--problem-dir",
            str(problem_dir),
            "--out",
            str(mutable_output),
            "--image",
            "example.invalid/task-capsule:latest",
            "--outer-capsule",
        ],
    )
    assert mutable_result.exit_code != 0
    assert "digest-pinned outer capsule" in str(mutable_result.exception)
    assert not mutable_output.exists()


def test_export_taiga_cli_options_document_trusted_capsule_contract() -> None:
    local_help = CliRunner().invoke(local_app, ["export-taiga", "--help"])
    assert local_help.exit_code == 0, local_help.output
    assert "--outer-capsule" in local_help.output
    assert "Trusted assertion" in local_help.output

    signature = inspect.signature(plugin_commands.export_taiga)
    outer_option = signature.parameters["outer_capsule"].default
    assert outer_option.default is False
    assert outer_option.param_decls == ("--outer-capsule",)
    assert "digest-pinned" in outer_option.help

    template_option = (
        inspect.signature(plugin_commands.create_problem).parameters["template"].default
    )
    assert "software-engineering" in template_option.help


def test_plugin_export_taiga_forwards_only_explicit_outer_capsule_assertion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_export_taiga(
        problem_dir: Path,
        output: Path,
        *,
        image_ref: str,
        image_is_outer_capsule: bool,
    ) -> dict[str, str]:
        calls.append(
            {
                "problem_dir": problem_dir,
                "output": output,
                "image_ref": image_ref,
                "image_is_outer_capsule": image_is_outer_capsule,
            }
        )
        return {"task_id": "fake"}

    monkeypatch.setattr(plugin_commands, "export_taiga_impl", fake_export_taiga)
    problem_dir = tmp_path / "problem"
    output = tmp_path / "metadata.json"

    plugin_commands.export_taiga(
        problem_dir,
        output,
        "LOCAL_IMAGE",
        False,
    )
    plugin_commands.export_taiga(
        problem_dir,
        output,
        _OUTER_IMAGE,
        True,
    )

    assert [call["image_is_outer_capsule"] for call in calls] == [False, True]
    assert [call["image_ref"] for call in calls] == ["LOCAL_IMAGE", _OUTER_IMAGE]


def test_capability_export_rejects_mutable_outer_image(tmp_path: Path) -> None:
    problem_dir = _write_problem(tmp_path / "mutable-outer", _native_task())

    with pytest.raises(ValueError, match="digest-pinned outer capsule"):
        build_job_payload(
            problem_dir,
            image_ref="example.invalid/task-capsule:latest",
            image_is_outer_capsule=True,
        )


def test_capability_export_allows_declared_sse_in_outer_capsule(
    tmp_path: Path,
) -> None:
    problem_dir = _write_problem(
        tmp_path / "sse",
        _native_task(mcp_transport="sse"),
    )

    payload = build_job_payload(
        problem_dir,
        image_ref="LOCAL_IMAGE",
        image_is_outer_capsule=True,
    )

    tools = _taiga_problem(payload)["extra_fields"]["task_metadata"][
        "capability_summary"
    ]["tools"]
    assert tools[-1] == {
        "access": "agent",
        "kind": "mcp",
        "name": "database-tools",
        "service": "database",
        "transport": "sse",
    }


def test_capability_export_rejects_unaudited_task_local_mcp_transport(
    tmp_path: Path,
) -> None:
    problem_dir = _write_problem(
        tmp_path / "stdio",
        _native_task(mcp_transport="stdio"),
    )

    with pytest.raises(ValueError, match="support only declared SSE"):
        build_job_payload(
            problem_dir,
            image_ref="LOCAL_IMAGE",
            image_is_outer_capsule=True,
        )


def test_capability_export_rejects_mutable_child_image(tmp_path: Path) -> None:
    task = _native_task()
    task["services"][1]["image"] = "example.invalid/database:latest"
    problem_dir = _write_problem(tmp_path / "mutable-child", task)

    with pytest.raises(ValueError, match="must be pinned"):
        build_job_payload(
            problem_dir,
            image_ref="LOCAL_IMAGE",
            image_is_outer_capsule=True,
        )


def test_capability_export_fails_when_peak_has_no_taiga_mapping(
    tmp_path: Path,
) -> None:
    task = _native_task()
    task["environment"]["required_resources"] = "16vcpu+128gib"
    task["agent"]["resources"]["cpus"] = 16
    problem_dir = _write_problem(tmp_path / "unmappable", task)

    with pytest.raises(ValueError, match="cannot map to any Taiga CPU enum"):
        build_job_payload(
            problem_dir,
            image_ref="LOCAL_IMAGE",
            image_is_outer_capsule=True,
        )


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> _FixedDateTime:
        return cls(2026, 1, 2, 3, 4, 5, tzinfo=tz or UTC)


@pytest.mark.parametrize(
    ("relative_path", "expected_sha256"),
    [
        (
            "alignerr_plugin/src/alignerr_plugin/starter_templates/ml",
            "96e8adbfd950b7a3ae38a559d552920c96f321de88f77bb1e2c1883349ef00d3",
        ),
        pytest.param(
            "examples/mujoco-pendulum",
            "01745b17207d6e811bd2e5b66a051d33b836820fe1d22b983ed84d076b8360f2",
            marks=requires_examples("mujoco-pendulum"),
        ),
        pytest.param(
            "examples/openfoam-hydrofoil-flap",
            "72e87a91000f68947b6807234fee4216049eae8d89fa2fc7a81f4c08d956e586",
            marks=requires_examples("openfoam-hydrofoil-flap"),
        ),
        pytest.param(
            "examples/opensees-base-isolation",
            "38ccfb772d1a1c8ea2fc97c9f3466ca4e259caddf1e92ff10416dd10de22df5c",
            marks=requires_examples("opensees-base-isolation"),
        ),
    ],
)
def test_legacy_payload_snapshot_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    expected_sha256: str,
) -> None:
    monkeypatch.delenv(taiga_exporter.QA_CPU_RESOURCE_ENV, raising=False)
    monkeypatch.delenv("LBX_TRUSTED_CALIBRATION_DIR", raising=False)
    monkeypatch.delenv("LBX_REQUIRE_TRUSTED_CONTINUOUS_EVALUATION", raising=False)
    monkeypatch.setattr(taiga_exporter, "datetime", _FixedDateTime)
    monkeypatch.setattr(taiga_exporter.time, "time", lambda: 1_767_323_045)
    monkeypatch.setattr(taiga_exporter, "BASE_IMAGE_TAG", "snapshot-base-tag")
    monkeypatch.setattr(taiga_exporter, "CPU_BASE_IMAGE_TAG", "snapshot-cpu-tag")
    monkeypatch.setattr(
        taiga_exporter,
        "_tpu_base_image_tag",
        lambda: "snapshot-tpu-tag",
    )
    monkeypatch.setattr(
        taiga_exporter,
        "_graphics_base_image_tag",
        lambda: "snapshot-graphics-tag",
    )
    monkeypatch.setattr(
        taiga_exporter,
        "_blackwell_base_image_tag",
        lambda: "snapshot-blackwell-tag",
    )

    payload = build_job_payload(ROOT / relative_path, image_ref="LOCAL_IMAGE")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert hashlib.sha256(encoded).hexdigest() == expected_sha256
