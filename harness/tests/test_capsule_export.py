from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tomllib
from pathlib import Path
from typing import Any

import pytest
import tomli_w
import yaml
from alignerr_plugin.capabilities import resolve_capabilities, resolve_services
from alignerr_plugin.capsule import (
    export_image_bundle,
    export_task_capsule,
    load_capabilities,
)
from alignerr_plugin.exporters.harbor import export_harbor
from alignerr_plugin.materialization import (
    WorkspaceSpec,
    materialize_workspace_seed,
    require_local_dockerfile_target,
    workspace_dockerfile_overlay,
)
from alignerr_plugin.schemas import TaskToml
from alignerr_plugin.validators.task.creator import TaskCreator

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64
_PINNED_DATABASE = f"example.invalid/database@{_DIGEST_B}"


def _base_task() -> dict[str, Any]:
    return {
        "schema_version": "1.2",
        "task": {
            "name": "labelbox/capsule-export-test",
            "description": "Capability export fixture.",
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
                "build_timeout_sec": 900,
            },
        },
        "verifier": {
            "timeout_sec": 600,
            "user": "root",
            "env": [],
            "resources": {
                "cpus": 2,
                "memory_mb": 2048,
                "storage_mb": 4096,
                "gpus": 0,
                "build_timeout_sec": 300,
            },
        },
        "difficulty": {
            "task_type": "mujoco",
            "domain": "model_environment_construction",
            "reward_type": "multi_deterministic_rubrics",
        },
    }


def _capability_task() -> dict[str, Any]:
    task = _base_task()
    task.update(
        {
            "workspace": {
                "seed": "starter",
                "root": "/workdir",
                "agent_cwd": "/workdir/src",
                "git_baseline": True,
            },
            "result": {
                "output_root": "/tmp/output",
                "reward_file": "grade.json",
                "reward_key": "score",
            },
            "volumes": [{"name": "database-data"}],
            "services": [
                {
                    "name": "main",
                    "role": "main",
                    "build": {
                        "context": "environment/main",
                        "dockerfile": "Dockerfile",
                        "args": {"APP_MODE": "benchmark"},
                        "platform": "linux/amd64",
                        "target": "runtime",
                        "pull": True,
                        "no_cache": True,
                    },
                    "user": "agent",
                    "env": {"DATABASE_HOST": "database"},
                    "depends_on": [{"service": "database", "condition": "healthy"}],
                },
                {
                    "name": "database",
                    "role": "sidecar",
                    "image": _PINNED_DATABASE,
                    "resources": {"platform": "linux/amd64"},
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
                        "pg_dump app > /tmp/database.sql.tmp",
                    ],
                    "timeout_sec": 45,
                    "destination": "/tmp/database.sql",
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
                    "exclude": ["__pycache__"],
                },
            ],
            "mcp_servers": [
                {
                    "name": "database-tools",
                    "transport": "sse",
                    "url": "http://database:8080/sse",
                    "service": "database",
                }
            ],
        }
    )
    return task


def _write_problem(problem_dir: Path, task: dict[str, Any]) -> Path:
    problem_dir.mkdir(parents=True)
    (problem_dir / "task.toml").write_text(tomli_w.dumps(task))
    (problem_dir / "instruction.md").write_text(
        "Repair the application and write results under /tmp/output.\n"
    )
    main = problem_dir / "environment" / "main"
    main.mkdir(parents=True)
    (main / "Dockerfile").write_text(
        "FROM python:3.13-slim AS runtime\nARG APP_MODE\nWORKDIR /workdir\n"
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
    (problem_dir / "private").mkdir()
    (problem_dir / "private" / "answer.txt").write_text("sealed\n")
    (problem_dir / "solution").mkdir()
    (problem_dir / "solution" / "solve.sh").write_text("#!/bin/sh\n")
    (problem_dir / "calibration.lock.json").write_text('{"locked": true}\n')
    return problem_dir


def _write_legacy_problem(problem_dir: Path) -> Path:
    task = _base_task()
    task["schema_version"] = "1.1"
    task.pop("agent")
    task.pop("verifier")
    problem_dir.mkdir(parents=True)
    (problem_dir / "task.toml").write_text(tomli_w.dumps(task))
    (problem_dir / "instruction.md").write_text("Write /tmp/output/result.txt.\n")
    (problem_dir / "environment").mkdir()
    (problem_dir / "environment" / "Dockerfile").write_text(
        "FROM python:3.13-slim\nWORKDIR /workdir\n"
    )
    (problem_dir / "scorer" / "data").mkdir(parents=True)
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n    return 1.0\n"
    )
    (problem_dir / "data").mkdir()
    return problem_dir


class FakeDocker:
    def __init__(self, *, reverse_tar_order: bool = False) -> None:
        self.commands: list[list[str]] = []
        self.reverse_tar_order = reverse_tar_order

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
        self.commands.append(command)
        stdout = ""
        if command[1:3] == ["image", "inspect"]:
            image_ref = command[-1]
            image_id = _DIGEST_C if "database@" in image_ref else _DIGEST_A
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
            members = [
                ("manifest.json", json.dumps(sorted(refs)).encode()),
                ("repositories", b"{}"),
            ]
            if self.reverse_tar_order:
                members.reverse()
            with tarfile.open(output_path, "w") as archive:
                for index, (name, content) in enumerate(members):
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    info.mtime = 123_456 + index
                    info.uid = 501
                    info.gid = 20
                    archive.addfile(info, io.BytesIO(content))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_unpinned_external_image_is_rejected_before_docker(tmp_path: Path) -> None:
    task = _base_task()
    task["services"] = [{"name": "main", "role": "main", "image": "python:3.13-slim"}]
    problem_dir = _write_problem(tmp_path / "problem", task)
    services = resolve_services(
        {"services": [{"name": "main", "role": "main", "image": "python:3.13-slim"}]}
    )
    docker = FakeDocker()

    with pytest.raises(ValueError, match="must be pinned"):
        export_image_bundle(
            problem_dir,
            services,
            tmp_path / "bundle",
            trusted_build=True,
            runner=docker,
        )

    assert docker.commands == []


def test_dockerfile_target_must_name_a_local_stage() -> None:
    dockerfile = (
        "FROM --platform=linux/amd64 python:3.13 AS builder\n"
        "FROM builder AS runtime\n"
    )

    assert (
        require_local_dockerfile_target(
            dockerfile,
            "RUNTIME",
            label="test build",
        )
        == "runtime"
    )
    with pytest.raises(ValueError, match="not a declared local Dockerfile stage"):
        require_local_dockerfile_target(
            dockerfile,
            "missing",
            label="test build",
        )


def test_capsule_rejects_missing_target_before_docker(tmp_path: Path) -> None:
    task = _capability_task()
    task["services"][0]["build"]["target"] = "missing"
    problem_dir = _write_problem(tmp_path / "problem", task)
    services = resolve_capabilities(TaskToml.model_validate(task)).services
    docker = FakeDocker()

    with pytest.raises(ValueError, match="not a declared local Dockerfile stage"):
        export_image_bundle(
            problem_dir,
            services,
            tmp_path / "bundle",
            trusted_build=True,
            runner=docker,
        )

    assert docker.commands == []


def test_harbor_rejects_missing_agent_target(tmp_path: Path) -> None:
    task = _capability_task()
    task["services"][0]["build"]["target"] = "missing"
    problem_dir = _write_problem(tmp_path / "problem", task)

    with pytest.raises(ValueError, match="not a declared local Dockerfile stage"):
        export_harbor(problem_dir, tmp_path / "harbor")


def test_child_images_require_explicit_trusted_build_phase(tmp_path: Path) -> None:
    task = _base_task()
    task["services"] = [
        {
            "name": "main",
            "role": "main",
            "image": f"example.invalid/main@{_DIGEST_A}",
            "resources": {"platform": "linux/amd64"},
        }
    ]
    problem_dir = _write_problem(tmp_path / "problem", task)
    services = resolve_services(task)
    docker = FakeDocker()

    with pytest.raises(PermissionError, match="trusted build phase"):
        export_image_bundle(
            problem_dir,
            services,
            tmp_path / "bundle",
            trusted_build=False,
            runner=docker,
        )

    assert docker.commands == []


def test_service_resolution_accepts_typed_schema_models() -> None:
    task_data = _capability_task()
    task_data["workspace"] = {
        "seed": "starter",
        "root": "/workdir",
        "agent_cwd": "/workdir/src",
    }
    task = TaskToml.model_validate(task_data)

    capabilities = resolve_capabilities(task)

    assert [(service.name, service.role) for service in capabilities.services] == [
        ("main", "agent"),
        ("database", "sidecar"),
    ]
    assert capabilities.agent_resources["cpus"] == 4
    assert capabilities.verifier_resources["cpus"] == 2
    assert capabilities.mcp_servers[0]["url"] == "http://database:8080/sse"


def test_workspace_commit_baseline_is_verified_and_agent_cwd_is_created() -> None:
    commit = "d" * 40
    dockerfile = workspace_dockerfile_overlay(
        WorkspaceSpec(
            root="/workspace",
            seed="starter",
            agent_cwd="/workspace/repo",
            git_baseline=commit,
        ),
        user="agent",
    )

    assert 'test "$(git -C /workspace rev-parse --verify HEAD^{commit})"' in dockerfile
    assert commit in dockerfile
    assert f"update-ref {commit}" not in dockerfile
    assert "mkdir -p /workspace /workspace/repo" in dockerfile
    assert "command -v git" in dockerfile
    assert "apt-get install -y --no-install-recommends git" in dockerfile
    assert dockerfile.index("mkdir -p") < dockerfile.index("chown -R agent:agent")


def _write_git_workspace_seed(task_root: Path) -> tuple[Path, str]:
    seed = task_root / "starter"
    seed.mkdir(parents=True, exist_ok=True)
    (seed / "app.py").write_text("print('pinned seed')\n")
    subprocess.run(["git", "init", "-q", str(seed)], check=True)
    subprocess.run(["git", "-C", str(seed), "add", "."], check=True)
    environment = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
    }
    subprocess.run(
        [
            "git",
            "-C",
            str(seed),
            "-c",
            "user.name=Alignerr",
            "-c",
            "user.email=alignerr@local",
            "commit",
            "-q",
            "-m",
            "seed",
        ],
        check=True,
        env=environment,
    )
    commit = subprocess.run(
        ["git", "-C", str(seed), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return seed, commit


def _materialized_tree(root: Path) -> list[tuple[str, bytes, int, int]]:
    return [
        (
            path.relative_to(root).as_posix(),
            path.read_bytes() if path.is_file() else b"",
            stat.S_IMODE(path.stat().st_mode),
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
    ]


def test_commit_pinned_workspace_survives_deterministic_materialization(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "problem"
    _, commit = _write_git_workspace_seed(task_root)
    workspace = WorkspaceSpec(
        root="/workspace",
        seed="starter",
        agent_cwd="/workspace",
        git_baseline=commit,
        init_policy="copy",
    )
    first = tmp_path / "first" / ".alignerr-workspace-seed"
    second = tmp_path / "second" / ".alignerr-workspace-seed"

    materialize_workspace_seed(task_root, first, workspace)
    materialize_workspace_seed(task_root, second, workspace)

    for destination in (first, second):
        resolved = subprocess.run(
            [
                "git",
                "-C",
                str(destination),
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert resolved == commit
        assert not (destination / ".git" / "config").exists()
        assert not (destination / ".git" / "hooks").exists()
    assert _materialized_tree(first) == _materialized_tree(second)
    assert (
        "!.alignerr-workspace-seed/.git/**"
        in (first.parent / ".dockerignore").read_text()
    )


@pytest.mark.parametrize("git_baseline", [False, True])
def test_regular_workspace_seed_is_exempted_from_dockerignore(
    tmp_path: Path,
    git_baseline: bool,
) -> None:
    task_root = tmp_path / "problem"
    seed = task_root / "starter"
    seed.mkdir(parents=True)
    (seed / "app.py").write_text("print('regular seed')\n")
    context = tmp_path / "context"
    context.mkdir()
    (context / ".dockerignore").write_text(".*\n")
    destination = context / ".alignerr-workspace-seed"
    workspace = WorkspaceSpec(
        root="/workspace",
        seed="starter",
        agent_cwd="/workspace",
        git_baseline=git_baseline,
        init_policy="copy",
    )

    materialize_workspace_seed(task_root, destination, workspace)

    assert (destination / "app.py").read_text() == "print('regular seed')\n"
    dockerignore = (context / ".dockerignore").read_text()
    assert "!.alignerr-workspace-seed/\n" in dockerignore
    assert "!.alignerr-workspace-seed/.git/**\n" in dockerignore


@pytest.mark.parametrize(
    "attack",
    ["gitdir", "symlink", "alternates", "hook", "config"],
)
def test_commit_pinned_workspace_rejects_malicious_git_metadata(
    tmp_path: Path,
    attack: str,
) -> None:
    task_root = tmp_path / "problem"
    seed, commit = _write_git_workspace_seed(task_root)
    git_dir = seed / ".git"
    if attack == "gitdir":
        shutil.rmtree(git_dir)
        git_dir.write_text("gitdir: ../../../outside.git\n")
    elif attack == "symlink":
        target = git_dir / "safe-target"
        target.write_text("data\n")
        (git_dir / "linked").symlink_to(target.name)
    elif attack == "alternates":
        info = git_dir / "objects" / "info"
        info.mkdir(exist_ok=True)
        (info / "alternates").write_text("../../../../outside.git/objects\n")
    elif attack == "hook":
        hook = git_dir / "hooks" / "post-checkout"
        hook.write_text("#!/bin/sh\nexec /tmp/payload\n")
        hook.chmod(0o755)
    else:
        with (git_dir / "config").open("a") as config:
            config.write("[core]\n\thooksPath = ../../../outside-hooks\n")

    workspace = WorkspaceSpec(
        seed="starter",
        git_baseline=commit,
        init_policy="overlay",
    )
    with pytest.raises(ValueError, match=r"\.git|gitdir|alternates|hook"):
        materialize_workspace_seed(task_root, tmp_path / "output", workspace)


def test_image_bundle_manifest_and_archive_are_deterministic(tmp_path: Path) -> None:
    problem_dir = _write_problem(tmp_path / "problem", _capability_task())
    services = resolve_capabilities(_capability_task()).services

    first = export_image_bundle(
        problem_dir,
        services,
        tmp_path / "first",
        trusted_build=True,
        runner=FakeDocker(),
    )
    second = export_image_bundle(
        problem_dir,
        services,
        tmp_path / "second",
        trusted_build=True,
        runner=FakeDocker(reverse_tar_order=True),
    )

    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert first.manifest == second.manifest
    assert [row["service"] for row in first.manifest["images"]] == [
        "main",
        "database",
    ]
    assert {row["role"] for row in first.manifest["images"]} == {"main", "sidecar"}
    assert all(
        row["image_digest"].startswith("sha256:") for row in first.manifest["images"]
    )
    database = next(
        row for row in first.manifest["images"] if row["service"] == "database"
    )
    assert database["image_digest"] == _DIGEST_C
    assert database["source_digest"] == _DIGEST_B
    assert database["platform"] == "linux/amd64"
    assert database["os"] == "linux"
    assert database["architecture"] == "amd64"
    assert {row["image_digest"] for row in first.manifest["images"]} == {
        _DIGEST_A,
        _DIGEST_C,
    }


def test_capability_harbor_export_projects_native_fields(tmp_path: Path) -> None:
    problem_dir = _write_problem(tmp_path / "problem", _capability_task())
    output_dir = tmp_path / "harbor"

    export_harbor(problem_dir, output_dir)

    task = tomllib.loads((output_dir / "task.toml").read_text())
    compose = yaml.safe_load(
        (output_dir / "environment" / "docker-compose.yaml").read_text()
    )

    assert task["schema_version"] == "1.4"
    assert task["agent"]["user"] == "agent"
    assert task["verifier"]["user"] == "root"
    assert task["verifier"]["env"] == {}
    assert task["environment"]["cpus"] == 4
    assert task["environment"]["memory_mb"] == 4096
    assert task["environment"]["storage_mb"] == 10_000
    assert task["environment"]["network_mode"] == "no-network"
    assert task["verifier"]["environment_mode"] == "separate"
    assert task["verifier"]["environment"]["cpus"] == 2
    assert task["verifier"]["environment"]["memory_mb"] == 2048
    assert task["verifier"]["environment"]["network_mode"] == "no-network"
    assert task["environment"]["mcp_servers"] == [
        {
            "name": "database-tools",
            "transport": "sse",
            "url": "http://database:8080/sse",
        }
    ]
    assert task["artifacts"] == [
        {
            "source": "/tmp/database.sql",
            "destination": "database.sql",
            "service": "database",
        },
        {
            "source": "/workdir/src",
            "destination": "sources/src",
            "exclude": ["__pycache__"],
            "service": "main",
        },
        {
            "source": "/workdir/pyproject.toml",
            "destination": "sources/pyproject.toml",
            "exclude": ["__pycache__"],
            "service": "main",
        },
    ]
    capture = task["verifier"]["collect"][0]
    assert capture["service"] == "database"
    assert "pg_dump app" in capture["command"]
    assert "mv /tmp/database.sql.tmp /tmp/database.sql" in capture["command"]
    assert capture["timeout_sec"] == 45.0

    assert set(compose["services"]) == {"main", "database"}
    assert compose["services"]["database"]["image"] == _PINNED_DATABASE
    assert compose["services"]["database"]["expose"] == ["5432"]
    assert compose["services"]["main"]["depends_on"] == {
        "database": {"condition": "service_healthy"}
    }
    assert compose["services"]["main"]["platform"] == "linux/amd64"
    assert compose["volumes"] == {"database-data": {}}
    assert (output_dir / "environment" / "Dockerfile").is_file()
    assert (output_dir / "tests" / "Dockerfile").is_file()
    assert (output_dir / "tests" / "test.sh").is_file()
    agent_dockerfile = (output_dir / "environment" / "Dockerfile").read_text()
    verifier_dockerfile = (output_dir / "tests" / "Dockerfile").read_text()
    assert "COPY .alignerr-workspace-seed/ /workdir/" in agent_dockerfile
    assert agent_dockerfile.rstrip().endswith("USER agent")
    assert "COPY .alignerr-workspace-seed/ /workdir/" in verifier_dockerfile
    assert "RUN mkdir -p -- /tmp /tmp/output /workdir" in verifier_dockerfile
    assert verifier_dockerfile.rstrip().endswith("USER root")


def test_harbor_preserves_compose_build_options_at_valid_locations(
    tmp_path: Path,
) -> None:
    task = _capability_task()
    task["services"].append(
        {
            "name": "worker",
            "role": "sidecar",
            "build": {
                "context": "environment/worker",
                "dockerfile": "Containerfile",
                "args": {"FEATURE": "enabled"},
                "target": "worker-runtime",
                "platform": "linux/arm64",
                "pull": True,
                "no_cache": True,
            },
        }
    )
    problem_dir = _write_problem(tmp_path / "problem", task)
    worker = problem_dir / "environment" / "worker"
    worker.mkdir()
    (worker / "Containerfile").write_text(
        'FROM alpine:3.22 AS worker-runtime\nCMD ["sleep", "infinity"]\n'
    )

    output_dir = tmp_path / "harbor"
    export_harbor(problem_dir, output_dir)
    compose = yaml.safe_load(
        (output_dir / "environment" / "docker-compose.yaml").read_text()
    )
    worker_service = compose["services"]["worker"]

    assert worker_service["platform"] == "linux/arm64"
    assert worker_service["build"] == {
        "context": "./services/worker",
        "dockerfile": "Containerfile",
        "args": {"FEATURE": "enabled"},
        "target": "worker-runtime",
        "pull": True,
        "no_cache": True,
    }


def test_harbor_compose_remaps_shared_agent_network_name(tmp_path: Path) -> None:
    task = _capability_task()
    task["services"][0]["name"] = "workspace-agent"
    task["artifacts"][1]["service"] = "workspace-agent"
    task["services"].append(
        {
            "name": "observer",
            "role": "sidecar",
            "image": f"example.invalid/observer@{_DIGEST_C}",
            "network_mode": "share",
            "network_share_target": "workspace-agent",
        }
    )
    problem_dir = _write_problem(tmp_path / "problem", task)
    output_dir = tmp_path / "harbor"

    export_harbor(problem_dir, output_dir)

    compose = yaml.safe_load(
        (output_dir / "environment" / "docker-compose.yaml").read_text()
    )
    assert compose["services"]["observer"]["network_mode"] == "service:main"


def test_export_rejects_task_symlink_escape_without_publishing(
    tmp_path: Path,
) -> None:
    problem_dir = _write_problem(tmp_path / "problem", _capability_task())
    outside = tmp_path / "host-secret.txt"
    outside.write_text("do not copy\n")
    (problem_dir / "starter" / "escape").symlink_to(outside)
    output_dir = tmp_path / "harbor"

    with pytest.raises(ValueError, match="symlink .* escapes task root"):
        export_harbor(problem_dir, output_dir)

    assert not output_dir.exists()


def test_capability_context_materializes_all_task_inputs_and_force_is_explicit(
    tmp_path: Path,
) -> None:
    problem_dir = _write_problem(tmp_path / "problem", _capability_task())
    output_dir = tmp_path / "harbor"
    export_harbor(problem_dir, output_dir)

    for relative in (
        "starter/src/app.py",
        "solution/solve.sh",
        "scorer/compute_score.py",
        "scorer/data/private.json",
        "private/answer.txt",
        "data/public.txt",
        "calibration.lock.json",
    ):
        assert (output_dir / "_task_inputs" / relative).is_file()

    marker = output_dir / "stale.txt"
    marker.write_text("stale\n")
    with pytest.raises(FileExistsError, match="force=True"):
        export_harbor(problem_dir, output_dir)
    assert marker.is_file()

    export_harbor(problem_dir, output_dir, force=True)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("privileged", True),
        ("devices", ["/dev/disk0"]),
        ("volumes_from", ["host-service"]),
        ("capabilities", ["SYS_ADMIN"]),
        ("volumes", ["/host:/container"]),
    ],
)
def test_export_rejects_unsafe_service_projection_before_publish(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    task = _capability_task()
    task["services"][0][field] = value
    problem_dir = _write_problem(tmp_path / "problem", task)
    output_dir = tmp_path / "harbor"

    with pytest.raises(ValueError):
        export_harbor(problem_dir, output_dir)

    assert not output_dir.exists()


def test_export_rejects_verifier_dependency_and_shared_volume(
    tmp_path: Path,
) -> None:
    task = _base_task()
    task["volumes"] = [{"name": "shared-state"}]
    task["result"] = {
        "output_root": "/tmp/output/verifier",
        "reward_file": "grade.json",
        "reward_key": "score",
    }
    task["services"] = [
        {
            "name": "main",
            "role": "main",
            "image": f"example.invalid/main@{_DIGEST_A}",
            "user": "agent",
            "resources": {"platform": "linux/amd64"},
            "volumes": [
                {"volume": "shared-state", "target": "/workspace", "mode": "rw"}
            ],
        },
        {
            "name": "verifier",
            "role": "verifier",
            "image": f"example.invalid/verifier@{_DIGEST_C}",
            "resources": {"platform": "linux/amd64"},
            "depends_on": [{"service": "main"}],
            "volumes": [{"volume": "shared-state", "target": "/results", "mode": "ro"}],
        },
    ]
    problem_dir = _write_problem(tmp_path / "problem", task)

    with pytest.raises(ValueError, match="verifier service.*must not depend"):
        export_harbor(problem_dir, tmp_path / "harbor")


def test_export_rejects_main_to_verifier_dependency_and_shared_volume(
    tmp_path: Path,
) -> None:
    task = _base_task()
    task["result"] = {
        "output_root": "/tmp/output/verifier",
        "reward_file": "grade.json",
        "reward_key": "score",
    }
    task["services"] = [
        {
            "name": "main",
            "role": "main",
            "image": f"example.invalid/main@{_DIGEST_A}",
            "user": "agent",
            "resources": {"platform": "linux/amd64"},
            "depends_on": [{"service": "verifier"}],
        },
        {
            "name": "verifier",
            "role": "verifier",
            "image": f"example.invalid/verifier@{_DIGEST_C}",
            "resources": {"platform": "linux/amd64"},
        },
    ]
    problem_dir = _write_problem(tmp_path / "main-dependency", task)
    with pytest.raises(ValueError, match="must not depend on verifier"):
        export_harbor(problem_dir, tmp_path / "main-dependency-export")

    task["evaluation"] = {
        "engine": "rubric_task",
        "metadata": {"trusted_post_agent_verifier": True},
    }
    task["volumes"] = [{"name": "shared-state"}]
    task["services"][0].pop("depends_on")
    task["services"][0]["volumes"] = [
        {"volume": "shared-state", "target": "/workspace", "mode": "rw"}
    ]
    task["services"][1]["depends_on"] = [{"service": "main"}]
    task["services"][1]["volumes"] = [
        {"volume": "shared-state", "target": "/results", "mode": "ro"}
    ]
    problem_dir = _write_problem(tmp_path / "shared-volume", task)
    with pytest.raises(ValueError, match="named volumes cannot be shared"):
        export_harbor(problem_dir, tmp_path / "shared-volume-export")


def test_staged_export_rejects_source_nesting_symlinks_and_unowned_force(
    tmp_path: Path,
) -> None:
    problem_dir = _write_legacy_problem(tmp_path / "problem")
    with pytest.raises(ValueError, match="inside task source tree"):
        export_harbor(problem_dir, problem_dir / "export")

    real_output = tmp_path / "real-output"
    real_output.mkdir()
    symlink_output = tmp_path / "linked-output"
    symlink_output.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        export_harbor(problem_dir, symlink_output)

    unowned = tmp_path / "unowned"
    unowned.mkdir()
    with pytest.raises(ValueError, match="created by this exporter"):
        export_harbor(problem_dir, unowned, force=True)


def test_capability_image_stamping_requires_and_preserves_digest(
    tmp_path: Path,
) -> None:
    problem_dir = _write_problem(tmp_path / "problem", _capability_task())
    scorer = problem_dir / "scorer" / "compute_score.py"
    scorer.write_text(scorer.read_text() + "\n# llm_criterion\n")
    with pytest.raises(ValueError, match="immutable"):
        export_harbor(problem_dir, tmp_path / "bad", image_ref="example.invalid:latest")

    image = f"example.invalid/export@{_DIGEST_A}"
    output_dir = tmp_path / "harbor"
    export_harbor(problem_dir, output_dir, image_ref=image)
    exported = tomllib.loads((output_dir / "task.toml").read_text())
    assert exported["environment"]["docker_image"] == image
    assert exported["verifier"]["env"] == {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}
    assert tomllib.loads((output_dir / "tests" / "task.toml").read_text()) == exported
    assert (
        tomllib.loads((output_dir / "tests" / "task" / "task.toml").read_text())
        == exported
    )


def test_legacy_harbor_ignores_short_image_ref(tmp_path: Path) -> None:
    problem_dir = _write_legacy_problem(tmp_path / "problem")
    output_dir = tmp_path / "harbor"

    export_harbor(
        problem_dir,
        output_dir,
        image_ref="gcr.io/example/legacy@sha256:abc123",
    )

    exported = tomllib.loads((output_dir / "task.toml").read_text())
    assert "docker_image" not in exported["environment"]


def test_isolated_service_resources_create_internal_compose_network(
    tmp_path: Path,
) -> None:
    task = _capability_task()
    task["services"][0]["resources"] = {"network": "none"}
    task["services"][1]["resources"]["network"] = "isolated"
    problem_dir = _write_problem(tmp_path / "problem", task)
    output_dir = tmp_path / "harbor"

    export_harbor(problem_dir, output_dir)

    compose = yaml.safe_load(
        (output_dir / "environment" / "docker-compose.yaml").read_text()
    )
    assert compose["networks"]["alignerr-isolated"] == {"internal": True}
    assert "alignerr-isolated" in compose["services"]["database"]["networks"]
    assert compose["services"]["main"]["network_mode"] == "none"


def test_generated_compose_is_accepted_by_docker_when_available(
    tmp_path: Path,
) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable")
    version = subprocess.run(
        [docker, "compose", "version"],
        text=True,
        capture_output=True,
        check=False,
    )
    if version.returncode != 0:
        pytest.skip("Docker Compose plugin is unavailable")

    problem_dir = _write_problem(tmp_path / "problem", _capability_task())
    output_dir = tmp_path / "harbor"
    export_harbor(problem_dir, output_dir)
    subprocess.run(
        [
            docker,
            "compose",
            "--file",
            str(output_dir / "environment" / "docker-compose.yaml"),
            "config",
            "--quiet",
        ],
        text=True,
        capture_output=True,
        check=True,
    )


def test_outer_capsule_context_embeds_bundle_and_existing_rubric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem_dir = _write_problem(tmp_path / "problem", _capability_task())
    _, commit = _write_git_workspace_seed(problem_dir)
    task_data = tomllib.loads((problem_dir / "task.toml").read_text())
    task_data["workspace"]["git_baseline"] = commit
    (problem_dir / "task.toml").write_text(tomli_w.dumps(task_data))
    output_dir = tmp_path / "capsule"

    result = export_task_capsule(
        problem_dir,
        output_dir,
        trusted_build=True,
        runner=FakeDocker(),
    )

    dockerfile = result.dockerfile.read_text()
    manifest = json.loads(result.capsule_manifest.read_text())
    compose = yaml.safe_load(result.compose_file.read_text())
    installer = (output_dir / "install-nested-docker.sh").read_text()
    service_runtime = (
        output_dir
        / "taiga_runtime"
        / "rubric"
        / "src"
        / "rubric"
        / "service_runtime.py"
    ).read_text()
    materialized_task_toml = output_dir / "task" / "task.toml"
    materialized_task = tomllib.loads(materialized_task_toml.read_text())

    assert "COPY taiga_runtime/rubric/ /mcp_server/" in dockerfile
    assert "COPY docker-compose.yaml manifest.json /task/capsule/" in dockerfile
    assert (
        "COPY --chown=root:root task/.alignerr-workspace-seed/ "
        "/task/.alignerr-workspace-seed/" in dockerfile
    )
    assert 'CMD ["/opt/lbx-runtime/.venv/bin/rubric", "mcp"]' in dockerfile
    assert manifest["compose_file"] == "docker-compose.yaml"
    assert manifest["workspace"]["seed"] == ".alignerr-workspace-seed"
    assert materialized_task["workspace"]["seed"] == ".alignerr-workspace-seed"
    authored_task = tomllib.loads((problem_dir / "task.toml").read_text())
    assert authored_task["workspace"]["seed"] == "starter"
    assert [image["service"] for image in manifest["images"]] == [
        "main",
        "database",
    ]
    assert all(image["archive_sha256"] for image in manifest["images"])
    assert set(compose["services"]) == {"main", "database"}
    for service in compose["services"].values():
        assert service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]
    assert result.image_bundle.archive_path.is_file()
    assert result.image_bundle.manifest_path.is_file()
    for relative in (
        "task/starter/src/app.py",
        "task/solution/solve.sh",
        "task/scorer/data/private.json",
        "task/private/answer.txt",
        "task/calibration.lock.json",
    ):
        assert (output_dir / relative).is_file()
    assert "@sha256:" in dockerfile
    assert "uv:latest" not in dockerfile
    assert "docker-compose-plugin" not in installer
    assert "docker-compose-linux-${compose_arch}" in installer
    assert 'test "$(docker compose version --short)" = "2.29.7"' in installer
    assert '"--no-deps"' in service_runtime
    assert "verifier-wrapper.sh" in service_runtime
    runtime_seed = output_dir / "task" / ".alignerr-workspace-seed"
    assert not (output_dir / "task" / "starter" / ".git").exists()
    monkeypatch.syspath_prepend(str(output_dir / "grader" / "src"))
    monkeypatch.syspath_prepend(str(output_dir / "taiga_runtime" / "rubric" / "src"))
    from rubric.service_config import load_task_service_config
    from rubric.service_runtime import TaskServiceRuntime

    runtime_config = load_task_service_config(materialized_task_toml)
    assert runtime_config is not None
    assert runtime_config.workspace is not None
    assert runtime_config.workspace.seed == ".alignerr-workspace-seed"
    runtime = TaskServiceRuntime.__new__(TaskServiceRuntime)
    runtime.config = runtime_config
    archive = runtime._workspace_seed_archive(runtime_config.workspace.seed)
    archived_seed = tmp_path / "runtime-archived-seed"
    archived_seed.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        handle.extractall(archived_seed, filter="data")
    assert (archived_seed / ".git").is_dir()
    assert (
        subprocess.run(
            [
                "git",
                "-C",
                str(archived_seed),
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == commit
    )
    assert (
        subprocess.run(
            ["git", "-C", str(runtime_seed), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == commit
    )
    subprocess.run(
        ["git", "-C", str(runtime_seed), "fsck", "--strict", "--no-dangling"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(runtime_seed), "cat-file", "-e", f"{commit}^{{tree}}"],
        check=True,
    )
    status = subprocess.run(
        ["git", "-C", str(runtime_seed), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""
    checkout = tmp_path / "runtime-checkout"
    checkout.mkdir()
    subprocess.run(
        [
            "git",
            "-C",
            str(runtime_seed),
            f"--work-tree={checkout}",
            "checkout",
            "--force",
            commit,
            "--",
            ".",
        ],
        check=True,
    )
    assert (checkout / "app.py").read_text() == "print('pinned seed')\n"


def test_outer_capsule_rewrites_regular_workspace_seed_for_runtime(
    tmp_path: Path,
) -> None:
    problem_dir = _write_problem(tmp_path / "problem", _capability_task())

    result = export_task_capsule(
        problem_dir,
        tmp_path / "capsule",
        trusted_build=True,
        runner=FakeDocker(),
    )

    materialized = tomllib.loads(
        (result.context_dir / "task" / "task.toml").read_text()
    )
    assert materialized["workspace"]["seed"] == ".alignerr-workspace-seed"
    assert (
        result.context_dir / "task" / ".alignerr-workspace-seed" / "src" / "app.py"
    ).is_file()
    manifest = json.loads(result.capsule_manifest.read_text())
    assert manifest["workspace"]["seed"] == ".alignerr-workspace-seed"


def test_evaluation_only_capability_generates_implicit_main_and_verifier(
    tmp_path: Path,
) -> None:
    task = _base_task()
    task["evaluation"] = {"engine": "rubric_task"}
    task["agent"]["resources"]["platform"] = "linux/amd64"
    problem_dir = _write_problem(tmp_path / "problem", task)
    output_dir = tmp_path / "harbor"

    capabilities = load_capabilities(problem_dir)
    export_harbor(problem_dir, output_dir)

    assert capabilities.agent_service is not None
    assert capabilities.agent_service.build is not None
    assert capabilities.agent_service.build.context == "environment/main"
    assert capabilities.agent_service.build.platform == "linux/amd64"
    assert capabilities.agent_service.compose["networks"] == {"alignerr-isolated": {}}
    compose = yaml.safe_load(
        (output_dir / "environment" / "docker-compose.yaml").read_text()
    )
    assert set(compose["services"]) == {"main"}
    assert compose["services"]["main"]["networks"] == {"alignerr-isolated": {}}
    assert (output_dir / "tests" / "Dockerfile").is_file()
    exported = tomllib.loads((output_dir / "task.toml").read_text())
    assert exported["agent"]["user"] == "agent"
    assert exported["verifier"]["user"] == "root"


@pytest.mark.parametrize(
    ("network", "expected"),
    [
        ("none", {"network_mode": "none"}),
        ("internet", {}),
    ],
)
def test_implicit_agent_preserves_authored_network_policy(
    tmp_path: Path,
    network: str,
    expected: dict[str, object],
) -> None:
    task = _base_task()
    task["evaluation"] = {"engine": "rubric_task"}
    task["agent"]["resources"]["network"] = network
    problem_dir = _write_problem(tmp_path / network, task)
    output_dir = tmp_path / f"harbor-{network}"

    capabilities = load_capabilities(problem_dir)
    export_harbor(problem_dir, output_dir)

    assert capabilities.agent_service is not None
    compose = capabilities.agent_service.compose
    exported = yaml.safe_load(
        (output_dir / "environment" / "docker-compose.yaml").read_text()
    )["services"]["main"]
    if expected:
        assert {key: compose[key] for key in expected} == expected
        assert {key: exported[key] for key in expected} == expected
    else:
        assert "network_mode" not in compose and "networks" not in compose
        assert "network_mode" not in exported and "networks" not in exported


def test_verifier_export_preserves_workdir_and_prepares_result_parents(
    tmp_path: Path,
) -> None:
    task = _capability_task()
    task["services"].append(
        {
            "name": "verifier",
            "role": "verifier",
            "build": {
                "context": "environment/verifier",
                "platform": "linux/amd64",
            },
        }
    )
    task["result"] = {
        "output_root": "/tmp/output/verifier",
        "reward_file": "results/grade.json",
        "reward_key": "score",
    }
    problem_dir = _write_problem(tmp_path / "problem", task)
    verifier_context = problem_dir / "environment" / "verifier"
    verifier_context.mkdir()
    (verifier_context / "Dockerfile").write_text(
        "FROM python:3.13-slim\nWORKDIR /authored-verifier\n"
    )

    output_dir = tmp_path / "harbor"
    export_harbor(problem_dir, output_dir)

    dockerfile = (output_dir / "tests" / "Dockerfile").read_text()
    assert "WORKDIR /authored-verifier" in dockerfile
    assert dockerfile.count("WORKDIR ") == 1
    assert "/tmp/output/verifier/results" in dockerfile

    capsule = export_task_capsule(
        problem_dir,
        tmp_path / "capsule",
        trusted_build=True,
        runner=FakeDocker(),
    )
    capsule_compose = yaml.safe_load(capsule.compose_file.read_text())
    assert capsule_compose["services"]["verifier"]["network_mode"] == "none"
    assert "networks" not in capsule_compose["services"]["verifier"]


def test_legacy_harbor_export_snapshot_is_unchanged(tmp_path: Path) -> None:
    problem_dir = _write_legacy_problem(tmp_path / "legacy")
    source_task = tomllib.loads((problem_dir / "task.toml").read_text())
    output_dir = tmp_path / "harbor"

    export_harbor(problem_dir, output_dir)

    snapshot = {
        "top_level": sorted(path.name for path in output_dir.iterdir()),
        "tests": sorted(path.name for path in (output_dir / "tests").iterdir()),
        "has_compose": (output_dir / "environment" / "docker-compose.yaml").exists(),
        "has_tests_dockerfile": (output_dir / "tests" / "Dockerfile").exists(),
        "test_sh_sha256": hashlib.sha256(
            (output_dir / "tests" / "test.sh").read_bytes()
        ).hexdigest(),
        "dockerfile_sha256": hashlib.sha256(
            (output_dir / "environment" / "Dockerfile").read_bytes()
        ).hexdigest(),
    }
    assert snapshot == {
        "top_level": [
            ".alignerr-export-owned",
            "environment",
            "instruction.md",
            "task.toml",
            "tests",
        ],
        "tests": ["test.sh"],
        "has_compose": False,
        "has_tests_dockerfile": False,
        "test_sh_sha256": "a8876160fef935c669637142d0ecfc7ac8d5224e131f8ddb3b7040570ff152b0",
        "dockerfile_sha256": "09d44cdadbcc98c2395310bced9b2aa1bf49e93bebf3da42f6440cdbe1d99444",
    }
    assert tomllib.loads((output_dir / "task.toml").read_text()) == {
        **source_task,
        "metadata": {},
    }


def test_legacy_harbor_preserves_solver_security_and_calibration_variants(
    tmp_path: Path,
) -> None:
    problem_dir = _write_legacy_problem(tmp_path / "legacy")
    task_path = problem_dir / "task.toml"
    task = tomllib.loads(task_path.read_text())
    task["difficulty"]["task_type"] = "cfd"
    task["delivery"] = {"platform": "prometheus"}
    task["environment"]["required_resources"] = "24vcpu+200gib+h100/1"
    task["difficulty"]["domain"] = "aerodynamics"
    task_path.write_text(tomli_w.dumps(task))
    (problem_dir / "calibration.lock.json").write_text('{"locked": true}\n')
    (problem_dir / "solution").mkdir()
    (problem_dir / "solution" / "solve.py").write_text("print('oracle')\n")
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "# llm_criterion\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return 1.0\n"
    )

    output_dir = tmp_path / "harbor"
    export_harbor(problem_dir, output_dir)
    exported = tomllib.loads((output_dir / "task.toml").read_text())
    dockerfile = (output_dir / "environment" / "Dockerfile").read_text()

    assert exported["agent"]["user"] == "agent"
    assert exported["verifier"]["user"] == "root"
    assert exported["verifier"]["env"] == {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}
    assert exported["metadata"]["runtime_notices"]
    assert (output_dir / "solution" / "solve.py").is_file()
    assert (output_dir / "environment" / "calibration.lock.json").is_file()
    assert "requirements-solvers.txt" in dockerfile
    assert "/mcp_server/calibration/calibration.lock.json" in dockerfile


def test_software_scaffold_exports_seeded_non_root_harbor_image(
    tmp_path: Path,
) -> None:
    problem_dir = TaskCreator().create_structure(
        tmp_path / "problems",
        {
            "name": "labelbox/software-harbor-smoke",
            "template": "software-engineering",
        },
    )
    output_dir = tmp_path / "harbor"

    export_harbor(problem_dir, output_dir)

    task = tomllib.loads((output_dir / "task.toml").read_text())
    dockerfile = (output_dir / "environment" / "Dockerfile").read_text()
    assert task["agent"]["user"] == "agent"
    assert task["verifier"]["user"] == "root"
    assert task["verifier"]["env"] == {}
    assert (output_dir / "starter" / "normalizer.py").is_file()
    assert (output_dir / "environment" / "workspace_seed" / "normalizer.py").is_file()
    assert (output_dir / "solution" / "solve.sh").is_file()
    assert (output_dir / "tests" / "test.sh").is_file()
    assert not (output_dir / "scorer").exists()
    assert "id -u agent" in dockerfile
    assert "COPY --chown=agent:agent workspace_seed/ /tmp/output/repo/" in dockerfile
    assert "git -C /tmp/output/repo init -q" in dockerfile
    assert "chmod 0700 /mcp_server" in dockerfile
    assert dockerfile.rstrip().endswith("USER agent")


def test_harbor_normalizes_valid_iso_verifier_env_shape(tmp_path: Path) -> None:
    task = _capability_task()
    task["verifier"]["env"] = ["VERIFIER_TOKEN"]
    problem_dir = _write_problem(tmp_path / "problem", task)
    output_dir = tmp_path / "harbor"

    export_harbor(problem_dir, output_dir)

    exported = tomllib.loads((output_dir / "task.toml").read_text())
    assert "env" not in exported["agent"]
    assert exported["verifier"]["env"] == {"VERIFIER_TOKEN": "${VERIFIER_TOKEN}"}


def test_exported_tasks_validate_with_optional_harbor_task_config(
    tmp_path: Path,
) -> None:
    config_module = pytest.importorskip("harbor.models.task.config")
    task_config = config_module.TaskConfig

    software = TaskCreator().create_structure(
        tmp_path / "problems",
        {
            "name": "labelbox/harbor-task-config-smoke",
            "template": "software-engineering",
        },
    )
    candidates = [software]
    wal = Path(__file__).resolve().parents[2] / "examples" / "wal-recovery-ordering"
    if wal.is_dir():
        candidates.append(wal)

    for index, problem_dir in enumerate(candidates):
        output_dir = tmp_path / f"harbor-{index}"
        export_harbor(problem_dir, output_dir)
        task_config.model_validate_toml((output_dir / "task.toml").read_text())


def test_software_export_uses_declared_workspace_seed_without_services(
    tmp_path: Path,
) -> None:
    problem_dir = TaskCreator().create_structure(
        tmp_path / "problems",
        {
            "name": "labelbox/software-custom-seed",
            "template": "software-engineering",
        },
    )
    (problem_dir / "starter").rename(problem_dir / "authored-repository")
    task_path = problem_dir / "task.toml"
    task = tomllib.loads(task_path.read_text())
    task["workspace"] = {
        "seed": "authored-repository",
        "root": "/workdir",
        "agent_cwd": "/workdir",
        "init_policy": "copy",
    }
    task_path.write_text(tomli_w.dumps(task))

    output_dir = tmp_path / "harbor"
    export_harbor(problem_dir, output_dir)

    assert (
        output_dir / "environment" / ".alignerr-workspace-seed" / "normalizer.py"
    ).is_file()
    compose_path = output_dir / "environment" / "docker-compose.yaml"
    assert compose_path.is_file()
    assert "main" in yaml.safe_load(compose_path.read_text())["services"]


def test_software_export_image_permissions_when_docker_usable(
    tmp_path: Path,
) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable")
    daemon = subprocess.run(
        [docker, "info"],
        text=True,
        capture_output=True,
        check=False,
    )
    if daemon.returncode != 0:
        pytest.skip("Docker daemon is unavailable")
    stale_probes = subprocess.run(
        [
            docker,
            "ps",
            "-a",
            "--filter",
            "name=alignerr-docker-probe",
            "--format",
            "{{.ID}}",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if stale_probes.stdout.strip():
        pytest.skip("Docker daemon has stalled short-lived container probes")

    probe_name = (
        "alignerr-docker-probe-"
        + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    )
    try:
        probe = subprocess.run(
            [
                docker,
                "run",
                "--rm",
                "--name",
                probe_name,
                "--entrypoint",
                "/bin/true",
                "python:3.13-slim",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(
                [docker, "rm", "--force", probe_name],
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            pass
        pytest.skip("Docker daemon cannot start short-lived containers")
    if probe.returncode != 0:
        pytest.skip(f"Docker container probe failed: {probe.stderr.strip()}")

    problem_dir = TaskCreator().create_structure(
        tmp_path / "problems",
        {
            "name": "labelbox/software-harbor-image-smoke",
            "template": "software-engineering",
        },
    )
    output_dir = tmp_path / "harbor"
    export_harbor(problem_dir, output_dir)

    image = (
        "alignerr-software-harbor-smoke:"
        + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    )
    try:
        subprocess.run(
            [
                docker,
                "build",
                "--tag",
                image,
                str(output_dir / "environment"),
            ],
            text=True,
            check=True,
            timeout=900,
        )
        subprocess.run(
            [
                docker,
                "run",
                "--rm",
                "--entrypoint",
                "/bin/sh",
                image,
                "-ec",
                (
                    'test "$(id -u)" = 1000; '
                    'test "$(id -un)" = agent; '
                    "test -f /tmp/output/repo/normalizer.py; "
                    "test -d /tmp/output/repo/.git; "
                    "test ! -r /mcp_server/data/hidden_cases.json; "
                    "test ! -r /mcp_server/grader/compute_score.py"
                ),
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=180,
        )
    finally:
        subprocess.run(
            [docker, "image", "rm", "--force", image],
            text=True,
            capture_output=True,
            check=False,
        )
