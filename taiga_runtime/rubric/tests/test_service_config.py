from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from alignerr_plugin.capabilities import resolve_capabilities
from alignerr_plugin.capsule import (
    MaterializedServiceImage,
    capsule_compose_data,
    load_capabilities,
)
from alignerr_plugin.utils import load_task_toml
from rubric.capsule_runtime import CapsuleBundle, CapsuleRuntimeError
from rubric.service_config import (
    RuntimeOperatorRoots,
    ServiceConfigurationError,
    load_task_service_config,
)


def test_legacy_task_without_capability_services_is_a_noop(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text("""
[task]
name = "legacy"

[environment]
required_resources = "2vcpu+8gb"
""")

    assert load_task_service_config(task_toml) is None


@pytest.mark.parametrize(
    "native_section",
    [
        """
[workspace]
root = "/workspace"
init_policy = "empty"
git_baseline = false
checkpoint_restore = false
""",
        "artifacts = []\n",
        "captures = []\n",
        "mcp_servers = []\n",
        """
[evaluation]
engine = "legacy_runner"
""",
        "gates = []\n",
        "reports = []\n",
        "[result]\n",
    ],
)
def test_each_native_capability_section_activates_runtime(
    tmp_path: Path,
    native_section: str,
) -> None:
    task_toml = tmp_path / "task.toml"
    if native_section.startswith(("[", "\n[")):
        content = f'[task]\nname = "native-section"\n{native_section}'
    else:
        content = f'{native_section}\n[task]\nname = "native-section"\n'
    task_toml.write_text(content)

    config = load_task_service_config(task_toml)

    assert config is not None
    assert config.main_service == "main"
    assert config.services[0].role == "main"
    assert config.primary_reward == "score"


def test_schema_export_projection_activates_implicit_main_without_services(
    tmp_path: Path,
) -> None:
    problem = tmp_path / "problem"
    environment = problem / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM scratch\n")
    task_toml = problem / "task.toml"
    task_toml.write_text("""
schema_version = "1.2"

[task]
name = "native/implicit-main"

[environment]
required_resources = "4vcpu+16gib"

[workspace]
root = "/workspace"
agent_cwd = "/workspace"
init_policy = "empty"
git_baseline = false
checkpoint_restore = false

[evaluation]
engine = "legacy_runner"

[difficulty]
task_type = "software_engineering"
domain = "repo_debugging"
reward_type = "multi_deterministic_rubrics"
""")

    schema_task = load_task_toml(problem)
    assert resolve_capabilities(schema_task).services == ()
    exported = load_capabilities(problem)
    assert [(service.name, service.role) for service in exported.services] == [
        ("main", "agent")
    ]
    digest = f"sha256:{"a" * 64}"
    image = MaterializedServiceImage(
        name="main",
        role="agent",
        digest=digest,
        archive_ref="alignerr-capsule/main:locked",
        source="build",
        source_ref=None,
        platform="linux/amd64",
        os="linux",
        architecture="amd64",
    )
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "docker-compose.yaml").write_text(
        json.dumps(capsule_compose_data(exported, (image,)))
    )
    (capsule / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "alignerr.task-capsule.v1",
                "compose_file": "docker-compose.yaml",
                "agent_service": "main",
                "verifier_service": None,
                "images": [{"service": "main", "role": "main"}],
            }
        )
    )
    roots = RuntimeOperatorRoots(
        capsule=capsule,
        state=tmp_path / "state",
        sealed=tmp_path / "sealed",
    )

    config = load_task_service_config(task_toml, operator_roots=roots)

    assert config is not None
    assert [(service.name, service.role) for service in config.services] == [
        ("main", "main")
    ]
    assert config.main_service == "main"
    assert config.verifier_service is None
    assert config.workspace is not None
    assert config.primary_reward == "score"


def test_schema_shaped_service_capture_and_artifact_config(
    tmp_path: Path,
) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text(f"""
[task]
name = "native/service-task"

[workspace]
seed = "workspace"
root = "/workspace"
agent_cwd = "/workspace/repo"
init_policy = "overlay"
git_baseline = false
clean_paths = ["build"]
checkpoint_restore = false

[agent]
user = "1000:1000"

[verifier]
timeout_sec = 91

[result]
output_root = "/tmp/output"
reward_file = "grade.json"
reward_key = "reward"

[[services]]
name = "main"
role = "main"
image = "example/main@sha256:{"1" * 64}"
user = "1000:1000"

[[services]]
name = "database"
role = "sidecar"
image = "example/db@sha256:{"2" * 64}"

[[services]]
name = "migrate"
role = "init"
image = "example/init@sha256:{"3" * 64}"

[[services]]
name = "verifier"
role = "verifier"
image = "example/verifier@sha256:{"4" * 64}"

[[captures]]
name = "database-dump"
service = "database"
command = ["pg_dump", "--file", "/tmp/db.dump.tmp"]
timeout_sec = 30
atomic_destination = "/tmp/db.dump"
accepted_exit_codes = [0]
failure_policy = "infrastructure"

[[artifacts]]
kind = "service"
name = "database-dump"
source = "/tmp/db.dump"
service = "database"
destination = "evidence/db.dump"
max_bytes = 1024
max_depth = 4
""")

    roots = RuntimeOperatorRoots(
        capsule=tmp_path / "capsule-root",
        state=tmp_path / "state-root",
        sealed=tmp_path / "sealed-root",
    )
    config = load_task_service_config(task_toml, operator_roots=roots)

    assert config is not None
    assert config.capsule_dir == roots.capsule
    assert config.main_service == "main"
    assert config.agent_services == ("main", "database", "migrate")
    assert config.verifier_service == "verifier"
    assert config.agent_user == "1000:1000"
    assert config.agent_workdir == "/workspace/repo"
    assert config.workspace is not None
    assert config.workspace.init_policy == "overlay"
    assert config.workspace.clean_paths == ("build",)
    assert config.workspace.checkpoint_restore is False
    assert config.captures[0].service == "database"
    assert config.captures[0].command == "pg_dump --file /tmp/db.dump.tmp"
    assert config.captures[0].atomic_destination == "/tmp/db.dump"
    assert config.captures[0].failure_policy == "infrastructure"
    assert config.artifacts[0].destination == "evidence/db.dump"
    assert config.artifacts[0].max_depth == 4
    assert config.tools == ()
    assert config.verifier_result_paths == ("/tmp/output/grade.json",)
    assert config.primary_reward == "reward"
    assert config.subscores_key == "subscores"


def test_task_selected_runtime_paths_are_rejected(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text("""
[task]
name = "path-escape"

[[services]]
name = "main"
role = "main"
user = "1000:1000"

[service_runtime]
state_dir = "/"
sealed_dir = "/mcp_server/grader"
""")

    with pytest.raises(ServiceConfigurationError, match="operator-owned"):
        load_task_service_config(task_toml)


def test_task_local_sse_mcp_declaration_is_accepted(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text("""
[task]
name = "mcp"

[[services]]
name = "main"
role = "main"
user = "1000:1000"

[[tools.mcp_servers]]
name = "browser"
transport = "sse"
service = "main"
url = "http://main:3080/sse"
""")

    config = load_task_service_config(task_toml)

    assert config is not None
    assert config.tools[0].name == "browser"
    assert config.tools[0].transport == "sse"
    assert config.tools[0].url == "http://main:3080/sse"


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        (
            'transport = "stdio"\ncommand = ["node", "server.js"]',
            "only audited SSE",
        ),
        (
            'transport = "sse"\nurl = "http://attacker.example:3080/sse"',
            "declared service DNS",
        ),
        (
            (
                'transport = "sse"\nurl = "http://main:3080/sse"\n'
                'headers = { Authorization = "secret" }'
            ),
            "credentials or headers",
        ),
    ],
)
def test_task_local_mcp_transport_validation(
    tmp_path: Path,
    declaration: str,
    expected: str,
) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text(f"""
[task]
name = "mcp-invalid"

[[services]]
name = "main"
role = "main"
user = "1000:1000"

[[tools.mcp_servers]]
name = "browser"
service = "main"
{declaration}
""")

    with pytest.raises(ServiceConfigurationError, match=expected):
        load_task_service_config(task_toml)


def test_operator_roots_reject_protected_and_symlink_paths(tmp_path: Path) -> None:
    target = tmp_path / "real-state"
    target.mkdir()
    symlink = tmp_path / "state-link"
    symlink.symlink_to(target, target_is_directory=True)

    with pytest.raises(ServiceConfigurationError, match="protected"):
        RuntimeOperatorRoots(
            capsule=tmp_path / "capsule",
            state=Path("/mcp_server/grader/runtime"),
            sealed=tmp_path / "sealed",
        )
    with pytest.raises(ServiceConfigurationError, match="symlink"):
        RuntimeOperatorRoots(
            capsule=tmp_path / "capsule",
            state=symlink / "runtime",
            sealed=tmp_path / "sealed",
        )


def test_main_service_must_use_unprivileged_proxy_user(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text(f"""
[task]
name = "bad-root"

[[services]]
name = "main"
role = "main"
image = "example/main@sha256:{"1" * 64}"
user = "00:00"
""")

    with pytest.raises(Exception, match="non-root"):
        load_task_service_config(task_toml)


def test_capsule_manifest_indirection_and_archive_digest_lock(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "images.docker.tar"
    archive.write_bytes(b"locked archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    image_digest = f"sha256:{"a" * 64}"
    image_manifest = {
        "schema": "alignerr.task-capsule.images.v1",
        "images": [
            {
                "service": "main",
                "role": "main",
                "archive": archive.name,
                "archive_sha256": archive_sha,
                "image_ref": "lbx-capsule/main:locked",
                "image_digest": image_digest,
            }
        ],
    }
    # The generated outer image flattens these two files into the capsule root,
    # while the outer manifest retains their context-relative images/ prefix.
    (tmp_path / "images.manifest.json").write_text(json.dumps(image_manifest))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "alignerr.task-capsule.v1",
                "image_manifest": "images/images.manifest.json",
                "image_archive": "images/images.docker.tar",
            }
        )
    )
    (tmp_path / "docker-compose.yaml").write_text("services: {}\n")

    bundle = CapsuleBundle.load(tmp_path)

    assert bundle.images[0].archive_path == archive
    bundle.verify_archives()

    archive.write_bytes(b"mutated")
    with pytest.raises(CapsuleRuntimeError, match="archive digest mismatch"):
        bundle.verify_archives()
