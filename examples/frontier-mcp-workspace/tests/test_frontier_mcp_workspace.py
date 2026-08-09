from __future__ import annotations

import asyncio
import fcntl
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import tomllib
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "taiga_runtime/rubric/src"))

from alignerr_plugin.capabilities import resolve_capabilities
from alignerr_plugin.capsule import (
    MaterializedServiceImage,
    capsule_compose_data,
)
from alignerr_plugin.schemas import TaskToml
from rubric.service_config import (
    RuntimeOperatorRoots,
    ToolEndpoint,
    load_task_service_config,
)
from rubric.tool_runtime import ToolRegistry


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task() -> TaskToml:
    return TaskToml.model_validate(tomllib.loads((ROOT / "task.toml").read_text()))


def _capsule_compose(task: TaskToml) -> dict:
    capabilities = resolve_capabilities(task)
    images = tuple(
        MaterializedServiceImage(
            name=service.name,
            role=service.role,
            digest=f"sha256:{index:064x}",
            archive_ref=f"local/{service.name}:locked",
            source="build",
            platform="linux/amd64",
            os="linux",
            architecture="amd64",
        )
        for index, service in enumerate(capabilities.services, start=1)
    )
    return capsule_compose_data(capabilities, images)


def test_native_schema_runtime_mcp_and_shared_volume(tmp_path: Path) -> None:
    task = _task()
    roots = RuntimeOperatorRoots(
        capsule=tmp_path / "capsule",
        state=tmp_path / "state",
        sealed=tmp_path / "sealed",
    )
    runtime = load_task_service_config(ROOT / "task.toml", operator_roots=roots)

    assert runtime is not None
    assert runtime.main_service == "main"
    assert runtime.verifier_service == "verifier"
    assert [tool.name for tool in runtime.tools] == ["playwright"]
    assert runtime.tools[0].url == "http://browser-mcp:3080/sse"
    assert runtime.tools[0].readiness_url == "http://browser-mcp:3080/health"
    assert task.mcp_servers[0].transport == "sse"
    assert task.difficulty.license == "self_generated"
    assert "taiga" not in task.metadata
    mounts = {
        service.name: {mount.volume: mount.mode for mount in service.volumes}
        for service in task.services
    }
    assert mounts["workspace"]["workspace-shared"] == "rw"
    assert mounts["browser-mcp"]["workspace-shared"] == "ro"
    assert mounts["main"]["workspace-shared"] == "ro"
    assert any(
        output.required and output.path == "/tmp/output/grade.json"
        for output in task.outputs
    )
    assert any(
        artifact.destination == "repo"
        and artifact.service == "main"
        and artifact.source == "/workspace/repo"
        for artifact in task.artifacts
    )


def test_generated_capsule_compose_is_immutable_and_valid(tmp_path: Path) -> None:
    compose = _capsule_compose(_task())
    services = compose["services"]

    assert set(services) == {"workspace", "browser-mcp", "main", "verifier"}
    assert all(service["pull_policy"] == "never" for service in services.values())
    assert all("build" not in service for service in services.values())
    assert "workspace-shared:/shared:ro" in services["browser-mcp"]["volumes"]
    assert "workspace-shared:/shared:ro" in services["main"]["volumes"]
    assert (
        services["main"]["depends_on"]["browser-mcp"]["condition"] == "service_healthy"
    )

    compose_path = tmp_path / "compose.yaml"
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=True))
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable; trusted CI runs Compose config")
    try:
        version = subprocess.run(
            [docker, "compose", "version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("Docker Compose availability check timed out")
    if version.returncode != 0:
        pytest.skip("Docker Compose is unavailable; trusted CI runs Compose config")
    try:
        subprocess.run(
            [docker, "compose", "-f", str(compose_path), "config", "--quiet"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("Docker Compose config validation timed out")


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_health(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"MCP process exited with {process.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health",
                timeout=0.2,
            ) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.02)
    raise TimeoutError("MCP process did not become healthy")


def _endpoint(port: int) -> ToolEndpoint:
    return ToolEndpoint(
        name="playwright",
        transport="sse",
        service="browser-mcp",
        url=f"http://browser-mcp:{port}/sse",
        command=(),
        readiness_kind="http",
        readiness_service="browser-mcp",
        readiness_command=None,
        readiness_url=f"http://browser-mcp:{port}/health",
        readiness_host=None,
        readiness_port=None,
        readiness_timeout_s=5,
        readiness_interval_s=0.05,
    )


def test_local_stdlib_sse_mcp_lists_and_calls_browser_tools(tmp_path: Path) -> None:
    shared = tmp_path / "workspace.json"
    shared.write_text(
        json.dumps(
            {
                "schema_version": "synthetic-claims-workspace.v1",
                "title": "Synthetic Claims Inbox",
                "cases": [{"claim_id": "V-001"}, {"claim_id": "V-002"}],
            }
        )
    )
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "environment/mcp/server.py")],
        env={
            **os.environ,
            "MCP_HOST": "127.0.0.1",
            "MCP_PORT": str(port),
            "SHARED_PATH": str(shared),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_health(port, process)
        endpoint = _endpoint(port)
        registry = ToolRegistry(
            (endpoint,),
            sse_url_resolver=lambda declared: replace(
                declared,
                url=f"http://127.0.0.1:{port}/sse",
            ).url
            or "",
        )
        registry.set_ready("playwright", True)

        listed = asyncio.run(registry.list_tools("playwright"))
        called = asyncio.run(registry.call_tool("playwright", "browser_snapshot", {}))
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)

    assert [tool["name"] for tool in listed["tools"]] == [
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
    ]
    assert called["isError"] is False
    assert called["structuredContent"]["title"] == "Synthetic Claims Inbox"
    assert called["structuredContent"]["case_count"] == 2


def test_child_images_are_pinned_and_non_root_except_trusted_verifier() -> None:
    dockerfiles = sorted((ROOT / "environment").glob("*/Dockerfile"))
    assert len(dockerfiles) == 4
    for dockerfile in dockerfiles:
        text = dockerfile.read_text()
        assert (
            "FROM python:3.13-slim-bookworm@sha256:"
            "bb73517d48bd32016e15eade0c009b2724ec3a025a9975b5cd9b251d0dcadb33" in text
        )
        if dockerfile.parent.name != "verifier":
            assert "USER root" not in text
            assert "USER " in text

    outer = (ROOT / "environment/Dockerfile").read_text()
    assert "COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/" in outer
    assert "find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600" in outer
    verifier_source = (ROOT / "environment/verifier/verify.py").read_text()
    assert "subprocess" not in verifier_source


def test_independent_verifier_publishes_canonical_result(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    evidence = artifacts / "evidence"
    repo = artifacts / "repo"
    evidence.mkdir(parents=True)
    repo.mkdir()
    shutil.copy2(ROOT / "solution/files/processor.py", repo / "processor.py")
    (evidence / "workspace.json").write_text(
        json.dumps(
            {
                "schema_version": "synthetic-claims-workspace.v1",
                "title": "Synthetic Claims Inbox",
                "cases": [{"claim_id": "V-001"}, {"claim_id": "V-002"}],
            }
        )
    )
    output = tmp_path / "output"

    subprocess.run(
        ["python3", str(ROOT / "environment/verifier/verify.py")],
        check=True,
        env={
            **os.environ,
            "ARTIFACT_ROOT": str(artifacts),
            "OUTPUT_ROOT": str(output),
        },
        timeout=10,
    )

    result = json.loads((output / "grade.json").read_text())
    assert result["schema_version"] == "frontier-mcp-verifier.v1"
    assert result["score"] == 1.0
    assert set(result["subscores"]) == {
        "source_contract",
        "sealed_workspace_evidence",
    }


def test_host_oracle_noop_and_introspection_cheat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUBRIC_AGENT_UID", str(os.getuid()))
    monkeypatch.setenv("RUBRIC_AGENT_GID", str(os.getgid()))
    candidate_requests: list[bytes] = []

    def run_local_candidate(
        cmd,
        *,
        cwd_fd,
        stdin_bytes=None,
        env=None,
        timeout_s=None,
        max_output_bytes,
        **_kwargs,
    ):
        candidate_requests.append(stdin_bytes or b"")
        try:
            candidate_cwd = os.readlink(f"/proc/self/fd/{cwd_fd}")
        except OSError:
            raw_path = fcntl.fcntl(
                cwd_fd,
                getattr(fcntl, "F_GETPATH", 50),
                b"\0" * 1024,
            )
            candidate_cwd = os.fsdecode(raw_path.split(b"\0", 1)[0])
        completed = subprocess.run(
            cmd,
            input=stdin_bytes,
            cwd=candidate_cwd,
            env=env,
            check=False,
            capture_output=True,
            timeout=timeout_s,
        )
        return subprocess.CompletedProcess(
            cmd,
            completed.returncode,
            stdout=completed.stdout[: max_output_bytes + 1],
            stderr=b"",
        )

    monkeypatch.setattr(
        "grading.helpers.run_submitted_executable",
        run_local_candidate,
    )
    scorer = _module(ROOT / "scorer/compute_score.py", "mcp_example_scorer")
    oracle = tmp_path / "oracle"
    noop = tmp_path / "noop"
    cheat = tmp_path / "cheat"
    subprocess.run(
        ["bash", str(ROOT / "solution/solve.sh")],
        check=True,
        env={**os.environ, "LBT_OUTPUT_DIR": str(oracle)},
        timeout=10,
    )
    subprocess.run(
        ["bash", str(ROOT / "baselines/noop.sh")],
        check=True,
        env={**os.environ, "LBT_OUTPUT_DIR": str(noop)},
        timeout=10,
    )
    subprocess.run(
        ["bash", str(ROOT / "baselines/noop.sh")],
        check=True,
        env={**os.environ, "LBT_OUTPUT_DIR": str(cheat)},
        timeout=10,
    )
    shutil.copy2(
        ROOT / "tests/introspection_cheat.py",
        cheat / "repo/processor.py",
    )

    oracle_grade = scorer.TASK.grade(
        workspace=oracle,
        private=ROOT / "scorer/data",
    )
    noop_grade = scorer.TASK.grade(
        workspace=noop,
        private=ROOT / "scorer/data",
    )
    cheat_grade = scorer.TASK.grade(
        workspace=cheat,
        private=ROOT / "scorer/data",
    )

    assert oracle_grade.score() == pytest.approx(1.0), (
        oracle_grade.subscores,
        oracle_grade.criterion_logs,
    )
    assert noop_grade.score() == pytest.approx(0.0)
    assert cheat_grade.score() == pytest.approx(0.0)
    assert candidate_requests
    for request_bytes in candidate_requests:
        assert b'"expected"' not in request_bytes
        request = json.loads(request_bytes)
        assert set(request) == {"input"}
        assert set(request["input"]) == {
            "billed_cents",
            "claim_id",
            "contract_cap_cents",
            "covered",
        }
    assert "expected" not in scorer.DRIVER_SOURCE
    assert "claim_cases.json" not in scorer.DRIVER_SOURCE
