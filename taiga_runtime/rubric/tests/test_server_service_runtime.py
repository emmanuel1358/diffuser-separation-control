from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from rubric import server
from rubric.service_runtime import (
    CaptureInfrastructureError,
    CommandResult,
    GraderWorkspaceHandoff,
    ServiceRuntimeAgentError,
    VerifierResult,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.start_calls = 0
        self.restart_calls = 0
        self.cleanup_calls = 0
        self.commands: list[str] = []
        self.edits: list[dict[str, object]] = []
        self.mcp_requests: list[tuple] = []
        self.final_result: VerifierResult | None = None
        self.final_error: Exception | None = None

    def start(self) -> None:
        self.start_calls += 1

    def restart_main(self) -> None:
        self.restart_calls += 1

    def exec_main(self, command: str) -> CommandResult:
        self.commands.append(command)
        return CommandResult(("docker", "compose", "exec"), 0, stdout="nested\n")

    def edit_main_file(self, **kwargs):
        self.edits.append(kwargs)
        return f"edited {kwargs['path']}"

    async def task_mcp_list_tools(self, server_name: str) -> dict:
        self.mcp_requests.append(("list", server_name))
        return {"tools": [{"name": "navigate"}]}

    async def task_mcp_call(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict | None,
    ) -> dict:
        self.mcp_requests.append(("call", server_name, tool_name, arguments))
        return {
            "content": [{"type": "text", "text": "ok"}],
            "isError": False,
        }

    def finalize_and_verify(self) -> VerifierResult | None:
        if self.final_error is not None:
            raise self.final_error
        return self.final_result

    def grader_handoff(self) -> GraderWorkspaceHandoff:
        return GraderWorkspaceHandoff(
            workspace=Path("/sealed/workspace"),
            manifest=Path("/sealed/workspace/manifest.json"),
        )

    def cleanup(self) -> None:
        self.cleanup_calls += 1


def test_setup_problem_starts_nested_runtime_only_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(server, "_SERVICE_RUNTIME", runtime)
    monkeypatch.setattr(
        server, "lock_down_grader_private", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(server, "lock_down_public_readonly", lambda *_args: None)
    monkeypatch.setattr(server, "_verify_continuous_evaluation", lambda _fields: False)

    prompt = asyncio.run(
        server.setup_problem(
            "nested-task",
            extra_fields={"task_prompt": "Work in the nested service."},
        )
    )

    assert prompt == "Work in the nested service."
    assert runtime.start_calls == 1


def test_bash_and_editor_route_to_main_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(server, "_SERVICE_RUNTIME", runtime)

    bash_result = asyncio.run(server.bash("id", restart=True))
    editor_result = asyncio.run(
        server.str_replace_editor(
            command="create",
            path="/workspace/note.txt",
            file_text="hello\n",
        )
    )

    assert runtime.restart_calls == 1
    assert runtime.commands == ["id"]
    assert bash_result.output == "nested\n"
    assert bash_result.error is None
    assert runtime.edits[0]["path"] == "/workspace/note.txt"
    assert editor_result.output == "edited /workspace/note.txt"


def test_generic_task_mcp_tools_route_only_list_and_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(server, "_SERVICE_RUNTIME", runtime)

    listed = asyncio.run(server.task_mcp_list_tools("browser"))
    called = asyncio.run(
        server.task_mcp_call(
            "browser",
            "navigate",
            {"url": "http://example.test"},
        )
    )

    assert listed == {"tools": [{"name": "navigate"}]}
    assert called["content"][0]["text"] == "ok"
    assert runtime.mcp_requests == [
        ("list", "browser"),
        (
            "call",
            "browser",
            "navigate",
            {"url": "http://example.test"},
        ),
    ]


def test_nested_verifier_result_is_canonical_grade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = FakeRuntime()
    runtime.final_result = VerifierResult(
        payload={
            "score": 0.4,
            "subscores": {"score": 0.4},
            "weights": {"score": 1.0},
            "metadata": {"source": "nested-verifier"},
        },
        result_dir=tmp_path,
        exit_code=0,
    )
    monkeypatch.setattr(server, "_SERVICE_RUNTIME", runtime)

    grade = asyncio.run(server.grade_problem("nested-task", transcript=""))

    assert grade.subscores == {"score": pytest.approx(0.4)}
    assert grade.env_internal_failure is None
    assert grade.metadata["service_runtime"] == "nested-docker"
    assert grade.metadata["source"] == "nested-verifier"


def test_capture_failure_is_reported_as_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    runtime.final_error = CaptureInfrastructureError("database capture failed")
    monkeypatch.setattr(server, "_SERVICE_RUNTIME", runtime)

    grade = asyncio.run(server.grade_problem("nested-task", transcript=""))

    assert grade.env_internal_failure is True
    assert grade.metadata["failure_classification"] == (
        "service_runtime_infrastructure"
    )
    assert "database capture failed" in grade.env_internal_failure_logs[0]


def test_missing_main_artifact_remains_an_agent_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    runtime.final_error = ServiceRuntimeAgentError("required output is missing")
    monkeypatch.setattr(server, "_SERVICE_RUNTIME", runtime)

    grade = asyncio.run(server.grade_problem("nested-task", transcript=""))

    assert grade.env_internal_failure is False
    assert grade.metadata["failure_classification"] == ("service_runtime_agent_fault")


def test_outer_grader_receives_explicit_sealed_workspace_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    sentinel = object()

    def fake_evaluate(*_args, **_kwargs):
        assert server.os.environ["LBX_SERVICE_ARTIFACT_SNAPSHOT"] == "/sealed/workspace"
        assert (
            server.os.environ["LBX_SERVICE_ARTIFACT_MANIFEST"]
            == "/sealed/workspace/manifest.json"
        )
        return sentinel

    monkeypatch.setattr(server, "_SERVICE_RUNTIME", runtime)
    monkeypatch.setattr(server, "_evaluate", fake_evaluate)

    result = asyncio.run(
        server.grade_problem(
            "nested-task",
            transcript="",
            extra_fields={"test_file": "compute_score.py"},
        )
    )

    assert result is sentinel


@pytest.mark.parametrize(
    "grader_source",
    [
        """
def compute_score(workspace, trajectory, private):
    expected = SERVICE_ARTIFACT_ROOT / "candidate.txt"
    return 0.73 if workspace == SERVICE_ARTIFACT_ROOT and expected.read_text() == "sealed" else 0.0
""",
        """
from pathlib import Path
def compute_score():
    expected = SERVICE_ARTIFACT_ROOT / "candidate.txt"
    return 0.73 if Path.cwd() == SERVICE_ARTIFACT_ROOT and expected.read_text() == "sealed" else 0.0
""",
    ],
)
def test_function_graders_use_sealed_snapshot_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    grader_source: str,
) -> None:
    snapshot = tmp_path / "sealed"
    snapshot.mkdir()
    (snapshot / "candidate.txt").write_text("sealed")
    monkeypatch.setenv("LBX_SERVICE_ARTIFACT_SNAPSHOT", str(snapshot))
    monkeypatch.setenv(
        "LBX_SERVICE_ARTIFACT_MANIFEST",
        str(snapshot / "manifest.json"),
    )

    grade = server._evaluate(grader_source)

    assert grade.subscores == {"score": pytest.approx(0.73)}


def test_legacy_task_config_keeps_runtime_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text('[task]\nname = "legacy"\n')
    monkeypatch.setenv("RUBRIC_TASK_TOML_PATH", str(task_toml))
    monkeypatch.setattr(server, "_SERVICE_RUNTIME", server._SERVICE_RUNTIME_UNSET)

    assert server._task_service_runtime() is None
