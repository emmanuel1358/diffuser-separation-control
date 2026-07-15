from __future__ import annotations

import asyncio
import sys
import types

from lbx_rl_tasks_harness.runtimes.claude_code import (
    _build_rubric_mcp_server,
    _effective_model,
    _invoke_bridge_tool,
    _model_facing_tool_name,
)


def test_effective_model_prefers_cli_then_env(monkeypatch) -> None:
    monkeypatch.setenv("LBX_RL_CLAUDE_CODE_MODEL", "claude-sonnet-4-5")

    assert _effective_model("claude-fable-5") == "claude-fable-5"
    assert _effective_model(None) == "claude-sonnet-4-5"


class FakeBridgeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict):
        self.calls.append(args)
        return [{"type": "text", "text": "ok"}]


class FakeBridge:
    def __init__(self) -> None:
        self.bash = FakeBridgeTool("bash")
        self.editor = FakeBridgeTool("str_replace_based_edit_tool")

    def agent_tools(self) -> list[FakeBridgeTool]:
        return [self.bash, self.editor]


def test_build_rubric_mcp_server_uses_existing_bridge(monkeypatch) -> None:
    captured = {}

    def fake_tool(*, name, description, input_schema):
        def decorate(func):
            func._tool_name = name
            func._description = description
            func._input_schema = input_schema
            return func

        return decorate

    def fake_create_sdk_mcp_server(*, name, version, tools):
        captured.update({"name": name, "version": version, "tools": tools})
        return captured

    module = types.ModuleType("claude_agent_sdk")
    module.tool = fake_tool
    module.create_sdk_mcp_server = fake_create_sdk_mcp_server
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)

    server = _build_rubric_mcp_server(FakeBridge())

    assert server["name"] == "rubric"
    assert [tool._tool_name for tool in server["tools"]] == [
        "bash",
        "str_replace_editor",
    ]


def test_invoke_bridge_tool_returns_sdk_content() -> None:
    tool = FakeBridgeTool("bash")

    result = asyncio.run(_invoke_bridge_tool(tool, {"command": "echo ok"}))

    assert tool.calls == [{"command": "echo ok"}]
    assert result == {"content": [{"type": "text", "text": "ok"}]}


def test_model_facing_tool_name_matches_production_edit_tool() -> None:
    assert (
        _model_facing_tool_name("mcp__rubric__str_replace_editor")
        == "str_replace_based_edit_tool"
    )
    assert _model_facing_tool_name("mcp__rubric__bash") == "bash"
