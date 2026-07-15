from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from typing import Any

from lbx_rl_tasks_harness.mcp_bridge import RubricMcpBridge


@dataclass
class FakeTool:
    name: str


class FakeSessionManager:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def test_rubric_mcp_bridge_uses_installed_console_script(monkeypatch) -> None:
    captured_config: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, config: dict[str, Any]) -> None:
            captured_config.update(config)

        def session(self, name: str) -> FakeSessionManager:
            assert name == "rubric"
            return FakeSessionManager()

    async def fake_load_mcp_tools(_session: object) -> list[FakeTool]:
        return [
            FakeTool("setup_problem"),
            FakeTool("grade_problem"),
            FakeTool("bash"),
            FakeTool("str_replace_editor"),
        ]

    client_module = types.ModuleType("langchain_mcp_adapters.client")
    client_module.MultiServerMCPClient = FakeClient
    tools_module = types.ModuleType("langchain_mcp_adapters.tools")
    tools_module.load_mcp_tools = fake_load_mcp_tools
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", client_module)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.tools", tools_module)

    async def enter_bridge() -> None:
        async with RubricMcpBridge("container-123"):
            pass

    asyncio.run(enter_bridge())

    rubric_args = captured_config["rubric"]["args"]
    assert rubric_args[:7] == [
        "exec",
        "-i",
        "container-123",
        "env",
        "FASTMCP_LOG_LEVEL=ERROR",
        "LOG_LEVEL=ERROR",
        "/opt/lbx-runtime/.venv/bin/rubric",
    ]
    assert rubric_args[-1] == "mcp"
    assert "uv" not in rubric_args
