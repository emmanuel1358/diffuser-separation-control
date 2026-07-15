from __future__ import annotations

import os
from contextlib import AbstractAsyncContextManager
from typing import Any


def rubric_mcp_stdio_config(container_id: str) -> dict[str, Any]:
    return {
        "command": "docker",
        "args": [
            "exec",
            "-i",
            container_id,
            "env",
            "FASTMCP_LOG_LEVEL=ERROR",
            "LOG_LEVEL=ERROR",
            "/opt/lbx-runtime/.venv/bin/rubric",
            "mcp",
        ],
        "env": {
            **os.environ,
            "FASTMCP_LOG_LEVEL": "ERROR",
            "LOG_LEVEL": "ERROR",
        },
        "transport": "stdio",
    }


class RubricMcpBridge(AbstractAsyncContextManager["RubricMcpBridge"]):
    """Connect LangChain tools to the rubric MCP server inside a task container."""

    def __init__(self, container_id: str) -> None:
        self._container_id = container_id
        self._client: Any = None
        self._session_cm: Any = None
        self._tools_by_name: dict[str, Any] = {}

    async def __aenter__(self) -> "RubricMcpBridge":
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langchain_mcp_adapters.tools import load_mcp_tools
        except ImportError as exc:
            raise RuntimeError(
                "MCP bridge dependencies are missing. Install with `uv sync`."
            ) from exc

        self._client = MultiServerMCPClient(
            {"rubric": rubric_mcp_stdio_config(self._container_id)}
        )
        self._session_cm = self._client.session("rubric")
        session = await self._session_cm.__aenter__()
        tools = await load_mcp_tools(session)
        self._tools_by_name = {tool.name: tool for tool in tools}
        missing = {
            "setup_problem",
            "grade_problem",
            "bash",
            "str_replace_editor",
        } - set(self._tools_by_name)
        if missing:
            raise RuntimeError(
                f"rubric MCP server is missing required tools: {sorted(missing)}; "
                f"registered={sorted(self._tools_by_name)}"
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)

    def setup_problem_tool(self):
        return self._tools_by_name["setup_problem"]

    def grade_problem_tool(self):
        return self._tools_by_name["grade_problem"]

    def agent_tools(self) -> list[Any]:
        out = []
        for source_name, model_name in {
            "bash": "bash",
            "str_replace_editor": "str_replace_based_edit_tool",
        }.items():
            tool = self._tools_by_name[source_name]
            if tool.name != model_name:
                tool.name = model_name
            out.append(tool)
        return out
