from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from rubric.service_config import ToolEndpoint
from rubric.tool_runtime import (
    ToolRegistry,
    ToolRegistryError,
    ToolRequestError,
)


def _endpoint() -> ToolEndpoint:
    return ToolEndpoint(
        name="browser",
        transport="sse",
        service="browser",
        url="http://browser:3080/sse",
        command=(),
        readiness_kind=None,
        readiness_service="browser",
        readiness_command=None,
        readiness_url=None,
        readiness_host=None,
        readiness_port=None,
        readiness_timeout_s=10.0,
        readiness_interval_s=0.1,
    )


class FakeSession:
    def __init__(self, *_args, **_kwargs) -> None:
        self.initialized = False
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def initialize(self):
        self.initialized = True
        return SimpleNamespace()

    async def list_tools(self, cursor=None):
        assert self.initialized
        assert cursor is None
        return SimpleNamespace(
            tools=[
                {
                    "name": "navigate",
                    "description": "Navigate safely",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                    },
                    "_meta": {"secret": "drop"},
                }
            ],
            nextCursor=None,
        )

    async def call_tool(
        self,
        name,
        arguments,
        read_timeout_seconds=None,
    ):
        assert self.initialized
        assert read_timeout_seconds is not None
        self.calls.append((name, arguments))
        return SimpleNamespace(
            content=[
                {
                    "type": "text",
                    "text": f"visited {arguments['url']}",
                    "transport": object(),
                }
            ],
            structuredContent={"visited": arguments["url"]},
            isError=False,
        )


def _mock_registry(
    session_factory=FakeSession,
) -> tuple[ToolRegistry, list[dict]]:
    connections: list[dict] = []

    @asynccontextmanager
    async def fake_sse(url, **kwargs):
        connections.append({"url": url, **kwargs})
        yield object(), object()

    registry = ToolRegistry(
        (_endpoint(),),
        sse_url_resolver=lambda _endpoint: "http://127.0.0.1:3080/sse",
        sse_client_factory=fake_sse,
        session_factory=session_factory,
    )
    return registry, connections


def test_mocked_sse_proxy_lists_and_calls_only_tools_methods() -> None:
    registry, connections = _mock_registry()
    registry.set_ready("browser", True)

    listed = asyncio.run(registry.list_tools("browser"))
    called = asyncio.run(
        registry.call_tool(
            "browser",
            "navigate",
            {"url": "http://example.test"},
        )
    )

    assert listed == {
        "tools": [
            {
                "name": "navigate",
                "description": "Navigate safely",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                },
            }
        ]
    }
    assert called == {
        "content": [
            {
                "type": "text",
                "text": "visited http://example.test",
            }
        ],
        "structuredContent": {"visited": "http://example.test"},
        "isError": False,
    }
    assert len(connections) == 2
    assert all(connection["headers"] is None for connection in connections)
    assert all(
        connection["url"] == "http://127.0.0.1:3080/sse" for connection in connections
    )


def test_sse_http_client_disables_redirects_proxies_and_revalidates_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://attacker.invalid:8080")
    registry, connections = _mock_registry()
    registry.set_ready("browser", True)
    asyncio.run(registry.list_tools("browser"))
    factory = connections[0]["httpx_client_factory"]
    client = factory(timeout=httpx.Timeout(5.0))

    assert client.follow_redirects is False
    assert client._trust_env is False
    request_hook = client._event_hooks["request"][0]

    async def check_requests() -> None:
        await request_hook(httpx.Request("GET", "http://127.0.0.1:3080/sse"))
        await request_hook(
            httpx.Request(
                "POST",
                "http://127.0.0.1:3080/messages/?session_id=ok",
            )
        )
        with pytest.raises(ToolRegistryError, match="effective endpoint"):
            await request_hook(
                httpx.Request(
                    "GET",
                    "http://169.254.169.254/latest/meta-data",
                )
            )
        with pytest.raises(ToolRegistryError, match="effective endpoint"):
            await request_hook(httpx.Request("GET", "http://127.0.0.1:3080/redirected"))
        await client.aclose()

    asyncio.run(check_requests())


def test_readiness_transport_and_user_errors_are_classified() -> None:
    registry, connections = _mock_registry()

    with pytest.raises(ToolRegistryError, match="not ready"):
        asyncio.run(registry.list_tools("browser"))
    assert connections == []

    registry.set_ready("browser", True)
    with pytest.raises(ToolRequestError, match="invalid MCP tool name"):
        asyncio.run(registry.call_tool("browser", "bad tool", {}))
    with pytest.raises(ToolRequestError, match="non-finite"):
        asyncio.run(registry.call_tool("browser", "navigate", {"x": float("nan")}))
    with pytest.raises(ToolRequestError, match="encoded bytes"):
        asyncio.run(
            registry.call_tool(
                "browser",
                "navigate",
                {"value": "x" * (300 * 1024)},
            )
        )
    assert connections == []
    with pytest.raises(ToolRequestError, match="unknown tool"):
        asyncio.run(registry.call_tool("browser", "missing", {}))
    assert len(connections) == 1


def test_transport_failure_is_infrastructure() -> None:
    @asynccontextmanager
    async def failing_sse(_url, **_kwargs):
        raise OSError("connection refused")
        yield  # pragma: no cover

    registry = ToolRegistry(
        (_endpoint(),),
        sse_url_resolver=lambda _endpoint: "http://127.0.0.1:3080/sse",
        sse_client_factory=failing_sse,
    )
    registry.set_ready("browser", True)

    with pytest.raises(ToolRegistryError, match="connection refused"):
        asyncio.run(registry.list_tools("browser"))


def test_oversized_remote_output_is_infrastructure() -> None:
    class HugeSession(FakeSession):
        async def call_tool(self, *_args, **_kwargs):
            return SimpleNamespace(
                content=[{"type": "text", "text": "x" * (2 * 1024 * 1024)}],
                structuredContent=None,
                isError=False,
            )

    registry, _connections = _mock_registry(HugeSession)
    registry.set_ready("browser", True)

    with pytest.raises(ToolRegistryError, match="encoded bytes"):
        asyncio.run(registry.call_tool("browser", "navigate", {}))


def test_local_sse_mcp_integration() -> None:
    uvicorn = pytest.importorskip("uvicorn")
    child = FastMCP("local-child")

    @child.tool()
    def echo(value: str) -> dict[str, str]:
        return {"echo": value}

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            child.sse_app(),
            host="127.0.0.1",
            port=port,
            log_level="error",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2.0)
        pytest.fail("local MCP SSE server did not start")

    endpoint = replace(_endpoint(), url=f"http://browser:{port}/sse")
    registry = ToolRegistry(
        (endpoint,),
        sse_url_resolver=lambda _endpoint: f"http://127.0.0.1:{port}/sse",
    )
    registry.set_ready("browser", True)
    try:
        listed = asyncio.run(registry.list_tools("browser"))
        called = asyncio.run(registry.call_tool("browser", "echo", {"value": "live"}))
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)

    assert [tool["name"] for tool in listed["tools"]] == ["echo"]
    assert called["isError"] is False
    assert called["structuredContent"] == {"echo": "live"}
