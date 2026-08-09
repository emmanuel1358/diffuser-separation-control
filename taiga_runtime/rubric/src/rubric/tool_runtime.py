"""Audited task-local MCP registry and bounded SSE client proxy."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from types import TracebackType
from typing import Any, Protocol
from urllib.parse import urlsplit

import anyio
import httpx
from grading.faults import InfrastructureFault
from mcp import ClientSession
from mcp.client.sse import sse_client
from rubric.service_config import ToolEndpoint

_SERVER_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_REMOTE_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_INPUT_MAX_BYTES = 256 * 1024
_INPUT_MAX_DEPTH = 16
_INPUT_MAX_NODES = 10_000
_OUTPUT_MAX_BYTES = 1024 * 1024
_OUTPUT_MAX_DEPTH = 32
_OUTPUT_MAX_NODES = 25_000
_MAX_TOOL_PAGES = 16
_MAX_TOOLS = 512
_CONNECT_TIMEOUT_S = 5.0
_READ_TIMEOUT_S = 30.0
_CALL_TIMEOUT_S = 60.0


class ToolRegistryError(InfrastructureFault):
    """A declared task-local MCP server is unavailable."""


class UnsupportedToolTransportError(ToolRegistryError):
    """No secure proxy implementation is registered for this transport."""


class ToolRequestError(ValueError):
    """Agent-controlled MCP request data is invalid."""


class SseClientContext(Protocol):
    async def __aenter__(self) -> tuple[Any, Any]: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


@dataclass(frozen=True, slots=True)
class ToolReadiness:
    """Current readiness state for one declared tool."""

    endpoint: ToolEndpoint
    ready: bool
    detail: str | None = None


class ToolRegistry:
    """Thread-safe readiness state plus an allowlisted MCP SSE proxy."""

    def __init__(
        self,
        endpoints: Iterable[ToolEndpoint] = (),
        *,
        sse_url_resolver: Callable[[ToolEndpoint], str] | None = None,
        sse_client_factory: Callable[..., SseClientContext] = sse_client,
        session_factory: Callable[..., Any] = ClientSession,
    ) -> None:
        self._lock = threading.RLock()
        self._endpoints: dict[str, ToolEndpoint] = {}
        self._readiness: dict[str, ToolReadiness] = {}
        self._sse_url_resolver = sse_url_resolver or self._declared_url
        self._sse_client_factory = sse_client_factory
        self._session_factory = session_factory
        for endpoint in endpoints:
            if endpoint.name in self._endpoints:
                raise ToolRegistryError(f"duplicate task-local tool {endpoint.name!r}")
            self._endpoints[endpoint.name] = endpoint
            self._readiness[endpoint.name] = ToolReadiness(
                endpoint=endpoint,
                ready=False,
                detail="service runtime has not completed readiness checks",
            )

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._endpoints)

    def endpoint(self, name: str) -> ToolEndpoint:
        if not isinstance(name, str) or not _SERVER_NAME_RE.fullmatch(name):
            raise ToolRequestError(f"invalid task-local MCP server name: {name!r}")
        with self._lock:
            try:
                return self._endpoints[name]
            except KeyError as exc:
                raise ToolRequestError(
                    f"unknown task-local MCP server {name!r}"
                ) from exc

    def set_ready(
        self, name: str, ready: bool, detail: str | None = None
    ) -> ToolReadiness:
        with self._lock:
            endpoint = self.endpoint(name)
            state = ToolReadiness(endpoint=endpoint, ready=ready, detail=detail)
            self._readiness[name] = state
            return state

    def readiness(self, name: str) -> ToolReadiness:
        with self._lock:
            self.endpoint(name)
            return self._readiness[name]

    def require_ready(self, name: str) -> ToolEndpoint:
        with self._lock:
            state = self.readiness(name)
            if not state.ready:
                detail = f": {state.detail}" if state.detail else ""
                raise ToolRegistryError(
                    f"task-local MCP server {name!r} is not ready{detail}"
                )
            return state.endpoint

    @staticmethod
    def _declared_url(endpoint: ToolEndpoint) -> str:
        if endpoint.url is None:
            raise ToolRegistryError(
                f"task-local MCP server {endpoint.name!r} has no SSE URL"
            )
        return endpoint.url

    @staticmethod
    def _validated_resolved_url(endpoint: ToolEndpoint, resolved: str) -> str:
        if endpoint.url is None:
            raise ToolRegistryError(
                f"task-local MCP server {endpoint.name!r} has no SSE URL"
            )
        try:
            declared = urlsplit(endpoint.url)
            target = urlsplit(resolved)
            target_port = target.port
        except ValueError as exc:
            raise ToolRegistryError(
                f"task-local MCP server {endpoint.name!r} resolved an invalid URL"
            ) from exc
        if (
            target.scheme != "http"
            or target.username is not None
            or target.password is not None
            or target.query
            or target.fragment
            or target_port != declared.port
            or target.path != declared.path
        ):
            raise ToolRegistryError(
                f"task-local MCP server {endpoint.name!r} resolved outside its "
                "declared SSE endpoint"
            )
        hostname = target.hostname
        if hostname == endpoint.service:
            return resolved
        try:
            address = ipaddress.ip_address(hostname or "")
        except ValueError as exc:
            raise ToolRegistryError(
                f"task-local MCP server {endpoint.name!r} did not resolve to an IP"
            ) from exc
        if not (address.is_private or address.is_loopback):
            raise ToolRegistryError(
                f"task-local MCP server {endpoint.name!r} resolved to a non-private IP"
            )
        return resolved

    @classmethod
    def _secure_httpx_client_factory(
        cls,
        endpoint: ToolEndpoint,
        resolved: str,
    ) -> Callable[..., httpx.AsyncClient]:
        expected = urlsplit(cls._validated_resolved_url(endpoint, resolved))

        async def validate_request(request: httpx.Request) -> None:
            effective = urlsplit(str(request.url))
            try:
                effective_port = effective.port
            except ValueError as exc:
                raise ToolRegistryError(
                    f"task-local MCP server {endpoint.name!r} requested an "
                    "invalid effective endpoint"
                ) from exc
            if (
                effective.scheme != expected.scheme
                or effective.hostname != expected.hostname
                or effective_port != expected.port
                or effective.username is not None
                or effective.password is not None
                or request.method not in {"GET", "POST"}
                or (request.method == "GET" and effective.path != expected.path)
            ):
                raise ToolRegistryError(
                    f"task-local MCP server {endpoint.name!r} attempted an "
                    f"unapproved effective endpoint: {request.url}"
                )

        def factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            if headers or auth is not None:
                raise ToolRegistryError(
                    f"task-local MCP server {endpoint.name!r} attempted to add "
                    "HTTP credentials or headers"
                )
            return httpx.AsyncClient(
                event_hooks={"request": [validate_request]},
                follow_redirects=False,
                timeout=timeout,
                trust_env=False,
            )

        return factory

    @asynccontextmanager
    async def _session(self, endpoint: ToolEndpoint) -> AsyncIterator[Any]:
        if endpoint.transport != "sse":
            raise UnsupportedToolTransportError(
                f"task-local MCP server {endpoint.name!r} uses unsupported "
                f"transport {endpoint.transport!r}"
            )
        try:
            resolved = self._validated_resolved_url(
                endpoint, self._sse_url_resolver(endpoint)
            )
            async with self._sse_client_factory(
                resolved,
                headers=None,
                timeout=_CONNECT_TIMEOUT_S,
                sse_read_timeout=_READ_TIMEOUT_S,
                httpx_client_factory=self._secure_httpx_client_factory(
                    endpoint,
                    resolved,
                ),
            ) as streams:
                read_stream, write_stream = streams
                async with self._session_factory(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=_READ_TIMEOUT_S),
                ) as session:
                    with anyio.fail_after(_READ_TIMEOUT_S):
                        await session.initialize()
                    yield session
        except (ToolRegistryError, ToolRequestError):
            raise
        except Exception as exc:
            raise ToolRegistryError(
                f"task-local MCP server {endpoint.name!r} transport failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _bounded_json(
        value: Any,
        *,
        label: str,
        max_bytes: int,
        max_depth: int,
        max_nodes: int,
        require_object: bool = False,
    ) -> Any:
        if require_object and not isinstance(value, Mapping):
            raise ToolRequestError(f"{label} must be a JSON object")
        nodes = 0
        stack = [(value, 0)]
        while stack:
            current, depth = stack.pop()
            nodes += 1
            if nodes > max_nodes:
                raise ToolRequestError(f"{label} exceeds {max_nodes} JSON nodes")
            if depth > max_depth:
                raise ToolRequestError(f"{label} exceeds JSON depth {max_depth}")
            if current is None or isinstance(current, (str, bool, int)):
                continue
            if isinstance(current, float):
                if not math.isfinite(current):
                    raise ToolRequestError(f"{label} contains a non-finite number")
                continue
            if isinstance(current, Mapping):
                for key, item in current.items():
                    if not isinstance(key, str):
                        raise ToolRequestError(
                            f"{label} contains a non-string object key"
                        )
                    stack.append((item, depth + 1))
                continue
            if isinstance(current, list):
                stack.extend((item, depth + 1) for item in current)
                continue
            raise ToolRequestError(
                f"{label} contains non-JSON value {type(current).__name__}"
            )
        try:
            encoded = json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
            raise ToolRequestError(f"{label} is not valid JSON: {exc}") from exc
        if len(encoded) > max_bytes:
            raise ToolRequestError(f"{label} exceeds {max_bytes} encoded bytes")
        return json.loads(encoded)

    @staticmethod
    def _wire_mapping(value: Any, *, label: str) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            dumped = value.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            )
        elif isinstance(value, Mapping):
            dumped = dict(value)
        else:
            raise ToolRegistryError(
                f"{label} returned unsupported value {type(value).__name__}"
            )
        if not isinstance(dumped, dict):
            raise ToolRegistryError(f"{label} did not serialize to an object")
        dumped.pop("_meta", None)
        return dumped

    @classmethod
    def _sanitize_tool(cls, value: Any) -> dict[str, Any]:
        raw = cls._wire_mapping(value, label="tools/list")
        allowed = {
            "annotations",
            "description",
            "inputSchema",
            "name",
            "outputSchema",
            "title",
        }
        return {key: item for key, item in raw.items() if key in allowed}

    @classmethod
    def _sanitize_content(cls, value: Any) -> dict[str, Any]:
        raw = cls._wire_mapping(value, label="tools/call content")
        allowed = {
            "annotations",
            "data",
            "description",
            "mimeType",
            "name",
            "resource",
            "text",
            "title",
            "type",
            "uri",
        }
        return {key: item for key, item in raw.items() if key in allowed}

    @classmethod
    def _bounded_output(cls, value: Any, *, label: str) -> Any:
        try:
            return cls._bounded_json(
                value,
                label=label,
                max_bytes=_OUTPUT_MAX_BYTES,
                max_depth=_OUTPUT_MAX_DEPTH,
                max_nodes=_OUTPUT_MAX_NODES,
            )
        except ToolRequestError as exc:
            raise ToolRegistryError(str(exc)) from exc

    async def _list_tools_from_session(
        self, session: Any, server_name: str
    ) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _page in range(_MAX_TOOL_PAGES):
            with anyio.fail_after(_CALL_TIMEOUT_S):
                result = await session.list_tools(cursor=cursor)
            raw_tools = getattr(result, "tools", None)
            if not isinstance(raw_tools, list):
                raise ToolRegistryError(
                    f"task-local MCP server {server_name!r} returned malformed "
                    "tools/list"
                )
            tools.extend(self._sanitize_tool(tool) for tool in raw_tools)
            if len(tools) > _MAX_TOOLS:
                raise ToolRegistryError(
                    f"task-local MCP server {server_name!r} returned too many tools"
                )
            cursor = getattr(result, "nextCursor", None)
            if cursor is None:
                cursor = getattr(result, "next_cursor", None)
            if not cursor:
                break
            if not isinstance(cursor, str) or len(cursor.encode()) > 1024:
                raise ToolRegistryError(
                    f"task-local MCP server {server_name!r} returned an invalid cursor"
                )
        else:
            raise ToolRegistryError(
                f"task-local MCP server {server_name!r} exceeded pagination limit"
            )
        return tools

    async def list_tools(self, name: str) -> dict[str, Any]:
        """Connect to one ready server and proxy only MCP ``tools/list``."""
        endpoint = self.require_ready(name)
        async with self._session(endpoint) as session:
            tools = await self._list_tools_from_session(session, name)
        return self._bounded_output({"tools": tools}, label="tools/list output")

    async def call_tool(
        self,
        name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Connect to one ready server and proxy only MCP ``tools/call``."""
        endpoint = self.require_ready(name)
        if not isinstance(tool_name, str) or not _REMOTE_TOOL_NAME_RE.fullmatch(
            tool_name
        ):
            raise ToolRequestError(f"invalid MCP tool name: {tool_name!r}")
        bounded_arguments = self._bounded_json(
            {} if arguments is None else arguments,
            label="MCP tool arguments",
            max_bytes=_INPUT_MAX_BYTES,
            max_depth=_INPUT_MAX_DEPTH,
            max_nodes=_INPUT_MAX_NODES,
            require_object=True,
        )
        async with self._session(endpoint) as session:
            available_tools = await self._list_tools_from_session(session, name)
            if tool_name not in {
                tool.get("name")
                for tool in available_tools
                if isinstance(tool.get("name"), str)
            }:
                raise ToolRequestError(
                    f"unknown tool {tool_name!r} on task-local MCP server {name!r}"
                )
            with anyio.fail_after(_CALL_TIMEOUT_S):
                result = await session.call_tool(
                    tool_name,
                    bounded_arguments,
                    read_timeout_seconds=timedelta(seconds=_CALL_TIMEOUT_S),
                )
        raw_content = getattr(result, "content", None)
        if not isinstance(raw_content, list):
            raise ToolRegistryError(
                f"task-local MCP server {name!r} returned malformed tools/call"
            )
        payload: dict[str, Any] = {
            "content": [self._sanitize_content(item) for item in raw_content],
            "isError": bool(
                getattr(result, "isError", getattr(result, "is_error", False))
            ),
        }
        structured = getattr(
            result,
            "structuredContent",
            getattr(result, "structured_content", None),
        )
        if structured is not None:
            payload["structuredContent"] = structured
        return self._bounded_output(payload, label="tools/call output")
