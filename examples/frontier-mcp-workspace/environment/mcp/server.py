from __future__ import annotations

import json
import os
import queue
import secrets
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SHARED_PATH = Path(os.environ.get("SHARED_PATH", "/shared/workspace.json"))
SESSIONS: dict[str, queue.Queue[dict[str, Any]]] = {}
SESSIONS_LOCK = threading.Lock()
TOOLS = [
    {
        "name": "browser_navigate",
        "description": "Navigate the synthetic browser to the task workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "browser_snapshot",
        "description": "Return a deterministic accessibility-style workspace snapshot.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "browser_click",
        "description": "Acknowledge a synthetic element click.",
        "inputSchema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}},
            "required": ["ref"],
            "additionalProperties": False,
        },
    },
]


def _wire(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _snapshot() -> dict[str, Any]:
    value = json.loads(SHARED_PATH.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "synthetic-claims-workspace.v1"
        or not isinstance(value.get("cases"), list)
    ):
        raise ValueError("shared workspace has the wrong schema")
    return {
        "title": str(value.get("title", "")),
        "case_count": len(value["cases"]),
        "tree": ["main", "heading Synthetic Claims Inbox", "list Claims"],
    }


def _tool_result(name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be an object")
    if name == "browser_navigate":
        url = arguments.get("url")
        if not isinstance(url, str):
            raise ValueError("browser_navigate requires url")
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "workspace"
            or parsed.port != 18073
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("browser_navigate only accepts the task workspace URL")
        structured = {"url": url, "title": _snapshot()["title"]}
        text = f"Navigated to {url}"
    elif name == "browser_snapshot":
        structured = _snapshot()
        text = "\n".join(structured["tree"])
    elif name == "browser_click":
        reference = arguments.get("ref")
        if not isinstance(reference, str) or not reference:
            raise ValueError("browser_click requires ref")
        structured = {"clicked": reference}
        text = f"Clicked {reference}"
    else:
        raise KeyError(name)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
        "isError": False,
    }


def _response(request: Any) -> dict[str, Any] | None:
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request"},
        }
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params")
    if request_id is None:
        return None
    try:
        if method == "initialize":
            requested = params if isinstance(params, dict) else {}
            result: Any = {
                "protocolVersion": requested.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "synthetic-browser-mcp",
                    "version": "1.0.0",
                },
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            if not isinstance(params, dict):
                raise ValueError("tools/call params must be an object")
            result = _tool_result(params.get("name"), params.get("arguments", {}))
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Method not found"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except KeyError:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": "Unknown tool"},
        }
    except (TypeError, ValueError) as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": str(exc)},
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SyntheticMCP/1"

    def do_GET(self) -> None:
        if self.path == "/health":
            body = b'{"status":"ok"}\n'
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if urllib.parse.urlsplit(self.path).path != "/sse":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        session_id = secrets.token_hex(16)
        messages: queue.Queue[dict[str, Any]] = queue.Queue()
        with SESSIONS_LOCK:
            SESSIONS[session_id] = messages
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            self._event("endpoint", f"/messages/?session_id={session_id}")
            while True:
                try:
                    message = messages.get(timeout=10)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self._event("message", _wire(message).decode())
        except (BrokenPipeError, ConnectionResetError, OSError):
            return
        finally:
            with SESSIONS_LOCK:
                SESSIONS.pop(session_id, None)

    def _event(self, event: str, data: str) -> None:
        payload = f"event: {event}\ndata: {data}\n\n".encode()
        self.wfile.write(payload)
        self.wfile.flush()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path not in {"/messages", "/messages/"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        session_ids = urllib.parse.parse_qs(parsed.query).get("session_id", [])
        length = int(self.headers.get("Content-Length", "0"))
        if len(session_ids) != 1 or not 0 < length <= 262144:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        with SESSIONS_LOCK:
            messages = SESSIONS.get(session_ids[0])
        if messages is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            request = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        response = _response(request)
        self.send_response(HTTPStatus.ACCEPTED)
        self.send_header("Content-Length", "0")
        self.end_headers()
        if response is not None:
            messages.put(response)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class Server(ThreadingHTTPServer):
    daemon_threads = True


def main() -> None:
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "3080"))
    Server((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
