from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SHARED_PATH = Path(os.environ.get("SHARED_PATH", "/shared/workspace.json"))
WORKSPACE = {
    "schema_version": "synthetic-claims-workspace.v1",
    "title": "Synthetic Claims Inbox",
    "cases": [
        {
            "claim_id": "V-001",
            "covered": True,
            "billed_cents": 8500,
            "contract_cap_cents": 6000,
        },
        {
            "claim_id": "V-002",
            "covered": False,
            "billed_cents": 3000,
            "contract_cap_cents": 5000,
        },
    ],
}


def _encode(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_encode(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_shared() -> dict[str, Any]:
    value = json.loads(SHARED_PATH.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "synthetic-claims-workspace.v1"
        or not isinstance(value.get("cases"), list)
    ):
        raise ValueError("shared workspace has the wrong schema")
    return value


def capture(destination: Path) -> None:
    _atomic_write(destination, _load_shared())


class Handler(BaseHTTPRequestHandler):
    server_version = "SyntheticWorkspace/1"

    def _bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._bytes(HTTPStatus.OK, "application/json", b'{"status":"ok"}\n')
            return
        if self.path == "/api/cases":
            self._bytes(HTTPStatus.OK, "application/json", _encode(_load_shared()))
            return
        if self.path == "/":
            title = _load_shared()["title"]
            body = (
                "<!doctype html><html><head><title>"
                + str(title)
                + "</title></head><body><main><h1>"
                + str(title)
                + "</h1><p>Deterministic browser mock target.</p></main></body></html>"
            ).encode()
            self._bytes(HTTPStatus.OK, "text/html; charset=utf-8", body)
            return
        self._bytes(
            HTTPStatus.NOT_FOUND, "application/json", b'{"error":"not_found"}\n'
        )

    def log_message(self, _format: str, *_args: object) -> None:
        return


def serve() -> None:
    _atomic_write(SHARED_PATH, WORKSPACE)
    port = int(os.environ.get("WORKSPACE_PORT", "18073"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path)
    arguments = parser.parse_args()
    if arguments.capture is None:
        serve()
    else:
        capture(arguments.capture)
