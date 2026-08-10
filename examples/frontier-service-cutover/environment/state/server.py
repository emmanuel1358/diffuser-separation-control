from __future__ import annotations

import argparse
import json
import os
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

STATE_PATH = Path(os.environ.get("STATE_PATH", "/data/state.json"))
LOCK = threading.RLock()


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


def _empty_state() -> dict[str, Any]:
    return {"schema_version": "cutover-state.v1", "records": []}


def _load() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return _empty_state()
    value = json.loads(STATE_PATH.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "cutover-state.v1"
        or not isinstance(value.get("records"), list)
    ):
        raise ValueError("persisted state has the wrong schema")
    return value


def _replace_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = sorted(
        (
            {
                "id": int(record["id"]),
                "legacy": str(record["legacy"]),
                "current": str(record["current"]),
            }
            for record in records
        ),
        key=lambda record: record["id"],
    )
    state = {"schema_version": "cutover-state.v1", "records": normalized}
    _atomic_write(STATE_PATH, state)
    return state


def _upsert(record: dict[str, Any]) -> dict[str, Any]:
    state = _load()
    by_id = {int(item["id"]): item for item in state["records"]}
    normalized = {
        "id": int(record["id"]),
        "legacy": str(record["legacy"]),
        "current": str(record["current"]),
    }
    by_id[normalized["id"]] = normalized
    return _replace_records(list(by_id.values()))


def capture(destination: Path) -> None:
    with LOCK:
        state = _load()
        _atomic_write(destination, state)


class Handler(BaseHTTPRequestHandler):
    server_version = "SyntheticState/1"

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = _encode(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 65536:
            raise ValueError("invalid request size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise TypeError("request must be an object")
        return value

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path == "/snapshot":
            with LOCK:
                self._json(HTTPStatus.OK, _load())
            return
        if parsed.path != "/record":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            record_id = int(urllib.parse.parse_qs(parsed.query)["id"][0])
            with LOCK:
                record = next(
                    item for item in _load()["records"] if item["id"] == record_id
                )
            self._json(HTTPStatus.OK, record)
        except (KeyError, StopIteration, TypeError, ValueError):
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown_record"})

    def do_POST(self) -> None:
        try:
            value = self._request_json()
            with LOCK:
                if self.path == "/seed":
                    records = value.get("records")
                    if not isinstance(records, list):
                        raise ValueError("seed records must be a list")
                    state = _replace_records(records)
                elif self.path == "/events":
                    state = _upsert(value)
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
            self._json(HTTPStatus.OK, state)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_record"})

    def log_message(self, _format: str, *_args: object) -> None:
        return


def serve() -> None:
    with LOCK:
        if not STATE_PATH.exists():
            _atomic_write(STATE_PATH, _empty_state())
    port = int(os.environ.get("STATE_PORT", "7000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path)
    arguments = parser.parse_args()
    if arguments.capture is None:
        serve()
    else:
        capture(arguments.capture)
