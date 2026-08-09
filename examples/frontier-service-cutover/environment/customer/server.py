from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

STATE_URL = os.environ.get("STATE_URL", "http://state:7000")
API_BASE = os.environ.get("API_BASE", "http://main:8080")
SHARED_RESULT = Path(os.environ.get("SHARED_RESULT", "/shared/customer-results.json"))
EVENT_RECORDS = (
    {"id": 3, "legacy": "blue-old", "current": "blue"},
    {"id": 4, "legacy": "green-old", "current": "green"},
    {"id": 5, "legacy": "violet-old", "current": "violet"},
    {"id": 6, "legacy": "white-old", "current": "white"},
)
EXPECTED_RECORDS = (
    {"id": 1, "legacy": "red-old", "current": "red"},
    {"id": 2, "legacy": "gold-old", "current": "gold"},
    *EVENT_RECORDS,
)


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


def _post_json(url: str, value: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=_encode(value),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError("service returned a non-object")
    return payload


def _lookup(record_id: int) -> dict[str, Any]:
    query = urllib.parse.urlencode({"id": record_id})
    with urllib.request.urlopen(f"{API_BASE}/lookup?{query}", timeout=3) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError("main service returned a non-object")
    return payload


class LoadRun:
    def __init__(self) -> None:
        self._finalize_lock = threading.Lock()
        self._result: dict[str, Any] | None = None
        self._threads = [
            threading.Thread(
                target=self._write_records,
                args=(EVENT_RECORDS[offset::2],),
                daemon=True,
                name=f"synthetic-load-{offset}",
            )
            for offset in range(2)
        ]

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    @staticmethod
    def _write_records(records: tuple[dict[str, Any], ...]) -> None:
        for record in records:
            for _attempt in range(50):
                try:
                    _post_json(f"{STATE_URL}/events", record)
                    break
                except OSError:
                    time.sleep(0.05)
            for _attempt in range(10):
                try:
                    _lookup(record["id"])
                    break
                except OSError:
                    time.sleep(0.05)

    @staticmethod
    def _final_check(record: dict[str, Any]) -> dict[str, Any]:
        observed: dict[str, Any] | None = None
        for _attempt in range(50):
            try:
                observed = _lookup(record["id"])
                break
            except OSError:
                time.sleep(0.05)
        expected = {"id": record["id"], "value": record["current"]}
        return {
            "id": record["id"],
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
        }

    def finalize(self) -> dict[str, Any]:
        with self._finalize_lock:
            if self._result is not None:
                return self._result
            for thread in self._threads:
                thread.join(timeout=5)
            checks = [self._final_check(record) for record in EXPECTED_RECORDS]
            passed = sum(int(check["passed"]) for check in checks)
            self._result = {
                "schema_version": "cutover-customer-results.v1",
                "finalized": True,
                "passed": passed,
                "total": len(checks),
                "checks": checks,
            }
            _atomic_write(SHARED_RESULT, self._result)
            return self._result

    def result(self) -> dict[str, Any]:
        with self._finalize_lock:
            return self._result or {
                "schema_version": "cutover-customer-results.v1",
                "finalized": False,
                "passed": 0,
                "total": len(EXPECTED_RECORDS),
                "checks": [],
            }


RUN = LoadRun()


class Handler(BaseHTTPRequestHandler):
    server_version = "SyntheticCustomer/1"

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = _encode(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
        elif self.path == "/results":
            self._json(HTTPStatus.OK, RUN.result())
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path == "/finalize":
            self._json(HTTPStatus.OK, RUN.finalize())
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, _format: str, *_args: object) -> None:
        return


def capture(destination: Path) -> None:
    self_url = os.environ.get("CUSTOMER_SELF_URL", "http://127.0.0.1:9000")
    with urllib.request.urlopen(f"{self_url}/results", timeout=5) as response:
        value = json.load(response)
    if not isinstance(value, dict) or value.get("finalized") is not True:
        raise ValueError("customer run was not finalized")
    _atomic_write(destination, value)


def serve() -> None:
    RUN.start()
    port = int(os.environ.get("CUSTOMER_PORT", "9000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path)
    arguments = parser.parse_args()
    if arguments.capture is None:
        serve()
    else:
        capture(arguments.capture)
