from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

TARGET_FIELD = "current"


def transform(record: dict[str, Any]) -> dict[str, Any]:
    return {"id": int(record["id"]), "value": str(record[TARGET_FIELD])}


def _state_record(record_id: int) -> dict[str, Any]:
    state_url = os.environ.get("STATE_URL", "http://state:7000")
    query = urllib.parse.urlencode({"id": record_id})
    with urllib.request.urlopen(f"{state_url}/record?{query}", timeout=3) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError("state service returned a non-object")
    return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "SyntheticCutover/1"

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path != "/lookup":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            raw_id = urllib.parse.parse_qs(parsed.query)["id"][0]
            record = _state_record(int(raw_id))
            self._json(HTTPStatus.OK, transform(record))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": type(exc).__name__})

    def log_message(self, _format: str, *_args: object) -> None:
        return


def serve() -> None:
    port = int(os.environ.get("APP_PORT", "8080"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    arguments = parser.parse_args()
    if arguments.serve:
        serve()
