"""Trusted hidden driver executed only through the candidate process boundary."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time

MAX_INPUT_BYTES = 64 * 1024
MAX_WORKER_OUTPUT_BYTES = 4096
WORKER_TIMEOUT_SECONDS = 5.0

WORKER_SOURCE = r"""
from __future__ import annotations

import json
import os
import sys

protocol_fd = os.dup(1)
null_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(null_fd, 1)
os.dup2(null_fd, 2)
os.close(null_fd)

loads = json.loads
dumps = json.dumps
write = os.write

request = loads(sys.stdin.buffer.read(65536).decode("utf-8", errors="strict"))
sys.path.insert(0, os.getcwd())

try:
    from normalizer import normalize_slug

    actual = normalize_slug(request["value"])
except BaseException as exc:
    response = {"ok": False, "error": type(exc).__name__}
else:
    response = {"ok": type(actual) is str, "actual": actual}

encoded = (
    dumps(
        response,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    + "\n"
).encode("utf-8")
write(protocol_fd, encoded)
os.close(protocol_fd)
"""


def _strict_object(data: bytes) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        data.decode("utf-8", errors="strict"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value {item!r}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError("JSON document must be an object")
    return value


def _validated_cases(data: bytes) -> list[dict[str, str]]:
    if not data or len(data) > MAX_INPUT_BYTES:
        raise ValueError("hidden case payload has an invalid size")
    payload = _strict_object(data)
    if set(payload) != {"schema_version", "cases"}:
        raise ValueError("hidden case payload has unexpected fields")
    if payload["schema_version"] != "slug-normalizer-cases.v1":
        raise ValueError("hidden case payload has an unsupported schema")
    cases = payload["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 64:
        raise ValueError("hidden case payload must contain 1..64 cases")
    for case in cases:
        if (
            not isinstance(case, dict)
            or set(case) != {"input", "expected"}
            or not isinstance(case["input"], str)
            or not isinstance(case["expected"], str)
            or len(case["input"]) > 4096
            or len(case["expected"]) > 4096
        ):
            raise ValueError("hidden case payload contains an invalid case")
    return cases


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _bounded_worker(request: bytes) -> bytes | None:
    environment = {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", WORKER_SOURCE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd(),
            env=environment,
            close_fds=True,
            start_new_session=True,
        )
    except OSError:
        return None

    assert process.stdin is not None
    assert process.stdout is not None
    captured = bytearray()
    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    try:
        process.stdin.write(request)
        process.stdin.close()
        descriptor = process.stdout.fileno()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                return None
            readable, _, _ = select.select([descriptor], [], [], remaining)
            if not readable:
                _kill_process_group(process)
                return None
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            captured.extend(chunk)
            if len(captured) > MAX_WORKER_OUTPUT_BYTES:
                _kill_process_group(process)
                return None
        try:
            returncode = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            return None
        return bytes(captured) if returncode == 0 else None
    except (BrokenPipeError, OSError):
        _kill_process_group(process)
        return None
    finally:
        _kill_process_group(process)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()


def _case_passes(case: dict[str, str]) -> bool:
    request = json.dumps(
        {"value": case["input"]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    output = _bounded_worker(request)
    if output is None:
        return False
    try:
        response = _strict_object(output)
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
    return (
        set(response) == {"ok", "actual"}
        and response["ok"] is True
        and type(response["actual"]) is str
        and response["actual"] == case["expected"]
    )


def main() -> None:
    data = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    cases = _validated_cases(data)
    passed = sum(_case_passes(case) for case in cases)
    result = {
        "schema_version": "slug-normalizer-result.v1",
        "passed": passed,
        "total": len(cases),
    }
    sys.stdout.write(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
