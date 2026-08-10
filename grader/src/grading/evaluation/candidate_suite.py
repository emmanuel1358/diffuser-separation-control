"""Deterministic, bounded command suites for committed workspace candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from grading.faults import AgentFault, GraderFault

_MAX_ARGV_ITEMS = 256
_MAX_ARGV_BYTES = 64 * 1024
_MAX_ENV_ITEMS = 256
_MAX_ENV_BYTES = 64 * 1024
_MAX_STDIN_BYTES = 64 * 1024 * 1024
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_SUITE_OUTPUT_BYTES = 128 * 1024 * 1024
_MAX_REPEATS = 32
_MAX_TIMEOUT_S = 3600.0
_MAX_JSON_LINES = 100_000
_MAX_JSON_NODES = 1_000_000
_MAX_JSON_DEPTH = 128
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CandidateSuiteContext(Protocol):
    """The RubricContext surface used by this module."""

    def run_candidate(
        self,
        cmd: list[str],
        *,
        stdin_bytes: bytes | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float = 120.0,
        max_output_bytes: int = 16 * 1024 * 1024,
    ) -> Any: ...

    def candidate_operation(
        self,
        label: str,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...


def _encoded_size(value: str, *, label: str) -> int:
    if "\0" in value:
        raise ValueError(f"{label} must not contain NUL")
    try:
        return len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid UTF-8") from exc


def _finite_positive(value: float, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a positive finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite number") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return normalized


@dataclass(frozen=True)
class CandidateCommandSpec:
    """One immutable, repeatable command run through the candidate boundary."""

    argv: tuple[str, ...]
    cwd: str = "."
    env: tuple[tuple[str, str], ...] = ()
    stdin_bytes: bytes | None = None
    timeout_s: float = 120.0
    max_output_bytes: int = 16 * 1024 * 1024
    repeats: int = 1
    allowed_return_codes: tuple[int, ...] = (0,)
    deterministic_stdout: bool = False
    max_attempt_elapsed_s: float | None = None
    max_total_elapsed_s: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.argv, (str, bytes)):
            raise TypeError("candidate argv must be a sequence of strings")
        argv = tuple(self.argv)
        if not argv or len(argv) > _MAX_ARGV_ITEMS:
            raise ValueError(
                f"candidate argv must contain 1..{_MAX_ARGV_ITEMS} entries"
            )
        if any(not isinstance(part, str) or not part for part in argv):
            raise ValueError("candidate argv entries must be non-empty strings")
        argv_bytes = sum(
            _encoded_size(part, label="candidate argv entry") for part in argv
        )
        if argv_bytes > _MAX_ARGV_BYTES:
            raise ValueError(f"candidate argv exceeds {_MAX_ARGV_BYTES} bytes")

        if not isinstance(self.cwd, str) or not self.cwd:
            raise ValueError("candidate cwd must be a non-empty relative path")
        cwd_path = Path(self.cwd)
        if cwd_path.is_absolute() or ".." in cwd_path.parts:
            raise ValueError("candidate cwd must stay relative to the workspace")
        cwd = cwd_path.as_posix() or "."
        if _encoded_size(cwd, label="candidate cwd") > 4096:
            raise ValueError("candidate cwd exceeds 4096 bytes")

        raw_env: Iterable[Any]
        if isinstance(self.env, Mapping):
            raw_env = self.env.items()
        else:
            raw_env = self.env
        raw_env_items = tuple(raw_env)
        if len(raw_env_items) > _MAX_ENV_ITEMS:
            raise ValueError(f"candidate env has more than {_MAX_ENV_ITEMS} entries")
        normalized_env: list[tuple[str, str]] = []
        keys: list[str] = []
        env_bytes = 0
        for item in raw_env_items:
            if isinstance(item, (str, bytes)):
                raise TypeError("candidate env entries must be key/value pairs")
            try:
                key, value = item
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "candidate env entries must be key/value pairs"
                ) from exc
            if not isinstance(key, str) or not _ENV_KEY_RE.fullmatch(key):
                raise ValueError(f"invalid candidate environment key {key!r}")
            if not isinstance(value, str):
                raise TypeError(f"candidate environment value for {key!r} is not text")
            normalized_env.append((key, value))
            keys.append(key)
            env_bytes += _encoded_size(key, label="candidate environment key")
            env_bytes += _encoded_size(
                value,
                label=f"candidate environment value for {key!r}",
            )
        if len(keys) != len(set(keys)):
            raise ValueError("candidate env keys must be unique")
        if env_bytes > _MAX_ENV_BYTES:
            raise ValueError(f"candidate env exceeds {_MAX_ENV_BYTES} bytes")
        env = tuple(sorted(normalized_env))

        stdin_bytes = self.stdin_bytes
        if stdin_bytes is not None:
            if not isinstance(stdin_bytes, bytes):
                raise ValueError("candidate stdin_bytes must be bytes or None")
            if len(stdin_bytes) > _MAX_STDIN_BYTES:
                raise ValueError(f"candidate stdin exceeds {_MAX_STDIN_BYTES} bytes")

        timeout_s = _finite_positive(self.timeout_s, label="candidate timeout_s")
        if timeout_s > _MAX_TIMEOUT_S:
            raise ValueError(f"candidate timeout_s exceeds {_MAX_TIMEOUT_S:g}")
        if (
            not isinstance(self.max_output_bytes, int)
            or isinstance(self.max_output_bytes, bool)
            or not 1 <= self.max_output_bytes <= _MAX_OUTPUT_BYTES
        ):
            raise ValueError(
                f"candidate max_output_bytes must be in 1..{_MAX_OUTPUT_BYTES}"
            )
        if (
            not isinstance(self.repeats, int)
            or isinstance(self.repeats, bool)
            or not 1 <= self.repeats <= _MAX_REPEATS
        ):
            raise ValueError(f"candidate repeats must be in 1..{_MAX_REPEATS}")
        if self.repeats * self.max_output_bytes > _MAX_SUITE_OUTPUT_BYTES:
            raise ValueError(
                f"candidate suite capture exceeds {_MAX_SUITE_OUTPUT_BYTES} bytes"
            )
        if not isinstance(self.deterministic_stdout, bool):
            raise TypeError("candidate deterministic_stdout must be a bool")

        if isinstance(self.allowed_return_codes, (str, bytes)):
            raise TypeError("candidate allowed_return_codes must be integers")
        raw_return_codes = tuple(self.allowed_return_codes)
        if (
            not raw_return_codes
            or len(raw_return_codes) > 32
            or any(
                not isinstance(code, int)
                or isinstance(code, bool)
                or not 0 <= code <= 255
                for code in raw_return_codes
            )
        ):
            raise ValueError(
                "candidate allowed_return_codes must contain 1..32 integers in 0..255"
            )
        allowed_return_codes = tuple(sorted(set(raw_return_codes)))

        max_attempt_elapsed_s = self.max_attempt_elapsed_s
        if max_attempt_elapsed_s is not None:
            max_attempt_elapsed_s = _finite_positive(
                max_attempt_elapsed_s,
                label="candidate max_attempt_elapsed_s",
            )
            if max_attempt_elapsed_s > _MAX_TIMEOUT_S:
                raise ValueError(
                    f"candidate max_attempt_elapsed_s exceeds {_MAX_TIMEOUT_S:g}"
                )
        max_total_elapsed_s = self.max_total_elapsed_s
        if max_total_elapsed_s is not None:
            max_total_elapsed_s = _finite_positive(
                max_total_elapsed_s,
                label="candidate max_total_elapsed_s",
            )
            if max_total_elapsed_s > _MAX_TIMEOUT_S * _MAX_REPEATS:
                raise ValueError(
                    "candidate max_total_elapsed_s exceeds the suite maximum"
                )

        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "env", env)
        object.__setattr__(self, "timeout_s", timeout_s)
        object.__setattr__(self, "allowed_return_codes", allowed_return_codes)
        object.__setattr__(
            self,
            "max_attempt_elapsed_s",
            max_attempt_elapsed_s,
        )
        object.__setattr__(self, "max_total_elapsed_s", max_total_elapsed_s)

    def spec_dict(self) -> dict[str, Any]:
        """Stable JSON-serializable plan fragment without embedding stdin bytes."""
        return {
            "schema_version": "candidate-command-suite.v1",
            "argv": list(self.argv),
            "cwd": self.cwd,
            "env": {key: value for key, value in self.env},
            "stdin_bytes": (
                None
                if self.stdin_bytes is None
                else {
                    "size": len(self.stdin_bytes),
                    "sha256": hashlib.sha256(self.stdin_bytes).hexdigest(),
                }
            ),
            "timeout_s": self.timeout_s,
            "max_output_bytes": self.max_output_bytes,
            "repeats": self.repeats,
            "allowed_return_codes": list(self.allowed_return_codes),
            "deterministic_stdout": self.deterministic_stdout,
            "max_attempt_elapsed_s": self.max_attempt_elapsed_s,
            "max_total_elapsed_s": self.max_total_elapsed_s,
        }


@dataclass(frozen=True)
class CandidateAttempt:
    """One bounded candidate process outcome."""

    index: int
    returncode: int
    stdout: bytes
    elapsed_s: float

    @property
    def stdout_sha256(self) -> str:
        return hashlib.sha256(self.stdout).hexdigest()

    def parse_json(self, context: CandidateSuiteContext, **kwargs: Any) -> Any:
        return parse_candidate_json(context, self.stdout, **kwargs)

    def parse_jsonl(
        self,
        context: CandidateSuiteContext,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        return parse_candidate_jsonl(context, self.stdout, **kwargs)


@dataclass(frozen=True)
class CandidateSuiteMetadata:
    """Aggregate process metadata; candidate stdout is never interpreted as score."""

    attempt_count: int
    total_elapsed_s: float
    min_elapsed_s: float
    max_elapsed_s: float
    mean_elapsed_s: float
    return_codes: tuple[int, ...]
    stdout_sha256: tuple[str, ...]
    deterministic_stdout: bool


@dataclass(frozen=True)
class CandidateSuiteResult:
    attempts: tuple[CandidateAttempt, ...]
    metadata: CandidateSuiteMetadata


def run_candidate_suite(
    context: CandidateSuiteContext,
    spec: CandidateCommandSpec,
) -> CandidateSuiteResult:
    """Run every attempt through ``RubricContext.run_candidate``."""
    if not isinstance(spec, CandidateCommandSpec):
        raise GraderFault("candidate suite requires a CandidateCommandSpec")

    suite_started = time.monotonic()
    attempts: list[CandidateAttempt] = []
    expected_stdout: bytes | None = None
    env = dict(spec.env) if spec.env else None

    for index in range(spec.repeats):
        attempt_started = time.monotonic()
        effective_timeout_s = spec.timeout_s
        if spec.max_attempt_elapsed_s is not None:
            effective_timeout_s = min(
                effective_timeout_s,
                spec.max_attempt_elapsed_s,
            )
        if spec.max_total_elapsed_s is not None:
            remaining_s = spec.max_total_elapsed_s - (attempt_started - suite_started)
            if remaining_s <= 0.0:
                raise AgentFault(
                    "candidate suite exceeded the "
                    f"{spec.max_total_elapsed_s:g}s total performance budget"
                )
            effective_timeout_s = min(effective_timeout_s, remaining_s)
        completed = context.run_candidate(
            list(spec.argv),
            stdin_bytes=spec.stdin_bytes,
            cwd=spec.cwd,
            env=env,
            timeout_s=effective_timeout_s,
            max_output_bytes=spec.max_output_bytes,
        )
        attempt_finished = time.monotonic()
        elapsed_s = attempt_finished - attempt_started

        if completed.returncode not in spec.allowed_return_codes:
            raise AgentFault(
                f"candidate suite attempt {index + 1} returned "
                f"{completed.returncode}, expected one of "
                f"{list(spec.allowed_return_codes)}"
            )
        if not isinstance(completed.stdout, bytes):
            raise AgentFault(
                f"candidate suite attempt {index + 1} returned non-bytes stdout"
            )
        if len(completed.stdout) > spec.max_output_bytes:
            raise AgentFault(
                f"candidate suite attempt {index + 1} exceeded the stdout limit"
            )
        if (
            spec.max_attempt_elapsed_s is not None
            and elapsed_s > spec.max_attempt_elapsed_s
        ):
            raise AgentFault(
                f"candidate suite attempt {index + 1} exceeded the "
                f"{spec.max_attempt_elapsed_s:g}s performance budget"
            )
        total_elapsed_s = attempt_finished - suite_started
        if (
            spec.max_total_elapsed_s is not None
            and total_elapsed_s > spec.max_total_elapsed_s
        ):
            raise AgentFault(
                "candidate suite exceeded the "
                f"{spec.max_total_elapsed_s:g}s total performance budget"
            )

        stdout = bytes(completed.stdout)
        if spec.deterministic_stdout:
            if expected_stdout is None:
                expected_stdout = stdout
            elif stdout != expected_stdout:
                raise AgentFault(
                    f"candidate suite stdout changed on attempt {index + 1}"
                )
        attempts.append(
            CandidateAttempt(
                index=index,
                returncode=completed.returncode,
                stdout=stdout,
                elapsed_s=elapsed_s,
            )
        )

    finished = time.monotonic()
    total_elapsed_s = finished - suite_started
    if (
        spec.max_total_elapsed_s is not None
        and total_elapsed_s > spec.max_total_elapsed_s
    ):
        raise AgentFault(
            "candidate suite exceeded the "
            f"{spec.max_total_elapsed_s:g}s total performance budget"
        )
    elapsed = tuple(attempt.elapsed_s for attempt in attempts)
    attempt_tuple = tuple(attempts)
    metadata = CandidateSuiteMetadata(
        attempt_count=len(attempt_tuple),
        total_elapsed_s=total_elapsed_s,
        min_elapsed_s=min(elapsed),
        max_elapsed_s=max(elapsed),
        mean_elapsed_s=sum(elapsed) / len(elapsed),
        return_codes=tuple(attempt.returncode for attempt in attempt_tuple),
        stdout_sha256=tuple(attempt.stdout_sha256 for attempt in attempt_tuple),
        deterministic_stdout=(len({attempt.stdout for attempt in attempt_tuple}) == 1),
    )
    return CandidateSuiteResult(attempts=attempt_tuple, metadata=metadata)


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _loads_strict_json(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_json_object_without_duplicates,
        parse_constant=_reject_json_constant,
    )


def _bounded_json_shape(value: Any, *, max_depth: int, max_nodes: int) -> int:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise ValueError(f"JSON output exceeds the {max_nodes}-node limit")
        if depth > max_depth:
            raise ValueError(f"JSON output exceeds the {max_depth}-level depth limit")
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    return nodes


def _validate_json_limits(
    *,
    max_bytes: int,
    max_depth: int,
    max_nodes: int,
) -> None:
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or not 1 <= max_bytes <= _MAX_OUTPUT_BYTES
    ):
        raise GraderFault(f"candidate JSON max_bytes must be in 1..{_MAX_OUTPUT_BYTES}")
    if (
        not isinstance(max_depth, int)
        or isinstance(max_depth, bool)
        or not 1 <= max_depth <= _MAX_JSON_DEPTH
    ):
        raise GraderFault(f"candidate JSON max_depth must be in 1..{_MAX_JSON_DEPTH}")
    if (
        not isinstance(max_nodes, int)
        or isinstance(max_nodes, bool)
        or not 1 <= max_nodes <= _MAX_JSON_NODES
    ):
        raise GraderFault(f"candidate JSON max_nodes must be in 1..{_MAX_JSON_NODES}")


def _parse_candidate_json(
    data: bytes,
    *,
    max_bytes: int,
    max_depth: int,
    max_nodes: int,
    require_object: bool,
) -> Any:
    if not isinstance(data, bytes):
        raise TypeError("candidate JSON output must be bytes")
    if len(data) > max_bytes:
        raise ValueError(f"candidate JSON output exceeds {max_bytes} bytes")
    document = _loads_strict_json(data.decode("utf-8", errors="strict"))
    _bounded_json_shape(document, max_depth=max_depth, max_nodes=max_nodes)
    if require_object and not isinstance(document, dict):
        raise ValueError("candidate JSON output must be an object")
    return document


def parse_candidate_json(
    context: CandidateSuiteContext,
    data: bytes,
    *,
    max_bytes: int = 16 * 1024 * 1024,
    max_depth: int = 64,
    max_nodes: int = 100_000,
    require_object: bool = False,
) -> Any:
    """Strictly parse bounded UTF-8 JSON under ``candidate_operation``."""
    _validate_json_limits(
        max_bytes=max_bytes,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )
    if not isinstance(require_object, bool):
        raise GraderFault("candidate JSON require_object must be a bool")
    return context.candidate_operation(
        "candidate JSON output",
        _parse_candidate_json,
        data,
        max_bytes=max_bytes,
        max_depth=max_depth,
        max_nodes=max_nodes,
        require_object=require_object,
    )


def _parse_candidate_jsonl(
    data: bytes,
    *,
    max_bytes: int,
    max_lines: int,
    max_depth: int,
    max_nodes: int,
    expected_lines: int | None,
    allow_extra_lines: bool,
    reject_duplicate_lines: bool,
    require_object: bool,
) -> tuple[Any, ...]:
    if not isinstance(data, bytes):
        raise TypeError("candidate JSONL output must be bytes")
    if len(data) > max_bytes:
        raise ValueError(f"candidate JSONL output exceeds {max_bytes} bytes")
    text = data.decode("utf-8", errors="strict")
    lines = text.splitlines()
    if len(lines) > max_lines:
        raise ValueError(f"candidate JSONL output exceeds {max_lines} lines")
    if any(not line.strip() for line in lines):
        raise ValueError("candidate JSONL output contains an empty line")
    if expected_lines is not None:
        if len(lines) < expected_lines:
            raise ValueError(
                f"candidate JSONL output has {len(lines)} lines, "
                f"expected {expected_lines}"
            )
        if len(lines) > expected_lines and not allow_extra_lines:
            raise ValueError(
                f"candidate JSONL output has {len(lines) - expected_lines} extra lines"
            )

    documents: list[Any] = []
    fingerprints: set[str] = set()
    total_nodes = 0
    for index, line in enumerate(lines, start=1):
        document = _loads_strict_json(line)
        total_nodes += _bounded_json_shape(
            document,
            max_depth=max_depth,
            max_nodes=max_nodes - total_nodes,
        )
        if require_object and not isinstance(document, dict):
            raise ValueError(f"candidate JSONL line {index} must be an object")
        if reject_duplicate_lines:
            fingerprint = json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            if fingerprint in fingerprints:
                raise ValueError(f"candidate JSONL line {index} is a duplicate")
            fingerprints.add(fingerprint)
        documents.append(document)
    return tuple(documents)


def parse_candidate_jsonl(
    context: CandidateSuiteContext,
    data: bytes,
    *,
    max_bytes: int = 16 * 1024 * 1024,
    max_lines: int = 10_000,
    max_depth: int = 64,
    max_nodes: int = 100_000,
    expected_lines: int | None = None,
    allow_extra_lines: bool = False,
    reject_duplicate_lines: bool = False,
    require_object: bool = True,
) -> tuple[Any, ...]:
    """Strictly parse bounded UTF-8 JSONL under ``candidate_operation``."""
    _validate_json_limits(
        max_bytes=max_bytes,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )
    if (
        not isinstance(max_lines, int)
        or isinstance(max_lines, bool)
        or not 1 <= max_lines <= _MAX_JSON_LINES
    ):
        raise GraderFault(f"candidate JSONL max_lines must be in 1..{_MAX_JSON_LINES}")
    if expected_lines is not None and (
        not isinstance(expected_lines, int)
        or isinstance(expected_lines, bool)
        or not 0 <= expected_lines <= max_lines
    ):
        raise GraderFault("candidate JSONL expected_lines must be in 0..max_lines")
    if not all(
        isinstance(value, bool)
        for value in (allow_extra_lines, reject_duplicate_lines, require_object)
    ):
        raise GraderFault("candidate JSONL option flags must be bools")
    if allow_extra_lines and expected_lines is None:
        raise GraderFault("candidate JSONL allow_extra_lines requires expected_lines")
    return context.candidate_operation(
        "candidate JSONL output",
        _parse_candidate_jsonl,
        data,
        max_bytes=max_bytes,
        max_lines=max_lines,
        max_depth=max_depth,
        max_nodes=max_nodes,
        expected_lines=expected_lines,
        allow_extra_lines=allow_extra_lines,
        reject_duplicate_lines=reject_duplicate_lines,
        require_object=require_object,
    )


__all__ = [
    "CandidateAttempt",
    "CandidateCommandSpec",
    "CandidateSuiteMetadata",
    "CandidateSuiteResult",
    "parse_candidate_json",
    "parse_candidate_jsonl",
    "run_candidate_suite",
]
