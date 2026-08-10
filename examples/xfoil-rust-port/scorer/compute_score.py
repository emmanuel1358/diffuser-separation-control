"""Declarative behavioral rubric for the XFOIL-to-Rust transformation task."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from grading import AgentFault
from grading.evaluation import (
    RubricCriterion,
    RubricEvaluation,
    RubricTask,
    TrustedJson,
    WorkspaceArtifact,
)
from grading.kfold import DEFAULT_WIPE_ROOTS, _agent_wipe_roots
from grading.runtime_hardening import kill_pre_grade_agent_processes

PROTOCOL = "transform/v1"
SUITE_WEIGHTS = {
    "geometry": 0.25,
    "inviscid": 0.25,
    "viscous": 0.25,
    "polar_state": 0.25,
}
_TASK_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()
_MONITOR_SCHEMA = "xfoil-ptrace-monitor/v1"
_MONITOR_SHA256 = "9de50f4741cb4120c96a5ec049b7ac50d7f10ae9a721ea17e994aa6e4e15b816"
_MAX_CANDIDATE_BINARY_BYTES = 64 * 1024 * 1024
_RUST_INCLUDE_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_])include(?![A-Za-z0-9_])",
)
_AGENT_SMUGGLE_PROTECTED = (Path("/tmp/output"),)


class ComparisonRule:
    __slots__ = ("path", "kind", "atol", "rtol", "falloff", "weight")

    def __init__(
        self,
        path: str,
        kind: str = "exact",
        atol: float = 0.0,
        rtol: float = 0.0,
        falloff: float = 5.0,
        weight: float = 1.0,
    ) -> None:
        self.path = path
        self.kind = kind
        self.atol = atol
        self.rtol = rtol
        self.falloff = falloff
        self.weight = weight


def _rules_for(
    request: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[ComparisonRule, ...]:
    operation = str(request["operation"])
    if expected.get("status") != "ok":
        return (ComparisonRule(path="status"),)
    if operation == "naca_geometry":
        return (
            ComparisonRule(path="status"),
            ComparisonRule(
                path="observations.coordinates",
                kind="numeric_array",
                atol=5e-4,
                falloff=8.0,
                weight=5.0,
            ),
        )
    if operation == "analyze_inviscid":
        return (
            ComparisonRule(path="status", weight=2.0),
            ComparisonRule(
                path="observations.cl",
                kind="numeric",
                atol=0.02,
                weight=4.0,
            ),
            ComparisonRule(
                path="observations.cm",
                kind="numeric",
                atol=0.01,
                weight=2.0,
            ),
        )
    if operation == "analyze_viscous":
        return (
            ComparisonRule(path="status", weight=2.0),
            ComparisonRule(
                path="observations.cl",
                kind="numeric",
                atol=0.035,
                weight=3.0,
            ),
            ComparisonRule(
                path="observations.cd",
                kind="numeric",
                atol=0.0015,
                weight=4.0,
            ),
            ComparisonRule(
                path="observations.cm",
                kind="numeric",
                atol=0.02,
            ),
            ComparisonRule(
                path="observations.xtr_upper",
                kind="numeric",
                atol=0.08,
                falloff=4.0,
            ),
            ComparisonRule(
                path="observations.xtr_lower",
                kind="numeric",
                atol=0.08,
                falloff=4.0,
            ),
        )
    if operation == "polar":
        return (
            ComparisonRule(path="status", weight=2.0),
            ComparisonRule(
                path="observations.alpha",
                kind="numeric_array",
                atol=1e-6,
                weight=2.0,
            ),
            ComparisonRule(
                path="observations.cl",
                kind="numeric_array",
                atol=0.04,
                weight=3.0,
            ),
            ComparisonRule(
                path="observations.cd",
                kind="numeric_array",
                atol=0.002,
                weight=4.0,
            ),
            ComparisonRule(
                path="observations.cm",
                kind="numeric_array",
                atol=0.025,
            ),
        )
    raise ValueError(f"unsupported hidden transformation operation {operation!r}")


def _validated_requests(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("hidden_requests.json must contain a non-empty list")
    requests: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    suite_counts = {suite: 0 for suite in SUITE_WEIGHTS}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("hidden transformation requests must be objects")
        case_id = str(item.get("case_id", ""))
        suite = str(item.get("suite", ""))
        if not case_id or case_id in case_ids:
            raise ValueError(
                "hidden transformation case IDs must be non-empty and unique"
            )
        if item.get("protocol") != PROTOCOL:
            raise ValueError(f"hidden case {case_id!r} has the wrong protocol")
        if suite not in suite_counts:
            raise ValueError(f"hidden case {case_id!r} has unknown suite {suite!r}")
        if not item.get("operation"):
            raise ValueError(f"hidden case {case_id!r} has no operation")
        case_ids.add(case_id)
        suite_counts[suite] += 1
        requests.append(item)
    missing_suites = [suite for suite, count in suite_counts.items() if count == 0]
    if missing_suites:
        raise ValueError(f"hidden requests omit suites {missing_suites}")
    return requests


def _parse_response_stream(
    raw: bytes | str,
    expected_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    text = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != len(expected_ids):
        raise ValueError(
            f"expected {len(expected_ids)} JSONL responses, received {len(lines)}"
        )
    responses: dict[str, dict[str, Any]] = {}
    required = {
        "protocol",
        "case_id",
        "status",
        "observations",
        "events",
        "output_files",
    }
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("each response must be a JSON object")
        missing = required - set(value)
        if missing:
            raise ValueError(f"response is missing fields {sorted(missing)}")
        case_id = str(value["case_id"])
        if value["protocol"] != PROTOCOL:
            raise ValueError(f"response {case_id!r} has the wrong protocol")
        if case_id in responses:
            raise ValueError(f"response case ID {case_id!r} is duplicated")
        if not isinstance(value["status"], str):
            raise ValueError(f"response {case_id!r} status must be text")
        if not isinstance(value["observations"], dict):
            raise ValueError(f"response {case_id!r} observations must be an object")
        if not isinstance(value["events"], list):
            raise ValueError(f"response {case_id!r} events must be a list")
        if not isinstance(value["output_files"], dict):
            raise ValueError(f"response {case_id!r} output_files must be an object")
        responses[case_id] = value
    if set(responses) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(responses))
        extra = sorted(set(responses) - set(expected_ids))
        raise ValueError(f"response case IDs differ: missing={missing}, extra={extra}")
    return responses


def _value_at_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _array_shape(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return None
    if not value:
        return (0,)
    child_shapes = [_array_shape(item) for item in value]
    scalar_children = all(shape is None for shape in child_shapes)
    if scalar_children:
        return (len(value),)
    if any(shape is None for shape in child_shapes):
        return None
    first = child_shapes[0]
    if any(shape != first for shape in child_shapes):
        return None
    assert first is not None
    return (len(value), *first)


def _flatten_array(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return [value]
    flattened: list[Any] = []
    for item in value:
        flattened.extend(_flatten_array(item))
    return flattened


def _numeric_score(
    context,
    actual: Any,
    expected: Any,
    rule: ComparisonRule,
) -> float:
    # Null / non-numeric candidate values (e.g. serde_json NaN → null) must cost
    # only this rule, matching the absent-field path — not abort all suites.
    if actual is None or isinstance(actual, bool):
        return 0.0
    try:
        actual_number = context.number(
            actual,
            label=f"candidate {rule.path}",
            source="candidate",
        )
    except AgentFault:
        return 0.0
    expected_number = context.number(
        expected,
        label=f"trusted {rule.path}",
        source="trusted",
    )
    error = abs(actual_number - expected_number)
    tolerance = max(rule.atol, abs(expected_number) * rule.rtol)
    if error <= tolerance:
        return 1.0
    if tolerance <= 0.0:
        return 0.0
    return max(0.0, 1.0 - ((error - tolerance) / (rule.falloff * tolerance)))


def _rule_score(
    context,
    actual_response: dict[str, Any],
    expected_response: dict[str, Any],
    rule: ComparisonRule,
) -> float:
    actual = _value_at_path(actual_response, rule.path)
    expected = _value_at_path(expected_response, rule.path)
    if actual is _MISSING or expected is _MISSING:
        return 0.0
    if rule.kind == "exact":
        return 1.0 if actual == expected else 0.0
    if rule.kind == "numeric":
        return _numeric_score(context, actual, expected, rule)
    if rule.kind == "numeric_array":
        actual_shape = _array_shape(actual)
        expected_shape = _array_shape(expected)
        if actual_shape is None or actual_shape != expected_shape:
            return 0.0
        pairs = zip(_flatten_array(actual), _flatten_array(expected), strict=True)
        scores = [
            _numeric_score(context, actual_value, expected_value, rule)
            for actual_value, expected_value in pairs
        ]
        return context.mean(
            scores,
            label=f"{rule.path} element agreement",
            empty="agent_fault",
        )
    context.grader_failure(f"unknown comparison kind {rule.kind!r}")
    raise AssertionError("unreachable")


def _suite_scores(
    context,
    requests: list[dict[str, Any]],
    expected_by_id: dict[str, dict[str, Any]],
    actual_by_id: dict[str, dict[str, Any]],
) -> dict[str, float]:
    case_scores: dict[str, list[float]] = {suite: [] for suite in SUITE_WEIGHTS}
    for request in requests:
        case_id = str(request["case_id"])
        rules = _rules_for(request, expected_by_id[case_id])
        weighted = 0.0
        total_weight = 0.0
        for rule in rules:
            weighted += rule.weight * _rule_score(
                context,
                actual_by_id[case_id],
                expected_by_id[case_id],
                rule,
            )
            total_weight += rule.weight
        case_scores[str(request["suite"])].append(
            context.ratio(
                weighted,
                total_weight,
                label=f"{request['suite']} case agreement",
                zero="grader_fault",
            )
        )
    return {
        suite: context.mean(
            scores,
            label=f"{suite} suite agreement",
            empty="grader_fault",
        )
        for suite, scores in case_scores.items()
    }


def _find_public_runner() -> Path:
    for candidate in (
        Path("/data/transform_runner.py"),
        _TASK_ROOT / "data" / "transform_runner.py",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("transform_runner.py is unavailable")


def _hide_public_reference() -> list[tuple[Path, int]]:
    paths: list[Path] = []
    xfoil = shutil.which("xfoil")
    if xfoil:
        paths.append(Path(xfoil).resolve())
    for candidate in (
        Path("/data/xfoil-source"),
        Path("/data/xfoil-source.tar.gz"),
        Path("/data/xfoil-source.tar.gz.sha256"),
        Path("/data/transform_runner.py"),
    ):
        if candidate.exists():
            paths.append(candidate)

    changed: list[tuple[Path, int]] = []
    try:
        for path in paths:
            mode = os.lstat(path).st_mode & 0o777
            changed.append((path, mode))
            os.chmod(path, 0o700)
    except OSError:
        _restore_public_reference(changed)
        raise
    return changed


def _restore_public_reference(changed: list[tuple[Path, int]]) -> None:
    for path, mode in reversed(changed):
        os.chmod(path, mode)


def _scrub_agent_smuggle_roots(
    *,
    roots: tuple[Path, ...] | None = None,
    protected: tuple[Path, ...] = _AGENT_SMUGGLE_PROTECTED,
    cargo_config: Path = Path("/tmp/.cargo"),
    agent_uid: int | None = None,
) -> None:
    """Remove durable agent-planted trees that survive into grade time."""
    if os.geteuid() != 0:
        return

    if agent_uid is None:
        try:
            agent_uid = int(os.environ.get("RUBRIC_AGENT_UID", "1000"))
        except ValueError as exc:
            raise AgentFault("RUBRIC_AGENT_UID must be an integer") from exc

    try:
        if cargo_config.is_symlink() or cargo_config.is_file():
            cargo_config.unlink(missing_ok=True)
        elif cargo_config.is_dir():
            shutil.rmtree(cargo_config)
    except OSError as exc:
        raise AgentFault(f"could not remove agent Cargo config: {exc}") from exc

    if roots is None:
        roots = tuple(Path(path) for path in _agent_wipe_roots(DEFAULT_WIPE_ROOTS))

    protected_paths = {Path(os.path.realpath(path)) for path in protected}
    stack = list(dict.fromkeys(Path(os.path.realpath(root)) for root in roots))

    def is_protected(path: Path) -> bool:
        return any(
            path == protected_path or protected_path in path.parents
            for protected_path in protected_paths
        )

    while stack:
        root = stack.pop()
        if is_protected(root):
            continue
        try:
            if not root.is_dir():
                continue
            entries = list(os.scandir(root))
        except OSError as exc:
            raise AgentFault(
                f"could not inspect agent scratch root {root}: {exc}"
            ) from exc

        for entry in entries:
            path = Path(entry.path)
            if is_protected(path):
                continue
            try:
                info = entry.stat(follow_symlinks=False)
                is_directory = entry.is_dir(follow_symlinks=False)
                if info.st_uid == agent_uid:
                    if is_directory:
                        shutil.rmtree(path)
                    else:
                        path.unlink(missing_ok=True)
                elif is_directory:
                    stack.append(path)
            except OSError as exc:
                raise AgentFault(
                    f"could not remove agent scratch entry {path}: {exc}"
                ) from exc


def _quiesce_between_stages(context) -> None:
    """Kill leftover agent-uid processes between grading stages (root only)."""

    def _run() -> int:
        if os.geteuid() != 0:
            return 0
        if not Path("/proc").is_dir():
            return 0
        return kill_pre_grade_agent_processes()

    context.trusted_operation("inter-stage agent process quiesce", _run)


def _dependency_tables(
    data: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Collect Cargo dependency tables that may declare ``path`` specs."""
    tables: list[tuple[str, dict[str, Any]]] = []
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        deps = data.get(section)
        if isinstance(deps, dict):
            tables.append((section, deps))

    workspace = data.get("workspace")
    if isinstance(workspace, dict):
        deps = workspace.get("dependencies")
        if isinstance(deps, dict):
            tables.append(("workspace.dependencies", deps))

    target = data.get("target")
    if isinstance(target, dict):
        for cfg_name, body in target.items():
            if not isinstance(body, dict):
                continue
            for section in ("dependencies", "dev-dependencies", "build-dependencies"):
                deps = body.get(section)
                if isinstance(deps, dict):
                    tables.append((f"target.{cfg_name}.{section}", deps))

    patch = data.get("patch")
    if isinstance(patch, dict):
        for source, crates in patch.items():
            if isinstance(crates, dict):
                tables.append((f"patch.{source}", crates))
    return tables


def _reject_path_dependency(
    *,
    relative: Path,
    table_label: str,
    name: str,
    path_value: str,
    manifest_parent: Path,
    root_resolved: Path,
) -> None:
    # Absolute paths are always out-of-tree. Relative ``..`` segments are fine
    # when they stay inside the candidate workspace (e.g. ``../rustfoil-core``).
    if path_value.startswith("/"):
        raise AgentFault(
            f"Cargo.toml {relative} path dependency {name!r} in {table_label} "
            f"escapes the candidate tree ({path_value!r})"
        )
    resolved = (manifest_parent / path_value).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise AgentFault(
            f"Cargo.toml {relative} path dependency {name!r} in {table_label} "
            f"resolves outside the candidate tree"
        ) from exc


def _mask_rust_comments_and_strings(source: str) -> str:
    """Mask Rust comments and strings while preserving code positions."""
    masked = list(source)
    length = len(source)

    def blank(start: int, end: int) -> None:
        for index in range(start, min(end, length)):
            if masked[index] not in "\r\n":
                masked[index] = " "

    def raw_string_end(start: int) -> int | None:
        raw_index = start
        if (
            source[start] in {"b", "c"}
            and start + 1 < length
            and source[start + 1] == "r"
        ):
            raw_index += 1
        elif source[start] != "r":
            return None

        cursor = raw_index + 1
        while cursor < length and source[cursor] == "#":
            cursor += 1
        if cursor >= length or source[cursor] != '"':
            return None

        hashes = source[raw_index + 1 : cursor]
        terminator = '"' + hashes
        closing = source.find(terminator, cursor + 1)
        return length if closing < 0 else closing + len(terminator)

    def char_literal_end(start: int) -> int | None:
        quote_index = (
            start + 1
            if source[start] == "b" and start + 1 < length and source[start + 1] == "'"
            else start
        )
        if source[quote_index] != "'" or quote_index + 1 >= length:
            return None

        content_index = quote_index + 1
        if source[content_index] == "\\":
            cursor = content_index + 2
            while cursor < length and source[cursor] not in {"'", "\r", "\n"}:
                cursor += 1
            return cursor + 1 if cursor < length and source[cursor] == "'" else None

        closing = content_index + 1
        return closing + 1 if closing < length and source[closing] == "'" else None

    index = 0
    while index < length:
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            end = length if end < 0 else end
            blank(index, end)
            index = end
            continue

        if source.startswith("/*", index):
            depth = 1
            cursor = index + 2
            while cursor < length and depth:
                if source.startswith("/*", cursor):
                    depth += 1
                    cursor += 2
                elif source.startswith("*/", cursor):
                    depth -= 1
                    cursor += 2
                else:
                    cursor += 1
            blank(index, cursor)
            index = cursor
            continue

        char_end = char_literal_end(index)
        if char_end is not None:
            blank(index, char_end)
            index = char_end
            continue

        raw_end = raw_string_end(index)
        if raw_end is not None:
            blank(index, raw_end)
            index = raw_end
            continue

        quote_index = (
            index + 1
            if source[index] in {"b", "c"}
            and index + 1 < length
            and source[index + 1] == '"'
            else index
        )
        if source[quote_index] == '"':
            cursor = quote_index + 1
            while cursor < length:
                if source[cursor] == "\\":
                    cursor += 2
                elif source[cursor] == '"':
                    cursor += 1
                    break
                else:
                    cursor += 1
            blank(index, cursor)
            index = cursor
            continue

        index += 1

    return "".join(masked)


def _rust_source_uses_include(source: str) -> bool:
    return (
        _RUST_INCLUDE_IDENTIFIER.search(_mask_rust_comments_and_strings(source))
        is not None
    )


def _reject_rust_include_macros(candidate_root: Path) -> None:
    for source_path in sorted(
        path
        for path in candidate_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".rs", ".rsi", ".inc"}
    ):
        try:
            source = source_path.read_text(encoding="utf-8")
            relative = source_path.relative_to(candidate_root)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise AgentFault(
                f"candidate Rust source could not be inspected: {exc}"
            ) from exc
        if _rust_source_uses_include(source):
            raise AgentFault(
                f"candidate Rust source {relative} uses a forbidden include macro; "
                "use ordinary in-repository modules instead"
            )


def _reject_disallowed_cargo_manifest(candidate_root: Path) -> None:
    """Reject delegated Cargo behavior and compile-time source inclusion."""
    for cargo_dir in candidate_root.rglob(".cargo"):
        if cargo_dir.is_dir():
            try:
                relative = cargo_dir.relative_to(candidate_root)
            except ValueError:
                relative = cargo_dir
            raise AgentFault(
                f"candidate must not ship a .cargo directory ({relative}); "
                "rustc-wrapper and other cargo config overrides are forbidden"
            )

    manifests = sorted(candidate_root.rglob("Cargo.toml"))
    if not manifests:
        raise AgentFault("candidate workspace has no Cargo.toml")

    root_resolved = candidate_root.resolve()
    for manifest in manifests:
        try:
            relative = manifest.relative_to(candidate_root)
        except ValueError as exc:
            raise AgentFault(f"Cargo.toml escaped candidate root: {manifest}") from exc
        if any(part == "target" for part in relative.parts):
            continue
        try:
            data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise AgentFault(f"unreadable Cargo.toml at {relative}: {exc}") from exc

        package = data.get("package")
        if isinstance(package, dict) and "build" in package:
            raise AgentFault(
                f"Cargo.toml {relative} declares a custom build script "
                f"(build = {package['build']!r}); build scripts are forbidden"
            )

        lib = data.get("lib")
        if isinstance(lib, dict):
            crate_types = lib.get("crate-type")
            types = (
                [crate_types]
                if isinstance(crate_types, str)
                else list(crate_types or [])
            )
            if "proc-macro" in types or lib.get("proc-macro") is True:
                raise AgentFault(
                    f"Cargo.toml {relative} declares a proc-macro crate; forbidden"
                )

        for table_label, deps in _dependency_tables(data):
            if table_label == "build-dependencies" or table_label.endswith(
                ".build-dependencies"
            ):
                raise AgentFault(
                    f"Cargo.toml {relative} declares {table_label}; forbidden"
                )
            for name, spec in deps.items():
                if not isinstance(spec, dict) or "path" not in spec:
                    continue
                _reject_path_dependency(
                    relative=relative,
                    table_label=table_label,
                    name=str(name),
                    path_value=str(spec["path"]),
                    manifest_parent=manifest.parent,
                    root_resolved=root_resolved,
                )
    _reject_rust_include_macros(candidate_root)


def _find_candidate_monitor() -> Path:
    for candidate in (
        Path("/mcp_server/grader/candidate_monitor.py"),
        _TASK_ROOT / "scorer" / "candidate_monitor.py",
    ):
        if candidate.is_file():
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != _MONITOR_SHA256:
                raise ValueError(
                    f"candidate monitor digest mismatch: expected {_MONITOR_SHA256}, "
                    f"received {digest}"
                )
            return candidate
    raise FileNotFoundError("candidate_monitor.py is unavailable")


def _parse_monitor_envelope(raw: bytes | str) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected one monitor envelope, received {len(lines)} lines")
    value = json.loads(lines[0])
    if not isinstance(value, dict) or value.get("schema") != _MONITOR_SCHEMA:
        raise ValueError("candidate monitor returned the wrong schema")
    status = value.get("status")
    if status not in {
        "ok",
        "policy_violation",
        "candidate_failure",
        "candidate_timeout",
        "monitor_failure",
    }:
        raise ValueError(f"candidate monitor returned invalid status {status!r}")
    return value


def _candidate_build_directory() -> Path:
    if os.geteuid() != 0:
        raise RuntimeError("candidate build directory setup requires root")
    directory = Path(tempfile.mkdtemp(prefix="xfoil-candidate-build-"))
    home = directory / "home"
    home.mkdir(mode=0o700)
    for path in (directory, home):
        os.chown(path, 1000, 1000)
        os.chmod(path, 0o700)
    return directory


def _seal_candidate_binary(build_directory: Path) -> Path:
    source = build_directory / "release" / "transform-candidate"
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_fd = os.open(source, flags)
    try:
        metadata = os.fstat(source_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("candidate build output is not a regular file")
        if not metadata.st_mode & 0o111:
            raise RuntimeError("candidate build output is not executable")
        if metadata.st_size > _MAX_CANDIDATE_BINARY_BYTES:
            raise RuntimeError("candidate build output exceeds the sealed size limit")

        sealed_directory = Path(tempfile.mkdtemp(prefix="xfoil-sealed-binary-"))
        destination = sealed_directory / "transform-candidate"
        with os.fdopen(os.dup(source_fd), "rb") as source_handle:
            with destination.open("xb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle)
        destination.chmod(0o555)
        sealed_directory.chmod(0o555)
        return destination
    finally:
        os.close(source_fd)


def _run_monitored_candidate(
    context,
    binary: Path,
) -> tuple[bytes | None, dict[str, Any]]:
    monitor = context.trusted_operation(
        "candidate monitor discovery",
        _find_candidate_monitor,
    )
    result = context.run_solver(
        [
            "python3",
            "-I",
            "-B",
            str(monitor),
            "--binary",
            str(binary),
            "--cwd",
            str(context.candidate.path),
            "--requests",
            str(context.private / "hidden_requests.json"),
            "--timeout",
            "600",
        ],
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/root",
            "LC_ALL": "C",
        },
        timeout_s=620,
        max_output_bytes=48 * 1024 * 1024,
    )
    if not result.ok:
        context.grader_failure(
            "trusted candidate monitor failed "
            f"(returncode={result.returncode}, timeout={result.timed_out}, "
            f"output_limit={result.output_exceeded})"
        )
    envelope = context.trusted_operation(
        "candidate monitor envelope parsing",
        _parse_monitor_envelope,
        result.output,
    )
    status = envelope["status"]
    if status == "monitor_failure":
        context.grader_failure(
            f"trusted candidate monitor reported {envelope.get('error', 'failure')}"
        )
    metadata = {
        "status": status,
        "exec_events": envelope.get("exec_events"),
        "emulator_exec_events": envelope.get("emulator_exec_events"),
        "exec_paths": envelope.get("exec_paths"),
        "thread_clones": envelope.get("thread_clones"),
        "violation": envelope.get("violation"),
        "setup_error": envelope.get("setup_error"),
    }
    if status == "policy_violation":
        return None, metadata
    if status == "candidate_timeout":
        context.reject_candidate("candidate timed out under the process monitor")
    if status == "candidate_failure":
        setup_error = envelope.get("setup_error")
        details = [setup_error] if isinstance(setup_error, str) else []
        try:
            stderr = base64.b64decode(
                envelope.get("stderr_b64", ""),
                validate=True,
            ).decode("utf-8", errors="replace")
        except (TypeError, ValueError):
            stderr = ""
        if stderr.strip():
            details.append(stderr.strip()[-1000:])
        detail = f": {'; '.join(details)}" if details else ""
        context.reject_candidate(
            f"candidate exited with return code {envelope.get('returncode')}{detail}"
        )
    if envelope.get("exec_events") != 1:
        context.grader_failure("candidate monitor did not observe one initial exec")
    try:
        stdout = base64.b64decode(envelope["stdout_b64"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        context.grader_failure(f"candidate monitor stdout envelope is invalid: {exc}")
    return stdout, metadata


def evaluate(context) -> RubricEvaluation:
    requests = context.trusted_operation(
        "hidden transformation request validation",
        _validated_requests,
        context.fixture("requests"),
    )
    case_ids = tuple(str(request["case_id"]) for request in requests)
    runner = context.trusted_operation("public runner discovery", _find_public_runner)
    reference = context.run_solver(
        [
            "python3",
            str(runner),
            "legacy",
            str(context.private / "hidden_requests.json"),
        ],
        timeout_s=1200,
        max_output_bytes=32 * 1024 * 1024,
    )
    if not reference.ok:
        context.grader_failure(
            "trusted XFOIL reference execution failed "
            f"(returncode={reference.returncode}, timeout={reference.timed_out}, "
            f"output_limit={reference.output_exceeded})"
        )
    expected_by_id = context.trusted_operation(
        "trusted reference response parsing",
        _parse_response_stream,
        reference.output,
        case_ids,
    )

    if shutil.which("cargo") is None:
        context.grader_failure("the grading image does not provide cargo")

    changed = _hide_public_reference()
    build_directory: Path | None = None
    sealed_binary: Path | None = None
    try:
        context.trusted_operation(
            "cargo manifest policy check",
            _reject_disallowed_cargo_manifest,
            context.candidate.path,
        )
        context.trusted_operation(
            "agent smuggle-root scrub",
            _scrub_agent_smuggle_roots,
        )

        build_directory = context.trusted_operation(
            "candidate build directory setup",
            _candidate_build_directory,
        )
        safe_path = "/usr/local/bin:/usr/bin:/bin"
        build = context.run_candidate(
            [
                "cargo",
                "build",
                "--release",
                "--offline",
                "--bin",
                "transform-candidate",
                "--target-dir",
                str(build_directory),
                "--config",
                "build.rustc-wrapper=''",
            ],
            env={
                "PATH": safe_path,
                "HOME": str(build_directory / "home"),
                "LC_ALL": "C",
                "CARGO_HOME": "/opt/cargo-home",
                "CARGO_NET_OFFLINE": "true",
                "RUSTC_WRAPPER": "",
                "CARGO_BUILD_RUSTC_WRAPPER": "",
            },
            timeout_s=1200,
            max_output_bytes=2 * 1024 * 1024,
        )
        if build.returncode != 0:
            context.reject_candidate(
                f"cargo build failed with return code {build.returncode}"
            )

        _quiesce_between_stages(context)

        sealed_binary = context.trusted_operation(
            "candidate binary sealing",
            _seal_candidate_binary,
            build_directory,
        )
        candidate_stdout, monitor_metadata = _run_monitored_candidate(
            context,
            sealed_binary,
        )
        if candidate_stdout is None:
            scores = {suite: 0.0 for suite in SUITE_WEIGHTS}
        else:
            actual_by_id = context.candidate_operation(
                "candidate JSONL response parsing",
                _parse_response_stream,
                candidate_stdout,
                case_ids,
            )
            scores = _suite_scores(
                context,
                requests,
                expected_by_id,
                actual_by_id,
            )
    finally:
        if sealed_binary is not None:
            context.trusted_operation(
                "sealed candidate cleanup",
                shutil.rmtree,
                sealed_binary.parent,
                ignore_errors=True,
            )
        if build_directory is not None:
            context.trusted_operation(
                "candidate build cleanup",
                shutil.rmtree,
                build_directory,
                ignore_errors=True,
            )
        _restore_public_reference(changed)

    return RubricEvaluation(
        subscores=scores,
        metadata={
            "protocol": PROTOCOL,
            "suite_case_counts": {
                suite: sum(1 for request in requests if request["suite"] == suite)
                for suite in SUITE_WEIGHTS
            },
            "process_monitor": monitor_metadata,
        },
    )


TASK = RubricTask(
    artifact=WorkspaceArtifact(
        "repo",
        max_files=5_000,
        max_total_bytes=256 * 1024 * 1024,
        max_file_bytes=16 * 1024 * 1024,
        clean_paths=(".git", "target"),
        forbidden_names=("build.rs",),
        forbidden_suffixes=(
            ".f",
            ".f77",
            ".f90",
            ".for",
            ".o",
            ".a",
            ".so",
            ".dylib",
            ".dll",
            ".exe",
            ".class",
            ".jar",
            ".wasm",
            ".pyc",
        ),
        # Keep only tokens that imply executable cheating in source. Prose
        # tripwires (compiler names / data paths) previously zeroed genuine
        # ports via documentation comments. Runtime hide + ptrace already
        # block invoking XFOIL / Fortran / exec.
        forbidden_text_patterns=(
            "Command::new",
            "std::process::Command",
            "process::Command",
            'extern "C"',
            "libloading",
            "dlopen",
        ),
        text_suffixes=(
            ".rs",
            ".rsi",
            ".inc",
            ".toml",
            ".lock",
            ".py",
            ".sh",
            ".c",
            ".h",
            ".cc",
            ".cpp",
            ".cxx",
            ".hpp",
            ".go",
            ".java",
            ".js",
            ".ts",
            ".cs",
            ".swift",
            ".kt",
            ".kts",
            ".rb",
        ),
        reject_native_payloads=True,
    ),
    fixtures={"requests": TrustedJson("hidden_requests.json", require_object=False)},
    criteria=tuple(
        RubricCriterion(
            id=suite,
            weight=weight,
            description=f"{suite.replace('_', ' ')} behavioral compatibility",
        )
        for suite, weight in SUITE_WEIGHTS.items()
    ),
    evaluate=evaluate,
    security_tier="sealed_rescore",
)
