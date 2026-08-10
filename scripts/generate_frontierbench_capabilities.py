#!/usr/bin/env python3
"""Generate deterministic, capability-only evidence for Frontier-Bench.

The generator intentionally reads only production ``task.toml`` files,
production Compose overrides, and production ``tests/test.sh`` entrypoints. It
does not read instructions, hidden tests, fixtures, solutions, or credential
stores. Environment-variable values and shell commands are never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by CLI setup failures
    raise SystemExit(
        "PyYAML is required; run this script with `uv run python` from the "
        "template repository."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT.parent / "frontier-bench"
DEFAULT_MATRIX_PATH = REPO_ROOT / "docs" / "frontierbench_capabilities.json"
DEFAULT_DOCS_PATH = REPO_ROOT / "docs" / "FRONTIERBENCH_CAPABILITIES.md"

MATRIX_SCHEMA_VERSION = "frontierbench-capability-matrix.v1"
GENERATOR_VERSION = 1
EXPECTED_TASK_COUNT = 74
EXPECTED_SLUGS_SHA256 = (
    "d90cdc3a869482daa0d6d04655da2da1b9a33b8817944addb5c6dfb1e76b44d5"
)
EXCLUDED_NON_FRONTIER_NAMES = ("xfoil",)
COMPOSE_NAMES = (
    "docker-compose.yaml",
    "docker-compose.yml",
    "compose.yaml",
    "compose.yml",
)

SCHEMA_FIXTURE_TEST = (
    "harness/tests/test_frontierbench_capabilities.py::"
    "test_native_conformance_fixture_covers_schema_and_export"
)
RUNTIME_FIXTURE_TEST = (
    "taiga_runtime/rubric/tests/test_frontierbench_conformance.py::"
    "test_native_conformance_fixture_loads_runtime_primitives"
)
CONFORMANCE_FIXTURE = "harness/tests/fixtures/frontierbench_native_conformance.toml"


def _native_primitive(
    label: str,
    *,
    schema: str,
    harbor: str,
    taiga: str,
    runtime: str,
) -> dict[str, Any]:
    return {
        "kind": "native_primitive",
        "label": label,
        "schema": schema,
        "backend_mapping": {
            "harbor": harbor,
            "taiga": taiga,
            "runtime": runtime,
        },
        "conformance_required": True,
    }


def _mapping_policy(label: str, decision: str) -> dict[str, Any]:
    return {
        "kind": "mapping_policy",
        "label": label,
        "decision": decision,
        "conformance_required": False,
    }


MAPPING_CATALOG: dict[str, dict[str, Any]] = {
    "workspace.lifecycle": _native_primitive(
        "Seeded writable workspace and checkpoint lifecycle",
        schema="alignerr_plugin.schemas.WorkspaceSection",
        harbor="workspace seed overlay in the main/verifier image contexts",
        taiga="outer-capsule workspace identity and code_root",
        runtime="WorkspaceRuntimeSpec and TaskServiceRuntime._initialize_workspace",
    ),
    "artifact.file": _native_primitive(
        "Single regular-file artifact",
        schema="alignerr_plugin.schemas.FileArtifact",
        harbor="schema 1.4 artifacts projection",
        taiga="digest-locked capsule artifact summary",
        runtime="ServiceArtifact and TaskServiceRuntime._collect_artifact",
    ),
    "artifact.tree": _native_primitive(
        "Bounded recursive tree artifact",
        schema="alignerr_plugin.schemas.TreeArtifact",
        harbor="schema 1.4 artifacts projection with excludes",
        taiga="digest-locked capsule artifact summary",
        runtime="bounded no-symlink tree collection and sealing",
    ),
    "artifact.path_set": _native_primitive(
        "Grouped set of artifact paths",
        schema="alignerr_plugin.schemas.PathSetArtifact",
        harbor="one native artifact row per source",
        taiga="digest-locked capsule artifact summary",
        runtime="expanded bounded ServiceArtifact rows",
    ),
    "artifact.binary": _native_primitive(
        "Mode-aware binary artifact",
        schema="alignerr_plugin.schemas.BinaryArtifact",
        harbor="schema 1.4 artifacts projection",
        taiga="digest-locked capsule artifact summary",
        runtime="mode-preserving sealed artifact collection",
    ),
    "artifact.service": _native_primitive(
        "Artifact collected from a named service",
        schema="alignerr_plugin.schemas.ServiceArtifact",
        harbor="service-qualified native artifact projection",
        taiga="outer-capsule service artifact summary",
        runtime="quiesced service collection into a root-owned snapshot",
    ),
    "service.graph": _native_primitive(
        "Main, sidecar, init, and verifier service graph",
        schema="alignerr_plugin.schemas.ServiceSpec",
        harbor="native environment/docker-compose.yaml projection",
        taiga="digest-locked child images in an outer nested-Docker capsule",
        runtime="TaskServiceConfig and TaskServiceRuntime service ownership",
    ),
    "service.build": _native_primitive(
        "Trusted task-relative child image build",
        schema="alignerr_plugin.schemas.ServiceBuild",
        harbor="materialized service build contexts",
        taiga="trusted-build child image bundle",
        runtime="runtime rejects Compose build directives",
    ),
    "service.dependency": _native_primitive(
        "Service startup dependency and readiness condition",
        schema="alignerr_plugin.schemas.ServiceDependency",
        harbor="Compose depends_on projection",
        taiga="nested Compose depends_on projection",
        runtime="declared-service graph validation and readiness waits",
    ),
    "service.healthcheck": _native_primitive(
        "Bounded service healthcheck",
        schema="alignerr_plugin.schemas.ServiceHealthcheck",
        harbor="Compose healthcheck projection",
        taiga="nested Compose healthcheck projection",
        runtime="container health-state readiness gate",
    ),
    "service.environment": _native_primitive(
        "Named service environment",
        schema="alignerr_plugin.schemas.ServiceSpec.env",
        harbor="Compose environment projection",
        taiga="nested Compose environment projection",
        runtime="orchestration variables and socket values rejected",
    ),
    "service.port": _native_primitive(
        "Container service port",
        schema="alignerr_plugin.schemas.PortSpec",
        harbor="Compose expose/ports projection",
        taiga="nested Compose expose/ports projection",
        runtime="service-local network only; no host namespace",
    ),
    "service.network": _native_primitive(
        "Isolated or shared service network",
        schema="alignerr_plugin.schemas.ServiceSpec.network_mode",
        harbor="Compose network_mode and aliases projection",
        taiga="nested Compose network projection",
        runtime="host namespace joins are rejected",
    ),
    "service.named_volume": _native_primitive(
        "Backend-managed named volume",
        schema="alignerr_plugin.schemas.NamedVolume and VolumeMount",
        harbor="Compose named volume projection",
        taiga="nested Compose named volume projection",
        runtime="bind mounts and runtime sockets are rejected",
    ),
    "service.shared_memory": _native_primitive(
        "Bounded service shared-memory request",
        schema="alignerr_plugin.schemas.ServiceSpec.shm_mb",
        harbor="Compose shm_size projection",
        taiga="nested Compose shm_size projection",
        runtime="validated child Compose configuration",
    ),
    "service.capability": _native_primitive(
        "Allowlisted Linux service capability",
        schema="alignerr_plugin.schemas.ServiceSpec.capabilities",
        harbor="Compose cap_add projection",
        taiga="nested Compose cap_add projection",
        runtime="dangerous capabilities rejected before startup",
    ),
    "capture.pre_verification": _native_primitive(
        "Ordered failure-atomic pre-verification capture",
        schema="alignerr_plugin.schemas.CaptureSpec",
        harbor="verifier.collect projection",
        taiga="outer-capsule capture summary",
        runtime="CaptureHook ordering, timeout, fault policy, and atomic publish",
    ),
    "mcp.sse": _native_primitive(
        "Task-local SSE MCP endpoint",
        schema="alignerr_plugin.schemas.SSEMCPServer",
        harbor="environment.mcp_servers projection",
        taiga=(
            "audited service-DNS SSE proxy inside a trusted digest-locked outer "
            "capsule; unbundled export is rejected"
        ),
        runtime="service-DNS-only bounded SSE ToolRegistry proxy",
    ),
    "resource.agent": _native_primitive(
        "Agent phase resource and timeout envelope",
        schema="alignerr_plugin.schemas.ResourceSpec and AgentSection",
        harbor="environment resource projection",
        taiga="required_resources capacity validation and timeout projection",
        runtime="main-service command and startup time bounds",
    ),
    "resource.verifier": _native_primitive(
        "Independent verifier resource and timeout envelope",
        schema="alignerr_plugin.schemas.ResourceSpec and VerifierSection",
        harbor="separate verifier environment projection",
        taiga="outer-capsule verifier phase peak and grading timeout",
        runtime="separate verifier service timeout and result collection",
    ),
    "browser.sidecar": _native_primitive(
        "Browser automation sidecar",
        schema="ServiceSpec plus SSEMCPServer",
        harbor="browser/Playwright sidecar and named-volume graph",
        taiga=(
            "trusted outer capsule with audited SSE proxy; arbitrary unbundled "
            "SSE remains rejected"
        ),
        runtime="service readiness plus bounded task-local MCP proxy",
    ),
    "evaluation.engine": _native_primitive(
        "Verifier-owned evaluation entrypoint",
        schema="alignerr_plugin.schemas.EvaluationSection",
        harbor="separate tests image and tests/test.sh",
        taiga="image-baked rubric test_file bridge",
        runtime="root-owned grader or separate verifier service",
    ),
    "evaluation.gate": _native_primitive(
        "Structural, behavioral, performance, or determinism gate",
        schema="alignerr_plugin.schemas.GateSpec",
        harbor="task metadata and verifier-owned gate execution",
        taiga="redacted ordered capability summary",
        runtime="verifier-owned gate command with bounded result publication",
    ),
    "evaluation.report": _native_primitive(
        "Bounded structured verifier report",
        schema="alignerr_plugin.schemas.ReportSpec",
        harbor="declared verifier result path",
        taiga="redacted capability report summary",
        runtime="selected report paths sealed with verifier results",
    ),
    "result.canonical": _native_primitive(
        "Canonical reward, subscores, reports, and fault policy",
        schema="alignerr_plugin.schemas.ResultSection",
        harbor="verifier reward/report contract",
        taiga="output_directory and in-image grading result",
        runtime="canonical numeric verifier payload and fault separation",
    ),
    "security.sealed_verifier": _native_primitive(
        "Agent/verifier separation and sealed handoff",
        schema="strict paths, users, images, mounts, and capabilities",
        harbor="non-root agent plus separate root verifier",
        taiga="digest-locked outer capsule under Firecracker",
        runtime="root-owned snapshots, no socket inheritance, fail-closed results",
    ),
    "determinism.repeated_gate": _native_primitive(
        "Repeated deterministic evaluation gate",
        schema="GateSpec.repetitions and EvaluationSection.repetitions",
        harbor="verifier-owned repeated execution",
        taiga="ordered gate/repetition evidence in capability summary",
        runtime="sealed candidate snapshot reused across repetitions",
    ),
    "toolchain.image_owned": _native_primitive(
        "Heavy compiler, solver, browser, or accelerator toolchain",
        schema="ServiceBuild plus ResourceSpec",
        harbor="trusted image build; toolchain remains image-owned",
        taiga="trusted child image bundle or accelerator base image",
        runtime="no runtime package installation primitive",
    ),
    "compat.source_schema": _mapping_policy(
        "Legacy source schema identity",
        "Record source value; native authoring emits the current TaskToml schema.",
    ),
    "metadata.task": _mapping_policy(
        "Task identity metadata",
        "Map to TaskSection while excluding instructions and descriptions from evidence.",
    ),
    "metadata.provenance": _mapping_policy(
        "Non-runtime provenance metadata",
        "Preserve category/subcategory and numeric expert estimate; omit PII and referral values.",
    ),
    "environment.runtime_env": _mapping_policy(
        "Runtime environment variable declaration",
        "Record names only; materialize reviewed values through ServiceSpec.env or backend secret injection.",
    ),
    "evaluation.oracle_env": _mapping_policy(
        "Oracle-only environment declaration",
        "Keep in the trusted solution boundary; do not expose values to the agent.",
    ),
    "workspace.skills": _mapping_policy(
        "Image-owned agent skills path",
        "Bake skills into the main ServiceBuild and expose them beneath WorkspaceSection.agent_cwd.",
    ),
}

REQUIRED_NATIVE_PRIMITIVES = tuple(
    sorted(
        primitive_id
        for primitive_id, entry in MAPPING_CATALOG.items()
        if entry["kind"] == "native_primitive"
        and entry.get("conformance_required") is True
    )
)


TASK_FIELD_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"schema_version"), ("compat.source_schema",)),
    (
        re.compile(r"artifacts(?:\[\]\.(?:source|service|exclude))?"),
        (
            "artifact.file",
            "artifact.tree",
            "artifact.path_set",
            "artifact.binary",
            "artifact.service",
        ),
    ),
    (re.compile(r"task(?:\..+)?"), ("metadata.task",)),
    (re.compile(r"metadata(?:\..+)?"), ("metadata.provenance",)),
    (re.compile(r"agent"), ("resource.agent",)),
    (re.compile(r"agent\.timeout_sec"), ("resource.agent",)),
    (
        re.compile(r"verifier"),
        ("resource.verifier", "security.sealed_verifier"),
    ),
    (
        re.compile(r"verifier\.timeout_sec"),
        ("resource.verifier", "evaluation.engine"),
    ),
    (
        re.compile(r"verifier\.environment_mode"),
        ("resource.verifier", "security.sealed_verifier"),
    ),
    (
        re.compile(r"verifier\.collect(?:\[\]\.(?:command|service|timeout_sec))?"),
        ("capture.pre_verification", "artifact.service"),
    ),
    (
        re.compile(r"verifier\.env(?:\.\*)?"),
        ("environment.runtime_env", "security.sealed_verifier"),
    ),
    (
        re.compile(
            r"verifier\.environment(?:\.(?:allow_internet|build_timeout_sec|"
            r"cpus|memory_mb|storage_mb|gpus|gpu_types|mcp_servers))?"
        ),
        ("resource.verifier",),
    ),
    (re.compile(r"environment"), ("resource.agent",)),
    (
        re.compile(
            r"environment\.(?:build_timeout_sec|cpus|memory_mb|storage_mb|"
            r"gpus|gpu_types)"
        ),
        ("resource.agent",),
    ),
    (
        re.compile(r"environment\.mcp_servers"),
        ("mcp.sse",),
    ),
    (
        re.compile(r"environment\.mcp_servers\[\]\.(?:name|transport|url)"),
        ("mcp.sse",),
    ),
    (
        re.compile(
            r"environment\.healthcheck(?:\.(?:command|interval_sec|retries|"
            r"start_interval_sec|start_period_sec|timeout_sec))?"
        ),
        ("service.healthcheck",),
    ),
    (
        re.compile(r"environment\.env(?:\.\*)?"),
        ("environment.runtime_env",),
    ),
    (re.compile(r"environment\.skills_dir"), ("workspace.skills",)),
    (re.compile(r"solution(?:\.env(?:\.\*)?)?"), ("evaluation.oracle_env",)),
)

COMPOSE_FIELD_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"services"), ("service.graph",)),
    (re.compile(r"services\.\*"), ("service.graph",)),
    (
        re.compile(
            r"services\.\*\.build(?:\.(?:context|dockerfile|no_cache|target|args))?"
        ),
        ("service.build", "toolchain.image_owned"),
    ),
    (re.compile(r"services\.\*\.image"), ("service.graph", "service.build")),
    (re.compile(r"services\.\*\.pull_policy"), ("service.build",)),
    (
        re.compile(r"services\.\*\.depends_on(?:\.\*(?:\.condition)?)?"),
        ("service.dependency", "service.graph"),
    ),
    (
        re.compile(r"services\.\*\.healthcheck(?:\..+)?"),
        ("service.healthcheck",),
    ),
    (
        re.compile(r"services\.\*\.environment(?:\.\*)?"),
        ("service.environment",),
    ),
    (
        re.compile(r"services\.\*\.(?:expose|ports)"),
        ("service.port",),
    ),
    (
        re.compile(r"services\.\*\.volumes"),
        ("service.named_volume",),
    ),
    (re.compile(r"volumes(?:\.\*)?"), ("service.named_volume",)),
    (
        re.compile(r"services\.\*\.network_mode"),
        ("service.network",),
    ),
    (
        re.compile(r"services\.\*\.shm_size"),
        ("service.shared_memory",),
    ),
    (
        re.compile(r"services\.\*\.cap_add"),
        ("service.capability", "security.sealed_verifier"),
    ),
    (
        re.compile(r"services\.\*\.command"),
        ("service.graph",),
    ),
    (
        re.compile(r"services\.\*\.(?:cpus|mem_limit)"),
        ("resource.agent",),
    ),
)


ENTRYPOINT_PATTERN_SPECS: dict[str, dict[str, Any]] = {
    "test.pytest": {
        "group": "test",
        "pattern": re.compile(r"(?i)\bpytest\b"),
        "mapping": ("evaluation.engine", "evaluation.gate"),
    },
    "test.node": {
        "group": "test",
        "pattern": re.compile(r"(?i)\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b"),
        "mapping": ("evaluation.engine", "toolchain.image_owned"),
    },
    "report.ctrf": {
        "group": "report",
        "pattern": re.compile(r"(?i)(?:--ctrf\b|ctrf(?:_[^/\s]+)?\.json)"),
        "mapping": ("evaluation.report",),
    },
    "report.junit": {
        "group": "report",
        "pattern": re.compile(r"(?i)(?:--junit(?:xml)?\b|junit\.xml)"),
        "mapping": ("evaluation.report",),
    },
    "result.reward_text": {
        "group": "result",
        "pattern": re.compile(r"/logs/verifier/reward\.txt\b"),
        "mapping": ("result.canonical",),
    },
    "result.reward_json": {
        "group": "result",
        "pattern": re.compile(r"/logs/verifier/reward\.json\b"),
        "mapping": ("result.canonical",),
    },
    "security.canary_marker": {
        "group": "security",
        "pattern": re.compile(r"harbor-canary GUID"),
        "mapping": ("security.sealed_verifier",),
    },
    "security.root_only_reward": {
        "group": "security",
        "pattern": re.compile(
            r"(?is)(?:chmod\s+700\s+[^\n]*(?:verifier|LOG_DIR)|"
            r"(?:verifier|reward)[^\n]{0,100}root-only)"
        ),
        "mapping": ("security.sealed_verifier", "result.canonical"),
    },
    "security.privilege_drop": {
        "group": "security",
        "pattern": re.compile(
            r"\b(?:setpriv|runuser|su\s+(?:nobody|agent|postgres))\b"
        ),
        "mapping": ("security.sealed_verifier",),
    },
    "security.anti_cheat": {
        "group": "cheat",
        "pattern": re.compile(
            r"(?i)(?:anti[- ]?cheat|forbidden|reward.{0,60}forg|"
            r"forg.{0,60}reward|bypass|pre[- ]?plant)"
        ),
        "mapping": ("security.sealed_verifier", "evaluation.gate"),
    },
    "security.fault_separation": {
        "group": "security",
        "pattern": re.compile(
            r"(?i)(?:infra(?:structure)?[_ -](?:error|fault)|"
            r"agent[_ -](?:fail|fault)|verifier infra)"
        ),
        "mapping": ("result.canonical", "security.sealed_verifier"),
    },
    "security.stale_output_clear": {
        "group": "cheat",
        "pattern": re.compile(
            r"(?i)(?:rm\s+-f[^\n]*(?:reward|result|dump|patch)|"
            r"pre[- ]?clear|stale[^\n]{0,80}(?:output|artifact|result))"
        ),
        "mapping": ("capture.pre_verification", "security.sealed_verifier"),
    },
    "determinism.explicit": {
        "group": "determinism",
        "pattern": re.compile(
            r"(?i)(?:determin|reproduc|repeat(?:ed|ability)|"
            r"for\s+\w+\s+in\s+1\s+2\s+3)"
        ),
        "mapping": ("determinism.repeated_gate", "evaluation.gate"),
    },
    "browser.entrypoint": {
        "group": "browser",
        "pattern": re.compile(
            r"(?i)\b(?:playwright|chromium|chrome|browser|novnc|selenium)\b"
        ),
        "mapping": ("browser.sidecar", "toolchain.image_owned"),
    },
    "accelerator.entrypoint": {
        "group": "toolchain",
        "pattern": re.compile(r"(?i)\b(?:cuda|nvidia|gpu|h100|a100|jax|triton)\b"),
        "mapping": ("resource.agent", "toolchain.image_owned"),
    },
}

HEAVY_TOOLCHAIN_SPECS: dict[str, re.Pattern[str]] = {
    "browser_stack": re.compile(
        r"(?i)\b(?:playwright|chromium|chrome|browser|novnc|selenium|"
        r"frontend|nextjs|react)\b"
    ),
    "cad_eda": re.compile(
        r"(?i)\b(?:freecad|openroad|verilog|systemverilog|qemu|uefi)\b"
    ),
    "database_stack": re.compile(
        r"(?i)\b(?:postgres|mysql|redis|sqlite|kafka|odoo|memcached|"
        r"mvcc|lsm|database)\b"
    ),
    "formal_proof": re.compile(r"(?i)\b(?:lean4?|lake|coq|rocq)\b"),
    "gpu_ml": re.compile(r"(?i)\b(?:cuda|nvidia|gpu|h100|a100|jax|triton)\b"),
    "native_compiler": re.compile(r"(?i)\b(?:gcc|g\+\+|clang|cmake|make|ninja)\b"),
    "node_web": re.compile(r"(?i)\b(?:node|npm|pnpm|yarn|bun|nextjs|react)\b"),
    "rust": re.compile(r"(?i)\b(?:rust|cargo)\b"),
    "scientific_solver": re.compile(
        r"(?i)\b(?:meep|fdtd|glm|ode|pde|solver|topology)\b"
    ),
}

_BINARY_SUFFIXES = frozenset(
    {
        ".a",
        ".bin",
        ".class",
        ".dll",
        ".dump",
        ".exe",
        ".o",
        ".obj",
        ".pyc",
        ".rdb",
        ".so",
        ".wasm",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    )


def _git_revision(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    revision = completed.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"unexpected source git revision: {revision!r}")
    return revision


def _source_relative(path: Path, source_root: Path) -> str:
    return path.relative_to(source_root).as_posix()


def _input_digest(paths: Sequence[Path], source_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = _source_relative(path, source_root).encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _walk_field_paths(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path
            yield from _walk_field_paths(item, path)
    elif isinstance(value, list):
        list_prefix = f"{prefix}[]"
        for item in value:
            yield from _walk_field_paths(item, list_prefix)


def _normalize_task_field(path: str) -> str:
    for prefix in ("verifier.env.", "environment.env.", "solution.env."):
        if path.startswith(prefix):
            return f"{prefix}*"
    return path


def _walk_compose_field_paths(
    value: Any,
    parts: tuple[str, ...] = (),
) -> Iterable[str]:
    if not isinstance(value, Mapping):
        return
    for key, item in value.items():
        if parts in {
            ("services",),
            ("volumes",),
            ("networks",),
            ("services", "*", "depends_on"),
            ("services", "*", "environment"),
        }:
            normalized = "*"
        else:
            normalized = str(key)
        child = (*parts, normalized)
        yield ".".join(child)
        yield from _walk_compose_field_paths(item, child)


def _field_mapping(
    path: str,
    rules: Sequence[tuple[re.Pattern[str], tuple[str, ...]]],
) -> list[str]:
    for pattern, mapping in rules:
        if pattern.fullmatch(path):
            return list(mapping)
    return []


def _field_inventory(
    paths_by_task: Mapping[str, set[str]],
    rules: Sequence[tuple[re.Pattern[str], tuple[str, ...]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    all_paths = sorted(
        {
            field_path
            for task_paths in paths_by_task.values()
            for field_path in task_paths
        }
    )
    inventory: list[dict[str, Any]] = []
    unmapped: list[str] = []
    for field_path in all_paths:
        mapping = _field_mapping(field_path, rules)
        if not mapping:
            unmapped.append(field_path)
        inventory.append(
            {
                "path": field_path,
                "task_count": sum(
                    field_path in task_paths for task_paths in paths_by_task.values()
                ),
                "mapping": mapping,
            }
        )
    return inventory, unmapped


def _artifact_shape(source: str, *, exclude_count: int) -> str:
    path = PurePosixPath(source)
    if source.endswith("/") or exclude_count:
        return "tree"
    if path.suffix.lower() in _BINARY_SUFFIXES:
        return "binary"
    if path.suffix:
        return "file"
    return "ambiguous_path"


def _artifact_fact(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        source = raw
        service = "main"
        declaration = "string"
        exclude_count = 0
        explicit_service = False
    elif isinstance(raw, Mapping):
        source = str(raw.get("source") or "")
        service = str(raw.get("service") or "main")
        declaration = "table"
        excludes = raw.get("exclude") or []
        exclude_count = len(excludes) if isinstance(excludes, list) else 0
        explicit_service = "service" in raw
    else:
        raise TypeError(f"artifact declaration must be a string or table: {raw!r}")
    if not source:
        raise ValueError("artifact declaration is missing a source path")

    observed_shape = _artifact_shape(source, exclude_count=exclude_count)
    if service != "main":
        native_candidates = ["artifact.service"]
    elif observed_shape == "tree":
        native_candidates = ["artifact.tree"]
    elif observed_shape == "file":
        native_candidates = ["artifact.file"]
    elif observed_shape == "binary":
        native_candidates = ["artifact.binary"]
    else:
        native_candidates = [
            "artifact.binary",
            "artifact.file",
            "artifact.tree",
        ]
    return {
        "source": source,
        "service": service,
        "declaration": declaration,
        "observed_shape": observed_shape,
        "exclude_pattern_count": exclude_count,
        "explicit_service": explicit_service,
        "native_candidates": native_candidates,
    }


def _environment_names(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return sorted(str(key) for key in value)
    names: set[str] = set()
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, str):
                continue
            name = item.split("=", 1)[0].strip()
            if name:
                names.add(name)
    return sorted(names)


def _dependency_edges(services: Mapping[str, Any]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for service_name, raw_service in services.items():
        if not isinstance(raw_service, Mapping):
            continue
        dependencies = raw_service.get("depends_on")
        if isinstance(dependencies, Mapping):
            rows = dependencies.items()
        elif isinstance(dependencies, list):
            rows = ((dependency, None) for dependency in dependencies)
        else:
            rows = ()
        for dependency, details in rows:
            condition = "service_started"
            if isinstance(details, Mapping):
                condition = str(details.get("condition") or condition)
            edges.append(
                {
                    "service": str(service_name),
                    "depends_on": str(dependency),
                    "condition": condition,
                }
            )
    return sorted(
        edges,
        key=lambda edge: (
            edge["service"],
            edge["depends_on"],
            edge["condition"],
        ),
    )


def _service_ports(raw_service: Mapping[str, Any]) -> list[str]:
    values = raw_service.get("expose") or raw_service.get("ports") or []
    if not isinstance(values, list):
        return []
    ports: set[str] = set()
    for value in values:
        if isinstance(value, (str, int)):
            ports.add(str(value))
        elif isinstance(value, Mapping):
            target = value.get("target") or value.get("container_port")
            if target is not None:
                ports.add(str(target))
    return sorted(ports)


def _compose_facts(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "present": False,
            "service_count": 1,
            "services": ["main"],
            "edges": [],
            "build_services": [],
            "external_image_services": [],
            "digest_pinned_external_image_count": 0,
            "healthcheck_services": [],
            "named_volumes": [],
            "service_volume_mount_counts": {},
            "environment_variable_names": {},
            "exposed_ports": {},
            "network_modes": {},
            "linux_capabilities": {},
            "shared_memory_services": [],
            "native_primitives": [],
        }
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    raw_services = payload.get("services") or {}
    if not isinstance(raw_services, Mapping) or not raw_services:
        raise ValueError(f"{path} must contain a non-empty services mapping")

    services = {
        str(name): value
        for name, value in raw_services.items()
        if isinstance(value, Mapping)
    }
    if len(services) != len(raw_services):
        raise ValueError(f"{path} has a non-mapping service definition")
    external_image_services = sorted(
        name for name, value in services.items() if value.get("image")
    )
    digest_pinned = sum(
        bool(
            re.fullmatch(
                r"[^\s@]+@sha256:[0-9a-fA-F]{64}",
                str(services[name].get("image") or ""),
            )
        )
        for name in external_image_services
    )
    named_volumes = payload.get("volumes") or {}
    if not isinstance(named_volumes, Mapping):
        raise ValueError(f"{path} top-level volumes must be a mapping")

    primitives = {
        "service.graph",
        "service.dependency",
        "security.sealed_verifier",
    }
    if any(service.get("build") for service in services.values()):
        primitives.update({"service.build", "toolchain.image_owned"})
    if any(service.get("healthcheck") for service in services.values()):
        primitives.add("service.healthcheck")
    if any(service.get("environment") for service in services.values()):
        primitives.add("service.environment")
    if any(
        service.get("expose") or service.get("ports") for service in services.values()
    ):
        primitives.add("service.port")
    if named_volumes or any(service.get("volumes") for service in services.values()):
        primitives.add("service.named_volume")
    if any(service.get("network_mode") for service in services.values()):
        primitives.add("service.network")
    if any(service.get("shm_size") for service in services.values()):
        primitives.add("service.shared_memory")
    if any(service.get("cap_add") for service in services.values()):
        primitives.add("service.capability")

    return {
        "present": True,
        "service_count": len(services),
        "services": sorted(services),
        "edges": _dependency_edges(services),
        "build_services": sorted(
            name for name, service in services.items() if service.get("build")
        ),
        "external_image_services": external_image_services,
        "digest_pinned_external_image_count": digest_pinned,
        "healthcheck_services": sorted(
            name for name, service in services.items() if service.get("healthcheck")
        ),
        "named_volumes": sorted(str(name) for name in named_volumes),
        "service_volume_mount_counts": {
            name: len(service.get("volumes") or [])
            for name, service in sorted(services.items())
            if service.get("volumes")
        },
        "environment_variable_names": {
            name: _environment_names(service.get("environment"))
            for name, service in sorted(services.items())
            if service.get("environment")
        },
        "exposed_ports": {
            name: _service_ports(service)
            for name, service in sorted(services.items())
            if _service_ports(service)
        },
        "network_modes": {
            name: str(service["network_mode"])
            for name, service in sorted(services.items())
            if service.get("network_mode")
        },
        "linux_capabilities": {
            name: sorted(str(value) for value in service.get("cap_add") or [])
            for name, service in sorted(services.items())
            if service.get("cap_add")
        },
        "shared_memory_services": sorted(
            name for name, service in services.items() if service.get("shm_size")
        ),
        "native_primitives": sorted(primitives),
    }


def _capture_facts(verifier: Mapping[str, Any]) -> dict[str, Any]:
    raw_hooks = verifier.get("collect") or []
    if not isinstance(raw_hooks, list):
        raise TypeError("[verifier].collect must be a list")
    services: list[str] = []
    timeouts: list[float] = []
    atomic_publish_count = 0
    tolerant_exit_count = 0
    for raw_hook in raw_hooks:
        if not isinstance(raw_hook, Mapping):
            raise TypeError("verifier collect hook must be a table")
        services.append(str(raw_hook.get("service") or "main"))
        timeout = raw_hook.get("timeout_sec")
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
            timeouts.append(float(timeout))
        command = str(raw_hook.get("command") or "")
        if re.search(
            r"(?is)(?:\.tmp.{0,400}\bmv\b|os\.replace\s*\(|" r"temp.{0,400}atomic)",
            command,
        ):
            atomic_publish_count += 1
        if "|| true" in command:
            tolerant_exit_count += 1
    return {
        "count": len(raw_hooks),
        "services": sorted(set(services)),
        "timeouts_sec": sorted(timeouts),
        "atomic_publish_count": atomic_publish_count,
        "tolerant_exit_count": tolerant_exit_count,
        "native_primitives": (
            ["capture.pre_verification", "security.sealed_verifier"]
            if raw_hooks
            else []
        ),
    }


def _mcp_facts(environment: Mapping[str, Any]) -> dict[str, Any]:
    raw_servers = environment.get("mcp_servers") or []
    if not isinstance(raw_servers, list):
        raise TypeError("[environment].mcp_servers must be a list")
    servers: list[dict[str, Any]] = []
    for raw_server in raw_servers:
        if not isinstance(raw_server, Mapping):
            raise TypeError("MCP server declaration must be a table")
        raw_url = str(raw_server.get("url") or "")
        parsed = urlsplit(raw_url) if raw_url else None
        servers.append(
            {
                "name": str(raw_server.get("name") or ""),
                "transport": str(raw_server.get("transport") or ""),
                "service_host": parsed.hostname if parsed else None,
                "port": parsed.port if parsed else None,
            }
        )
    return {
        "count": len(servers),
        "servers": sorted(servers, key=lambda row: row["name"]),
        "native_primitives": ["mcp.sse"] if servers else [],
    }


def _resource_fields(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    allowed = (
        "allow_internet",
        "build_timeout_sec",
        "cpus",
        "memory_mb",
        "storage_mb",
        "gpus",
        "gpu_types",
    )
    return {name: value[name] for name in allowed if name in value}


def _resource_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
    environment = payload.get("environment") or {}
    verifier = payload.get("verifier") or {}
    agent = payload.get("agent") or {}
    if not all(
        isinstance(section, Mapping) for section in (environment, verifier, agent)
    ):
        raise TypeError("task environment, verifier, and agent must be tables")
    verifier_environment = verifier.get("environment") or {}
    if not isinstance(verifier_environment, Mapping):
        raise TypeError("[verifier.environment] must be a table")
    verifier_env = verifier.get("env") or {}
    agent_resources = _resource_fields(environment)
    verifier_resources = _resource_fields(verifier_environment)
    return {
        "agent": {
            "timeout_sec": agent.get("timeout_sec"),
            **agent_resources,
        },
        "verifier": {
            "timeout_sec": verifier.get("timeout_sec"),
            "environment_mode": verifier.get("environment_mode"),
            "explicit_resources": bool(verifier_resources),
            "environment_variable_names": _environment_names(verifier_env),
            **verifier_resources,
        },
        "native_primitives": [
            "resource.agent",
            "resource.verifier",
            "security.sealed_verifier",
        ],
    }


def _entrypoint_facts(text: str) -> dict[str, Any]:
    matched = [
        pattern_id
        for pattern_id, spec in ENTRYPOINT_PATTERN_SPECS.items()
        if spec["pattern"].search(text)
    ]
    if not any(pattern_id.startswith("test.") for pattern_id in matched):
        matched.append("test.custom_shell")
    if not any(pattern_id.startswith("result.reward_") for pattern_id in matched):
        matched.append("result.delegated_to_test_suite")

    frameworks: list[str] = []
    if "test.pytest" in matched:
        frameworks.append("pytest")
    if "test.node" in matched:
        frameworks.append("node-test")
    if "test.custom_shell" in matched:
        frameworks.append("custom-shell")

    reward_files: list[str] = []
    if "result.reward_text" in matched:
        reward_files.append("/logs/verifier/reward.txt")
    if "result.reward_json" in matched:
        reward_files.append("/logs/verifier/reward.json")

    report_formats: list[str] = []
    if "report.ctrf" in matched:
        report_formats.append("ctrf")
    if "report.junit" in matched:
        report_formats.append("junit")

    groups: dict[str, list[str]] = {
        "security": [],
        "determinism": [],
        "cheat_resistance": [],
        "browser": [],
    }
    for pattern_id in matched:
        spec = ENTRYPOINT_PATTERN_SPECS.get(pattern_id)
        if spec is None:
            continue
        group = spec["group"]
        if group == "security":
            groups["security"].append(pattern_id)
        elif group == "determinism":
            groups["determinism"].append(pattern_id)
        elif group == "cheat":
            groups["cheat_resistance"].append(pattern_id)
        elif group == "browser":
            groups["browser"].append(pattern_id)

    mappings = {
        mapping
        for pattern_id in matched
        for mapping in (
            ENTRYPOINT_PATTERN_SPECS.get(pattern_id, {}).get("mapping")
            or (
                ("evaluation.engine",)
                if pattern_id == "test.custom_shell"
                else ("result.canonical",)
            )
        )
    }
    return {
        "interpreter": (
            "bash"
            if text.startswith("#!/bin/bash") or text.startswith("#!/usr/bin/env bash")
            else "bash-inferred"
        ),
        "frameworks": frameworks,
        "direct_reward_files": reward_files,
        "reward_writer": ("entrypoint" if reward_files else "delegated_to_test_suite"),
        "report_formats": report_formats,
        "patterns": sorted(matched),
        **{name: sorted(values) for name, values in groups.items()},
        "native_primitives": sorted(mappings),
    }


def _toolchain_facts(
    *,
    slug: str,
    metadata: Mapping[str, Any],
    entrypoint_text: str,
    compose: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> list[str]:
    tags = metadata.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    compose_identifiers = " ".join(
        [
            *compose.get("services", []),
            *compose.get("build_services", []),
            *compose.get("external_image_services", []),
        ]
    )
    haystack = " ".join(
        [
            slug,
            str(metadata.get("category") or ""),
            str(metadata.get("subcategory") or ""),
            *(str(tag) for tag in tags),
            compose_identifiers,
            entrypoint_text,
        ]
    )
    flags = {
        name
        for name, pattern in HEAVY_TOOLCHAIN_SPECS.items()
        if pattern.search(haystack)
    }
    if (resources.get("agent") or {}).get("gpus", 0):
        flags.add("gpu_accelerator")
    if (resources.get("verifier") or {}).get("gpus", 0):
        flags.add("verifier_gpu")
    if compose.get("present"):
        flags.add("multi_container")
    return sorted(flags)


def _metadata_facts(metadata: Mapping[str, Any]) -> dict[str, Any]:
    estimate = metadata.get("expert_time_estimate_hours")
    if isinstance(estimate, bool) or not isinstance(estimate, (int, float)):
        estimate = None
    return {
        "category": str(metadata.get("category") or ""),
        "subcategory": str(metadata.get("subcategory") or ""),
        "expert_time_estimate_hours": estimate,
    }


def _task_primitives(
    *,
    artifacts: Sequence[Mapping[str, Any]],
    compose: Mapping[str, Any],
    captures: Mapping[str, Any],
    mcp: Mapping[str, Any],
    resources: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    browser: bool,
    heavy_toolchains: Sequence[str],
) -> list[str]:
    primitives = {
        "evaluation.engine",
        "result.canonical",
        "resource.agent",
        "resource.verifier",
        "security.sealed_verifier",
    }
    for artifact in artifacts:
        primitives.update(artifact["native_candidates"])
    if len(artifacts) > 1:
        primitives.add("artifact.path_set")
    primitives.update(compose.get("native_primitives", []))
    primitives.update(captures.get("native_primitives", []))
    primitives.update(mcp.get("native_primitives", []))
    primitives.update(resources.get("native_primitives", []))
    primitives.update(evaluation.get("native_primitives", []))
    if browser:
        primitives.add("browser.sidecar")
    if heavy_toolchains:
        primitives.add("toolchain.image_owned")
    return sorted(primitives)


def _task_record(
    *,
    source_root: Path,
    task_path: Path,
    test_path: Path,
    compose_path: Path | None,
) -> tuple[dict[str, Any], set[str], set[str]]:
    payload = tomllib.loads(task_path.read_text())
    slug = task_path.parent.name
    task_section = payload.get("task") or {}
    if not isinstance(task_section, Mapping):
        raise TypeError(f"{task_path} [task] must be a table")
    task_name = str(task_section.get("name") or "")
    if task_name.rsplit("/", 1)[-1] != slug:
        raise ValueError(
            f"{task_path} task name {task_name!r} does not end in slug {slug!r}"
        )
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise TypeError(f"{task_path} artifacts must be a list")
    artifacts = [_artifact_fact(raw) for raw in raw_artifacts]

    verifier = payload.get("verifier") or {}
    environment = payload.get("environment") or {}
    metadata = payload.get("metadata") or {}
    if not all(
        isinstance(section, Mapping) for section in (verifier, environment, metadata)
    ):
        raise TypeError(f"{task_path} has a non-table required section")

    compose = _compose_facts(compose_path)
    captures = _capture_facts(verifier)
    mcp = _mcp_facts(environment)
    resources = _resource_facts(payload)
    entrypoint_text = test_path.read_text(errors="strict")
    evaluation = _entrypoint_facts(entrypoint_text)
    heavy_toolchains = _toolchain_facts(
        slug=slug,
        metadata=metadata,
        entrypoint_text=entrypoint_text,
        compose=compose,
        resources=resources,
    )
    browser_evidence: list[str] = []
    if any(server["name"].lower() == "playwright" for server in mcp["servers"]):
        browser_evidence.append("mcp:playwright")
    if any(
        re.search(r"(?i)(?:playwright|browser|chrom|novnc)", service)
        for service in compose["services"]
    ):
        browser_evidence.append("compose:browser-service")
    if "browser_stack" in heavy_toolchains:
        browser_evidence.append("static-toolchain-indicator")
    if evaluation["browser"]:
        browser_evidence.append("test-entrypoint-indicator")
    browser_evidence = sorted(set(browser_evidence))

    service_scoped = sum(artifact["explicit_service"] for artifact in artifacts)
    sidecar_scoped = sum(artifact["service"] != "main" for artifact in artifacts)
    artifact_contract = {
        "count": len(artifacts),
        "declaration_shapes": dict(
            sorted(Counter(item["declaration"] for item in artifacts).items())
        ),
        "observed_shapes": dict(
            sorted(Counter(item["observed_shape"] for item in artifacts).items())
        ),
        "service_scoped_count": service_scoped,
        "sidecar_scoped_count": sidecar_scoped,
        "path_set_candidate": len(artifacts) > 1,
        "items": artifacts,
    }

    record = {
        "slug": slug,
        "source_schema_version": payload.get("schema_version"),
        "source_files": {
            "task_toml": {
                "path": _source_relative(task_path, source_root),
                "sha256": _sha256_file(task_path),
            },
            "test_entrypoint": {
                "path": _source_relative(test_path, source_root),
                "sha256": _sha256_file(test_path),
            },
            "compose": (
                {
                    "path": _source_relative(compose_path, source_root),
                    "sha256": _sha256_file(compose_path),
                }
                if compose_path
                else None
            ),
        },
        "metadata": _metadata_facts(metadata),
        "artifact_contract": artifact_contract,
        "services": compose,
        "collect_hooks": captures,
        "mcp": mcp,
        "resources": resources,
        "browser": {
            "required": bool(browser_evidence),
            "evidence": browser_evidence,
        },
        "heavy_toolchain_flags": heavy_toolchains,
        "evaluation": evaluation,
    }
    record["native_primitives"] = _task_primitives(
        artifacts=artifacts,
        compose=compose,
        captures=captures,
        mcp=mcp,
        resources=resources,
        evaluation=evaluation,
        browser=bool(browser_evidence),
        heavy_toolchains=heavy_toolchains,
    )

    task_fields = {_normalize_task_field(path) for path in _walk_field_paths(payload)}
    compose_fields = (
        set(_walk_compose_field_paths(yaml.safe_load(compose_path.read_text()) or {}))
        if compose_path
        else set()
    )
    return record, task_fields, compose_fields


def _numeric_range(tasks: Sequence[Mapping[str, Any]], *keys: str) -> dict[str, Any]:
    values: list[float] = []
    for task in tasks:
        current: Any = task
        for key in keys:
            current = current.get(key) if isinstance(current, Mapping) else None
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            values.append(float(current))
    if not values:
        return {"declared_count": 0, "min": None, "max": None}
    canonical = lambda value: int(value) if value.is_integer() else value
    return {
        "declared_count": len(values),
        "min": canonical(min(values)),
        "max": canonical(max(values)),
    }


def _summary(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    schema_versions = Counter(
        (
            str(task["source_schema_version"])
            if task["source_schema_version"] is not None
            else "implicit"
        )
        for task in tasks
    )
    categories = Counter(task["metadata"]["category"] for task in tasks)
    artifact_shapes = Counter(
        item["observed_shape"]
        for task in tasks
        for item in task["artifact_contract"]["items"]
    )
    artifact_declarations = Counter(
        item["declaration"]
        for task in tasks
        for item in task["artifact_contract"]["items"]
    )
    heavy_flags = Counter(
        flag for task in tasks for flag in task["heavy_toolchain_flags"]
    )
    compose_tasks = [task for task in tasks if task["services"]["present"]]
    collect_tasks = [task for task in tasks if task["collect_hooks"]["count"]]
    service_artifact_tasks = [
        task for task in tasks if task["artifact_contract"]["sidecar_scoped_count"]
    ]
    mcp_tasks = [task for task in tasks if task["mcp"]["count"]]
    gpu_tasks = [task for task in tasks if task["resources"]["agent"].get("gpus", 0)]
    verifier_gpu_tasks = [
        task for task in tasks if task["resources"]["verifier"].get("gpus", 0)
    ]
    verifier_resource_tasks = [
        task for task in tasks if task["resources"]["verifier"]["explicit_resources"]
    ]
    browser_tasks = [task for task in tasks if task["browser"]["required"]]
    return {
        "counts": {
            "tasks": len(tasks),
            "artifact_entries": sum(
                task["artifact_contract"]["count"] for task in tasks
            ),
            "compose_tasks": len(compose_tasks),
            "compose_services": sum(
                task["services"]["service_count"] for task in compose_tasks
            ),
            "collect_hook_tasks": len(collect_tasks),
            "collect_hooks": sum(
                task["collect_hooks"]["count"] for task in collect_tasks
            ),
            "sidecar_artifact_tasks": len(service_artifact_tasks),
            "sidecar_artifacts": sum(
                task["artifact_contract"]["sidecar_scoped_count"]
                for task in service_artifact_tasks
            ),
            "mcp_tasks": len(mcp_tasks),
            "mcp_servers": sum(task["mcp"]["count"] for task in mcp_tasks),
            "gpu_tasks": len(gpu_tasks),
            "verifier_gpu_tasks": len(verifier_gpu_tasks),
            "explicit_verifier_resource_tasks": len(verifier_resource_tasks),
            "browser_tasks": len(browser_tasks),
            "pytest_entrypoints": sum(
                "pytest" in task["evaluation"]["frameworks"] for task in tasks
            ),
            "ctrf_entrypoints": sum(
                "ctrf" in task["evaluation"]["report_formats"] for task in tasks
            ),
            "direct_text_reward_entrypoints": sum(
                "/logs/verifier/reward.txt" in task["evaluation"]["direct_reward_files"]
                for task in tasks
            ),
            "direct_json_reward_entrypoints": sum(
                "/logs/verifier/reward.json"
                in task["evaluation"]["direct_reward_files"]
                for task in tasks
            ),
            "explicit_determinism_entrypoints": sum(
                bool(task["evaluation"]["determinism"]) for task in tasks
            ),
            "explicit_cheat_resistance_entrypoints": sum(
                bool(task["evaluation"]["cheat_resistance"]) for task in tasks
            ),
        },
        "schema_versions": dict(sorted(schema_versions.items())),
        "categories": dict(sorted(categories.items())),
        "artifact_declaration_shapes": dict(sorted(artifact_declarations.items())),
        "artifact_observed_shapes": dict(sorted(artifact_shapes.items())),
        "heavy_toolchain_flags": dict(sorted(heavy_flags.items())),
        "resource_ranges": {
            "agent_timeout_sec": _numeric_range(
                tasks, "resources", "agent", "timeout_sec"
            ),
            "verifier_timeout_sec": _numeric_range(
                tasks, "resources", "verifier", "timeout_sec"
            ),
            "build_timeout_sec": _numeric_range(
                tasks, "resources", "agent", "build_timeout_sec"
            ),
            "agent_cpus": _numeric_range(tasks, "resources", "agent", "cpus"),
            "agent_memory_mb": _numeric_range(tasks, "resources", "agent", "memory_mb"),
            "agent_storage_mb": _numeric_range(
                tasks, "resources", "agent", "storage_mb"
            ),
        },
        "task_sets": {
            "compose": [task["slug"] for task in compose_tasks],
            "collect_hooks": [task["slug"] for task in collect_tasks],
            "sidecar_artifacts": [task["slug"] for task in service_artifact_tasks],
            "mcp": [task["slug"] for task in mcp_tasks],
            "gpu": [task["slug"] for task in gpu_tasks],
            "verifier_gpu": [task["slug"] for task in verifier_gpu_tasks],
            "explicit_verifier_resources": [
                task["slug"] for task in verifier_resource_tasks
            ],
            "browser": [task["slug"] for task in browser_tasks],
        },
    }


def _pattern_inventory(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pattern_ids = sorted(
        {pattern_id for task in tasks for pattern_id in task["evaluation"]["patterns"]}
    )
    inventory: list[dict[str, Any]] = []
    for pattern_id in pattern_ids:
        spec = ENTRYPOINT_PATTERN_SPECS.get(pattern_id)
        if spec is None:
            mapping = (
                ["evaluation.engine"]
                if pattern_id == "test.custom_shell"
                else ["result.canonical"]
            )
            group = "test" if pattern_id == "test.custom_shell" else "result"
        else:
            mapping = list(spec["mapping"])
            group = str(spec["group"])
        inventory.append(
            {
                "id": pattern_id,
                "group": group,
                "task_count": sum(
                    pattern_id in task["evaluation"]["patterns"] for task in tasks
                ),
                "mapping": mapping,
            }
        )
    return inventory


def _validate_mapping_references(matrix: Mapping[str, Any]) -> None:
    known = set(matrix["mapping_catalog"])
    references: set[str] = set()
    for inventory_name in ("task_toml_fields", "compose_fields", "test_patterns"):
        for row in matrix["observed_inventory"][inventory_name]:
            references.update(row["mapping"])
    for task in matrix["tasks"]:
        references.update(task["native_primitives"])
        for artifact in task["artifact_contract"]["items"]:
            references.update(artifact["native_candidates"])
    unknown = sorted(references - known)
    if unknown:
        raise ValueError(f"matrix references unknown mappings: {unknown}")


def build_matrix(source_root: Path) -> dict[str, Any]:
    """Parse the production corpus and return deterministic capability facts."""
    source_root = source_root.resolve()
    task_root = source_root / "tasks"
    if not task_root.is_dir():
        raise FileNotFoundError(f"Frontier-Bench task root not found: {task_root}")

    task_paths = sorted(task_root.glob("*/task.toml"))
    slugs = [path.parent.name for path in task_paths]
    if len(slugs) != len(set(slugs)):
        raise ValueError("production task slugs are not unique")
    if len(slugs) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TASK_COUNT} production tasks, found {len(slugs)}"
        )
    slug_digest = _sha256_bytes("".join(f"{slug}\n" for slug in slugs).encode())
    if slug_digest != EXPECTED_SLUGS_SHA256:
        raise ValueError(
            "production slug set changed: "
            f"expected {EXPECTED_SLUGS_SHA256}, got {slug_digest}"
        )
    excluded = [
        slug
        for slug in slugs
        if any(name in slug.lower() for name in EXCLUDED_NON_FRONTIER_NAMES)
    ]
    if excluded:
        raise ValueError(f"non-Frontier exclusions appeared in task slugs: {excluded}")

    records: list[dict[str, Any]] = []
    task_fields_by_task: dict[str, set[str]] = {}
    compose_fields_by_task: dict[str, set[str]] = {}
    source_inputs: list[Path] = []
    for task_path in task_paths:
        slug = task_path.parent.name
        test_path = task_path.parent / "tests" / "test.sh"
        if not test_path.is_file():
            raise FileNotFoundError(f"missing production test entrypoint: {test_path}")
        compose_candidates = [
            task_path.parent / "environment" / name
            for name in COMPOSE_NAMES
            if (task_path.parent / "environment" / name).is_file()
        ]
        if len(compose_candidates) > 1:
            raise ValueError(
                f"{slug} has multiple production Compose entrypoints: "
                f"{compose_candidates}"
            )
        compose_path = compose_candidates[0] if compose_candidates else None
        record, task_fields, compose_fields = _task_record(
            source_root=source_root,
            task_path=task_path,
            test_path=test_path,
            compose_path=compose_path,
        )
        records.append(record)
        task_fields_by_task[slug] = task_fields
        compose_fields_by_task[slug] = compose_fields
        source_inputs.extend([task_path, test_path])
        if compose_path:
            source_inputs.append(compose_path)

    task_inventory, unmapped_task_fields = _field_inventory(
        task_fields_by_task, TASK_FIELD_RULES
    )
    compose_inventory, unmapped_compose_fields = _field_inventory(
        compose_fields_by_task, COMPOSE_FIELD_RULES
    )
    pattern_inventory = _pattern_inventory(records)
    unmapped_patterns = [row["id"] for row in pattern_inventory if not row["mapping"]]
    unmapped_fields = sorted(
        [
            *(f"task.toml:{path}" for path in unmapped_task_fields),
            *(f"compose:{path}" for path in unmapped_compose_fields),
        ]
    )
    if unmapped_fields or unmapped_patterns:
        raise ValueError(
            "unmapped observed capability surface: "
            f"fields={unmapped_fields}, patterns={unmapped_patterns}"
        )

    matrix: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "generator": {
            "path": "scripts/generate_frontierbench_capabilities.py",
            "version": GENERATOR_VERSION,
        },
        "source": {
            "repository": "harbor-framework/frontier-bench",
            "git_revision": _git_revision(source_root),
            "scope": "tasks/*",
            "task_count": len(records),
            "slug_set_sha256": slug_digest,
            "input_file_count": len(source_inputs),
            "input_sha256": _input_digest(source_inputs, source_root),
            "parsed_entrypoints": {
                "task_toml": len(task_paths),
                "compose": sum(task["services"]["present"] for task in records),
                "tests_test_sh": len(records),
            },
            "excluded_non_frontier_names": list(EXCLUDED_NON_FRONTIER_NAMES),
        },
        "privacy": {
            "task_instructions_read": False,
            "hidden_tests_or_fixtures_read": False,
            "solutions_read": False,
            "credential_stores_read": False,
            "environment_or_command_values_serialized": False,
            "evidence_content": "metadata_and_capability_facts_only",
        },
        "summary": _summary(records),
        "observed_inventory": {
            "task_toml_fields": task_inventory,
            "compose_fields": compose_inventory,
            "test_patterns": pattern_inventory,
            "unmapped_fields": [],
            "unmapped_patterns": [],
        },
        "mapping_catalog": MAPPING_CATALOG,
        "required_native_primitives": list(REQUIRED_NATIVE_PRIMITIVES),
        "conformance_fixtures": {
            "schema": {
                "fixture": CONFORMANCE_FIXTURE,
                "test": SCHEMA_FIXTURE_TEST,
                "covers": list(REQUIRED_NATIVE_PRIMITIVES),
            },
            "export": {
                "fixture": CONFORMANCE_FIXTURE,
                "test": SCHEMA_FIXTURE_TEST,
                "covers": list(REQUIRED_NATIVE_PRIMITIVES),
            },
            "runtime": {
                "fixture": CONFORMANCE_FIXTURE,
                "test": RUNTIME_FIXTURE_TEST,
                "covers": list(REQUIRED_NATIVE_PRIMITIVES),
            },
        },
        "tasks": records,
    }
    _validate_mapping_references(matrix)
    return matrix


def _range_text(value: Mapping[str, Any], unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return (
        f"{value['min']}{suffix} to {value['max']}{suffix} "
        f"({value['declared_count']} declarations)"
    )


def _service_graph_text(task: Mapping[str, Any]) -> str:
    edges = task["services"]["edges"]
    if not edges:
        return "no dependency edges"
    return "; ".join(
        f"`{edge['service']}` -> `{edge['depends_on']}` " f"({edge['condition']})"
        for edge in edges
    )


def render_docs(matrix: Mapping[str, Any]) -> str:
    """Render the human-readable corpus and backend mapping report."""
    source = matrix["source"]
    summary = matrix["summary"]
    counts = summary["counts"]
    ranges = summary["resource_ranges"]
    tasks = {task["slug"]: task for task in matrix["tasks"]}
    artifact_declarations = ", ".join(
        f"{count} {shape}"
        for shape, count in summary["artifact_declaration_shapes"].items()
    )
    artifact_shapes = ", ".join(
        f"{count} {shape}"
        for shape, count in summary["artifact_observed_shapes"].items()
    )
    lines = [
        "# Frontier-Bench Capability Conformance",
        "",
        "This document is generated from capability-only production metadata. "
        "Regenerate it and the JSON matrix with:",
        "",
        "```bash",
        "uv run python scripts/generate_frontierbench_capabilities.py "
        "--source ../frontier-bench",
        "```",
        "",
        f"Source revision: `{source['git_revision']}`. Input digest: "
        f"`sha256:{source['input_sha256']}`. The corpus contains exactly "
        f"{counts['tasks']} unique production slugs "
        f"(`sha256:{source['slug_set_sha256']}`).",
        "",
        "The generator reads only `tasks/*/task.toml`, optional production "
        "Compose overrides, and `tasks/*/tests/test.sh`. It does not read task "
        "instructions, hidden tests or fixtures, solutions, or credential "
        "stores. Environment values and shell commands are not emitted.",
        "",
        "## Corpus counts",
        "",
        f"- {counts['artifact_entries']} artifact entries: "
        f"{artifact_declarations}; observed path shapes: {artifact_shapes}.",
        f"- {counts['compose_tasks']} Compose tasks with "
        f"{counts['compose_services']} declared services.",
        f"- {counts['collect_hook_tasks']} tasks use "
        f"{counts['collect_hooks']} pre-verification collect hooks.",
        f"- {counts['sidecar_artifact_tasks']} tasks collect "
        f"{counts['sidecar_artifacts']} artifacts from non-main services.",
        f"- {counts['mcp_tasks']} task declares {counts['mcp_servers']} "
        "task-local MCP server.",
        f"- {counts['gpu_tasks']} tasks request an agent GPU; "
        f"{counts['verifier_gpu_tasks']} also request a verifier GPU.",
        f"- {counts['explicit_verifier_resource_tasks']} tasks declare an "
        "independent verifier resource block.",
        f"- {counts['browser_tasks']} tasks have static browser-stack evidence.",
        f"- {counts['pytest_entrypoints']} verifier entrypoints invoke pytest; "
        f"{counts['ctrf_entrypoints']} emit CTRF directly.",
        f"- {counts['direct_text_reward_entrypoints']} entrypoints reference "
        "`reward.txt` directly and "
        f"{counts['direct_json_reward_entrypoints']} reference `reward.json`; "
        "the remainder delegate reward publication to their invoked suite.",
        f"- {counts['explicit_determinism_entrypoints']} entrypoints contain an "
        "explicit repetition/determinism signal and "
        f"{counts['explicit_cheat_resistance_entrypoints']} contain an explicit "
        "anti-cheat or stale-output signal.",
        "",
        "Artifact `ambiguous_path` means the legacy declaration does not say "
        "whether a suffix-less path is a file, tree, or mode-preserving binary. "
        "The JSON records all valid native candidates instead of guessing.",
        "",
        "## Time and resources",
        "",
        f"- Agent timeout: {_range_text(ranges['agent_timeout_sec'], 'seconds')}.",
        f"- Verifier timeout: "
        f"{_range_text(ranges['verifier_timeout_sec'], 'seconds')}.",
        f"- Build timeout: " f"{_range_text(ranges['build_timeout_sec'], 'seconds')}.",
        f"- Agent CPU: {_range_text(ranges['agent_cpus'], 'vCPU')}.",
        f"- Agent memory: {_range_text(ranges['agent_memory_mb'], 'MiB')}.",
        f"- Agent storage: {_range_text(ranges['agent_storage_mb'], 'MiB')}.",
        "",
        "The native mapping uses independent `ResourceSpec` declarations for "
        "agent and verifier phases. Harbor projects those to its main and "
        "separate-verifier environments. Taiga preserves the selected resource "
        "enum, validates known phase peaks for outer capsules, and rejects "
        "nested-Docker GPU/TPU child requests that Firecracker cannot satisfy.",
        "",
        "## Production service graphs",
        "",
    ]
    for task in matrix["tasks"]:
        if not task["services"]["present"]:
            continue
        services = ", ".join(f"`{name}`" for name in task["services"]["services"])
        volumes = task["services"]["named_volumes"]
        volume_text = (
            "; named volumes: " + ", ".join(f"`{name}`" for name in volumes)
            if volumes
            else ""
        )
        lines.append(
            f"- **{task['slug']}** ({task['services']['service_count']} services: "
            f"{services}) — {_service_graph_text(task)}{volume_text}."
        )

    lines.extend(
        [
            "",
            "Every graph maps to `ServiceSpec`, `ServiceDependency`, "
            "`ServiceHealthcheck`, `PortSpec`, and backend-managed `NamedVolume` "
            "as observed. Local build contexts become trusted `ServiceBuild` "
            "inputs. External images must be promoted to digest-pinned references "
            "before native export; runtime Compose builds, host binds, privileged "
            "containers, host namespaces, and runtime sockets are rejected.",
            "",
            "## Representative mappings",
            "",
        ]
    )

    wal = tasks["wal-recovery-ordering"]
    live = tasks["live-database-cutover"]
    medical = tasks["medical-claims-processing"]
    jax = tasks["jax-speedrun-gpu"]
    lines.extend(
        [
            "- **wal-recovery-ordering**: "
            f"{wal['artifact_contract']['count']} tree artifact, "
            f"patterns `{', '.join(wal['evaluation']['determinism'] + wal['evaluation']['cheat_resistance'])}`. "
            "Maps to a bounded `TreeArtifact`, structural/performance/determinism "
            "gates, canonical reward, and sealed non-root verifier execution.",
            "- **live-database-cutover**: "
            f"{live['services']['service_count']} services, "
            f"{live['collect_hooks']['count']} collect hooks, "
            f"{live['artifact_contract']['service_scoped_count']} service-qualified "
            "artifacts, and independent verifier resources. Maps to the full "
            "service graph, ordered atomic captures, sealed service artifacts, "
            "and separate verifier result handling.",
            "- **medical-claims-processing**: "
            f"{medical['services']['service_count']} services, named volume "
            f"`{medical['services']['named_volumes'][0]}`, one SSE MCP server, "
            "browser evidence, and a sidecar artifact. Maps to the browser "
            "sidecar/MCP readiness contract and shared named volume. Harbor can "
            "project it natively; Taiga supports the declared SSE endpoint only "
            "inside a trusted outer capsule through the audited service-DNS proxy. "
            "Arbitrary unbundled SSE export remains fail-closed.",
            "- **jax-speedrun-gpu**: agent and verifier each request "
            f"{jax['resources']['agent']['gpus']} "
            f"{jax['resources']['agent']['gpu_types'][0]} GPU with independent "
            "timeouts and storage. It maps to separate agent/verifier "
            "`ResourceSpec` envelopes and the accelerator backend.",
            "- **XFOIL**: excluded. No XFOIL slug or matching production task "
            f"exists in the {counts['tasks']}-task Frontier-Bench source scope at "
            "the recorded revision.",
            "",
            "## Native schema and backend mapping",
            "",
            "| Capability | Native primitive | Harbor | Taiga | Runtime |",
            "|---|---|---|---|---|",
        ]
    )

    mapping_groups = [
        (
            "Workspace",
            "workspace.lifecycle",
        ),
        (
            "Artifact shapes",
            "artifact.file, artifact.tree, artifact.path_set, artifact.binary, artifact.service",
        ),
        (
            "Compose graph",
            "service.graph, service.build, service.dependency, service.healthcheck, service.named_volume",
        ),
        (
            "Collect hooks",
            "capture.pre_verification",
        ),
        (
            "MCP/browser",
            "mcp.sse, browser.sidecar",
        ),
        (
            "Phase resources",
            "resource.agent, resource.verifier",
        ),
        (
            "Reward/test/report",
            "evaluation.engine, evaluation.gate, evaluation.report, result.canonical",
        ),
        (
            "Security/determinism",
            "security.sealed_verifier, determinism.repeated_gate",
        ),
        (
            "Heavy toolchains",
            "toolchain.image_owned",
        ),
    ]
    for label, primitive_ids in mapping_groups:
        first_id = primitive_ids.split(", ", 1)[0]
        entry = matrix["mapping_catalog"][first_id]
        backend = entry["backend_mapping"]
        lines.append(
            f"| {label} | `{primitive_ids}` | {backend['harbor']} | "
            f"{backend['taiga']} | {backend['runtime']} |"
        )

    lines.extend(
        [
            "",
            "The complete per-field and per-pattern mapping is in "
            "`docs/frontierbench_capabilities.json`. `unmapped_fields` and "
            "`unmapped_patterns` are empty; the generator fails if a new observed "
            "field or classified capability pattern lacks a mapping decision.",
            "",
            "## Conformance fixtures",
            "",
            f"`{CONFORMANCE_FIXTURE}` declares every required native primitive. "
            f"`{SCHEMA_FIXTURE_TEST}` validates the schema and export projection; "
            f"`{RUNTIME_FIXTURE_TEST}` validates the dependency-free production "
            "runtime parser. The matrix records the exact primitive coverage for "
            "all three layers.",
            "",
            "Check deterministic regeneration without writing files:",
            "",
            "```bash",
            "uv run python scripts/generate_frontierbench_capabilities.py "
            "--source ../frontier-bench --check",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_or_check(path: Path, content: str, *, check: bool) -> bool:
    if check:
        try:
            current = path.read_text()
        except FileNotFoundError:
            print(f"missing generated file: {path}", file=sys.stderr)
            return False
        if current != content:
            print(f"generated file is stale: {path}", file=sys.stderr)
            return False
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate capability-only Frontier-Bench conformance evidence.")
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Frontier-Bench repository root (default: sibling checkout).",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help="Machine-readable JSON output path.",
    )
    parser.add_argument(
        "--docs-output",
        type=Path,
        default=DEFAULT_DOCS_PATH,
        help="Human-readable Markdown output path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated output differs; do not write files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = build_matrix(args.source)
    matrix_content = _stable_json(matrix)
    docs_content = render_docs(matrix)
    matrix_ok = _write_or_check(
        args.matrix_output.resolve(),
        matrix_content,
        check=args.check,
    )
    docs_ok = _write_or_check(
        args.docs_output.resolve(),
        docs_content,
        check=args.check,
    )
    if not matrix_ok or not docs_ok:
        return 1
    action = "verified" if args.check else "generated"
    print(
        f"{action} {len(matrix['tasks'])} task records at "
        f"{matrix['source']['git_revision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
