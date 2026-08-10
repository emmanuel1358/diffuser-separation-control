"""Nested-Docker service, capture, and verifier runtime for Taiga Firecracker."""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import io
import ipaddress
import json
import math
import os
import shlex
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from grading.faults import AgentFault, InfrastructureFault
from rubric.capsule_runtime import (
    CapsuleBundle,
    CapsuleRuntimeError,
    validate_service_name,
)
from rubric.service_config import (
    CaptureHook,
    ServiceArtifact,
    ServiceConfigurationError,
    ServiceSpec,
    TaskServiceConfig,
    ToolEndpoint,
    load_task_service_config,
)
from rubric.tool_runtime import ToolRegistry

_TRUNCATION_MARKER = b"\n...[output truncated by task service runtime]...\n"
_SOCKET_MARKERS = (
    "docker.sock",
    "/var/run/docker.sock",
    "/run/docker.sock",
    "containerd.sock",
)
_MAX_WORKSPACE_SEED_BYTES = 256 * 1024 * 1024
_MAX_WORKSPACE_SEED_FILES = 100_000
_MAX_WORKSPACE_SEED_DEPTH = 128
_MAX_ARTIFACT_STREAM_BYTES = 2 * 1024**3
_MAX_ARTIFACT_STREAM_FILES = 100_000


class ServiceRuntimeError(InfrastructureFault):
    """Base class for trusted nested-runtime failures."""


class ServiceStartupError(ServiceRuntimeError):
    """Nested dockerd or an agent service failed to become ready."""


class ServiceSecurityError(ServiceRuntimeError):
    """The capsule would expose orchestration authority to a child."""


class CaptureInfrastructureError(ServiceRuntimeError):
    """A trusted capture operation failed."""


class VerifierInfrastructureError(ServiceRuntimeError):
    """The separate verifier could not run or publish a canonical result."""


class ServiceRuntimeAgentError(AgentFault):
    """The agent failed to publish a required main-service artifact."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Bounded result from one argv-based host subprocess."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class ManagedProcess(Protocol):
    """Subset of ``subprocess.Popen`` used by the runtime."""

    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...


class RuntimeCommandRunner(Protocol):
    """Injectable subprocess boundary used by all host-side commands."""

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_s: float,
        max_output_bytes: int,
        input_bytes: bytes | None = None,
    ) -> CommandResult: ...

    def start(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        output_path: Path,
        max_output_bytes: int,
    ) -> ManagedProcess: ...

    def stream_to_file(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        output_path: Path,
        timeout_s: float,
        max_bytes: int,
        max_error_bytes: int,
    ) -> CommandResult: ...

    def stop_process_group(
        self, process: ManagedProcess, *, grace_s: float
    ) -> None: ...


def _decode_output(value: bytes, *, truncated: bool) -> str:
    if truncated:
        value += _TRUNCATION_MARKER
    return value.decode("utf-8", errors="replace")


class SubprocessCommandRunner:
    """Production argv runner with bounded memory and process-group cleanup."""

    @staticmethod
    def _drain(
        stream: Any,
        *,
        limit: int,
        output: bytearray,
        truncated: list[bool],
    ) -> None:
        try:
            while chunk := stream.read(64 * 1024):
                remaining = max(0, limit - len(output))
                if remaining:
                    output.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated[0] = True
        finally:
            stream.close()

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_s: float,
        max_output_bytes: int,
        input_bytes: bytes | None = None,
    ) -> CommandResult:
        args = tuple(str(value) for value in argv)
        stdin_handle = None
        if input_bytes is not None:
            # This handle must outlive Popen construction and closes in the
            # wait/timeout finally block below.
            stdin_handle = tempfile.TemporaryFile()  # noqa: SIM115
            stdin_handle.write(input_bytes)
            stdin_handle.seek(0)
        try:
            proc = subprocess.Popen(
                args,
                stdin=stdin_handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(env),
                start_new_session=True,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            if stdin_handle is not None:
                stdin_handle.close()
            raise
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        stdout_truncated = [False]
        stderr_truncated = [False]
        stdout_thread = threading.Thread(
            target=self._drain,
            kwargs={
                "stream": proc.stdout,
                "limit": max_output_bytes,
                "output": stdout_buffer,
                "truncated": stdout_truncated,
            },
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._drain,
            kwargs={
                "stream": proc.stderr,
                "limit": max_output_bytes,
                "output": stderr_buffer,
                "truncated": stderr_truncated,
            },
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self.stop_process_group(proc, grace_s=1.0)
        finally:
            if stdin_handle is not None:
                stdin_handle.close()
        stdout_thread.join(timeout=2.0)
        stderr_thread.join(timeout=2.0)
        return CommandResult(
            argv=args,
            returncode=(
                proc.returncode if proc.returncode is not None else -signal.SIGKILL
            ),
            stdout=_decode_output(bytes(stdout_buffer), truncated=stdout_truncated[0]),
            stderr=_decode_output(bytes(stderr_buffer), truncated=stderr_truncated[0]),
            timed_out=timed_out,
            stdout_truncated=stdout_truncated[0],
            stderr_truncated=stderr_truncated[0],
        )

    def start(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        output_path: Path,
        max_output_bytes: int,
    ) -> ManagedProcess:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            tuple(str(value) for value in argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(env),
            start_new_session=True,
        )

        def drain_log() -> None:
            remaining = max_output_bytes
            assert proc.stdout is not None
            with output_path.open("wb", buffering=0) as output:
                try:
                    while chunk := proc.stdout.read(64 * 1024):
                        if remaining > 0:
                            written = chunk[:remaining]
                            output.write(written)
                            remaining -= len(written)
                finally:
                    proc.stdout.close()

        threading.Thread(
            target=drain_log,
            daemon=True,
            name=f"dockerd-log-{proc.pid}",
        ).start()
        return proc

    def stream_to_file(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        output_path: Path,
        timeout_s: float,
        max_bytes: int,
        max_error_bytes: int,
    ) -> CommandResult:
        args = tuple(str(value) for value in argv)
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env),
            start_new_session=True,
        )
        stderr_buffer = bytearray()
        stderr_truncated = [False]
        limit_exceeded = threading.Event()

        def copy_stdout() -> None:
            written = 0
            assert proc.stdout is not None
            try:
                with output_path.open("xb") as output:
                    while chunk := proc.stdout.read(64 * 1024):
                        remaining = max_bytes - written
                        if len(chunk) > remaining:
                            if remaining > 0:
                                output.write(chunk[:remaining])
                            limit_exceeded.set()
                            self.stop_process_group(proc, grace_s=0.1)
                            return
                        output.write(chunk)
                        written += len(chunk)
            finally:
                proc.stdout.close()

        stdout_thread = threading.Thread(target=copy_stdout, daemon=True)
        assert proc.stderr is not None
        stderr_thread = threading.Thread(
            target=self._drain,
            kwargs={
                "stream": proc.stderr,
                "limit": max_error_bytes,
                "output": stderr_buffer,
                "truncated": stderr_truncated,
            },
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self.stop_process_group(proc, grace_s=1.0)
        stdout_thread.join(timeout=2.0)
        stderr_thread.join(timeout=2.0)
        return CommandResult(
            argv=args,
            returncode=(
                proc.returncode if proc.returncode is not None else -signal.SIGKILL
            ),
            stderr=_decode_output(
                bytes(stderr_buffer),
                truncated=stderr_truncated[0],
            ),
            timed_out=timed_out,
            stdout_truncated=limit_exceeded.is_set(),
            stderr_truncated=stderr_truncated[0],
        )

    def stop_process_group(self, process: ManagedProcess, *, grace_s: float) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            pass
        deadline = time.monotonic() + max(0.0, grace_s)
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                pass


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One path sealed out of a live child service."""

    service: str
    source: str
    destination: str
    host_path: Path
    sha256: str
    bytes: int
    files: int
    mode: int | None


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    """Atomically published host snapshot."""

    root: Path
    records: tuple[ArtifactRecord, ...]
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class VerifierResult:
    """Canonical verifier payload and sealed result files."""

    payload: dict[str, Any]
    result_dir: Path
    exit_code: int


@dataclass(frozen=True, slots=True)
class GraderWorkspaceHandoff:
    """Explicit immutable workspace made available to the outer grader."""

    workspace: Path
    manifest: Path


def _tail_file(path: Path, limit: int = 16 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read(limit).decode(errors="replace")
    except OSError:
        return ""


def _assert_contained_path(
    path: Path, root: Path, *, label: str, allow_root: bool = False
) -> None:
    if not path.is_absolute() or not root.is_absolute() or ".." in path.parts:
        raise ServiceSecurityError(f"{label} must be an absolute normalized path")
    if path == root:
        if not allow_root:
            raise ServiceSecurityError(f"{label} must be beneath {root}")
    elif root not in path.parents:
        raise ServiceSecurityError(f"{label} escapes operator root {root}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ServiceSecurityError(f"{label} has a symlink ancestor: {current}")
        if not current.exists():
            break


def _assert_private_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ServiceSecurityError(f"{label} is not a real directory: {path}")
    details = path.stat()
    owner = details.st_uid
    effective_uid = os.geteuid()
    expected_uid = 0 if effective_uid == 0 else effective_uid
    if owner != expected_uid:
        raise ServiceSecurityError(
            f"{label} is owned by uid {owner}, expected {expected_uid}: {path}"
        )
    if stat.S_IMODE(details.st_mode) & 0o022:
        raise ServiceSecurityError(f"{label} is group/world writable: {path}")


def _mkdir_private(path: Path) -> bool:
    """Create a directory privately; never mutate permissions on existing paths."""
    if path.exists() or path.is_symlink():
        _assert_private_directory(path, label="runtime directory")
        return False
    path.mkdir(parents=False, mode=0o700)
    _assert_private_directory(path, label="runtime directory")
    return True


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


class TaskServiceRuntime:
    """Own nested dockerd, Compose services, capture, and verification."""

    def __init__(
        self,
        config: TaskServiceConfig,
        *,
        runner: RuntimeCommandRunner | None = None,
        bundle: CapsuleBundle | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        for endpoint in config.tools:
            try:
                parsed_tool_url = urlsplit(endpoint.url or "")
                parsed_tool_port = parsed_tool_url.port
            except ValueError as exc:
                raise ServiceConfigurationError(
                    f"task-local MCP server {endpoint.name!r} has an invalid URL"
                ) from exc
            if (
                endpoint.transport != "sse"
                or parsed_tool_url.scheme != "http"
                or parsed_tool_url.hostname != endpoint.service
                or parsed_tool_port is None
                or parsed_tool_url.username is not None
                or parsed_tool_url.password is not None
                or parsed_tool_url.query
                or parsed_tool_url.fragment
            ):
                raise ServiceConfigurationError(
                    f"task-local MCP server {endpoint.name!r} bypassed the "
                    "audited SSE service-DNS policy"
                )
        _assert_contained_path(
            config.capsule_dir,
            config.operator_roots.capsule,
            label="capsule directory",
            allow_root=True,
        )
        _assert_contained_path(
            config.state_dir,
            config.operator_roots.state,
            label="state directory",
        )
        _assert_contained_path(
            config.sealed_dir,
            config.operator_roots.sealed,
            label="sealed directory",
        )
        self.runner = runner or SubprocessCommandRunner()
        self.bundle = bundle
        self.tools = ToolRegistry(
            config.tools,
            sse_url_resolver=self._resolve_tool_sse_url,
        )
        self._sleep = sleep
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._daemon: ManagedProcess | None = None
        self._compose_validated = False
        self._compose_services: dict[str, Mapping[str, Any]] = {}
        self._started = False
        self._closed = False
        self._agent_stack_stopped = False
        self._snapshot: ArtifactSnapshot | None = None
        self._verifier_result: VerifierResult | None = None
        self._finalized_without_verifier = False
        self._state_created = False
        self._sealed_prepared = False
        self.cleanup_errors: list[str] = []
        self._socket_path = self.config.state_dir / "docker.sock"
        self._daemon_log = self.config.state_dir / "dockerd.log"
        path_value = (
            os.environ.get("PATH")
            or "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        self._docker_env = {
            "PATH": path_value,
            "HOME": "/root",
            "DOCKER_HOST": f"unix://{self._socket_path}",
            "TMPDIR": str(self.config.state_dir / "tmp"),
        }

    @classmethod
    def from_task_toml(
        cls,
        task_toml_path: Path = Path("/task/task.toml"),
        **kwargs: Any,
    ) -> TaskServiceRuntime | None:
        """Construct an active runtime, or ``None`` for a legacy task."""
        config = load_task_service_config(task_toml_path)
        return cls(config, **kwargs) if config is not None else None

    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def artifact_snapshot(self) -> ArtifactSnapshot | None:
        return self._snapshot

    @property
    def verifier_service(self) -> str | None:
        """Resolve a verifier declared by task config or capsule metadata."""
        if self.config.verifier_service is not None:
            return self.config.verifier_service
        if self.bundle is None:
            return None
        if self.bundle.verifier_service is not None:
            return self.bundle.verifier_service
        return next(
            (
                image.service
                for image in self.bundle.images
                if image.role.lower() in {"verifier", "grader"}
            ),
            None,
        )

    def _agent_service_names(self) -> tuple[str, ...]:
        verifier = self.verifier_service
        return tuple(
            service.name
            for service in self.config.services
            if service.name != verifier and service.role != "verifier"
        )

    def _compose_service_specs(self) -> tuple[ServiceSpec, ...]:
        specs = list(self.config.services)
        verifier = self.verifier_service
        if verifier is not None and all(spec.name != verifier for spec in specs):
            specs.append(ServiceSpec(name=verifier, role="verifier"))
        return tuple(specs)

    def _run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        input_bytes: bytes | None = None,
    ) -> CommandResult:
        return self.runner.run(
            argv,
            env=self._docker_env,
            timeout_s=timeout_s,
            max_output_bytes=self.config.max_output_bytes,
            input_bytes=input_bytes,
        )

    def _stream_to_file(
        self,
        argv: Sequence[str],
        *,
        output_path: Path,
        timeout_s: float,
        max_bytes: int,
    ) -> CommandResult:
        return self.runner.stream_to_file(
            argv,
            env=self._docker_env,
            output_path=output_path,
            timeout_s=timeout_s,
            max_bytes=max_bytes,
            max_error_bytes=self.config.max_output_bytes,
        )

    def _checked(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        error_type: type[ServiceRuntimeError] = ServiceRuntimeError,
        description: str,
        input_bytes: bytes | None = None,
    ) -> CommandResult:
        result = self._run(argv, timeout_s=timeout_s, input_bytes=input_bytes)
        if result.timed_out:
            raise error_type(f"{description} timed out after {timeout_s:g}s")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            suffix = f": {detail}" if detail else ""
            raise error_type(
                f"{description} exited with status {result.returncode}{suffix}"
            )
        return result

    def _docker_args(self, *args: str) -> tuple[str, ...]:
        return ("docker", *args)

    def _compose_args(self, *args: str) -> tuple[str, ...]:
        if self.bundle is None:
            raise ServiceStartupError("task capsule has not been loaded")
        return (
            "docker",
            "compose",
            "--ansi",
            "never",
            "-f",
            str(self.bundle.compose_path),
            "--project-name",
            self.config.project_name,
            *args,
        )

    def _prepare_state(self) -> None:
        state_root = self.config.operator_roots.state
        _assert_contained_path(
            state_root, state_root, label="state root", allow_root=True
        )
        if not state_root.exists():
            if not state_root.parent.is_dir() or state_root.parent.is_symlink():
                raise ServiceSecurityError(
                    f"state root parent is unavailable: {state_root.parent}"
                )
            _mkdir_private(state_root)
        else:
            _assert_private_directory(state_root, label="state root")
        if self.config.state_dir.exists() or self.config.state_dir.is_symlink():
            raise ServiceSecurityError(
                "refusing to reuse pre-existing service state directory: "
                f"{self.config.state_dir}"
            )
        self._state_created = _mkdir_private(self.config.state_dir)
        _mkdir_private(self.config.state_dir / "tmp")
        _mkdir_private(self.config.state_dir / "data")
        _mkdir_private(self.config.state_dir / "exec")

    def _prepare_sealed(self) -> None:
        if self._sealed_prepared:
            return
        sealed_root = self.config.operator_roots.sealed
        _assert_contained_path(
            sealed_root, sealed_root, label="sealed root", allow_root=True
        )
        if not sealed_root.exists():
            if not sealed_root.parent.is_dir() or sealed_root.parent.is_symlink():
                raise ServiceSecurityError(
                    f"sealed root parent is unavailable: {sealed_root.parent}"
                )
            _mkdir_private(sealed_root)
        else:
            _assert_private_directory(sealed_root, label="sealed root")
        _mkdir_private(self.config.sealed_dir)
        self._sealed_prepared = True

    def _start_daemon(self) -> None:
        args = (
            "dockerd",
            "--host",
            f"unix://{self._socket_path}",
            "--data-root",
            str(self.config.state_dir / "data"),
            "--exec-root",
            str(self.config.state_dir / "exec"),
            "--pidfile",
            str(self.config.state_dir / "dockerd.pid"),
            "--log-level",
            "error",
        )
        try:
            self._daemon = self.runner.start(
                args,
                env=self._docker_env,
                output_path=self._daemon_log,
                max_output_bytes=self.config.max_output_bytes,
            )
        except (OSError, RuntimeError) as exc:
            raise ServiceStartupError(f"could not start nested dockerd: {exc}") from exc

        deadline = self._monotonic() + self.config.daemon_timeout_s
        last_detail = ""
        while self._monotonic() < deadline:
            if self._daemon.poll() is not None:
                detail = _tail_file(self._daemon_log)
                raise ServiceStartupError(
                    "nested dockerd exited before readiness"
                    + (f": {detail.strip()}" if detail.strip() else "")
                )
            result = self._run(
                self._docker_args("info", "--format", "{{json .ServerVersion}}"),
                timeout_s=min(5.0, self.config.daemon_timeout_s),
            )
            if result.returncode == 0 and not result.timed_out:
                return
            last_detail = (result.stderr or result.stdout).strip()
            self._sleep(0.25)
        raise ServiceStartupError(
            f"nested dockerd did not become ready within "
            f"{self.config.daemon_timeout_s:g}s"
            + (f": {last_detail}" if last_detail else "")
        )

    def _load_capsule_images(self) -> None:
        if self.bundle is None:
            raise ServiceStartupError("task capsule has not been loaded")
        self.bundle.verify_archives()
        loaded_archives: set[Path] = set()
        for image in self.bundle.images:
            if image.archive_path not in loaded_archives:
                self._checked(
                    self._docker_args(
                        "image", "load", "--input", str(image.archive_path)
                    ),
                    timeout_s=self.config.startup_timeout_s,
                    error_type=ServiceStartupError,
                    description=(
                        f"loading image archive for service {image.service!r}"
                    ),
                )
                loaded_archives.add(image.archive_path)
            inspected = self._checked(
                self._docker_args("image", "inspect", image.image_ref),
                timeout_s=30.0,
                error_type=ServiceStartupError,
                description=f"inspecting loaded image for service {image.service!r}",
            )
            try:
                rows = json.loads(inspected.stdout)
            except json.JSONDecodeError as exc:
                raise ServiceStartupError(
                    f"docker returned invalid image metadata for {image.service!r}"
                ) from exc
            if (
                not isinstance(rows, list)
                or len(rows) != 1
                or not isinstance(rows[0], dict)
            ):
                raise ServiceStartupError(
                    f"docker returned unexpected image metadata for {image.service!r}"
                )
            row = rows[0]
            observed = {str(row.get("Id") or "")}
            repo_digests = row.get("RepoDigests")
            if isinstance(repo_digests, list):
                for value in repo_digests:
                    if isinstance(value, str) and "@" in value:
                        observed.add(value.rsplit("@", 1)[1])
            if image.image_digest not in observed:
                raise ServiceStartupError(
                    f"loaded image digest mismatch for service {image.service!r}: "
                    f"expected {image.image_digest}"
                )

    @staticmethod
    def _environment_dict(value: Any) -> dict[str, str]:
        if isinstance(value, Mapping):
            return {str(key): str(item) for key, item in value.items()}
        output: dict[str, str] = {}
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and "=" in item:
                    key, raw = item.split("=", 1)
                    output[key] = raw
        return output

    def _assert_safe_service(self, service: str, row: Mapping[str, Any]) -> None:
        values: list[str] = []
        volumes = row.get("volumes", [])
        if not isinstance(volumes, list):
            raise ServiceSecurityError(
                f"service {service!r} has malformed volume declarations"
            )
        for volume in volumes:
            if isinstance(volume, str):
                values.append(volume)
                source = volume.split(":", 1)[0]
                if source.startswith(("/", ".", "~")):
                    raise ServiceSecurityError(
                        f"service {service!r} contains a bind mount"
                    )
            elif isinstance(volume, Mapping):
                values.extend(
                    str(volume.get(name) or "") for name in ("source", "target", "type")
                )
                if str(volume.get("type") or "").lower() == "bind":
                    raise ServiceSecurityError(
                        f"service {service!r} contains a bind mount"
                    )
            else:
                raise ServiceSecurityError(
                    f"service {service!r} has malformed volume declarations"
                )
        for name in ("devices", "volumes_from"):
            raw = row.get(name)
            if raw not in (None, [], {}):
                raise ServiceSecurityError(f"service {service!r} cannot declare {name}")
        if row.get("privileged") not in (None, False):
            raise ServiceSecurityError(f"service {service!r} cannot run privileged")
        capabilities = row.get("cap_add", row.get("capabilities", []))
        capability_values = (
            [str(value).upper().removeprefix("CAP_") for value in capabilities]
            if isinstance(capabilities, list)
            else [str(capabilities).upper().removeprefix("CAP_")]
        )
        dangerous_capabilities = {
            "ALL",
            "DAC_OVERRIDE",
            "DAC_READ_SEARCH",
            "MKNOD",
            "SYS_ADMIN",
            "SYS_CHROOT",
            "SYS_MODULE",
            "SYS_RAWIO",
        }
        forbidden_capabilities = sorted(
            dangerous_capabilities.intersection(capability_values)
        )
        if forbidden_capabilities:
            raise ServiceSecurityError(
                f"service {service!r} requests dangerous capabilities: "
                + ", ".join(forbidden_capabilities)
            )
        for name in (
            "cgroup",
            "ipc",
            "network_mode",
            "pid",
            "userns_mode",
            "uts",
        ):
            if str(row.get(name) or "").lower() == "host":
                raise ServiceSecurityError(
                    f"service {service!r} cannot join the host {name} namespace"
                )
        security_options = row.get("security_opt", [])
        if not isinstance(security_options, list):
            raise ServiceSecurityError(
                f"service {service!r} has malformed security_opt"
            )
        normalized_security = {
            str(option).lower().replace("=", ":") for option in security_options
        }
        if any("unconfined" in option for option in normalized_security):
            raise ServiceSecurityError(
                f"service {service!r} cannot disable container confinement"
            )
        if not any(
            option
            in {
                "no-new-privileges",
                "no-new-privileges:1",
                "no-new-privileges:true",
            }
            for option in normalized_security
        ):
            raise ServiceSecurityError(
                f"service {service!r} must set no-new-privileges"
            )
        cap_drop = row.get("cap_drop", [])
        if not isinstance(cap_drop, list) or "ALL" not in {
            str(value).upper().removeprefix("CAP_") for value in cap_drop
        }:
            raise ServiceSecurityError(
                f"service {service!r} must drop all capabilities"
            )
        joined = "\n".join(values).lower()
        configured_socket = str(self._socket_path).lower()
        if any(marker in joined for marker in _SOCKET_MARKERS) or (
            configured_socket and configured_socket in joined
        ):
            raise ServiceSecurityError(
                f"service {service!r} attempts to mount or inherit a container "
                "runtime socket"
            )
        environment = self._environment_dict(row.get("environment"))
        for key in ("DOCKER_HOST", "CONTAINER_HOST", "CONTAINERD_ADDRESS"):
            if environment.get(key):
                raise ServiceSecurityError(
                    f"service {service!r} must not receive orchestration variable {key}"
                )

    @staticmethod
    def _service_dependencies(row: Mapping[str, Any]) -> set[str]:
        raw = row.get("depends_on", {})
        if isinstance(raw, Mapping):
            return {str(name) for name in raw}
        if isinstance(raw, list):
            return {str(name) for name in raw}
        if raw in (None, ""):
            return set()
        raise ServiceSecurityError("service has malformed depends_on")

    @staticmethod
    def _named_volume_sources(row: Mapping[str, Any]) -> set[str]:
        sources: set[str] = set()
        volumes = row.get("volumes", [])
        if not isinstance(volumes, list):
            return sources
        for volume in volumes:
            if isinstance(volume, str):
                source = volume.split(":", 1)[0]
                if source and not source.startswith(("/", ".", "~")):
                    sources.add(source)
            elif isinstance(volume, Mapping):
                if str(volume.get("type") or "volume").lower() != "volume":
                    continue
                source = volume.get("source")
                if isinstance(source, str) and source:
                    sources.add(source)
        return sources

    def _validate_compose(self) -> None:
        configured = self._checked(
            self._compose_args("config", "--format", "json"),
            timeout_s=30.0,
            error_type=ServiceStartupError,
            description="validating capsule Compose configuration",
        )
        try:
            payload = json.loads(configured.stdout)
        except json.JSONDecodeError as exc:
            raise ServiceStartupError(
                "docker compose config returned invalid JSON"
            ) from exc
        services = payload.get("services") if isinstance(payload, Mapping) else None
        if not isinstance(services, Mapping):
            raise ServiceStartupError(
                "docker compose config did not contain a services table"
            )
        if self.bundle is None:
            raise ServiceStartupError("task capsule has not been loaded")
        specs = self._compose_service_specs()
        expected_services = {spec.name for spec in specs}
        unexpected_services = sorted(set(services) - expected_services)
        if unexpected_services:
            raise ServiceSecurityError(
                "Compose contains undeclared services that could start as "
                f"dependencies: {unexpected_services}"
            )
        for spec in specs:
            row = services.get(spec.name)
            if not isinstance(row, Mapping):
                raise ServiceStartupError(
                    f"configured service {spec.name!r} is absent from Compose"
                )
            if row.get("build") not in (None, {}):
                raise ServiceSecurityError(
                    f"runtime Compose service {spec.name!r} contains a build; "
                    "child images must be built in trusted packaging"
                )
            image = row.get("image")
            if not isinstance(image, str) or not image:
                raise ServiceSecurityError(
                    f"runtime Compose service {spec.name!r} has no image"
                )
            locked = self.bundle.image_for_service(spec.name)
            if image != locked.image_ref:
                raise ServiceSecurityError(
                    f"Compose image for service {spec.name!r} is not the "
                    "digest-locked capsule image reference"
                )
            pull_policy = row.get("pull_policy")
            if pull_policy not in (None, "never"):
                raise ServiceSecurityError(
                    f"service {spec.name!r} permits mutable runtime pulls"
                )
            self._assert_safe_service(spec.name, row)
        verifier = self.verifier_service
        if verifier is not None:
            verifier_row = services[verifier]
            if str(verifier_row.get("network_mode") or "").lower() != "none":
                raise ServiceSecurityError(
                    "verifier service must set network_mode='none'"
                )
            verifier_dependencies = self._service_dependencies(verifier_row)
            if verifier_dependencies:
                raise ServiceSecurityError(
                    "verifier service cannot depend on agent services"
                )
            if verifier_row.get("links") not in (None, [], {}):
                raise ServiceSecurityError(
                    "verifier service cannot link to agent services"
                )
            for namespace in ("ipc", "pid"):
                if str(verifier_row.get(namespace) or "").startswith("service:"):
                    raise ServiceSecurityError(
                        f"verifier service cannot share {namespace} with agent "
                        "services"
                    )
            verifier_volumes = self._named_volume_sources(verifier_row)
            for name, raw_row in services.items():
                if name == verifier or not isinstance(raw_row, Mapping):
                    continue
                dependencies = self._service_dependencies(raw_row)
                if verifier in dependencies:
                    raise ServiceSecurityError(
                        f"agent service {name!r} cannot depend on verifier"
                    )
                for namespace in ("ipc", "network_mode", "pid"):
                    if str(raw_row.get(namespace) or "") == f"service:{verifier}":
                        raise ServiceSecurityError(
                            f"agent service {name!r} cannot share {namespace} "
                            "with verifier"
                        )
                shared_volumes = verifier_volumes.intersection(
                    self._named_volume_sources(raw_row)
                )
                if shared_volumes:
                    raise ServiceSecurityError(
                        "agent and verifier services cannot share volumes: "
                        f"{sorted(shared_volumes)}"
                    )
        self._compose_services = {
            str(name): row
            for name, row in services.items()
            if isinstance(name, str) and isinstance(row, Mapping)
        }
        self._compose_validated = True

    def _service_container_id(self, service: str) -> str | None:
        service = validate_service_name(service)
        result = self._run(
            self._compose_args("ps", "--all", "--quiet", service),
            timeout_s=10.0,
        )
        if result.returncode != 0 or result.timed_out:
            return None
        return next(
            (line.strip() for line in result.stdout.splitlines() if line.strip()), None
        )

    def _resolve_tool_sse_url(self, endpoint: ToolEndpoint) -> str:
        if not self._started or self._agent_stack_stopped:
            raise ServiceRuntimeError(
                f"task-local MCP server {endpoint.name!r} is not running"
            )
        if endpoint.url is None:
            raise ServiceConfigurationError(
                f"task-local MCP server {endpoint.name!r} has no SSE URL"
            )
        container_id = self._service_container_id(endpoint.service)
        if container_id is None:
            raise ServiceRuntimeError(
                f"task-local MCP service {endpoint.service!r} has no container"
            )
        inspected = self._checked(
            self._docker_args(
                "inspect",
                "--format",
                "{{json .NetworkSettings.Networks}}",
                container_id,
            ),
            timeout_s=10.0,
            error_type=ServiceRuntimeError,
            description=(f"resolving task-local MCP service {endpoint.service!r}"),
        )
        try:
            networks = json.loads(inspected.stdout)
        except json.JSONDecodeError as exc:
            raise ServiceRuntimeError(
                f"nested Docker returned invalid network metadata for "
                f"{endpoint.service!r}"
            ) from exc
        if not isinstance(networks, Mapping):
            raise ServiceRuntimeError(
                f"nested Docker returned malformed network metadata for "
                f"{endpoint.service!r}"
            )
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
        for network_name in sorted(networks):
            network = networks[network_name]
            if not isinstance(network, Mapping):
                continue
            for field in ("IPAddress", "GlobalIPv6Address"):
                raw = network.get(field)
                if not isinstance(raw, str) or not raw:
                    continue
                try:
                    address = ipaddress.ip_address(raw)
                except ValueError:
                    continue
                if address.is_private and not (
                    address.is_loopback
                    or address.is_link_local
                    or address.is_multicast
                    or address.is_unspecified
                ):
                    addresses.append(address)
        if not addresses:
            raise ServiceRuntimeError(
                f"task-local MCP service {endpoint.service!r} has no private "
                "container address"
            )
        parsed = urlsplit(endpoint.url)
        address = addresses[0]
        hostname = f"[{address}]" if address.version == 6 else str(address)
        netloc = f"{hostname}:{parsed.port}"
        return urlunsplit(("http", netloc, parsed.path, "", ""))

    def _service_state(self, service: str) -> dict[str, Any] | None:
        container_id = self._service_container_id(service)
        if container_id is None:
            return None
        result = self._run(
            self._docker_args("inspect", "--format", "{{json .State}}", container_id),
            timeout_s=10.0,
        )
        if result.returncode != 0 or result.timed_out:
            return None
        try:
            state = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        return state if isinstance(state, dict) else None

    @staticmethod
    def _state_ready(spec: ServiceSpec, state: Mapping[str, Any]) -> bool:
        status = str(state.get("Status") or "").lower()
        if spec.role == "init":
            return status == "exited" and int(state.get("ExitCode", -1)) == 0
        if status != "running" and state.get("Running") is not True:
            return False
        health = state.get("Health")
        if isinstance(health, Mapping):
            return str(health.get("Status") or "").lower() == "healthy"
        return True

    @staticmethod
    def _state_failed(spec: ServiceSpec, state: Mapping[str, Any]) -> str | None:
        status = str(state.get("Status") or "").lower()
        health = state.get("Health")
        health_status = (
            str(health.get("Status") or "").lower()
            if isinstance(health, Mapping)
            else ""
        )
        if health_status == "unhealthy":
            return "reported unhealthy"
        if spec.role == "init" and status == "exited":
            exit_code = int(state.get("ExitCode", -1))
            if exit_code != 0:
                return f"init container exited with status {exit_code}"
        if spec.role != "init" and status in {"dead", "exited", "removing"}:
            return f"container entered state {status!r}"
        return None

    def _wait_services(
        self,
        specs: Sequence[ServiceSpec],
        *,
        timeout_s: float,
        error_type: type[ServiceRuntimeError] = ServiceStartupError,
    ) -> None:
        deadline = self._monotonic() + timeout_s
        last_states: dict[str, Any] = {}
        pending = {spec.name: spec for spec in specs}
        while pending and self._monotonic() < deadline:
            for name, spec in tuple(pending.items()):
                state = self._service_state(name)
                if state is None:
                    continue
                last_states[name] = state
                failure = self._state_failed(spec, state)
                if failure:
                    raise error_type(f"service {name!r} {failure}")
                if self._state_ready(spec, state):
                    pending.pop(name)
            if pending:
                self._sleep(0.25)
        if pending:
            summary = json.dumps(last_states, sort_keys=True, default=str)
            raise error_type(
                f"services did not become ready within {timeout_s:g}s: "
                f"{sorted(pending)}; last states={summary}"
            )

    def _exec_service_argv(
        self,
        service: str,
        command: Sequence[str],
        *,
        user: str | None = None,
        workdir: str | None = None,
        timeout_s: float,
        input_bytes: bytes | None = None,
    ) -> CommandResult:
        service = validate_service_name(service)
        args: list[str] = ["exec", "--no-TTY"]
        if user is not None:
            args.extend(("--user", user))
        if workdir is not None:
            args.extend(("--workdir", workdir))
        args.append(service)
        args.extend(str(value) for value in command)
        return self._run(
            self._compose_args(*args),
            timeout_s=timeout_s,
            input_bytes=input_bytes,
        )

    def _exec_service_shell(
        self,
        service: str,
        command: str,
        *,
        user: str | None,
        timeout_s: float,
        workdir: str | None = None,
    ) -> CommandResult:
        spec = self.config.service(service)
        return self._exec_service_argv(
            service,
            (spec.shell, "-lc", command),
            user=user,
            workdir=workdir,
            timeout_s=timeout_s,
        )

    def _checked_main_setup(
        self,
        command: Sequence[str],
        *,
        description: str,
        input_bytes: bytes | None = None,
    ) -> CommandResult:
        result = self._exec_service_argv(
            self.config.main_service,
            command,
            user=self.config.agent_user,
            timeout_s=self.config.startup_timeout_s,
            input_bytes=input_bytes,
        )
        if result.timed_out:
            raise ServiceStartupError(f"{description} timed out")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ServiceStartupError(
                f"{description} exited with status {result.returncode}"
                + (f": {detail}" if detail else "")
            )
        return result

    def _workspace_seed_archive(self, seed: str) -> bytes:
        task_root = self.config.task_toml_path.parent
        _assert_contained_path(task_root, task_root, label="task root", allow_root=True)
        _assert_contained_path(
            task_root / seed,
            task_root,
            label="workspace seed",
            allow_root=True,
        )
        try:
            resolved_root = task_root.resolve(strict=True)
            seed_path = (resolved_root / seed).resolve(strict=True)
        except OSError as exc:
            raise ServiceStartupError(
                f"workspace seed {seed!r} is unavailable: {exc}"
            ) from exc
        if seed_path != resolved_root and resolved_root not in seed_path.parents:
            raise ServiceSecurityError(f"workspace seed escapes task root: {seed!r}")
        candidates = [seed_path]
        if seed_path.is_dir():
            candidates.extend(sorted(seed_path.rglob("*")))
        total_bytes = 0
        files = 0
        for candidate in candidates:
            if candidate.is_symlink():
                raise ServiceSecurityError(
                    f"workspace seed contains a symlink: {candidate}"
                )
            relative = candidate.relative_to(seed_path)
            if len(relative.parts) > _MAX_WORKSPACE_SEED_DEPTH:
                raise ServiceSecurityError(
                    f"workspace seed exceeds depth {_MAX_WORKSPACE_SEED_DEPTH}"
                )
            if candidate.is_file():
                files += 1
                total_bytes += candidate.stat().st_size
            elif not candidate.is_dir():
                raise ServiceSecurityError(
                    f"workspace seed contains a special file: {candidate}"
                )
            if (
                files > _MAX_WORKSPACE_SEED_FILES
                or total_bytes > _MAX_WORKSPACE_SEED_BYTES
            ):
                raise ServiceSecurityError("workspace seed exceeds runtime limits")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as handle:
            handle.add(
                seed_path,
                arcname="." if seed_path.is_dir() else seed_path.name,
                recursive=True,
            )
        if archive.tell() > _MAX_WORKSPACE_SEED_BYTES + 16 * 1024 * 1024:
            raise ServiceSecurityError("workspace seed archive exceeds runtime limit")
        return archive.getvalue()

    def _initialize_workspace(self) -> None:
        workspace = self.config.workspace
        if workspace is None:
            return
        marker_payload = json.dumps(
            {
                "checkpoint_restore": workspace.checkpoint_restore,
                "clean_paths": workspace.clean_paths,
                "git_baseline": workspace.git_baseline,
                "init_policy": workspace.init_policy,
                "root": workspace.root,
                "seed": workspace.seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        marker_path = f"/tmp/.lbx-workspace-{self.config.project_name}.json"
        if workspace.checkpoint_restore:
            marker = self._exec_service_argv(
                self.config.main_service,
                (
                    "/bin/sh",
                    "-c",
                    'test -f "$1" && test ! -L "$1" && cat -- "$1"',
                    "lbx-workspace",
                    marker_path,
                ),
                user=self.config.agent_user,
                timeout_s=30.0,
            )
            if (
                marker.returncode == 0
                and not marker.timed_out
                and marker.stdout == marker_payload
            ):
                return
        create_script = (
            'set -eu; mkdir -p -- "$1" "$2"; '
            'test -d "$1" && test ! -L "$1"; '
            'test -d "$2" && test ! -L "$2"'
        )
        self._checked_main_setup(
            (
                "/bin/sh",
                "-c",
                create_script,
                "lbx-workspace",
                workspace.root,
                workspace.agent_cwd,
            ),
            description="creating the configured workspace",
        )
        if workspace.init_policy in {"copy", "empty"}:
            clear_script = (
                'set -eu; find "$1" -mindepth 1 -maxdepth 1 ' "-exec rm -rf -- {} +"
            )
            self._checked_main_setup(
                (
                    "/bin/sh",
                    "-c",
                    clear_script,
                    "lbx-workspace",
                    workspace.root,
                ),
                description="clearing the configured workspace",
            )
        if workspace.init_policy in {"copy", "overlay"}:
            assert workspace.seed is not None
            archive = self._workspace_seed_archive(workspace.seed)
            self._checked_main_setup(
                ("tar", "-xf", "-", "-C", workspace.root),
                description="provisioning the configured workspace seed",
                input_bytes=archive,
            )
        elif workspace.init_policy == "reuse":
            self._checked_main_setup(
                ("test", "-d", workspace.root),
                description="validating the reusable workspace",
            )
        for clean_path in workspace.clean_paths:
            self._checked_main_setup(
                ("rm", "-rf", "--", f"{workspace.root}/{clean_path}"),
                description=f"cleaning workspace path {clean_path!r}",
            )
        if workspace.git_baseline is True:
            self._checked_main_setup(
                ("git", "-C", workspace.root, "init"),
                description="initializing workspace git baseline",
            )
            self._checked_main_setup(
                ("git", "-C", workspace.root, "add", "-A"),
                description="staging workspace git baseline",
            )
            self._checked_main_setup(
                (
                    "git",
                    "-C",
                    workspace.root,
                    "-c",
                    "user.name=LBX Runtime",
                    "-c",
                    "user.email=runtime@invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--allow-empty",
                    "-m",
                    "LBX workspace baseline",
                ),
                description="committing workspace git baseline",
            )
        elif isinstance(workspace.git_baseline, str):
            self._checked_main_setup(
                (
                    "git",
                    "-C",
                    workspace.root,
                    "rev-parse",
                    "--verify",
                    f"{workspace.git_baseline}^{{commit}}",
                ),
                description="validating immutable workspace git baseline",
            )
        marker_script = (
            'set -eu; target="$1"; tmp="$target.tmp-$$"; '
            "trap 'rm -f -- \"$tmp\"' EXIT HUP INT TERM; "
            'cat > "$tmp"; chmod 0600 "$tmp"; mv -f -- "$tmp" "$target"; trap - EXIT'
        )
        self._checked_main_setup(
            ("/bin/sh", "-c", marker_script, "lbx-workspace", marker_path),
            description="publishing workspace checkpoint marker",
            input_bytes=marker_payload.encode(),
        )

    def _wait_tools(self) -> None:
        for endpoint in self.config.tools:
            readiness_command = endpoint.readiness_command
            if endpoint.readiness_kind == "http" and endpoint.readiness_url:
                quoted_url = shlex.quote(endpoint.readiness_url)
                readiness_command = (
                    "if command -v curl >/dev/null 2>&1; then "
                    f"curl --fail --silent --show-error --max-time 5 {quoted_url} "
                    ">/dev/null; elif command -v wget >/dev/null 2>&1; then "
                    f"wget -q -T 5 -O /dev/null {quoted_url}; else exit 127; fi"
                )
            elif (
                endpoint.readiness_kind == "tcp"
                and endpoint.readiness_host
                and endpoint.readiness_port is not None
            ):
                host = repr(endpoint.readiness_host)
                port = endpoint.readiness_port
                readiness_command = "python3 -c " + shlex.quote(
                    "import socket; "
                    f"s=socket.create_connection(({host},{port}),5); s.close()"
                )
            elif endpoint.readiness_kind not in (None, "command"):
                raise ServiceStartupError(
                    f"tool {endpoint.name!r} has incomplete or unsupported "
                    f"readiness kind {endpoint.readiness_kind!r}"
                )
            if readiness_command is None:
                self.tools.set_ready(
                    endpoint.name,
                    True,
                    f"service {endpoint.service!r} passed container readiness",
                )
                continue
            deadline = self._monotonic() + endpoint.readiness_timeout_s
            last_detail = ""
            while self._monotonic() < deadline:
                result = self._exec_service_shell(
                    endpoint.readiness_service,
                    readiness_command,
                    user=None,
                    timeout_s=min(10.0, endpoint.readiness_timeout_s),
                )
                if result.returncode == 0 and not result.timed_out:
                    self.tools.set_ready(endpoint.name, True)
                    break
                last_detail = (result.stderr or result.stdout).strip()
                self._sleep(endpoint.readiness_interval_s)
            else:
                self.tools.set_ready(endpoint.name, False, last_detail or "timeout")
                raise ServiceStartupError(
                    f"tool {endpoint.name!r} did not become ready within "
                    f"{endpoint.readiness_timeout_s:g}s"
                )

    def _verify_main_uids(self) -> None:
        checks = (
            (None, "default main-service user"),
            (self.config.agent_user, "configured agent proxy user"),
        )
        for user, label in checks:
            result = self._exec_service_argv(
                self.config.main_service,
                ("id", "-u"),
                user=user,
                timeout_s=30.0,
            )
            raw_uid = result.stdout.strip()
            if (
                result.returncode != 0
                or result.timed_out
                or not raw_uid.isdecimal()
                or int(raw_uid, 10) == 0
            ):
                detail = (result.stderr or result.stdout).strip()
                raise ServiceSecurityError(
                    f"{label} must resolve to a nonzero UID inside the container"
                    + (f": {detail}" if detail else "")
                )

    def start(self) -> None:
        """Start the nested daemon and the agent-facing Compose stack once."""
        with self._lock:
            if self._started:
                return
            if self._closed:
                raise ServiceRuntimeError("service runtime is already closed")
            try:
                if self.bundle is None:
                    self.bundle = CapsuleBundle.load(
                        self.config.capsule_dir,
                        manifest_name=self.config.capsule_manifest,
                        compose_override=self.config.compose_file,
                    )
                self._prepare_state()
                self._start_daemon()
                self._load_capsule_images()
                self._validate_compose()
                verifier_service = self.verifier_service
                specs = [
                    service
                    for service in self.config.services
                    if service.role != "verifier" and service.name != verifier_service
                ]
                self._checked(
                    self._compose_args(
                        "up",
                        "--detach",
                        "--no-build",
                        "--pull",
                        "never",
                        *(service.name for service in specs),
                    ),
                    timeout_s=self.config.startup_timeout_s,
                    error_type=ServiceStartupError,
                    description="starting nested agent service stack",
                )
                self._wait_services(specs, timeout_s=self.config.startup_timeout_s)
                self._verify_main_uids()
                self._initialize_workspace()
                self._wait_tools()
                self._started = True
            except (
                CapsuleRuntimeError,
                ServiceConfigurationError,
                ServiceRuntimeError,
            ):
                self.cleanup()
                raise
            except Exception as exc:
                self.cleanup()
                raise ServiceStartupError(
                    f"unexpected nested service startup failure: {type(exc).__name__}: {exc}"
                ) from exc

    def exec_main(
        self, command: str, *, timeout_s: float | None = None
    ) -> CommandResult:
        """Execute an agent command in ``main`` as its unprivileged user."""
        with self._lock:
            if not self._started or self._agent_stack_stopped:
                return CommandResult(
                    argv=(),
                    returncode=1,
                    stderr="main service is not running",
                )
            return self._exec_service_argv(
                self.config.main_service,
                (self.config.agent_shell, "-lc", command),
                user=self.config.agent_user,
                workdir=self.config.agent_workdir,
                timeout_s=timeout_s or self.config.command_timeout_s,
            )

    async def task_mcp_list_tools(self, server: str) -> dict[str, Any]:
        """Proxy a bounded MCP ``tools/list`` request to a ready child service."""
        return await self.tools.list_tools(server)

    async def task_mcp_call(
        self,
        server: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Proxy a bounded MCP ``tools/call`` request to a ready child service."""
        return await self.tools.call_tool(server, tool_name, arguments)

    def restart_main(self) -> None:
        """Restart only the main service, then re-apply health readiness."""
        with self._lock:
            if not self._started or self._agent_stack_stopped:
                raise ServiceRuntimeError("main service is not running")
            self._checked(
                self._compose_args("restart", self.config.main_service),
                timeout_s=self.config.startup_timeout_s,
                error_type=ServiceStartupError,
                description="restarting main service",
            )
            self._wait_services(
                (self.config.service(self.config.main_service),),
                timeout_s=self.config.startup_timeout_s,
            )

    def _resolve_agent_path(self, raw_path: str) -> str:
        target = PurePosixPath(raw_path)
        if not target.is_absolute():
            target = PurePosixPath(self.config.agent_workdir) / target
        if ".." in target.parts:
            raise PermissionError(f"agent path contains '..': {raw_path!r}")
        roots = (
            PurePosixPath(self.config.agent_workdir),
            PurePosixPath("/tmp/output"),
            *(
                (PurePosixPath(self.config.workspace.root),)
                if self.config.workspace is not None
                else ()
            ),
        )
        if not any(target == root or target.is_relative_to(root) for root in roots):
            raise PermissionError(
                f"path {raw_path!r} is outside nested agent roots "
                f"{tuple(str(root) for root in roots)}"
            )
        return str(target)

    def read_main_file(self, raw_path: str) -> str:
        """Read one regular main-service file as the agent identity."""
        target = self._resolve_agent_path(raw_path)
        result = self._exec_service_argv(
            self.config.main_service,
            (
                "/bin/sh",
                "-c",
                'test -f "$1" && test ! -L "$1" && cat -- "$1"',
                "lbx-editor",
                target,
            ),
            user=self.config.agent_user,
            workdir=self.config.agent_workdir,
            timeout_s=min(60.0, self.config.command_timeout_s),
        )
        if result.timed_out:
            raise TimeoutError(f"reading {target} timed out")
        if result.returncode != 0:
            raise OSError(
                (result.stderr or result.stdout).strip() or f"cannot read {target}"
            )
        if result.stdout_truncated:
            raise OSError(
                f"{target} exceeds the {self.config.max_output_bytes}-byte read limit"
            )
        if len(result.stdout.encode()) > self.config.editor_max_bytes:
            raise OSError(
                f"{target} exceeds the {self.config.editor_max_bytes}-byte editor limit"
            )
        return result.stdout

    def _write_main_bytes(self, raw_path: str, content: bytes) -> None:
        """Atomically write bounded bytes as the main-service agent identity."""
        target = self._resolve_agent_path(raw_path)
        if len(content) > self.config.editor_max_bytes:
            raise OSError(
                f"content exceeds the {self.config.editor_max_bytes}-byte limit"
            )
        script = (
            'set -eu; target="$1"; parent=$(dirname -- "$target"); '
            'mkdir -p -- "$parent"; tmp="$parent/.lbx-edit-$$"; '
            "trap 'rm -f -- \"$tmp\"' EXIT HUP INT TERM; "
            'cat > "$tmp"; chmod 0600 "$tmp"; mv -f -- "$tmp" "$target"; trap - EXIT'
        )
        result = self._exec_service_argv(
            self.config.main_service,
            ("/bin/sh", "-c", script, "lbx-editor", target),
            user=self.config.agent_user,
            workdir=self.config.agent_workdir,
            timeout_s=min(60.0, self.config.command_timeout_s),
            input_bytes=content,
        )
        if result.timed_out:
            raise TimeoutError(f"writing {target} timed out")
        if result.returncode != 0:
            raise OSError(
                (result.stderr or result.stdout).strip() or f"cannot write {target}"
            )

    def write_main_file(self, raw_path: str, content: str) -> None:
        """Atomically write one text file as the main-service agent."""
        self._write_main_bytes(raw_path, content.encode())

    def edit_main_file(
        self,
        *,
        command: str,
        path: str,
        file_text: str = "",
        old_str: str = "",
        new_str: str = "",
        insert_line: int = 0,
        insert_text: str = "",
    ) -> str:
        """Implement the rubric editor contract inside ``main``."""
        target = self._resolve_agent_path(path)
        if command == "view":
            return self.read_main_file(target)
        if command == "create":
            self.write_main_file(target, file_text)
            return f"created {target}"
        if command == "str_replace":
            if not old_str:
                raise ValueError("old_str and new_str are required")
            text = self.read_main_file(target)
            if old_str not in text:
                raise ValueError("old_str not found")
            self.write_main_file(target, text.replace(old_str, new_str, 1))
            return f"updated {target}"
        if command == "insert":
            text = self.read_main_file(target)
            lines = text.splitlines()
            lines.insert(max(0, int(insert_line)), insert_text)
            self.write_main_file(target, "\n".join(lines) + "\n")
            return f"updated {target}"
        raise ValueError(f"unsupported command: {command}")

    def copy_file_to_main(self, source: Path, destination: str) -> None:
        """Copy a bounded regular host file into ``main`` without a socket mount."""
        if source.is_symlink() or not source.is_file():
            raise OSError(f"copy source is not a regular file: {source}")
        if source.stat().st_size > self.config.editor_max_bytes:
            raise OSError(f"copy source exceeds {self.config.editor_max_bytes} bytes")
        self._write_main_bytes(destination, source.read_bytes())

    def copy_file_from_main(self, source: str, destination: Path) -> None:
        """Atomically copy a bounded regular file out of ``main``."""
        target = self._resolve_agent_path(source)
        result = self._exec_service_argv(
            self.config.main_service,
            (
                "/bin/sh",
                "-c",
                'test -f "$1" && test ! -L "$1" && base64 < "$1"',
                "lbx-copy",
                target,
            ),
            user=self.config.agent_user,
            workdir=self.config.agent_workdir,
            timeout_s=min(60.0, self.config.command_timeout_s),
        )
        if result.timed_out:
            raise TimeoutError(f"copying {target} timed out")
        if result.returncode != 0:
            raise OSError(
                (result.stderr or result.stdout).strip() or f"cannot copy {target}"
            )
        if result.stdout_truncated:
            raise OSError(
                f"{target} exceeds the {self.config.max_output_bytes}-byte copy limit"
            )
        try:
            content = base64.b64decode(result.stdout, validate=False)
        except (ValueError, TypeError) as exc:
            raise OSError(f"main service returned invalid base64 for {target}") from exc
        if len(content) > self.config.editor_max_bytes:
            raise OSError(
                f"{target} exceeds the {self.config.editor_max_bytes}-byte copy limit"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def _run_capture_hook(self, hook: CaptureHook) -> None:
        if hook.atomic_destination is not None:
            cleanup = self._exec_service_argv(
                hook.service,
                (
                    "/bin/sh",
                    "-c",
                    'rm -f -- "$1" "$1.tmp" "$1.lbx-tmp"',
                    "lbx-capture",
                    hook.atomic_destination,
                ),
                user=hook.user,
                timeout_s=min(30.0, hook.timeout_s),
            )
            if cleanup.returncode != 0 or cleanup.timed_out:
                raise CaptureInfrastructureError(
                    f"could not clear atomic capture path "
                    f"{hook.atomic_destination!r} in {hook.service!r}"
                )
        result = self._exec_service_shell(
            hook.service,
            hook.command,
            user=hook.user,
            timeout_s=hook.timeout_s,
        )
        if result.timed_out:
            message = (
                f"capture hook in service {hook.service!r} timed out after "
                f"{hook.timeout_s:g}s"
            )
        elif result.returncode not in hook.accepted_exit_codes:
            detail = (result.stderr or result.stdout).strip()
            message = (
                f"capture hook in service {hook.service!r} exited with "
                f"status {result.returncode}" + (f": {detail}" if detail else "")
            )
        else:
            message = ""
        if message:
            if hook.failure_policy == "agent":
                raise ServiceRuntimeAgentError(message)
            raise CaptureInfrastructureError(message)
        if hook.atomic_destination is not None:
            check = self._exec_service_argv(
                hook.service,
                (
                    "/bin/sh",
                    "-c",
                    'test -e "$1" && test ! -e "$1.tmp" && test ! -e "$1.lbx-tmp"',
                    "lbx-capture",
                    hook.atomic_destination,
                ),
                user=hook.user,
                timeout_s=min(30.0, hook.timeout_s),
            )
            if check.returncode != 0 or check.timed_out:
                raise CaptureInfrastructureError(
                    f"capture hook in service {hook.service!r} did not publish "
                    f"atomic destination {hook.atomic_destination!r}"
                )

    @staticmethod
    def _remove_excluded(root: Path, patterns: tuple[str, ...]) -> None:
        if not patterns or not root.is_dir():
            return
        candidates = sorted(
            root.rglob("*"), key=lambda path: len(path.parts), reverse=True
        )
        for candidate in candidates:
            relative = candidate.relative_to(root).as_posix()
            if not any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
                continue
            if candidate.is_symlink() or candidate.is_file():
                candidate.unlink()
            elif candidate.is_dir():
                shutil.rmtree(candidate)

    @staticmethod
    def _measure_and_digest(path: Path, *, max_depth: int) -> tuple[int, int, str]:
        digest = hashlib.sha256()
        total_bytes = 0
        files = 0
        paths = [path] if not path.is_dir() else sorted(path.rglob("*"))
        for candidate in paths:
            if candidate.is_symlink():
                raise ServiceSecurityError(
                    f"captured artifact contains a symlink: {candidate}"
                )
            relative = (
                candidate.name
                if candidate == path
                else candidate.relative_to(path).as_posix()
            )
            depth = 0 if candidate == path else len(candidate.relative_to(path).parts)
            if depth > max_depth:
                raise CaptureInfrastructureError(
                    f"captured artifact exceeds depth {max_depth}: {candidate}"
                )
            if candidate.is_dir():
                digest.update(f"D\0{relative}\0".encode())
                continue
            if not candidate.is_file():
                raise ServiceSecurityError(
                    f"captured artifact is not a regular file: {candidate}"
                )
            files += 1
            size = candidate.stat().st_size
            total_bytes += size
            digest.update(f"F\0{relative}\0{size}\0".encode())
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return total_bytes, files, digest.hexdigest()

    @staticmethod
    def _extract_artifact_archive(
        archive_path: Path,
        destination: Path,
        artifact: ServiceArtifact,
    ) -> None:
        source_name = PurePosixPath(artifact.source).name
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = archive.getmembers()
            if not members:
                raise CaptureInfrastructureError(
                    f"artifact {artifact.source!r} produced an empty archive"
                )
            validated: list[tuple[tarfile.TarInfo, Path]] = []
            seen_targets: set[Path] = set()
            total_bytes = 0
            files = 0
            for member in members:
                raw_parts = tuple(
                    part
                    for part in PurePosixPath(member.name).parts
                    if part not in {"", "."}
                )
                if (
                    PurePosixPath(member.name).is_absolute()
                    or ".." in raw_parts
                    or not raw_parts
                ):
                    raise ServiceSecurityError(
                        f"artifact archive contains unsafe path {member.name!r}"
                    )
                relative_parts = (
                    raw_parts[1:] if raw_parts[0] == source_name else raw_parts
                )
                if len(relative_parts) > artifact.max_depth:
                    raise CaptureInfrastructureError(
                        f"artifact {artifact.source!r} exceeds depth "
                        f"{artifact.max_depth}"
                    )
                target = destination.joinpath(*relative_parts)
                if target in seen_targets:
                    raise ServiceSecurityError(
                        f"artifact archive repeats path {member.name!r}"
                    )
                seen_targets.add(target)
                if member.isdir():
                    pass
                elif member.isreg():
                    files += 1
                    total_bytes += member.size
                else:
                    raise ServiceSecurityError(
                        f"artifact archive contains links or special files: "
                        f"{member.name!r}"
                    )
                if files > min(
                    artifact.max_files,
                    _MAX_ARTIFACT_STREAM_FILES,
                ):
                    raise CaptureInfrastructureError(
                        f"artifact {artifact.source!r} exceeds "
                        f"{artifact.max_files} files"
                    )
                if total_bytes > artifact.max_bytes:
                    raise CaptureInfrastructureError(
                        f"artifact {artifact.source!r} exceeds "
                        f"{artifact.max_bytes} bytes"
                    )
                validated.append((member, target))

            for member, target in validated:
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise CaptureInfrastructureError(
                        f"artifact archive file is unreadable: {member.name!r}"
                    )
                with extracted, target.open("xb") as output:
                    remaining = member.size
                    while remaining:
                        chunk = extracted.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise CaptureInfrastructureError(
                                f"artifact archive file was truncated: "
                                f"{member.name!r}"
                            )
                        output.write(chunk)
                        remaining -= len(chunk)
                target.chmod(member.mode & 0o777)

    def _collect_artifact(
        self, artifact: ServiceArtifact, staging: Path
    ) -> ArtifactRecord | None:
        container_id = self._service_container_id(artifact.service)
        if container_id is None:
            raise CaptureInfrastructureError(
                f"service {artifact.service!r} disappeared before artifact capture"
            )
        destination = staging / PurePosixPath(artifact.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        archive_path = staging / f".artifact-{uuid.uuid4().hex}.tar"
        stream_limit = min(
            _MAX_ARTIFACT_STREAM_BYTES,
            artifact.max_bytes
            + min(artifact.max_files, _MAX_ARTIFACT_STREAM_FILES) * 2048
            + 8 * 1024 * 1024,
        )
        result = self._stream_to_file(
            self._docker_args(
                "cp",
                f"{container_id}:{artifact.source}",
                "-",
            ),
            output_path=archive_path,
            timeout_s=self.config.startup_timeout_s,
            max_bytes=stream_limit,
        )
        if (
            result.returncode != 0
            or result.timed_out
            or result.stdout_truncated
            or not archive_path.is_file()
        ):
            try:
                archive_path.unlink()
            except OSError:
                pass
            if not artifact.required:
                return None
            detail = (result.stderr or result.stdout).strip()
            message = (
                f"required artifact {artifact.source!r} could not be streamed "
                f"from service {artifact.service!r}" + (f": {detail}" if detail else "")
            )
            if result.stdout_truncated:
                message += f": archive exceeded {stream_limit} bytes"
            if artifact.service == self.config.main_service:
                raise ServiceRuntimeAgentError(message)
            raise CaptureInfrastructureError(message)
        try:
            self._extract_artifact_archive(
                archive_path,
                destination,
                artifact,
            )
        finally:
            try:
                archive_path.unlink()
            except OSError:
                pass
        self._remove_excluded(destination, artifact.exclude)
        if artifact.kind in {"file", "binary"} and not destination.is_file():
            raise CaptureInfrastructureError(
                f"{artifact.kind} artifact {artifact.source!r} is not a file"
            )
        if artifact.kind == "tree" and not destination.is_dir():
            raise CaptureInfrastructureError(
                f"tree artifact {artifact.source!r} is not a directory"
            )
        total_bytes, files, digest = self._measure_and_digest(
            destination, max_depth=artifact.max_depth
        )
        if total_bytes > artifact.max_bytes:
            raise CaptureInfrastructureError(
                f"artifact {artifact.source!r} exceeds {artifact.max_bytes} bytes"
            )
        if files > artifact.max_files:
            raise CaptureInfrastructureError(
                f"artifact {artifact.source!r} exceeds {artifact.max_files} files"
            )
        preserved_mode = (
            stat.S_IMODE(destination.stat().st_mode)
            if artifact.kind == "binary" and artifact.preserve_mode
            else None
        )
        if preserved_mode is not None and preserved_mode & 0o7000:
            raise ServiceSecurityError(
                f"binary artifact {artifact.source!r} has unsafe special mode bits"
            )
        return ArtifactRecord(
            service=artifact.service,
            source=artifact.source,
            destination=artifact.destination,
            host_path=destination,
            sha256=digest,
            bytes=total_bytes,
            files=files,
            mode=preserved_mode,
        )

    @staticmethod
    def _seal_tree(
        root: Path, *, preserved_modes: Mapping[Path, int] | None = None
    ) -> None:
        preserved_modes = preserved_modes or {}
        for candidate in sorted(
            root.rglob("*"), key=lambda path: len(path.parts), reverse=True
        ):
            if os.geteuid() == 0:
                os.chown(candidate, 0, 0, follow_symlinks=False)
            if candidate.is_dir():
                candidate.chmod(0o700)
            elif candidate.is_file():
                candidate.chmod(preserved_modes.get(candidate, 0o600))
        if os.geteuid() == 0:
            os.chown(root, 0, 0)
        root.chmod(0o700)

    def _stop_agent_stack(self, *, best_effort: bool = False) -> None:
        if self._agent_stack_stopped or not self._compose_validated:
            return
        result = self._run(
            self._compose_args("stop", "--timeout", "10", *self._agent_service_names()),
            timeout_s=30.0,
        )
        self._agent_stack_stopped = True
        if not best_effort and (result.returncode != 0 or result.timed_out):
            raise CaptureInfrastructureError(
                "could not stop nested agent service stack"
            )

    def _pause_services(self, services: Sequence[str], *, description: str) -> None:
        names = tuple(dict.fromkeys(services))
        if not names:
            return
        result = self._run(
            self._compose_args("pause", *names),
            timeout_s=30.0,
        )
        if result.returncode != 0 or result.timed_out:
            raise CaptureInfrastructureError(f"could not {description}")

    def _capture_snapshot(self) -> ArtifactSnapshot:
        if self._snapshot is not None:
            return self._snapshot
        self._prepare_sealed()
        staging = Path(
            tempfile.mkdtemp(prefix=".snapshot-", dir=self.config.sealed_dir)
        )
        records: list[ArtifactRecord] = []
        try:
            for hook in self.config.captures:
                self._run_capture_hook(hook)
            agent_reachable_services = [
                service.name
                for service in self.config.services
                if service.role in {"main", "sidecar"}
            ]
            self._pause_services(
                agent_reachable_services,
                description=(
                    "freeze the entire agent-reachable service graph before "
                    "artifact capture"
                ),
            )
            for artifact in self.config.artifacts:
                record = self._collect_artifact(artifact, staging)
                if record is not None:
                    records.append(record)
            self._stop_agent_stack()

            manifest = {
                "schema_version": "lbx-service-artifacts.v1",
                "artifacts": [
                    {
                        "service": record.service,
                        "source": record.source,
                        "destination": record.destination,
                        "sha256": record.sha256,
                        "bytes": record.bytes,
                        "files": record.files,
                        "mode": record.mode,
                    }
                    for record in records
                ],
            }
            manifest_tmp = staging / ".manifest.json.tmp"
            manifest_path = staging / "manifest.json"
            with manifest_tmp.open("w") as handle:
                json.dump(manifest, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(manifest_tmp, manifest_path)
            self._seal_tree(
                staging,
                preserved_modes={
                    record.host_path: record.mode
                    for record in records
                    if record.mode is not None
                },
            )
            published = self.config.sealed_dir / f"snapshot-{uuid.uuid4().hex}"
            os.replace(staging, published)
            published_records = tuple(
                ArtifactRecord(
                    service=record.service,
                    source=record.source,
                    destination=record.destination,
                    host_path=published / record.destination,
                    sha256=record.sha256,
                    bytes=record.bytes,
                    files=record.files,
                    mode=record.mode,
                )
                for record in records
            )
            self._snapshot = ArtifactSnapshot(
                root=published,
                records=published_records,
                manifest_path=published / "manifest.json",
            )
            return self._snapshot
        except Exception:
            self._stop_agent_stack(best_effort=True)
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _wait_verifier(self, service: str) -> tuple[str, int]:
        deadline = self._monotonic() + self.config.verifier_timeout_s
        last_state: dict[str, Any] | None = None
        while self._monotonic() < deadline:
            container_id = self._service_container_id(service)
            state = self._service_state(service)
            if container_id is not None and state is not None:
                last_state = state
                status = str(state.get("Status") or "").lower()
                if status == "exited":
                    return container_id, int(state.get("ExitCode", -1))
                if status == "dead":
                    raise VerifierInfrastructureError(
                        f"verifier service {service!r} entered dead state"
                    )
            self._sleep(0.25)
        raise VerifierInfrastructureError(
            f"verifier service {service!r} timed out after "
            f"{self.config.verifier_timeout_s:g}s; last state={last_state!r}"
        )

    def _verifier_compose_args(
        self, override_path: Path, *args: str
    ) -> tuple[str, ...]:
        if self.bundle is None:
            raise VerifierInfrastructureError("task capsule has not been loaded")
        return (
            "docker",
            "compose",
            "--ansi",
            "never",
            "-f",
            str(self.bundle.compose_path),
            "-f",
            str(override_path),
            "--project-name",
            self.config.project_name,
            *args,
        )

    def _verifier_image_config(self, service: str) -> dict[str, Any]:
        if self.bundle is None:
            raise VerifierInfrastructureError("task capsule has not been loaded")
        image = self.bundle.image_for_service(service)
        inspected = self._checked(
            self._docker_args(
                "image",
                "inspect",
                "--format",
                "{{json .Config}}",
                image.image_ref,
            ),
            timeout_s=30.0,
            error_type=VerifierInfrastructureError,
            description=f"inspecting verifier image config for {service!r}",
        )
        try:
            config = json.loads(inspected.stdout)
        except json.JSONDecodeError as exc:
            raise VerifierInfrastructureError(
                "nested Docker returned invalid verifier image config"
            ) from exc
        if not isinstance(config, dict):
            raise VerifierInfrastructureError(
                "nested Docker returned malformed verifier image config"
            )
        return config

    @staticmethod
    def _container_command(
        value: Any,
        fallback: Sequence[str],
        *,
        label: str,
    ) -> tuple[str, ...]:
        if value is None:
            return tuple(fallback)
        if isinstance(value, str) and value:
            return ("/bin/sh", "-c", value)
        if isinstance(value, list) and all(
            isinstance(item, str) and item for item in value
        ):
            return tuple(value)
        raise VerifierInfrastructureError(f"verifier {label} is malformed")

    def _write_verifier_override(
        self, service: str, snapshot: ArtifactSnapshot
    ) -> tuple[Path, dict[str, Path], str]:
        image_config = self._verifier_image_config(service)
        compose_row = self._compose_services.get(service, {})
        expected_workdir = str(image_config.get("WorkingDir") or "")
        compose_workdir = compose_row.get("working_dir")
        if compose_workdir not in (None, "", expected_workdir):
            raise VerifierInfrastructureError(
                "verifier Compose service overrides the image WORKDIR"
            )
        image_entrypoint = self._container_command(
            image_config.get("Entrypoint"),
            (),
            label="image entrypoint",
        )
        image_command = self._container_command(
            image_config.get("Cmd"),
            (),
            label="image command",
        )
        entrypoint = self._container_command(
            compose_row.get("entrypoint"),
            image_entrypoint,
            label="entrypoint",
        )
        command = self._container_command(
            compose_row.get("command"),
            image_command,
            label="command",
        )
        original_argv = (*entrypoint, *command)
        if not original_argv:
            raise VerifierInfrastructureError(
                "verifier image has no executable entrypoint or command"
            )

        result_root = self.config.state_dir / "verifier-output"
        _mkdir_private(result_root)
        result_files: dict[str, Path] = {}
        result_mounts: list[dict[str, Any]] = []
        for index, result_path in enumerate(self.config.verifier_result_paths):
            host_path = result_root / f"{index:02d}-result"
            descriptor = os.open(
                host_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.close(descriptor)
            result_files[result_path] = host_path
            result_mounts.append(
                {
                    "type": "bind",
                    "source": str(host_path),
                    "target": result_path,
                    "read_only": False,
                }
            )

        wrapper_path = self.config.state_dir / "verifier-wrapper.sh"
        clear_lines: list[str] = []
        for path in self.config.verifier_result_paths:
            quoted = shlex.quote(path)
            clear_lines.extend(
                (
                    f"test -f {quoted}",
                    f"test ! -L {quoted}",
                    f": > {quoted}",
                )
            )
        clear_commands = "\n".join(clear_lines)
        wrapper = (
            "#!/bin/sh\n"
            "set -eu\n"
            f"ulimit -f {(self.config.max_output_bytes + 511) // 512}\n"
            f"{clear_commands}\n"
            'exec "$@"\n'
        ).encode()
        descriptor = os.open(
            wrapper_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o700,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(wrapper)
            handle.flush()
            os.fsync(handle.fileno())

        override_path = self.config.state_dir / "verifier-override.json"
        payload = {
            "services": {
                service: {
                    "cap_drop": ["ALL"],
                    "command": list(original_argv),
                    "depends_on": {},
                    "entrypoint": [
                        "/bin/sh",
                        "/lbx/runtime/verifier-wrapper.sh",
                    ],
                    "environment": {
                        "LBX_SERVICE_ARTIFACT_MANIFEST": (
                            "/lbx/service-artifacts/manifest.json"
                        ),
                        "LBX_SERVICE_ARTIFACT_SNAPSHOT": "/lbx/service-artifacts",
                    },
                    "network_mode": "none",
                    "security_opt": ["no-new-privileges:true"],
                    "volumes": [
                        {
                            "type": "bind",
                            "source": str(snapshot.root),
                            "target": "/lbx/service-artifacts",
                            "read_only": True,
                        },
                        {
                            "type": "bind",
                            "source": str(wrapper_path),
                            "target": "/lbx/runtime/verifier-wrapper.sh",
                            "read_only": True,
                        },
                        *result_mounts,
                    ],
                }
            }
        }
        content = (json.dumps(payload, sort_keys=True) + "\n").encode()
        descriptor = os.open(
            override_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                override_path.unlink()
            except OSError:
                pass
            raise
        return override_path, result_files, expected_workdir

    def _collect_verifier_results(
        self, result_files: Mapping[str, Path]
    ) -> tuple[Path, dict[str, Path]]:
        self._prepare_sealed()
        staging = Path(
            tempfile.mkdtemp(prefix=".verifier-results-", dir=self.config.sealed_dir)
        )
        collected: dict[str, Path] = {}
        try:
            for index, source in enumerate(self.config.verifier_result_paths):
                basename = PurePosixPath(source).name or f"result-{index}"
                destination = staging / f"{index:02d}-{basename}"
                host_path = result_files[source]
                if host_path.is_symlink() or not host_path.is_file():
                    raise VerifierInfrastructureError(
                        f"verifier result is not a regular file: {source}"
                    )
                if host_path.stat().st_size > self.config.max_output_bytes:
                    raise VerifierInfrastructureError(
                        f"verifier result exceeds "
                        f"{self.config.max_output_bytes} bytes: {source}"
                    )
                written = 0
                with (
                    host_path.open("rb") as input_file,
                    destination.open("xb") as output_file,
                ):
                    while chunk := input_file.read(64 * 1024):
                        written += len(chunk)
                        if written > self.config.max_output_bytes:
                            raise VerifierInfrastructureError(
                                f"verifier result exceeds "
                                f"{self.config.max_output_bytes} bytes: {source}"
                            )
                        output_file.write(chunk)
                if written:
                    collected[source] = destination
                elif source == self.config.verifier_reward_path:
                    raise VerifierInfrastructureError(
                        "verifier did not write its canonical reward file"
                    )
            if self.config.verifier_reward_path not in collected:
                raise VerifierInfrastructureError(
                    "verifier did not produce its declared canonical reward file "
                    f"{self.config.verifier_reward_path!r}"
                )
            self._seal_tree(staging)
            published = self.config.sealed_dir / f"verifier-results-{uuid.uuid4().hex}"
            os.replace(staging, published)
            return published, {
                source: published / path.name for source, path in collected.items()
            }
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _canonical_payload(self, results: Mapping[str, Path]) -> dict[str, Any]:
        reward_key = self.config.primary_reward
        if not isinstance(reward_key, str) or not reward_key:
            raise VerifierInfrastructureError(
                "verifier result requires an explicitly configured reward_key"
            )
        source = self.config.verifier_reward_path
        path = results.get(source)
        if path is None:
            raise VerifierInfrastructureError(
                f"verifier result is missing canonical reward path {source!r}"
            )
        if PurePosixPath(source).suffix.lower() != ".json":
            raise VerifierInfrastructureError(
                "canonical verifier reward must be a JSON object"
            )

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON constant {value}")

        try:
            payload = json.loads(
                path.read_text(),
                parse_constant=reject_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise VerifierInfrastructureError(
                f"invalid verifier JSON result {source!r}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise VerifierInfrastructureError(
                f"verifier JSON result {source!r} must be an object"
            )

        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, float) and not math.isfinite(value):
                raise VerifierInfrastructureError(
                    f"verifier JSON result {source!r} contains a non-finite number"
                )
            if isinstance(value, Mapping):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)

        score_value = payload.get(reward_key)
        if not _is_number(score_value):
            raise VerifierInfrastructureError(
                f"verifier reward_key {reward_key!r} is missing or not finite"
            )
        score = float(score_value)
        normalized = dict(payload)
        normalized["score"] = score
        configured_subscores = (
            payload.get(self.config.subscores_key)
            if self.config.subscores_key
            else None
        )
        if isinstance(configured_subscores, Mapping) and all(
            isinstance(key, str) and _is_number(value)
            for key, value in configured_subscores.items()
        ):
            normalized["subscores"] = {
                key: float(value) for key, value in configured_subscores.items()
            }
        else:
            normalized["subscores"] = {reward_key: score}
        raw_weights = payload.get("weights")
        if (
            isinstance(raw_weights, Mapping)
            and set(raw_weights) == set(normalized["subscores"])
            and all(_is_number(value) for value in raw_weights.values())
        ):
            normalized["weights"] = {
                str(key): float(value) for key, value in raw_weights.items()
            }
        else:
            count = len(normalized["subscores"]) or 1
            normalized["weights"] = {
                key: 1.0 / count for key in normalized["subscores"]
            }
        metadata = (
            dict(payload["metadata"])
            if isinstance(payload.get("metadata"), Mapping)
            else {}
        )
        metadata["service_runtime_result"] = source
        normalized["metadata"] = metadata
        return normalized

    def _run_verifier(self, snapshot: ArtifactSnapshot) -> VerifierResult:
        service = self.verifier_service
        if service is None:
            raise VerifierInfrastructureError("no verifier service is configured")
        override_path, result_files, expected_workdir = self._write_verifier_override(
            service, snapshot
        )
        self._checked(
            self._verifier_compose_args(
                override_path,
                "create",
                "--no-deps",
                "--no-build",
                "--pull",
                "never",
                service,
            ),
            timeout_s=self.config.startup_timeout_s,
            error_type=VerifierInfrastructureError,
            description=f"creating verifier service {service!r}",
        )
        container_id = self._service_container_id(service)
        if container_id is None:
            raise VerifierInfrastructureError(
                f"verifier service {service!r} has no container after create"
            )
        workdir_result = self._checked(
            self._docker_args(
                "inspect",
                "--format",
                "{{json .Config.WorkingDir}}",
                container_id,
            ),
            timeout_s=10.0,
            error_type=VerifierInfrastructureError,
            description="verifying preserved verifier image WORKDIR",
        )
        try:
            actual_workdir = json.loads(workdir_result.stdout)
        except json.JSONDecodeError as exc:
            raise VerifierInfrastructureError(
                "nested Docker returned invalid verifier WORKDIR metadata"
            ) from exc
        if actual_workdir != expected_workdir:
            raise VerifierInfrastructureError(
                "verifier launch did not preserve the image WORKDIR"
            )
        self._checked(
            self._verifier_compose_args(override_path, "start", service),
            timeout_s=self.config.startup_timeout_s,
            error_type=VerifierInfrastructureError,
            description=f"starting verifier service {service!r}",
        )
        container_id, exit_code = self._wait_verifier(service)
        if exit_code != 0:
            raise VerifierInfrastructureError(
                f"verifier service {service!r} exited with status {exit_code}"
            )
        result_dir, result_paths = self._collect_verifier_results(result_files)
        payload = self._canonical_payload(
            {
                self.config.verifier_reward_path: result_paths[
                    self.config.verifier_reward_path
                ]
            }
        )
        metadata = dict(payload.get("metadata") or {})
        metadata["verifier_exit_code"] = exit_code
        metadata["service_artifact_manifest"] = str(snapshot.manifest_path)
        payload["metadata"] = metadata
        return VerifierResult(
            payload=payload, result_dir=result_dir, exit_code=exit_code
        )

    def finalize_and_verify(self) -> VerifierResult | None:
        """Capture once, stop agent services, and run the separate verifier."""
        with self._lock:
            if self._verifier_result is not None:
                return self._verifier_result
            if self._finalized_without_verifier:
                return None
            if not self._started:
                raise ServiceRuntimeError("service runtime was not started")
            snapshot = self._capture_snapshot()
            if self.verifier_service is None:
                self._finalized_without_verifier = True
                return None
            try:
                self._verifier_result = self._run_verifier(snapshot)
                return self._verifier_result
            except (ServiceRuntimeAgentError, ServiceRuntimeError):
                raise
            except Exception as exc:
                raise VerifierInfrastructureError(
                    f"unexpected verifier failure: {type(exc).__name__}: {exc}"
                ) from exc

    def grader_handoff(self) -> GraderWorkspaceHandoff:
        """Return the sealed workspace only after the agent stack is stopped."""
        with self._lock:
            if self._snapshot is None or not self._agent_stack_stopped:
                raise ServiceRuntimeError(
                    "sealed grader workspace is unavailable before finalization"
                )
            if (
                self._snapshot.root.is_symlink()
                or not self._snapshot.root.is_dir()
                or self._snapshot.manifest_path.is_symlink()
                or not self._snapshot.manifest_path.is_file()
            ):
                raise ServiceSecurityError("sealed grader workspace was replaced")
            return GraderWorkspaceHandoff(
                workspace=self._snapshot.root,
                manifest=self._snapshot.manifest_path,
            )

    def cleanup(self) -> None:
        """Idempotently stop Compose and the nested daemon process group."""
        with self._lock:
            if self._closed:
                return
            if self._compose_validated and self._daemon is not None:
                try:
                    self._run(
                        self._compose_args(
                            "down",
                            "--remove-orphans",
                            "--volumes",
                            "--timeout",
                            "10",
                        ),
                        timeout_s=30.0,
                    )
                except Exception as exc:  # noqa: BLE001 - cleanup boundary
                    self.cleanup_errors.append(
                        f"compose cleanup failed: {type(exc).__name__}: {exc}"
                    )
            if self._daemon is not None:
                try:
                    self.runner.stop_process_group(self._daemon, grace_s=3.0)
                except Exception as exc:  # noqa: BLE001 - cleanup boundary
                    self.cleanup_errors.append(
                        f"dockerd cleanup failed: {type(exc).__name__}: {exc}"
                    )
            self._daemon = None
            self._started = False
            self._closed = True
            if self._state_created:
                try:
                    _assert_contained_path(
                        self.config.state_dir,
                        self.config.operator_roots.state,
                        label="state cleanup directory",
                    )
                    shutil.rmtree(self.config.state_dir)
                except (OSError, ServiceSecurityError) as exc:
                    self.cleanup_errors.append(
                        f"state cleanup failed: {type(exc).__name__}: {exc}"
                    )

    close = cleanup
