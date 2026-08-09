"""Safe, staged task-input materialization for exported build contexts."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import uuid
import zlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IGNORED_NAMES = frozenset(
    {
        ".DS_Store",
        ".git",
        ".pytest_cache",
        "__pycache__",
    }
)
_OUTPUT_OWNER_MARKER = ".alignerr-export-owned"
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_DOCKERFILE_STAGE_RE = re.compile(
    r"^\s*FROM(?:\s+--\S+)*\s+\S+\s+AS\s+([A-Za-z0-9_.-]+)\s*(?:#.*)?$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_MAX_GIT_METADATA_BYTES = 256 * 1024 * 1024
_MAX_GIT_OBJECT_BYTES = 64 * 1024 * 1024
_MAX_GIT_METADATA_FILES = 100_000
_MAX_REACHABLE_GIT_OBJECTS = 50_000
_UNSAFE_GIT_CONFIG_TOKENS = (
    "[include",
    "alternates",
    "filter.",
    "fsmonitor",
    "hookspath",
    "include.path",
    "sshcommand",
    "worktree",
)


def require_local_dockerfile_target(
    dockerfile_text: str,
    target: str,
    *,
    label: str,
) -> str:
    """Return a target only when the Dockerfile declares it as a local stage."""
    normalized = target.strip()
    stages = {
        match.group(1).casefold(): match.group(1)
        for match in _DOCKERFILE_STAGE_RE.finditer(dockerfile_text)
    }
    declared = stages.get(normalized.casefold())
    if not normalized or declared is None:
        raise ValueError(
            f"{label} target {target!r} is not a declared local Dockerfile stage"
        )
    return declared


@dataclass(frozen=True)
class WorkspaceSpec:
    """Schema-independent workspace seed and writable-root contract."""

    root: str = "/workdir"
    seed: str | None = None
    agent_cwd: str = "/workdir"
    git_baseline: bool | str = False
    init_policy: str = "empty"
    clean_paths: tuple[str, ...] = ()


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            result = dump(mode="python", exclude_none=True)
        except TypeError:
            result = dump()
        return dict(result) if isinstance(result, Mapping) else {}
    return {}


def resolve_workspace(task: Any) -> WorkspaceSpec:
    """Read the optional workspace contract without importing schema classes."""
    task_data = _mapping(task)
    workspace = _mapping(task_data.get("workspace"))
    if not workspace:
        return WorkspaceSpec()
    root = str(workspace.get("root") or "/workdir")
    return WorkspaceSpec(
        root=root,
        seed=str(workspace["seed"]) if workspace.get("seed") else None,
        agent_cwd=str(workspace.get("agent_cwd") or root),
        git_baseline=workspace.get("git_baseline", True),
        init_policy=str(workspace.get("init_policy") or "copy"),
        clean_paths=tuple(str(path) for path in workspace.get("clean_paths", [])),
    )


def _assert_inside(path: Path, root: Path, *, label: str) -> Path:
    root = root.resolve()
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes task root {root}: {path}") from exc
    return resolved


def resolve_task_path(task_root: Path, value: str, *, label: str) -> Path:
    """Resolve a task-relative path and reject host escapes."""
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be task-relative: {value!r}")
    return _assert_inside(task_root / candidate, task_root, label=label)


def validate_no_external_symlinks(source: Path, task_root: Path) -> None:
    """Reject symlinks whose targets leave the task directory."""
    source = _assert_inside(source, task_root, label="materialization source")
    candidates = [source]
    if source.is_dir() and not source.is_symlink():
        for directory, dirnames, filenames in os.walk(source, followlinks=False):
            base = Path(directory)
            candidates.extend(base / name for name in (*dirnames, *filenames))
    for candidate in candidates:
        if not candidate.is_symlink():
            continue
        target = candidate.parent / os.readlink(candidate)
        _assert_inside(target, task_root, label=f"symlink {candidate}")


def copy_path_safe(
    source: Path,
    destination: Path,
    *,
    task_root: Path,
    ignored_names: frozenset[str] = _IGNORED_NAMES,
) -> None:
    """Copy a task path while preserving only task-contained symlinks."""
    validate_no_external_symlinks(source, task_root)
    if source.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(os.readlink(source), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(
            source,
            destination,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                *ignored_names,
                "*.pyc",
            ),
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)


def copy_directory_contents_safe(
    source: Path,
    destination: Path,
    *,
    task_root: Path,
    ignored_names: frozenset[str] = _IGNORED_NAMES,
) -> None:
    """Copy sorted directory contents without following external symlinks."""
    validate_no_external_symlinks(source, task_root)
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir(), key=lambda path: path.name):
        if child.name in ignored_names or child.suffix == ".pyc":
            continue
        copy_path_safe(
            child,
            destination / child.name,
            task_root=task_root,
            ignored_names=ignored_names,
        )


def materialize_task_inputs(task_root: Path, destination: Path) -> None:
    """Copy all authored task inputs needed by agent, solution, and verifier."""
    task_root = task_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(task_root.iterdir(), key=lambda path: path.name):
        if child.name in _IGNORED_NAMES or child.suffix == ".pyc":
            continue
        copy_path_safe(child, destination / child.name, task_root=task_root)


def materialize_workspace_seed(
    task_root: Path,
    destination: Path,
    workspace: WorkspaceSpec,
) -> None:
    """Materialize the immutable workspace seed into a build-context directory."""
    destination.mkdir(parents=True, exist_ok=True)
    if workspace.init_policy == "empty" or workspace.seed is None:
        return
    seed = resolve_task_path(task_root, workspace.seed, label="workspace seed")
    if not seed.exists():
        raise ValueError(f"workspace seed does not exist: {seed}")
    if seed.is_dir():
        copy_directory_contents_safe(
            seed,
            destination,
            task_root=task_root,
        )
        if isinstance(workspace.git_baseline, str):
            if workspace.init_policy not in {"copy", "overlay"}:
                raise ValueError(
                    "commit-pinned git_baseline requires workspace init_policy "
                    "'copy' or 'overlay'"
                )
            _materialize_pinned_git_metadata(
                seed,
                destination,
                workspace.git_baseline,
            )
    else:
        if isinstance(workspace.git_baseline, str):
            raise ValueError(
                "commit-pinned git_baseline requires a directory workspace seed"
            )
        copy_path_safe(seed, destination / seed.name, task_root=task_root)
    _preserve_workspace_seed_in_dockerignore(destination.parent)


def _validate_source_git_directory(git_dir: Path) -> None:
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise ValueError(
            "commit-pinned workspace seed requires an internal .git directory; "
            "gitdir files and symlinks are forbidden"
        )
    total_bytes = 0
    file_count = 0
    for directory, dirnames, filenames in os.walk(git_dir, followlinks=False):
        base = Path(directory)
        for name in (*dirnames, *filenames):
            candidate = base / name
            relative = candidate.relative_to(git_dir)
            if candidate.is_symlink():
                raise ValueError(
                    f"workspace .git metadata contains forbidden symlink: {relative}"
                )
            mode = os.lstat(candidate).st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(
                    "workspace .git metadata contains a non-regular file: "
                    f"{relative}"
                )
            file_count += 1
            size = candidate.stat().st_size
            total_bytes += size
            if (
                file_count > _MAX_GIT_METADATA_FILES
                or total_bytes > _MAX_GIT_METADATA_BYTES
            ):
                raise ValueError("workspace .git metadata exceeds safe size limits")
            if "objects" in relative.parts and size > _MAX_GIT_OBJECT_BYTES:
                raise ValueError(f"workspace git object is oversized: {relative}")
            if relative.name in {"alternates", "commondir"}:
                raise ValueError(
                    f"workspace .git metadata contains forbidden {relative.name!r}"
                )
            if "hooks" in relative.parts and not relative.name.endswith(".sample"):
                raise ValueError(f"workspace .git contains unsafe hook: {relative}")
            if relative.name == "config":
                if size > 1024 * 1024:
                    raise ValueError("workspace git config exceeds safe size limit")
                config = candidate.read_text(errors="replace").lower()
                if any(token in config for token in _UNSAFE_GIT_CONFIG_TOKENS):
                    raise ValueError(
                        f"workspace .git contains unsafe config: {relative}"
                    )


def _trusted_git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return environment


def _trusted_git_command(repository: Path, *arguments: str) -> list[str]:
    return [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-C",
        str(repository),
        *arguments,
    ]


def _trusted_git(
    repository: Path,
    *arguments: str,
    text: bool,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        _trusted_git_command(repository, *arguments),
        check=True,
        capture_output=True,
        text=text,
        timeout=15,
        env=_trusted_git_environment(),
    )


def _reachable_git_objects(seed: Path, commit: str) -> list[str]:
    output = _trusted_git(
        seed,
        "rev-list",
        "--objects",
        "--no-object-names",
        commit,
        text=True,
    ).stdout
    objects = output.splitlines()
    if not objects or len(objects) > _MAX_REACHABLE_GIT_OBJECTS:
        raise ValueError("workspace reachable git object graph exceeds safe limits")
    if any(not _GIT_COMMIT_RE.fullmatch(object_id) for object_id in objects):
        raise ValueError("workspace reachable git object graph contains invalid IDs")
    return objects


def _read_reachable_git_objects(
    seed: Path,
    object_ids: list[str],
) -> list[tuple[str, str, bytes]]:
    completed = subprocess.run(
        _trusted_git_command(seed, "cat-file", "--batch"),
        input=("".join(f"{object_id}\n" for object_id in object_ids)).encode(),
        check=True,
        capture_output=True,
        timeout=60,
        env=_trusted_git_environment(),
    )
    payload = completed.stdout
    position = 0
    total_size = 0
    objects: list[tuple[str, str, bytes]] = []
    for expected_id in object_ids:
        line_end = payload.find(b"\n", position)
        if line_end < 0:
            raise ValueError("workspace git object batch is truncated")
        header = payload[position:line_end].decode("ascii", errors="strict").split()
        if len(header) != 3 or header[0] != expected_id:
            raise ValueError("workspace git object batch has an invalid header")
        object_type = header[1]
        if object_type not in {"blob", "commit", "tag", "tree"}:
            raise ValueError(f"workspace git object has forbidden type {object_type!r}")
        try:
            object_size = int(header[2])
        except ValueError as exc:
            raise ValueError("workspace git object has an invalid size") from exc
        if object_size < 0 or object_size > _MAX_GIT_OBJECT_BYTES:
            raise ValueError(f"workspace git object {expected_id!r} is oversized")
        total_size += object_size
        if total_size > _MAX_GIT_METADATA_BYTES:
            raise ValueError("workspace reachable git objects exceed safe size limits")
        content_start = line_end + 1
        content_end = content_start + object_size
        if (
            content_end >= len(payload)
            or payload[content_end : content_end + 1] != b"\n"
        ):
            raise ValueError("workspace git object batch has invalid framing")
        content = payload[content_start:content_end]
        object_payload = f"{object_type} {object_size}\0".encode() + content
        actual_id = hashlib.sha1(object_payload, usedforsecurity=False).hexdigest()
        if actual_id != expected_id:
            raise ValueError(
                f"workspace git object {expected_id!r} failed identity verification"
            )
        objects.append((expected_id, object_type, object_payload))
        position = content_end + 1
    if payload[position:]:
        raise ValueError("workspace git object batch contains trailing data")
    return objects


def _verify_materialized_git_repository(destination: Path, commit: str) -> None:
    try:
        _trusted_git(destination, "fsck", "--strict", "--no-dangling", text=True)
        _trusted_git(
            destination,
            "cat-file",
            "-e",
            f"{commit}^{{tree}}",
            text=True,
        )
        listing = _trusted_git(
            destination,
            "ls-tree",
            "-r",
            commit,
            text=True,
        ).stdout
        for row in listing.splitlines():
            mode = row.split(maxsplit=1)[0]
            if mode not in {"100644", "100755"}:
                raise ValueError(
                    f"workspace commit contains unsupported tree mode {mode!r}"
                )
        _trusted_git(destination, "read-tree", commit, text=True)
        status = _trusted_git(destination, "status", "--short", text=True).stdout
        if status:
            raise ValueError(
                "workspace seed files do not exactly match the pinned commit: "
                f"{status.strip()}"
            )
        with tempfile.TemporaryDirectory(prefix="alignerr-git-checkout-") as temporary:
            checkout = Path(temporary)
            _trusted_git(
                destination,
                f"--work-tree={checkout}",
                "checkout",
                "--force",
                commit,
                "--",
                ".",
                text=True,
            )
            for candidate in checkout.rglob("*"):
                if candidate.is_symlink():
                    raise ValueError(
                        "workspace pinned commit checkout contains a symlink"
                    )
                if candidate.is_file():
                    with candidate.open("rb") as handle:
                        for _ in iter(lambda: handle.read(1024 * 1024), b""):
                            pass
        _write_deterministic_git_index(destination, commit)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"materialized workspace git repository failed verification: {exc}"
        ) from exc


def _write_deterministic_git_index(destination: Path, commit: str) -> None:
    """Build an index from tree IDs without host inode or timestamp metadata."""
    tree = _trusted_git(
        destination,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        commit,
        text=False,
    ).stdout
    index_path = destination / ".git" / "index"
    temporary_index = destination / ".git" / "index.canonical"
    temporary_index.unlink(missing_ok=True)
    environment = _trusted_git_environment()
    environment["GIT_INDEX_FILE"] = str(temporary_index)
    subprocess.run(
        _trusted_git_command(destination, "update-index", "-z", "--index-info"),
        input=tree,
        check=True,
        capture_output=True,
        timeout=15,
        env=environment,
    )
    os.replace(temporary_index, index_path)


def _materialize_pinned_git_metadata(
    seed: Path,
    destination: Path,
    commit: str,
) -> None:
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError("workspace git_baseline must be a 40-character commit ID")
    source_git = seed / ".git"
    _validate_source_git_directory(source_git)
    try:
        head = _trusted_git(
            seed,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            text=True,
        ).stdout.strip()
        if head != commit:
            raise ValueError(
                f"workspace seed HEAD {head!r} does not match git_baseline {commit!r}"
            )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"cannot verify workspace git_baseline {commit!r}: {exc}"
        ) from exc
    object_ids = _reachable_git_objects(seed, commit)
    objects = _read_reachable_git_objects(seed, object_ids)

    destination_git = destination / ".git"
    for relative in (
        "objects/info",
        "objects/pack",
        "refs/heads",
        "refs/tags",
    ):
        (destination_git / relative).mkdir(parents=True, exist_ok=True)
    for object_id, _, object_payload in objects:
        object_path = destination_git / "objects" / object_id[:2] / object_id[2:]
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_bytes(zlib.compress(object_payload, level=9))
    head_path = destination_git / "HEAD"
    head_path.write_text(f"{commit}\n")
    _verify_materialized_git_repository(destination, commit)
    for candidate in sorted(
        destination_git.rglob("*"),
        key=lambda path: (len(path.parts), path.as_posix()),
        reverse=True,
    ):
        candidate.chmod(0o755 if candidate.is_dir() else 0o644)
        os.utime(candidate, (0, 0), follow_symlinks=False)
    destination_git.chmod(0o755)
    os.utime(destination_git, (0, 0), follow_symlinks=False)


def _preserve_workspace_seed_in_dockerignore(context: Path) -> None:
    dockerignore = context / ".dockerignore"
    rules = (
        "!.alignerr-workspace-seed/\n"
        "!.alignerr-workspace-seed/.git/\n"
        "!.alignerr-workspace-seed/.git/**\n"
    )
    existing = dockerignore.read_text() if dockerignore.is_file() else ""
    if rules not in existing:
        separator = "" if not existing or existing.endswith("\n") else "\n"
        dockerignore.write_text(existing + separator + rules)


def workspace_dockerfile_overlay(
    workspace: WorkspaceSpec,
    *,
    user: str,
    seed_directory: str = ".alignerr-workspace-seed",
    verifier: bool = False,
) -> str:
    """Render deterministic seed, git-baseline, workdir, and user instructions."""
    root = shlex.quote(workspace.root)
    cwd = workspace.agent_cwd
    lines = [
        "",
        "# Alignerr materialized workspace contract.",
        "USER root",
    ]
    if not verifier:
        lines.extend(
            [
                (
                    "RUN if ! id -u agent >/dev/null 2>&1; then "
                    "if command -v useradd >/dev/null 2>&1; then "
                    "groupadd -f -g 1000 agent && useradd -u 1000 -g 1000 -m "
                    "-s /bin/sh agent; "
                    "elif command -v adduser >/dev/null 2>&1; then "
                    "addgroup -g 1000 agent 2>/dev/null || true; "
                    "adduser -D -u 1000 -G agent agent; "
                    "else echo 'cannot create non-root agent user' >&2; "
                    "exit 1; fi; fi"
                ),
            ]
        )
    lines.extend(
        [
            f"RUN mkdir -p {root} {shlex.quote(cwd)}",
            f"COPY {seed_directory}/ {workspace.root.rstrip('/')}/",
        ]
    )
    if workspace.clean_paths:
        paths = " ".join(
            shlex.quote(f"{workspace.root.rstrip('/')}/{path}")
            for path in workspace.clean_paths
        )
        lines.append(f"RUN rm -rf -- {paths}")
    if workspace.git_baseline:
        lines.append(
            "RUN if ! command -v git >/dev/null 2>&1; then "
            "if command -v apt-get >/dev/null 2>&1; then "
            "apt-get update && apt-get install -y --no-install-recommends git "
            "&& rm -rf /var/lib/apt/lists/*; "
            "elif command -v apk >/dev/null 2>&1; then apk add --no-cache git; "
            "elif command -v dnf >/dev/null 2>&1; then "
            "dnf install -y git && dnf clean all; "
            "else echo 'workspace git baseline requires git' >&2; exit 1; fi; fi"
        )
        if isinstance(workspace.git_baseline, str):
            expected = shlex.quote(workspace.git_baseline)
            lines.append(
                f"RUN test -d {root}/.git "
                f'&& test "$(git -C {root} rev-parse --verify '
                f'HEAD^{{commit}})" = {expected}'
            )
        else:
            lines.append(
                f"RUN if [ -d {root} ]; then git -C {root} init -q "
                f"&& git -C {root} add -A "
                f"&& git -C {root} -c user.name=alignerr "
                f"-c user.email=alignerr@local commit -q --allow-empty -m seed "
                f"&& git -C {root} update-ref refs/tags/alignerr-seed HEAD; fi"
            )
    if verifier:
        lines.extend(
            [
                f"RUN chown -R root:root {root} && chmod -R go-w {root}",
                "USER root",
            ]
        )
    else:
        lines.extend(
            [
                f"RUN chown -R {shlex.quote(user)}:{shlex.quote(user)} {root}",
                f"WORKDIR {cwd}",
                f"USER {user}",
            ]
        )
    return "\n".join(lines) + "\n"


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_lexical_symlinks(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"output path contains symlink component: {current}")


@contextmanager
def staged_output_directory(
    target: Path,
    *,
    force: bool,
    source_tree: Path | None = None,
) -> Iterator[Path]:
    """Yield a sibling stage and atomically publish it on successful exit."""
    target = _absolute_lexical(target)
    _reject_lexical_symlinks(target)
    if source_tree is not None:
        source_lexical = _absolute_lexical(source_tree)
        source_resolved = source_lexical.resolve(strict=True)
        try:
            target.relative_to(source_lexical)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"output directory must not be inside task source tree: {target}"
            )
        try:
            target.resolve(strict=False).relative_to(source_resolved)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"resolved output directory must not be inside task source tree: "
                f"{target}"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise FileExistsError(
            f"output directory already exists: {target}; pass force=True to replace it"
        )
    if target.exists() and force:
        target_mode = os.lstat(target).st_mode
        marker = target / _OUTPUT_OWNER_MARKER
        if not stat.S_ISDIR(target_mode) or marker.is_symlink() or not marker.is_file():
            raise ValueError(
                "force may only replace a directory created by this exporter: "
                f"{target}"
            )
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    (stage / _OUTPUT_OWNER_MARKER).write_text("alignerr-export-v1\n")
    backup: Path | None = None
    try:
        yield stage
        if target.exists():
            backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
            os.replace(target, backup)
        try:
            os.replace(stage, target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


__all__ = [
    "WorkspaceSpec",
    "copy_directory_contents_safe",
    "copy_path_safe",
    "materialize_task_inputs",
    "materialize_workspace_seed",
    "require_local_dockerfile_target",
    "resolve_task_path",
    "resolve_workspace",
    "staged_output_directory",
    "validate_no_external_symlinks",
    "workspace_dockerfile_overlay",
]
