"""Content-addressed deploy-time mount substrate.

Large datasets and model weights are mounted read-only at deploy time instead of
baked into per-task images. The flow:

1. The author declares ``[[preloaded_files]]`` in ``task.toml`` (a local
   ``source`` tree or an ``hf_repo``).
2. ``scripts/sync_mount.sh`` packs each source into a content-addressed squashfs
   (``<task>/<name>-<sha256:16>.squashfs``), uploads it once to a shared cache
   prefix (reused across tasks -- no duplication), and stamps the concrete
   ``remote_path`` entries into ``.alignerr/preloaded_files.json``.
3. The exporter reads that manifest and emits ``preloaded_files`` on the Boreal
   problem entry; the platform mounts each at ``local_path`` read-only.

This module holds the pure, testable pieces: content addressing, the manifest
shape, and QA-facing source archive notices. The shell scripts call into the
same naming so producer and consumer agree.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

PRELOADED_MANIFEST_PATH = Path(".alignerr") / "preloaded_files.json"
RESERVED_TRUSTED_MOUNT_PATHS = frozenset({"/mcp_server/calibration"})
QA_VISIBLE_PUBLIC_MAX_FILES = 256
QA_VISIBLE_PUBLIC_MAX_BYTES = 1024 * 1024 * 1024
_PROMPT_FILE_LEFT_BOUNDARY = r"A-Za-z0-9_.-"
_PROMPT_FILE_RIGHT_CONTINUATION = r"(?:[A-Za-z0-9_/-]|\.[A-Za-z0-9_])"


def content_address(path: Path) -> str:
    """Deterministic 16-char content hash of a file or directory tree.

    For a directory, hashes the sorted ``(relpath, bytes)`` of every regular
    file so identical trees produce identical addresses (cross-task dedup).
    """
    digest = hashlib.sha256()
    path = Path(path)
    if path.is_dir():
        files = sorted(
            (p for p in path.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(path).as_posix(),
        )
        for file in files:
            digest.update(file.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(file.read_bytes())
            digest.update(b"\0")
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def remote_squashfs_name(task_id: str, entry_name: str, address: str) -> str:
    """Content-addressed remote object name (shared cache prefix-relative)."""
    return f"{task_id}/{entry_name}-{address}.squashfs"


def _prompt_mentions_file(prompt: str, aliases: tuple[str, ...]) -> bool:
    """Match a file alias as a path token, never inside a longer filename."""
    return any(
        re.search(
            rf"(?<![{_PROMPT_FILE_LEFT_BOUNDARY}])"
            rf"{re.escape(alias)}"
            rf"(?!{_PROMPT_FILE_RIGHT_CONTINUATION})",
            prompt,
            flags=re.IGNORECASE,
        )
        is not None
        for alias in aliases
    )


def qa_visible_public_file_selection(
    source: Path,
    mount_path: str,
    *,
    prompt_text: str = "",
) -> tuple[list[tuple[str, str]] | None, bool]:
    """Select explicit Taiga payload files and report if they cover the tree.

    Taiga Data Quality can inspect regular ``preloaded_files`` entries but sees a
    squashfs mount as one opaque payload file. Small regular trees are expanded
    in full. For a large or mixed tree, regular files named verbatim in the
    prompt are still expanded so QA can resolve the task's public-file contract.
    The boolean is true only when the selected files replace the complete tree;
    callers must retain the canonical baked/squashfs tree for partial selections.
    """
    source = Path(source)
    if not source.is_dir():
        return None, False
    root = PurePosixPath(mount_path)
    files: list[tuple[str, str, int]] = []
    total_bytes = 0
    has_nonregular = False
    for path in sorted(source.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode) or path.name == ".gitkeep":
            continue
        if not stat.S_ISREG(info.st_mode):
            has_nonregular = True
            continue
        relative = path.relative_to(source).as_posix()
        total_bytes += int(info.st_size)
        files.append((relative, str(root / relative), int(info.st_size)))

    if not files:
        return None, False
    if (
        not has_nonregular
        and len(files) <= QA_VISIBLE_PUBLIC_MAX_FILES
        and total_bytes <= QA_VISIBLE_PUBLIC_MAX_BYTES
    ):
        return (
            [(relative, local_path) for relative, local_path, _size in files],
            True,
        )

    referenced = [
        (relative, local_path, size)
        for relative, local_path, size in files
        if prompt_text
        and _prompt_mentions_file(
            prompt_text,
            (relative, PurePosixPath(relative).name, local_path),
        )
    ]
    if (
        not referenced
        or len(referenced) > QA_VISIBLE_PUBLIC_MAX_FILES
        or sum(size for _relative, _local_path, size in referenced)
        > QA_VISIBLE_PUBLIC_MAX_BYTES
    ):
        return None, False
    return (
        [(relative, local_path) for relative, local_path, _size in referenced],
        False,
    )


def qa_visible_public_files(
    source: Path,
    mount_path: str,
    *,
    prompt_text: str = "",
) -> list[tuple[str, str]] | None:
    """Return the bounded QA-visible file list without completeness metadata."""
    files, _complete = qa_visible_public_file_selection(
        source,
        mount_path,
        prompt_text=prompt_text,
    )
    return files


# Conventional dataset dirs auto-mounted (read-only) by trusted CI for known
# task types, so the raw dataset is mounted at deploy time instead of baked into
# the per-task image. Each entry is (source-relative-to-problem-dir, mount_path).
# Mirrors the ML_Envs convention (data/public -> /data, data/private ->
# /mcp_server/data) using this template's layout.
AUTO_MOUNT_SPECS: dict[str, list[tuple[str, str]]] = {
    "ml": [
        ("data", "/data"),
        ("scorer/data", "/mcp_server/data"),
    ],
}


def _tree_has_content(path: Path) -> bool:
    """True when a directory holds at least one real file (ignoring ``.gitkeep``)."""
    path = Path(path)
    if not path.is_dir():
        return False
    return any(p.is_file() and p.name != ".gitkeep" for p in path.rglob("*"))


def auto_mount_entries(
    problem_dir: Path, task_type: str, *, hidden_env: str = ""
) -> list[tuple[str, str]]:
    """Implicit conventional mounts for a task type (e.g. ``ml``).

    Returns ``[(source_rel, mount_path), ...]`` for each conventional dataset dir
    that actually has content, so trusted CI can pack/upload/mount it read-only
    instead of baking it into the image. Empty/missing trees are skipped (they
    keep the baked-empty fallback). Unknown task types return ``[]``.

    Hidden-env tasks keep their env source **baked** rather than mounted:

    * ``env``/``hybrid`` never mount ``scorer/data`` (``/mcp_server/data``), so
      the hidden ``env.py`` is baked into the image (the root supervisor loads it
      from disk, not a deploy mount).
    * a pure ``env`` task additionally never mounts ``data`` (``/data``), so the
      agent-facing ``env_client.py`` stays baked too -- a pure-env task mounts
      nothing. ``hybrid`` may still mount ``/data`` for genuinely large static
      public data.
    """
    problem_dir = Path(problem_dir)
    specs = AUTO_MOUNT_SPECS.get((task_type or "").strip().lower(), [])
    hidden_env_norm = (hidden_env or "").strip().lower()
    hidden_env_active = hidden_env_norm in {"env", "hybrid"}
    env_only = hidden_env_norm == "env"
    return [
        (source_rel, mount_path)
        for source_rel, mount_path in specs
        if not (hidden_env_active and mount_path == "/mcp_server/data")
        if not (env_only and mount_path == "/data")
        if _tree_has_content(problem_dir / source_rel)
    ]


def manifest_entry(
    remote_path: str, local_path: str, *, read_only: bool = True
) -> dict[str, Any]:
    """One Boreal/Taiga ``preloaded_files`` entry."""
    return {
        "remote_path": remote_path,
        "local_path": local_path,
        "is_read_only": bool(read_only),
    }


def merge_manifest_entries(
    existing: list[dict[str, Any]],
    additions: list[dict[str, Any]],
    *,
    trusted: bool = False,
) -> list[dict[str, Any]]:
    """Merge mounts by local path, protecting framework-owned destinations."""

    merged = {
        str(entry.get("local_path")): dict(entry)
        for entry in existing
        if isinstance(entry, dict) and entry.get("local_path")
    }
    for entry in additions:
        local_path = str(entry.get("local_path") or "")
        if not local_path:
            raise ValueError("preloaded entry local_path must be non-empty")
        if local_path in RESERVED_TRUSTED_MOUNT_PATHS and not trusted:
            raise ValueError(
                f"preloaded mount path {local_path!r} is reserved for trusted CI"
            )
        if local_path in RESERVED_TRUSTED_MOUNT_PATHS and not bool(
            entry.get("is_read_only", True)
        ):
            raise ValueError(f"trusted mount {local_path!r} must be read-only")
        merged[local_path] = dict(entry)
    return [merged[path] for path in sorted(merged)]


def load_preloaded_manifest(problem_dir: Path) -> list[dict[str, Any]]:
    """Return the stamped manifest (``[]`` when none has been produced)."""
    manifest_path = Path(problem_dir) / PRELOADED_MANIFEST_PATH
    if not manifest_path.exists():
        return []
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        data = data.get("preloaded_files", [])
    return [entry for entry in data if isinstance(entry, dict)]


def render_preloaded_notice(manifest: list[dict[str, Any]]) -> str:
    """QA-only NOTICE.md text for source archives that omit mounted payloads."""
    mounts = [
        str(entry.get("local_path")) for entry in manifest if entry.get("local_path")
    ]
    if not mounts:
        return ""
    lines = [
        "# NOTICE",
        "",
        "## Deploy-Time Mounts",
        "",
        "Some task data is provided through Taiga `preloaded_files` read-only "
        "mounts instead of being baked into the Docker image or registered "
        "Docker source archive.",
        "",
        "Trusted CI slims those source directories before image build and source "
        "registration, so their contents may be absent from the Docker source "
        "archive by design.",
        "",
        "Authoritative mounted paths:",
        "",
    ]
    lines.extend(f"- `{mount}`" for mount in sorted(set(mounts)))
    lines.extend(
        [
            "",
            "The full mount metadata remains on the problem entry under "
            "`preloaded_files`.",
        ]
    )
    return "\n".join(lines) + "\n"
