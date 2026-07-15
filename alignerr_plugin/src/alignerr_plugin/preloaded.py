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
from pathlib import Path
from typing import Any

PRELOADED_MANIFEST_PATH = Path(".alignerr") / "preloaded_files.json"


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


def auto_mount_entries(problem_dir: Path, task_type: str, *, hidden_env: str = "") -> list[tuple[str, str]]:
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


def manifest_entry(remote_path: str, local_path: str, *, read_only: bool = True) -> dict[str, Any]:
    """One Boreal/Taiga ``preloaded_files`` entry."""
    return {
        "remote_path": remote_path,
        "local_path": local_path,
        "is_read_only": bool(read_only),
    }


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
    mounts = [str(entry.get("local_path")) for entry in manifest if entry.get("local_path")]
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
            "The full mount metadata remains on the problem entry under " "`preloaded_files`.",
        ]
    )
    return "\n".join(lines) + "\n"
