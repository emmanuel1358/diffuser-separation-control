#!/usr/bin/env python3
"""Upload a task's Docker image source code to Taiga (docker-images family).

Called from the grade-fork-pr workflow right after a task's image is built+pushed,
so the source that produced the deployed image is registered against that exact
image digest. This is what the "Docker Image Source Code" sidebar shows, and it
is the precondition for applying status tags (e.g. "Ready for Customer") to the
problem version.

Flow (matches the docker-images endpoints):
  1. POST /api/docker-images/presigned-upload-url {file_size_bytes}
       -> {upload_url, upload_id, required_headers, ...}
  2. PUT the .tar.gz to upload_url with required_headers (direct to GCS;
     bypasses IAP, which is why inline upload fails on large files).
  3. POST /api/docker-images/register {upload_id, image_name, docker_build_cmd,
     description}.
  4. GET /api/docker-images/{id}/download and confirm gzip magic bytes (the list
     endpoint's source_code_s3_path is unreliable).

The tarball is the BUILD SOURCE, not the image, and it is registered through
Taiga's docker-images API only. It is never emitted as ``preloaded_files`` and is
not mounted into the agent runtime.

Unlike ML_Envs, ISO source archives intentionally preserve reviewer/debug
materials that may be useful for delivery review but must not be agent-visible:
``solution/``, ``.alignerr/``, ``baselines/``, ``data-generation/``, and non-ML
``data/`` / ``scorer/data/`` contents. The archive also includes the shared
``grader/`` package, with the task's ``environment/Dockerfile`` placed at the
archive root so ``docker build -f Dockerfile .`` works directly.

For ML mounted datasets, the grade workflow's mount step (``scripts/sync_mount.sh``)
empties those source dirs in the build context before the image is built
(``ml`` data -> /data, scorer/data -> /mcp_server/data, plus any declared
``[[preloaded_files]]`` sources), leaving a ``.gitkeep`` placeholder. Archiving
the context as-is keeps the registered source faithful to the pushed image:
mounted data is absent and served read-only from GCS at deploy; baked data for
other task types remains in the source archive. .git / caches / compiled python
are excluded everywhere.

This script is fail-closed: digest-pinned images only, non-empty archives,
missing ``--include`` sources are fatal when ``--require-includes`` is set (CI
default), transient API failures are retried, and registration is verified via
download before returning success. Idempotent skip only when an exact-image
record already has a downloadable gzip source.

Auth: TAIGA_TOKEN (Bearer). Base URL: TAIGA_BASE_URL (default taiga.ant.dev).

Usage:
    python3 scripts/upload_docker_source.py \
        --image-name "<registry/repo@sha256:...>" \
        --problem-dir "problems/<task_id>" \
        [--repo-root .] \
        [--build-cmd "docker buildx build -f Dockerfile ..."] \
        [--description "..."] [--dry-run] [--require-includes] [--retries 5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

# Shared source roots (relative to repo root) needed to rebuild a task image.
# The base image is pulled from the registry by tag, so base/ source is not
# required here; grader/ is COPYied into the image and must be present.
SOURCE_ROOTS = ("grader",)
# Directory/file names dropped anywhere in the tree.
EXCLUDE_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
    ".venv",
}
EXCLUDE_SUFFIXES = (".pyc", ".pyo")
# A valid gzip of just Dockerfile + problem scaffolding is always larger than
# this; smaller archives almost always mean an empty/broken context.
MIN_TARBALL_BYTES = 64
DEFAULT_RETRIES = 5
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0)


def _base_url() -> str:
    return os.environ.get("TAIGA_BASE_URL", "https://taiga.ant.dev").rstrip("/")


def _token() -> str:
    tok = (os.environ.get("TAIGA_TOKEN") or "").strip()
    if not tok:
        sys.exit("TAIGA_TOKEN is not set.")
    return tok


def _req(
    method: str,
    url: str,
    *,
    headers: dict,
    body: bytes | None = None,
    timeout: int = 300,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        return 0, str(e.reason if hasattr(e, "reason") else e).encode()
    except (TimeoutError, OSError) as e:
        # Python 3.13+ can raise these from getresponse/read outside URLError;
        # treat as soft failures so upload_with_retries can retry.
        return 0, str(e).encode()


def _member_skip(name: str) -> bool:
    parts = name.split("/")
    if any(p in EXCLUDE_NAMES for p in parts):
        return True
    return name.endswith(EXCLUDE_SUFFIXES)


def _tar_filter(ti: tarfile.TarInfo):
    """Drop vcs / caches / compiled python anywhere in the tree."""
    return None if _member_skip(ti.name) else ti


def require_digest_pinned(image_name: str) -> None:
    """Fail closed unless the image is digest-pinned (repo@sha256:...)."""
    if "@sha256:" not in image_name:
        sys.exit(
            f"image_name must be digest-pinned (repo@sha256:...), got: {image_name!r}"
        )


def build_tarball(
    repo_root: str,
    problem_dir: str,
    out_path: str,
    extra_files: list | None = None,
    *,
    require_includes: bool = False,
) -> int:
    """Tar the problem dir as-is (i.e. post mount-slim) + shared SOURCE_ROOTS,
    with the task Dockerfile duplicated at the archive root. Returns compressed
    bytes.
    """
    problem_abs = os.path.join(repo_root, problem_dir)
    if not os.path.isdir(problem_abs):
        sys.exit(f"problem dir not found: {problem_abs}")
    dockerfile = os.path.join(problem_abs, "environment", "Dockerfile")
    if not os.path.isfile(dockerfile):
        sys.exit(f"environment/Dockerfile not found under {problem_dir}")

    problem_arcname = problem_dir.rstrip("/")
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(problem_abs, arcname=problem_arcname, filter=_tar_filter)
        for rel in SOURCE_ROOTS:
            abs_root = os.path.join(repo_root, rel)
            if os.path.isdir(abs_root):
                tar.add(abs_root, arcname=rel, filter=_tar_filter)
        # Dockerfile at the archive root so `docker build -f Dockerfile .` works.
        tar.add(dockerfile, arcname="Dockerfile")
        for src, arcname in extra_files or []:
            if os.path.isfile(src):
                tar.add(src, arcname=arcname)
                print(f"  + included {arcname}")
            elif require_includes:
                sys.exit(f"required include missing: {src} (arcname={arcname})")
            else:
                sys.stderr.write(f"  (include skipped; not found: {src})\n")
    size = os.path.getsize(out_path)
    if size < MIN_TARBALL_BYTES:
        sys.exit(
            f"source tarball too small ({size} bytes); refusing to register empty/broken archive"
        )
    return size


def find_registered(image_name: str) -> dict | None:
    """Return the exact-image docker-images record, or None."""
    base = _base_url() + "/api"
    auth = {"Authorization": f"Bearer {_token()}"}
    query = image_name.rsplit("@", 1)[-1] if "@" in image_name else image_name
    status, raw = _req(
        "GET",
        f"{base}/docker-images?search={urllib.parse.quote(query)}",
        headers=auth,
    )
    if status // 100 != 2 or not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    rows = (
        data if isinstance(data, list) else data.get("data") or data.get("items") or []
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        if (row.get("image_name") or row.get("image")) == image_name:
            return row
    return None


def already_registered(image_name: str) -> bool:
    """True when an exact-image source record already exists (idempotent skip).

    Prefer ``source_verified`` / ``find_registered`` + ``verify_download`` for
    fail-closed paths; this thin wrapper remains for tests/callers.
    """
    return find_registered(image_name) is not None


def verify_download(image_id: str) -> bool:
    """True when ``/docker-images/{id}/download`` returns a gzip stream.

    The list view often returns ``source_code_s3_path=null`` even for covered
    images, so only the gzip magic on the download stream is reliable.
    """
    if not image_id:
        return False
    base = _base_url() + "/api"
    auth = {"Authorization": f"Bearer {_token()}"}
    req = urllib.request.Request(
        f"{base}/docker-images/{image_id}/download",
        headers=auth,
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read(2)[:2] == b"\x1f\x8b"
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return False


def source_verified(image_name: str) -> str | None:
    """Return the record id when source is registered AND downloadable."""
    rec = find_registered(image_name)
    if not rec:
        return None
    image_id = str(rec.get("id") or "")
    if not image_id:
        return None
    return image_id if verify_download(image_id) else None


def upload(
    image_name: str, build_cmd: str, description: str, tar_path: str, size: int
) -> tuple[int, str | None]:
    """Register source for ``image_name``. Returns (exit_code, record_id)."""
    base = _base_url() + "/api"
    auth = {"Authorization": f"Bearer {_token()}"}

    status, raw = _req(
        "POST",
        f"{base}/docker-images/presigned-upload-url",
        headers={**auth, "Content-Type": "application/json"},
        body=json.dumps({"file_size_bytes": size}).encode(),
    )
    if status // 100 != 2:
        sys.stderr.write(f"presigned-upload-url failed HTTP {status}: {raw[:400]!r}\n")
        return 1, None
    try:
        slot = json.loads(raw)
    except json.JSONDecodeError:
        sys.stderr.write(f"presigned-upload-url returned non-JSON: {raw[:400]!r}\n")
        return 1, None
    upload_url = slot.get("upload_url")
    upload_id = slot.get("upload_id")
    if not upload_url or not upload_id:
        sys.stderr.write(f"presigned-upload-url missing fields: {slot!r}\n")
        return 1, None
    required_headers = dict(slot.get("required_headers") or {})
    required_headers.setdefault("Content-Type", "application/gzip")

    with open(tar_path, "rb") as f:
        data = f.read()
    if len(data) != size:
        sys.stderr.write(
            f"tarball size drifted before PUT ({len(data)} != {size}); refusing\n"
        )
        return 1, None
    status, raw = _req(
        "PUT",
        upload_url,
        headers={str(k): str(v) for k, v in required_headers.items()},
        body=data,
    )
    if status // 100 != 2:
        sys.stderr.write(f"PUT to upload_url failed HTTP {status}: {raw[:400]!r}\n")
        return 1, None

    status, raw = _req(
        "POST",
        f"{base}/docker-images/register",
        headers={**auth, "Content-Type": "application/json"},
        body=json.dumps(
            {
                "upload_id": upload_id,
                "image_name": image_name,
                "docker_build_cmd": build_cmd,
                "description": description,
            }
        ).encode(),
    )
    if status // 100 != 2:
        sys.stderr.write(f"register failed HTTP {status}: {raw[:400]!r}\n")
        return 1, None
    try:
        rec = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        rec = {}
    image_id = str(rec.get("id") or "") or None
    print(
        f"registered docker-image source: id={image_id} image_name={image_name} ({size} bytes)"
    )
    return 0, image_id


def upload_with_retries(
    image_name: str,
    build_cmd: str,
    description: str,
    tar_path: str,
    size: int,
    *,
    retries: int = DEFAULT_RETRIES,
) -> int:
    """Upload + download-verify, retrying transient failures. Fail-closed."""
    attempts = max(1, retries)
    last_err = "unknown failure"
    for attempt in range(1, attempts + 1):
        # Another concurrent submit may have landed source between retries.
        existing = source_verified(image_name)
        if existing:
            print(
                f"Source already registered and download-verified for {image_name} "
                f"(id={existing}); skipping."
            )
            return 0

        code, image_id = upload(image_name, build_cmd, description, tar_path, size)
        if code == 0 and image_id:
            # Call verify once per attempt: a flaky first probe must not force a
            # second call that can disagree and fall through to fatal failure.
            if verify_download(image_id):
                print(f"download-verified gzip source for id={image_id}")
                return 0
            last_err = f"register succeeded (id={image_id}) but download verify failed"
        elif code == 0 and not image_id:
            # Register returned 2xx without an id — fall back to search+verify.
            verified = source_verified(image_name)
            if verified:
                print(f"download-verified gzip source for id={verified}")
                return 0
            last_err = (
                "register succeeded but record id missing and search verify failed"
            )
        else:
            last_err = f"upload attempt {attempt} failed"

        if attempt >= attempts:
            break
        delay = RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
        print(
            f"::warning::Docker source upload attempt {attempt}/{attempts} failed "
            f"({last_err}); retrying in {delay:.0f}s",
            flush=True,
        )
        time.sleep(delay)

    sys.stderr.write(
        f"FATAL: failed to register downloadable Docker source for {image_name} "
        f"after {attempts} attempt(s): {last_err}\n"
    )
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload a task's Docker source to Taiga")
    ap.add_argument(
        "--image-name", required=True, help="deployed image ref (digest-pinned)"
    )
    ap.add_argument("--problem-dir", required=True, help="e.g. problems/<task_id>")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--build-cmd", default=None)
    ap.add_argument("--description", default=None)
    ap.add_argument(
        "--include",
        action="append",
        nargs=2,
        default=[],
        metavar=("SRC", "ARCNAME"),
        help="extra file to add to the archive at ARCNAME (repeatable)",
    )
    ap.add_argument(
        "--require-includes",
        action="store_true",
        help="fail if any --include SRC path is missing (CI default)",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"upload+verify attempts before failing (default {DEFAULT_RETRIES})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build the tarball and print its size; upload nothing",
    )
    args = ap.parse_args()

    require_digest_pinned(args.image_name)
    if args.retries < 1:
        sys.exit("--retries must be >= 1")

    problem_dir = args.problem_dir.rstrip("/")
    build_cmd = args.build_cmd or (
        "docker buildx build -f Dockerfile "
        f"--build-arg PROBLEM_DIR={problem_dir} "
        "-t " + args.image_name.split("@")[0] + " ."
    )
    description = args.description or (
        f"Source for {os.path.basename(problem_dir)} "
        "(matches the pushed image build context; mounted datasets are served "
        "read-only from GCS, not in this archive). "
        f"Rebuild: {build_cmd}"
    )

    if not args.dry_run:
        existing = source_verified(args.image_name)
        if existing:
            print(
                f"Source already registered and download-verified for {args.image_name} "
                f"(id={existing}); skipping."
            )
            return 0

    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "source.tar.gz")
        size = build_tarball(
            os.path.abspath(args.repo_root),
            problem_dir,
            tar_path,
            extra_files=args.include,
            require_includes=args.require_includes,
        )
        print(f"built source tarball: {size} bytes")
        if args.dry_run:
            print("[dry-run] not uploading")
            return 0
        return upload_with_retries(
            args.image_name,
            build_cmd,
            description,
            tar_path,
            size,
            retries=args.retries,
        )


if __name__ == "__main__":
    sys.exit(main())
