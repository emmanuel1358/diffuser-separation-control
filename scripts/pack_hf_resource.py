#!/usr/bin/env python3
"""Resolve, download, and pack a single Hugging Face repo into a squashfs laid out
as that repo's HF hub cache folder, for read-only mounting into a task container.

Packs the huggingface_hub cache folder (models--<org>--<name>/{refs,snapshots,
blobs}) so that, mounted at /tmp/.cache/huggingface/hub/<folder>, an agent can
from_pretrained("<org>/<name>") entirely offline. Content-addressed by the repo's
immutable commit sha, resolved from a cheap repo_info metadata call so the cache
can be checked before paying for a multi-GB download.

Subcommands (selected by flags; each prints one JSON object, or for --list one
JSON object per line, to stdout; all human progress goes to stderr):

  --metadata <metadata.json> --list
      Read metadata.json:hf_resources, normalize each entry, and print one
      normalized resource object per line. No network.

  --resource '<json>' --resolve-only
      Resolve the repo's commit sha and derived names. One small metadata call,
      no weights downloaded. Prints the resolved object.

  --resource '<json>' --pack --out-dir DIR
      Resolve, snapshot_download the repo into a temp hub, write refs/main (and
      refs/<revision> for a pinned tag/branch) so it resolves offline by name,
      and pack the cache folder into DIR/<remote-basename>. Prints the resolved
      object with "packed_path" set.

A "resource" is either a string ("org/name") or an object:
  {"repo_id": "org/name", "revision": "main", "repo_type": "model",
   "allow_patterns": ["*.safetensors", "*.json"], "ignore_patterns": ["*.bin"]}

revision defaults to "main", repo_type to "model". allow_patterns/ignore_patterns
narrow what is downloaded (and thus packed); when either is set the object name
gets a short pattern-hash suffix so a narrowed variant is a distinct cache
object from the full repo.

Gated repos need a token at pack time: set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN)
in the environment running the deploy. There is no network at task RUNTIME, so
the download must happen here, on the build/deploy host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn


VALID_REPO_TYPES = {"model", "dataset"}
# A revision matching a 40-hex sha needs no live resolution -- it IS the address.
_SHA_RE = re.compile(r"\A[0-9a-fA-F]{40}\Z")
HF_HUB_CACHE_DIR = "/tmp/.cache/huggingface/hub"
# Central cross-task cache prefix: each repo@sha is packed/uploaded once.
REMOTE_PREFIX = "cache/huggingface"


def die(msg: str) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# --- Pure helpers (no network) ---------------------------------------------

def normalize_resource(entry: object) -> dict:
    """Normalize a string or object hf_resources entry into a full dict; raises
    ValueError on a malformed entry."""
    if isinstance(entry, str):
        entry = {"repo_id": entry}
    if not isinstance(entry, dict):
        raise ValueError(f"hf_resources entry must be a string or object, got {type(entry).__name__}")

    repo_id = entry.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("hf_resources entry is missing a non-empty 'repo_id'")
    repo_id = repo_id.strip()
    if repo_id.count("/") > 1 or repo_id.startswith("/") or repo_id.endswith("/"):
        raise ValueError(f"hf_resources repo_id {repo_id!r} is not a valid '[org/]name' id")

    repo_type = entry.get("repo_type", "model")
    if repo_type not in VALID_REPO_TYPES:
        raise ValueError(
            f"hf_resources repo_type {repo_type!r} for {repo_id!r} must be one of {sorted(VALID_REPO_TYPES)}"
        )

    revision = entry.get("revision", "main")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"hf_resources revision for {repo_id!r} must be a non-empty string")
    revision = revision.strip()

    def _patterns(key: str) -> list[str] | None:
        val = entry.get(key)
        if val is None:
            return None
        if not isinstance(val, list) or not all(isinstance(p, str) and p for p in val):
            raise ValueError(f"hf_resources {key} for {repo_id!r} must be a list of non-empty strings")
        return list(val)

    allow_patterns = _patterns("allow_patterns")
    ignore_patterns = _patterns("ignore_patterns")

    unknown = set(entry) - {"repo_id", "repo_type", "revision", "allow_patterns", "ignore_patterns"}
    if unknown:
        raise ValueError(f"hf_resources entry for {repo_id!r} has unknown keys {sorted(unknown)}")

    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "allow_patterns": allow_patterns,
        "ignore_patterns": ignore_patterns,
    }


def repo_folder_name(repo_id: str, repo_type: str) -> str:
    """The cache folder huggingface_hub names for a repo, e.g.
    models--meta-llama--Llama-3.1-8B."""
    return f"{repo_type}s--" + repo_id.replace("/", "--")


def pattern_suffix(allow_patterns: list[str] | None, ignore_patterns: list[str] | None) -> str:
    """Deterministic suffix distinguishing a pattern-narrowed download from the
    full repo; empty when no patterns are set."""
    if not allow_patterns and not ignore_patterns:
        return ""
    payload = json.dumps(
        {"allow": sorted(allow_patterns or []), "ignore": sorted(ignore_patterns or [])},
        sort_keys=True,
    )
    return "-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def derive_names(resource: dict, commit_sha: str) -> dict:
    """Derive the cache folder, mount target, remote dir, and remote object name."""
    folder = repo_folder_name(resource["repo_id"], resource["repo_type"])
    suffix = pattern_suffix(resource["allow_patterns"], resource["ignore_patterns"])
    remote_dir = f"{REMOTE_PREFIX}/{folder}"
    return {
        "repo_folder_name": folder,
        "mount_local_path": f"{HF_HUB_CACHE_DIR}/{folder}",
        "remote_dir": remote_dir,
        "remote_basename": f"{commit_sha}{suffix}.squashfs",
        "remote_name": f"{remote_dir}/{commit_sha}{suffix}.squashfs",
    }


def _attach_names(resource: dict, commit_sha: str) -> dict:
    """Bind (resource, commit_sha) into the full resolved object -- the one place
    that ties a resource to a sha, so all derived names use a single sha."""
    out = dict(resource)
    out["commit_sha"] = commit_sha
    out.update(derive_names(resource, commit_sha))
    return out


# --- Network helpers (huggingface_hub) -------------------------------------

def _require_hf():
    try:
        import huggingface_hub  # noqa: F401
        return huggingface_hub
    except ImportError:
        die(
            "huggingface_hub is not installed on this (build/deploy) host. "
            "Install it where build_and_push.sh runs: pip install huggingface_hub"
        )


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def resolve(resource: dict) -> dict:
    """Resolve the immutable commit sha for repo@revision and attach derived names.

    A sha-pinned revision resolves fully offline (the sha IS the answer). A
    symbolic revision makes one repo_info call (no weights); that call
    authenticates, so a gated repo on a moving ref needs the token every deploy."""
    revision = resource["revision"]
    if _SHA_RE.match(revision):
        commit_sha = revision.lower()
    else:
        hf = _require_hf()
        api = hf.HfApi(token=_hf_token())
        try:
            info = api.repo_info(
                resource["repo_id"],
                repo_type=resource["repo_type"],
                revision=revision,
            )
        except Exception as exc:  # huggingface_hub raises a family of HfHubHTTPError types
            die(
                f"could not resolve {resource['repo_id']}@{revision} "
                f"(repo_type={resource['repo_type']}): {exc}. "
                "If the repo is gated, set HF_TOKEN on the deploy host."
            )
        commit_sha = getattr(info, "sha", None)
        if not commit_sha:
            die(f"repo_info for {resource['repo_id']} returned no commit sha")
    return _attach_names(resource, commit_sha)


def _refs_for_revision(revision: str) -> list[str]:
    """Refs to write so the repo resolves offline by name: always "main", plus a
    symbolic revision. A bare sha needs no ref (resolves via snapshots/<sha>)."""
    refs = ["main"]
    if revision != "main" and not _SHA_RE.match(revision):
        refs.append(revision)
    return refs


def _ensure_ref(repo_dir: Path, ref: str, commit_sha: str) -> None:
    """Write <repo_dir>/refs/<ref> = commit_sha (idempotent; ref may contain '/').
    snapshot_download fetches by sha and writes no refs, so these let
    from_pretrained resolve offline by name."""
    ref_path = repo_dir / "refs" / ref
    if ref_path.is_file() and ref_path.read_text(encoding="utf-8").strip() == commit_sha:
        return
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(commit_sha, encoding="utf-8")


def pack(resource: dict, out_dir: Path, commit_sha: str | None = None) -> dict:
    """Download and pack; returns the resolved object with 'packed_path' set.

    A given commit_sha is a pre-resolved sha to reuse (so the bytes, object name,
    and mount all share one sha); otherwise resolve here."""
    hf = _require_hf()
    if commit_sha is not None:
        if not _SHA_RE.match(commit_sha):
            die(f"--commit-sha must be a 40-hex commit sha, got {commit_sha!r}")
        resolved = _attach_names(resource, commit_sha.lower())
    else:
        resolved = resolve(resource)
    commit_sha = resolved["commit_sha"]
    folder = resolved["repo_folder_name"]

    if not _have("mksquashfs"):
        die("mksquashfs not found (install squashfs-tools) on the deploy host.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / resolved["remote_basename"]

    with tempfile.TemporaryDirectory(prefix="hfcache-") as tmp:
        cache_dir = Path(tmp) / "hub"
        cache_dir.mkdir(parents=True, exist_ok=True)
        log(f"Downloading {resource['repo_id']}@{commit_sha} ({resource['repo_type']}) ...")
        # Fetch by sha for an immutable snapshot; refs are backfilled below.
        hf.snapshot_download(
            resource["repo_id"],
            repo_type=resource["repo_type"],
            revision=commit_sha,
            cache_dir=str(cache_dir),
            allow_patterns=resource["allow_patterns"],
            ignore_patterns=resource["ignore_patterns"],
            token=_hf_token(),
        )
        repo_dir = cache_dir / folder
        if not repo_dir.is_dir():
            die(f"expected cache folder {repo_dir} was not created by snapshot_download")
        # Write refs so the repo resolves offline by name.
        for ref in _refs_for_revision(resource["revision"]):
            _ensure_ref(repo_dir, ref, commit_sha)

        log(f"Packing {repo_dir} -> {out_path}")
        if out_path.exists():
            out_path.unlink()
        # zstd level 3; squashfs root is repo_dir's contents (refs/ snapshots/
        # blobs/), with symlinks preserved so they resolve within the mount.
        subprocess.run(
            ["mksquashfs", str(repo_dir), str(out_path),
             "-comp", "zstd", "-Xcompression-level", "3", "-no-recovery"],
            check=True,
            stdout=sys.stderr.fileno(),
        )

    resolved["packed_path"] = str(out_path)
    resolved["packed_bytes"] = out_path.stat().st_size
    return resolved


def _have(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None


# --- CLI --------------------------------------------------------------------

def _load_resources_from_metadata(metadata_path: Path) -> list[dict]:
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"could not parse {metadata_path}: {exc}")
    raw = data.get("hf_resources", []) if isinstance(data, dict) else []
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        die(f"{metadata_path}:hf_resources must be a list")
    out = []
    for entry in raw:
        try:
            out.append(normalize_resource(entry))
        except ValueError as exc:
            die(f"{metadata_path}: {exc}")
    # Two entries for the same (repo_id, repo_type) mount at the same cache dir
    # and collide (patterns can't disambiguate the path); reject up front.
    seen: set[tuple[str, str]] = set()
    for r in out:
        key = (r["repo_id"], r["repo_type"])
        if key in seen:
            die(
                f"{metadata_path}: hf_resources lists {r['repo_id']!r} "
                f"({r['repo_type']}) more than once; entries for the same repo "
                "mount at the same HF cache dir and collide. Use a single entry "
                "(combine allow_patterns, or omit them for the full repo)."
            )
        seen.add(key)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", help="path to a task metadata.json (with --list)")
    ap.add_argument("--resource", help="a single resource as a JSON string")
    ap.add_argument("--list", action="store_true", help="print normalized resources from --metadata")
    ap.add_argument("--resolve-only", action="store_true", help="resolve sha + names; no download")
    ap.add_argument("--pack", action="store_true", help="resolve, download, and pack (needs --out-dir)")
    ap.add_argument("--out-dir", help="output directory for --pack")
    ap.add_argument("--commit-sha", help="pre-resolved commit sha to reuse for --pack (single-resolve)")
    args = ap.parse_args(argv)

    if args.list:
        if not args.metadata:
            die("--list requires --metadata")
        for res in _load_resources_from_metadata(Path(args.metadata)):
            print(json.dumps(res, sort_keys=True))
        return 0

    if not args.resource:
        die("one of --list (with --metadata) or --resource is required")
    try:
        resource = normalize_resource(json.loads(args.resource))
    except (json.JSONDecodeError, ValueError) as exc:
        die(f"bad --resource: {exc}")

    if args.pack:
        if not args.out_dir:
            die("--pack requires --out-dir")
        print(json.dumps(pack(resource, Path(args.out_dir), commit_sha=args.commit_sha), sort_keys=True))
        return 0

    # default / --resolve-only
    print(json.dumps(resolve(resource), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
