"""Digest-locked task-capsule metadata used by the nested Docker runtime.

The authoring/export package deliberately is not imported here.  Taiga's rubric
image contains this small runtime package, while authoring dependencies are not
part of the production trust boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grading.faults import InfrastructureFault

_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SERVICE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CapsuleRuntimeError(InfrastructureFault):
    """The trusted task capsule is missing, malformed, or inconsistent."""


def validate_service_name(value: str) -> str:
    """Return a validated Docker Compose service name."""
    value = value.strip()
    if not _SERVICE_NAME_RE.fullmatch(value):
        raise CapsuleRuntimeError(
            f"invalid service name {value!r}; expected an alphanumeric first "
            "character followed by alphanumerics, '.', '_', or '-'"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_json_object(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise CapsuleRuntimeError(
            f"capsule manifest is unavailable at {path}: {exc}"
        ) from exc
    if not path.is_file() or path.is_symlink():
        raise CapsuleRuntimeError(f"capsule manifest is not a regular file: {path}")
    if stat.st_size > _MAX_MANIFEST_BYTES:
        raise CapsuleRuntimeError(
            f"capsule manifest exceeds {_MAX_MANIFEST_BYTES} bytes: {path}"
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapsuleRuntimeError(
            f"could not parse capsule manifest {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CapsuleRuntimeError("capsule manifest must contain a JSON object")
    return payload


def _contained_path(root: Path, raw: str, *, label: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        raise CapsuleRuntimeError(f"{label} must be relative to the capsule directory")
    resolved_root = root.resolve(strict=True)
    current = resolved_root
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise CapsuleRuntimeError(f"{label} has a symlink ancestor: {current}")
        if not current.exists():
            break
    resolved = (resolved_root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise CapsuleRuntimeError(
            f"{label} escapes the capsule directory: {raw!r}"
        ) from exc
    return resolved


def _contained_file_with_basename_fallback(root: Path, raw: str, *, label: str) -> Path:
    """Resolve an embedded file, tolerating a flattened COPY destination."""
    path = _contained_path(root, raw, label=label)
    if path.is_file() and not path.is_symlink():
        return path
    basename = Path(raw).name
    fallback = _contained_path(root, basename, label=label)
    if fallback.is_file() and not fallback.is_symlink():
        return fallback
    return path


def _first_string(mapping: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _image_rows(
    payload: Mapping[str, Any],
) -> Iterable[tuple[str | None, Mapping[str, Any]]]:
    raw = payload.get("images")
    if raw is None:
        raw = payload.get("archives")
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, Mapping):
                yield None, entry
        return
    if isinstance(raw, Mapping):
        for service, entry in raw.items():
            if isinstance(service, str) and isinstance(entry, Mapping):
                yield service, entry
        return

    # A few capsule builders naturally use a service-keyed manifest.  Accept
    # that representation too, but only rows that actually declare an archive.
    services = payload.get("services")
    if isinstance(services, Mapping):
        for service, entry in services.items():
            if (
                isinstance(service, str)
                and isinstance(entry, Mapping)
                and ("archive" in entry or "archive_path" in entry)
            ):
                yield service, entry
    elif isinstance(services, list):
        for entry in services:
            if isinstance(entry, Mapping) and (
                "archive" in entry or "archive_path" in entry
            ):
                yield None, entry


@dataclass(frozen=True, slots=True)
class CapsuleImageArchive:
    """One trusted child image archive embedded in the outer task image."""

    service: str
    role: str
    archive_path: Path
    archive_sha256: str
    image_ref: str
    image_digest: str

    def verify_archive(self) -> None:
        """Verify the archive before it is handed to nested dockerd."""
        try:
            is_regular = (
                self.archive_path.is_file() and not self.archive_path.is_symlink()
            )
        except OSError as exc:
            raise CapsuleRuntimeError(
                f"could not inspect image archive for {self.service!r}: {exc}"
            ) from exc
        if not is_regular:
            raise CapsuleRuntimeError(
                f"image archive for {self.service!r} is not a regular file: "
                f"{self.archive_path}"
            )
        actual = _sha256_file(self.archive_path)
        if actual != self.archive_sha256:
            raise CapsuleRuntimeError(
                f"archive digest mismatch for service {self.service!r}: "
                f"expected {self.archive_sha256}, got {actual}"
            )


@dataclass(frozen=True, slots=True)
class CapsuleBundle:
    """Validated capsule compose file and child image archives."""

    root: Path
    manifest_path: Path
    compose_path: Path
    images: tuple[CapsuleImageArchive, ...]
    agent_service: str | None = None
    verifier_service: str | None = None

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        manifest_name: str = "manifest.json",
        compose_override: str | None = None,
    ) -> CapsuleBundle:
        """Read and structurally validate a capsule manifest."""
        current = Path(root.anchor)
        for part in root.parts[1:]:
            current /= part
            if current.is_symlink():
                raise CapsuleRuntimeError(
                    f"capsule path has a symlink ancestor: {current}"
                )
            if not current.exists():
                break
        try:
            resolved_root = root.resolve(strict=True)
        except OSError as exc:
            raise CapsuleRuntimeError(
                f"capsule directory is unavailable at {root}: {exc}"
            ) from exc
        if not resolved_root.is_dir() or resolved_root.is_symlink():
            raise CapsuleRuntimeError(
                f"capsule path is not a directory: {resolved_root}"
            )

        manifest_path = _contained_path(
            resolved_root, manifest_name, label="capsule manifest path"
        )
        if (
            manifest_name == "manifest.json"
            and not manifest_path.is_file()
            and not manifest_path.is_symlink()
        ):
            manifest_path = _contained_path(
                resolved_root,
                "capsule.manifest.json",
                label="capsule manifest path",
            )
        outer_payload = _bounded_json_object(manifest_path)
        compose_name = compose_override or _first_string(
            outer_payload, "compose_file", "compose_path"
        )
        if compose_name is None:
            compose_name = "docker-compose.yaml"
        compose_path = _contained_path(
            resolved_root, compose_name, label="capsule compose path"
        )
        if not compose_path.is_file() or compose_path.is_symlink():
            raise CapsuleRuntimeError(
                f"capsule compose file is not a regular file: {compose_path}"
            )

        payload = outer_payload
        image_manifest_name = _first_string(outer_payload, "image_manifest")
        image_manifest_path = manifest_path
        if image_manifest_name is not None:
            image_manifest_path = _contained_file_with_basename_fallback(
                resolved_root,
                image_manifest_name,
                label="capsule image manifest path",
            )
            payload = _bounded_json_object(image_manifest_path)
        outer_archive_name = _first_string(outer_payload, "image_archive")

        images: list[CapsuleImageArchive] = []
        seen_services: set[str] = set()
        for keyed_service, row in _image_rows(payload):
            archive_value = row.get("archive")
            archive_meta: Mapping[str, Any] = (
                archive_value if isinstance(archive_value, Mapping) else {}
            )
            archive_name = (
                archive_value.strip()
                if isinstance(archive_value, str) and archive_value.strip()
                else _first_string(row, "archive_path", "path")
                or _first_string(archive_meta, "path", "file")
            )
            service = keyed_service or _first_string(row, "service", "name")
            role = _first_string(row, "role") or "sidecar"
            archive_sha256 = _first_string(
                row, "archive_sha256", "archive_digest", "sha256"
            ) or _first_string(archive_meta, "sha256", "digest")
            image_ref = _first_string(row, "runtime_ref", "image_ref", "image", "ref")
            image_digest = _first_string(
                row, "image_digest", "digest", "content_digest", "image_id"
            )

            if not all(
                (service, archive_name, archive_sha256, image_ref, image_digest)
            ):
                raise CapsuleRuntimeError(
                    "each capsule image requires service, archive path, archive "
                    "SHA-256, image reference, and image digest"
                )
            service = validate_service_name(service)
            if service in seen_services:
                raise CapsuleRuntimeError(
                    f"capsule contains duplicate image service {service!r}"
                )
            seen_services.add(service)
            if not _SHA256_HEX_RE.fullmatch(archive_sha256):
                raise CapsuleRuntimeError(
                    f"invalid archive SHA-256 for service {service!r}"
                )
            if not _SHA256_DIGEST_RE.fullmatch(image_digest):
                raise CapsuleRuntimeError(
                    f"invalid image digest for service {service!r}: {image_digest!r}"
                )
            if "@" in image_ref and not image_ref.endswith(f"@{image_digest}"):
                raise CapsuleRuntimeError(
                    f"digest-pinned image reference for {service!r} disagrees "
                    "with the capsule image digest"
                )
            archive_candidates = [
                _contained_path(
                    resolved_root,
                    archive_name,
                    label=f"archive path for service {service!r}",
                )
            ]
            if image_manifest_path.parent != resolved_root:
                archive_candidates.append(
                    _contained_path(
                        image_manifest_path.parent,
                        archive_name,
                        label=f"archive path for service {service!r}",
                    )
                )
            if outer_archive_name is not None:
                archive_candidates.append(
                    _contained_file_with_basename_fallback(
                        resolved_root,
                        outer_archive_name,
                        label=f"archive path for service {service!r}",
                    )
                )
            archive_path = next(
                (
                    candidate
                    for candidate in archive_candidates
                    if candidate.is_file() and not candidate.is_symlink()
                ),
                archive_candidates[0],
            )
            images.append(
                CapsuleImageArchive(
                    service=service,
                    role=role,
                    archive_path=archive_path,
                    archive_sha256=archive_sha256,
                    image_ref=image_ref,
                    image_digest=image_digest,
                )
            )

        if not images:
            raise CapsuleRuntimeError(
                "capsule manifest declares no child image archives"
            )
        agent_service = _first_string(outer_payload, "agent_service", "main_service")
        verifier_service = _first_string(outer_payload, "verifier_service")
        if agent_service is not None:
            agent_service = validate_service_name(agent_service)
        if verifier_service is not None:
            verifier_service = validate_service_name(verifier_service)
        return cls(
            root=resolved_root,
            manifest_path=manifest_path,
            compose_path=compose_path,
            images=tuple(images),
            agent_service=agent_service,
            verifier_service=verifier_service,
        )

    def verify_archives(self) -> None:
        """Verify every embedded archive before loading any of them."""
        verified: dict[Path, str] = {}
        for image in self.images:
            previous = verified.get(image.archive_path)
            if previous is not None:
                if previous != image.archive_sha256:
                    raise CapsuleRuntimeError(
                        "capsule assigns conflicting digests to archive "
                        f"{image.archive_path}"
                    )
                continue
            image.verify_archive()
            verified[image.archive_path] = image.archive_sha256

    def image_for_service(self, service: str) -> CapsuleImageArchive:
        """Return the locked image row for ``service``."""
        service = validate_service_name(service)
        for image in self.images:
            if image.service == service:
                return image
        raise CapsuleRuntimeError(
            f"service {service!r} has no digest-locked image in the capsule"
        )
