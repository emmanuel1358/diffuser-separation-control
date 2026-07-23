"""Private, reproducible context for one evaluation attempt."""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

EVALUATION_NONCE_ENV = "LBX_EVALUATION_NONCE"
EVALUATION_PLAN_ATTESTED_ENV = "LBX_EVALUATION_PLAN_ATTESTED"
MAX_COMMITTED_FILES = 10_000
MAX_COMMITTED_BYTES = 4 * 1024 * 1024 * 1024


def _framed(value: bytes) -> bytes:
    return len(value).to_bytes(8, byteorder="big") + value


def _update_array_digest(digest: Any, value: Any) -> None:
    import numpy as np

    array = np.asarray(value)
    digest.update(_framed(str(array.dtype).encode("ascii", errors="replace")))
    digest.update(
        _framed(repr(tuple(int(part) for part in array.shape)).encode("ascii"))
    )
    digest.update(_framed(np.ascontiguousarray(array).tobytes()))


def artifact_digest(raw_arrays: Mapping[str, tuple[Any, Any]]) -> str:
    """Digest candidate predictions without including private truth values."""
    digest = hashlib.sha256()
    for name in sorted(raw_arrays):
        digest.update(_framed(name.encode("utf-8")))
        prediction, _truth = raw_arrays[name]
        _update_array_digest(digest, prediction)
    return digest.hexdigest()


def workspace_artifact_digest(root: Path) -> str:
    """Commit every regular file in the submitted artifact tree."""
    root = Path(root)
    digest = hashlib.sha256()
    if not root.is_dir():
        raise ValueError(f"artifact workspace is not a directory: {root}")
    files = [path for path in sorted(root.rglob("*")) if not path.is_dir()]
    if len(files) > MAX_COMMITTED_FILES:
        raise ValueError(
            f"artifact workspace has {len(files)} files, over limit "
            f"{MAX_COMMITTED_FILES}"
        )
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                f"artifact workspace contains non-regular entry {relative!r}"
            )
        total_bytes += int(info.st_size)
        if total_bytes > MAX_COMMITTED_BYTES:
            raise ValueError(
                f"artifact workspace exceeds {MAX_COMMITTED_BYTES} committed bytes"
            )
        digest.update(_framed(relative.encode("utf-8")))
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(int(info.st_size).to_bytes(8, byteorder="big"))
        digest.update(file_digest.digest())
    return digest.hexdigest()


@dataclass(frozen=True)
class EvaluationContext:
    """Private-nonce-derived deterministic seeds plus a public commitment."""

    task_digest: str
    artifact_digest: str
    nonce: str
    seed_material: bytes
    commitment: str
    attested: bool

    @classmethod
    def create(
        cls,
        *,
        task_digest: str,
        raw_arrays: Mapping[str, tuple[Any, Any]],
    ) -> "EvaluationContext":
        return cls.create_from_artifact_digest(
            task_digest=task_digest,
            candidate_digest=artifact_digest(raw_arrays),
        )

    @classmethod
    def create_from_artifact_digest(
        cls,
        *,
        task_digest: str,
        candidate_digest: str,
    ) -> "EvaluationContext":
        nonce = os.environ.get(EVALUATION_NONCE_ENV)
        attested = bool(nonce and os.environ.get(EVALUATION_PLAN_ATTESTED_ENV) == "1")

        # A fresh private nonce is generated only after the artifact has been
        # committed. It therefore provides challenge unpredictability without a
        # long-lived shared secret. Local fallback remains deterministic and is
        # never presented as attested evidence.
        secret_bytes = (
            nonce.encode("utf-8")
            if nonce
            else hashlib.sha256(f"local:{task_digest}".encode()).digest()
        )
        effective_nonce = nonce or candidate_digest
        message = (
            f"lbx-evaluation.v1\0{task_digest}\0{candidate_digest}\0"
            f"{effective_nonce}"
        ).encode("utf-8")
        seed_material = hmac.new(secret_bytes, message, hashlib.sha256).digest()
        commitment = hashlib.sha256(seed_material).hexdigest()
        return cls(
            task_digest=task_digest,
            artifact_digest=candidate_digest,
            nonce=effective_nonce,
            seed_material=seed_material,
            commitment=commitment,
            attested=attested,
        )

    def seed(self, label: str) -> int:
        material = hmac.new(
            self.seed_material,
            label.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return int.from_bytes(material[:8], byteorder="big", signed=False)


__all__ = [
    "EVALUATION_NONCE_ENV",
    "EVALUATION_PLAN_ATTESTED_ENV",
    "EvaluationContext",
    "artifact_digest",
    "workspace_artifact_digest",
]
