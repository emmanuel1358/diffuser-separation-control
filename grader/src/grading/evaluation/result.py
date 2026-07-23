"""Public receipts and root-only traces for evaluation decisions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

EVALUATION_TRACE_PATH_ENV = "LBX_EVALUATION_TRACE_PATH"
PUBLIC_RECEIPT_SCHEMA = "continuous-evaluation-receipt.v1"
PRIVATE_TRACE_SCHEMA = "continuous-evaluation-trace.v1"
RUBRIC_PRIVATE_TRACE_SCHEMA = "rubric-evaluation-trace.v1"


@dataclass(frozen=True)
class TargetDecision:
    accepted: bool
    reason: str

    def public_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reason": self.reason}


@dataclass(frozen=True)
class PublicEvaluationReceipt:
    protocol: str
    plan_sha256: str
    seed_commitment: str
    attested: bool
    challenge_count: int
    family_alpha: float
    decisions: Mapping[str, TargetDecision]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PUBLIC_RECEIPT_SCHEMA,
            "protocol": self.protocol,
            "plan_sha256": self.plan_sha256,
            "seed_commitment": self.seed_commitment,
            "attested": self.attested,
            "challenge_count": self.challenge_count,
            "family_alpha": self.family_alpha,
            "decisions": {
                name: decision.public_dict()
                for name, decision in sorted(self.decisions.items())
            },
        }


def write_private_trace(
    *,
    protocol: str,
    plan_sha256: str,
    seed_commitment: str,
    targets: Mapping[str, Mapping[str, Any]],
    replay: Mapping[str, Any] | None = None,
) -> None:
    """Write exact statistics only when a root-side runner supplied a sink."""
    raw_path = os.environ.get(EVALUATION_TRACE_PATH_ENV)
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": (
            RUBRIC_PRIVATE_TRACE_SCHEMA
            if protocol.startswith("declarative-rubric")
            else PRIVATE_TRACE_SCHEMA
        ),
        "protocol": protocol,
        "plan_sha256": plan_sha256,
        "seed_commitment": seed_commitment,
        "replay": dict(replay or {}),
        "targets": {name: dict(value) for name, value in sorted(targets.items())},
    }
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(
            f"could not create private evaluation trace securely at {path}: {exc}"
        ) from exc
    with os.fdopen(fd, "wb", closefd=True) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "EVALUATION_TRACE_PATH_ENV",
    "PRIVATE_TRACE_SCHEMA",
    "PUBLIC_RECEIPT_SCHEMA",
    "RUBRIC_PRIVATE_TRACE_SCHEMA",
    "PublicEvaluationReceipt",
    "TargetDecision",
    "write_private_trace",
]
