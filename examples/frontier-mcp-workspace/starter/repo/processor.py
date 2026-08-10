from __future__ import annotations

from typing import Any

POLICY_VERSION = "legacy"


def adjudicate(claim: dict[str, Any]) -> dict[str, Any]:
    billed_cents = int(claim["billed_cents"])
    return {
        "claim_id": str(claim["claim_id"]),
        "decision": "pay",
        "payable_cents": billed_cents,
    }
