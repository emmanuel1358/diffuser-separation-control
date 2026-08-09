from __future__ import annotations

from typing import Any

POLICY_VERSION = "deterministic-v1"


def adjudicate(claim: dict[str, Any]) -> dict[str, Any]:
    covered = bool(claim["covered"])
    billed_cents = int(claim["billed_cents"])
    contract_cap_cents = int(claim["contract_cap_cents"])
    return {
        "claim_id": str(claim["claim_id"]),
        "decision": "pay" if covered else "deny",
        "payable_cents": min(billed_cents, contract_cap_cents) if covered else 0,
    }
