from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

SEED = {
    "records": [
        {"id": 1, "legacy": "red-old", "current": "red"},
        {"id": 2, "legacy": "gold-old", "current": "gold"},
    ]
}


def main() -> None:
    state_url = os.environ.get("STATE_URL", "http://state:7000")
    body = json.dumps(SEED, sort_keys=True, separators=(",", ":")).encode()
    request = urllib.request.Request(
        f"{state_url}/seed",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()

    marker = {
        "schema_version": "cutover-init.v1",
        "seed_count": len(SEED["records"]),
    }
    target = Path("/tmp/init.json")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(temporary, target)


if __name__ == "__main__":
    main()
