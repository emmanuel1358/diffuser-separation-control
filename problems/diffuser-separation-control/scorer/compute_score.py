from __future__ import annotations

import json
from pathlib import Path

from grading.faults import AgentFault


def compute_score(workspace: str, trajectory=None, private=None):
    workspace_path = Path(workspace)

    candidates = [
        workspace_path / "output" / "diffuser_design.json",
        workspace_path / "diffuser_design.json",
        Path("/tmp/output/diffuser_design.json"),
    ]

    submission = next((p for p in candidates if p.exists()), None)

    if submission is None:
        raise AgentFault("missing diffuser_design.json")

    with submission.open("r", encoding="utf-8") as f:
        data = json.load(f)

    angle = float(data["half_angle_deg"])
    length = float(data["length_ratio"])
    inlet = float(data["inlet_extension_m"])

    if angle == 7.0 and length == 6.0 and inlet == 0.5:
        return {"score": 1.0}

    return {"score": 0.0}


if __name__ == "__main__":
    import sys

    workspace = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    print(json.dumps(compute_score(workspace), indent=2))