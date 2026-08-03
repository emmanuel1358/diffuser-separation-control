
from __future__ import annotations

import json
from pathlib import Path

from grading.faults import AgentFault
from grading.helpers import load_submission_or_fault


def compute_score(workspace: str, trajectory=None, private=None):
    filename = Path("output") / "diffuser_design.json"

    # Safe submission loader provided by the harness
    text = load_submission_or_fault(workspace, filename)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AgentFault(f"invalid JSON: {e}") from e

    try:
        angle = float(data["half_angle_deg"])
        length = float(data["length_ratio"])
        inlet = float(data["inlet_extension_m"])
    except KeyError as e:
        raise AgentFault(f"missing field: {e}")
    except (TypeError, ValueError) as e:
        raise AgentFault(f"invalid numeric value: {e}")

    if angle == 7.0 and length == 6.0 and inlet == 0.5:
        return {"score": 1.0}

    return {"score": 0.0}


if __name__ == "__main__":
    import sys

    workspace = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    print(json.dumps(compute_score(workspace), indent=2))
