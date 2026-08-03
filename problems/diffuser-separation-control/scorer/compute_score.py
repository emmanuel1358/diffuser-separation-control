from __future__ import annotations

from pathlib import Path

from grading.faults import AgentFault
from grading.helpers import load_json


def compute_score(workspace: str, trajectory=None, private=None):
    submission_path = Path(workspace) / "output" / "diffuser_design.json"

    # load_json returns a dict directly; it also validates the path safely
    data = load_json(submission_path)

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
    import json
    import sys

    workspace = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    print(json.dumps(compute_score(workspace), indent=2))