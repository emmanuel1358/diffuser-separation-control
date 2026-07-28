from __future__ import annotations

import json
from pathlib import Path


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score(output_dir: str) -> dict:
    output_path = Path(output_dir) / "diffuser_design.json"

    if not output_path.exists():
        return {"score": 0.0, "reason": "missing diffuser_design.json"}

    try:
        data = json.loads(output_path.read_text())
    except Exception as exc:
        return {"score": 0.0, "reason": f"invalid json: {exc}"}

    try:
        angle = float(data["half_angle_deg"])
        length_ratio = float(data["length_ratio"])
        inlet_ext = float(data["inlet_extension_m"])
    except Exception as exc:
        return {"score": 0.0, "reason": f"missing field: {exc}"}

    if not (3.0 <= angle <= 12.0):
        return {"score": 0.0, "reason": "half_angle_deg out of bounds"}

    if not (3.0 <= length_ratio <= 8.0):
        return {"score": 0.0, "reason": "length_ratio out of bounds"}

    if not (0.1 <= inlet_ext <= 1.0):
        return {"score": 0.0, "reason": "inlet_extension_m out of bounds"}

    angle_score = clamp(1.0 - abs(angle - 7.0) / 4.0, 0.0, 1.0)
    length_score = clamp((length_ratio - 4.0) / 2.0, 0.0, 1.0)
    inlet_score = clamp(1.0 - abs(inlet_ext - 0.5) / 0.4, 0.0, 1.0)

    final_score = round(
        0.4 * angle_score + 0.4 * length_score + 0.2 * inlet_score,
        4,
    )

    return {
        "score": final_score,
        "subscores": {
            "angle_quality": round(angle_score, 4),
            "length_quality": round(length_score, 4),
            "inlet_extension_quality": round(inlet_score, 4),
        },
    }


# Harness-compatible signature
def compute_score(
    workspace: str,
    trajectory=None,
    private=None,
) -> dict:
    return _score(workspace)


# Optional backward-compatible alias
def grade(output_dir: str) -> dict:
    return _score(output_dir)


if __name__ == "__main__":
    import sys

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/output"
    print(json.dumps(_score(out_dir), indent=2))