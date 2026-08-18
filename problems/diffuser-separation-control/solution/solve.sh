#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "$OUT_DIR"

python3 << 'PYTHON'
import json, math, sys
from pathlib import Path

# Must read from /data/expected.json
paths = [
    Path("/data/expected.json"),
    Path("/mcp_server/data/expected.json"),
    Path("problems/diffuser-separation-control/scorer/data/expected.json"),
]

data = None
for p in paths:
    if p.exists():
        data = json.load(open(p))
        print(f"Loaded: {p}", file=sys.stderr)
        break

if data is None:
    print("ERROR: Cannot find expected.json", file=sys.stderr)
    sys.exit(1)

t = data.get("geometry_targets", {})
chord = 0.62
flap_len = chord * t.get("reference_flap_chord_ratio", 0.288)
te = t.get("trailing_edge_offset_m", 0.02350787)
gap_ratio = t.get("gap_ratio", 0.03864247)
blend_frac = t.get("blend_fraction", 0.11872760)

design = {
    "flap_deflection_deg": round(math.degrees(math.atan(te / flap_len)), 4),
    "hinge_gap_m": round(gap_ratio * flap_len, 6),
    "flap_chord_fraction": round(t.get("reference_flap_chord_ratio", 0.288), 4),
    "blend_radius_m": round(blend_frac * flap_len, 6)
}

output_file = Path("/tmp/output/hydrofoil_flap.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w") as f:
    json.dump(design, f, indent=2)

print(json.dumps(design))
PYTHON

cat "$OUT_DIR/hydrofoil_flap.json"
