#!/usr/bin/env bash
# _rev: 2026-06-20.b7prc04t3_batch12r
set -euo pipefail
PROBLEM_DIR="${PROBLEM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROBLEM_DIR

required_json=(
  "$PROBLEM_DIR/data/hydrofoil_flap_template.json"
  "$PROBLEM_DIR/data/public_operating_envelope.json"
  "$PROBLEM_DIR/data/public_baseline_summary.json"
  "$PROBLEM_DIR/data/public_calibration_samples.json"
  "$PROBLEM_DIR/data/public_transfer_guidance.json"
  "$PROBLEM_DIR/scorer/data/hidden_conditions.json"
  "$PROBLEM_DIR/scorer/data/expected.json"
)
for path in "${required_json[@]}"; do
  test -s "$path"
  python3 -m json.tool "$path" >/dev/null
done

python3 -m py_compile \
  "$PROBLEM_DIR/scorer/compute_score.py" \
  "$PROBLEM_DIR/scorer/openfoam_case.py"

bash -n "$PROBLEM_DIR/solution/solve.sh"
bash -n "$PROBLEM_DIR/solution/render.sh"
bash -n "$PROBLEM_DIR/baselines/naive.sh"

python3 - <<'PY'
import json
import math
import os
from pathlib import Path
problem = Path(os.environ["PROBLEM_DIR"])
task = (problem / "task.toml").read_text()
assert "/tmp/output/hydrofoil_flap.json" in task, "missing task output declaration"
for forbidden in ("/tmp/output/public_openfoam_metrics.json", "/tmp/output/openfoam_run_manifest.json"):
    assert forbidden not in task, f"solver-provenance output must not be agent-facing: {forbidden}"
public = json.loads((problem / "data/public_operating_envelope.json").read_text())
assert "public_validation_case" in public, "public validation case missing from public envelope"
validation_case = public["public_validation_case"]
for key in ("public_speed_m_per_s", "public_trim_angle_deg", "public_submergence_factor", "public_reynolds_number"):
    assert key in validation_case, f"public validation case missing public-prefixed field {key}"
for forbidden in ("public_openfoam_probe_case", "trim_bias_deg", "blockMesh", "checkMesh", "simpleFoam", "OpenFOAM"):
    assert forbidden not in json.dumps(public), f"public envelope leaks solver/probe wording: {forbidden}"
transfer = json.loads((problem / "data/public_transfer_guidance.json").read_text())
assert transfer.get("public_offdesign_cases"), "public transfer bridge cases missing"
for bridge_case in transfer["public_offdesign_cases"]:
    assert "public_operating_point" in bridge_case, "bridge case must use public_operating_point wrapper"
    assert "trim_bias_deg" not in bridge_case, "public bridge case must not reuse private-case trim key"
assert transfer.get("broad_performance_bands"), "public transfer broad performance bands missing"
conditions = json.loads((problem / "scorer/data/hidden_conditions.json").read_text())
of = conditions["openfoam"]
assert of.get("total_timeout_sec", 9999) <= 600, "total OpenFOAM timeout budget exceeds 600s target"
per = int(of.get("per_case_timeout_sec", 9999))
block = int(of.get("block_mesh_timeout_sec", 9999))
solver = int(of.get("simple_foam_timeout_sec", 9999))
assert block + solver <= per, "blockMesh + simpleFoam timeouts exceed per-case budget"
assert per * len(conditions.get("cases", [])) <= 600, "worst-case hidden OpenFOAM timeouts exceed 600s budget"
expected = json.loads((problem / "scorer/data/expected.json").read_text())
assert "official_digest_sha256" not in expected, "digest shortcut must not be present"
score_text = (problem / "scorer/compute_score.py").read_text()
for token in ("public_validation_case", "score_resolution_rebalanced"):
    assert token in score_text, f"compute_score missing expected verification token {token}"
for forbidden in ("public_openfoam_required_artifact_gate", "public_openfoam_artifact_exists", "public_openfoam_solver_evidence"):
    assert forbidden not in score_text, f"solver evidence scoring must not remain: {forbidden}"
for forbidden in ("non_oracle_public_ceiling", "non_oracle_score_ceiling"):
    assert forbidden not in score_text, f"hard non-oracle score ceiling must not remain: {forbidden}"

calib = json.loads((problem / "data/public_calibration_samples.json").read_text())
assert "metric_windows" in calib, "public metric windows missing"
assert calib["metric_windows"]["lift_coefficient"]["floor_weight"] > 0.0, "lift public-window floor missing"
pack = public["packaging_context"]
assert pack["target_trailing_edge_offset_band_m"][0] <= 0.0235 <= pack["target_trailing_edge_offset_band_m"][1], "target trailing-edge band must contain the high-credit reference geometry"
assert pack["target_blockage_fraction_band"][0] <= 0.147 <= pack["target_blockage_fraction_band"][1], "target blockage band must contain the high-credit reference geometry"
import ast
module = ast.parse(score_text)
weights = None
for node in module.body:
    if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "WEIGHTS":
        weights = ast.literal_eval(node.value)
        break
assert isinstance(weights, dict), "WEIGHTS dictionary missing"
exact_total = sum(weights[k] for k in ("nominal_lift_window", "hidden_lift_window", "nominal_drag_fit", "hidden_drag_fit"))
assert exact_total <= 0.25 + 1e-9, f"exact lift/drag target weight too high: {exact_total}"
assert abs(sum(weights.values()) - 1.0) <= 1e-9, "rubric weights must sum to 1.0"

PY

# Check only active runtime/proof paths for stale render/video dependencies. Do not
# scan tests/test.sh itself, because the grep pattern below intentionally contains
# the forbidden tokens and would otherwise make this test self-fail.
render_ref_file="$(mktemp)"
render_scan_paths=()
for path in   "$PROBLEM_DIR/.alignerr"   "$PROBLEM_DIR/task.toml"   "$PROBLEM_DIR/scorer"   "$PROBLEM_DIR/solution"   "$PROBLEM_DIR/data"   "$PROBLEM_DIR/baselines"; do
  if [ -e "$path" ]; then
    render_scan_paths+=("$path")
  fi
done
if [ "${#render_scan_paths[@]}" -gt 0 ] && grep -RInE "rendering\.mp4|rendering\.png|ffprobe|ffmpeg" "${render_scan_paths[@]}" >"$render_ref_file"; then
  cat "$render_ref_file"
  echo "unexpected active render/video references" >&2
  exit 1
fi
rm -f "$render_ref_file"
