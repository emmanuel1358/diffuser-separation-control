#!/usr/bin/env bash
set -euo pipefail

PROBLEM_DIR="${PROBLEM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROBLEM_DIR

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then PYTHON_BIN="python"
  else echo "No python available"; exit 1; fi
fi

for f in \
  instruction.md \
  task.toml \
  metadata.json \
  data/building_description.md \
  data/design_schema.json \
  data/public_model_summary.py \
  data/public_isolation_model.py \
  data/public_analysis_envelope.json \
  data/graded_case_sampling_policy.json \
  data/isolation_starter.json \
  scorer/compute_score.py \
  scorer/data/hidden_cases.json \
  solution/solve.sh \
  solution/oracle_search.py \
  baselines/naive.sh \
  baselines/solve.sh \
  environment/Dockerfile; do
  test -s "$PROBLEM_DIR/$f" || { echo "Missing or empty file: $f"; exit 1; }
done

grep -q "/tmp/output/isolation_design.json" "$PROBLEM_DIR/instruction.md"
! grep -q "/tmp/output/solver_evidence.json" "$PROBLEM_DIR/instruction.md"
! grep -q "public_isolation_probe.py" "$PROBLEM_DIR/instruction.md"
! grep -q "OpenSees\|OpenSeesPy\|opensees\|ops\." "$PROBLEM_DIR/instruction.md"
grep -q '/mcp_server/public_data/' "$PROBLEM_DIR/environment/Dockerfile"
grep -q "openseespy import ok\|openseespy.opensees" "$PROBLEM_DIR/environment/Dockerfile"
grep -q 'domain = "structural_mechanics"' "$PROBLEM_DIR/task.toml"
grep -q "rendered = false" "$PROBLEM_DIR/task.toml"
grep -q "base-isolation\|base isolation\|isolation system" "$PROBLEM_DIR/instruction.md"

if grep -R "rendering.mp4\|rendering.png\|ffprobe\|ffmpeg\|wx5890kie\|seismic_retrofit" \
  "$PROBLEM_DIR/instruction.md" "$PROBLEM_DIR/task.toml" "$PROBLEM_DIR/data" \
  "$PROBLEM_DIR/scorer" "$PROBLEM_DIR/solution" 2>/dev/null; then
  echo "Found stale/render reference."
  exit 1
fi

${PYTHON_BIN} -m json.tool "$PROBLEM_DIR/data/design_schema.json" >/dev/null
${PYTHON_BIN} -m json.tool "$PROBLEM_DIR/data/public_analysis_envelope.json" >/dev/null
${PYTHON_BIN} -m json.tool "$PROBLEM_DIR/data/graded_case_sampling_policy.json" >/dev/null
${PYTHON_BIN} -m json.tool "$PROBLEM_DIR/data/isolation_starter.json" >/dev/null
${PYTHON_BIN} -m json.tool "$PROBLEM_DIR/scorer/data/hidden_cases.json" >/dev/null
${PYTHON_BIN} -m json.tool "$PROBLEM_DIR/metadata.json" >/dev/null
# py_compile only (does not execute imports, so no numpy/openseespy needed on host)
${PYTHON_BIN} -m py_compile "$PROBLEM_DIR/data/public_isolation_model.py"
${PYTHON_BIN} -m py_compile "$PROBLEM_DIR/data/public_model_summary.py"
${PYTHON_BIN} -m py_compile "$PROBLEM_DIR/solution/oracle_search.py"

# Regression: a non-converged response has residual=None.  Worst-case
# aggregation must return a finite failure envelope, not raise float(None).
${PYTHON_BIN} - <<'PY'
import math, os, sys
from pathlib import Path

problem = Path(os.environ["PROBLEM_DIR"])
sys.path.insert(0, str(problem / "data"))
try:
    import public_isolation_model as model
except ModuleNotFoundError as exc:
    if exc.name == "numpy":
        print("skipping model import regression: numpy unavailable on host")
        sys.exit(0)
    raise

original_run_case = model.run_case
responses = iter([
    {
        "converged": True,
        "peak_isolator_disp_in": 4.0,
        "peak_interstory_drift_ratio": 0.001,
        "peak_floor_acceleration_g": 0.2,
        "peak_base_shear_coeff": 0.1,
        "residual_isolator_disp_in": 0.5,
    },
    {
        "converged": False,
        "peak_isolator_disp_in": 6.0,
        "peak_interstory_drift_ratio": 0.002,
        "peak_floor_acceleration_g": 0.3,
        "peak_base_shear_coeff": 0.2,
        "residual_isolator_disp_in": None,
    },
])
try:
    model.run_case = lambda design, case: next(responses)
    result = model.evaluate_design({}, [{}, {}])
finally:
    model.run_case = original_run_case

assert result["all_converged"] is False
assert result["worst_case"]["residual_isolator_disp_in"] == model.FAILED_METRIC_VALUE
assert all(math.isfinite(v) for v in result["worst_case"].values())

empty = model.evaluate_design({}, [])
assert empty["all_converged"] is False
assert all(v == model.FAILED_METRIC_VALUE for v in empty["worst_case"].values())
print("non-convergence aggregation regression passed")
PY

${PYTHON_BIN} - <<'PY'
import importlib.util, json, os, pathlib, sys, tempfile

problem = pathlib.Path(os.environ["PROBLEM_DIR"])

# compute_score imports only the stdlib at module scope -> safe to import on host.
spec = importlib.util.spec_from_file_location("compute_score", problem / "scorer" / "compute_score.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

hidden = json.loads((problem / "scorer" / "data" / "hidden_cases.json").read_text())
assert len(hidden["hidden_cases"]) == 6, "expected six hidden cases"
weights = module.numeric_weights(hidden)
assert abs(sum(weights.values()) - 1.0) < 1e-9, "normalized weights must sum to 1"

# Agent-caused failures raise AgentFault (runtime keeps a clean 0.0), no solver needed.
def expect_agent_fault(ws, msg):
    try:
        module.compute_score(ws, private=problem / "scorer" / "data")
    except module.AgentFault:
        return
    raise AssertionError(msg)

with tempfile.TemporaryDirectory() as tmp:
    ws = pathlib.Path(tmp)
    expect_agent_fault(ws, "missing output must raise AgentFault")
    (ws / "isolation_design.json").write_text(
        json.dumps({"isolation_system": {"Qd_kip": 320.0, "Kd_kip_per_in": 20.0, "Dy_in": 0.6}})
    )
    result = module.compute_score(ws, private=problem / "scorer" / "data")
    assert "score" in result, "valid design should be scored directly without solver evidence"

# Design validation guards.
assert module.validate_design({"isolation_system": {"Qd_kip": 320.0, "Kd_kip_per_in": 20.0, "Dy_in": 0.6}})["errors"] == []
assert module.validate_design({"isolation_system": {"Qd_kip": 5.0, "Kd_kip_per_in": 20.0, "Dy_in": 0.6}})["errors"], "out-of-range Qd must fail"
assert module.validate_design({"isolation_system": {"Qd_kip": True, "Kd_kip_per_in": 20.0, "Dy_in": 0.6}})["errors"], "boolean must fail"
assert module.validate_design({"isolation_system": {"Qd_kip": 320.0, "Kd_kip_per_in": 20.0}})["errors"], "missing key must fail"
assert module.validate_design({"isolation_system": {"Qd_kip": 320.0, "Kd_kip_per_in": 20.0, "Dy_in": 0.6, "x": 1}})["errors"], "extra key must fail"
print("scorer contract checks passed")
PY

# Public/hidden scoring-config alignment (disclosed envelope must match the grader).
${PYTHON_BIN} - <<'PY'
import json, os
from pathlib import Path
problem = Path(os.environ["PROBLEM_DIR"])
env = json.loads((problem / "data" / "public_analysis_envelope.json").read_text())
hid = json.loads((problem / "scorer" / "data" / "hidden_cases.json").read_text())
hs = hid["scoring"]; es = env["scoring"]
for k, v in hs["targets"].items():
    assert abs(float(env["performance_targets"][k]) - float(v)) < 1e-12, f"target mismatch {k}"
assert abs(float(env["moat_capacity_in"]) - float(hs["moat_capacity_in"])) < 1e-12
assert abs(float(env["recentering_capacity_in"]) - float(hs["recentering_capacity_in"])) < 1e-12
assert es["lower_ratio_curve"] == hs["lower_ratio_curve"], "lower curve must match"
assert es["moat_gate_curve"] == hs["moat_gate_curve"], "moat curve must match"
assert es["recentering_gate_curve"] == hs["recentering_gate_curve"], "recentering curve must match"
assert es["weights"] == hs["weights"], "weights must match"
assert abs(float(es["final_score_exponent"]) - float(hs["final_score_exponent"])) < 1e-12
# Hidden case parameters must lie inside the disclosed band.
band = env["ground_motion_band"]
pmin, pmax = band["pga_g_range"]; tmin, tmax = band["predominant_period_tp_sec_range"]; dmin, dmax = band["duration_sec_range"]
for c in hid["hidden_cases"]:
    assert pmin <= c["pga_g"] <= pmax, f"pga out of band: {c}"
    assert tmin <= c["tp_sec"] <= tmax, f"tp out of band: {c}"
    assert dmin <= c["duration_sec"] <= dmax, f"duration out of band: {c}"
    smin, smax = band["phase_seed_integer_range"]
    assert smin <= c["phase_seed"] <= smax, f"phase seed out of band: {c}"
assert band["hidden_case_count"] == len(hid["hidden_cases"])
# Hidden model markers.
modeltext = (problem / "data" / "public_isolation_model.py").read_text()
for needle in ("ops.model", "ops.node", "ops.element", "ops.eigen", "ops.analyze", "UniformExcitation", "ElasticPP"):
    assert needle in modeltext, f"model missing OpenSeesPy marker: {needle}"
scorer = (problem / "scorer" / "compute_score.py").read_text()
assert "solver_evidence.json" not in scorer
assert "public_data" in scorer, "grader must reach the root-readable public copy"
# No exec/eval of submitted code in the grader.
for bad in ("exec(", "eval(", "runpy.run_path", "spec.loader.exec_module"):
    assert bad not in scorer, f"grader must not use {bad}"
print("public/hidden alignment checks passed")
PY

echo "ALL TESTS PASSED"
