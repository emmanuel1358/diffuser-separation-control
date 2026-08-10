#!/usr/bin/env bash
set -euo pipefail

PROBLEM_DIR="${PROBLEM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROBLEM_DIR
export PYTHONDONTWRITEBYTECODE=1

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  echo "No Python interpreter available"
  exit 1
fi

for path in \
  README.md \
  instruction.md \
  metadata.json \
  task.toml \
  environment/Dockerfile \
  starter/normalizer.py \
  starter/tests/test_normalizer.py \
  scorer/compute_score.py \
  scorer/evaluation.plan.json \
  scorer/data/candidate_driver.py \
  scorer/data/hidden_cases.json \
  solution/files/normalizer.py \
  solution/solve.sh \
  baselines/noop.sh; do
  test -s "$PROBLEM_DIR/$path" || {
    echo "Missing or empty file: $path"
    exit 1
  }
done

"$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
from pathlib import Path

from alignerr_plugin.utils import load_task_toml
from grading.evaluation import RubricTask
from grading.evaluation.plan import check_evaluation_plan, validate_serialized_plan

problem = Path(os.environ["PROBLEM_DIR"])
task = load_task_toml(problem)
assert task.environment.allow_internet is False
assert task.agent.timeout_sec == 7200
assert task.agent.user == "agent"
assert task.verifier.timeout_sec == 7200
assert task.verifier.user == "root"
assert task.difficulty.task_type == "software_engineering"
assert task.difficulty.domain == "repo_debugging"
assert task.difficulty.reward_type == "multi_deterministic_rubrics"
assert [output.path for output in task.outputs] == ["/tmp/output/repo"]

metadata = json.loads((problem / "metadata.json").read_text(encoding="utf-8"))
assert metadata["benchmark"] == "taiga_task"
assert metadata["problem_data"]["instance_id"]

scorer_path = problem / "scorer" / "compute_score.py"
spec = importlib.util.spec_from_file_location("starter_software_grader", scorer_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert isinstance(module.TASK, RubricTask)
assert not hasattr(module, "compute_score")

plan = json.loads(
    (problem / "scorer" / "evaluation.plan.json").read_text(encoding="utf-8")
)
assert validate_serialized_plan(plan) == module.TASK.evaluation_plan.sha256
assert check_evaluation_plan(problem).status == "unchanged"

dockerfile = (problem / "environment" / "Dockerfile").read_text(encoding="utf-8")
assert "ARG AGENT_UID=1000" in dockerfile
assert "chmod 0700" in dockerfile
assert "chmod 0600" in dockerfile

scorer = scorer_path.read_text(encoding="utf-8")
assert "WorkspaceArtifact" in scorer
assert "run_candidate_suite" in scorer

for path in problem.rglob("*.py"):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")

print("static schema and evaluation-plan checks passed")
PY

(
  cd "$PROBLEM_DIR/starter"
  "$PYTHON_BIN" -m unittest discover -s tests -v
)

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
ORACLE_OUT="$TMP_DIR/oracle"
NOOP_OUT="$TMP_DIR/noop"

LBT_OUTPUT_DIR="$ORACLE_OUT" bash "$PROBLEM_DIR/solution/solve.sh" >/dev/null
LBT_OUTPUT_DIR="$NOOP_OUT" bash "$PROBLEM_DIR/baselines/noop.sh" >/dev/null

run_hidden_driver() {
  local repository="$1"
  (
    cd "$repository"
    "$PYTHON_BIN" -I -B "$PROBLEM_DIR/scorer/data/candidate_driver.py" \
      < "$PROBLEM_DIR/scorer/data/hidden_cases.json"
  )
}

ORACLE_RESULT="$(run_hidden_driver "$ORACLE_OUT/repo")"
NOOP_RESULT="$(run_hidden_driver "$NOOP_OUT/repo")"
export ORACLE_RESULT NOOP_RESULT

"$PYTHON_BIN" - <<'PY'
import json
import os

oracle = json.loads(os.environ["ORACLE_RESULT"])
noop = json.loads(os.environ["NOOP_RESULT"])

assert oracle["passed"] == oracle["total"], oracle
assert noop["passed"] < noop["total"], noop

oracle_grade = float(oracle["passed"] == oracle["total"])
noop_grade = float(noop["passed"] == noop["total"])
assert oracle_grade == 1.0
assert noop_grade == 0.0

print(f"oracle host grade: {oracle_grade:.1f}")
print(f"no-op host grade: {noop_grade:.1f}")
PY

echo "SOFTWARE ENGINEERING STARTER TESTS PASSED"
