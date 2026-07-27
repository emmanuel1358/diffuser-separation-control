from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from grading import AgentFault, GraderFault
from grading.evaluation import (
    JsonArtifact,
    NumericField,
    RegularFileArtifact,
    RubricCriterion,
    RubricEvaluation,
    RubricTask,
    TrustedJson,
)
from grading.evaluation.plan import validate_serialized_plan
from grading.numeric import NumericContractError, safe_mean, safe_ratio


def _task(*, evaluate=None, required: bool = False, fixtures=None) -> RubricTask:
    return RubricTask(
        artifact=JsonArtifact(
            "design.json",
            required_keys=("value",),
            numeric_fields=(NumericField("value"),),
        ),
        criteria=(
            RubricCriterion(
                id="quality",
                weight=1.0,
                description="Candidate quality",
                required=required,
            ),
        ),
        fixtures=fixtures or {},
        evaluate=evaluate or (lambda context: {"quality": context.candidate["value"]}),
    )


def _workspace(tmp_path: Path, payload: bytes = b'{"value": 0.75}') -> Path:
    workspace = tmp_path / "output"
    workspace.mkdir()
    workspace.joinpath("design.json").write_bytes(payload)
    return workspace


def test_rubric_task_grades_typed_candidate(tmp_path: Path) -> None:
    grade = _task().grade(workspace=_workspace(tmp_path), private=tmp_path)

    assert grade.score() == pytest.approx(0.75)
    assert grade.subscores == {"quality": 0.75}
    assert grade.metadata["return_shape"] == "declarative_rubric"
    assert grade.metadata["evaluation"]["protocol"] == "declarative-rubric.v1"


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff\xfe{}",
        b"{",
        b"[]",
        ('{"value": %s}' % (10**400)).encode(),
        ('{"value": %s}' % ("[" * 2000 + "]" * 2000)).encode(),
    ],
)
def test_bad_json_content_is_agent_fault(tmp_path: Path, payload: bytes) -> None:
    with pytest.raises(AgentFault):
        _task().grade(workspace=_workspace(tmp_path, payload), private=tmp_path)


def test_symlink_artifact_is_agent_fault(tmp_path: Path) -> None:
    workspace = tmp_path / "output"
    workspace.mkdir()
    truth = tmp_path / "truth.json"
    truth.write_text('{"value": 1.0}')
    os.symlink(truth, workspace / "design.json")

    with pytest.raises(AgentFault):
        _task().grade(workspace=workspace, private=tmp_path)


def test_symlink_workspace_parent_is_agent_fault(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    (private / "design.json").write_text('{"value": 1.0}')
    workspace = tmp_path / "output"
    os.symlink(private, workspace)

    with pytest.raises(AgentFault):
        _task().grade(workspace=workspace, private=tmp_path)


def test_regular_file_artifact_returns_immutable_snapshot(tmp_path: Path) -> None:
    workspace = tmp_path / "output"
    workspace.mkdir()
    original = workspace / "policy.py"
    original.write_text("VALUE = 1\n")

    submitted = RegularFileArtifact("policy.py").load(workspace)
    original.write_text("VALUE = 2\n")

    assert submitted.original_path == original
    assert submitted.path != original
    assert submitted.path.read_text() == "VALUE = 1\n"


def test_unclassified_evaluator_exception_is_kept_zero(tmp_path: Path) -> None:
    def evaluate(_context):
        raise ZeroDivisionError("candidate-dependent denominator")

    grade = _task(evaluate=evaluate).grade(
        workspace=_workspace(tmp_path),
        private=tmp_path,
    )

    assert grade.score() == 0.0
    assert grade.env_internal_failure is False
    assert grade.metadata["critical_operator_alert"] is True
    assert "traceback" not in grade.metadata
    assert (
        "ZeroDivisionError"
        in grade.criterion_logs["unclassified_grader_crash"]["error_message"]
    )


def test_declared_zero_denominator_policy_becomes_agent_fault(tmp_path: Path) -> None:
    def evaluate(context):
        return {
            "quality": context.ratio(
                context.candidate["value"],
                0,
                label="candidate efficiency",
                zero="agent_fault",
            )
        }

    with pytest.raises(AgentFault, match="denominator is zero"):
        _task(evaluate=evaluate).grade(
            workspace=_workspace(tmp_path),
            private=tmp_path,
        )


def test_missing_trusted_fixture_is_grader_fault(tmp_path: Path) -> None:
    task = _task(fixtures={"truth": TrustedJson("truth.json")})

    with pytest.raises(GraderFault):
        task.grade(workspace=_workspace(tmp_path), private=tmp_path / "private")


def test_required_criterion_zeroes_headline(tmp_path: Path) -> None:
    task = _task(
        required=True,
        evaluate=lambda _context: RubricEvaluation(
            subscores={"quality": 0.4},
            metadata={"domain": "fixture"},
        ),
    )

    grade = task.grade(workspace=_workspace(tmp_path), private=tmp_path)

    assert grade.subscores == {"quality": 0.4}
    assert grade.score() == 0.0
    assert grade.metadata["domain"] == "fixture"


def test_rubric_receipt_redacts_replay_and_writes_private_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "evaluation-details.json"
    monkeypatch.setenv("LBX_EVALUATION_TRACE_PATH", str(trace))

    grade = _task().grade(workspace=_workspace(tmp_path), private=tmp_path)
    receipt = grade.metadata["evaluation"]
    private = json.loads(trace.read_text())

    assert "artifact_digest" not in receipt
    assert "nonce" not in receipt
    assert private["schema_version"] == "rubric-evaluation-trace.v1"
    assert len(private["replay"]["artifact_digest"]) == 64
    assert private["targets"]["quality"]["score"] == pytest.approx(0.75)


def test_evaluator_must_return_exact_criterion_set(tmp_path: Path) -> None:
    task = _task(evaluate=lambda _context: {"wrong": 1.0})

    with pytest.raises(GraderFault, match="criterion mismatch"):
        task.grade(workspace=_workspace(tmp_path), private=tmp_path)


def test_shared_numeric_helpers_require_explicit_empty_zero_policies() -> None:
    assert safe_ratio(1, 0, zero="zero") == 0.0
    assert safe_mean([], empty="zero") == 0.0
    with pytest.raises(NumericContractError):
        safe_ratio(1, 0)
    with pytest.raises(NumericContractError):
        safe_mean([])


def test_rubric_spec_is_stable_and_json_serializable() -> None:
    first = _task()
    second = _task()

    assert first.spec_sha256 == second.spec_sha256
    assert json.loads(json.dumps(first.spec_dict()))["protocol"] == (
        "declarative-rubric.v1"
    )

    plan = first.evaluation_plan
    payload = {**plan.to_dict(), "plan_sha256": plan.sha256}
    assert payload["schema_version"] == "evaluation-plan.v2"
    assert validate_serialized_plan(payload) == plan.sha256


def _write_rubric_problem(problem_dir: Path) -> None:
    scorer = problem_dir / "scorer"
    scorer.mkdir(parents=True)
    scorer.joinpath("compute_score.py").write_text(
        "from grading.evaluation import JsonArtifact, NumericField, "
        "RubricCriterion, RubricTask\n"
        "\n"
        "def evaluate(context):\n"
        "    return {'quality': float(context.candidate['value'])}\n"
        "\n"
        "TASK = RubricTask(\n"
        "    artifact=JsonArtifact(\n"
        "        'design.json',\n"
        "        required_keys=('value',),\n"
        "        numeric_fields=(NumericField('value'),),\n"
        "    ),\n"
        "    criteria=(RubricCriterion('quality', weight=1.0),),\n"
        "    evaluate=evaluate,\n"
        ")\n",
        encoding="utf-8",
    )


def test_refresh_and_check_evaluation_plan(tmp_path: Path) -> None:
    from grading.evaluation.plan import check_evaluation_plan, refresh_evaluation_plan

    problem = tmp_path / "rubric-task"
    _write_rubric_problem(problem)

    first = refresh_evaluation_plan(problem)
    assert first.status == "written"
    assert first.path.is_file()

    second = refresh_evaluation_plan(problem)
    assert second.status == "unchanged"
    assert check_evaluation_plan(problem).status == "unchanged"

    first.path.write_text("{}\n", encoding="utf-8")
    assert check_evaluation_plan(problem).status == "stale"
    assert refresh_evaluation_plan(problem).status == "written"

    first.path.unlink()
    blocked = check_evaluation_plan(problem)
    assert blocked.status == "missing"
    assert "evaluation.plan.json" in blocked.message
    assert not first.path.exists()
