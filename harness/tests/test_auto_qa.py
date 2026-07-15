from __future__ import annotations

import asyncio
import json

from lbx_rl_tasks_harness import auto_qa, runner
from lbx_rl_tasks_harness.message_text import message_text
from lbx_rl_tasks_harness.models import HarnessProblem, OutputSpec


class SequenceModel:
    """Returns queued completions in order; records how many times it was called."""

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.calls = 0

    async def ainvoke(self, prompt: str):
        self.calls += 1
        content = self._contents[min(self.calls - 1, len(self._contents) - 1)]

        class Message:
            pass

        message = Message()
        message.content = content
        return message


class FakeModel:
    prompt = ""
    reasoning = None

    async def ainvoke(self, prompt: str):
        self.prompt = prompt

        class Message:
            content = """{
              "overall_assessment": "needs_changes",
              "summary": "Task is close but needs stronger rollout checks.",
              "is_solvable": {"value": true, "evidence": "The prompt declares /tmp/output/model.xml."},
              "scientifically_correct": {"value": false, "evidence": "No robustness perturbations are present."},
              "confidence": "high",
              "blocking_issues": ["Missing robustness checks in scorer/compute_score.py"],
              "non_blocking_feedback": ["Add a naive baseline."],
              "checks": [
                {
                  "check": "Prompt clarity",
                  "status": "pass",
                  "reason": "Deliverables are explicit.",
                  "recommendation": ""
                }
              ]
            }"""

        return Message()


def _problem(tmp_path):
    repo = tmp_path
    problem_dir = repo / "problems" / "demo"
    (repo / "docs").mkdir(parents=True)
    (repo / "project_guidelines").mkdir()
    (repo / "docs" / "RUBRIC_GUIDANCE.md").write_text("Rubric guidance")
    (repo / "docs" / "GRADING.md").write_text("Grading guidance")
    (repo / "project_guidelines" / "mujoco_environments.md").write_text(
        "MuJoCo guidance"
    )
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (problem_dir / "scorer").mkdir(parents=True)
    (problem_dir / "instruction.md").write_text("Write /tmp/output/model.xml")
    (problem_dir / "task.toml").write_text("[task]\nname='demo'\n")
    (problem_dir / "metadata.json").write_text(
        '{"benchmark": "taiga_task", "problem_data": {"instance_id": "demo"}}'
    )
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private): return 1.0"
    )
    return HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/model.xml")],
        source_problem_dir=problem_dir,
    )


def _cfd_problem(tmp_path):
    repo = tmp_path
    problem_dir = repo / "problems" / "demo-cfd"
    (repo / "docs").mkdir(parents=True)
    (repo / "project_guidelines" / "cfd").mkdir(parents=True)
    (repo / "docs" / "RUBRIC_GUIDANCE.md").write_text("Rubric guidance")
    (repo / "docs" / "GRADING.md").write_text("Grading guidance")
    (repo / "project_guidelines" / "cfd" / "cfd_environments.md").write_text(
        "CFD guidance: keep the agent-facing surface solver-agnostic."
    )
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (problem_dir / "scorer").mkdir(parents=True)
    (problem_dir / "data").mkdir()
    (problem_dir / "instruction.md").write_text(
        "Run OpenFOAM with blockMesh and checkMesh before writing /tmp/output/design.json."
    )
    (problem_dir / "task.toml").write_text(
        "[task]\nname='demo-cfd'\n[difficulty]\ntask_type='cfd'\n"
    )
    (problem_dir / "metadata.json").write_text(
        '{"benchmark": "taiga_task", "problem_data": {"instance_id": "demo-cfd"}}'
    )
    (problem_dir / "data" / "openfoam_probe.py").write_text("blockMesh checkMesh")
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private): return 1.0"
    )
    return HarnessProblem(
        id="demo-cfd",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/design.json")],
        source_problem_dir=problem_dir,
    )


def _structures_problem(tmp_path):
    repo = tmp_path
    problem_dir = repo / "problems" / "demo-structures"
    (repo / "docs").mkdir(parents=True)
    (repo / "project_guidelines" / "strctural_engineering").mkdir(parents=True)
    (repo / "docs" / "RUBRIC_GUIDANCE.md").write_text("Rubric guidance")
    (repo / "docs" / "GRADING.md").write_text("Grading guidance")
    (
        repo
        / "project_guidelines"
        / "strctural_engineering"
        / "STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md"
    ).write_text("Structural guidance keeps instruction.md solver agnostic.")
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (problem_dir / "scorer").mkdir(parents=True)
    (problem_dir / "data").mkdir()
    (problem_dir / "instruction.md").write_text(
        "Design a retrofit before writing /tmp/output/retrofit_design.json."
    )
    (problem_dir / "task.toml").write_text(
        "[task]\nname='demo-structures'\n[difficulty]\ntask_type='structures'\n"
    )
    (problem_dir / "metadata.json").write_text(
        '{"benchmark": "taiga_task", "problem_data": {"instance_id": "demo-structures"}}'
    )
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private): return 1.0"
    )
    return HarnessProblem(
        id="demo-structures",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/retrofit_design.json")],
        source_problem_dir=problem_dir,
    )


def test_run_auto_qa_check_uses_autoqa_env_and_normalizes(monkeypatch, tmp_path):
    fake_model = FakeModel()
    monkeypatch.setenv("LBX_RL_AUTOQA_MODEL", "openai:gpt-5.5")
    monkeypatch.setenv("LBX_RL_AUTOQA_REASONING_EFFORT", "xhigh")
    monkeypatch.setenv("LBX_RL_HARNESS_MODEL", "claude-opus-4-7")
    monkeypatch.setattr(auto_qa, "has_key_for_model", lambda _model: True)
    monkeypatch.setattr(
        auto_qa,
        "_build_autoqa_model",
        lambda model, *, reasoning_effort: fake_model,
    )

    result = asyncio.run(
        auto_qa.run_auto_qa_check(
            _problem(tmp_path),
            proof={"harness_result": {"score": 0.2}},
            grade_payload={"score": 0.2},
            rubric_quality={"status": "completed", "checks": []},
        )
    )

    assert result["status"] == "completed"
    assert result["model"] == "openai:gpt-5.5"
    assert result["reasoning_effort"] == "xhigh"
    assert result["overall_assessment"] == "needs_changes"
    assert result["is_solvable"]["value"] is True
    assert result["scientifically_correct"]["value"] is False
    assert {check["check"] for check in result["checks"]} == set(auto_qa.AUTOQA_CHECKS)
    assert "SUBMITTED BUILD PROOF" in fake_model.prompt
    assert "SUBMITTED RUBRIC QUALITY REVIEW" in fake_model.prompt
    assert "You are an expert task QA reviewer" in fake_model.prompt
    assert "You are an expert MuJoCo task QA reviewer" not in fake_model.prompt
    assert "MuJoCo physics validity" in fake_model.prompt
    assert "Solver-agnostic surface leakage" not in fake_model.prompt


def test_autoqa_prompt_for_cfd_includes_solver_leak_check(
    tmp_path,
):
    prompt = auto_qa._autoqa_prompt(
        _cfd_problem(tmp_path),
        proof={},
        grade_payload={"score": 0.3},
        rubric_quality={"status": "completed", "checks": []},
    )

    assert "Task type: cfd" in prompt
    assert "solver-agnostic" in prompt
    assert "CFD physics validity" in prompt
    assert "MuJoCo physics validity" not in prompt
    assert "invalid CFD physics" in prompt
    assert "invalid MuJoCo physics" not in prompt
    assert "Solver-agnostic surface leakage" in prompt
    assert "Numerical solver grader quality" not in prompt
    assert "grader QA reviewer" not in prompt
    assert "inspect ONLY instruction.md" in prompt
    assert "mentions a solver" in prompt
    assert "Physics-grounded scoring details are allowed" in prompt
    assert "Do not use public/preloaded data" in prompt
    assert "agent-facing surfaces" not in prompt
    assert "cfd_environments.md" in prompt
    assert "Solver usage contract" not in prompt
    assert "Public solver evidence path" not in prompt
    assert "must require the agent to run a public CFD solver diagnostic" not in prompt


def test_autoqa_prompt_for_structures_includes_solver_leak_check(tmp_path):
    prompt = auto_qa._autoqa_prompt(
        _structures_problem(tmp_path),
        proof={},
        grade_payload={"score": 0.3},
        rubric_quality={"status": "completed", "checks": []},
    )

    assert "Task type: structures" in prompt
    assert "Structural physics validity" in prompt
    assert "Solver-agnostic surface leakage" in prompt
    assert "Numerical solver grader quality" not in prompt


def test_autoqa_checks_add_solver_leak_only_for_numerical_solvers():
    assert "Solver-agnostic surface leakage" not in auto_qa._autoqa_checks("")
    assert "Solver-agnostic surface leakage" not in auto_qa._autoqa_checks("mujoco")
    assert "Solver-agnostic surface leakage" in auto_qa._autoqa_checks("cfd")
    assert "Solver-agnostic surface leakage" in auto_qa._autoqa_checks("structures")
    assert "Numerical solver grader quality" not in auto_qa._autoqa_checks("cfd")
    assert "Numerical solver grader quality" not in auto_qa._autoqa_checks("structures")


def test_build_autoqa_model_sets_openai_reasoning(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "langchain_openai.ChatOpenAI",
        FakeChatOpenAI,
    )

    auto_qa._build_autoqa_model("openai:gpt-5.5", reasoning_effort="xhigh")

    assert captured["model"] == "gpt-5.5"
    assert captured["reasoning"] == {"effort": "xhigh"}
    assert captured["max_tokens"] is None


def test_run_auto_qa_check_skips_when_openai_key_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(auto_qa, "has_key_for_model", lambda _model: False)

    result = asyncio.run(
        auto_qa.run_auto_qa_check(
            _problem(tmp_path),
            proof={},
            grade_payload={"score": 0.0},
            rubric_quality={"status": "completed"},
            model_name="openai:gpt-5.5",
            reasoning_effort="xhigh",
        )
    )

    assert result["status"] == "skipped"
    assert result["model"] == "openai:gpt-5.5"
    assert result["reasoning_effort"] == "xhigh"
    assert "OPENAI_API_KEY" in result["reason"]


_VALID_AUTOQA_JSON = '{"overall_assessment": "pass", "summary": "ok", "checks": []}'


def test_run_auto_qa_check_retries_then_recovers_on_empty(monkeypatch, tmp_path):
    # First completion is blank (transient), retry yields valid JSON -> completed.
    model = SequenceModel(["", _VALID_AUTOQA_JSON])
    monkeypatch.setattr(auto_qa, "AUTOQA_RETRY_BACKOFF_SEC", 0.0)
    monkeypatch.setattr(auto_qa, "has_key_for_model", lambda _m: True)
    monkeypatch.setattr(
        auto_qa, "_build_autoqa_model", lambda model_name, *, reasoning_effort: model
    )

    result = asyncio.run(
        auto_qa.run_auto_qa_check(
            _problem(tmp_path),
            proof={},
            grade_payload={"score": 0.2},
            rubric_quality={"status": "completed", "checks": []},
            model_name="openai:gpt-5.5",
        )
    )

    assert result["status"] == "completed"
    assert model.calls == 2  # retried once after the blank completion


def test_run_auto_qa_check_empty_output_is_explainable_advisory_error(
    monkeypatch, tmp_path
):
    # Persistently blank output -> retried the max times, then an explainable
    # status="error" (advisory) instead of a raw JSONDecodeError.
    model = SequenceModel([""])
    monkeypatch.setattr(auto_qa, "AUTOQA_RETRY_BACKOFF_SEC", 0.0)
    monkeypatch.setattr(auto_qa, "has_key_for_model", lambda _m: True)
    monkeypatch.setattr(
        auto_qa, "_build_autoqa_model", lambda model_name, *, reasoning_effort: model
    )

    result = asyncio.run(
        auto_qa.run_auto_qa_check(
            _problem(tmp_path),
            proof={},
            grade_payload={"score": 0.2},
            rubric_quality={"status": "completed", "checks": []},
            model_name="claude-opus-4-7",
            reasoning_effort="xhigh",
        )
    )

    assert result["status"] == "error"
    assert model.calls == auto_qa.AUTOQA_MAX_EMPTY_RETRIES + 1
    assert "empty output" in result["reason"]
    assert "claude-opus-4-7" in result["reason"]
    assert "JSONDecodeError" not in result["reason"]
    assert "Expecting value" not in result["reason"]


def test_run_auto_qa_check_non_parseable_output_keeps_raw_excerpt(monkeypatch, tmp_path):
    # Non-empty but non-JSON output -> not retried (non-blank), explainable error
    # carrying a raw excerpt for diagnosis.
    model = SequenceModel(["I cannot help with that request."])
    monkeypatch.setattr(auto_qa, "has_key_for_model", lambda _m: True)
    monkeypatch.setattr(
        auto_qa, "_build_autoqa_model", lambda model_name, *, reasoning_effort: model
    )

    result = asyncio.run(
        auto_qa.run_auto_qa_check(
            _problem(tmp_path),
            proof={},
            grade_payload={"score": 0.2},
            rubric_quality={"status": "completed", "checks": []},
            model_name="claude-opus-4-7",
        )
    )

    assert result["status"] == "error"
    assert model.calls == 1  # non-blank output is not retried
    assert "non-parseable" in result["reason"]
    assert result["raw_output_excerpt"] == "I cannot help with that request."


def _write_min_proof(problem):
    proof_path = problem.source_problem_dir / runner.PROOF_PATH
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(
        json.dumps(
            {"harness_result": {"score": 0.2, "rubric_quality": {"status": "completed"}}}
        )
    )
    return proof_path


def test_run_auto_qa_only_infra_error_is_non_blocking(monkeypatch, tmp_path):
    # An infra status="error" must NOT fail the required step (advisory).
    problem = _problem(tmp_path)
    _write_min_proof(problem)

    async def fake_check(*_a, **_k):
        return {"status": "error", "reason": "empty output", "checks": []}

    monkeypatch.setattr(runner, "run_auto_qa_check", fake_check)

    result = runner.run_auto_qa_only(problem, require=True)
    assert result["status"] == "error"  # returned, did not raise


def test_run_auto_qa_only_skipped_still_blocks(monkeypatch, tmp_path):
    # A "skipped" result (missing key / no source dir) is a real misconfiguration.
    problem = _problem(tmp_path)
    _write_min_proof(problem)

    async def fake_check(*_a, **_k):
        return {"status": "skipped", "reason": "OPENAI_API_KEY is not set", "checks": []}

    monkeypatch.setattr(runner, "run_auto_qa_check", fake_check)

    try:
        runner.run_auto_qa_only(problem, require=True)
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("run_auto_qa_only should raise on skipped with require=True")


def test_message_text_ignores_reasoning_blocks_before_json() -> None:
    class Message:
        content = [
            {"type": "reasoning", "reasoning": "considering {not valid json}"},
            {"type": "text", "text": '{"overall_assessment": "pass"}'},
        ]

    assert message_text(Message()) == '{"overall_assessment": "pass"}'
