from __future__ import annotations

import asyncio

from lbx_rl_tasks_harness import rubric_quality
from lbx_rl_tasks_harness.models import HarnessProblem, OutputSpec


class FakeModel:
    prompt = ""

    async def ainvoke(self, _prompt: str):
        self.prompt = _prompt

        class Message:
            content = """{
              "summary": "Looks measurable.",
              "checks": [
                {
                  "check": "Prompt alignment",
                  "status": "pass",
                  "reason": "Criteria match the prompt.",
                  "recommendation": ""
                }
              ]
            }"""

        return Message()


def test_run_rubric_quality_check_normalizes_model_json(monkeypatch, tmp_path):
    problem_dir = tmp_path / "problems" / "demo"
    (problem_dir / "scorer").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "RUBRIC_GUIDANCE.md").write_text("Guidance")
    (problem_dir / "instruction.md").write_text("Do the task")
    (problem_dir / "task.toml").write_text("[task]\nname='demo'\n")
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(): pass"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")

    fake_model = FakeModel()
    monkeypatch.setattr(rubric_quality, "has_key_for_model", lambda _model: True)
    monkeypatch.setattr(rubric_quality, "_build_model", lambda _model: fake_model)

    result = asyncio.run(
        rubric_quality.run_rubric_quality_check(
            HarnessProblem(
                id="demo",
                source_format="problem-dir",
                prompt="demo",
                outputs=[OutputSpec(path="/tmp/output/result.txt")],
                source_problem_dir=problem_dir,
            ),
            grade_payload={"score": 1.0},
            model_name="claude-opus-4-7",
        ),
    )

    assert result["status"] == "completed"
    assert result["summary"] == "Looks measurable."
    assert result["checks"][0] == {
        "check": "Prompt alignment",
        "status": "pass",
        "reason": "Criteria match the prompt.",
        "recommendation": "",
    }
    assert {check["check"] for check in result["checks"]} == set(
        rubric_quality.RUBRIC_QUALITY_CHECKS
    )
    assert "TASK PROMPT / INSTRUCTIONS" in fake_model.prompt
    assert "SCORER RUBRIC CRITERIA / compute_score.py" in fake_model.prompt
    assert "STANDARD RUBRIC GUIDANCE" in fake_model.prompt
    assert (
        "decide whether the rubric criteria adhere to the guidance" in fake_model.prompt
    )
    assert "do not grade the agent submission" in fake_model.prompt.lower()


def test_parse_json_object_accepts_fenced_json() -> None:
    assert rubric_quality._parse_json_object('```json\n{"summary": "ok"}\n```') == {
        "summary": "ok"
    }
