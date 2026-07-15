from __future__ import annotations

from rich.console import Console

from lbx_rl_tasks_harness.trajectory import _github_actions_preview
from lbx_rl_tasks_harness.trajectory import _normalize_tool_content
from lbx_rl_tasks_harness.trajectory import print_auto_qa_review
from lbx_rl_tasks_harness.trajectory import StreamingTrajectoryRenderer
from lbx_rl_tasks_harness.trajectory import print_rubric_quality_review


def test_normalize_langchain_tool_text_envelope() -> None:
    content = [
        {
            "type": "text",
            "text": '{\n  "output": "hello\\n",\n  "error": null,\n  "system": null\n}',
        }
    ]

    assert _normalize_tool_content(content) == {
        "output": "hello\n",
        "error": None,
        "system": None,
    }


def test_github_actions_preview_preserves_plain_text_lines() -> None:
    assert _github_actions_preview("first\nsecond") == "first\nsecond"
    assert _github_actions_preview({"output": "first\nsecond"}) == (
        '{"output": "first\\nsecond"}'
    )


class _ToolMessage:
    type = "tool"
    name = "bash"
    content = [
        {"type": "text", "text": '{"output": "hello", "error": null, "system": null}'}
    ]


def test_streaming_renderer_prints_only_new_messages() -> None:
    console = Console(record=True, force_terminal=False, width=100)
    renderer = StreamingTrajectoryRenderer(console=console)

    renderer.print_new_messages([_ToolMessage()])
    renderer.print_new_messages([_ToolMessage()])
    renderer.finish()

    text = console.export_text()
    assert text.count("Tool Result: bash") == 1
    assert "hello" in text


def test_print_rubric_quality_review_outputs_table() -> None:
    console = Console(record=True, force_terminal=False, width=100)

    print_rubric_quality_review(
        {
            "status": "completed",
            "model": "claude-opus-4-7",
            "summary": "Rubric is measurable.",
            "checks": [
                {
                    "check": "Measurability",
                    "status": "pass",
                    "reason": "Uses deterministic predicates.",
                    "recommendation": "",
                }
            ],
        },
        console=console,
    )

    text = console.export_text()
    assert "Rubric Quality Review" in text
    assert "Status: completed | Model: claude-opus-4-7" in text
    assert "Measurability" in text
    assert "Uses deterministic predicates." in text


def test_print_auto_qa_review_outputs_assessment() -> None:
    console = Console(record=True, force_terminal=False, width=100)

    print_auto_qa_review(
        {
            "status": "completed",
            "model": "openai:gpt-5.5",
            "reasoning_effort": "xhigh",
            "overall_assessment": "needs_changes",
            "confidence": "high",
            "summary": "Needs stronger perturbation checks.",
            "is_solvable": {"value": True, "evidence": "Outputs are declared."},
            "scientifically_correct": {"value": False, "evidence": "Physics gap."},
            "blocking_issues": ["Missing robustness rollout."],
            "non_blocking_feedback": ["Add a naive baseline."],
            "checks": [
                {
                    "check": "MuJoCo physics validity",
                    "status": "fail",
                    "reason": "No perturbations.",
                    "recommendation": "Add fixed perturbation rollouts.",
                }
            ],
        },
        console=console,
    )

    text = console.export_text()
    assert "Auto QA Result" in text
    assert "Overall: needs_changes" in text
    assert "Reasoning:" in text
    assert "xhigh" in text
    assert "Missing robustness rollout." in text
    assert "MuJoCo physics validity" in text
