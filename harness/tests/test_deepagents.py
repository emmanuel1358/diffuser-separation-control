from __future__ import annotations

from types import SimpleNamespace

from lbx_rl_tasks_harness.llm import (
    ANTHROPIC_MAX_TOKENS,
    anthropic_kwargs as _anthropic_kwargs,
)
from lbx_rl_tasks_harness.runtimes.deepagents import (
    _counts_agent_step,
    _deny_host_filesystem_permissions,
    _task_prompt_text,
)


def test_task_prompt_text_extracts_mcp_text_blocks() -> None:
    raw = [
        {"type": "text", "text": "Write /tmp/output/model.xml."},
        {"type": "text", "text": "Then write /tmp/output/policy.py."},
    ]

    assert _task_prompt_text(raw) == (
        "Write /tmp/output/model.xml.\nThen write /tmp/output/policy.py."
    )


def test_anthropic_kwargs_use_opus_4_7_output_limit_and_adaptive_thinking() -> None:
    kwargs = _anthropic_kwargs("claude-opus-4-7")

    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["max_tokens"] == ANTHROPIC_MAX_TOKENS == 128000
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_counts_agent_step_ignores_initial_input_messages() -> None:
    assert not _counts_agent_step(SimpleNamespace(type="human"))
    assert not _counts_agent_step(SimpleNamespace(type="system"))
    assert _counts_agent_step(SimpleNamespace(type="ai"))
    assert _counts_agent_step(SimpleNamespace(type="tool"))


def test_deepagents_denies_host_filesystem_tools() -> None:
    permissions = _deny_host_filesystem_permissions()

    assert len(permissions) == 1
    rule = permissions[0]
    assert set(rule.operations) == {"read", "write"}
    assert rule.paths == ["/**"]
    assert rule.mode == "deny"
