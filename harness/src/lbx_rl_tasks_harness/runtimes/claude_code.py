from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from alignerr_plugin.local_runtime import SLIM_FALLBACK_TURN_BONUS

from lbx_rl_tasks_harness.docker import (
    build_task_image,
    copy_output_from_container,
    start_task_container,
    stop_task_container,
)
from lbx_rl_tasks_harness.mcp_bridge import RubricMcpBridge
from lbx_rl_tasks_harness.models import HarnessProblem
from lbx_rl_tasks_harness.prompts import LOCAL_SYSTEM_PROMPT
from lbx_rl_tasks_harness.runtimes.common import (
    extra_fields_for_mcp,
    localize_accelerator_prompt,
    parse_mcp_tool_payload,
    task_prompt_text,
)
from lbx_rl_tasks_harness.trajectory import (
    print_grader_results,
    StreamingTrajectoryRenderer,
    write_agent_trajectory,
)

DEFAULT_CLAUDE_CODE_MODEL = "claude-fable-5"
CLAUDE_CODE_MODEL_ENV = "LBX_RL_CLAUDE_CODE_MODEL"
DEFAULT_MAX_TURNS = 300


def effective_model_name(model_name: str | None) -> str:
    return (
        model_name
        or os.environ.get(CLAUDE_CODE_MODEL_ENV, "").strip()
        or DEFAULT_CLAUDE_CODE_MODEL
    )


_effective_model = effective_model_name


def _synthetic_tool_call_id() -> str:
    return "toolu_" + uuid.uuid4().hex[:22]


def _synthetic_message_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def _coerce_token_count(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict):
        return sum(_coerce_token_count(v) for v in value.values())
    return 0


def _usage_to_tuple(usage: dict[str, Any] | None) -> tuple[int, int, int, int]:
    if not usage:
        return 0, 0, 0, 0
    input_tokens = _coerce_token_count(usage.get("input_tokens"))
    output_tokens = _coerce_token_count(usage.get("output_tokens"))
    cache_read = _coerce_token_count(usage.get("cache_read_input_tokens"))
    cache_creation = _coerce_token_count(usage.get("cache_creation_input_tokens"))
    return input_tokens, output_tokens, cache_read, cache_creation


def _model_facing_tool_name(mcp_name: str) -> str:
    if not mcp_name.startswith("mcp__"):
        return mcp_name
    parts = mcp_name.split("__", 2)
    if len(parts) != 3:
        return mcp_name
    tool = parts[2]
    if tool == "str_replace_editor":
        return "str_replace_based_edit_tool"
    return tool


def _assistant_to_ai_message(msg: Any, fallback_stop: str) -> AIMessage:
    content_blocks: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for block in getattr(msg, "content", []) or []:
        cls = type(block).__name__
        if cls == "TextBlock":
            content_blocks.append({"type": "text", "text": block.text})
        elif cls == "ThinkingBlock":
            content_blocks.append(
                {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", "") or "",
                    "signature": getattr(block, "signature", None),
                }
            )
        elif cls == "ToolUseBlock":
            tool_id = getattr(block, "id", None) or _synthetic_tool_call_id()
            raw_name = getattr(block, "name", "") or ""
            args = getattr(block, "input", None) or {}
            display_name = _model_facing_tool_name(raw_name)
            content_blocks.append(
                {
                    "id": tool_id,
                    "caller": {"type": "direct"},
                    "input": args,
                    "name": display_name,
                    "type": "tool_use",
                }
            )
            tool_calls.append(
                {
                    "name": display_name,
                    "args": args,
                    "id": tool_id,
                    "type": "tool_call",
                }
            )
        else:
            content_blocks.append({"type": cls.lower(), "raw": repr(block)[:500]})

    input_tokens, output_tokens, cache_read, cache_creation = _usage_to_tuple(
        getattr(msg, "usage", None)
    )
    stop_reason = getattr(msg, "stop_reason", None) or fallback_stop
    message_id = getattr(msg, "message_id", None) or _synthetic_message_id()
    model = getattr(msg, "model", None) or DEFAULT_CLAUDE_CODE_MODEL
    ai = AIMessage(
        content=content_blocks,
        tool_calls=tool_calls,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {
                "cache_read": cache_read,
                "cache_creation": cache_creation,
            },
        },
        response_metadata={
            "id": message_id,
            "model": model,
            "model_name": model,
            "model_provider": "anthropic",
            "stop_reason": stop_reason,
            "usage": {
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        },
    )
    ai.id = message_id
    return ai


def _tool_result_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(raw or "")


def _user_to_tool_messages(msg: Any) -> list[ToolMessage]:
    out: list[ToolMessage] = []
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return out
    for block in content:
        if type(block).__name__ != "ToolResultBlock":
            continue
        tool_use_id = getattr(block, "tool_use_id", "") or _synthetic_tool_call_id()
        is_error = bool(getattr(block, "is_error", False))
        text = _tool_result_text(getattr(block, "content", ""))
        payload = {
            "output": "" if is_error else text,
            "error": text if is_error else "",
            "base64_image": None,
            "system": None,
        }
        out.append(
            ToolMessage(
                content=[
                    {
                        "type": "text",
                        "text": json.dumps(payload),
                        "id": "lc_" + uuid.uuid4().hex[:32],
                    }
                ],
                tool_call_id=tool_use_id,
                name="",
            )
        )
    return out


async def _invoke_bridge_tool(bridge_tool: Any, args: dict[str, Any]) -> dict[str, Any]:
    result = await bridge_tool.ainvoke(args)
    return {"content": [{"type": "text", "text": _tool_result_text(result)}]}


def _build_rubric_mcp_server(bridge: RubricMcpBridge) -> Any:
    from claude_agent_sdk import create_sdk_mcp_server, tool

    bridge_tools = {
        bridge_tool.name: bridge_tool for bridge_tool in bridge.agent_tools()
    }
    bash_bridge_tool = bridge_tools["bash"]
    editor_bridge_tool = bridge_tools["str_replace_based_edit_tool"]

    @tool(
        name="bash",
        description="Run a shell command in the agent workdir.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "restart": {"type": "boolean"},
            },
            "required": ["command"],
        },
    )
    async def bash_tool(args: dict[str, Any]) -> dict[str, Any]:
        return await _invoke_bridge_tool(bash_bridge_tool, args)

    @tool(
        name="str_replace_editor",
        description="Create, view, and edit files in the agent workdir or /tmp/output.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "path": {"type": "string"},
                "file_text": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                "insert_line": {"type": "integer"},
                "insert_text": {"type": "string"},
                "view_range": {"type": "array"},
            },
            "required": ["command", "path"],
        },
    )
    async def str_replace_editor_tool(args: dict[str, Any]) -> dict[str, Any]:
        return await _invoke_bridge_tool(editor_bridge_tool, args)

    return create_sdk_mcp_server(
        name="rubric",
        version="1.0.0",
        tools=[bash_tool, str_replace_editor_tool],
    )


def _build_tmux_mcp_server(container_id: str) -> Any:
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool(
        name="tmux",
        description=(
            "Run a tmux command inside the task container. Use this for "
            "long-running processes that outlive a single bash call."
        ),
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )
    async def tmux_tool(args: dict[str, Any]) -> dict[str, Any]:
        command = args.get("command", "")
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                ["docker", "exec", container_id, "bash", "-lc", f"tmux {command}"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            text = (
                f"exit_code={proc.returncode}\n"
                f"stdout:\n{proc.stdout[-6000:]}\n"
                f"stderr:\n{proc.stderr[-6000:]}"
            )
        except subprocess.TimeoutExpired:
            text = "tmux command timed out after 120s"
        except Exception as exc:  # noqa: BLE001
            text = f"tmux tool error: {exc}"
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server(name="host_tmux", version="1.0.0", tools=[tmux_tool])


async def _run_sdk_agent(
    task_prompt: str,
    *,
    bridge: RubricMcpBridge,
    container_id: str,
    model_name: str,
    max_steps: int | None,
) -> list[BaseMessage]:
    if shutil.which("claude") is None:
        raise RuntimeError(
            "`claude` CLI is required for --runtime claude-code. Install Claude "
            "Code and run `claude /login`, or use --runtime deepagents with "
            "ANTHROPIC_API_KEY."
        )
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
    except ImportError as exc:
        raise RuntimeError(
            "Claude Code runtime requires claude-agent-sdk. Install with `uv sync`."
        ) from exc

    options = ClaudeAgentOptions(
        model=model_name,
        system_prompt=LOCAL_SYSTEM_PROMPT,
        max_turns=max_steps or DEFAULT_MAX_TURNS,
        mcp_servers={
            "rubric": _build_rubric_mcp_server(bridge),
            "host_tmux": _build_tmux_mcp_server(container_id),
        },
        allowed_tools=[
            "mcp__rubric__bash",
            "mcp__rubric__str_replace_editor",
            "mcp__host_tmux__tmux",
        ],
        disallowed_tools=[],
        tools=[],
        permission_mode="bypassPermissions",
        effort="max",
        env={key: value for key, value in os.environ.items() if value is not None},
    )

    messages: list[BaseMessage] = [HumanMessage(content=task_prompt)]
    renderer = StreamingTrajectoryRenderer()
    renderer.print_new_messages(messages)
    async for sdk_msg in query(prompt=task_prompt, options=options):
        cls = type(sdk_msg).__name__
        if cls == "AssistantMessage":
            messages.append(_assistant_to_ai_message(sdk_msg, fallback_stop="tool_use"))
        elif cls == "UserMessage":
            tool_messages = _user_to_tool_messages(sdk_msg)
            prior_names: dict[str, str] = {}
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    for tool_call in message.tool_calls:
                        prior_names.setdefault(tool_call["id"], tool_call["name"])
                    break
            for tool_message in tool_messages:
                if tool_message.tool_call_id in prior_names:
                    tool_message.name = prior_names[tool_message.tool_call_id]
                messages.append(tool_message)
        renderer.print_new_messages(messages)
    renderer.finish()
    return messages


async def run_claude_code(
    problem: HarnessProblem,
    workspace: Path,
    transcript_path: Path,
    model_name: str | None = None,
    max_steps: int | None = None,
    flavor: str = "auto",
) -> dict[str, Any]:
    effective_model = effective_model_name(model_name)
    build = build_task_image(problem, flavor=flavor)
    image_tag = build.image_tag
    if build.flavor_fallback:
        # Slim fallback: agent pip-installs ML wheels at runtime, so add a turn bonus.
        max_steps = (max_steps or DEFAULT_MAX_TURNS) + SLIM_FALLBACK_TURN_BONUS
        print(
            f"[slim fallback] agent turn budget bumped to {max_steps} "
            f"(+{SLIM_FALLBACK_TURN_BONUS}).",
            flush=True,
        )
    started = start_task_container(image_tag)
    try:
        async with RubricMcpBridge(started.container_id) as bridge:
            extra_fields = extra_fields_for_mcp(problem, image_tag)
            task_prompt = await bridge.setup_problem_tool().ainvoke(
                {
                    "problem_id": problem.id,
                    "extra_fields": extra_fields,
                    "use_hinted_problem": False,
                }
            )
            task_prompt = task_prompt_text(task_prompt)
            task_prompt = localize_accelerator_prompt(problem, task_prompt)
            messages = await _run_sdk_agent(
                task_prompt,
                bridge=bridge,
                container_id=started.container_id,
                model_name=effective_model,
                max_steps=max_steps,
            )
            transcript_path.write_text(repr({"messages": messages}))
            write_agent_trajectory(
                messages, transcript_path.with_name("trajectory.json")
            )
            grade = await bridge.grade_problem_tool().ainvoke(
                {
                    "problem_id": problem.id,
                    "transcript": transcript_path.read_text(),
                    "extra_fields": extra_fields,
                }
            )
            copy_output_from_container(started.container_id, workspace)
            grade_payload = parse_mcp_tool_payload(grade)
            print_grader_results(grade_payload)
            return grade_payload
    finally:
        stop_task_container(started.container_id)
