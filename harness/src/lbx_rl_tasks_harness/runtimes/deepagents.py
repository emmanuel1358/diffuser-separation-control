from __future__ import annotations

from pathlib import Path
from typing import Any

from alignerr_plugin.local_runtime import SLIM_FALLBACK_TURN_BONUS

from lbx_rl_tasks_harness.docker import (
    build_task_image,
    copy_output_from_container,
    start_task_container,
    stop_task_container,
)
from lbx_rl_tasks_harness.mcp_bridge import RubricMcpBridge
from lbx_rl_tasks_harness.models import HarnessProblem
from lbx_rl_tasks_harness.llm import build_model as _build_model
from lbx_rl_tasks_harness.models_config import (
    default_model_from_env,
    has_key_for_model,
    required_env_var,
)
from lbx_rl_tasks_harness.prompts import LOCAL_SYSTEM_PROMPT
from lbx_rl_tasks_harness.runtimes.common import (
    extra_fields_for_mcp as _extra_fields_for_mcp,
    localize_accelerator_prompt as _localize_accelerator_prompt,
    parse_mcp_tool_payload as _parse_mcp_tool_payload,
    task_prompt_text as _task_prompt_text,
)
from lbx_rl_tasks_harness.tmux_tool import build_tmux_tool
from lbx_rl_tasks_harness.trajectory import (
    print_grader_results,
    StreamingTrajectoryRenderer,
    write_agent_trajectory,
)


def _install_local_system_prompt() -> None:
    try:
        import deepagents.graph as graph
    except ImportError as exc:
        raise RuntimeError(
            "DeepAgents runtime dependencies are missing. Install with `uv sync`."
        ) from exc
    graph.BASE_AGENT_PROMPT = LOCAL_SYSTEM_PROMPT


def _counts_agent_step(message: Any) -> bool:
    return str(getattr(message, "type", "")).lower() not in {"human", "system"}


def _deny_host_filesystem_permissions() -> list[Any]:
    """Disable DeepAgents' built-in host filesystem tools.

    The local runtime is supposed to model an agent inside the task container.
    DeepAgents also injects host-side `ls`/`read_file`/`write_file` tools by
    default; if those can touch `/tmp/output`, the agent may create artifacts on
    the host while the grader checks the container's `/tmp/output`.
    """
    try:
        from deepagents.middleware.permissions import FilesystemPermission
    except ImportError as exc:
        raise RuntimeError(
            "DeepAgents permissions are unavailable. Install with `uv sync`."
        ) from exc
    return [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        )
    ]


async def run_deepagents(
    problem: HarnessProblem,
    workspace: Path,
    transcript_path: Path,
    model_name: str | None = None,
    max_steps: int | None = None,
    flavor: str = "auto",
) -> dict[str, Any]:
    task_model = problem.metadata.get("runner", {}).get("api_model_name")
    effective_model = model_name or default_model_from_env(
        task_model if isinstance(task_model, str) else None
    )
    if not has_key_for_model(effective_model):
        env_var = required_env_var(effective_model)
        raise RuntimeError(
            f"{env_var} is required for --runtime deepagents with model {effective_model!r}"
        )
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:
        raise RuntimeError(
            "DeepAgents runtime dependencies are missing. Install with `uv sync`."
        ) from exc

    build = build_task_image(problem, flavor=flavor)
    image_tag = build.image_tag
    if build.flavor_fallback and max_steps is not None:
        # Slim fallback: extend the step cap to offset runtime pip-install overhead.
        max_steps = max_steps + SLIM_FALLBACK_TURN_BONUS
        print(
            f"[slim fallback] agent step budget bumped to {max_steps} "
            f"(+{SLIM_FALLBACK_TURN_BONUS}).",
            flush=True,
        )
    started = start_task_container(image_tag)
    try:
        async with RubricMcpBridge(started.container_id) as bridge:
            extra_fields = _extra_fields_for_mcp(problem, image_tag)
            task_prompt = await bridge.setup_problem_tool().ainvoke(
                {
                    "problem_id": problem.id,
                    "extra_fields": extra_fields,
                    "use_hinted_problem": False,
                }
            )
            task_prompt = _task_prompt_text(task_prompt)
            task_prompt = _localize_accelerator_prompt(problem, task_prompt)

            model = _build_model(effective_model)
            _install_local_system_prompt()
            agent = create_deep_agent(
                model=model,
                tools=[*bridge.agent_tools(), build_tmux_tool(started.container_id)],
                system_prompt=None,
                permissions=_deny_host_filesystem_permissions(),
            )
            renderer = StreamingTrajectoryRenderer()
            messages: list[Any] = []
            steps = 0

            async for state in agent.astream(
                {"messages": [{"role": "user", "content": task_prompt}]},
                stream_mode="values",
            ):
                if isinstance(state, dict) and "messages" in state:
                    messages = list(state["messages"])
                    renderer.print_new_messages(messages)
                    last = messages[-1] if messages else None
                    last_type = str(getattr(last, "type", "")).lower()
                    if last is None or not _counts_agent_step(last):
                        continue

                    steps += 1

                    if max_steps is not None and steps >= max_steps:
                        pending_tool_calls = (
                            last is not None
                            and last_type == "ai"
                            and bool(getattr(last, "tool_calls", None))
                        )

                        if not pending_tool_calls:
                            break

            renderer.finish()
            agent_result = {"messages": messages}
            transcript_path.write_text(repr(agent_result))
            if messages:
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
            grade_payload = _parse_mcp_tool_payload(grade)
            print_grader_results(grade_payload)
            return grade_payload
    finally:
        stop_task_container(started.container_id)
