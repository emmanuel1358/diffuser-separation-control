from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.json import JSON
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


def _message_type(message: Any) -> str:
    return str(getattr(message, "type", "") or message.__class__.__name__).lower()


def _message_content(message: Any) -> Any:
    return getattr(message, "content", "")


def _tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None)
    return calls if isinstance(calls, list) else []


def _as_pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _clip(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n... truncated {len(text) - limit} chars ..."


def _json_preview(value: Any, limit: int = 1200) -> str:
    """Return a log-safe JSON string preview for GitHub Actions."""
    return _clip(json.dumps(value, default=str, ensure_ascii=False), limit)


def _github_actions_preview(value: Any, limit: int = 1200) -> str:
    """Return readable text for CI logs without JSON-escaping plain strings."""
    if isinstance(value, str):
        return _clip(value, limit)
    return _json_preview(value, limit)


def _parse_jsonish(value: str) -> Any | None:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _normalize_tool_content(value: Any) -> Any:
    """Collapse LangChain content envelopes into the result users care about."""
    if isinstance(value, list):
        text_parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text") or ""))
            else:
                return value
        joined = "\n".join(part for part in text_parts if part).strip()
        if not joined:
            return ""
        parsed = _parse_jsonish(joined)
        return parsed if parsed is not None else joined
    if isinstance(value, str):
        parsed = _parse_jsonish(value)
        return parsed if parsed is not None else value
    return value


def _content_blocks(content: Any) -> list[Any]:
    if isinstance(content, list):
        return content
    return [content]


def _render_message(message: Any, step: int) -> tuple[list[Any], int]:
    kind = _message_type(message)
    content = _message_content(message)
    renderables: list[Any] = []
    if kind in {"human", "system"}:
        return renderables, step
    if kind == "ai":
        text_parts: list[str] = []
        for block in _content_blocks(content):
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in {"text", "thinking", "reasoning"}:
                    text_parts.append(
                        str(block.get("text") or block.get("thinking") or "")
                    )
            else:
                text_parts.append(str(block))
        text = "\n".join(part for part in text_parts if part).strip()
        if text:
            renderables.append(
                Panel(
                    Syntax(_clip(text), "markdown", word_wrap=True),
                    title=f"Step {step:02d} - Model Thinking",
                    border_style="magenta",
                )
            )
            step += 1
        for call in _tool_calls(message):
            name = str(call.get("name") or "tool")
            args = call.get("args") or {}
            renderables.append(
                Panel(
                    JSON(_as_pretty_json(args)),
                    title=f"Step {step:02d} - Tool Call: {name}",
                    border_style="yellow",
                )
            )
            step += 1
        return renderables, step
    if kind == "tool":
        name = str(getattr(message, "name", "") or "tool_result")
        renderables.append(
            _render_text_or_json_renderable(
                content,
                title=f"Step {step:02d} - Tool Result: {name}",
            )
        )
        return renderables, step + 1
    renderables.append(
        _render_text_or_json_renderable(
            content,
            title=f"Step {step:02d} - {message.__class__.__name__}",
        )
    )
    return renderables, step + 1


def _render_text_or_json_renderable(value: Any, *, title: str) -> Panel:
    value = _normalize_tool_content(value)
    if isinstance(value, dict) and set(value).issubset({"output", "error", "system"}):
        output = str(value.get("output") or "").strip()
        error = str(value.get("error") or "").strip()
        system = str(value.get("system") or "").strip()
        body_parts: list[str] = []
        if output:
            body_parts.append(f"[output]\n{output}")
        if error:
            body_parts.append(f"[error]\n{error}")
        if system:
            body_parts.append(f"[system]\n{system}")
        body = "\n\n".join(body_parts) or "(no output)"
        return Panel(
            Syntax(_clip(body), "text", word_wrap=True),
            title=title,
            border_style="cyan",
        )
    if isinstance(value, (dict, list)):
        return Panel(JSON(_as_pretty_json(value)), title=title, border_style="cyan")
    text = str(value or "").strip()
    return Panel(
        Syntax(_clip(text), "text", word_wrap=True),
        title=title,
        border_style="cyan",
    )


def print_agent_trajectory(
    messages: list[Any], *, console: Console | None = None
) -> None:
    console = console or Console()
    console.rule("[bold blue]Local Agent Trajectory")
    renderer = StreamingTrajectoryRenderer(console=console)
    for message in messages:
        renderer.print_message(message)
    console.rule("[bold blue]End Trajectory")


class StreamingTrajectoryRenderer:
    def __init__(self, *, console: Console | None = None) -> None:
        self.console = console or Console()
        self.step = 1
        self._seen_count = 0
        self._started = False
        self._github_actions = os.environ.get("GITHUB_ACTIONS") == "true"

    def print_new_messages(self, messages: list[Any]) -> None:
        if not self._started:
            if self._github_actions:
                print(
                    "Agent trajectory is streaming. Full fidelity is saved to trajectory.json."
                )
            else:
                self.console.rule("[bold blue]Local Agent Trajectory")
            self._started = True
        for message in messages[self._seen_count :]:
            self.print_message(message)
        self._seen_count = len(messages)

    def print_message(self, message: Any) -> None:
        if self._github_actions:
            self._print_github_actions_message(message)
            return
        renderables, self.step = _render_message(message, self.step)
        if renderables:
            self.console.print(Group(*renderables))

    def finish(self) -> None:
        if self._started and not self._github_actions:
            self.console.rule("[bold blue]End Trajectory")

    def _print_github_actions_message(self, message: Any) -> None:
        kind = _message_type(message)
        if kind in {"human", "system"}:
            return
        if kind == "ai":
            content = _message_content(message)
            text_parts: list[str] = []
            for block in _content_blocks(content):
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type in {"text", "thinking", "reasoning"}:
                        text_parts.append(
                            str(block.get("text") or block.get("thinking") or "")
                        )
                else:
                    text_parts.append(str(block))
            text = "\n".join(part for part in text_parts if part).strip()
            if text:
                self._print_github_group("model", text)
            for call in _tool_calls(message):
                self._print_github_group(
                    f"tool_call {call.get('name') or 'tool'}", call.get("args") or {}
                )
            return
        if kind == "tool":
            self._print_github_group(
                f"tool_result {getattr(message, 'name', None) or 'tool'}",
                _normalize_tool_content(_message_content(message)),
            )
            return
        self._print_github_group(
            kind, _normalize_tool_content(_message_content(message))
        )

    def _print_github_group(self, label: str, value: Any) -> None:
        title = f"Step {self.step:02d} - {label}"
        print(f"::group::{title}")
        print(_github_actions_preview(value))
        print("::endgroup::")
        self.step += 1


def write_agent_trajectory(messages: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, message in enumerate(messages, start=1):
        rows.append(
            {
                "index": idx,
                "type": _message_type(message),
                "name": getattr(message, "name", None),
                "content": _message_content(message),
                "tool_calls": _tool_calls(message),
                "usage": getattr(message, "usage_metadata", None),
            }
        )
    path.write_text(_as_pretty_json(rows) + "\n")


def print_grader_results(
    grade_payload: dict[str, Any], *, console: Console | None = None
) -> None:
    console = console or Console()
    subscores = grade_payload.get("subscores") or {}
    weights = grade_payload.get("weights") or {}
    metadata = grade_payload.get("metadata") or {}
    score = (
        metadata.get("score")
        or metadata.get("headline_score")
        or subscores.get("score", 0.0)
    )

    console.rule("[bold green]Grader Results")
    table = Table(title=f"Final Score: {float(score):.6f}", show_lines=False)
    table.add_column("Criterion", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    for name, value in subscores.items():
        table.add_row(
            str(name), f"{float(value):.3f}", f"{float(weights.get(name, 0.0)):.3f}"
        )
    console.print(table)

    breakdown = (
        metadata.get("rubric_breakdown") or metadata.get("structured_subscores") or []
    )
    if isinstance(breakdown, list) and breakdown:
        details = Table(title="Criterion Details", show_lines=True)
        details.add_column("Criterion")
        details.add_column("Pass", justify="center")
        details.add_column("Reasoning")
        for item in breakdown[:20]:
            if not isinstance(item, dict):
                continue
            label = (
                item.get("label")
                or item.get("description")
                or item.get("criterion")
                or item.get("name")
            )
            passed = item.get("passed")
            details.add_row(
                str(label),
                "yes" if passed is True else "no" if passed is False else "",
                Text(str(item.get("reasoning") or ""), overflow="fold"),
            )
        console.print(details)


def print_rubric_quality_review(
    quality: dict[str, Any], *, console: Console | None = None
) -> None:
    console = console or Console()
    console.rule("[bold magenta]Rubric Quality Review")

    status = str(quality.get("status") or "unknown")
    model = quality.get("model")
    summary = str(quality.get("summary") or quality.get("reason") or "")
    header = f"Status: {status}"
    if model:
        header += f" | Model: {model}"
    console.print(header)
    if summary:
        console.print(Text(summary, overflow="fold"))

    checks = quality.get("checks")
    if not isinstance(checks, list) or not checks:
        return

    table = Table(show_lines=True)
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Reason")
    table.add_column("Recommendation")
    for item in checks:
        if not isinstance(item, dict):
            continue
        table.add_row(
            str(item.get("check") or ""),
            str(item.get("status") or ""),
            Text(str(item.get("reason") or ""), overflow="fold"),
            Text(str(item.get("recommendation") or ""), overflow="fold"),
        )
    console.print(table)


class StreamingAutoQaRenderer:
    def __init__(
        self, *, console: Console | None = None, flush_chars: int = 800
    ) -> None:
        self.console = console or Console()
        self.flush_chars = flush_chars
        self._buffer: list[str] = []
        self._buffer_len = 0
        self._started = False
        self._chunk_index = 1

    def start(self) -> None:
        if not self._started:
            self.console.rule("[bold magenta]Auto QA Review")
            self._started = True

    def print_chunk(self, text: str) -> None:
        if not text:
            return
        self.start()
        self._buffer.append(text)
        self._buffer_len += len(text)
        if self._buffer_len >= self.flush_chars or "\n" in text:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer).strip()
        self._buffer = []
        self._buffer_len = 0
        if not text:
            return
        self.console.print(
            Panel(
                Syntax(_clip(text, 4000), "json", word_wrap=True),
                title=f"Auto QA Stream {self._chunk_index:02d}",
                border_style="magenta",
            )
        )
        self._chunk_index += 1

    def finish(self) -> None:
        self.flush()
        if self._started:
            self.console.rule("[bold magenta]End Auto QA Stream")


def print_auto_qa_review(
    auto_qa: dict[str, Any], *, console: Console | None = None
) -> None:
    console = console or Console()
    console.rule("[bold green]Auto QA Result")

    status = str(auto_qa.get("status") or "unknown")
    model = auto_qa.get("model")
    effort = auto_qa.get("reasoning_effort")
    assessment = str(auto_qa.get("overall_assessment") or "unknown")
    confidence = str(auto_qa.get("confidence") or "unknown")
    header = f"Status: {status} | Overall: {assessment} | Confidence: {confidence}"
    if model:
        header += f" | Model: {model}"
    if effort:
        header += f" | Reasoning: {effort}"
    console.print(header)

    summary = str(auto_qa.get("summary") or auto_qa.get("reason") or "")
    if summary:
        console.print(Text(summary, overflow="fold"))

    overview = Table(title="Assessment", show_lines=False)
    overview.add_column("Question", style="bold")
    overview.add_column("Answer", justify="center")
    overview.add_column("Evidence")
    for key, label in [
        ("is_solvable", "Solvable"),
        ("scientifically_correct", "Scientifically correct"),
    ]:
        item = auto_qa.get(key)
        if not isinstance(item, dict):
            continue
        value = "yes" if item.get("value") else "no"
        overview.add_row(
            label, value, Text(str(item.get("evidence") or ""), overflow="fold")
        )
    console.print(overview)

    blocking = auto_qa.get("blocking_issues")
    feedback = auto_qa.get("non_blocking_feedback")
    if isinstance(blocking, list) and blocking:
        console.print(
            Panel(
                "\n".join(f"- {issue}" for issue in blocking),
                title="Blocking Issues",
                border_style="red",
            )
        )
    if isinstance(feedback, list) and feedback:
        console.print(
            Panel(
                "\n".join(f"- {item}" for item in feedback),
                title="Feedback",
                border_style="yellow",
            )
        )

    checks = auto_qa.get("checks")
    if not isinstance(checks, list) or not checks:
        return

    table = Table(title="Auto QA Checks", show_lines=True)
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Reason")
    table.add_column("Recommendation")
    for item in checks:
        if not isinstance(item, dict):
            continue
        table.add_row(
            str(item.get("check") or ""),
            str(item.get("status") or ""),
            Text(str(item.get("reason") or ""), overflow="fold"),
            Text(str(item.get("recommendation") or ""), overflow="fold"),
        )
    console.print(table)
