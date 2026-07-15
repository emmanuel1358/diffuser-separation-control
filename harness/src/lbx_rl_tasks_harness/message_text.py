from __future__ import annotations

from typing import Any


def message_text(message: Any) -> str:
    """Extract final assistant text while ignoring reasoning-only content blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "reasoning":
                    continue
                text = (
                    item.get("text") or item.get("content") or item.get("output_text")
                )
                if text is not None:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)
