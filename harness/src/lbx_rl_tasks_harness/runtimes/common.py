from __future__ import annotations

import json
from typing import Any

from alignerr_plugin.runtime_notices import (
    GPU_NOTICE,
    TPU_NOTICE,
    accelerator_from_required_resources,
)

from lbx_rl_tasks_harness.models import HarnessProblem


def extra_fields_for_mcp(problem: HarnessProblem, image_tag: str) -> dict[str, Any]:
    """Mirror Boreal's problem entry while flattening nested extra_fields."""
    fields = dict(problem.taiga_problem or {})
    nested = fields.get("extra_fields")
    if isinstance(nested, dict):
        fields.update(nested)
    fields["image"] = image_tag
    return fields


def parse_mcp_tool_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list):
        text_parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    text_parts.append(str(text))
            elif isinstance(item, str):
                text_parts.append(item)
        if text_parts:
            return json.loads("\n".join(text_parts))
    return {"raw": str(raw)}


def task_prompt_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text is not None:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return str(raw)


def _accelerator_from_environment(environment: Any) -> str:
    if not isinstance(environment, dict):
        return ""
    return accelerator_from_required_resources(
        str(environment.get("required_resources") or "")
    )


def _accelerator_from_runtime_notices(values: Any) -> str:
    for item in values or []:
        if not isinstance(item, dict):
            continue
        accelerator = str(item.get("accelerator") or "").strip().lower()
        if accelerator in {"gpu", "tpu"}:
            return accelerator
        text = str(item.get("text") or "")
        if text == TPU_NOTICE:
            return "tpu"
        if text == GPU_NOTICE:
            return "gpu"
    return ""


def accelerator_kind_for_problem(problem: HarnessProblem) -> str:
    for resource in (
        problem.required_resources,
        (problem.taiga_problem or {}).get("required_resources"),
    ):
        accelerator = accelerator_from_required_resources(str(resource or ""))
        if accelerator:
            return accelerator

    taiga_extra = (problem.taiga_problem or {}).get("extra_fields")
    if isinstance(taiga_extra, dict):
        accelerator = _accelerator_from_runtime_notices(
            taiga_extra.get("runtime_notices")
        )
        if accelerator:
            return accelerator

    accelerator = _accelerator_from_runtime_notices(
        problem.metadata.get("runtime_notices")
    )
    if accelerator:
        return accelerator

    return _accelerator_from_environment(problem.metadata.get("environment"))


def _strip_legacy_accelerator_notices(prompt: str) -> str:
    text = prompt
    for notice in (GPU_NOTICE, TPU_NOTICE):
        text = text.replace(notice, "")
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + ("\n" if prompt.endswith("\n") and lines else "")


def localize_accelerator_prompt(problem: HarnessProblem, prompt: str) -> str:
    """Add local/CI accelerator fallback guidance without changing task exports."""
    accelerator = accelerator_kind_for_problem(problem)
    if not accelerator:
        return prompt

    prompt = _strip_legacy_accelerator_notices(prompt)
    if (
        "do not debug drivers" in prompt.lower()
        and "cpu-compatible fallback" in prompt.lower()
    ):
        return prompt

    label = "TPU" if accelerator == "tpu" else "GPU"
    guidance = (
        f"Local/GitHub harness note: this task may request a {label}, but this "
        "harness may or may not expose working accelerator hardware. Run one "
        "quick availability probe; if the accelerator is unavailable or broken, "
        "do not debug drivers. Switch to a CPU-compatible fallback quickly and "
        "still write every required artifact under `/tmp/output`."
    )
    return prompt.rstrip("\n") + "\n\n" + guidance
