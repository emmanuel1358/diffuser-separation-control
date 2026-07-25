from __future__ import annotations

from typing import Any

from alignerr_plugin.taiga_resources import is_h100_resource, is_tpu_resource

ACCELERATOR_NOTICE_KIND = "accelerator_availability"
GPU_NOTICE = "A GPU may be available."
TPU_NOTICE = "A TPU may be available."


def accelerator_from_required_resources(required_resources: str) -> str:
    """Return the accelerator family implied by a Taiga resource tier."""
    if is_tpu_resource(required_resources):
        return "tpu"
    return "gpu" if is_h100_resource(required_resources) else ""


def accelerator_runtime_notice(required_resources: str) -> dict[str, Any] | None:
    accelerator = accelerator_from_required_resources(required_resources)
    if not accelerator:
        return None
    text = TPU_NOTICE if accelerator == "tpu" else GPU_NOTICE
    return {
        "kind": ACCELERATOR_NOTICE_KIND,
        "accelerator": accelerator,
        "text": text,
        "default_enabled": True,
    }


def without_accelerator_runtime_notices(values: Any) -> list[dict[str, Any]]:
    """Drop generated accelerator notices while preserving unrelated notices."""
    out: list[dict[str, Any]] = []
    for item in values or []:
        if not isinstance(item, dict):
            continue
        if item.get("kind") == ACCELERATOR_NOTICE_KIND:
            continue
        if item.get("text") in {GPU_NOTICE, TPU_NOTICE}:
            continue
        out.append(dict(item))
    return out


def runtime_notices_for_resources(
    required_resources: str, existing: Any = None
) -> list[dict[str, Any]]:
    notices = without_accelerator_runtime_notices(existing)
    accelerator_notice = accelerator_runtime_notice(required_resources)
    if accelerator_notice is not None:
        notices.append(accelerator_notice)
    return notices
