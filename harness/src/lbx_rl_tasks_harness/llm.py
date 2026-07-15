from __future__ import annotations

from typing import Any

from lbx_rl_tasks_harness.models_config import (
    provider_for_model,
    strip_provider_prefix,
)

ANTHROPIC_MAX_TOKENS = 128000


def anthropic_kwargs(model_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "thinking": {"type": "adaptive"},
    }


def build_model(model_name: str):
    provider = provider_for_model(model_name)
    concrete_name = strip_provider_prefix(model_name)
    try:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(**anthropic_kwargs(concrete_name))
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=concrete_name)
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=concrete_name)
    except ImportError as exc:
        raise RuntimeError(
            "Model provider dependencies are missing. Install with `uv sync`."
        ) from exc
    raise AssertionError(provider)
