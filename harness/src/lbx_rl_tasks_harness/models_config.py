from __future__ import annotations

import os

DEFAULT_MODEL = "claude-fable-5"
GLOBAL_MODEL_ENV = "LBX_RL_HARNESS_MODEL"

SUPPORTED_MODEL_EXAMPLES = {
    "claude": [
        "claude-fable-5",
        "claude-opus-4-7",
        "claude-sonnet-4-5",
        "anthropic:claude-fable-5",
    ],
    "openai": [
        "openai:gpt-5.1",
        "gpt-5.1",
    ],
    "gemini": [
        "google:gemini-3-pro",
        "gemini-3-pro",
    ],
}


def provider_for_model(model_name: str) -> str:
    lowered = model_name.lower()
    if lowered.startswith("anthropic:") or lowered.startswith("claude-"):
        return "anthropic"
    if lowered.startswith("openai:") or lowered.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if lowered.startswith("google:") or lowered.startswith("gemini-"):
        return "google"
    raise ValueError(
        "Could not infer provider from model. Prefix the model with one of "
        "`anthropic:`, `openai:`, or `google:`."
    )


def provider_model_env_var(provider: str) -> str:
    if provider == "anthropic":
        return "ANTHROPIC_MODEL"
    if provider == "openai":
        return "OPENAI_MODEL"
    if provider == "google":
        return "GOOGLE_MODEL"
    raise AssertionError(provider)


def default_model_from_env(task_model: str | None = None) -> str:
    """Resolve model precedence for local runs.

    Order:
    1. `LBX_RL_HARNESS_MODEL`, for switching providers globally.
    2. Provider-specific model env matching `task_model`, e.g. `ANTHROPIC_MODEL`.
    3. `task.toml [runner].api_model_name`.
    4. Built-in default.
    """
    global_model = os.environ.get(GLOBAL_MODEL_ENV, "").strip()
    if global_model:
        return global_model
    if task_model:
        provider = provider_for_model(task_model)
        provider_model = os.environ.get(provider_model_env_var(provider), "").strip()
        if provider_model:
            return provider_model
        return task_model
    return DEFAULT_MODEL


def strip_provider_prefix(model_name: str) -> str:
    if ":" not in model_name:
        return model_name
    return model_name.split(":", 1)[1]


def required_env_var(model_name: str) -> str:
    provider = provider_for_model(model_name)
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "google":
        return "GOOGLE_API_KEY"
    raise AssertionError(provider)


def has_key_for_model(model_name: str) -> bool:
    return bool(os.environ.get(required_env_var(model_name), "").strip())
