"""AsyncOpenAI client construction for the UserAssistant agent.

Encapsulates all the reasoning-model (o-series) special-casing: temperature
pinned to 1.0, reasoning_effort default of ``high``, and a bumped timeout
pulled from ``timeouts.llm.reasoning`` when the default would be too short.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import AsyncOpenAI


@dataclass
class LLMClientConfig:
    """Resolved LLM configuration for the UserAssistant."""
    client: AsyncOpenAI
    model: str
    base_url: str
    temperature: float
    reasoning_effort: Optional[str]
    max_completion_tokens: Optional[int]


def build_llm_client(config: Dict[str, Any]) -> LLMClientConfig:
    """Resolve LLM settings from agent config and build an AsyncOpenAI client.

    Raises ``ValueError`` if no API key is available (via config or env var).
    """
    llm_root = config.get("llm", {}) or {}
    provider_name = llm_root.get("default_provider", "openai")
    provider_cfg = (llm_root.get("providers", {}) or {}).get(provider_name, {}) or {}

    api_key = (
        provider_cfg.get("api_key")
        or llm_root.get("api_key")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Missing OpenAI API key. Set OPENAI_API_KEY or "
            "llm.providers.openai.api_key in agents.yaml."
        )

    model = str(provider_cfg.get("model") or "gpt-4")
    temperature = float(provider_cfg.get("temperature", 0.7))
    max_completion_tokens_cfg = provider_cfg.get("max_completion_tokens")
    base_url = str(
        provider_cfg.get("base_url")
        or llm_root.get("base_url")
        or "https://api.openai.com/v1"
    )

    raw_timeout = (llm_root.get("retry", {}) or {}).get("timeout")
    try:
        timeout: Optional[float] = float(raw_timeout) if raw_timeout is not None else None
    except Exception:
        timeout = None

    # Reasoning-model adjustments (o-series on OpenAI).
    is_reasoning = model.startswith("o")
    if is_reasoning and "openai.com" in base_url.lower():
        temperature = 1.0
        reasoning_timeout = float(
            config.get("timeouts", {}).get("llm", {}).get("reasoning", 120.0)
        )
        if timeout is None or timeout < reasoning_timeout:
            timeout = reasoning_timeout

    reasoning_effort = (
        provider_cfg.get("reasoning_effort")
        or llm_root.get("reasoning_effort")
        or os.getenv("OPENAI_REASONING_EFFORT")
    )
    if reasoning_effort is None and is_reasoning and "openai.com" in base_url:
        reasoning_effort = "high"

    max_completion_tokens: Optional[int] = None
    if is_reasoning and max_completion_tokens_cfg is not None:
        try:
            max_completion_tokens = int(max_completion_tokens_cfg)
        except Exception:
            max_completion_tokens = None

    client_kwargs: Dict[str, Any] = {"api_key": str(api_key), "base_url": base_url}
    if timeout is not None:
        client_kwargs["timeout"] = float(timeout)
    client = AsyncOpenAI(**client_kwargs)

    return LLMClientConfig(
        client=client,
        model=model,
        base_url=base_url,
        temperature=temperature,
        reasoning_effort=str(reasoning_effort) if reasoning_effort else None,
        max_completion_tokens=max_completion_tokens,
    )


def build_llm_call_kwargs(cfg: LLMClientConfig) -> Dict[str, Any]:
    """Build extra kwargs for ``client.chat.completions.create``.

    Reasoning models reject ``temperature``; pass ``reasoning_effort`` and
    ``max_completion_tokens`` instead.
    """
    kwargs: Dict[str, Any] = {}
    if not cfg.model.startswith("o"):
        kwargs["temperature"] = cfg.temperature
    else:
        if cfg.reasoning_effort:
            kwargs["reasoning_effort"] = cfg.reasoning_effort
        if cfg.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = cfg.max_completion_tokens
    return kwargs
