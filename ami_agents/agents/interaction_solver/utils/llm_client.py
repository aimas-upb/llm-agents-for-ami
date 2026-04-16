"""AsyncOpenAI client construction for the InteractionSolver agent.

Differs from the UserAssistant's equivalent in two ways:

1. A ``planning.llm_planning`` yaml section can override the shared
   ``llm.providers.openai`` settings (planning-specific model/temperature).
2. ``max_tokens`` is supported for non-reasoning models, since BT-JSON
   plans can be long. Reasoning models (o-series) still only use
   ``max_completion_tokens``.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import AsyncOpenAI


@dataclass
class LLMClientConfig:
    """Resolved LLM configuration for the InteractionSolver."""
    client: AsyncOpenAI
    model: str
    base_url: str
    temperature: float
    reasoning_effort: Optional[str]
    max_tokens: Optional[int]
    max_completion_tokens: Optional[int]


def build_llm_client(config: Dict[str, Any]) -> LLMClientConfig:
    """Resolve planning LLM settings and build an AsyncOpenAI client."""
    llm_root = config.get("llm", {}) or {}
    provider_name = llm_root.get("default_provider", "openai")
    provider_cfg = (llm_root.get("providers", {}) or {}).get(provider_name, {}) or {}
    planning_llm = (config.get("planning", {}) or {}).get("llm_planning", {}) or {}

    model = str(planning_llm.get("model") or provider_cfg.get("model") or "gpt-4")
    temperature = float(planning_llm.get("temperature", provider_cfg.get("temperature", 0.7)))

    max_tokens_cfg = planning_llm.get("max_tokens", provider_cfg.get("max_tokens"))
    max_completion_tokens_cfg = planning_llm.get(
        "max_completion_tokens", provider_cfg.get("max_completion_tokens")
    )

    is_reasoning = model.startswith("o")
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    if is_reasoning:
        # Reasoning models reject ``max_tokens``; only carry the *_completion_* variant.
        if max_completion_tokens_cfg is not None:
            try:
                max_completion_tokens = int(max_completion_tokens_cfg)
            except Exception:
                max_completion_tokens = None
    else:
        default_max_tokens = provider_cfg.get("max_tokens", 1500)
        raw_mt = max_tokens_cfg if max_tokens_cfg is not None else default_max_tokens
        try:
            max_tokens = int(raw_mt) if raw_mt is not None else None
        except Exception:
            max_tokens = int(default_max_tokens) if default_max_tokens is not None else 1500
        if max_completion_tokens_cfg is not None:
            try:
                max_completion_tokens = int(max_completion_tokens_cfg)
            except Exception:
                max_completion_tokens = None

    reasoning_effort = (
        planning_llm.get("reasoning_effort")
        or provider_cfg.get("reasoning_effort")
        or llm_root.get("reasoning_effort")
        or os.getenv("OPENAI_REASONING_EFFORT")
    )

    api_key = (
        provider_cfg.get("api_key")
        or llm_root.get("api_key")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = str(
        provider_cfg.get("base_url")
        or llm_root.get("base_url")
        or "https://api.openai.com/v1"
    )
    raw_timeout = (llm_root.get("retry", {}) or {}).get("timeout")

    llm_timeouts = config.get("timeouts", {}).get("llm", {}) or {}
    default_timeout = float(llm_timeouts.get("default", 30.0))
    reasoning_timeout = float(llm_timeouts.get("reasoning", 120.0))

    try:
        timeout = float(raw_timeout) if raw_timeout is not None else default_timeout
    except Exception:
        timeout = default_timeout
    if is_reasoning and "openai.com" in base_url.lower() and timeout < reasoning_timeout:
        timeout = reasoning_timeout

    if reasoning_effort is None and is_reasoning and "openai.com" in base_url:
        reasoning_effort = "high"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    return LLMClientConfig(
        client=client,
        model=model,
        base_url=base_url,
        temperature=temperature,
        reasoning_effort=str(reasoning_effort) if reasoning_effort else None,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
    )
