"""
Tests that the UA's plan summary uses its own LLM config.

`summarize_plan` must call the client of `user_assistant.llm.plan_summarization`
(e.g. gpt-4o-mini at temperature 0.5), not the agent's default client, which
is built from the atomic-segmentation config.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ami_agents.agents.user_assistant import pipeline
from ami_agents.agents.user_assistant.utils.llm_client import (
    build_behaviour_llm_client,
    build_llm_call_kwargs,
)


# -- helpers -----------------------------------------------------------------

def _client(text: str) -> MagicMock:
    """An OpenAI-style client whose completion returns `text`."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]))
    return client


# -- tests -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_uses_the_plan_summarization_client():
    """The per-step client and kwargs are called; the default client never is."""
    # 1. An agent whose default (segmentation) and summary clients differ.
    default_client, summary_client = _client("wrong"), _client("Lights go on.")
    agent = SimpleNamespace(
        llm_client=default_client, llm_model="segmentation-model",
        build_llm_kwargs=lambda: {"reasoning_effort": "low"},
        get_llm_client_for_behaviour=lambda key: SimpleNamespace(
            client=summary_client, model=f"{key}-model"),
        build_llm_kwargs_for_behaviour=lambda key: {"temperature": 0.5},
    )

    # 2. Summarise, then check which client answered and with what.
    summary = await pipeline.summarize_plan(agent, MagicMock(), '{"tree": {}}')

    assert summary == "Lights go on."
    default_client.chat.completions.create.assert_not_called()
    call = summary_client.chat.completions.create.call_args.kwargs
    assert call["model"] == "plan_summarization-model"
    assert call["temperature"] == 0.5


def test_plan_summarization_config_is_read(monkeypatch):
    """The step's yaml block reaches its client and call kwargs: model, temperature, max_tokens."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    config = {"llm": {
        "atomic_segmentation": {"model": "gpt-6-luna", "reasoning_effort": "low"},
        "plan_summarization": {"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 4096},
    }}

    cfg = build_behaviour_llm_client(config, "plan_summarization")

    assert cfg.model == "gpt-4o-mini"
    assert build_llm_call_kwargs(cfg) == {"temperature": 0.5, "max_tokens": 4096}
