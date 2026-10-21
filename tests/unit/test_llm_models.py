"""
Tests for the shared reasoning-model check and the UA's call kwargs.

`is_reasoning_model` decides which output keys a call sends: a reasoning
model (e.g. gpt-6-luna) gets reasoning_effort / max_completion_tokens, a
non-reasoning one (e.g. gpt-4o-mini) temperature / max_tokens.
"""

import pytest

from ami_agents.agents.user_assistant.utils.llm_client import (
    build_behaviour_llm_client,
    build_llm_call_kwargs,
)
from ami_agents.shared.utils.llm_models import is_reasoning_model


# -- is_reasoning_model ------------------------------------------------------

@pytest.mark.parametrize("model", [
    "o1", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5-nano", "gpt-6-luna", "GPT-6-luna", "gpt-10",
])
def test_reasoning_models(model):
    """The o-series and every gpt major from 5 upward are reasoning models."""
    assert is_reasoning_model(model)


@pytest.mark.parametrize("model", [
    "gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4", "openai/gpt-4o", "", None,
])
def test_non_reasoning_models(model):
    """Earlier gpt families, and names that only start with 'o', are not."""
    assert not is_reasoning_model(model)


# -- UA call kwargs ----------------------------------------------------------

def test_reasoning_step_sends_reasoning_effort(monkeypatch):
    """gpt-6-luna with reasoning_effort: low is called as a reasoning model."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    config = {"llm": {"atomic_segmentation": {"model": "gpt-6-luna", "reasoning_effort": "low"}}}

    kwargs = build_llm_call_kwargs(build_behaviour_llm_client(config, "atomic_segmentation"))

    assert kwargs["reasoning_effort"] == "low"
    assert "temperature" not in kwargs and "max_tokens" not in kwargs


def test_non_reasoning_step_sends_max_tokens(monkeypatch):
    """A non-reasoning step's max_tokens reaches the call, with its temperature."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    config = {"llm": {"plan_summarization": {
        "model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 4096}}}

    kwargs = build_llm_call_kwargs(build_behaviour_llm_client(config, "plan_summarization"))

    assert kwargs == {"temperature": 0.5, "max_tokens": 4096}
