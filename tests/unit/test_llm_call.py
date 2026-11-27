"""`complete`: one answer over the API the client's endpoint serves.

Covered, with fake clients (no network):
  * api.openai.com goes over the Responses API, any other endpoint over Chat
    Completions, and a benchmark's `_ami_tool_api` overrides either;
  * the Chat Completions call is the one the agents always made;
  * the Responses call: system text as instructions, reasoning effort and
    token limit renamed, temperature only for non-reasoning models, JSON
    formats moved to `text.format`;
  * a reply cut by the token limit reports "length" either way.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ami_agents.shared.utils.llm_call import (
    chat_kwargs,
    complete,
    responses_kwargs,
)

MESSAGES = [{"role": "system", "content": "You segment requests."},
            {"role": "user", "content": "turn on the light"}]


def _client(base_url, api=None):
    """A fake AsyncOpenAI client answering "ok" on both APIs."""
    choice = SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")
    client = SimpleNamespace(
        base_url=base_url,
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(choices=[choice])))),
        responses=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
            output_text="ok", status="completed", incomplete_details=None))))
    if api:
        client._ami_tool_api = api
    return client


# -- which API ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_endpoint_goes_over_responses():
    client = _client("https://api.openai.com/v1/")
    reply = await complete(client, "gpt-6-luna", MESSAGES, reasoning_effort="low")
    assert reply.content == "ok" and reply.finish_reason == "stop"
    client.responses.create.assert_awaited_once()
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_local_endpoint_keeps_chat_completions():
    """Ollama and the like: exactly the call the agents always made."""
    client = _client("http://localhost:11434/v1/")
    reply = await complete(client, "qwen3:4b", MESSAGES, temperature=0.0, max_tokens=512)
    assert reply.content == "ok"
    client.chat.completions.create.assert_awaited_once_with(
        model="qwen3:4b", messages=MESSAGES, temperature=0.0, max_tokens=512)
    client.responses.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("override, used", [("chat", "chat"), ("responses", "responses")])
async def test_a_benchmark_override_wins(override, used):
    """The bench's `_ami_tool_api` forces one transport for every phase."""
    client = _client("https://api.openai.com/v1/" if override == "chat" else "http://x/v1", override)
    await complete(client, "gpt-6-luna", MESSAGES)
    assert client.responses.create.await_count == (used == "responses")
    assert client.chat.completions.create.await_count == (used == "chat")


# -- the arguments ----------------------------------------------------------------

def test_responses_arguments_for_a_reasoning_model():
    kwargs = responses_kwargs("gpt-6-luna", MESSAGES, temperature=0.5, max_tokens=99,
                              max_completion_tokens=8192, reasoning_effort="low")
    assert kwargs == {"model": "gpt-6-luna", "instructions": "You segment requests.",
                      "input": [{"role": "user", "content": "turn on the light"}],
                      "reasoning": {"effort": "low"}, "max_output_tokens": 8192}


def test_responses_arguments_for_a_plain_model_and_json_formats():
    kwargs = responses_kwargs("gpt-4o-mini", MESSAGES, temperature=0.5, max_tokens=256,
                              reasoning_effort="low", response_format={"type": "json_object"})
    assert kwargs["temperature"] == 0.5 and "reasoning" not in kwargs
    assert kwargs["max_output_tokens"] == 256
    assert kwargs["text"] == {"format": {"type": "json_object"}}
    schema = responses_kwargs("gpt-6-luna", MESSAGES, response_format={
        "type": "json_schema", "json_schema": {"name": "guards", "strict": True,
                                               "schema": {"type": "object"}}})
    assert schema["text"] == {"format": {"type": "json_schema", "name": "guards",
                                         "schema": {"type": "object"}, "strict": True}}


def test_chat_arguments_leave_out_what_a_model_rejects():
    reasoning = chat_kwargs("gpt-6-luna", MESSAGES, temperature=0.5, max_tokens=99,
                            max_completion_tokens=8192, reasoning_effort="low")
    assert reasoning == {"model": "gpt-6-luna", "messages": MESSAGES,
                         "reasoning_effort": "low", "max_completion_tokens": 8192}
    plain = chat_kwargs("gpt-4o-mini", MESSAGES, temperature=0.5, reasoning_effort="low")
    assert plain == {"model": "gpt-4o-mini", "messages": MESSAGES, "temperature": 0.5}


# -- the reply ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_reply_cut_by_the_token_limit_is_length():
    client = _client("https://api.openai.com/v1/")
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"intents": [', status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"))
    reply = await complete(client, "gpt-6-luna", MESSAGES)
    assert reply.finish_reason == "length" and reply.content.startswith("{")
