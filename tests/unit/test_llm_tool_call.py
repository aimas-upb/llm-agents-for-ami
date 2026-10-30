"""Unit tests for `ToolCallSession` (ami_agents/shared/utils/llm_tool_call.py).

The session runs one forced tool call over the Responses API on OpenAI's
endpoint and over Chat Completions on any other. These tests check, on fake
clients, that each backend gets its own request shape, that a retry carries
the feedback where that API expects it, and that the BT planner works
unchanged on both.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
from ami_agents.bt_planning.planning.schema import GENERATE_BT_TOOL
from ami_agents.shared.utils.llm_tool_call import CHAT, RESPONSES, ToolCallSession, api_for

ARGS = {"tree": {}, "explanation": "cannot be done", "impossible": True}


# -- fakes --------------------------------------------------------------------

def _openai_client(*responses):
    """A client on OpenAI's endpoint whose `responses.create` returns ``responses``."""
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1/"
    client.responses.create = AsyncMock(side_effect=list(responses))
    return client


def _local_client(*responses):
    """A client on a local endpoint whose `chat.completions.create` returns ``responses``."""
    client = MagicMock()
    client.base_url = "http://localhost:11434/v1/"
    client.chat.completions.create = AsyncMock(side_effect=list(responses))
    return client


def _function_call(args, response_id="resp_1", call_id="call_1"):
    """A Responses API response holding one function call with ``args``."""
    item = SimpleNamespace(type="function_call", call_id=call_id,
                           name="generate_behavior_tree",
                           arguments=args if isinstance(args, str) else json.dumps(args))
    return SimpleNamespace(id=response_id, output=[item], output_text="")


def _text_response(text, response_id="resp_1"):
    """A Responses API response with text and no function call."""
    item = SimpleNamespace(type="message")
    return SimpleNamespace(id=response_id, output=[item], output_text=text)


def _chat_tool_call(args, call_id="call_1"):
    """A Chat Completions response with one tool call carrying ``args``."""
    call = SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name="generate_behavior_tree", arguments=json.dumps(args)))
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _session(client, model="gpt-6-luna", **settings):
    """A session for the BT tool on ``client``."""
    return ToolCallSession(client, model, system="SYSTEM", user="USER",
                           tool=GENERATE_BT_TOOL, **settings)


# -- choosing the API ---------------------------------------------------------

class TestApiFor:
    def test_openai_endpoint_uses_responses(self):
        # 1. OpenAI's own endpoint, as the AsyncOpenAI client reports it.
        assert api_for(SimpleNamespace(base_url="https://api.openai.com/v1/")) == RESPONSES

    def test_other_endpoints_use_chat(self):
        # 1. A local server, and a client with no base_url (a test double).
        assert api_for(SimpleNamespace(base_url="http://localhost:11434/v1/")) == CHAT
        assert api_for(SimpleNamespace()) == CHAT


# -- the Responses API ----------------------------------------------------------

class TestResponses:
    @pytest.mark.asyncio
    async def test_reasoning_model_gets_tools_and_effort_together(self):
        # 1. One call for gpt-6-luna, the pair Chat Completions rejects.
        client = _openai_client(_function_call(ARGS))
        reply = await _session(client, reasoning_effort="low", temperature=0.3,
                               max_output_tokens=4000).ask()

        # 2. The request: tool and effort together, no temperature.
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["instructions"] == "SYSTEM"
        assert kwargs["input"] == [{"role": "user", "content": "USER"}]
        assert kwargs["tools"][0]["name"] == "generate_behavior_tree"
        assert kwargs["tools"][0]["strict"] is False
        assert kwargs["tool_choice"] == {"type": "function", "name": "generate_behavior_tree"}
        assert kwargs["reasoning"] == {"effort": "low"}
        assert kwargs["max_output_tokens"] == 4000
        assert "temperature" not in kwargs

        # 3. The reply: the call's arguments.
        assert reply.called and reply.arguments == ARGS

    @pytest.mark.asyncio
    async def test_retry_continues_the_previous_response_with_the_feedback(self):
        # 1. Two calls, with feedback on the first reply.
        client = _openai_client(_function_call(ARGS, "resp_1", "call_7"),
                                _function_call(ARGS, "resp_2", "call_8"))
        session = _session(client)
        await session.ask()
        session.correct("bad tree")
        await session.ask()

        # 2. The retry sends only the feedback, as the call's output, after resp_1.
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["previous_response_id"] == "resp_1"
        assert kwargs["instructions"] == "SYSTEM"
        assert kwargs["input"] == [{"type": "function_call_output", "call_id": "call_7",
                                    "output": "bad tree"}]

    @pytest.mark.asyncio
    async def test_text_only_reply_is_corrected_with_a_user_message(self):
        # 1. The model wrote text instead of calling the tool.
        client = _openai_client(_text_response("no tool"), _function_call(ARGS, "resp_2"))
        session = _session(client)
        reply = await session.ask()
        assert not reply.called and reply.content == "no tool"

        # 2. The feedback goes back as the user's turn.
        session.correct("call the tool")
        await session.ask()
        assert client.responses.create.call_args.kwargs["input"] == [
            {"role": "user", "content": "call the tool"}]

    @pytest.mark.asyncio
    async def test_invalid_arguments_are_reported(self):
        # 1. A call whose arguments are not JSON.
        client = _openai_client(_function_call("{not json"))
        reply = await _session(client).ask()

        # 2. Called, but no arguments, and why.
        assert reply.called and reply.arguments is None
        assert "not valid JSON" in reply.error


# -- Chat Completions -------------------------------------------------------------

class TestChat:
    @pytest.mark.asyncio
    async def test_local_model_gets_messages_and_temperature(self):
        # 1. A non-reasoning local model.
        client = _local_client(_chat_tool_call(ARGS))
        reply = await _session(client, model="qwen3:8b", temperature=0.0,
                               max_output_tokens=2000).ask()

        # 2. The Chat Completions request, as the planner sent it before.
        kwargs = client.chat.completions.create.call_args.kwargs
        assert [m["role"] for m in kwargs["messages"]][:2] == ["system", "user"]
        assert kwargs["tools"] == [GENERATE_BT_TOOL]
        assert kwargs["tool_choice"] == {"type": "function",
                                         "function": {"name": "generate_behavior_tree"}}
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_completion_tokens"] == 2000
        assert reply.arguments == ARGS

    @pytest.mark.asyncio
    async def test_retry_appends_a_tool_message(self):
        # 1. Two calls, with feedback on the first tool call.
        client = _local_client(_chat_tool_call(ARGS, "call_3"), _chat_tool_call(ARGS))
        session = _session(client, model="qwen3:8b")
        await session.ask()
        session.correct("bad tree")
        await session.ask()

        # 2. The retry's messages: after system and user, the assistant's call,
        #    then the tool's feedback (the list is the session's own, so the
        #    second reply is appended after them).
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[2]["role"] == "assistant"
        assert messages[3] == {"role": "tool", "tool_call_id": "call_3",
                               "content": "bad tree"}


# -- the BT planner on both backends ------------------------------------------------

class TestPlannerOnResponses:
    @pytest.mark.asyncio
    async def test_impossible_goal_on_openai_goes_through_responses(self):
        # 1. An infeasible goal, planned by gpt-6-luna on OpenAI's endpoint:
        #    the case that failed with tools + reasoning_effort on chat.
        client = _openai_client(_function_call(ARGS))
        result = await AsyncBTPlanner(max_attempts=1).generate_bt(
            intents=["turn on the dishwasher"], affordances=[], client=client,
            model="gpt-6-luna", reasoning_effort="low")

        # 2. Planned over the Responses API, never Chat Completions.
        assert result["impossible"] is True
        assert result["explanation"] == "cannot be done"
        client.responses.create.assert_awaited_once()
        client.chat.completions.create.assert_not_called()
