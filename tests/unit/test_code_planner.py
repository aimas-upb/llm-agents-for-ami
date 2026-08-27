"""
Unit tests for AsyncCodeBTPlanner (direct-code plan generation).

Mirrors the mocked-client pattern of test_bt_planner_affordance_refs.py;
the point is that code mode shares the IR mode's post-generation pipeline
(affordance resolution, settling annotation, validation, retry feedback).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
from ami_agents.bt_planning.planning.code_planner import AsyncCodeBTPlanner

from .test_bt_planner_affordance_refs import AFFORDANCES, LIGHT_ACTION, LIGHT_PROPERTY


def _content_response(content: str):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client(*responses):
    client = MagicMock()
    if len(responses) == 1:
        client.chat.completions.create = AsyncMock(return_value=responses[0])
    else:
        client.chat.completions.create = AsyncMock(side_effect=list(responses))
    return client


GOOD_CODE = (
    "```python\n"
    "current = env['light308']['brightness']\n"
    "tree = sequence('brighten',\n"
    "    action('light308/setBrightness', {'brightness': min(current + 50, 100)}),\n"
    "    condition('light308/brightness', current, op='>'))\n"
    "```"
)


class TestCodeGeneration:
    @pytest.mark.asyncio
    async def test_fenced_code_produces_resolved_tree(self):
        planner = AsyncCodeBTPlanner(max_attempts=1)
        client = _client(_content_response(GOOD_CODE))

        result = await planner.generate_bt(
            intents=["make it brighter"],
            affordances=AFFORDANCES,
            state={"artifacts": {"light308": {"brightness": 10}}},
            client=client,
            model="gpt-4o-mini",
        )

        assert result["plan_mode"] == "python_code"
        assert result["generated_code"].startswith("current =")
        tree = result["tree"]
        assert tree["children"][0]["action_url"] == LIGHT_ACTION["target"]
        assert tree["children"][0]["parameters"] == {"brightness": 60}
        assert tree["children"][1]["property_url"] == LIGHT_PROPERTY["target"]
        assert tree["children"][1]["expected_value"] == 10

    @pytest.mark.asyncio
    async def test_api_call_is_plain_completion(self):
        planner = AsyncCodeBTPlanner(max_attempts=1)
        client = _client(_content_response(GOOD_CODE))

        await planner.generate_bt(
            intents=["make it brighter"],
            affordances=AFFORDANCES,
            state={"artifacts": {"light308": {"brightness": 10}}},
            client=client,
            model="gpt-4o-mini",
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    @pytest.mark.asyncio
    async def test_ir_mode_still_forces_tool_choice(self):
        planner = AsyncBTPlanner(max_attempts=1)
        client = _client(
            _content_response('```json\n{"tree": {"name": "N", "type": "action", '
                              '"affordance_id": "light308/setBrightness"}, '
                              '"explanation": "", "impossible": false}\n```')
        )

        await planner.generate_bt(
            intents=["x"], affordances=AFFORDANCES, client=client, model="gpt-4o-mini"
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["tool_choice"]["function"]["name"] == "generate_behavior_tree"

    @pytest.mark.asyncio
    async def test_exec_error_feeds_retry_then_success(self):
        planner = AsyncCodeBTPlanner(max_attempts=2)
        bad = _content_response("```python\ntree = env['nope']['x']\n```")
        client = _client(bad, _content_response(GOOD_CODE))

        result = await planner.generate_bt(
            intents=["make it brighter"],
            affordances=AFFORDANCES,
            state={"artifacts": {"light308": {"brightness": 10}}},
            client=client,
            model="gpt-4o-mini",
        )

        assert client.chat.completions.create.await_count == 2
        assert result["tree"]["children"][0]["action_url"] == LIGHT_ACTION["target"]

        # Feedback goes through the user role (plain-completion conversation)
        # and carries the exec error.
        retry_messages = client.chat.completions.create.call_args.kwargs["messages"]
        user_feedback = [m for m in retry_messages if m.get("role") == "user"]
        assert any("KeyError" in m["content"] for m in user_feedback)

    @pytest.mark.asyncio
    async def test_no_code_block_is_retryable(self):
        planner = AsyncCodeBTPlanner(max_attempts=1)
        client = _client(_content_response("I cannot plan this right now, sorry."))

        result = await planner.generate_bt(
            intents=["x"], affordances=AFFORDANCES, client=client, model="gpt-4o-mini"
        )

        assert result["tree"] == {}
        assert "python" in result["explanation"]

    @pytest.mark.asyncio
    async def test_last_fence_with_tree_wins(self):
        content = (
            "Here is the API recap:\n"
            "```python\n"
            "# action(affordance_id, parameters)\n"
            "```\n"
            "And the plan:\n" + GOOD_CODE
        )
        planner = AsyncCodeBTPlanner(max_attempts=1)
        client = _client(_content_response(content))

        result = await planner.generate_bt(
            intents=["make it brighter"],
            affordances=AFFORDANCES,
            state={"artifacts": {"light308": {"brightness": 10}}},
            client=client,
            model="gpt-4o-mini",
        )

        assert result["tree"]["children"][0]["parameters"] == {"brightness": 60}

    @pytest.mark.asyncio
    async def test_mark_impossible_short_circuits(self):
        planner = AsyncCodeBTPlanner(max_attempts=1)
        client = _client(
            _content_response("```python\nmark_impossible('no light affordance')\n```")
        )

        result = await planner.generate_bt(
            intents=["x"], affordances=AFFORDANCES, client=client, model="gpt-4o-mini"
        )

        assert result["impossible"] is True
        assert result["tree"] == {}
        assert result["explanation"] == "no light affordance"
        assert result["plan_mode"] == "python_code"

    @pytest.mark.asyncio
    async def test_hallucinated_ref_goes_through_shared_resolution(self):
        planner = AsyncCodeBTPlanner(max_attempts=1)
        client = _client(
            _content_response("```python\ntree = action('light999/doesNotExist')\n```")
        )

        result = await planner.generate_bt(
            intents=["x"], affordances=AFFORDANCES, client=client, model="gpt-4o-mini"
        )

        assert result["tree"] == {}
        assert "unknown affordance_id" in result["explanation"]

    @pytest.mark.asyncio
    async def test_settling_time_annotation_applies_in_code_mode(self):
        affordance = dict(LIGHT_ACTION)
        affordance["settling_time_seconds"] = 60.0
        code = (
            "```python\n"
            "tree = sequence('plan',\n"
            "    action('light308/setBrightness', {'brightness': 60}),\n"
            "    wait_condition('light308/brightness', 60, op='>=', timeout_seconds=5))\n"
            "```"
        )
        planner = AsyncCodeBTPlanner(max_attempts=1)
        client = _client(_content_response(code))

        result = await planner.generate_bt(
            intents=["x"],
            affordances=[affordance, LIGHT_PROPERTY],
            client=client,
            model="gpt-4o-mini",
        )

        tree = result["tree"]
        assert tree["children"][0]["settling_time_seconds"] == 60.0
        # The shared pipeline raises the wait timeout to settling + margin.
        assert tree["children"][1]["timeout_seconds"] >= 60.0


class TestParseCodeContent:
    def test_plain_fence(self):
        code = AsyncCodeBTPlanner._parse_code_content("```python\ntree = action('a/b')\n```")
        assert code == "tree = action('a/b')"

    def test_unfenced_script_with_tree_assignment(self):
        code = AsyncCodeBTPlanner._parse_code_content("tree = action('a/b')")
        assert code == "tree = action('a/b')"

    def test_prose_returns_none(self):
        assert AsyncCodeBTPlanner._parse_code_content("I refuse.") is None
        assert AsyncCodeBTPlanner._parse_code_content(None) is None
        assert AsyncCodeBTPlanner._parse_code_content("") is None

    def test_py_fence_variant(self):
        code = AsyncCodeBTPlanner._parse_code_content("```py\ntree = action('a/b')\n```")
        assert code == "tree = action('a/b')"
