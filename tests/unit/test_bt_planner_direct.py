"""
Unit tests for DirectCodeBTPlanner with a mocked AsyncOpenAI client.
"""

import json
from types import SimpleNamespace

import pytest

from ami_agents.bt_planning.planning.bt_planner_direct import (
    DirectCodeBTPlanner,
    extract_impossible_subgoals,
    strip_code_fences,
)


def _make_client(message):
    """Return an object exposing chat.completions.create as an async callable."""

    async def _create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )


def _tool_message(code, explanation="ok"):
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            arguments=json.dumps({"code": code, "explanation": explanation})
        ),
    )
    return SimpleNamespace(content=None, tool_calls=[tool_call])


def _text_message(content):
    return SimpleNamespace(content=content, tool_calls=None)


AFFORDANCES = [
    {"affordance_id": "a1", "affordance_type": "action", "target": "http://x/on"}
]


class TestHelpers:
    def test_extract_impossible(self):
        code = "tree = None\n# IMPOSSIBLE: no oven affordance\n"
        assert extract_impossible_subgoals(code) == ["no oven affordance"]

    def test_strip_fences(self):
        assert strip_code_fences("```python\ntree = 1\n```") == "tree = 1"


@pytest.mark.asyncio
class TestGenerate:
    async def test_tool_call_extracts_code(self):
        code = "tree = py_trees.behaviours.Success(name='ok')"
        planner = DirectCodeBTPlanner(max_attempts=1)
        result = await planner.generate_bt(
            intents=["turn on"], affordances=AFFORDANCES,
            client=_make_client(_tool_message(code)), model="gpt-4o",
        )
        assert result["tree"] is None
        assert result["code"] == code
        assert result["impossible"] is False

    async def test_impossible_flag_from_comment(self):
        code = "# IMPOSSIBLE: no such device\ntree = py_trees.behaviours.Success(name='x')"
        planner = DirectCodeBTPlanner(max_attempts=1)
        result = await planner.generate_bt(
            intents=["do"], affordances=AFFORDANCES,
            client=_make_client(_tool_message(code)), model="gpt-4o",
        )
        assert result["impossible"] is True
        assert result["detected_impossible"] == ["no such device"]

    async def test_syntax_error_fails_after_retries(self):
        planner = DirectCodeBTPlanner(max_attempts=2)
        result = await planner.generate_bt(
            intents=["do"], affordances=AFFORDANCES,
            client=_make_client(_tool_message("tree = (")), model="gpt-4o",
        )
        assert result["code"] == ""
        assert "syntax error" in result["explanation"]

    async def test_text_fallback_parses_python_block(self):
        content = "Here is the tree:\n```python\ntree = py_trees.behaviours.Success(name='ok')\n```"
        planner = DirectCodeBTPlanner(max_attempts=1)
        result = await planner.generate_bt(
            intents=["do"], affordances=AFFORDANCES,
            client=_make_client(_text_message(content)), model="qwen",
            use_tool_calling=False,
        )
        assert "Success" in result["code"]
        assert result["impossible"] is False
