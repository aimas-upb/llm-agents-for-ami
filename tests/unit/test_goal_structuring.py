"""GoalStructuringBehaviour: route, context, one LLM call, resolution.

No SPADE run loop, no EnvExplorer, no LLM: the behaviour's `run` is awaited
directly, with the pipeline RPCs and the LLM client patched.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ami_agents.agents.user_assistant.behaviours.goal_structuring import (
    GoalStructuringBehaviour,
)
from ami_agents.agents.user_assistant.prompts.goal_structuring_prompts import (
    GOAL_CONTEXT_PLACEHOLDER,
    PROMPTS,
    render_goal_prompt,
)

MODULE = "ami_agents.agents.user_assistant.behaviours.goal_structuring"


def _llm_returning(payload):
    completion = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(payload)))])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return SimpleNamespace(client=client, model="test-model")


def _behaviour(text, qualifiers):
    behaviour = GoalStructuringBehaviour(text, qualifiers)
    behaviour.agent = MagicMock()
    behaviour.agent.config = {}
    return behaviour


async def _run(behaviour, llm_payload, context="CTX", candidates=None):
    with patch(f"{MODULE}.pipeline") as pl, \
         patch(f"{MODULE}.build_behaviour_llm_client",
               return_value=_llm_returning(llm_payload)) as llm, \
         patch(f"{MODULE}.build_llm_call_kwargs", return_value={}):
        pl.fetch_goal_context = AsyncMock(return_value=context)
        pl.resolve_candidates = AsyncMock(return_value=candidates or {})
        await behaviour.run()
    return pl, llm


class TestPrompts:
    @pytest.mark.parametrize("structure", ["simple", "dependency"])
    def test_each_prompt_has_one_context_slot(self, structure):
        assert PROMPTS[structure].count(GOAL_CONTEXT_PLACEHOLDER) == 1
        rendered = render_goal_prompt(structure, "THE HOME")
        assert "THE HOME" in rendered and GOAL_CONTEXT_PLACEHOLDER not in rendered

    def test_only_the_dependency_prompt_binds(self):
        assert "## Binding goals and predicates" in PROMPTS["dependency"]
        assert "## Binding goals and predicates" not in PROMPTS["simple"]


class TestStructuring:
    @pytest.mark.asyncio
    async def test_simple_goal_gets_actions_only_context(self):
        b = _behaviour("turn on kitchen light 1", ["explicit", "achievement"])
        pl, llm = await _run(b, {"goal_specificity": "explicit",
                                 "goal_kind": "achievement",
                                 "artifact_name": "kitchen_on_off_light_1"})
        kwargs = pl.fetch_goal_context.await_args.kwargs
        assert (kwargs["with_properties"], kwargs["with_environment"]) == (False, False)
        system = llm.return_value.client.chat.completions.create.await_args.kwargs[
            "messages"][0]["content"]
        assert "CTX" in system and "## Binding" not in system
        assert b.result.structure == "simple"
        assert b.result.goals["G1"].artifact_name == "kitchen_on_off_light_1"

    @pytest.mark.asyncio
    async def test_dependency_goal_gets_the_full_context(self):
        b = _behaviour("when washer 1 finishes turn on the light",
                       ["explicit", "logical_dependency", "achievement"])
        pl, _ = await _run(b, {
            "goals": {"G1": {"goal_specificity": "explicit",
                             "timing": {"starts": "on_trigger"}}},
            "predicates": {"P1": {"predicate_specificity": "explicit",
                                  "predicate_subject": "device_property",
                                  "role": "trigger", "applies_to": ["G1"]}}})
        kwargs = pl.fetch_goal_context.await_args.kwargs
        assert kwargs["with_properties"] and kwargs["with_environment"]
        assert b.result.structure == "dependency"
        assert b.result.predicates["P1"].role == "trigger"

    @pytest.mark.asyncio
    async def test_no_context_means_no_structure(self):
        b = _behaviour("turn on the light", ["incomplete", "achievement"])
        _, llm = await _run(b, {}, context="")
        assert b.result is None and b.context_unavailable
        llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_incomplete_goal_with_one_fit_is_promoted(self):
        b = _behaviour("start the dishwasher", ["incomplete", "achievement"])
        pl, _ = await _run(
            b,
            {"goal_specificity": "incomplete", "goal_kind": "achievement",
             "artifact_class": "homeont:Dishwasher",
             "affordance_class": "homeont:SetOnOffCommand"},
            candidates={"outcome": "found", "entries": [{
                "artifact_name": "kitchen_dishwasher_1",
                "artifact_type": "homeont:Dishwasher",
                "workspace_class": "homeont:Kitchen",
                "affordance_name": "onOff"}]})
        pl.resolve_candidates.assert_awaited_once()
        goal = b.result.goals["G1"]
        assert goal.goal_specificity == "explicit"
        assert goal.location_class == "homeont:Kitchen"

    @pytest.mark.asyncio
    async def test_a_goal_naming_neither_room_nor_device_is_forwarded_as_is(self):
        b = _behaviour("turn it on", ["incomplete", "achievement"])
        pl, _ = await _run(b, {"goal_specificity": "incomplete",
                               "goal_kind": "achievement",
                               "affordance_class": "homeont:SetOnOffCommand"})
        pl.resolve_candidates.assert_not_awaited()
        assert b.result.goals["G1"].goal_specificity == "incomplete"

    @pytest.mark.asyncio
    async def test_unparseable_llm_output_is_an_error(self):
        b = _behaviour("turn on the light", ["incomplete", "achievement"])
        await _run(b, ["not", "an", "object"])
        assert b.result is None and b.error and not b.context_unavailable
