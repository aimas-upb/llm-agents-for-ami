"""Explicit goals the deterministic pass cannot plan go to the small model,
against their own device -- never to the general workflow."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rdflib import Graph

from ami_agents.agents.interaction_solver.behaviours.explicit_goal_llm_planning import (
    ExplicitGoalLLMPlanningBehaviour,
)
from ami_agents.agents.interaction_solver.behaviours.explicit_goal_planning import (
    ExplicitGoalPlanningBehaviour,
)
from ami_agents.agents.interaction_solver.utils.llm_client import build_llm_client
from ami_agents.bt_planning.planning.bt_planner import AsyncBTPlanner
from ami_agents.shared.models.goal_structure import GoalSpec
from ami_agents.shared.utils.vocabulary import close_types

HOME = """
@prefix hmas: <https://purl.org/hmas/> .
@prefix homeont: <http://example.org/homeont/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix js: <https://www.w3.org/2019/wot/json-schema#> .

<http://h/ac#artifact> a hmas:Artifact, homeont:AirConditioner ; td:title "ac" ;
    td:hasActionAffordance [ a td:ActionAffordance, homeont:SetModeCommand ;
        td:title "hvacMode" ; td:hasForm [ hctl:hasTarget <http://h/ac/hvacMode> ] ;
        td:hasInputSchema [ a js:IntegerSchema ; js:enum 0, 3, 4 ;
                            js:description "0 = Off. 3 = Cool. 4 = Heat." ] ] .
<http://h/fan#artifact> a hmas:Artifact, homeont:Fan ; td:title "fan" ;
    td:hasActionAffordance [ a td:ActionAffordance, homeont:SetOnOffCommand ;
        td:title "onOff" ; td:hasForm [ hctl:hasTarget <http://h/fan/onOff> ] ;
        td:hasInputSchema [ a js:BooleanSchema ] ] .
"""


@pytest.fixture
def graph():
    g = Graph().parse(data=HOME, format="turtle")
    close_types(g)
    return g


def _goal(**fields):
    base = {"goal_specificity": "explicit", "goal_kind": "achievement", "goal_effect": "set",
            "intent_text": "set the AC to cooling", "artifact_name": "ac",
            "artifact_class": "homeont:AirConditioner",
            "affordance_class": "homeont:SetModeCommand", "affordance_name": "hvacMode",
            "parameter_name": "NA", "target_value_text": "cooling"}
    return GoalSpec.from_dict({**base, **fields})


def _agent(planner_result=None, planner_error=None):
    agent = MagicMock()
    agent.explicit_llm = SimpleNamespace(client="small-client", model="gpt-5-nano",
                                         temperature=0.0, reasoning_effort="low",
                                         max_completion_tokens=None)
    generate = AsyncMock(return_value=planner_result, side_effect=planner_error)
    agent.explicit_bt_planner = SimpleNamespace(generate_bt=generate)
    return agent


async def _run_llm(graph, goals, agent):
    behaviour = ExplicitGoalLLMPlanningBehaviour(goals, graph)
    behaviour.agent = agent
    await behaviour.run()
    return behaviour.results


class TestLLMPlanning:
    @pytest.mark.asyncio
    async def test_the_small_model_plans_against_the_one_device(self, graph):
        tree = {"type": "action", "action_url": "http://h/ac/hvacMode",
                "parameters": {"value": 3}}
        agent = _agent({"tree": tree, "explanation": "cool"})
        (entry,) = await _run_llm(graph, [_goal()], agent)
        assert entry == {"tree": tree, "explanation": "cool", "source": "explicit_llm"}

        kwargs = agent.explicit_bt_planner.generate_bt.await_args.kwargs
        assert kwargs["client"] == "small-client" and kwargs["model"] == "gpt-5-nano"
        assert kwargs["temperature"] is None            # reasoning model
        assert {a["artifact_name"] for a in kwargs["affordances"]} == {"ac"}
        assert '"cooling"' in kwargs["goal_brief"]
        assert kwargs["intents"] == ["set the AC to cooling"]

    @pytest.mark.asyncio
    async def test_no_single_device_is_impossible_without_an_llm_call(self, graph):
        agent = _agent({"tree": {"type": "action"}})
        (entry,) = await _run_llm(graph, [_goal(artifact_class="homeont:HeatPump")], agent)
        assert entry["impossible"] and entry["tree"] is None
        agent.explicit_bt_planner.generate_bt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_planner_saying_impossible_is_passed_on(self, graph):
        agent = _agent({"tree": {}, "impossible": True, "explanation": "no such mode"})
        (entry,) = await _run_llm(graph, [_goal()], agent)
        assert entry["impossible"] and entry["explanation"] == "no such mode"

    @pytest.mark.asyncio
    async def test_a_planner_failure_is_an_error_entry(self, graph):
        agent = _agent(planner_error=RuntimeError("rate limited"))
        (entry,) = await _run_llm(graph, [_goal()], agent)
        assert entry["error"] == "explicit_planning_failed" and entry["tree"] is None

    @pytest.mark.asyncio
    async def test_an_empty_tree_is_an_error_entry(self, graph):
        agent = _agent({"tree": {}, "explanation": "LLM call failed: x"})
        (entry,) = await _run_llm(graph, [_goal()], agent)
        assert entry["error"] == "explicit_planning_failed"


class _FakeLLMBehaviour:
    """Records the goals handed over; plans each as a marker tree."""

    handed = []

    def __init__(self, goals, graph, logger=None):
        type(self).handed = [g.affordance_name for g in goals]
        self.results = [{"tree": {"type": "action", "name": f"llm:{g.affordance_name}"}}
                        for g in goals]

    async def join(self):
        return None


class TestExplicitPlanning:
    @pytest.mark.asyncio
    async def test_only_goals_needing_a_value_go_to_the_small_model(self, graph):
        module = "ami_agents.agents.interaction_solver.behaviours.explicit_goal_planning"
        goals = [
            _goal(target_value_determined="3"),                  # deterministic
            _goal(),                                             # needs a value -> LLM
            _goal(artifact_name="fan", artifact_class="homeont:Fan",
                  affordance_class="homeont:SetOnOffCommand", affordance_name="onOff",
                  target_value_determined="true"),               # deterministic
        ]
        behaviour = ExplicitGoalPlanningBehaviour(goals)
        behaviour.agent = MagicMock()
        with patch(f"{module}.fetch_environment_graph", AsyncMock(return_value=graph)), \
             patch(f"{module}.ExplicitGoalLLMPlanningBehaviour", _FakeLLMBehaviour):
            await behaviour.run()

        assert _FakeLLMBehaviour.handed == ["hvacMode"]
        assert behaviour.results[0]["tree"]["parameters"] == {"value": 3}
        assert behaviour.results[1]["tree"]["name"] == "llm:hvacMode"
        assert behaviour.results[2]["tree"]["parameters"] == {"value": True}
        assert all(r is not None for r in behaviour.results)


    @pytest.mark.asyncio
    async def test_a_modify_the_tds_cannot_plan_goes_to_the_small_model(self, graph):
        # The AC's actions state no saref:actsUpon: nothing says what to read.
        module = "ami_agents.agents.interaction_solver.behaviours.explicit_goal_planning"
        goal = _goal(goal_effect="modify", target_value_determined="1",
                     affordance_name="hvacMode")
        behaviour = ExplicitGoalPlanningBehaviour([goal])
        behaviour.agent = MagicMock()
        with patch(f"{module}.fetch_environment_graph", AsyncMock(return_value=graph)), \
             patch(f"{module}.ExplicitGoalLLMPlanningBehaviour", _FakeLLMBehaviour):
            await behaviour.run()
        assert _FakeLLMBehaviour.handed == ["hvacMode"]


class TestPlannerAndConfig:
    @pytest.mark.asyncio
    async def test_the_goal_brief_reaches_the_user_message(self):
        seen = {}

        async def create(**kwargs):
            seen["messages"] = kwargs["messages"]
            raise RuntimeError("stop after capturing the prompt")

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        await AsyncBTPlanner(max_attempts=1).generate_bt(
            intents=["set the AC to cooling"], affordances=[], client=client,
            model="gpt-5-nano", goal_brief='- device: "ac"')
        user = seen["messages"][1]["content"]
        assert user.startswith("set the AC to cooling")
        assert 'Structured goal:\n- device: "ac"' in user

    def test_the_explicit_section_configures_the_small_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        config = {"planning": {
            "llm_planning": {"model": "gpt-5-mini", "reasoning_effort": "medium"},
            "explicit_planning": {"model": "gpt-5-nano", "reasoning_effort": "low"}}}
        small = build_llm_client(config, section="explicit_planning")
        default = build_llm_client(config)
        assert (small.model, small.reasoning_effort) == ("gpt-5-nano", "low")
        assert (default.model, default.reasoning_effort) == ("gpt-5-mini", "medium")


# --------------------------------------------------------------------------
# Modify goals the TDs cannot plan: the small model names a recipe
# --------------------------------------------------------------------------

# A light whose action states no saref:actsUpon -- the deterministic path
# cannot tell which property to read, the model has to.
LIGHT = """
@prefix hmas: <https://purl.org/hmas/> .
@prefix homeont: <http://example.org/homeont/> .
@prefix saref: <https://saref.etsi.org/core/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix js: <https://www.w3.org/2019/wot/json-schema#> .

<http://h/light#artifact> a hmas:Artifact, homeont:DimmableLight ; td:title "light" ;
    td:hasActionAffordance [ a td:ActionAffordance, saref:SetAbsoluteLevelCommand ;
        td:title "brightness" ; td:hasForm [ hctl:hasTarget <http://h/light/actions/brightness> ] ;
        td:hasInputSchema [ a js:IntegerSchema ; js:minimum 1 ; js:maximum 254 ] ] ;
    td:hasPropertyAffordance [ a td:PropertyAffordance, homeont:DimmableLightBrightness ;
        td:title "brightness" ; td:hasForm [ hctl:hasTarget <http://h/light/properties/brightness> ] ;
        td:hasOutputSchema [ a js:IntegerSchema ] ] .
"""


def _modify_goal():
    return GoalSpec.from_dict({
        "goal_specificity": "explicit", "goal_kind": "achievement", "goal_effect": "modify",
        "intent_text": "dim the light by 20%", "artifact_name": "light",
        "artifact_class": "homeont:DimmableLight",
        "affordance_class": "saref:SetAbsoluteLevelCommand", "affordance_name": "brightness",
        "parameter_name": "NA", "target_value_text": "by 20%",
        "target_value_determined": "-20", "target_value_is_percentage": True})


def _recipe_agent(answer):
    """An agent whose small model answers `answer` (a JSON string)."""
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        message = SimpleNamespace(content=answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    agent = MagicMock()
    agent.explicit_llm = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        model="gpt-5-nano", temperature=0.0, reasoning_effort="low",
        max_completion_tokens=None)
    return agent, seen


@pytest.fixture
def light():
    g = Graph().parse(data=LIGHT, format="turtle")
    close_types(g)
    return g


class TestModifyRecipe:
    @pytest.mark.asyncio
    async def test_the_models_recipe_is_built_into_the_tree(self, light):
        agent, seen = _recipe_agent(
            '{"action": "brightness", "parameter": "value", "property": "brightness", '
            '"mode": "scale", "amount": -20, "min": 1, "max": 254}')
        (entry,) = await _run_llm(light, [_modify_goal()], agent)
        read, compute, action = entry["tree"]["children"]
        assert read["property_url"] == "http://h/light/properties/brightness"
        assert compute["args"]["mode"] == "scale" and compute["args"]["max"] == 254
        assert action["action_url"] == "http://h/light/actions/brightness"
        assert entry["source"] == "explicit_llm"
        # The brief tells the model the change and its kind.
        assert "- change: -20 (a percentage)" in seen["messages"][1]["content"]
        assert "temperature" not in seen and seen["reasoning_effort"] == "low"

    @pytest.mark.asyncio
    async def test_a_recipe_naming_what_is_not_there_is_an_error(self, light):
        agent, _ = _recipe_agent(
            '{"action": "dim", "parameter": "value", "property": "brightness", '
            '"mode": "scale", "amount": -20}')
        (entry,) = await _run_llm(light, [_modify_goal()], agent)
        assert entry["error"] == "explicit_planning_failed" and entry["tree"] is None
        assert "no action 'dim'" in entry["explanation"]

    @pytest.mark.asyncio
    async def test_the_model_saying_impossible_is_passed_on(self, light):
        agent, _ = _recipe_agent('{"impossible": "the light cannot be dimmed"}')
        (entry,) = await _run_llm(light, [_modify_goal()], agent)
        assert entry["impossible"] and entry["explanation"] == "the light cannot be dimmed"
