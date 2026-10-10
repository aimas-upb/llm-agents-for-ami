"""Incomplete achievement goals: choosing devices, and planning on them.

A hand-written home, closed under subclass inference as the InteractionSolver
closes its graph:

    kitchen      kitchen_light_1, kitchen_light_2   OnOffLight, onOff
    bedroom      bedroom_lamp                       OnOffLight, onOff
                 bedroom_ceiling                    DimmableLight, onOff
                 bedroom_fan                        Fan, onOff
    living room  living_fan                         Fan, onOff
    bathroom     bathroom_ac                        AirConditioner, two setpoints
"""

from unittest.mock import MagicMock, patch

import pytest
from rdflib import Graph

from ami_agents.agents.interaction_solver.behaviours.incomplete_goal_planning import (
    IncompleteGoalPlanningBehaviour,
)
from ami_agents.agents.interaction_solver.utils.explicit_goal_plan import (
    find_actions,
    plan_explicit_goal,
)
from ami_agents.agents.interaction_solver.utils.incomplete_goal_plan import (
    CLARIFY,
    IMPOSSIBLE,
    PLAN,
    Candidate,
    as_explicit,
    candidates,
    choose,
    combine,
)
from ami_agents.agents.interaction_solver.utils.specific_capability_context import (
    IMPOSSIBLE_WARNING,
    build_specific_context,
)
from ami_agents.shared.models.goal_structure import GoalSpec
from ami_agents.shared.utils.vocabulary import close_types

PREFIXES = """
@prefix hmas: <https://purl.org/hmas/> .
@prefix homeont: <http://example.org/homeont/> .
@prefix saref: <https://saref.etsi.org/core/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix js: <https://www.w3.org/2019/wot/json-schema#> .
"""


def _room(room, cls, *devices):
    contains = ", ".join(f"<http://h/{d}#artifact>" for d in devices)
    return (f"<http://h/{room}#ws> a hmas:Workspace ; hmas:contains {contains} ; "
            f"homeont:hasSpace <http://h/{room}#space> .\n"
            f"<http://h/{room}#space> a homeont:{cls} .\n")


def _device(name, cls, *actions):
    affs = " , ".join(
        f"[ a td:ActionAffordance, {command} ; td:title \"{title}\" ; "
        f"td:hasForm [ hctl:hasTarget <http://h/{name}/{title}> ] ; "
        f"td:hasInputSchema [ a {schema} ] ]"
        for title, command, schema in actions)
    return (f"<http://h/{name}#artifact> a hmas:Artifact, homeont:{cls} ; "
            f"td:title \"{name}\" ; td:hasActionAffordance {affs} .\n")


ONOFF = ("onOff", "homeont:SetOnOffCommand", "js:BooleanSchema")
HOME = PREFIXES + "".join([
    _room("kitchen", "Kitchen", "kitchen_light_1", "kitchen_light_2"),
    _room("bedroom", "Bedroom", "bedroom_lamp", "bedroom_ceiling", "bedroom_fan"),
    _room("living", "LivingRoom", "living_fan"),
    _room("bathroom", "Bathroom", "bathroom_ac"),
    _device("kitchen_light_1", "OnOffLight", ONOFF),
    _device("kitchen_light_2", "OnOffLight", ONOFF),
    _device("bedroom_lamp", "OnOffLight", ONOFF),
    _device("bedroom_ceiling", "DimmableLight", ONOFF),
    _device("bedroom_fan", "Fan", ONOFF),
    _device("living_fan", "Fan", ONOFF),
    _device("bathroom_ac", "AirConditioner",
            ("coolingSetpoint", "saref:SetAbsoluteLevelCommand", "js:NumberSchema"),
            ("heatingSetpoint", "saref:SetAbsoluteLevelCommand", "js:NumberSchema")),
])


@pytest.fixture(scope="module")
def graph():
    g = Graph().parse(data=HOME, format="turtle")
    close_types(g)
    return g


def _goal(**fields):
    base = {"goal_specificity": "incomplete", "goal_kind": "achievement",
            "goal_effect": "set", "intent_text": "turn on the lights",
            "affordance_class": "homeont:SetOnOffCommand", "affordance_name": "onOff",
            "parameter_name": "NA", "target_value_text": "on",
            "target_value_determined": "true"}
    return GoalSpec.from_dict({**base, **fields})


KITCHEN_LIGHTS = dict(location_class="homeont:Kitchen", artifact_class="homeont:OnOffLight")


def _choose(graph, goal):
    return choose(goal, candidates(graph, goal))


class TestCandidates:
    def test_without_a_device_title_every_fitting_device_is_found(self, graph):
        rows = find_actions(graph, _goal(**KITCHEN_LIGHTS))
        assert sorted(r["artifact_title"] for r in rows) == ["kitchen_light_1",
                                                             "kitchen_light_2"]

    def test_candidates_carry_their_class_and_room(self, graph):
        (lamp,) = candidates(graph, _goal(location_class="homeont:Bedroom",
                                          artifact_class="homeont:OnOffLight"))
        assert lamp == Candidate("bedroom_lamp", "homeont:OnOffLight",
                                 "homeont:Bedroom", "onOff")


class TestNoName:
    def test_all_plans_every_device(self, graph):
        choice = _choose(graph, _goal(quantifier="all", **KITCHEN_LIGHTS))
        assert choice.outcome == PLAN
        assert [c.title for c in choice.chosen] == ["kitchen_light_1", "kitchen_light_2"]

    def test_one_plans_the_first(self, graph):
        for quantifier in ("one", None):
            choice = _choose(graph, _goal(quantifier=quantifier, **KITCHEN_LIGHTS))
            assert [c.title for c in choice.chosen] == ["kitchen_light_1"]

    def test_nothing_fitting_is_impossible(self, graph):
        choice = _choose(graph, _goal(location_class="homeont:Bathroom",
                                      artifact_class="homeont:WindowCoveringController"))
        assert choice.outcome == IMPOSSIBLE
        assert "homeont:WindowCoveringController" in choice.reason

    def test_several_actions_on_one_device_are_asked_about(self, graph):
        choice = _choose(graph, _goal(location_class="homeont:Bathroom",
                                      artifact_class="homeont:AirConditioner",
                                      affordance_class="saref:SetAbsoluteLevelCommand",
                                      affordance_name=None, quantifier="all"))
        assert choice.outcome == CLARIFY
        assert "coolingSetpoint or heatingSetpoint" in choice.reason


class TestNoClass:
    def test_all_plans_every_kind_of_device(self, graph):
        # "turn off everything in the bedroom"
        choice = _choose(graph, _goal(location_class="homeont:Bedroom", quantifier="all"))
        assert choice.outcome == PLAN
        assert [c.title for c in choice.chosen] == ["bedroom_ceiling", "bedroom_fan",
                                                    "bedroom_lamp"]

    def test_one_of_several_kinds_is_asked_about(self, graph):
        choice = _choose(graph, _goal(location_class="homeont:Bedroom", quantifier="one"))
        assert choice.outcome == CLARIFY
        assert [c.title for c in choice.candidates] == ["bedroom_ceiling", "bedroom_fan",
                                                        "bedroom_lamp"]

    def test_one_kind_of_device_is_planned(self, graph):
        choice = _choose(graph, _goal(location_class="homeont:LivingRoom", quantifier="one"))
        assert choice.outcome == PLAN
        assert [c.title for c in choice.chosen] == ["living_fan"]


class TestNoLocation:
    def test_all_across_rooms_plans_every_device(self, graph):
        choice = _choose(graph, _goal(artifact_class="homeont:Fan", quantifier="all"))
        assert [c.title for c in choice.chosen] == ["bedroom_fan", "living_fan"]


class TestExplicitPerDevice:
    def test_a_chosen_device_becomes_an_explicit_goal(self, graph):
        goal = _goal(artifact_class="homeont:Fan", quantifier="all")
        fan = candidates(graph, goal)[0]
        explicit = as_explicit(goal, fan)
        assert (explicit.goal_specificity, explicit.artifact_name, explicit.location_class,
                explicit.quantifier) == ("explicit", "bedroom_fan", "homeont:Bedroom", None)
        assert plan_explicit_goal(graph, explicit)["tree"]["parameters"] == {"value": True}


class TestCombine:
    A = Candidate("a", None, None, "onOff")
    B = Candidate("b", None, None, "onOff")

    def test_several_trees_run_in_sequence_each_failure_tolerant(self):
        entry = combine([{"tree": {"type": "action", "name": "a"}, "explanation": "x"},
                         {"tree": {"type": "action", "name": "b"}, "explanation": "y"}],
                        [self.A, self.B])
        assert entry["tree"]["type"] == "sequence"
        assert [(w["type"], w["children"][0]["name"]) for w in entry["tree"]["children"]] == [
            ("ignore_failure", "a"), ("ignore_failure", "b")]

    def test_a_device_that_cannot_be_planned_is_named(self):
        entry = combine([{"tree": {"type": "action", "name": "a"}, "explanation": "x"},
                         {"tree": None, "impossible": True, "explanation": "out of range"}],
                        [self.A, self.B])
        assert [w["children"][0]["name"] for w in entry["tree"]["children"]] == ["a"]
        assert "b: out of range" in entry["explanation"]

    def test_no_device_planned_is_impossible(self):
        entry = combine([{"tree": None, "impossible": True, "explanation": "p"},
                         {"tree": None, "impossible": True, "explanation": "q"}],
                        [self.A, self.B])
        assert entry["impossible"] and entry["tree"] is None
        assert entry["explanation"] == "a: p; b: q"


class TestValueOnlyContext:
    def test_the_context_is_every_device_of_the_class_in_the_room(self, graph):
        goal = _goal(location_class="homeont:Bedroom", artifact_class="homeont:OnOffLight",
                     affordance_class=None, affordance_name=None,
                     target_value_text="blue", target_value_determined=None)
        context = build_specific_context(graph, goal)
        assert context.artifacts == ["bedroom_lamp"]
        assert IMPOSSIBLE_WARNING in context.goal_brief
        assert "homeont:OnOffLight in a homeont:Bedroom" in context.goal_brief

    def test_no_device_of_the_class_has_no_context(self, graph):
        goal = _goal(location_class="homeont:Kitchen", artifact_class="homeont:Fan",
                     affordance_class=None, affordance_name=None)
        assert build_specific_context(graph, goal) is None


# --------------------------------------------------------------------------
# The behaviour, with the per-device planners faked
# --------------------------------------------------------------------------

MODULE = "ami_agents.agents.interaction_solver.behaviours.incomplete_goal_planning"


class _FakeExplicit:
    """Plans deterministically, as the real explicit behaviour does for these."""

    seen = []

    def __init__(self, goals, logger=None, graph=None):
        type(self).seen = [g.artifact_name for g in goals]
        self.results = [plan_explicit_goal(graph, g) for g in goals]

    async def join(self):
        return None


class _FakeLLM:
    seen = []

    def __init__(self, goals, graph, logger=None):
        type(self).seen = [(g.goal_specificity, g.artifact_name) for g in goals]
        self.results = [{"tree": {"type": "action", "name": f"llm:{g.artifact_name}"},
                         "explanation": "llm"} for g in goals]

    async def join(self):
        return None


async def _plan(graph, goal):
    behaviour = IncompleteGoalPlanningBehaviour([goal])
    behaviour.agent = MagicMock()
    _FakeExplicit.seen, _FakeLLM.seen = [], []
    with patch(f"{MODULE}.fetch_environment_graph", return_value=None) as fetch, \
         patch(f"{MODULE}.ExplicitGoalPlanningBehaviour", _FakeExplicit), \
         patch(f"{MODULE}.ExplicitGoalLLMPlanningBehaviour", _FakeLLM):
        async def fetched(*_):
            return graph
        fetch.side_effect = fetched
        await behaviour.run()
    (entry,) = behaviour.results
    return entry


class TestBehaviour:
    @pytest.mark.asyncio
    async def test_set_on_all_devices_is_a_deterministic_sequence(self, graph):
        entry = await _plan(graph, _goal(quantifier="all", **KITCHEN_LIGHTS))
        assert _FakeExplicit.seen == ["kitchen_light_1", "kitchen_light_2"]
        assert entry["source"] == "incomplete"
        # Each device wrapped so one failing does not stop the next.
        wrappers = entry["tree"]["children"]
        assert {w["type"] for w in wrappers} == {"ignore_failure"}
        assert [w["children"][0]["action_url"] for w in wrappers] == [
            "http://h/kitchen_light_1/onOff", "http://h/kitchen_light_2/onOff"]

    @pytest.mark.asyncio
    async def test_modify_goes_through_the_explicit_path_per_device(self, graph):
        entry = await _plan(graph, _goal(quantifier="all", goal_effect="modify",
                                         **KITCHEN_LIGHTS))
        assert _FakeExplicit.seen == ["kitchen_light_1", "kitchen_light_2"]
        assert _FakeLLM.seen == []
        assert entry["tree"]["type"] == "sequence"

    @pytest.mark.asyncio
    async def test_a_goal_naming_no_action_is_planned_by_the_small_model(self, graph):
        entry = await _plan(graph, _goal(location_class="homeont:Hallway",
                                         artifact_class="homeont:OnOffLight",
                                         affordance_class=None, affordance_name=None,
                                         target_value_text="blue"))
        assert _FakeLLM.seen == [("incomplete", None)]
        assert entry["source"] == "incomplete"

    @pytest.mark.asyncio
    async def test_several_kinds_come_back_as_a_question(self, graph):
        entry = await _plan(graph, _goal(location_class="homeont:Bedroom"))
        assert entry["requires_clarification"] and entry["tree"] is None
        assert entry["explanation"].startswith("Which device do you mean")

    @pytest.mark.asyncio
    async def test_nothing_fitting_is_impossible_without_planning(self, graph):
        entry = await _plan(graph, _goal(location_class="homeont:Kitchen",
                                         artifact_class="homeont:Fan"))
        assert entry["impossible"]
        assert _FakeExplicit.seen == [] and _FakeLLM.seen == []
