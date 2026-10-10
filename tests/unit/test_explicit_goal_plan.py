"""Deterministic plans for explicit achievement goals.

The SimuHome graph is parsed from EnvExplorer's detailed dump, exactly as the
InteractionSolver receives it (no vocabulary merged in). A small hand-written
graph covers what SimuHome does not emit: a parameterized call, and two
actions sharing names.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdflib import Graph

from ami_agents.agents.env_explorer.utils.data_formatting import (
    format_capabilities_detailed_rdf,
)
from ami_agents.agents.interaction_solver.utils.explicit_goal_plan import (
    deterministic_goal,
    plan_explicit_goal,
)
from ami_agents.agents.interaction_solver.utils.plan_envelope import (
    envelope_plan_entries,
    plan_entries_from_envelope,
)
from ami_agents.shared.models.goal_structure import GoalSpec, GoalStructure
from ami_agents.shared.models.intents import Intent
from ami_agents.shared.utils.vocabulary import close_types

REPO = Path(__file__).resolve().parents[2]
EPISODE = REPO.parent / "SimuHome" / "data" / "benchmark" / "qt1_feasible_seed_1.json"
needs_corpus = pytest.mark.skipif(not EPISODE.is_file(),
                                  reason="SimuHome benchmark corpus not available")


@pytest.fixture(scope="module")
def isa_graph() -> Graph:
    from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD

    config = json.loads(EPISODE.read_text())["initial_home_config"]
    b = SimuHomeTD("http://localhost:8097", "test", config)
    agent = SimpleNamespace(
        discovery_complete=True,
        environment_map={r: SimpleNamespace(rdf=b.room_workspace(r).serialize(format="turtle"))
                         for r in config["rooms"]},
        artifacts={d["device_id"]: SimpleNamespace(thing_description=SimpleNamespace(
            rdf=b.artifact(r, d["device_id"]).serialize(format="turtle")))
            for r in config["rooms"] for d in config["rooms"][r].get("devices", [])})
    graph = Graph().parse(data=format_capabilities_detailed_rdf(agent), format="turtle")
    close_types(graph)     # as fetch_environment_graph does
    return graph


AC = {"goal_specificity": "explicit", "goal_kind": "achievement", "goal_effect": "set",
      "location_class": "homeont:Bathroom", "artifact_class": "homeont:AirConditioner",
      "artifact_name": "bathroom_air_conditioner_1", "parameter_name": "NA"}


def _goal(**fields):
    return GoalSpec.from_dict({**AC, **fields})


def _structure(**fields):
    return GoalStructure.from_dict({**AC, **fields}, intent_text="t", structure="simple")


class TestQualifies:
    def test_explicit_set_with_a_determined_value(self):
        assert deterministic_goal(_structure(affordance_name="hvacMode",
                                             target_value_determined="3"))

    def test_no_determined_value_still_qualifies(self):
        # Whether the action needs one is only known once it is found.
        assert deterministic_goal(_structure(affordance_name="hvacMode",
                                             target_value_determined=None))

    def test_an_explicit_modify_qualifies(self):
        # Planned as read -> compute -> set (modify_goal_plan).
        assert deterministic_goal(_structure(affordance_name="hvacMode", goal_effect="modify",
                                             target_value_determined="1"))

    @pytest.mark.parametrize("fields", [
        {"goal_specificity": "incomplete", "target_value_determined": "3"},
        {"goal_kind": "maintenance", "target_value_determined": "3"},
    ])
    def test_everything_else_does_not(self, fields):
        assert deterministic_goal(_structure(affordance_name="hvacMode", **fields)) is None

    def test_a_dependency_structure_does_not(self):
        structure = _structure(affordance_name="hvacMode", target_value_determined="3")
        structure.structure = "dependency"
        assert deterministic_goal(structure) is None


@needs_corpus
class TestSimuHome:
    def test_an_enum_value_becomes_a_one_action_tree(self, isa_graph):
        result = plan_explicit_goal(isa_graph, _goal(
            affordance_class="homeont:SetModeCommand", affordance_name="hvacMode",
            target_value_text="cooling mode", target_value_determined="3"))
        assert result["tree"] == {
            "type": "action", "name": "bathroom_air_conditioner_1.hvacMode",
            "action_url": "http://localhost:8097/workspaces/test/bathroom/artifacts/"
                          "bathroom_air_conditioner_1/actions/hvacMode",
            "parameters": {"value": 3}}
        assert not result.get("impossible")

    def test_a_boolean_is_typed(self, isa_graph):
        result = plan_explicit_goal(isa_graph, _goal(
            affordance_class="homeont:SetOnOffCommand", affordance_name="onOff",
            target_value_determined="true"))
        assert result["tree"]["parameters"] == {"value": True}

    def test_a_class_from_another_action_is_rejected(self, isa_graph):
        # The titles name hvacMode, the class is onOff's: the structure
        # contradicts itself, and is not planned.
        result = plan_explicit_goal(isa_graph, _goal(
            affordance_class="homeont:SetOnOffCommand", affordance_name="hvacMode",
            target_value_determined="3"))
        assert result["impossible"]

    def test_a_value_outside_the_enum_is_impossible(self, isa_graph):
        result = plan_explicit_goal(isa_graph, _goal(
            affordance_class="homeont:SetModeCommand", affordance_name="hvacMode",
            target_value_determined="2"))
        assert result["impossible"] and result["tree"] is None
        assert "permitted values" in result["explanation"]

    def test_a_value_of_the_wrong_type_is_impossible(self, isa_graph):
        result = plan_explicit_goal(isa_graph, _goal(
            affordance_class="homeont:SetModeCommand", affordance_name="hvacMode",
            target_value_determined="cool"))
        assert result["impossible"]

    def test_no_match_is_impossible(self, isa_graph):
        result = plan_explicit_goal(isa_graph, _goal(
            location_class="homeont:Kitchen",
            affordance_class="homeont:SetModeCommand", affordance_name="hvacMode",
            target_value_determined="3"))
        assert result["impossible"]
        assert "homeont:Kitchen" in result["explanation"]


ODD_HOME = """
@prefix hmas: <https://purl.org/hmas/> .
@prefix homeont: <http://example.org/homeont/> .
@prefix saref: <https://saref.etsi.org/core/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix js: <https://www.w3.org/2019/wot/json-schema#> .

<http://h/office/lamp#artifact> a hmas:Artifact ; td:title "lamp" ;
    td:hasActionAffordance [
        a td:ActionAffordance, saref:SetLevelCommand ; td:title "colour" ;
        td:hasForm [ hctl:hasTarget <http://h/lamp/colour> ] ;
        td:hasInputSchema [ a js:ObjectSchema ;
            js:properties [ a js:IntegerSchema ; js:propertyName "brightness" ;
                            js:minimum 0 ; js:maximum 100 ] ] ] ,
    [
        a td:ActionAffordance, saref:OnCommand ; td:title "power" ;
        td:hasForm [ hctl:hasTarget <http://h/lamp/power-a> ] ] ,
    [
        a td:ActionAffordance, saref:OnCommand ; td:title "power" ;
        td:hasForm [ hctl:hasTarget <http://h/lamp/power-b> ] ] .
"""


@pytest.fixture(scope="module")
def odd_graph() -> Graph:
    return Graph().parse(data=ODD_HOME, format="turtle")


def _lamp(**fields):
    base = {"goal_specificity": "explicit", "goal_kind": "achievement",
            "goal_effect": "set", "artifact_name": "lamp"}
    return GoalSpec.from_dict({**base, **fields})


class TestParameterizedAndAmbiguous:
    def test_a_parameterized_call_sends_the_named_parameter(self, odd_graph):
        result = plan_explicit_goal(odd_graph, _lamp(
            affordance_name="colour", parameter_name="brightness",
            target_value_determined="40"))
        assert result["tree"]["parameters"] == {"brightness": 40}

    def test_bounds_are_checked(self, odd_graph):
        result = plan_explicit_goal(odd_graph, _lamp(
            affordance_name="colour", parameter_name="brightness",
            target_value_determined="140"))
        assert result["impossible"] and "range" in result["explanation"]

    def test_a_parameterized_call_without_a_parameter_name_is_impossible(self, odd_graph):
        result = plan_explicit_goal(odd_graph, _lamp(
            affordance_name="colour", parameter_name="NA", target_value_determined="40"))
        assert result["impossible"]

    def test_two_actions_sharing_names_are_not_guessed_between(self, odd_graph):
        result = plan_explicit_goal(odd_graph, _lamp(
            affordance_name="power", target_value_determined="true"))
        assert result["impossible"] and "2 actions" in result["explanation"]


class TestEnvelopes:
    def test_entries_keep_their_order_and_flag_impossible(self):
        envelope = envelope_plan_entries(
            [{"tree": {"type": "action"}}, {"tree": None, "impossible": True}], "ws")
        assert [bool(p["tree"]) for p in envelope["plans"]] == [True, False]
        assert envelope["impossible"] is True

    def test_a_single_tree_reply_covers_its_intents(self):
        intents = [Intent("a"), Intent("b")]
        (entry,) = plan_entries_from_envelope({"tree": {"type": "action"},
                                               "explanation": "x"}, intents)
        assert entry["intents"] == [{"text_intent": "a"}, {"text_intent": "b"}]

    def test_an_error_reply_becomes_one_entry_per_intent(self):
        entries = plan_entries_from_envelope(
            {"error": "plan_generation_failed", "detail": "boom"}, [Intent("a"), Intent("b")])
        assert [e["intent"] for e in entries] == [{"text_intent": "a"}, {"text_intent": "b"}]
        assert all(e["tree"] is None for e in entries)


# Two devices that share a title and an action name, told apart only by class:
# an on/off light and a colour light (which homeont makes a dimmable light).
TWO_LIGHTS = """
@prefix hmas: <https://purl.org/hmas/> .
@prefix homeont: <http://example.org/homeont/> .
@prefix saref: <https://saref.etsi.org/core/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix js: <https://www.w3.org/2019/wot/json-schema#> .

<http://h/onoff#artifact> a hmas:Artifact, homeont:OnOffLight ; td:title "light" ;
    td:hasActionAffordance [ a td:ActionAffordance, homeont:SetOnOffCommand ;
        td:title "onOff" ; td:hasForm [ hctl:hasTarget <http://h/onoff/onOff> ] ;
        td:hasInputSchema [ a js:BooleanSchema ] ] .

<http://h/colour#artifact> a hmas:Artifact, homeont:ColorLight ; td:title "light" ;
    td:hasActionAffordance [ a td:ActionAffordance, homeont:SetOnOffCommand ;
        td:title "onOff" ; td:hasForm [ hctl:hasTarget <http://h/colour/onOff> ] ;
        td:hasInputSchema [ a js:BooleanSchema ] ] .
"""


class TestArtifactClass:
    @pytest.fixture(scope="class")
    def lights(self):
        graph = Graph().parse(data=TWO_LIGHTS, format="turtle")
        close_types(graph)
        return graph

    def _light(self, **fields):
        base = {"goal_specificity": "explicit", "goal_kind": "achievement",
                "goal_effect": "set", "artifact_name": "light",
                "affordance_class": "homeont:SetOnOffCommand",
                "affordance_name": "onOff", "target_value_determined": "true"}
        return GoalSpec.from_dict({**base, **fields})

    def test_the_class_tells_same_titled_devices_apart(self, lights):
        colour = plan_explicit_goal(lights, self._light(artifact_class="homeont:ColorLight"))
        assert colour["tree"]["action_url"] == "http://h/colour/onOff"
        on_off = plan_explicit_goal(lights, self._light(artifact_class="homeont:OnOffLight"))
        assert on_off["tree"]["action_url"] == "http://h/onoff/onOff"

    def test_a_superclass_covers_its_subclasses(self, lights):
        # homeont: a ColorLight is a DimmableLight, so asking for a dimmable
        # light named "light" finds the colour light -- and only it, since the
        # on/off light is not dimmable.
        result = plan_explicit_goal(lights, self._light(artifact_class="homeont:DimmableLight"))
        assert result["tree"]["action_url"] == "http://h/colour/onOff"

    def test_without_the_class_the_names_alone_are_ambiguous(self, lights):
        result = plan_explicit_goal(lights, self._light())
        assert result["impossible"] and "2 actions" in result["explanation"]

    @needs_corpus
    def test_a_wrong_class_matches_nothing(self, isa_graph):
        result = plan_explicit_goal(isa_graph, _goal(
            artifact_class="homeont:HeatPump",
            affordance_class="homeont:SetModeCommand", affordance_name="hvacMode",
            target_value_determined="3"))
        assert result["impossible"]
        assert "(a homeont:HeatPump)" in result["explanation"]


@needs_corpus
class TestGenericClasses:
    """A class more generic than the TD's still finds the action, once."""

    @pytest.mark.parametrize("fields", [
        {"artifact_class": "saref:Device"},
        {"artifact_class": "saref:HVAC"},
        {"affordance_class": "saref:OnCommand"},
        {"location_class": "homeont:BuildingSpace"},
    ])
    def test_a_generic_class_matches(self, isa_graph, fields):
        goal = _goal(**{"affordance_class": "homeont:SetOnOffCommand",
                        "affordance_name": "onOff",
                        "target_value_determined": "true", **fields})
        result = plan_explicit_goal(isa_graph, goal)
        assert result["tree"] is not None, result["explanation"]


class TestGenericClassOnTwoLights:
    def test_a_generic_class_cannot_tell_same_titled_devices_apart(self):
        graph = Graph().parse(data=TWO_LIGHTS, format="turtle")
        close_types(graph)
        goal = GoalSpec.from_dict({
            "goal_specificity": "explicit", "goal_kind": "achievement",
            "goal_effect": "set", "artifact_name": "light",
            "artifact_class": "saref:Device", "affordance_name": "onOff",
            "target_value_determined": "true"})
        result = plan_explicit_goal(graph, goal)
        assert result["impossible"] and "2 actions" in result["explanation"]



# Home Assistant's shape: the action is the value, and takes no input.
SWITCH = """
@prefix hmas: <https://purl.org/hmas/> .
@prefix homeont: <http://example.org/homeont/> .
@prefix saref: <https://saref.etsi.org/core/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .

<http://h/switch#artifact> a hmas:Artifact ; td:title "desk_switch" ;
    td:hasActionAffordance [ a td:ActionAffordance, saref:OnCommand ;
        td:title "SwitchTurnOn" ;
        td:hasForm [ hctl:hasTarget <http://h/desk_switch/ha/switch/turn_on> ] ] .
"""


class TestNoInputActions:
    def _switch_goal(self, **fields):
        base = {"goal_specificity": "explicit", "goal_kind": "achievement",
                "goal_effect": "set", "artifact_name": "desk_switch",
                "affordance_class": "saref:OnCommand", "affordance_name": "SwitchTurnOn",
                "parameter_name": "NA", "target_value_text": "on"}
        return GoalSpec.from_dict({**base, **fields})

    def test_an_action_without_input_is_planned_with_an_empty_body(self):
        graph = Graph().parse(data=SWITCH, format="turtle")
        result = plan_explicit_goal(graph, self._switch_goal())
        assert result["tree"] == {"type": "action", "name": "desk_switch.SwitchTurnOn",
                                  "action_url": "http://h/desk_switch/ha/switch/turn_on"}
        assert "parameters" not in result["tree"]

    def test_a_value_given_for_an_action_without_input_is_ignored(self):
        graph = Graph().parse(data=SWITCH, format="turtle")
        result = plan_explicit_goal(graph, self._switch_goal(target_value_determined="true"))
        assert result["tree"] is not None and "parameters" not in result["tree"]

    @needs_corpus
    def test_an_action_with_input_but_no_value_is_not_deterministic(self, isa_graph):
        result = plan_explicit_goal(isa_graph, _goal(
            affordance_class="homeont:SetModeCommand", affordance_name="hvacMode",
            target_value_text="cooling mode", target_value_determined=None))
        assert result is None
