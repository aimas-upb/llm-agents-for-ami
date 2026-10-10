"""Specific capability contexts: one goal's slice of the TD graph, for a small LLM.

The SimuHome graph is built as the InteractionSolver builds it -- EnvExplorer's
detailed dump, closed under subclass inference. A hand-written graph covers an
object (parameterized) input and two devices sharing a title.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdflib import Graph

from ami_agents.agents.env_explorer.utils.data_formatting import (
    format_capabilities_detailed_rdf,
)
from ami_agents.agents.interaction_solver.utils.specific_capability_context import (
    DIRECT_VALUE_KEY,
    build_specific_context,
    find_artifacts,
    goal_brief,
)
from ami_agents.shared.models.goal_structure import GoalSpec
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
    close_types(graph)
    return graph


AC = {"goal_specificity": "explicit", "goal_kind": "achievement", "goal_effect": "set",
      "location_class": "homeont:Bathroom", "artifact_class": "homeont:AirConditioner",
      "artifact_name": "bathroom_air_conditioner_1",
      "affordance_class": "homeont:SetModeCommand", "affordance_name": "hvacMode",
      "parameter_name": "NA", "target_value_text": "cooling mode"}


def _goal(**fields):
    return GoalSpec.from_dict({**AC, **fields})


def _by_name(context, name):
    return next(a for a in context.affordances if a["name"] == name)


@needs_corpus
class TestExplicitAchievement:
    def test_the_context_is_the_named_device_only(self, isa_graph):
        context = build_specific_context(isa_graph, _goal())
        assert context.artifacts == ["bathroom_air_conditioner_1"]
        assert {a["artifact_name"] for a in context.affordances} == {"bathroom_air_conditioner_1"}

    def test_it_holds_the_device_actions_and_readable_properties(self, isa_graph):
        context = build_specific_context(isa_graph, _goal())
        kinds = {a["type"] for a in context.affordances}
        assert kinds == {"action_affordance", "property_affordance"}
        assert "hvacMode" in {a["name"] for a in context.affordances
                              if a["type"] == "action_affordance"}

    def test_protocol_actions_are_left_out(self, isa_graph):
        context = build_specific_context(isa_graph, _goal())
        assert not any("subscribe" in (a["name"] or "").lower() for a in context.affordances)

    def test_a_direct_value_input_is_shown_as_the_value_field(self, isa_graph):
        action = _by_name(build_specific_context(isa_graph, _goal()), "hvacMode")
        schema = action["input_schema"]
        assert schema["required"] == [DIRECT_VALUE_KEY]
        value = schema["properties"][DIRECT_VALUE_KEY]
        assert value["type"] == "integer"
        assert 3 in value["enum"] and "3 = Cool" in value["description"]

    def test_the_action_carries_its_target_url(self, isa_graph):
        action = _by_name(build_specific_context(isa_graph, _goal()), "hvacMode")
        assert action["target"].endswith("/bathroom_air_conditioner_1/actions/hvacMode")
        assert action["form"]["method"] == "POST"

    def test_types_shown_are_the_most_specific(self, isa_graph):
        context = build_specific_context(isa_graph, _goal())
        action = _by_name(context, "coolingSetpoint")
        assert action["semantic_types"] == ["saref:SetAbsoluteLevelCommand"]
        assert "homeont:AirConditionerCoolingSetpoint" in action["description"]

    def test_a_generic_class_still_finds_the_device(self, isa_graph):
        context = build_specific_context(isa_graph, _goal(artifact_class="saref:Device",
                                                          location_class=None))
        assert context.artifacts == ["bathroom_air_conditioner_1"]

    def test_a_wrong_class_or_room_finds_nothing(self, isa_graph):
        assert build_specific_context(isa_graph, _goal(artifact_class="homeont:HeatPump")) is None
        assert build_specific_context(isa_graph, _goal(location_class="homeont:Kitchen")) is None


class TestDispatch:
    def test_a_goal_kind_without_a_helper_has_no_context(self):
        graph = Graph()
        assert build_specific_context(graph, _goal(goal_kind="maintenance")) is None
        assert build_specific_context(graph, _goal(goal_specificity="incomplete")) is None

    def test_the_brief_names_device_action_and_value_words(self):
        brief = goal_brief(_goal())
        assert '"bathroom_air_conditioner_1"' in brief
        assert '"hvacMode"' in brief
        assert '"cooling mode"' in brief
        assert "parameter" not in brief           # "NA": no parameter to name


ODD_HOME = """
@prefix hmas: <https://purl.org/hmas/> .
@prefix homeont: <http://example.org/homeont/> .
@prefix saref: <https://saref.etsi.org/core/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix js: <https://www.w3.org/2019/wot/json-schema#> .

<http://h/colour#artifact> a hmas:Artifact, homeont:ColorLight ; td:title "light" ;
    td:hasActionAffordance [ a td:ActionAffordance, saref:SetLevelCommand ;
        td:title "colour" ; td:hasForm [ hctl:hasTarget <http://h/colour/colour> ] ;
        td:hasInputSchema [ a js:ObjectSchema ;
            js:properties [ a js:IntegerSchema ; js:propertyName "brightness" ;
                            js:minimum 0 ; js:maximum 100 ] ;
            js:required "brightness" ] ] .

<http://h/onoff#artifact> a hmas:Artifact, homeont:OnOffLight ; td:title "light" ;
    td:hasActionAffordance [ a td:ActionAffordance, homeont:SetOnOffCommand ;
        td:title "onOff" ; td:hasForm [ hctl:hasTarget <http://h/onoff/onOff> ] ;
        td:hasInputSchema [ a js:BooleanSchema ] ] .
"""


@pytest.fixture(scope="module")
def odd_graph() -> Graph:
    graph = Graph().parse(data=ODD_HOME, format="turtle")
    close_types(graph)
    return graph


class TestShapesAndAmbiguity:
    def _light(self, **fields):
        base = {"goal_specificity": "explicit", "goal_kind": "achievement",
                "goal_effect": "set", "artifact_name": "light"}
        return GoalSpec.from_dict({**base, **fields})

    def test_an_object_input_keeps_its_named_parameters(self, odd_graph):
        context = build_specific_context(odd_graph, self._light(artifact_class="homeont:ColorLight"))
        (action,) = context.affordances
        assert action["input_schema"] == {
            "type": "object",
            "properties": {"brightness": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["brightness"]}

    def test_the_class_picks_one_of_two_same_titled_devices(self, odd_graph):
        context = build_specific_context(odd_graph, self._light(artifact_class="homeont:OnOffLight"))
        assert [a["target"] for a in context.affordances] == ["http://h/onoff/onOff"]

    def test_two_matching_devices_have_no_specific_context(self, odd_graph):
        assert len(find_artifacts(odd_graph, name="light")) == 2
        assert build_specific_context(odd_graph, self._light()) is None
