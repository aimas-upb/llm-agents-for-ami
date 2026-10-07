"""Resolving ENV_CAPABILITIES classes to affordances, against a real TD graph.

The graph is built offline from a SimuHome episode -- no simulator, no SHTD, no
LLM -- so every slot combination the resolver dispatches on is pinned by a test.
Nothing here reads a device: a capability is a fact about the TDs.

Skips if the SimuHome corpus is not checked out beside this repo.
"""

import json
from pathlib import Path

import pytest
from rdflib import Graph

from ami_agents.agents.env_explorer.behaviors.capability_query_behaviour import (
    capability_response,
)
from ami_agents.agents.env_explorer.utils.capability_resolution import (
    CapabilityOutcome,
    resolve_capability_request,
)
from ami_agents.agents.env_explorer.utils.state_resolution import load_vocabulary

REPO = Path(__file__).resolve().parents[2]
EPISODE = (REPO.parent / "SimuHome" / "data" / "benchmark"
           / "qt1_feasible_seed_1.json")

pytestmark = pytest.mark.skipif(
    not EPISODE.is_file(), reason="SimuHome benchmark corpus not available"
)


@pytest.fixture(scope="module")
def graph() -> Graph:
    """Discovered TDs + the vocabulary, as the resolver sees them."""
    from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD

    config = json.loads(EPISODE.read_text())["initial_home_config"]
    builder = SimuHomeTD("http://localhost:8097", "test", config)

    g = Graph()
    for room in config["rooms"]:
        g += builder.room_workspace(room)
        for device in config["rooms"][room].get("devices", []):
            g += builder.artifact(room, device["device_id"])
    return load_vocabulary(g)


def _names(result):
    return sorted({e.artifact_name for e in result.entries})


class TestDeviceProperty:
    def test_a_general_class_reaches_device_specific_leaves(self, graph):
        """"Can you dim anything?" -- LevelControlBrightness, no device named.

        The TDs type brightness as DimmableLightBrightness and TvBrightness;
        only `rdfs:subClassOf*` connects the parser's class to them.
        """
        result = resolve_capability_request(
            graph, device_property={"class": "homeont:LevelControlBrightness"})
        assert result.outcome is CapabilityOutcome.FOUND
        types = {e.affordance_type for e in result.entries}
        assert types == {"homeont:DimmableLightBrightness", "homeont:TvBrightness"}

    def test_an_actuatable_class_is_reported_as_changeable(self, graph):
        result = resolve_capability_request(
            graph, device_property={"class": "homeont:LevelControlBrightness"})
        assert {e.property_branch for e in result.entries} == {"actuatable"}
        assert {e.affordance_kind for e in result.entries} == {"property"}

    def test_permitted_bounds_come_from_the_schema(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:DimmableLight",
            device_property={"class": "homeont:LevelControlBrightness"})
        bounds = result.entries[0].permitted_values
        assert bounds == {"minimum": 1, "maximum": 254}

    def test_an_enum_carries_its_meaning(self, graph):
        """"What fan modes does the purifier have?" -- the codes and their gloss."""
        result = resolve_capability_request(
            graph, device_class="homeont:AirPurifier",
            device_property={"class": "homeont:FanControlFanMode"})
        assert result.outcome is CapabilityOutcome.FOUND
        values = result.entries[0].permitted_values
        assert values["enum"] == [0, 1, 2, 3, 4, 5, 6]
        assert "1 = Low" in values["meaning"]

    def test_a_capability_class_is_reported_as_supported(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:AirPurifier",
            device_property={"class": "homeont:FanControlFanModeSequence"})
        assert {e.property_branch for e in result.entries} == {"capability"}

    def test_the_room_is_named_even_when_not_asked_for(self, graph):
        result = resolve_capability_request(
            graph, device_property={"class": "homeont:LevelControlBrightness"})
        assert all(e.workspace_name for e in result.entries)


class TestCommand:
    def test_a_saref_command_reaches_its_homeont_specialisation(self, graph):
        """SetOnOffCommand specialises On/Off/Toggle; asking for On finds it."""
        result = resolve_capability_request(
            graph, location_class="homeont:Kitchen",
            command={"class": "saref:OnCommand"})
        assert result.outcome is CapabilityOutcome.FOUND
        assert {e.affordance_type for e in result.entries} == {"homeont:SetOnOffCommand"}
        assert {e.affordance_kind for e in result.entries} == {"action"}
        assert "kitchen_on_off_light_1" in _names(result)


class TestPropertyOrCommand:
    """A capability may be modelled as a changeable property, as a command, or
    both. When the parser fills both, both routes run and the answer is their
    union."""

    def test_both_routes_contribute(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:DimmableLight",
            device_property={"class": "homeont:LevelControlBrightness"},
            command={"class": "saref:OnCommand"})
        assert result.outcome is CapabilityOutcome.FOUND
        assert {e.affordance_kind for e in result.entries} == {"property", "action"}

    def test_the_command_alone_can_answer(self, graph):
        """The device has no such property, but has the command."""
        result = resolve_capability_request(
            graph, device_class="homeont:AirPurifier",
            device_property={"class": "homeont:LevelControlBrightness"},
            command={"class": "saref:OnCommand"})
        assert result.outcome is CapabilityOutcome.FOUND
        assert {e.affordance_kind for e in result.entries} == {"action"}

    def test_the_property_alone_can_answer(self, graph):
        """The device has the property, but no such command."""
        result = resolve_capability_request(
            graph, device_class="homeont:DimmableLight",
            device_property={"class": "homeont:LevelControlBrightness"},
            command={"class": "saref:OpenCommand"})
        assert result.outcome is CapabilityOutcome.FOUND
        assert {e.affordance_kind for e in result.entries} == {"property"}

    def test_neither_is_a_lack(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:AirPurifier",
            device_property={"class": "homeont:LevelControlBrightness"},
            command={"class": "saref:OpenCommand"})
        assert result.outcome is CapabilityOutcome.DEVICE_LACKS
        assert " or " in result.detail


class TestActsUpon:
    """A command with a property that is not actuatable answers only through
    the actions that state `saref:actsUpon` a property of that class."""

    def test_every_action_states_what_it_changes(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:DimmableLight",
            command={"class": "saref:SetAbsoluteLevelCommand"})
        assert result.outcome is CapabilityOutcome.FOUND
        for entry in result.entries:
            assert entry.acts_upon, entry.affordance_name
            assert entry.acts_upon[0]["affordance_name"] == entry.affordance_name

    def test_a_reported_state_nothing_acts_upon_is_a_lack(self, graph):
        """The dishwasher reports its operational state; no command changes it."""
        result = resolve_capability_request(
            graph, device_class="homeont:Dishwasher",
            device_property={"class": "homeont:OperationalStateOperationalState"},
            command={"class": "saref:Command"})
        assert result.outcome is CapabilityOutcome.DEVICE_LACKS
        assert result.query["property_branch"] == "state"

    def test_an_actuatable_property_still_unions(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:DimmableLight",
            device_property={"class": "homeont:LevelControlBrightness"},
            command={"class": "saref:OnCommand"})
        assert result.query["property_branch"] == "actuatable"
        assert {e.affordance_kind for e in result.entries} == {"property", "action"}


class TestEnvironmentVariable:
    def test_sensing_finds_the_reporting_device(self, graph):
        """"Can you tell how warm the bathroom is?" -- no command: sensing."""
        result = resolve_capability_request(
            graph, location_class="homeont:Bathroom",
            environment_variable={"class": "homeont:AirTemperature"})
        assert result.outcome is CapabilityOutcome.FOUND
        assert _names(result) == ["bathroom_air_conditioner_1"]
        assert result.entries[0].property_branch == "environment"

    def test_any_command_with_a_variable_finds_its_effects(self, graph):
        """"Can anything cool the living room?" -- saref:Command + variable."""
        result = resolve_capability_request(
            graph, location_class="homeont:LivingRoom",
            environment_variable={"class": "homeont:AirTemperature"},
            command={"class": "saref:Command"})
        assert result.outcome is CapabilityOutcome.FOUND
        assert _names(result) == ["living_room_air_conditioner_1"]
        assert {e.effect_on for e in result.entries} == {"homeont:AirTemperature"}
        assert {e.affordance_kind for e in result.entries} == {"action"}

    def test_a_specific_command_narrows_the_effects(self, graph):
        everything = resolve_capability_request(
            graph, location_class="homeont:LivingRoom",
            environment_variable={"class": "homeont:AirTemperature"},
            command={"class": "saref:Command"})
        modes = resolve_capability_request(
            graph, location_class="homeont:LivingRoom",
            environment_variable={"class": "homeont:AirTemperature"},
            command={"class": "homeont:SetModeCommand"})
        assert 0 < len(modes.entries) < len(everything.entries)
        assert {e.affordance_type for e in modes.entries} == {"homeont:SetModeCommand"}


class TestMetadata:
    def test_make_is_answered_from_the_thing(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:Refrigerator",
            device_property={"class": "schema:manufacturer"})
        assert result.outcome is CapabilityOutcome.FOUND
        assert result.entries[0].manufacturer == "LG Electronics"
        assert result.entries[0].affordance_name is None


class TestInventory:
    def test_a_room_alone_lists_what_is_there(self, graph):
        """"What can you control in the kitchen?" -- an inventory is the answer."""
        result = resolve_capability_request(graph, location_class="homeont:Kitchen")
        assert result.outcome is CapabilityOutcome.FOUND
        assert "kitchen_on_off_light_1" in _names(result)
        assert {e.affordance_kind for e in result.entries} == {"action", "property"}

    def test_only_domain_types_are_listed(self, graph):
        """WebSub subscription actions are plumbing, not capabilities."""
        result = resolve_capability_request(graph, location_class="homeont:Kitchen")
        assert all(e.affordance_type.startswith(("homeont:", "saref:"))
                   for e in result.entries)


class TestNegativeOutcomes:
    def test_a_named_device_without_the_capability(self, graph):
        """"Can the purifier dim?" -- the purifier is there, and cannot."""
        result = resolve_capability_request(
            graph, device_class="homeont:AirPurifier",
            device_property={"class": "homeont:LevelControlBrightness"})
        assert result.outcome is CapabilityOutcome.DEVICE_LACKS
        assert all(e.artifact_type == "homeont:AirPurifier" for e in result.entries)
        assert all(e.affordance_name is None for e in result.entries)

    def test_no_device_named_and_nothing_found_is_none(self, graph):
        result = resolve_capability_request(
            graph, location_class="homeont:Kitchen",
            device_property={"class": "homeont:LevelControlBrightness"})
        assert result.outcome is CapabilityOutcome.NONE
        assert result.entries == []

    def test_nothing_named_is_indeterminate(self, graph):
        result = resolve_capability_request(graph)
        assert result.outcome is CapabilityOutcome.INDETERMINATE


class TestResponse:
    def test_echoes_intent_and_performative(self, graph):
        result = resolve_capability_request(
            graph, location_class="homeont:Kitchen",
            command={"class": "saref:OnCommand"})
        response = capability_response(
            result, {"text_intent": "can you switch things on in the kitchen",
                     "request_performative": "query_if"})
        assert response["outcome"] == "found"
        assert response["request_performative"] == "query_if"
        assert response["text_intent"].startswith("can you")
        json.dumps(response)

    def test_performative_defaults_to_query(self, graph):
        response = capability_response(resolve_capability_request(graph), {})
        assert response["request_performative"] == "query"


class TestNamedDevice:
    """Goal structuring narrows a lookup to the device a goal names."""

    def test_the_name_narrows_the_answer_to_that_device(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:AirConditioner",
            command={"class": "homeont:SetModeCommand"},
            artifact_name="bathroom_air_conditioner_1")
        assert result.outcome == CapabilityOutcome.FOUND
        assert _names(result) == ["bathroom_air_conditioner_1"]
        assert result.query["artifact_name"] == "bathroom_air_conditioner_1"

    def test_entries_carry_the_room_class(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:AirConditioner",
            command={"class": "homeont:SetModeCommand"},
            artifact_name="bathroom_air_conditioner_1")
        assert {e.workspace_class for e in result.entries} == {"homeont:Bathroom"}

    def test_a_name_nothing_matches_is_none(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:AirConditioner",
            command={"class": "homeont:SetModeCommand"},
            artifact_name="garage_air_conditioner_9")
        assert result.outcome == CapabilityOutcome.NONE
        assert result.entries == []
