"""Resolving ENV_STATE classes to affordances, against a real TD graph.

The graph is built offline from a SimuHome episode -- no simulator, no SHTD, no
LLM -- so every outcome the resolver can produce is pinned by a test.

Skips if the SimuHome corpus is not checked out beside this repo.
"""

import json
from pathlib import Path

import pytest
from rdflib import Graph

from ami_agents.agents.env_explorer.utils.state_resolution import (
    StateOutcome,
    build_query,
    load_vocabulary,
    resolve_state_request,
)

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


class TestResolvedAffordance:
    def test_device_property_scoped_by_room_and_device(self, graph):
        """"How cold is the freezer in the kitchen?" -- all three slots bound."""
        result = resolve_state_request(
            graph,
            location_class="homeont:Kitchen",
            device_class="homeont:Freezer",
            device_property={"class": "homeont:CompartmentTemperature"},
        )
        assert result.outcome is StateOutcome.RESOLVED
        assert len(result.affordances) == 1
        affordance = result.affordances[0]
        assert affordance.target.endswith(
            "/kitchen/artifacts/kitchen_freezer_1/properties/temperature")

    def test_environment_variable_read_from_a_sensing_device(self, graph):
        """"How warm is the bathroom?" -- no device named.

        Answered by the air conditioner's own thermostat reading, which is an
        observation of the room's air.
        """
        result = resolve_state_request(
            graph,
            location_class="homeont:Bathroom",
            environment_variable={"class": "homeont:AirTemperature"},
        )
        assert result.outcome is StateOutcome.RESOLVED
        assert len(result.affordances) == 1
        assert "air_conditioner" in result.affordances[0].artifact

    def test_no_duplicate_rows_from_the_class_hierarchy(self, graph):
        """One device matching at two levels must yield one affordance.

        `rdfs:subClassOf*` binds the device class at every level, so an air
        conditioner matches as both homeont:AirConditioner and saref:HVAC.
        """
        result = resolve_state_request(
            graph,
            location_class="homeont:Bathroom",
            environment_variable={"class": "homeont:AirTemperature"},
        )
        targets = [a.target for a in result.affordances]
        assert len(targets) == len(set(targets))


class TestRetrievalTuple:
    """Each affordance carries the names the retrieval task speaks."""

    def test_names_come_from_the_graph(self, graph):
        result = resolve_state_request(
            graph,
            location_class="homeont:LivingRoom",
            device_class="homeont:Tv",
            device_property={"class": "homeont:ChannelCurrentChannel"},
        )
        assert result.outcome is StateOutcome.RESOLVED
        affordance = result.affordances[0]
        assert affordance.artifact_name == "living_room_tv_1"
        assert affordance.workspace_name == "Living Room"
        assert affordance.affordance_name == "currentChannel"
        assert affordance.affordance_type == "homeont:ChannelCurrentChannel"

    def test_artifact_type_is_the_device_class(self, graph):
        """Never null: the field exists to say what kind of device answered."""
        result = resolve_state_request(
            graph, location_class="homeont:Kitchen",
            device_property={"class": "homeont:OnOff"},
        )
        by_name = {a.artifact_name: a.artifact_type for a in result.affordances}
        assert by_name["kitchen_on_off_light_1"] == "homeont:OnOffLight"
        assert by_name["kitchen_dishwasher_1"] == "homeont:Dishwasher"
        assert all(t is not None for t in by_name.values())

    def test_an_object_valued_property_is_one_affordance(self, graph):
        """`currentChannel` has 7 sub-fields; it is still one property.

        Binding them in the query would return seven rows and make the outcome
        check see seven properties where there is one.
        """
        result = resolve_state_request(
            graph, device_property={"class": "homeont:ChannelCurrentChannel"})
        assert len(result.affordances) == 1
        assert result.affordances[0].parameter_name is None

    def test_workspace_name_absent_when_no_location_asked(self, graph):
        """Not a missing label -- the query never bound a space."""
        result = resolve_state_request(
            graph, device_class="homeont:Freezer",
            device_property={"class": "homeont:CompartmentTemperature"})
        assert result.affordances[0].workspace_name is None
        assert result.affordances[0].artifact_name == "kitchen_freezer_1"

    def test_value_is_absent_until_read(self, graph):
        """An unread affordance is visibly unread, not null-valued."""
        result = resolve_state_request(
            graph, location_class="homeont:Bathroom",
            environment_variable={"class": "homeont:AirTemperature"})
        assert "value" not in result.affordances[0].as_dict()


class TestNoAffordance:
    def test_nothing_senses_illuminance(self, graph):
        """The majority path: no device family in the corpus reports lux.

        The room workspace does expose an illuminance property, but that is
        simulator ground truth and the resolver must never reach for it.
        """
        result = resolve_state_request(
            graph,
            location_class="homeont:Kitchen",
            environment_variable={"class": "homeont:Illuminance"},
        )
        assert result.outcome is StateOutcome.NONE
        assert result.affordances == []
        assert "homeont:Kitchen" in result.detail
        assert "Illuminance" in result.detail

    def test_no_result_comes_from_a_room_level_property(self, graph):
        """Every answer is a device's affordance, never a room's property."""
        for variable in ("homeont:AirTemperature", "homeont:Illuminance",
                         "homeont:RelativeHumidity",
                         "homeont:Pm10MassConcentration"):
            result = resolve_state_request(
                graph, location_class="homeont:Kitchen",
                environment_variable={"class": variable},
            )
            for affordance in result.affordances:
                assert "/artifacts/" in affordance.target, affordance.target


class TestIndeterminate:
    def test_no_property_named_at_all(self, graph):
        result = resolve_state_request(
            graph, location_class="homeont:Kitchen", device_class="homeont:Freezer",
        )
        assert result.outcome is StateOutcome.INDETERMINATE
        assert result.affordances == []

    def test_decided_without_touching_the_graph(self):
        """A parse failure needs no query -- an empty graph is enough."""
        result = resolve_state_request(Graph())
        assert result.outcome is StateOutcome.INDETERMINATE


class TestMismatched:
    def test_over_broad_property_class(self, graph):
        """A branch root matches many unrelated properties."""
        result = resolve_state_request(
            graph,
            location_class="homeont:UtilityRoom",
            device_property={"class": "homeont:ActuatableDeviceProperty"},
        )
        assert result.outcome is StateOutcome.MISMATCHED
        assert len(result.property_classes) > 1

    def test_generic_onoff_matches_every_switchable_device(self, graph):
        """The worked example behind the prompt's specificity rule."""
        generic = resolve_state_request(
            graph, location_class="homeont:Bathroom",
            device_property={"class": "homeont:OnOff"},
        )
        assert generic.outcome is StateOutcome.MISMATCHED

        specific = resolve_state_request(
            graph, location_class="homeont:Bathroom",
            device_property={"class": "homeont:OnOffLightOnOff"},
        )
        assert specific.outcome is StateOutcome.RESOLVED
        assert len(specific.affordances) == 1


class TestVocabularyIsRequired:
    def test_saref_is_needed_for_the_device_constraint(self):
        """Without SAREF the path to saref:Device does not close.

        Asserted so a missing vendored file fails loudly here rather than
        masquerading as `no_affordance` at runtime.
        """
        from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD

        config = json.loads(EPISODE.read_text())["initial_home_config"]
        builder = SimuHomeTD("http://localhost:8097", "test", config)
        g = Graph()
        for room in config["rooms"]:
            g += builder.room_workspace(room)
            for device in config["rooms"][room].get("devices", []):
                g += builder.artifact(room, device["device_id"])
        g.parse(
            str(Path(__file__).resolve().parents[2] / "ami_agents" / "shared"
                / "ontologies" / "homeont.ttl"),
            format="turtle",
        )  # homeont only, no SAREF

        result = resolve_state_request(
            g, location_class="homeont:Bathroom",
            environment_variable={"class": "homeont:AirTemperature"},
        )
        assert result.outcome is StateOutcome.NONE


class TestQueryComposition:
    def test_slots_are_omitted_when_absent(self):
        with_location = build_query(
            location_class="homeont:Kitchen",
            property_class="homeont:CompartmentTemperature")
        assert "homeont:isSpaceOfWorkspace" in with_location

        without_location = build_query(
            device_class="homeont:Freezer",
            property_class="homeont:CompartmentTemperature")
        assert "homeont:isSpaceOfWorkspace" not in without_location
        assert "a/rdfs:subClassOf* homeont:Freezer" in without_location

    def test_device_class_is_a_path_not_an_exact_type(self):
        """An intermediate class must still reach its subclasses.

        `a homeont:Freezer` matches only an exact assertion; a parser answering
        with `saref:Appliance` would then match nothing at all.
        """
        query = build_query(device_class="saref:Appliance",
                            property_class="homeont:OnOff")
        assert "?artifact a/rdfs:subClassOf* saref:Appliance ." in query

    def test_device_ness_is_asked_as_a_path_check(self):
        """Binding `?devClass` returns one row per ancestor; EXISTS returns one."""
        query = build_query(property_class="homeont:AirTemperature")
        assert "FILTER EXISTS { ?artifact a/rdfs:subClassOf* saref:Device }" in query
        assert "?devClass" not in query
        assert "DISTINCT" in query.split("WHERE")[0]

    def test_names_are_required_not_optional(self):
        """A name is part of the tuple, so a TD lacking one is malformed."""
        query = build_query(location_class="homeont:Kitchen",
                            property_class="homeont:OnOff")
        assert "?artifact td:title ?artTitle" in query
        assert "?space rdfs:label ?spaceLabel ." in query
        assert "OPTIONAL" not in query
