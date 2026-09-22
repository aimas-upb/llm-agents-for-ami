"""An artifact's semantic types describe the device, not its surroundings.

The extraction used to walk every subject in the artifact graph, so a bathroom
air conditioner was typed with the room it stood in, every property it exposed
and every command it accepted -- 15 homeont classes where one was correct. These
tests pin the device's own types.

The graph is built offline from a SimuHome episode -- no simulator, no SHTD, no
LLM -- so the whole file runs in the unit suite.
"""

import json
from pathlib import Path

import pytest
from rdflib import Graph, URIRef

from ami_agents.environment.integration.integration_engine import IIntegrationEngine
from ami_agents.shared.utils.namespaces import domain_types

REPO = Path(__file__).resolve().parents[2]
EPISODE = (REPO.parent / "SimuHome" / "data" / "benchmark"
           / "qt1_feasible_seed_1.json")
BASE = "http://localhost:8097/workspaces/test"

pytestmark = pytest.mark.skipif(
    not EPISODE.is_file(), reason="SimuHome benchmark corpus not available"
)


@pytest.fixture(scope="module")
def builder():
    from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD

    config = json.loads(EPISODE.read_text())["initial_home_config"]
    return SimuHomeTD("http://localhost:8097", "test", config)


def semantic_types(builder, room: str, device: str):
    """The types the integration engine would store for one artifact."""
    graph = builder.artifact(room, device)
    uri = URIRef(f"{BASE}/{room}/artifacts/{device}")
    return IIntegrationEngine.extract_semantic_types(graph, uri)


class TestScopedToTheDevice:
    def test_device_family_is_present(self, builder):
        types = semantic_types(builder, "bathroom", "bathroom_air_conditioner_1")
        assert "homeont:AirConditioner" in types

    def test_the_containing_room_is_not_a_type_of_the_device(self, builder):
        """The artifact graph describes the room too; the device is not one."""
        types = semantic_types(builder, "bathroom", "bathroom_air_conditioner_1")
        assert "homeont:Bathroom" not in types
        assert "homeont:BuildingSpace" not in types

    def test_properties_and_commands_are_not_types_of_the_device(self, builder):
        """An affordance's class belongs to the affordance, not the artifact."""
        types = semantic_types(builder, "bathroom", "bathroom_air_conditioner_1")
        for leaked in ("homeont:AirTemperature", "homeont:AirConditionerOnOff",
                       "homeont:FanSpeed", "homeont:SetOnOffCommand",
                       "homeont:SetModeCommand"):
            assert leaked not in types, leaked

    def test_exactly_one_domain_type(self, builder):
        """What the one consumer of this list actually asks for."""
        types = semantic_types(builder, "bathroom", "bathroom_air_conditioner_1")
        assert domain_types(types) == ["homeont:AirConditioner"]

    def test_a_second_device_family(self, builder):
        types = semantic_types(builder, "kitchen", "kitchen_freezer_1")
        assert domain_types(types) == ["homeont:Freezer"]

    def test_asserted_ancestors_are_kept(self, builder):
        """`saref:Appliance` is stated in the TD, so it stays in the list.

        `domain_types()` filters it out for callers that want HomeOnt only.
        """
        types = semantic_types(builder, "kitchen", "kitchen_freezer_1")
        assert "saref:Appliance" in types

    def test_types_are_curies_not_iris(self, builder):
        """Stored short, so `domain_types` and prompt text agree on one form."""
        types = semantic_types(builder, "kitchen", "kitchen_freezer_1")
        assert all(not t.startswith("http") for t in types), types


class TestThingNodeResolution:
    def test_resolved_by_type_not_by_uri_fragment(self, builder):
        """The Thing is a different node from the artifact's own URI.

        The artifact URI carries only `hmas:ResourceProfile`; the device family
        is on the Thing it profiles. Resolving by type rather than by appending
        `#artifact` keeps this working if the serialisation changes.
        """
        graph = builder.artifact("bathroom", "bathroom_air_conditioner_1")
        uri = URIRef(f"{BASE}/bathroom/artifacts/bathroom_air_conditioner_1")

        thing = IIntegrationEngine.extract_thing_node(graph, uri)
        assert thing != uri
        assert domain_types(
            IIntegrationEngine.extract_semantic_types(graph, uri)
        ) == ["homeont:AirConditioner"]

    def test_falls_back_to_the_artifact_uri(self):
        """A graph with no Thing must not raise; it yields whatever is there."""
        graph = Graph()
        uri = URIRef("http://example.org/nothing")
        assert IIntegrationEngine.extract_thing_node(graph, uri) == uri
        assert IIntegrationEngine.extract_semantic_types(graph, uri) == []
