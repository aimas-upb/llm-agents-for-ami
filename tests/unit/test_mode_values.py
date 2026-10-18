"""A Matter Mode cluster's CurrentMode is described by the device's own modes.

Matter types CurrentMode as a plain uint8; which number is which mode is the
device's declaration, in its mandatory, fixed SupportedModes list
({Label, Mode, ModeTags}). The device description states them the way it
states a spec enum -- `js:enum` plus a description -- with the tags named from
Matter's ModeTag enum:

    washer   SupportedModes [{"label": "Heavy", "mode": 2, "ModeTags": [16386]}, ...]
    schema   enum [0..5]; "... 2 = Heavy (mode tags: Heavy) ..."

    _mode_items       the device's modes from its SupportedModes, tags named;
                      nothing for any other attribute
    the washer's TD   both the currentMode action's input and the property's
                      output carry them (a benchmark home, built offline)
"""

import json
from pathlib import Path

import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import RDF

from ami_agents.shared.utils.namespaces import JS, TD

REPO = Path(__file__).resolve().parents[2]
EPISODE = REPO.parent / "SimuHome" / "data" / "benchmark" / "qt4-3_feasible_seed_45.json"
WASHER = "utility_room_laundry_washer_1"

SUPPORTED = [{"label": "Normal", "mode": 0, "ModeTags": [16384]},
             {"label": "Heavy", "mode": 2, "ModeTags": [{"value": 16386}]},
             {"label": "Quick Normal", "mode": 4, "ModeTags": [1, 16384]}]


@pytest.fixture(scope="module")
def td_class():
    from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD
    return SimuHomeTD


class TestModeItems:
    def test_modes_come_from_supported_modes_with_named_tags(self, td_class):
        """Tags given bare or as {"value": n} both name from the spec."""
        items = td_class._mode_items(td_class, "LaundryWasherMode", "CurrentMode",
                                     {"SupportedModes": SUPPORTED})
        assert items == [{"value": 0, "name": "Normal", "tags": ["Normal"]},
                         {"value": 2, "name": "Heavy", "tags": ["Heavy"]},
                         {"value": 4, "name": "Quick Normal", "tags": ["Quick", "Normal"]}]
        assert td_class._mode_description(items).startswith(
            "0 = Normal (mode tags: Normal). 2 = Heavy (mode tags: Heavy).")

    def test_only_current_mode_with_a_list_is_described(self, td_class):
        assert td_class._mode_items(td_class, "LaundryWasherMode", "SupportedModes",
                                    {"SupportedModes": SUPPORTED}) == []
        assert td_class._mode_items(td_class, "LaundryWasherMode", "CurrentMode", {}) == []


@pytest.mark.skipif(not EPISODE.is_file(), reason="SimuHome benchmark corpus not available")
class TestWasherDescription:
    @pytest.fixture(scope="class")
    def graph(self, td_class):
        config = json.loads(EPISODE.read_text())["initial_home_config"]
        return td_class("http://localhost:8097", "test", config).artifact("utility_room", WASHER)

    def _schemas(self, graph):
        """The currentMode action's input value and property's output value."""
        out = []
        for affordance in graph.subjects(TD.title, None):
            if str(graph.value(affordance, TD.title)) != "currentMode":
                continue
            for predicate in (TD.hasInputSchema, TD.hasOutputSchema):
                schema = graph.value(affordance, predicate)
                if schema is None:
                    continue
                # A direct-value input may be wrapped in an object schema.
                members = list(graph.objects(schema, JS.properties)) or [schema]
                out.extend(members)
        return out

    def test_action_and_property_carry_the_modes(self, graph):
        schemas = self._schemas(graph)
        assert len(schemas) == 2
        for schema in schemas:
            assert sorted(int(v) for v in graph.objects(schema, JS.enum)) == [0, 1, 2, 3, 4, 5]
            assert "2 = Heavy (mode tags: Heavy)" in str(graph.value(schema, JS.description))
