"""A device's limits are properties of their own, as Matter reports them.

"What is the most the dimmer goes to?" is parsed as homeont:LevelControlMaxLevel
("the level control max level reported by the device"). The limit used to
exist only as js:maximum on the brightness it bounds, so no property had that
class and the answer was "does not provide Level Control Max Level". SHTD now
publishes each limit attribute (MaxLevel, MaxTemperature, MinMeasuredValue,
and their pairs) as a read-only property too, and keeps the bounds:

    brightness  DimmableLightBrightness   js:minimum 1, js:maximum 254
    maxLevel    LevelControlMaxLevel      (reads 254)
"""

import json
from pathlib import Path

import pytest
from rdflib.namespace import RDF

from ami_agents.shared.utils.namespaces import HOMEONT, JS, TD

REPO = Path(__file__).resolve().parents[2]
EPISODE = REPO.parent / "SimuHome" / "data" / "benchmark" / "qt1_feasible_seed_5.json"
LIGHT = "living_room_dimmable_light_1"


@pytest.fixture(scope="module")
def graph():
    """The living-room dimmer's TD, built offline from a benchmark home."""
    from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD
    if not EPISODE.exists():
        pytest.skip(f"SimuHome benchmark not found at {EPISODE}")
    config = json.loads(EPISODE.read_text())["initial_home_config"]
    return SimuHomeTD("http://localhost:8097", "test", config).artifact("living_room", LIGHT)


def _property(graph, title):
    """The property affordance titled ``title``, or None."""
    return next((p for p in graph.subjects(RDF.type, TD.PropertyAffordance)
                 if str(graph.value(p, TD.title)) == title), None)


def test_the_limit_is_a_property_typed_with_its_class(graph):
    """A question about the max level resolves to this property."""
    limit = _property(graph, "maxLevel")
    assert limit is not None and (limit, RDF.type, HOMEONT.LevelControlMaxLevel) in graph


def test_the_bounded_property_keeps_its_bounds(graph):
    """Publishing the limit takes nothing from the brightness schema."""
    brightness = _property(graph, "brightness")
    schema = graph.value(brightness, TD.hasOutputSchema) or brightness
    assert (int(graph.value(schema, JS.minimum)), int(graph.value(schema, JS.maximum))) == (1, 254)
