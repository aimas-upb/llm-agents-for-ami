"""lab308e's room environment as HASP serves it: SimuHome's shapes, derived
from a recorded Home Assistant deployment (`fixtures/lab308e_ha.json`) plus
the home's `semantic:` block (`lab308e.yaml`).

What is pinned, as SimuHome's td_builder emits it:

    <ws#place> homeont:hasEnvironment <ws#environment>
    <ws#environment> ssn:hasProperty <ws#illuminance>, <ws#desk_illuminance>, ...
    every variable: named, typed, a TD-SOSA observable property affordance,
        quantity kind and unit (where homeont has one), readable when observed
    every artifact: homeont:isLocatedIn <ws#place>
    sensors: sosa:Sensor, saref:Sensor, sosa:observes their variable
    the external light sensor: the outdoor environment's illuminance, not the room's

and that the ISA's ambiguous-goal context -- which starts from the room's
environment -- now finds lab308e's variables and the actions moving them.
"""

import sys
from pathlib import Path

import pytest
from rdflib import Graph, Literal, Namespace, RDF, URIRef

from test_hasp_resolution import AREA, SNAPSHOT, _hasp_on_fixture

REPO = Path(__file__).resolve().parents[5]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ami_agents.agents.env_explorer.utils.actuation_context import (  # noqa: E402
    build_actuation_context,
)
from ami_agents.agents.interaction_solver.utils.specific_capability_context import (  # noqa: E402
    build_specific_context,
)
from ami_agents.shared.models.goal_structure import GoalSpec  # noqa: E402
from ami_agents.shared.utils.vocabulary import close_types  # noqa: E402

SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
TD = Namespace("https://www.w3.org/2019/wot/td#")
QUDT = Namespace("http://qudt.org/schema/qudt/")
SAREF = Namespace("https://saref.etsi.org/core/")
HOMEONT = Namespace("http://example.org/homeont/")
TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")

# Served documents carry the requesting host (`_ttl_for_request`).
WS = f"http://testserver/workspaces/{AREA}"
ENVIRONMENT = URIRef(f"{WS}#environment")
ROOM_VARIABLES = ["air_temperature", "carbon_dioxide_concentration", "desk_illuminance",
                  "glare", "illuminance", "occupancy", "occupant_count", "relative_humidity"]


@pytest.fixture(scope="module")
def client():
    with _hasp_on_fixture() as test_client:
        yield test_client


@pytest.fixture(scope="module")
def graph(client):
    """The workspace TD and every artifact TD of lab308e, as served."""
    g = Graph()
    for path in [f"/workspaces/{AREA}"] + [
            f"/workspaces/{AREA}/artifacts/{d['name']}" for d in SNAPSHOT["devices"]]:
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)
        g.parse(data=response.text, format="turtle")
    return g


def _artifact(name: str) -> URIRef:
    return URIRef(f"{WS}/artifacts/{name}#artifact")


def _var(name: str) -> URIRef:
    return URIRef(f"{WS}#{name}")


# -- the room and its variables ---------------------------------------------------

class TestRoom:
    def test_the_place_has_the_environment(self, graph):
        place = URIRef(f"{WS}#place")
        assert (place, HOMEONT.hasEnvironment, ENVIRONMENT) in graph
        assert (place, RDF.type, HOMEONT.StudyRoom) in graph
        assert (ENVIRONMENT, HOMEONT.isEnvironmentOf, place) in graph

    def test_the_environment_has_its_variables(self, graph):
        found = sorted(str(v).rsplit("#", 1)[-1] for v in graph.objects(ENVIRONMENT, SSN.hasProperty))
        assert found == ROOM_VARIABLES

    @pytest.mark.parametrize("name", ROOM_VARIABLES)
    def test_each_variable_is_described_in_full(self, graph, name):
        variable = _var(name)
        # 1. A TD-SOSA observable property affordance, named after its class.
        assert (variable, RDF.type, TDSOSA.ObservablePropertyAffordance) in graph
        assert (variable, RDF.type, TD.PropertyAffordance) in graph
        assert (variable, TD.name, Literal(name)) in graph
        assert (variable, TD.title, Literal(name)) in graph
        assert (variable, SSN.isPropertyOf, ENVIRONMENT) in graph
        # 2. What it measures: every lab308e variable but occupancy (a yes/no) has both.
        if name != "occupancy":
            assert graph.value(variable, QUDT.hasQuantityKind) is not None, name
            assert graph.value(variable, QUDT.unit) is not None, name
        # 3. Readable: every lab308e variable has a sensor.
        assert (variable, TD.isObservable, Literal(True)) in graph
        assert (variable, TD.hasForm, None) in graph

    def test_comfort_ranges(self, graph):
        assert (_var("desk_illuminance"), TDSOSA.targetMinValue, Literal(300)) in graph
        assert (_var("glare"), TDSOSA.targetMaxValue, Literal(50)) in graph


# -- the devices -------------------------------------------------------------------

class TestDevices:
    def test_every_artifact_is_located_in_the_room(self, graph):
        for device in SNAPSHOT["devices"]:
            assert (_artifact(device["name"]), HOMEONT.isLocatedIn,
                    URIRef(f"{WS}#place")) in graph, device["name"]

    @pytest.mark.parametrize("sensor, variable", [
        ("internal_light_sensing_308e", "illuminance"),
        ("desk_light_sensing_308e", "desk_illuminance"),
        ("temperature_sensing_308e", "air_temperature"),
        ("humidity_sensing_308e", "relative_humidity"),
        ("co2_sensing_308e", "carbon_dioxide_concentration"),
        ("glare_sensing_308e", "glare"),
        ("presence_sensing_308e", "occupancy"),
        ("person_counter_308e", "occupant_count"),
    ])
    def test_sensors_observe_their_variable(self, graph, sensor, variable):
        art = _artifact(sensor)
        assert (art, RDF.type, SOSA.Sensor) in graph
        assert (art, RDF.type, SAREF.Sensor) in graph
        assert (art, SOSA.observes, _var(variable)) in graph

    def test_thermostats_observe_the_air(self, graph):
        for name in ("air_conditioner_308e", "heater_308e"):
            assert (_artifact(name), SOSA.observes, _var("air_temperature")) in graph
            assert (_artifact(name), RDF.type, SAREF.Sensor) not in graph

    def test_the_outdoor_light_is_not_the_rooms(self, graph):
        outdoor = _var("outdoor_illuminance")
        assert (_artifact("external_light_sensing_308e"), SOSA.observes, outdoor) in graph
        assert (outdoor, SSN.isPropertyOf, URIRef(f"{WS}#outdoor_environment")) in graph
        assert (ENVIRONMENT, SSN.hasProperty, outdoor) not in graph
        # No place has the outdoor environment.
        assert (None, HOMEONT.hasEnvironment, URIRef(f"{WS}#outdoor_environment")) not in graph

    def test_no_example_vocabulary_but_homeont(self, graph):
        for term in {t for triple in graph for t in triple if isinstance(t, URIRef)}:
            assert not str(term).startswith("http://example.org/") or \
                str(term).startswith(str(HOMEONT)), term
        assert (None, TDSOSA.affordsProperty, None) not in graph


# -- reading a variable ------------------------------------------------------------

def test_a_room_variable_reads_its_sensor(client):
    assert client.get(f"/workspaces/{AREA}/properties/illuminance").json() == 250
    assert client.get(f"/workspaces/{AREA}/properties/desk_illuminance").json() == 310
    assert client.get(f"/workspaces/{AREA}/properties/carbon_dioxide_concentration").json() == 650
    assert client.get(f"/workspaces/{AREA}/properties/no_such_variable").status_code == 404


# -- the agents' view ----------------------------------------------------------------

def test_an_ambiguous_goal_finds_what_lights_the_room(graph):
    """"The study room is too dark": the ISA's context starts from the room's
    environment, so before this model it found nothing on lab308e."""
    # 1. The goal, as the UA structures it.
    goal = GoalSpec.from_dict({
        "goal_specificity": "ambiguous", "goal_kind": "achievement",
        "intent_text": "the study room is too dark",
        "implied_environment_vars": [{
            "property_name": "illuminance", "property_class": "homeont:Illuminance",
            "property_sensed_space": {"space_class": "homeont:StudyRoom", "space_name": AREA}}]})
    # 2. The context over HASP's graph, closed under the ontology as the agents do.
    closed = Graph()
    closed += graph
    close_types(closed)
    context = build_specific_context(closed, goal)
    assert context is not None
    assert {"ambient_lights_308e", "task_lights_308e", "blinds_308e"} <= set(context.artifacts)
    # The ISA matches the room's variables by class: desk_illuminance is an
    # Illuminance of the same environment, so the desk lamp counts as lighting
    # the room too -- it does light part of it.
    assert "desk_lamp_308e" in context.artifacts
    # The window moves CO2, not light.
    assert "window_308e" not in context.artifacts


# -- regressions (icordos, 3a964c9 / 3813728, ported to the room-variable IRIs) ---------

def _closed(graph: Graph) -> Graph:
    """The served graph closed under the ontology, as the agents query it."""
    closed = Graph()
    closed += graph
    close_types(closed)
    return closed


def _ambiguous(variable_class: str, direction: str) -> GoalSpec:
    """An ambiguous goal on one lab308e room variable."""
    return GoalSpec.from_dict({
        "goal_kind": "achievement", "goal_specificity": "ambiguous",
        "implied_environment_vars": [{
            "property_name": variable_class.split(":")[1],
            "property_class": variable_class,
            "property_sensed_space": {"space_class": "homeont:StudyRoom", "space_name": AREA},
            "property_change_direction": direction}]})


def test_each_variable_has_exactly_one_environment(graph):
    """Every variable an action moves or a device observes belongs to one
    environment, which alone has it; no per-artifact environment remains; the
    lights and the light sensor name the very same room variable."""
    # 1. One owner, both directions.
    variables = (set(graph.objects(None, SOSA.actsOnProperty))
                 | set(graph.objects(None, SOSA.observes)))
    assert variables
    for variable in variables:
        owners = set(graph.objects(variable, SSN.isPropertyOf))
        assert len(owners) == 1, variable
        assert set(graph.subjects(SSN.hasProperty, variable)) == owners, variable
    assert not any("/artifacts/" in str(node) and str(node).endswith("#foi")
                   for node in graph.all_nodes())

    # 2. Shared references to the room's illuminance.
    illuminance = _var("illuminance")
    for name in ("ambient_lights_308e", "task_lights_308e"):
        actions = graph.objects(_artifact(name), TD.hasActionAffordance)
        assert any((actuation, SOSA.actsOnProperty, illuminance) in graph
                   for action in actions
                   for actuation in graph.objects(action, TDSOSA.hasEffectActuation))
    assert (_artifact("internal_light_sensing_308e"), SOSA.observes, illuminance) in graph


def test_the_window_lowers_co2(graph):
    """"The air is stuffy": the structurer sees a measurable CO2 variable, and
    the ISA's context finds the window lowering it and the CO2 sensor."""
    # 1. What the UA's structurer is shown of the room.
    closed = _closed(graph)
    view = build_actuation_context(closed, with_environment=True)
    variables = [var for ws in view["workspaces"]
                 for var in (ws.get("environment") or {}).get("variables", [])]
    assert any(var["class"] == "homeont:CarbonDioxideConcentration" for var in variables)

    # 2. The ISA's context for lowering it.
    context = build_specific_context(closed, _ambiguous(
        "homeont:CarbonDioxideConcentration", "decrease"))
    assert context is not None
    assert any(a["artifact_name"] == "window_308e" and a["name"] == "CoverOpenCover"
               and a["type"] == "action_affordance" for a in context.affordances)
    assert any(a["artifact_name"] == "co2_sensing_308e" and a["type"] == "property_affordance"
               for a in context.affordances)
    effects = [action for result in context.observable_property_hints["results"]
               for action in result["actions"]]
    assert any(action["artifact_title"] == "window_308e"
               and action["action_name"] == "CoverOpenCover"
               and action["direction"] == "decrease" for action in effects)


@pytest.mark.parametrize("variable, direction, actuator, sensor", [
    ("homeont:Illuminance", "increase", "ambient_lights_308e", "internal_light_sensing_308e"),
    ("homeont:Illuminance", "decrease", "blackout_blinds_308e", "internal_light_sensing_308e"),
    ("homeont:AirTemperature", "decrease", "air_conditioner_308e", "temperature_sensing_308e"),
])
def test_ambiguous_goals_find_room_actuators(graph, variable, direction, actuator, sensor):
    """The planner's room-variable join, not just device discovery."""
    context = build_specific_context(_closed(graph), _ambiguous(variable, direction))
    assert context is not None
    assert any(a["artifact_name"] == actuator and a["type"] == "action_affordance"
               for a in context.affordances)
    assert any(a["artifact_name"] == sensor and a["type"] == "property_affordance"
               for a in context.affordances)


def test_the_blinds_say_which_way_their_position_runs(graph):
    """lab308e's covers state Home Assistant's direction (percent open) in
    their own schemas, and nothing in the home claims Matter's (percent closed)."""
    JS_DESCRIPTION = URIRef("https://www.w3.org/2019/wot/json-schema#description")
    # 1. Every value-schema description in the served home.
    descriptions = [str(d) for d in graph.objects(None, JS_DESCRIPTION)]
    # 2. Some say percent open (the set_cover_position input, current_position);
    #    none says percent closed.
    assert any("Percent open: 0 is fully closed, 100 is fully open." in d
               for d in descriptions)
    assert not any("Percent closed" in d for d in descriptions)

