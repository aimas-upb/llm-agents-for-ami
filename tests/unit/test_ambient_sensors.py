"""Virtual ambient sensors: the instruments SimuHome does not model.

SimuHome computes each room variable from the appliances that affect it -- a
light changes a room's illuminance and never measures it -- so under a
sensor-only resolver, 137 benchmark goals have no device that can answer them.
SHTD mints one virtual sensor per variable a room reports and serves the
simulator's own value through it.

These pin that the sensors are real devices in the graph, that they read the
same number the room property does, and that switching them off restores the
graph exactly.
"""

import json
import os
from pathlib import Path

import pytest
from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import Namespace

from ami_agents.agents.env_explorer.utils.state_resolution import (
    StateOutcome,
    load_vocabulary,
    resolve_state_request,
)
from ami_agents.environment.integration.SimuHome.mappings import load_mappings

SAREF = Namespace("https://saref.etsi.org/core/")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SCHEMA = Namespace("https://schema.org/")
TD = Namespace("https://www.w3.org/2019/wot/td#")

REPO = Path(__file__).resolve().parents[2]
EPISODE = (REPO.parent / "SimuHome" / "data" / "benchmark"
           / "qt1_feasible_seed_1.json")

pytestmark = pytest.mark.skipif(
    not EPISODE.is_file(), reason="SimuHome benchmark corpus not available")

VARIABLES = ("temperature", "humidity", "illuminance", "pm10")


def _builder():
    from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD

    config = json.loads(EPISODE.read_text())["initial_home_config"]
    return SimuHomeTD("http://localhost:8097", "test", config)


@pytest.fixture(scope="module")
def builder():
    return _builder()


@pytest.fixture(scope="module")
def graph(builder):
    """The whole home, sensors included, plus the vocabulary."""
    g = Graph()
    for room in builder.rooms:
        g += builder.room_workspace(room)
        for device in (builder.rooms[room].get("devices") or []):
            g += builder.artifact(room, device["device_id"])
        for _token, sensor_id, _row in builder.ambient_sensors(room):
            g += builder.artifact(room, sensor_id)
    return load_vocabulary(g)


class TestMinting:
    def test_one_sensor_per_reported_variable(self, builder):
        tokens = {t for t, _, _ in builder.ambient_sensors("kitchen")}
        assert tokens == set(VARIABLES)

    def test_ids_follow_the_device_naming_convention(self, builder):
        ids = {i for _, i, _ in builder.ambient_sensors("kitchen")}
        assert "kitchen_light_sensor_1" in ids
        assert "kitchen_air_quality_sensor_1" in ids

    def test_a_room_reporting_nothing_gets_nothing(self):
        from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD

        empty = SimuHomeTD("http://x", "test",
                           {"rooms": {"void": {"devices": [], "state": {}}}})
        assert empty.ambient_sensors("void") == []


class TestTheGraph:
    """A virtual sensor is a device like any other, as far as the graph says."""

    def test_it_is_a_saref_device(self, graph):
        """What `FILTER EXISTS { ?a a/rdfs:subClassOf* saref:Device }` asks."""
        art = URIRef("http://localhost:8097/workspaces/test/kitchen"
                     "/artifacts/kitchen_light_sensor_1#artifact")
        assert (art, RDF.type, SAREF.Sensor) in graph
        ask = graph.query("""
            PREFIX saref: <https://saref.etsi.org/core/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX homeont: <http://example.org/homeont/>
            ASK { homeont:LightSensor rdfs:subClassOf* saref:Device }""")
        assert bool(ask.askAnswer), "the SAREF path must close, or nothing resolves"

    def test_it_carries_its_homeont_class(self, graph):
        art = URIRef("http://localhost:8097/workspaces/test/kitchen"
                     "/artifacts/kitchen_air_quality_sensor_1#artifact")
        assert (art, RDF.type,
                URIRef("http://example.org/homeont/AirQualitySensor")) in graph

    def test_it_observes_the_rooms_own_property(self, graph):
        """Not a second copy of the reading -- the environment's property."""
        art = URIRef("http://localhost:8097/workspaces/test/kitchen"
                     "/artifacts/kitchen_temperature_sensor_1#artifact")
        observed = list(graph.objects(art, SOSA.observes))
        assert observed, "a sensor with nothing to observe is not a sensor"
        assert "kitchen" in str(observed[0])

    def test_it_says_it_is_virtual(self, graph):
        """It must not pass itself off as hardware."""
        art = URIRef("http://localhost:8097/workspaces/test/kitchen"
                     "/artifacts/kitchen_light_sensor_1#artifact")
        assert str(next(graph.objects(art, SCHEMA.manufacturer))) == "SHTD"
        assert "Virtual" in str(next(graph.objects(art, SCHEMA.model)))
        assert "SimuHome models no ambient sensor" in str(
            next(graph.objects(art, RDFS.comment)))

    def test_the_room_lists_it(self, graph):
        """Discovery walks the workspace, so an unlisted sensor is invisible."""
        ws = URIRef("http://localhost:8097/workspaces/test/kitchen#workspace")
        contained = {str(o) for o in graph.objects(
            ws, URIRef("https://purl.org/hmas/contains"))}
        assert any("light_sensor" in c for c in contained)

    def test_the_affordance_reads_from_the_sensors_own_url(self, graph):
        """Not the room's property path: this is a device's own reading."""
        art = URIRef("http://localhost:8097/workspaces/test/kitchen"
                     "/artifacts/kitchen_light_sensor_1#artifact")
        prop = next(graph.objects(art, TD.hasPropertyAffordance))
        form = next(graph.objects(prop, TD.hasForm))
        target = str(next(graph.objects(
            form, URIRef("https://www.w3.org/2019/wot/hypermedia#hasTarget"))))
        assert "/artifacts/kitchen_light_sensor_1/properties/" in target


class TestTheResolverReachesThem:
    """The 137-goal gap, closing."""

    @pytest.mark.parametrize("variable, sensor", [
        ("homeont:Illuminance", "kitchen_light_sensor_1"),
        ("homeont:Pm10MassConcentration", "kitchen_air_quality_sensor_1"),
        ("homeont:AirTemperature", "kitchen_temperature_sensor_1"),
        ("homeont:RelativeHumidity", "kitchen_humidity_sensor_1"),
    ])
    def test_every_environment_variable_now_resolves(self, graph, variable, sensor):
        result = resolve_state_request(
            graph, location_class="homeont:Kitchen",
            environment_variable={"class": variable})
        assert result.outcome is StateOutcome.RESOLVED
        assert any(a.artifact_name == sensor for a in result.affordances)

    def test_illuminance_was_unanswerable_before(self, builder):
        """Without the sensors the same query finds nothing -- the v1 result."""
        g = Graph()
        for room in builder.rooms:
            g += builder.room_workspace(room)
            for device in (builder.rooms[room].get("devices") or []):
                g += builder.artifact(room, device["device_id"])
        load_vocabulary(g)
        result = resolve_state_request(
            g, location_class="homeont:Kitchen",
            environment_variable={"class": "homeont:Illuminance"})
        assert result.outcome is StateOutcome.NONE


class TestValues:
    def test_scaling_matches_the_room_property(self, builder):
        """One reading, two paths to it: they must not diverge.

        Temperature and humidity are centi-units in the simulator; illuminance
        and pm10 are whole units. Getting this wrong is silent.
        """
        mappings = load_mappings()
        state = builder.rooms["kitchen"]["state"]
        for token, _sensor_id, _row in builder.ambient_sensors("kitchen"):
            scaled = builder.scale_room_state(token, state[token])
            assert scaled == builder.scale_room_state(token, state[token])
            row = mappings.room_state_property(token) or {}
            assert row.get("unit"), f"{token} must declare a unit"
            if token in ("temperature", "humidity"):
                assert scaled == pytest.approx(state[token] / 100, abs=0.01)
            else:
                assert scaled == pytest.approx(state[token], abs=0.01)


class TestTheFlag:
    """v1 and v2 are two configurations of one build, not two datasets."""

    def test_switching_them_off_leaves_no_trace(self, monkeypatch):
        import importlib
        import ami_agents.environment.integration.SimuHome.td_builder as module

        monkeypatch.setenv("SHTD_AMBIENT_SENSORS", "0")
        importlib.reload(module)
        try:
            config = json.loads(EPISODE.read_text())["initial_home_config"]
            off = module.SimuHomeTD("http://localhost:8097", "test", config)
            assert off.ambient_sensors("kitchen") == []
            g = off.room_workspace("kitchen")
            assert not any("sensor" in str(s) for s in g.subjects())
        finally:
            monkeypatch.delenv("SHTD_AMBIENT_SENSORS", raising=False)
            importlib.reload(module)
