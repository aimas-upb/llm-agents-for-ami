"""ENV_STATE and ENV_CAPABILITY_QUERY resolution over HASP's own TDs.

The same resolvers the agents run (`state_resolution`, `capability_resolution`)
over the TDs HASP serves for lab308e, built offline from a recorded Home
Assistant snapshot (`fixtures/lab308e_ha.json`) -- no HA, no network.

These pin semantic parity with SimuHome's TD builder: a question that resolves
on a SimuHome home must resolve on an HA one too.
"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rdflib import Graph

import hasp as appmod

REPO = Path(__file__).resolve().parents[5]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ami_agents.agents.env_explorer.utils.capability_resolution import (  # noqa: E402
    CapabilityOutcome,
    resolve_capability_request,
)
from ami_agents.agents.env_explorer.utils.state_resolution import (  # noqa: E402
    StateOutcome,
    load_vocabulary,
    resolve_state_request,
)

FIXTURE = Path(__file__).parent / "fixtures" / "lab308e_ha.json"
HOME_CONFIG = Path(__file__).resolve().parents[1] / "lab308e.yaml"
SNAPSHOT = json.loads(FIXTURE.read_text())
AREA = "lab308e"
ROOM = "homeont:StudyRoom"


class FixtureHAWS:
    async def get_areas(self):
        return SNAPSHOT["areas"]

    async def get_devices(self, area_id=None):
        devices = SNAPSHOT["devices"]
        return devices if area_id is None else [
            d for d in devices if d.get("area_id") == area_id]

    async def get_entities(self):
        return SNAPSHOT["entities"]

    async def close(self):
        return None


class FixtureHAREST:
    async def get_states(self):
        return SNAPSHOT["states"]

    async def get_services(self):
        return SNAPSHOT["services"]

    async def close(self):
        return None


@contextmanager
def _hasp_on_fixture():
    """HASP's app, talking to the recorded snapshot instead of Home Assistant."""
    patch = pytest.MonkeyPatch()
    patch.setattr(appmod, "AREAS", set())
    patch.setattr(appmod, "BASE_WS_URI", "http://localhost:8080")
    patch.setattr(appmod, "TD_SOSA_ENV_VAR_OVERRIDES", {})
    patch.setattr(appmod, "TD_SOSA_PROPERTY_RANGES", {})
    patch.setattr(appmod, "TD_SOSA_SETTLING_TIMES", {})
    patch.setattr(appmod, "ha_client", FixtureHAWS())
    patch.setattr(appmod, "ha_rest", FixtureHAREST())
    patch.setenv("SEMANTIC_CONFIG", str(HOME_CONFIG))
    appmod.reload_semantics()
    try:
        yield TestClient(appmod.app)
    finally:
        patch.undo()
        appmod.reload_semantics()


@pytest.fixture(scope="module")
def client():
    with _hasp_on_fixture() as test_client:
        yield test_client


@pytest.fixture(scope="module")
def graph(client):
    """Every TD HASP serves for lab308e, plus the vocabulary."""
    g = Graph()
    response = client.get(f"/workspaces/{AREA}")
    assert response.status_code == 200, response.text
    g.parse(data=response.text, format="turtle")
    for device in SNAPSHOT["devices"]:
        response = client.get(f"/workspaces/{AREA}/artifacts/{device['name']}")
        assert response.status_code == 200, (device["name"], response.text)
        g.parse(data=response.text, format="turtle")
    return load_vocabulary(g)


def _names(result):
    entries = getattr(result, "entries", None)
    if entries is None:
        entries = result.affordances
    return sorted({e.artifact_name for e in entries})


class TestState:
    def test_room_air_temperature(self, graph):
        result = resolve_state_request(
            graph, location_class=ROOM,
            environment_variable={"class": "homeont:AirTemperature"})
        assert result.outcome is StateOutcome.RESOLVED
        assert "temperature_sensing_308e" in _names(result)

    def test_room_humidity(self, graph):
        result = resolve_state_request(
            graph, location_class=ROOM,
            environment_variable={"class": "homeont:RelativeHumidity"})
        assert _names(result) == ["humidity_sensing_308e"]

    def test_room_illuminance_from_every_light_sensor(self, graph):
        result = resolve_state_request(
            graph, location_class=ROOM,
            environment_variable={"class": "homeont:Illuminance"})
        assert _names(result) == ["desk_light_sensing_308e",
                                  "external_light_sensing_308e",
                                  "internal_light_sensing_308e"]

    @pytest.mark.parametrize("variable, sensor", [
        ("homeont:CarbonDioxideConcentration", "co2_sensing_308e"),
        ("homeont:Occupancy", "presence_sensing_308e"),
        ("homeont:OccupantCount", "person_counter_308e"),
        ("homeont:Glare", "glare_sensing_308e"),
    ])
    def test_lab308e_room_variables(self, graph, variable, sensor):
        result = resolve_state_request(
            graph, location_class=ROOM, environment_variable={"class": variable})
        assert _names(result) == [sensor]

    def test_a_light_is_on_or_off(self, graph):
        result = resolve_state_request(
            graph, device_class="homeont:DimmableLight",
            device_property={"class": "homeont:OnOff"})
        assert result.outcome is StateOutcome.RESOLVED
        assert "ambient_lights_308e" in _names(result)

    def test_who_makes_the_air_conditioner(self, graph):
        result = resolve_state_request(
            graph, device_class="homeont:AirConditioner",
            device_property={"class": "schema:manufacturer"})
        assert result.outcome is StateOutcome.RESOLVED_ARTIFACT
        assert result.affordances[0].manufacturer == "twrecked"


class TestCapabilities:
    def test_lights_can_be_dimmed_even_when_off(self, graph):
        """An off light reports no brightness attribute in HA, but it can
        still be dimmed: the capability comes from its supported modes."""
        result = resolve_capability_request(
            graph, location_class=ROOM,
            device_property={"class": "homeont:LevelControlBrightness"})
        assert result.outcome is CapabilityOutcome.FOUND
        assert _names(result) == ["ambient_lights_308e", "desk_lamp_308e",
                                  "task_lights_308e"]
        assert {e.property_branch for e in result.entries} == {"actuatable"}

    def test_air_conditioner_fan_modes(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:AirConditioner",
            device_property={"class": "homeont:FanControlFanMode"})
        assert result.outcome is CapabilityOutcome.FOUND
        values = result.entries[0].permitted_values
        assert values["enum"] == ["auto", "high", "low", "medium"]

    def test_setpoint_bounds(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:AirConditioner",
            device_property={"class": "homeont:TemperatureSetpoint"})
        assert result.entries[0].permitted_values["minimum"] == 18
        assert result.entries[0].permitted_values["maximum"] == 30

    def test_what_can_change_the_air_temperature(self, graph):
        result = resolve_capability_request(
            graph, location_class=ROOM,
            environment_variable={"class": "homeont:AirTemperature"},
            command={"class": "saref:Command"})
        assert result.outcome is CapabilityOutcome.FOUND
        assert {"air_conditioner_308e", "heater_308e"} <= set(_names(result))

    def test_what_can_be_switched_on(self, graph):
        """Climate devices are not among them: HASP exposes no climate
        turn_on (see `is_service_supported_for_entity`); they are switched on
        through their HVAC mode instead."""
        result = resolve_capability_request(
            graph, location_class=ROOM, command={"class": "saref:OnCommand"})
        assert result.outcome is CapabilityOutcome.FOUND
        assert {"ambient_lights_308e", "ceiling_fan_308e",
                "projector_308e"} <= set(_names(result))

    def test_hvac_mode_is_a_mode_command(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:AirConditioner",
            command={"class": "homeont:SetModeCommand"})
        assert result.outcome is CapabilityOutcome.FOUND

    def test_blinds_can_be_opened(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:WindowCoveringController",
            command={"class": "saref:OpenCommand"})
        assert _names(result) == ["blackout_blinds_308e", "blinds_308e",
                                  "window_308e"]

    def test_projector_volume_is_changeable(self, graph):
        result = resolve_capability_request(
            graph, device_class="homeont:MediaPlayer",
            device_property={"class": "homeont:Volume"})
        assert result.outcome is CapabilityOutcome.FOUND
        assert {e.property_branch for e in result.entries} == {"actuatable"}

    def test_the_room_inventory_lists_typed_affordances_only(self, graph):
        result = resolve_capability_request(graph, location_class=ROOM)
        assert result.outcome is CapabilityOutcome.FOUND
        assert all(e.affordance_type.startswith(("homeont:", "saref:"))
                   for e in result.entries)
        assert all(e.workspace_name == "lab308e" for e in result.entries)


class TestPropertyNames:
    def test_no_artifact_advertises_a_bare_state_property(self, graph):
        """Every entity's state has its own name (decided: always rename)."""
        rows = graph.query("""
            PREFIX td: <https://www.w3.org/2019/wot/td#>
            SELECT ?name WHERE { ?a td:hasPropertyAffordance ?p . ?p td:name ?name }
        """)
        names = {str(r.name) for r in rows}
        assert "state" not in names
        assert {"lightState", "coverState", "climateState"} <= names

    def test_each_state_name_reads_its_entity(self, client):
        url = f"/workspaces/{AREA}/artifacts/temperature_sensing_308e/properties/sensorState"
        assert client.get(url).json() == 22

    def test_bare_state_is_still_served_as_an_alias(self, client):
        url = f"/workspaces/{AREA}/artifacts/temperature_sensing_308e/properties/state"
        assert client.get(url).json() == 22


class TestArtifactNames:
    """Three names, three facts: td:title the instance, rdfs:label the kind of
    device, schema:model the product."""

    def _labels(self, graph, title):
        rows = graph.query("""
            PREFIX td: <https://www.w3.org/2019/wot/td#>
            PREFIX hmas: <https://purl.org/hmas/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label WHERE { ?a a hmas:Artifact ; td:title ?t ; rdfs:label ?label .
                                  FILTER(STR(?t) = "%s") }""" % title)
        return sorted(str(r.label) for r in rows)

    def test_the_label_is_the_kind_of_device(self, graph):
        assert self._labels(graph, "ambient_lights_308e") == ["Color Light"]
        assert self._labels(graph, "air_conditioner_308e") == ["Air Conditioner"]

    def test_an_untyped_device_has_no_label(self, graph):
        assert self._labels(graph, "clock_308e") == []


def test_no_action_carries_the_legacy_status_command(graph):
    """Actions are typed with their homeont / SAREF command, nothing else."""
    rows = graph.query("""
        PREFIX td: <https://www.w3.org/2019/wot/td#>
        ASK { ?a td:hasActionAffordance ?x . ?x a <http://example.org/StatusCommand> }""")
    assert not rows.askAnswer


def test_the_thermostat_read_is_a_get_command(graph):
    rows = graph.query("""
        PREFIX td: <https://www.w3.org/2019/wot/td#>
        PREFIX saref: <https://saref.etsi.org/core/>
        SELECT ?name WHERE { ?a td:hasActionAffordance ?x . ?x td:name ?name ; a saref:GetCommand }""")
    assert {str(r.name) for r in rows} == {"getThermostatState"}
