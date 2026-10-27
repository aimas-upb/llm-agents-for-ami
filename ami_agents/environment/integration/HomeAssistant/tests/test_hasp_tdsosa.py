"""The room environment in HASP's TDs: variables, observations and effects.

A small fake home (one light, one cover, one temperature sensor in area
`lab308`) served by HASP's app, checked for the shapes SimuHome's td_builder
emits and the agents query:

    artifact   homeont:isLocatedIn <ws#place>; a sensor `sosa:observes` <ws#var>;
               an action's actuation `sosa:actsOnProperty` <ws#var>
    workspace  <ws#place> homeont:hasEnvironment <ws#environment>
               <ws#environment> ssn:hasProperty <ws#var>, each var named,
               typed, with quantity kind / unit, and a form when observed
    routes     GET /workspaces/<ws>/properties/<var>; the two graph queries,
               which also accept the older variable keys (`luminosity`)
"""

import pytest
from fastapi.testclient import TestClient
from rdflib import Graph, Literal, Namespace, RDF, URIRef

import hasp as appmod
from ami_agents.shared.utils.namespaces import (  # noqa: E402
    HOMEONT,
    QUDT,
    SAREF,
    SOSA,
    SSN,
    TD,
    TDSOSA,
)



WS = "http://testserver/workspaces/lab308"
BASE = "http://localhost:8080/workspaces/lab308"


class FakeHAWS:
    async def get_areas(self):
        return [{"area_id": "lab308", "name": "Lab 308"}]

    async def get_devices(self, area_id=None):
        devices = [
            {"id": "dev_light", "name": "Main Light", "area_id": "lab308"},
            {"id": "dev_cover", "name": "Blinds", "area_id": "lab308"},
            {"id": "dev_temp", "name": "Temp Sensor", "area_id": "lab308"},
        ]
        if area_id is None:
            return devices
        return [d for d in devices if d.get("area_id") == area_id]

    async def get_entities(self):
        return [
            {"entity_id": "light.main_light", "device_id": "dev_light", "name": "Main Light", "area_id": "lab308"},
            {"entity_id": "cover.blinds", "device_id": "dev_cover", "name": "Blinds", "area_id": "lab308"},
            {"entity_id": "sensor.temp_sensor", "device_id": "dev_temp", "name": "Temp Sensor", "area_id": "lab308"},
        ]

    async def close(self):
        return None


class FakeHAREST:
    async def get_states(self):
        return [
            {"entity_id": "light.main_light", "state": "off", "attributes": {"friendly_name": "Main Light"}},
            {
                "entity_id": "cover.blinds",
                "state": "open",
                "attributes": {"friendly_name": "Blinds", "current_position": 100},
            },
            {
                "entity_id": "sensor.temp_sensor",
                "state": "21.5",
                "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
            },
        ]

    async def get_services(self):
        return [
            {"domain": "light", "services": {"turn_on": {"fields": {"entity_id": {}}}, "turn_off": {"fields": {"entity_id": {}}}}},
            {
                "domain": "cover",
                "services": {
                    "open_cover": {"target": {"entity": [{"domain": ["cover"]}]}, "fields": {}},
                    "close_cover": {"target": {"entity": [{"domain": ["cover"]}]}, "fields": {}},
                    "set_cover_position": {
                        "target": {"entity": [{"domain": ["cover"]}]},
                        "fields": {"position": {"selector": {"number": {"min": 0, "max": 100, "step": 1}}}},
                    },
                },
            },
            {"domain": "sensor", "services": {}},
        ]

    async def close(self):
        return None


@pytest.fixture(autouse=True)
def setup_tdsosa_test_env(monkeypatch):
    """HASP on the fake home, with the default mappings only (no home config)."""
    monkeypatch.setattr(appmod, "AREAS", set())
    monkeypatch.setattr(appmod, "BASE_WS_URI", "http://localhost:8080")
    monkeypatch.setattr(appmod, "ha_client", FakeHAWS())
    monkeypatch.setattr(appmod, "ha_rest", FakeHAREST())
    monkeypatch.setenv("SEMANTIC_CONFIG", "")
    appmod.reload_semantics()
    appmod.subscriptions.clear()
    yield
    monkeypatch.undo()
    appmod.reload_semantics()


@pytest.fixture
def home_config(tmp_path, monkeypatch):
    """A home config for the fake home: a comfort range, and the blinds moving
    glare instead of the room's light."""
    path = tmp_path / "home.yaml"
    path.write_text(
        "semantic:\n"
        "  devices:\n"
        "    Blinds:\n"
        "      effects:\n"
        "        - {class: 'homeont:Glare', services: [open_cover, close_cover]}\n"
        "  ranges:\n"
        "    illuminance: {min: 200, unit: lx}\n")
    monkeypatch.setenv("SEMANTIC_CONFIG", str(path))
    appmod.reload_semantics()


def _get(client, path: str) -> Graph:
    response = client.get(path)
    assert response.status_code == 200, response.text
    g = Graph()
    g.parse(data=response.text, format="turtle")
    return g


# -- artifacts ---------------------------------------------------------------

def test_a_light_action_moves_the_rooms_illuminance():
    g = _get(TestClient(appmod.app), "/workspaces/lab308/artifacts/main_light")
    art = URIRef(f"{WS}/artifacts/main_light#artifact")
    variable = URIRef(f"{WS}#illuminance")
    actuation = URIRef(f"{WS}/artifacts/main_light/actuations/light_turn_on_illuminance")

    # 1. The effect: an increasing actuation on the room's variable.
    assert (None, TDSOSA.hasEffectActuation, actuation) in g
    assert (actuation, RDF.type, TDSOSA.IncreasingActuation) in g
    assert (actuation, SOSA.actsOnProperty, variable) in g
    assert (actuation, TDSOSA.increasesObservableProperty, variable) in g
    # 2. The device: an actuator in the room, acting on its environment.
    assert (art, RDF.type, SAREF.Actuator) in g
    assert (art, HOMEONT.isLocatedIn, URIRef(f"{WS}#place")) in g
    assert (art, SOSA.hasFeatureOfInterest, URIRef(f"{WS}#environment")) in g
    assert (URIRef(f"{WS}#place"), HOMEONT.hasEnvironment, URIRef(f"{WS}#environment")) in g


def test_a_sensor_observes_the_rooms_air_temperature():
    g = _get(TestClient(appmod.app), "/workspaces/lab308/artifacts/temp_sensor")
    art = URIRef(f"{WS}/artifacts/temp_sensor#artifact")
    variable = URIRef(f"{WS}#air_temperature")

    # 1. The sensor and what it observes.
    assert (art, RDF.type, SOSA.Sensor) in g
    assert (art, RDF.type, SAREF.Sensor) in g
    assert (art, SOSA.observes, variable) in g
    assert (variable, SOSA.isObservedBy, art) in g
    # 2. Its reading: an observation affordance of the room's environment.
    reading = next(p for p in g.objects(art, TD.hasPropertyAffordance)
                   if (p, RDF.type, HOMEONT.AirTemperature) in g)
    assert (reading, RDF.type, TDSOSA.ObservablePropertyAffordance) in g
    assert (reading, SSN.isPropertyOf, URIRef(f"{WS}#environment")) in g
    assert str(g.value(reading, QUDT.unit)).endswith("unit/DEG_C")


def test_no_older_shapes_remain():
    client = TestClient(appmod.app)
    for name in ("main_light", "blinds", "temp_sensor"):
        g = _get(client, f"/workspaces/lab308/artifacts/{name}")
        assert (None, TDSOSA.affordsProperty, None) not in g
        assert (None, TDSOSA.settlingTime, None) not in g
        # Nothing in the example namespace but homeont.
        for term in {t for triple in g for t in triple if isinstance(t, URIRef)}:
            assert not str(term).startswith("http://example.org/") or \
                str(term).startswith(str(HOMEONT)), term


# -- the workspace -------------------------------------------------------------

def test_the_workspace_lists_its_variables():
    g = _get(TestClient(appmod.app), "/workspaces/lab308")
    environment = URIRef(f"{WS}#environment")
    temperature, illuminance = URIRef(f"{WS}#air_temperature"), URIRef(f"{WS}#illuminance")

    # 1. Both variables belong to the room's environment, in full.
    assert (environment, SSN.hasProperty, temperature) in g
    assert (environment, SSN.hasProperty, illuminance) in g
    for variable, name in ((temperature, "air_temperature"), (illuminance, "illuminance")):
        assert (variable, RDF.type, TDSOSA.ObservablePropertyAffordance) in g
        assert (variable, TD.name, Literal(name)) in g
        assert g.value(variable, QUDT.hasQuantityKind) is not None
    # 2. Only the observed one can be read.
    assert (temperature, TD.isObservable, Literal(True)) in g
    assert (temperature, TD.hasForm, None) in g
    assert (illuminance, TD.isObservable, Literal(False)) in g
    assert (illuminance, TD.hasForm, None) not in g


def test_a_room_variable_reads_its_sensor():
    client = TestClient(appmod.app)
    assert client.get("/workspaces/lab308/properties/air_temperature").json() == 21.5
    # No device observes the light level here.
    assert client.get("/workspaces/lab308/properties/illuminance").status_code == 404


def test_a_homes_range_and_effects(home_config):
    client = TestClient(appmod.app)
    g = _get(client, "/workspaces/lab308")
    illuminance = URIRef(f"{WS}#illuminance")
    assert (illuminance, TDSOSA.targetMinValue, Literal(200)) in g
    assert (illuminance, TDSOSA.targetUnit, Literal("lx")) in g
    # The blinds move glare, not the room's light, in this home.
    blinds = _get(client, "/workspaces/lab308/artifacts/blinds")
    assert (None, SOSA.actsOnProperty, URIRef(f"{WS}#glare")) in blinds
    assert (None, SOSA.actsOnProperty, illuminance) not in blinds


# -- the temperature sensor's own reading -------------------------------------------

def test_temperature_sensor_does_not_expose_synthetic_status_action():
    r = TestClient(appmod.app).get("/workspaces/lab308/artifacts/temp_sensor")
    assert r.status_code == 200
    assert "getTemperatureInDegc" not in r.text


def test_temperature_sensor_state_property_returns_number():
    r = TestClient(appmod.app).get("/workspaces/lab308/artifacts/temp_sensor/properties/state")
    assert r.status_code == 200
    assert r.json() == 21.5


# -- graph queries --------------------------------------------------------------

@pytest.mark.parametrize("asked", ["illuminance", "luminosity"])
def test_query_actions_affecting_observable_property(asked):
    r = TestClient(appmod.app).post(
        "/_graph/query/actions-affecting-observable-property",
        json={"workspace_id": "lab308", "observable_property": asked})
    assert r.status_code == 200
    payload = r.json()

    # 1. A name and an older key resolve to the same variable.
    assert payload["workspace_id"] == "lab308"
    assert payload["property_uri"] == f"{BASE}#illuminance"
    # 2. The light's and the cover's actions move it, both ways.
    action_names = {item["action_name"] for item in payload["actions"]}
    directions = {item["direction"] for item in payload["actions"]}
    assert {"LightTurnOn", "LightTurnOff", "CoverOpenCover", "CoverCloseCover"} <= action_names
    assert {"increase", "decrease"} <= directions
    assert all("settling_time_seconds" not in item for item in payload["actions"])


def test_query_actions_includes_target_range(home_config):
    r = TestClient(appmod.app).post(
        "/_graph/query/actions-affecting-observable-property",
        json={"workspace_id": "lab308", "observable_property": "illuminance"})
    payload = r.json()
    assert payload["target_min"] == 200.0
    assert payload["target_max"] is None
    assert payload["target_unit"] == "lx"


def test_query_action_effects_for_action_target():
    action_url = f"{BASE}/artifacts/blinds/ha/cover/close_cover"
    r = TestClient(appmod.app).post(
        "/_graph/query/action-effects",
        json={"workspace_id": "lab308", "action_url": action_url})
    assert r.status_code == 200
    effects = r.json()["effects"]
    assert len(effects) == 1
    assert effects[0]["action_target"] == action_url
    assert effects[0]["property_uri"] == f"{BASE}#illuminance"
    assert effects[0]["property_name"] == "illuminance"
    assert effects[0]["direction"] == "decrease"
