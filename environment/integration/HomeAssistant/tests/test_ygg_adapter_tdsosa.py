import urllib.parse

import pytest
from fastapi.testclient import TestClient
from rdflib import Graph, Namespace, RDF, URIRef

import ygg_ha_adapter as appmod


SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")


class FakeHAWS:
    async def get_areas(self):
        return [{"area_id": "lab308", "name": "Lab 308"}]

    async def get_devices(self, area_id=None):
        devices = [
            {"id": "dev_light", "name": "Main Light", "area_id": "lab308"},
            {"id": "dev_temp", "name": "Temp Sensor", "area_id": "lab308"},
        ]
        if area_id is None:
            return devices
        return [d for d in devices if d.get("area_id") == area_id]

    async def get_entities(self):
        return [
            {"entity_id": "light.main_light", "device_id": "dev_light", "name": "Main Light", "area_id": "lab308"},
            {"entity_id": "sensor.temp_sensor", "device_id": "dev_temp", "name": "Temp Sensor", "area_id": "lab308"},
        ]

    async def close(self):
        return None


class FakeHAREST:
    async def get_states(self):
        return [
            {"entity_id": "light.main_light", "state": "off", "attributes": {"friendly_name": "Main Light"}},
            {
                "entity_id": "sensor.temp_sensor",
                "state": "21.5",
                "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
            },
        ]

    async def get_services(self):
        return [
            {"domain": "light", "services": {"turn_on": {"fields": {"entity_id": {}}}, "turn_off": {"fields": {"entity_id": {}}}}},
            {"domain": "sensor", "services": {}},
        ]

    async def close(self):
        return None


@pytest.fixture(autouse=True)
def setup_tdsosa_test_env(monkeypatch):
    monkeypatch.setattr(appmod, "MONITOR_URL", "")
    monkeypatch.setattr(appmod, "EXPLORER_URL", "")
    monkeypatch.setattr(appmod, "AREAS", set())
    monkeypatch.setattr(appmod, "BASE_WS_URI", "http://localhost:8080")
    monkeypatch.setattr(appmod, "TD_SOSA_ENV_VAR_OVERRIDES", {})
    monkeypatch.setattr(appmod, "ha_client", FakeHAWS())
    monkeypatch.setattr(appmod, "ha_rest", FakeHAREST())


def _ttl_graph(ttl: str) -> Graph:
    g = Graph()
    g.parse(data=ttl, format="turtle")
    return g


def test_tdsosa_action_effect_added_for_light_turn_on():
    client = TestClient(appmod.app)
    artifact = urllib.parse.quote("Main Light", safe="")
    r = client.get(f"/workspaces/lab308/artifacts/{artifact}")
    assert r.status_code == 200
    g = _ttl_graph(r.text)

    env_uri = URIRef("http://testserver/workspaces/lab308/environment/luminosity")
    actuation_uri = URIRef(
        f"http://testserver/workspaces/lab308/artifacts/{artifact}/actuations/light_turn_on_increase"
    )

    assert (None, TDSOSA.hasEffectActuation, actuation_uri) in g
    assert (actuation_uri, RDF.type, TDSOSA.IncreasingActuation) in g
    assert (actuation_uri, SOSA.actsOnProperty, env_uri) in g
    assert (actuation_uri, TDSOSA.increasesObservableProperty, env_uri) in g
    assert (env_uri, RDF.type, SOSA.ObservableProperty) in g
    assert (env_uri, RDF.type, SOSA.ActuatableProperty) in g


def test_tdsosa_observable_property_links_added_for_sensor_state():
    client = TestClient(appmod.app)
    artifact = urllib.parse.quote("Temp Sensor", safe="")
    r = client.get(f"/workspaces/lab308/artifacts/{artifact}")
    assert r.status_code == 200
    g = _ttl_graph(r.text)

    env_uri = URIRef("http://testserver/workspaces/lab308/environment/thermal_comfort")
    foi_uri = URIRef(f"http://testserver/workspaces/lab308/artifacts/{artifact}/environment#foi")

    assert (None, TDSOSA.affordsProperty, env_uri) in g
    assert (None, RDF.type, TDSOSA.ObservablePropertyAffordance) in g
    assert (env_uri, RDF.type, SOSA.ObservableProperty) in g
    assert (foi_uri, SSN.hasProperty, env_uri) in g


def test_tdsosa_override_resolution_prefers_specific_key(monkeypatch):
    monkeypatch.setattr(
        appmod,
        "TD_SOSA_ENV_VAR_OVERRIDES",
        {
            "action:light.turn_on": "generic_action",
            "entity:light.main_light:action:light.turn_on": "entity_specific_action",
        },
    )

    resolved = appmod._resolve_env_var_override(
        entity={"entity_id": "light.main_light"},
        device={"id": "dev_light", "name": "Main Light"},
        artifact_label="Main Light",
        domain="light",
        service_name="turn_on",
    )
    assert resolved == "entity_specific_action"


def test_tdsosa_action_effect_uses_override_env_var(monkeypatch):
    monkeypatch.setattr(
        appmod,
        "TD_SOSA_ENV_VAR_OVERRIDES",
        {"action:light.turn_on": "desk_light_level"},
    )

    client = TestClient(appmod.app)
    artifact = urllib.parse.quote("Main Light", safe="")
    r = client.get(f"/workspaces/lab308/artifacts/{artifact}")
    assert r.status_code == 200
    g = _ttl_graph(r.text)

    env_uri = URIRef("http://testserver/workspaces/lab308/environment/desk_light_level")
    assert (None, TDSOSA.hasEffectActuation, None) in g
    assert (None, SOSA.actsOnProperty, env_uri) in g


def test_temperature_sensor_exposes_temperature_action():
    client = TestClient(appmod.app)
    artifact = urllib.parse.quote("Temp Sensor", safe="")
    r = client.get(f"/workspaces/lab308/artifacts/{artifact}")
    assert r.status_code == 200
    assert "getTemperatureInDegc" in r.text


def test_temperature_sensor_state_property_returns_number():
    client = TestClient(appmod.app)
    artifact = urllib.parse.quote("Temp Sensor", safe="")
    r = client.get(f"/workspaces/lab308/artifacts/{artifact}/properties/state")
    assert r.status_code == 200
    assert r.json() == 21.5


def test_temperature_sensor_state_override_applies_to_tdsosa(monkeypatch):
    monkeypatch.setattr(
        appmod,
        "TD_SOSA_ENV_VAR_OVERRIDES",
        {"entity:sensor.temp_sensor:state": "room_temperature"},
    )

    client = TestClient(appmod.app)
    artifact = urllib.parse.quote("Temp Sensor", safe="")
    r = client.get(f"/workspaces/lab308/artifacts/{artifact}")
    assert r.status_code == 200
    g = _ttl_graph(r.text)

    env_uri = URIRef("http://testserver/workspaces/lab308/environment/room_temperature")
    assert (None, TDSOSA.affordsProperty, env_uri) in g
    assert (env_uri, RDF.type, SOSA.ObservableProperty) in g
