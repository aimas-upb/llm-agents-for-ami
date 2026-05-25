import asyncio
import os
import types
import urllib.parse

import pytest
from fastapi.testclient import TestClient

import hasp as appmod


class FakeHAWS:
    def __init__(self, areas, devices_by_area, entities, states):
        self._areas = areas
        self._devices_by_area = devices_by_area
        self._entities = entities
        self._states = states

    async def get_areas(self):
        return self._areas

    async def get_devices(self, area_id=None):
        if area_id is None:
            all_devs = []
            for devs in self._devices_by_area.values():
                all_devs.extend(devs)
            return all_devs
        return self._devices_by_area.get(area_id, [])

    async def get_entities(self):
        return self._entities

    async def close(self):
        return None


class FakeHAREST:
    def __init__(self, states, services):
        self._states = states
        self._services = services
        self.calls = []

    async def get_states(self):
        return self._states

    async def get_services(self):
        return self._services

    async def call_service(self, domain, service, data):
        self.calls.append((domain, service, data))
        return {"ok": True}

    async def close(self):
        return None


@pytest.fixture(autouse=True)
def no_startup(monkeypatch):
    monkeypatch.setenv("AREAS", "")
    monkeypatch.setattr(appmod, "AREAS", set(), raising=False)
    appmod.subscriptions.clear()
    # Replace startup hook to avoid background cache init during tests
    async def _noop():
        return None
    monkeypatch.setattr(appmod, "_startup_cache", _noop, raising=False)


@pytest.fixture()
def sample_data(monkeypatch):
    areas = [{"area_id": "lab308", "name": "Lab 308"}]
    devices_by_area = {
        "lab308": [
            {"id": "dev1", "name": "Lab308 Light System", "area_id": "lab308"},
            {"id": "dev2", "name": "temperature_sensing_308", "area_id": "lab308"},
        ],
        "lab308_kitchen": [
            {"id": "dev4", "name": "home2_kitchen_thermostat", "area_id": "lab308_kitchen"},
        ],
        None: [
            {"id": "dev3", "name": "occupancy_sensor_308", "area_id": None},
        ],
    }
    entities = [
        {"entity_id": "light.lab308_light", "device_id": "dev1", "name": "Lab308 Light System", "area_id": "lab308"},
        {"entity_id": "sensor.temperature_sensing_308", "device_id": "dev2", "name": "Lab308.Entry.Temperature", "area_id": "lab308"},
        {"entity_id": "binary_sensor.occupancy_sensor_308", "device_id": "dev3", "name": "Lab308.Entry.Motion", "area_id": "lab308"},
        {"entity_id": "climate.home2_kitchen_thermostat", "device_id": "dev4", "name": "Home2.Kitchen.Thermostat", "area_id": "lab308_kitchen"},
    ]
    states = [
        {
            "entity_id": "light.lab308_light",
            "state": "off",
            "attributes": {"friendly_name": "Lamp"},
        },
        {
            "entity_id": "sensor.temperature_sensing_308",
            "state": "21.5",
            "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
        },
        {
            "entity_id": "binary_sensor.occupancy_sensor_308",
            "state": "on",
            "attributes": {"device_class": "occupancy"},
        },
        {
            "entity_id": "climate.home2_kitchen_thermostat",
            "state": "heat",
            "attributes": {
                "current_temperature": 20.0,
                "temperature": 21.0,
                "hvac_action": "heating",
                "hvac_modes": ["off", "heat"],
                "min_temp": 16,
                "max_temp": 28,
                "target_temp_step": 0.5,
            },
        },
    ]
    services = [
        {"domain": "light", "services": {"turn_on": {"fields": {"entity_id": {}}}, "turn_off": {"fields": {"entity_id": {}}}}},
        {"domain": "sensor", "services": {}},
    ]

    fake_ws = FakeHAWS(areas, devices_by_area, entities, states)
    fake_rest = FakeHAREST(states, services)
    monkeypatch.setattr(appmod, "ha_client", fake_ws)
    monkeypatch.setattr(appmod, "ha_rest", fake_rest)
    return {
        "ws": fake_ws,
        "rest": fake_rest,
        "areas": areas,
        "devices_by_area": devices_by_area,
        "entities": entities,
        "states": states,
        "services": services,
    }


def test_list_workspaces(sample_data):
    client = TestClient(appmod.app)
    r = client.get("/workspaces")
    assert r.status_code == 200
    body = r.text
    assert "workspaces/lab308#workspace" in body


def test_workspace(sample_data):
    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab308")
    assert r.status_code == 200
    ttl = r.text
    assert "workspaces/lab308#workspace" in ttl
    assert "artifacts/" in ttl


def test_list_artifacts(sample_data):
    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab308/artifacts")
    assert r.status_code == 200
    ttl = r.text
    assert "lab308_light#artifact" in ttl
    assert "temperature_sensing_308#artifact" in ttl
    assert "occupancy_sensor_308#artifact" in ttl
    assert "home2_kitchen_thermostat#artifact" in ttl


def test_get_artifact_builds_dynamic_actions(sample_data):
    client = TestClient(appmod.app)
    # Light device artifact exposes HA services and sensor action if any
    r = client.get("/workspaces/lab308/artifacts/Lab308%20Light%20System")
    assert r.status_code == 200
    ttl = r.text
    # Has CamelCase action name for light service
    assert "LightTurnOn" in ttl or "LightTurnOff" in ttl
    # Target points to /ha/light/turn_on
    assert "/ha/light/turn_on" in ttl or "/ha/light/turn_off" in ttl


def test_action_ha_service_forwarder(sample_data):
    client = TestClient(appmod.app)
    r = client.post("/workspaces/lab308/artifacts/Lab308%20Light%20System/ha/light/turn_on", json={})
    assert r.status_code == 200
    # Verify ha_rest.call_service was invoked
    calls = appmod.ha_rest.calls
    assert ("light", "turn_on", {"entity_id": "light.lab308_light"}) in calls


def test_action_ha_service_drops_unsupported_payload_fields(monkeypatch):
    areas = [{"area_id": "lab308", "name": "Lab 308"}]
    devices_by_area = {
        "lab308": [{"id": "dev1", "name": "blinds_308_cover", "area_id": "lab308"}],
    }
    entities = [
        {"entity_id": "cover.blinds_308_cover", "device_id": "dev1", "name": "blinds_308_cover", "area_id": "lab308"},
    ]
    states = [
        {"entity_id": "cover.blinds_308_cover", "state": "closed", "attributes": {}},
    ]
    services = [
        {
            "domain": "cover",
            "services": {
                "open_cover": {"fields": {}},
            },
        },
    ]

    fake_ws = FakeHAWS(areas, devices_by_area, entities, states)
    fake_rest = FakeHAREST(states, services)
    monkeypatch.setattr(appmod, "ha_client", fake_ws)
    monkeypatch.setattr(appmod, "ha_rest", fake_rest)

    client = TestClient(appmod.app)
    r = client.post(
        "/workspaces/lab308/artifacts/blinds_308_cover/ha/cover/open_cover",
        json={"open_close": False},
    )
    assert r.status_code == 200
    assert ("cover", "open_cover", {"entity_id": "cover.blinds_308_cover"}) in appmod.ha_rest.calls


def test_cover_artifact_hides_tilt_services_for_non_tilt_entity(monkeypatch):
    areas = [{"area_id": "lab308", "name": "Lab 308"}]
    devices_by_area = {
        "lab308": [{"id": "dev1", "name": "window_308_cover", "area_id": "lab308"}],
    }
    entities = [
        {"entity_id": "cover.window_308_cover", "device_id": "dev1", "name": "window_308_cover", "area_id": "lab308"},
    ]
    states = [
        {
            "entity_id": "cover.window_308_cover",
            "state": "closed",
            "attributes": {"supported_features": 0},
        },
    ]
    services = [
        {
            "domain": "cover",
            "services": {
                "open_cover": {"fields": {"entity_id": {}}},
                "open_cover_tilt": {"fields": {"entity_id": {}}},
                "set_cover_tilt_position": {"fields": {"tilt_position": {}}},
            },
        },
    ]

    fake_ws = FakeHAWS(areas, devices_by_area, entities, states)
    fake_rest = FakeHAREST(states, services)
    monkeypatch.setattr(appmod, "ha_client", fake_ws)
    monkeypatch.setattr(appmod, "ha_rest", fake_rest)

    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab308/artifacts/window_308_cover")
    assert r.status_code == 200
    ttl = r.text
    assert "/ha/cover/open_cover" in ttl
    assert "/ha/cover/open_cover_tilt" not in ttl
    assert "/ha/cover/set_cover_tilt_position" not in ttl


def test_action_ha_service_rejects_tilt_service_for_non_tilt_cover(monkeypatch):
    areas = [{"area_id": "lab308", "name": "Lab 308"}]
    devices_by_area = {
        "lab308": [{"id": "dev1", "name": "window_308_cover", "area_id": "lab308"}],
    }
    entities = [
        {"entity_id": "cover.window_308_cover", "device_id": "dev1", "name": "window_308_cover", "area_id": "lab308"},
    ]
    states = [
        {
            "entity_id": "cover.window_308_cover",
            "state": "closed",
            "attributes": {"supported_features": 0},
        },
    ]
    services = [
        {
            "domain": "cover",
            "services": {
                "open_cover_tilt": {"fields": {}},
            },
        },
    ]

    fake_ws = FakeHAWS(areas, devices_by_area, entities, states)
    fake_rest = FakeHAREST(states, services)
    monkeypatch.setattr(appmod, "ha_client", fake_ws)
    monkeypatch.setattr(appmod, "ha_rest", fake_rest)

    client = TestClient(appmod.app)
    r = client.post("/workspaces/lab308/artifacts/window_308_cover/ha/cover/open_cover_tilt", json={})
    assert r.status_code == 404
    assert r.json()["detail"] == "Service not supported by this entity"
    assert appmod.ha_rest.calls == []


def test_climate_artifact_hides_turn_off_and_turn_on_services(monkeypatch):
    areas = [{"area_id": "lab308", "name": "Lab 308"}]
    devices_by_area = {
        "lab308": [{"id": "dev1", "name": "heater_308e", "area_id": "lab308"}],
    }
    entities = [
        {"entity_id": "climate.heater_308e", "device_id": "dev1", "name": "heater_308e", "area_id": "lab308"},
    ]
    states = [
        {
            "entity_id": "climate.heater_308e",
            "state": "heat",
            "attributes": {
                "hvac_mode": "heat",
                "hvac_modes": ["off", "heat"],
                "temperature": 20,
                "current_temperature": 21,
                "min_temp": 16,
                "max_temp": 28,
            },
        },
    ]
    services = [
        {
            "domain": "climate",
            "services": {
                "turn_off": {"fields": {"entity_id": {}}},
                "turn_on": {"fields": {"entity_id": {}}},
                "set_hvac_mode": {"fields": {"entity_id": {}, "hvac_mode": {}}},
                "set_temperature": {"fields": {"entity_id": {}, "temperature": {}}},
            },
        },
    ]

    fake_ws = FakeHAWS(areas, devices_by_area, entities, states)
    fake_rest = FakeHAREST(states, services)
    monkeypatch.setattr(appmod, "ha_client", fake_ws)
    monkeypatch.setattr(appmod, "ha_rest", fake_rest)

    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab308/artifacts/heater_308e")
    assert r.status_code == 200
    ttl = r.text
    assert "/ha/climate/turn_off" not in ttl
    assert "/ha/climate/turn_on" not in ttl
    assert "/ha/climate/set_hvac_mode" in ttl
    assert "/ha/climate/set_temperature" in ttl


def test_action_ha_service_rejects_turn_off_for_climate_entity(monkeypatch):
    areas = [{"area_id": "lab308", "name": "Lab 308"}]
    devices_by_area = {
        "lab308": [{"id": "dev1", "name": "heater_308e", "area_id": "lab308"}],
    }
    entities = [
        {"entity_id": "climate.heater_308e", "device_id": "dev1", "name": "heater_308e", "area_id": "lab308"},
    ]
    states = [
        {
            "entity_id": "climate.heater_308e",
            "state": "heat",
            "attributes": {
                "hvac_mode": "heat",
                "hvac_modes": ["off", "heat"],
                "temperature": 20,
                "current_temperature": 21,
                "min_temp": 16,
                "max_temp": 28,
            },
        },
    ]
    services = [
        {
            "domain": "climate",
            "services": {
                "turn_off": {"fields": {}},
                "set_hvac_mode": {"fields": {"hvac_mode": {}}},
            },
        },
    ]

    fake_ws = FakeHAWS(areas, devices_by_area, entities, states)
    fake_rest = FakeHAREST(states, services)
    monkeypatch.setattr(appmod, "ha_client", fake_ws)
    monkeypatch.setattr(appmod, "ha_rest", fake_rest)

    client = TestClient(appmod.app)
    r = client.post("/workspaces/lab308/artifacts/heater_308e/ha/climate/turn_off", json={})
    assert r.status_code == 404
    assert r.json()["detail"] == "Service not supported by this entity"
    assert appmod.ha_rest.calls == []


def test_climate_artifact_hides_set_humidity_when_unsupported(monkeypatch):
    areas = [{"area_id": "lab308", "name": "Lab 308"}]
    devices_by_area = {
        "lab308": [{"id": "dev1", "name": "air_conditioner_308e", "area_id": "lab308"}],
    }
    entities = [
        {"entity_id": "climate.air_conditioner_308e", "device_id": "dev1", "name": "air_conditioner_308e", "area_id": "lab308"},
    ]
    states = [
        {
            "entity_id": "climate.air_conditioner_308e",
            "state": "cool",
            "attributes": {
                "hvac_mode": "cool",
                "hvac_modes": ["off", "cool", "heat"],
                "temperature": 22,
                "current_temperature": 24,
                "min_temp": 16,
                "max_temp": 28,
            },
        },
    ]
    services = [
        {
            "domain": "climate",
            "services": {
                "set_humidity": {"fields": {"entity_id": {}, "humidity": {"selector": {"number": {"min": 30, "max": 99}}}}},
                "set_hvac_mode": {"fields": {"entity_id": {}, "hvac_mode": {}}},
            },
        },
    ]

    fake_ws = FakeHAWS(areas, devices_by_area, entities, states)
    fake_rest = FakeHAREST(states, services)
    monkeypatch.setattr(appmod, "ha_client", fake_ws)
    monkeypatch.setattr(appmod, "ha_rest", fake_rest)

    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab308/artifacts/air_conditioner_308e")
    assert r.status_code == 200
    ttl = r.text
    assert "/ha/climate/set_humidity" not in ttl
    assert "/ha/climate/set_hvac_mode" in ttl


def test_action_ha_service_rejects_set_humidity_for_unsupported_climate_entity(monkeypatch):
    areas = [{"area_id": "lab308", "name": "Lab 308"}]
    devices_by_area = {
        "lab308": [{"id": "dev1", "name": "air_conditioner_308e", "area_id": "lab308"}],
    }
    entities = [
        {"entity_id": "climate.air_conditioner_308e", "device_id": "dev1", "name": "air_conditioner_308e", "area_id": "lab308"},
    ]
    states = [
        {
            "entity_id": "climate.air_conditioner_308e",
            "state": "cool",
            "attributes": {
                "hvac_mode": "cool",
                "hvac_modes": ["off", "cool", "heat"],
                "temperature": 22,
                "current_temperature": 24,
                "min_temp": 16,
                "max_temp": 28,
            },
        },
    ]
    services = [
        {
            "domain": "climate",
            "services": {
                "set_humidity": {"fields": {"humidity": {"selector": {"number": {"min": 30, "max": 99}}}}},
            },
        },
    ]

    fake_ws = FakeHAWS(areas, devices_by_area, entities, states)
    fake_rest = FakeHAREST(states, services)
    monkeypatch.setattr(appmod, "ha_client", fake_ws)
    monkeypatch.setattr(appmod, "ha_rest", fake_rest)

    client = TestClient(appmod.app)
    r = client.post("/workspaces/lab308/artifacts/air_conditioner_308e/ha/climate/set_humidity", json={"humidity": 50})
    assert r.status_code == 404
    assert r.json()["detail"] == "Service not supported by this entity"
    assert appmod.ha_rest.calls == []


def test_action_ha_service_no_entity(sample_data):
    client = TestClient(appmod.app)
    # Use sensor device but request light domain → should 404 for missing entity
    r = client.post("/workspaces/lab308/artifacts/temperature_sensing_308/ha/light/turn_on", json={})
    assert r.status_code == 404


def test_sensor_state_is_readable_via_property_not_dynamic_action(sample_data):
    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab308/artifacts/Lab308.Entry.Temperature")
    assert r.status_code == 200
    assert "getTemperatureInDegc" not in r.text
    resp = client.get("/workspaces/lab308/artifacts/Lab308.Entry.Temperature/properties/state")
    assert resp.status_code == 200
    assert resp.json() == 21.5
    resp2 = client.post("/workspaces/lab308/artifacts/Lab308.Entry.Temperature/getTemperatureInDegc")
    assert resp2.status_code == 404


def test_binary_sensor_state_is_readable_via_property_not_dynamic_action(sample_data):
    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab308/artifacts/Lab308.Entry.Motion")
    assert r.status_code == 200
    ttl = r.text
    assert "getOccupancyState" not in ttl
    resp = client.get("/workspaces/lab308/artifacts/Lab308.Entry.Motion/properties/state")
    assert resp.status_code == 200
    assert resp.json() == "on"
    resp2 = client.post("/workspaces/lab308/artifacts/Lab308.Entry.Motion/getOccupancyState")
    assert resp2.status_code == 404


def test_climate_dynamic_action_returns_state(sample_data):
    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab308/artifacts/Home2.Kitchen.Thermostat")
    assert r.status_code == 200
    ttl = r.text
    assert "getThermostatState" in ttl
    resp = client.post("/workspaces/lab308/artifacts/Home2.Kitchen.Thermostat/getThermostatState")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "heat"
    assert data["currentTemperature"] == 20.0
    assert data["targetTemperature"] == 21.0


def test_artifact_lookup_is_canonical(sample_data):
    client = TestClient(appmod.app)
    # Request using lowercase + underscores
    resp = client.get("/workspaces/lab308/artifacts/home2_kitchen_thermostat")
    assert resp.status_code == 200
    assert "Home2.Kitchen.Thermostat" in resp.text
