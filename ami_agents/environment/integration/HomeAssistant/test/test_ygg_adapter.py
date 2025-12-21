import asyncio
import os
import types
import urllib.parse
import json
import runpy
import importlib

import pytest
from fastapi.testclient import TestClient

import ygg_ha_adapter as appmod


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
def no_forwarder(monkeypatch):
    # Prevent background tasks during tests
    monkeypatch.setenv("AREAS", "")
    monkeypatch.setattr(appmod, "AREAS", set(), raising=False)
    # Replace startup hook to avoid scheduling tasks
    async def _noop():
        return None
    monkeypatch.setattr(appmod, "_startup_forwarder", _noop, raising=False)


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
            "entity_id": "binary_sensor.occupancy_sensor_308", "state": "on",
            "attributes": {"device_class": "occupancy"},
        },
        {
            "entity_id": "climate.home2_kitchen_thermostat", "state": "heat",
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
    assert "Lab308%20Light%20System#artifact" in ttl
    assert "Lab308.Entry.Temperature#artifact" in ttl
    assert "Lab308.Entry.Motion#artifact" in ttl
    assert "Home2.Kitchen.Thermostat#artifact" in ttl


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


def test_action_ha_service_no_entity(sample_data):
    client = TestClient(appmod.app)
    # Use sensor device but request light domain → should 404 for missing entity
    r = client.post("/workspaces/lab308/artifacts/temperature_sensing_308/ha/light/turn_on", json={})
    assert r.status_code == 404


def test_sensor_dynamic_action_returns_state(sample_data):
    client = TestClient(appmod.app)
    # For the temperature sensor artifact
    r = client.get("/workspaces/lab308/artifacts/Lab308.Entry.Temperature")
    assert r.status_code == 200
    # Dynamic sensor action
    action = "getTemperatureInDegc"
    resp = client.post(f"/workspaces/lab308/artifacts/Lab308.Entry.Temperature/{action}")
    assert resp.status_code == 200
    assert resp.text == "21.5"
    # Also accept form without device class
    resp2 = client.post("/workspaces/lab308/artifacts/Lab308.Entry.Temperature/getInDegc")
    assert resp2.status_code == 200
    assert resp2.text == "21.5"


def test_binary_sensor_action_support(sample_data):
    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab308/artifacts/Lab308.Entry.Motion")
    assert r.status_code == 200
    ttl = r.text
    assert "getOccupancyState" in ttl
    resp = client.post("/workspaces/lab308/artifacts/Lab308.Entry.Motion/getOccupancyState")
    assert resp.status_code == 200
    assert resp.text == "on"
    resp2 = client.post("/workspaces/lab308/artifacts/Lab308.Entry.Motion/getBinarySensorState")
    assert resp2.status_code == 200
    assert resp2.text == "on"


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


def test_forwarder_status_endpoint(sample_data, monkeypatch):
    # Simulate forwarder scheduled
    class DummyTask:
        def done(self):
            return False
    appmod.app.state.forward_task = DummyTask()
    client = TestClient(appmod.app)
    r = client.get("/_forwarder/status")
    assert r.status_code == 200
    data = r.json()
    assert "taskRunning" in data and data["taskRunning"] is True


@pytest.mark.asyncio
async def test_websub_subscribe_success(monkeypatch):
    # Clear existing subs
    appmod.subscriptions.clear()
    
    callback_url = "http://subscriber/cb"
    topic_url = "http://test/topic"
    
    # Mock the AsyncClient to handle the intent verification GET
    class MockResponse:
        def __init__(self, text, status_code=200):
            self.text = text
            self.status_code = status_code
        def raise_for_status(self):
            if self.status_code >= 400:
                # raise httpx.HTTPStatusError
                import httpx
                raise httpx.HTTPStatusError("Http Error", request=None, response=self)

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def get(self, url, params=None):
            # Check if this is the intent verification
            if url == callback_url and params and "hub.challenge" in params:
                # In a real scenario, the subscriber echoes the challenge
                return MockResponse(params["hub.challenge"])
            return MockResponse("Not Found", 404)

    monkeypatch.setattr(appmod.httpx, "AsyncClient", MockAsyncClient)

    client = TestClient(appmod.app)
    # Using data=... sends application/x-www-form-urlencoded, which is standard for WebSub
    payload = {
        "hub.mode": "subscribe",
        "hub.topic": topic_url,
        "hub.callback": callback_url,
        "hub.lease_seconds": "3600"
    }
    response = client.post("/hub/", data=payload)
    assert response.status_code == 202
    
    # Verify internal state
    subs = list(appmod.subscriptions.values())
    assert len(subs) == 1
    assert subs[0]["topic"] == topic_url
    assert subs[0]["callback"] == callback_url


@pytest.mark.asyncio
async def test_websub_intent_verification_failure(monkeypatch):
    appmod.subscriptions.clear()
    callback_url = "http://subscriber/cb_fail"
    topic_url = "http://test/topic"

    class MockResponse:
        def __init__(self, text, status_code):
            self.text = text
            self.status_code = status_code
        def raise_for_status(self):
            if self.status_code >= 400:
                import httpx
                raise httpx.HTTPStatusError("Http Error", request=None, response=self)

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def get(self, url, params=None):
             # Simulate subscriber returning 404
             return MockResponse("Not Found", 404)

    monkeypatch.setattr(appmod.httpx, "AsyncClient", MockAsyncClient)

    client = TestClient(appmod.app)
    payload = {
        "hub.mode": "subscribe",
        "hub.topic": topic_url,
        "hub.callback": callback_url
    }
    # My implementation raises 412 for HTTP errors/Request errors during verification
    response = client.post("/hub/", data=payload)
    assert response.status_code in (412, 409)
    assert len(appmod.subscriptions) == 0 # Should not have added subscription


@pytest.mark.asyncio
async def test_websub_unsubscribe_success(monkeypatch):
    appmod.subscriptions.clear()
    callback_url = "http://subscriber/cb"
    topic_url = "http://test/topic"
    
    # Pre-populate a subscription
    sub_id = f"{topic_url}-{callback_url}"
    appmod.subscriptions[sub_id] = {
        "topic": topic_url,
        "callback": callback_url,
        "lease_seconds": 3600
    }

    class MockResponse:
        def __init__(self, text, status_code=200):
            self.text = text
            self.status_code = status_code
        def raise_for_status(self):
            if self.status_code >= 400:
                import httpx
                raise httpx.HTTPStatusError("Http Error", request=None, response=self)

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def get(self, url, params=None):
            if url == callback_url and params and "hub.challenge" in params:
                return MockResponse(params["hub.challenge"])
            return MockResponse("Not Found", 404)

    monkeypatch.setattr(appmod.httpx, "AsyncClient", MockAsyncClient)

    client = TestClient(appmod.app)
    payload = {
        "hub.mode": "unsubscribe",
        "hub.topic": topic_url,
        "hub.callback": callback_url
    }
    response = client.post("/hub/", data=payload)
    assert response.status_code == 202
    
    # Verify removal
    assert sub_id not in appmod.subscriptions

def test_helpers_sanitize_and_sensor_name():
    assert appmod._sanitize_unit("°C") == "degc"
    assert appmod._sanitize_unit("%") == "percent"
    assert appmod._sanitize_unit("kWh") == "kwh"
    assert appmod._sanitize_unit("m/s") == "ms"

    assert appmod._sensor_action_name("temperature", "°C") == "getTemperatureInDegc"
    assert appmod._sensor_action_name(None, "°C") == "getInDegc"
    assert appmod._sensor_action_name("temperature", None) is None
    assert appmod._binary_sensor_action_names("occupancy")[0] == "getOccupancyState"
    assert appmod._binary_sensor_action_names(None)[0] == "getBinarySensorState"


def test_normalize_workspace_id(monkeypatch):
    original = appmod.AREAS
    monkeypatch.setattr(appmod, "AREAS", {"home2", "home3"})
    assert appmod._normalize_workspace_id("home2") == "home2"
    assert appmod._normalize_workspace_id("home2_kitchen") == "home2"
    assert appmod._normalize_workspace_id("home3_entry") == "home3"
    assert appmod._workspace_allowed("home2_kitchen") is True
    assert appmod._workspace_allowed("home4") is False
    monkeypatch.setattr(appmod, "AREAS", original)


def test_infer_value_and_type():
    v, t = appmod._infer_value_and_type("on")
    assert v is True and t.endswith("#boolean")
    v, t = appmod._infer_value_and_type("10")
    assert v == 10 and t.endswith("#integer")
    v, t = appmod._infer_value_and_type("10.5")
    assert v == 10.5 and t.endswith("#double")
    v, t = appmod._infer_value_and_type("abc")
    assert v == "abc" and t.endswith("#string")


def test_workspace_not_found(monkeypatch):
    # Minimal stubs
    class WS:
        async def get_areas(self):
            return []
    appmod.ha_client = WS()
    client = TestClient(appmod.app)
    r = client.get("/workspaces/unknown")
    assert r.status_code == 404


def test_artifact_not_found(monkeypatch):
    class WS:
        async def get_devices(self, area=None):
            return []
        async def get_entities(self):
            return []
    appmod.ha_client = WS()
    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab/artifacts/Nope")
    assert r.status_code == 404


def test_service_forwarder_not_found(monkeypatch):
    class WS:
        async def get_devices(self, area=None):
            devs = [{"id": "d", "name": "dev"}]
            if area is None:
                return devs
            return devs if area == "lab" else []
        async def get_entities(self):
            return [{"entity_id": "sensor.dev", "device_id": "d"}]
    class REST:
        async def get_services(self):
            return []
    appmod.ha_client = WS()
    appmod.ha_rest = REST()
    client = TestClient(appmod.app)
    r = client.post("/workspaces/lab/artifacts/dev/ha/light/turn_on", json={})
    assert r.status_code == 404


def test_dynamic_sensor_action_invalid_and_mismatch(monkeypatch):
    class WS:
        async def get_devices(self, area=None):
            devs = [{"id": "d", "name": "temp"}]
            if area is None:
                return devs
            return devs if area == "lab" else []
        async def get_entities(self):
            return [{"entity_id": "sensor.temp", "device_id": "d"}]
    class REST:
        async def get_states(self):
            return [{
                "entity_id": "sensor.temp",
                "state": "21",
                "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"}
            }]
    appmod.ha_client = WS()
    appmod.ha_rest = REST()
    client = TestClient(appmod.app)
    # invalid format
    r = client.post("/workspaces/lab/artifacts/temp/notAnAction")
    assert r.status_code == 404
    # mismatch: wrong unit
    r2 = client.post("/workspaces/lab/artifacts/temp/getTemperatureInDegf")
    assert r2.status_code == 404


def test_focus_other_endpoints(monkeypatch):
    client = TestClient(appmod.app)
    # The actual focus logic is tested in specific focus API tests.
    # This just checks other endpoints that were part of the old test.
    assert client.put("/workspaces/ws/artifacts/a", json={}).status_code == 200
    assert client.delete("/workspaces/ws/artifacts/a").status_code == 200

@pytest.mark.asyncio
async def test_focus_workspace_registers_subscription(monkeypatch):
    appmod.subscriptions.clear()
    client = TestClient(appmod.app)
    workspace_id = "lab308"
    callback_url = "http://agent/callback/ws"
    
    payload = {
        "callbackUrl": callback_url
    }
    response = client.post(f"/workspaces/{workspace_id}/focus", json=payload)
    assert response.status_code == 200
    assert "Focus succeeded" in response.text
    
    # Verify subscription was registered
    expected_topic = f"{appmod.BASE_WS_URI.rstrip('/')}/workspaces/{workspace_id}"
    assert f"{expected_topic}-{callback_url}" in appmod.subscriptions
    sub = appmod.subscriptions[f"{expected_topic}-{callback_url}"]
    assert sub["topic"] == expected_topic
    assert sub["callback"] == callback_url
    assert sub["type"] == "focus"

@pytest.mark.asyncio
async def test_focus_artifact_registers_subscription(monkeypatch, sample_data):
    appmod.subscriptions.clear()
    client = TestClient(appmod.app)
    workspace_id = "lab308"
    artifact_name = "Lab308 Light System" # Assuming this artifact exists from sample_data
    callback_url = "http://agent/callback/art"
    
    payload = {
        "artifactName": artifact_name,
        "callbackUrl": callback_url
    }
    response = client.post(f"/workspaces/{workspace_id}/focus", json=payload)
    assert response.status_code == 200
    assert "Focus succeeded" in response.text
    
    # Verify subscription was registered
    safe_artifact_name = urllib.parse.quote(artifact_name, safe="")
    expected_topic = f"{appmod.BASE_WS_URI.rstrip('/')}/workspaces/{workspace_id}/artifacts/{safe_artifact_name}#artifact"
    assert f"{expected_topic}-{callback_url}" in appmod.subscriptions
    sub = appmod.subscriptions[f"{expected_topic}-{callback_url}"]
    assert sub["topic"] == expected_topic
    assert sub["callback"] == callback_url
    assert sub["type"] == "focus"

@pytest.mark.asyncio
async def test_focus_missing_callback_url_succeeds(monkeypatch):
    appmod.subscriptions.clear()
    client = TestClient(appmod.app)
    workspace_id = "lab308"
    
    payload = {
        "artifactName": "some_artifact"
    }
    response = client.post(f"/workspaces/{workspace_id}/focus", json=payload)
    assert response.status_code == 200
    assert "Focus succeeded" in response.text


def test_import_requires_token(monkeypatch):
    # Import the module via run_path with HA_TOKEN unset should raise
    monkeypatch.delenv("HA_TOKEN", raising=False)
    
    # Also patch uvicorn to prevent server start if the check somehow fails
    import types, sys
    dummy_uvicorn = types.SimpleNamespace(run=lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, 'uvicorn', dummy_uvicorn)

    code = None
    try:
        # Correct path for ygg_ha_adapter.py
        adapter_path = os.path.join(os.path.dirname(__file__), "..", "ygg_ha_adapter.py")
        runpy.run_path(adapter_path, run_name="__main__")
    except RuntimeError as e:
        code = str(e)
    assert code == "HA_TOKEN env var required"


def test_list_workspaces_error(monkeypatch):
    class WS:
        async def get_areas(self):
            raise RuntimeError("boom")
    import ygg_ha_adapter as appmod2
    appmod2.ha_client = WS()
    from fastapi.testclient import TestClient
    client = TestClient(appmod2.app)
    r = client.get("/workspaces")
    assert r.status_code == 500


def test_get_artifact_error(monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("fail")
    monkeypatch.setattr(appmod, "_resolve_device_and_entities", boom)
    from fastapi.testclient import TestClient
    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab/artifacts/any")
    assert r.status_code == 500


@pytest.mark.asyncio
async def test_ws_handshake_and_build_map(monkeypatch):
    # Success handshake
    class WS:
        def __init__(self, msgs):
            self._msgs = msgs
            self.sent = []
        async def recv(self):
            return self._msgs.pop(0)
        async def send(self, s):
            self.sent.append(json.loads(s))
        async def close(self):
            return None
    async def fake_connect(url):
        msgs = [json.dumps({"type": "auth_required"}), json.dumps({"type": "auth_ok"})]
        return WS(msgs)
    monkeypatch.setattr(appmod.websockets, "connect", fake_connect)
    ws = await appmod._ws_handshake("ws://x", "tok")
    assert isinstance(ws, WS)

    # Build map
    async def fake_connect2(url):
        # After handshake, simulate result messages for devices and entities
        msgs = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            json.dumps({"type": "result", "success": True, "id": 1, "result": [{"id": "d1", "area_id": "a1"}]}),
            json.dumps({"type": "result", "success": True, "id": 2, "result": [{"entity_id": "sensor.x", "device_id": "d1"}]}),
        ]
        return WS(msgs)
    monkeypatch.setattr(appmod.websockets, "connect", fake_connect2)
    ent_to_area, ent_to_device, dev_by_id, ent_by_id = await appmod._build_entity_area_map()
    assert ent_to_area == {"sensor.x": "a1"}
    assert ent_to_device == {"sensor.x": "d1"}
    assert dev_by_id["d1"]["area_id"] == "a1"
    assert ent_by_id["sensor.x"]["device_id"] == "d1"

    # Error handshake
    async def fake_bad(url):
        msgs = [json.dumps({"type": "not_auth"})]
        return WS(msgs)
    monkeypatch.setattr(appmod.websockets, "connect", fake_bad)
    with pytest.raises(RuntimeError):
        await appmod._ws_handshake("ws://x", "tok")


def test_main_block(monkeypatch):
    # Ensure the __main__ path is exercised without starting a real server
    import types, sys
    dummy_uvicorn = types.SimpleNamespace(run=lambda *a, **kw: None)
    sys.modules['uvicorn'] = dummy_uvicorn
    monkeypatch.setenv("HA_TOKEN", "x")
    monkeypatch.setenv("HA_URL", "ws://localhost:8123/api/websocket")
    runpy.run_module("ygg_ha_adapter", run_name="__main__")