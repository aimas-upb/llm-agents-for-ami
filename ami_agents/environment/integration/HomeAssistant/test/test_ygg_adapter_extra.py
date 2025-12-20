import asyncio
import json
import os
import runpy
import importlib
import runpy

import pytest
from fastapi.testclient import TestClient

import ygg_ha_adapter as appmod


@pytest.fixture(autouse=True)
def reset_allowed_areas(monkeypatch):
    monkeypatch.setattr(appmod, "AREAS", set(), raising=False)


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


def test_focus_hub_update_delete(monkeypatch):
    # Mock AsyncClient for the /hub/ endpoint intent verification
    class MockResponse:
        def __init__(self, text, status_code=200):
            self.text = text
            self.status_code = status_code
        def raise_for_status(self):
            pass

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def get(self, url, params=None):
            if params and "hub.challenge" in params:
                return MockResponse(params["hub.challenge"])
            return MockResponse("Not Found", 404)

    monkeypatch.setattr(appmod.httpx, "AsyncClient", MockAsyncClient)

    client = TestClient(appmod.app)
    assert client.post("/workspaces/ws/focus", json={}).status_code == 200
    
    # Send valid WebSub subscription request
    payload = {
        "hub.mode": "subscribe",
        "hub.topic": "http://test/topic",
        "hub.callback": "http://test/cb"
    }
    assert client.post("/hub/", data=payload).status_code == 202
    
    assert client.put("/workspaces/ws/artifacts/a", json={}).status_code == 200
    assert client.delete("/workspaces/ws/artifacts/a").status_code == 200





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
