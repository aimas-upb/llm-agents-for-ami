import os
import runpy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import hasp as appmod


@pytest.fixture(autouse=True)
def reset_allowed_areas(monkeypatch):
    monkeypatch.setattr(appmod, "AREAS", set(), raising=False)
    appmod.subscriptions.clear()


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


def test_save_graph_snapshot(tmp_path, monkeypatch):
    class WS:
        async def get_areas(self):
            return [{"area_id": "lab", "name": "Lab"}]

        async def get_devices(self):
            return [{"id": "d1", "name": "Temp Sensor", "area_id": "lab"}]

        async def get_entities(self):
            return [{"entity_id": "sensor.temp", "device_id": "d1", "name": "Temp Sensor", "area_id": "lab"}]

    class REST:
        async def get_states(self):
            return [{
                "entity_id": "sensor.temp",
                "state": "21.5",
                "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
            }]

        async def get_services(self):
            return [{"domain": "sensor", "services": {}}]

    monkeypatch.setattr(appmod, "ha_client", WS())
    monkeypatch.setattr(appmod, "ha_rest", REST())
    monkeypatch.setattr(appmod, "GRAPH_SNAPSHOT_DIR", str(tmp_path))
    client = TestClient(appmod.app)

    r = client.post("/_graph/snapshot", json={"filename": "snapshot.ttl"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["filename"] == "snapshot.ttl"

    snapshot_path = Path(payload["path"])
    assert snapshot_path == tmp_path / "snapshot.ttl"
    assert snapshot_path.exists()
    assert "Temp Sensor" in snapshot_path.read_text(encoding="utf-8")


def test_save_graph_snapshot_rdfxml(tmp_path, monkeypatch):
    class WS:
        async def get_areas(self):
            return [{"area_id": "lab", "name": "Lab"}]

        async def get_devices(self):
            return [{"id": "d1", "name": "Temp Sensor", "area_id": "lab"}]

        async def get_entities(self):
            return [{"entity_id": "sensor.temp", "device_id": "d1", "name": "Temp Sensor", "area_id": "lab"}]

    class REST:
        async def get_states(self):
            return [{
                "entity_id": "sensor.temp",
                "state": "21.5",
                "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
            }]

        async def get_services(self):
            return [{"domain": "sensor", "services": {}}]

    monkeypatch.setattr(appmod, "ha_client", WS())
    monkeypatch.setattr(appmod, "ha_rest", REST())
    monkeypatch.setattr(appmod, "GRAPH_SNAPSHOT_DIR", str(tmp_path))
    client = TestClient(appmod.app)

    r = client.post("/_graph/snapshot", json={"filename": "snapshot.rdf", "format": "xml"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["filename"] == "snapshot.rdf"
    assert payload["format"] == "xml"

    snapshot_path = Path(payload["path"])
    assert snapshot_path == tmp_path / "snapshot.rdf"
    assert snapshot_path.exists()
    text = snapshot_path.read_text(encoding="utf-8")
    assert text.lstrip().startswith("<?xml")
    assert "rdf:RDF" in text


def test_focus_hub_update_delete(monkeypatch):
    class DummyClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, url, params=None):
            class R:
                status_code = 200
                text = params["hub.challenge"]
                def raise_for_status(self):
                    return None
            return R()

    monkeypatch.setattr(appmod.httpx, "AsyncClient", DummyClient)
    client = TestClient(appmod.app)
    cache = type("Cache", (), {
        "has_workspace": staticmethod(lambda ws: __import__("asyncio").sleep(0, result=True)),
        "get_artifact_contexts": staticmethod(lambda ws, art: __import__("asyncio").sleep(0, result=[{
            "artifact_topic": "http://localhost:8080/workspaces/ws/artifacts/a#artifact"
        }])),
    })()
    monkeypatch.setattr(appmod.app.state, "graph_cache", cache, raising=False)
    async def fake_ensure():
        return cache
    monkeypatch.setattr(appmod, "_ensure_graph_cache", fake_ensure)
    assert client.post("/workspaces/ws/focus", json={"callbackUrl": "http://callback.test/focus"}).status_code == 202
    assert client.post(
        "/hub/",
        data={
            "hub.mode": "subscribe",
            "hub.topic": "http://localhost:8080/workspaces/ws/artifacts/a#artifact",
            "hub.callback": "http://callback.test/websub",
        },
    ).status_code == 202
    assert client.put("/workspaces/ws/artifacts/a", json={}).status_code == 200
    assert client.delete("/workspaces/ws/artifacts/a").status_code == 200


def test_artifact_slug_prefers_entity_object_id(monkeypatch):
    class WS:
        async def get_areas(self):
            return [{"area_id": "lab", "name": "Lab"}]

        async def get_devices(self):
            return [{"id": "d1", "name": "Blinds Device", "area_id": "lab"}]

        async def get_entities(self):
            return [{"entity_id": "cover.blinds_308", "device_id": "d1", "name": "blinds_308 cover", "area_id": "lab"}]

    class REST:
        async def get_states(self):
            return [{
                "entity_id": "cover.blinds_308",
                "state": "open",
                "attributes": {"current_position": 100},
            }]

        async def get_services(self):
            return [{"domain": "cover", "services": {}}]

    monkeypatch.setattr(appmod, "ha_client", WS())
    monkeypatch.setattr(appmod, "ha_rest", REST())
    client = TestClient(appmod.app)

    listing = client.get("/workspaces/lab/artifacts")
    assert listing.status_code == 200
    assert "artifacts/blinds_308#artifact" in listing.text
    assert "artifacts/blinds_308%20cover#artifact" not in listing.text

    artifact = client.get("/workspaces/lab/artifacts/blinds_308")
    assert artifact.status_code == 200
    assert 'td:title "blinds_308 cover"' in artifact.text

    workspace = client.get("/workspaces/lab")
    assert workspace.status_code == 200
    assert "artifacts/blinds_308#artifact" in workspace.text
    assert "artifacts/blinds_308_cover#artifact" not in workspace.text


def test_platform_profile_uri_rewrites_to_request_base(monkeypatch):
    class WS:
        async def get_areas(self):
            return [{"area_id": "lab", "name": "Lab"}]

        async def get_devices(self):
            return []

        async def get_entities(self):
            return []

    class REST:
        async def get_states(self):
            return []

        async def get_services(self):
            return []

    monkeypatch.setattr(appmod, "ha_client", WS())
    monkeypatch.setattr(appmod, "ha_rest", REST())
    cache = appmod.HASPGraphCache(
        ws_client=appmod.ha_client,
        rest_client=appmod.ha_rest,
        base_uri="http://localhost:8008/",
        allowed_workspaces=set(),
        artifact_builder=appmod._build_cached_artifact_ttl,
    )
    monkeypatch.setattr(appmod.app.state, "graph_cache", cache, raising=False)

    async def fake_ensure():
        await cache.refresh()
        return cache

    monkeypatch.setattr(appmod, "_ensure_graph_cache", fake_ensure)
    client = TestClient(appmod.app)
    response = client.get("http://localhost:8080/")

    assert response.status_code == 200
    assert "<http://localhost:8080> a hmas:ResourceProfile" in response.text
    assert "localhost:8008" not in response.text


@pytest.mark.asyncio
async def test_process_state_changed_event_updates_cache_and_notifies(monkeypatch):
    events = []

    class Cache:
        def __init__(self):
            self.applied = []
        async def apply_state_change(self, entity_id, new_state):
            self.applied.append((entity_id, new_state))
        async def get_entity_contexts(self, entity_id):
            return [{
                "workspace_id": "lab308",
                "artifact_uri": "http://localhost:8080/workspaces/lab308/artifacts/Temp%20Sensor#artifact",
                "artifact_title": "Temp Sensor",
                "workspace_topic": "http://localhost:8080/workspaces/lab308",
                "artifact_topic": "http://localhost:8080/workspaces/lab308/artifacts/Temp%20Sensor#artifact",
            }]

    class DummyClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, url, json=None, headers=None):
            events.append((url, json))
            class R:
                status_code = 200
                text = "ok"
                def raise_for_status(self):
                    return None
            return R()

    cache = Cache()
    monkeypatch.setattr(appmod.app.state, "graph_cache", cache, raising=False)
    monkeypatch.setattr(appmod.httpx, "AsyncClient", DummyClient)
    appmod.subscriptions["artifact-sub"] = {
        "topic": "http://localhost:8080/workspaces/lab308/artifacts/Temp%20Sensor#artifact",
        "callback": "http://callback.test/notify",
        "lease_seconds": None,
        "timestamp": 0.0,
        "type": "websub",
    }

    await appmod._process_state_changed_event({
        "event_type": "state_changed",
        "time_fired": "2025-01-01T00:00:00Z",
        "data": {
            "entity_id": "sensor.temp_sensor",
            "new_state": {
                "entity_id": "sensor.temp_sensor",
                "state": "21.5",
                "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
            },
        },
    })

    assert cache.applied and cache.applied[0][0] == "sensor.temp_sensor"
    assert events and events[0][0] == "http://callback.test/notify"
    assert events[0][1]["artifactTitle"] == "Temp Sensor"


def test_import_requires_token(monkeypatch):
    # Import the module via run_path with HA_TOKEN unset should raise
    monkeypatch.delenv("HA_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HA_TOKEN env var required"):
        runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "hasp.py"), run_name="__main__")


def test_list_workspaces_error(monkeypatch):
    class WS:
        async def get_areas(self):
            raise RuntimeError("boom")
    import hasp as appmod2
    appmod2.ha_client = WS()
    from fastapi.testclient import TestClient
    client = TestClient(appmod2.app)
    r = client.get("/workspaces")
    assert r.status_code == 500


@pytest.mark.asyncio
async def test_shutdown_closes_clients(monkeypatch):
    called = {"ws": False, "rest": False}

    class Closable:
        def __init__(self, key):
            self.key = key

        async def close(self):
            called[self.key] = True
            return None

    monkeypatch.setattr(appmod, "ha_client", Closable("ws"))
    monkeypatch.setattr(appmod, "ha_rest", Closable("rest"))
    appmod.app.state.sync_task = None
    await appmod._shutdown()
    assert called == {"ws": True, "rest": True}


def test_get_artifact_error(monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("fail")
    monkeypatch.setattr(appmod, "_resolve_device_and_entities", boom)
    from fastapi.testclient import TestClient
    client = TestClient(appmod.app)
    r = client.get("/workspaces/lab/artifacts/any")
    assert r.status_code == 500


def test_main_block(monkeypatch):
    # Ensure the __main__ path is exercised without starting a real server
    import types, sys
    dummy_uvicorn = types.SimpleNamespace(run=lambda *a, **kw: None)
    sys.modules['uvicorn'] = dummy_uvicorn
    monkeypatch.setenv("HA_TOKEN", "x")
    monkeypatch.setenv("HA_URL", "ws://localhost:8123/api/websocket")
    runpy.run_module("hasp", run_name="__main__")
