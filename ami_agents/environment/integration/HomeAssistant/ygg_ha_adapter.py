#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import asyncio
import contextlib
import urllib.parse
import uuid
import httpx
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from rdflib import BNode, Graph, Literal, Namespace, RDF, URIRef

from http import HTTPStatus
import yaml

from ha_utils import (HomeAssistantWS, HomeAssistantRDF, HomeAssistantREST,
                      get_supported_service_fields)

# Load .env file automatically
def load_environment():
    """Load .env file from project root if it exists."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to project root: HomeAssistant/ -> integration/ -> environment/ -> ami_agents/ -> project root
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)

# Load environment variables from .env file
load_environment()

# Namespaces
BASE_FALLBACK = os.getenv("BASE_WS_URI", "http://localhost:8080/").rstrip("/") + "/"
WEBSUB = Namespace("https://purl.org/hmas/websub/")
HCTL   = Namespace("https://www.w3.org/2019/wot/hypermedia#")
JS     = Namespace("https://www.w3.org/2019/wot/json-schema#")
HMAS   = Namespace("https://purl.org/hmas/")
EX     = Namespace("http://example.org/")
WOTSEC = Namespace("https://www.w3.org/2019/wot/security#")
HTV    = Namespace("http://www.w3.org/2011/http#")
JACAMO = Namespace("https://purl.org/hmas/jacamo/")
TD     = Namespace("https://www.w3.org/2019/wot/td#")

WEBHOOK_VERIFY_TIMEOUT = 5.0 # seconds for webhook verification requests

# ---------------- Semantic config loader -----------------
def _load_semantic_config() -> Dict[str, Any]:
    """Load semantic type mappings from the YAML file pointed to by SEMANTIC_CONFIG env var."""
    path = os.getenv("SEMANTIC_CONFIG", "").strip()
    if not path:
        return {}
    # If path is relative, try relative to the HomeAssistant directory
    if not os.path.isabs(path):
        ha_dir = os.path.dirname(os.path.abspath(__file__))
        resolved = os.path.join(ha_dir, os.path.basename(path))
        if os.path.exists(resolved):
            path = resolved
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return (data or {}).get("semantic", {})
    except Exception as exc:
        print(f"Warning: could not load SEMANTIC_CONFIG from {path!r}: {exc}")
        return {}

_SEMANTIC_CFG: Dict[str, Any] = _load_semantic_config()
if _SEMANTIC_CFG:
    print(f"✓ Loaded semantic config: workspace_type={_SEMANTIC_CFG.get('workspace_type')}, {len(_SEMANTIC_CFG.get('devices', {}))} devices")
else:
    cfg_env = os.getenv("SEMANTIC_CONFIG", "NOT SET")
    print(f"⚠ Semantic config not loaded (SEMANTIC_CONFIG={cfg_env!r})")


def _semantic_workspace_type() -> Optional[URIRef]:
    """Return the ex: URIRef for the workspace type declared in the semantic config, or None."""
    raw = _SEMANTIC_CFG.get("workspace_type", "")
    if not raw or not raw.startswith("ex:"):
        return None
    return EX[raw[3:]]


def _semantic_artifact_type(device_name: str) -> Optional[URIRef]:
    """Return the ex: URIRef for the artifact type of the named device, or None."""
    devices = _SEMANTIC_CFG.get("devices", {})
    raw = (devices.get(device_name) or {}).get("artifact_type", "")
    if not raw or not raw.startswith("ex:"):
        return None
    return EX[raw[3:]]


def _semantic_action_type(device_name: str, svc_name: str) -> URIRef:
    """Return the ex: URIRef for the action type of device+service, falling back to EX.StatusCommand."""
    devices = _SEMANTIC_CFG.get("devices", {})
    raw = (devices.get(device_name) or {}).get("service_types", {}).get(svc_name, "")
    if raw and raw.startswith("ex:"):
        return EX[raw[3:]]
    return EX.StatusCommand

# XSD value type URIs for event payloads
XSD_BOOL   = "http://www.w3.org/2001/XMLSchema#boolean"
XSD_INT    = "http://www.w3.org/2001/XMLSchema#integer"
XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"

# ---------------- Config & App -----------------
HA_URL = os.getenv("HA_URL", "ws://poclea.go.ro:7589/api/websocket")
HA_TOKEN = os.getenv("HA_TOKEN", "")
if not HA_TOKEN:
    raise RuntimeError("HA_TOKEN env var required")
HA_BASE_URL = os.getenv("HA_BASE_URL")
if not HA_BASE_URL:
    HA_BASE_URL = HA_URL.replace("ws://", "http://").replace("wss://", "https://").split("/api/websocket")[0]

# Event forwarder configuration
AREAS = {a.strip() for a in os.getenv("AREAS", "").split(",") if a.strip()}  # allowed area_ids
BASE_WS_URI = os.getenv("BASE_WS_URI", BASE_FALLBACK)  # e.g., https://example.org/ws/lab

# In-memory store for WebSub subscriptions
# In a production environment, this would be a persistent database
subscriptions: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Yggdrasil to Home Assistant adapter")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ha_client = HomeAssistantWS(HA_URL, HA_TOKEN)
ha_rest = HomeAssistantREST(HA_BASE_URL, HA_TOKEN)

@app.on_event("shutdown")
async def _shutdown():
    # Stop background forwarder if running
    task = getattr(app.state, "forward_task", None)
    if task:
        print("Shutting down forwarder task...")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    await ha_client.close()
    await ha_rest.close()

# Optional: allow running directly and gracefully handling Ctrl+C
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
    except KeyboardInterrupt:
        # Uvicorn triggers FastAPI shutdown; this is a friendly log
        print("\nReceived Ctrl+C, shutting down...")

# ---------------- Helpers -----------------
def _sanitize_unit(unit: Optional[str]) -> Optional[str]:
    """Sanitize unit for use in action names/URLs.
    Maps common symbols and strips non-alphanumerics; returns lowercased token.
    """
    if not unit:
        return None
    mapping = {
        "°C": "degc",
        "°F": "degf",
        "%": "percent",
        "µg/m³": "ugm3",
        "μg/m³": "ugm3",
        "kWh": "kwh",
        "W": "w",
        "Wh": "wh",
        "V": "v",
        "A": "a",
        "lx": "lx",
    }
    if unit in mapping:
        return mapping[unit]
    # generic: keep letters/numbers only
    return "".join(ch for ch in unit if ch.isalnum()).lower() or None


def _camel_token(token: str) -> str:
    token = "".join(ch if ch.isalnum() else " " for ch in token)
    return "".join(part.capitalize() for part in token.split())

def _sensor_action_name(device_class: Optional[str], unit: Optional[str]) -> Optional[str]:
    """Build camelCase action name:
    - With device class: get<DeviceClass>In<Unit>
    - Without device class: getIn<Unit>
    Returns None if unit is missing/empty.
    """
    su_raw = _sanitize_unit(unit)
    if not su_raw:
        return None
    unit_cc = _camel_token(su_raw)
    dc = (device_class or "").strip()
    if dc:
        dc_cc = _camel_token(dc)
        return f"get{dc_cc}In{unit_cc}"
    return f"getIn{unit_cc}"


def _entity_display_name(entity: Optional[Dict[str, Any]], devices_by_id: Dict[str, Dict[str, Any]]) -> str:
    if not entity:
        return ""
    for key in ("name", "original_name"):
        val = entity.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    object_id = ""
    ent_id = entity.get("entity_id", "")
    if isinstance(ent_id, str) and "." in ent_id:
        object_id = ent_id.split(".", 1)[1]
    device = devices_by_id.get(entity.get("device_id"))
    device_name = (device or {}).get("name") if isinstance(device, dict) else None
    if device_name and object_id:
        return f"{device_name}.{object_id}"
    if device_name:
        return device_name
    return object_id or ent_id or "artifact"


def _sensor_action_names(device_class: Optional[str], unit: Optional[str]) -> List[str]:
    """Return canonical + fallback action names for numeric sensors."""
    names: List[str] = []
    for candidate in (
        _sensor_action_name(device_class, unit),
        _sensor_action_name(None, unit),
    ):
        if candidate and candidate not in names:
            names.append(candidate)
    return names


def _binary_sensor_action_names(device_class: Optional[str]) -> List[str]:
    """Return action names for binary sensors."""
    names: List[str] = []
    dc = (device_class or "").strip()
    if dc:
        names.append(f"get{_camel_token(dc)}State")
    names.append("getBinarySensorState")
    return names


def _canonical_label(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


def _normalize_workspace_id(area_id: Optional[str]) -> Optional[str]:
    if not area_id:
        return None
    if not AREAS:
        return area_id
    for ws in AREAS:
        if area_id == ws or area_id.startswith(ws + "_"):
            return ws
    return area_id


def _area_matches(workspace_id: str, candidate: Optional[str]) -> bool:
    if not candidate:
        return False
    if candidate == workspace_id:
        return True
    if candidate.startswith(workspace_id + "_"):
        return True
    norm = _normalize_workspace_id(candidate)
    return bool(norm and norm == workspace_id)


def _workspace_allowed(area_id: Optional[str]) -> bool:
    if not AREAS:
        return True
    normalized = _normalize_workspace_id(area_id)
    return bool(normalized and normalized in AREAS)


def _entity_matches_workspace(entity_id: Optional[str], workspace_id: str) -> bool:
    if not entity_id or "." not in entity_id:
        return False
    object_id = entity_id.split(".", 1)[1]
    return object_id.startswith(workspace_id + "_") or object_id.startswith(workspace_id)


def _format_climate_state(state: Dict[str, Any]) -> Dict[str, Any]:
    attrs = state.get("attributes", {}) if isinstance(state, dict) else {}
    return {
        "state": state.get("state"),
        "currentTemperature": attrs.get("current_temperature"),
        "targetTemperature": attrs.get("temperature"),
        "targetTemperatureLow": attrs.get("target_temp_low"),
        "targetTemperatureHigh": attrs.get("target_temp_high"),
        "hvacAction": attrs.get("hvac_action"),
        "hvacModes": attrs.get("hvac_modes"),
        "presetMode": attrs.get("preset_mode"),
        "availablePresetModes": attrs.get("preset_modes"),
        "minTemperature": attrs.get("min_temp"),
        "maxTemperature": attrs.get("max_temp"),
        "targetTemperatureStep": attrs.get("target_temp_step"),
    }


async def _resolve_device_and_entities(workspace_id: str, artifact_name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    decoded_name = urllib.parse.unquote(artifact_name)
    decoded_canon = _canonical_label(decoded_name)
    devices, entities = await _get_workspace_devices_and_entities(workspace_id)
    dev_by_id = {d["id"]: d for d in devices}
    # Try matching named entities first
    for entity in entities:
        label = entity.get("_artifact_label") or _entity_display_name(entity, dev_by_id)
        object_id = entity.get("entity_id", "").split(".", 1)[-1]
        candidates = {
            label,
            entity.get("_artifact_base_label", ""),
            entity.get("_artifact_slug", ""),
            _entity_display_name(entity, dev_by_id),
            entity.get("entity_id", ""),
            object_id,
        }
        matched = decoded_name in {c for c in candidates if c} or any(
            _canonical_label(c) == decoded_canon for c in candidates if c
        )
        if matched:
            device = dev_by_id.get(entity.get("device_id"))
            device_entities = [entity]
            return device or {}, device_entities, entity, label
    device = next((d for d in devices if d.get("name") == decoded_name), None)
    if not device:
        device = next((d for d in devices if _canonical_label(d.get("name", "")) == decoded_canon), None)
    if not device:
        raise HTTPException(status_code=404, detail="Artifact not found")
    device_entities = [e for e in entities if e.get("device_id") == device["id"]]
    if not device_entities:
        raise HTTPException(status_code=404, detail="No entities for artifact")
    return device, device_entities, None, decoded_name


async def _get_workspace_devices_and_entities(workspace_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    devices = await ha_client.get_devices()
    entities = await ha_client.get_entities()
    workspace_device_ids = {d["id"] for d in devices if _area_matches(workspace_id, d.get("area_id"))}
    workspace_entities: List[Dict[str, Any]] = []
    external_entities: List[Dict[str, Any]] = []
    for ent in entities:
        dev_id = ent.get("device_id")
        ent_area = ent.get("area_id")
        ent_match = _area_matches(workspace_id, ent_area) or _entity_matches_workspace(ent.get("entity_id"), workspace_id)
        if ent_match or (dev_id in workspace_device_ids):
            workspace_entities.append(ent)
            if dev_id:
                workspace_device_ids.add(dev_id)
        elif _area_matches(workspace_id, next((d.get("area_id") for d in devices if d.get("id") == dev_id), None)):
            external_entities.append(ent)
    filtered_devices = [d for d in devices if d["id"] in workspace_device_ids]
    dev_by_id = {d["id"]: d for d in filtered_devices}
    label_counts: Dict[str, int] = {}
    def _register_label(ent: Dict[str, Any]) -> None:
        label = _entity_display_name(ent, dev_by_id)
        ent["_artifact_base_label"] = label
        label_counts[label] = label_counts.get(label, 0) + 1
    filtered_entities: List[Dict[str, Any]] = []
    for ent in workspace_entities:
        if AREAS:
            ent_area = ent.get("area_id") or (dev_by_id.get(ent.get("device_id"), {}) or {}).get("area_id")
            if not _workspace_allowed(ent_area):
                continue
        filtered_entities.append(ent)
        _register_label(ent)
    workspace_entities = filtered_entities
    for ent in external_entities:
        if ent.get("device_id") in workspace_device_ids:
            if AREAS:
                ent_area = ent.get("area_id") or (dev_by_id.get(ent.get("device_id"), {}) or {}).get("area_id")
                if not _workspace_allowed(ent_area):
                    continue
            workspace_entities.append(ent)
            _register_label(ent)
    for ent in workspace_entities:
        base_label = ent.get("_artifact_base_label", "artifact")
        label = base_label
        if label_counts.get(base_label, 0) > 1:
            suffix = ent.get("entity_id", "")
            object_id = suffix.split(".", 1)[1] if isinstance(suffix, str) and "." in suffix else suffix
            label = f"{base_label} ({object_id})"
        ent["_artifact_label"] = label
        ent["_artifact_slug"] = urllib.parse.quote(label, safe="")
    return filtered_devices, workspace_entities

def _pick_entity(device_entities: List[Dict[str, Any]], domain: str) -> Optional[str]:
    for e in device_entities:
        if e.get("entity_id", "").startswith(domain + "."):
            return e["entity_id"]
    return None

# ---------------- Endpoints -----------------
@app.get("/", response_class=Response,
         responses={200: {"content": {"text/turtle": {}}}})
async def get_platform(request: Request):
    """Get the HypermediaMASPlatform representation."""
    try:
        areas = await ha_client.get_areas()
        # Filter areas based on AREAS configuration
        if AREAS:
            filtered_areas = [a for a in areas if _workspace_allowed(a.get("area_id"))]
        else:
            filtered_areas = areas

        rdf = HomeAssistantRDF(str(request.base_url))
        rdf.platform_to_rdf(filtered_areas)
        return Response(rdf.serialize(), media_type="text/turtle")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/workspaces", response_class=Response,
         responses={200: {"content": {"text/turtle": {}}}})
async def list_workspaces(request: Request):
    try:
        areas = await ha_client.get_areas()
        rdf = HomeAssistantRDF(str(request.base_url))
        for a in areas:
            rdf.workspace_to_rdf(a, [])
        return Response(rdf.serialize(), media_type="text/turtle")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ---------------- Event forwarder background task -----------------
def _infer_value_and_type(state: str) -> Tuple[Any, str]:
    if state in ("on", "off"):
        return (state == "on"), XSD_BOOL
    try:
        i = int(state)
        return i, XSD_INT
    except Exception:
        pass
    try:
        f = float(state)
        return f, XSD_DOUBLE
    except Exception:
        pass
    return state, XSD_STRING

async def _ws_handshake(url: str, token: str):
    ws = await websockets.connect(url)
    msg = json.loads(await ws.recv())
    if msg.get("type") != "auth_required":
        raise RuntimeError("Unexpected handshake")
    await ws.send(json.dumps({"type": "auth", "access_token": token}))
    msg = json.loads(await ws.recv())
    if msg.get("type") != "auth_ok":
        raise RuntimeError("Auth failed")
    return ws

async def _build_entity_area_map() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    ws = await _ws_handshake(HA_URL, HA_TOKEN)
    try:
        await ws.send(json.dumps({"id": 1, "type": "config/device_registry/list"}))
        await ws.send(json.dumps({"id": 2, "type": "config/entity_registry/list"}))
        devices = entities = None
        while devices is None or entities is None:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "result" and msg.get("success"):
                if msg.get("id") == 1:
                    devices = msg["result"]
                elif msg.get("id") == 2:
                    entities = msg["result"]
        dev_by_id = {d["id"]: d for d in devices}
        ent_to_area: Dict[str, str] = {}
        ent_to_device: Dict[str, str] = {}
        for e in entities:
            area_id = e.get("area_id")
            if not area_id and e.get("device_id"):
                area_id = dev_by_id.get(e["device_id"], {}).get("area_id")
            workspace_id = _normalize_workspace_id(area_id)
            if AREAS:
                if workspace_id not in AREAS:
                    continue
                ent_to_area[e["entity_id"]] = workspace_id
            elif area_id:
                ent_to_area[e["entity_id"]] = area_id
            if e.get("device_id"):
                ent_to_device[e["entity_id"]] = e["device_id"]
        ent_by_id = {e["entity_id"]: e for e in entities}
        return ent_to_area, ent_to_device, dev_by_id, ent_by_id
    finally:
        await ws.close()

async def distribute_to_websub_subscribers(http_client: httpx.AsyncClient, event_payload: Dict[str, Any]):
    """
    Distributes an event payload to all active WebSub subscribers
    whose topics match the event's artifact URI.
    """
    artifact_uri = event_payload.get("artifactUri")
    if not artifact_uri:
        print("WebSub distributor: Event payload missing artifactUri, skipping distribution.")
        return

    # Derive workspace URI from artifact URI for hierarchical matching
    # Expected format: .../workspaces/{id}/artifacts/{name}#artifact
    workspace_uri = None
    if "/artifacts/" in artifact_uri:
        workspace_uri = artifact_uri.split("/artifacts/")[0]

    # Iterate over a copy to prevent issues if subscriptions change during iteration
    for sub_id, sub in list(subscriptions.items()):
        topic = sub.get("topic")
        should_notify = False
        
        # 1. Exact match (Artifact subscription)
        if topic == artifact_uri:
            should_notify = True
        # 2. Workspace match (Workspace subscription)
        elif workspace_uri and (topic == workspace_uri or topic == f"{workspace_uri}#workspace"):
             should_notify = True
             
        if should_notify:
            callback_url = sub.get("callback")
            if not callback_url:
                print(f"WebSub distributor: Subscription {sub_id} missing callback URL.")
                continue

            try:
                print(f"WebSub distributor: Posting event for {artifact_uri} to callback: {callback_url}")
                r = await http_client.post(
                    callback_url,
                    json=event_payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-WebSub-Topic": artifact_uri, # Custom header for context
                    },
                    timeout=5 # Short timeout for callbacks
                )
                r.raise_for_status()
                print(f"WebSub distributor: Successfully delivered event to {callback_url} (Status: {r.status_code})")
            except httpx.RequestError as e:
                print(f"WebSub distributor: Failed to deliver event to {callback_url} for topic {artifact_uri}: Network error: {e}")
            except httpx.HTTPStatusError as e:
                print(f"WebSub distributor: Failed to deliver event to {callback_url} for topic {artifact_uri}: HTTP error: {e.response.status_code} - {e.response.text[:100]}")
            except Exception as e:
                print(f"WebSub distributor: An unexpected error occurred while delivering event to {callback_url}: {e}")

async def _event_forwarder_task():
    ent_to_area, ent_to_device, dev_by_id, ent_by_id = await _build_entity_area_map()
    print("Starting event forwarder task; areas=", (sorted(AREAS) if AREAS else "ALL"))
    async with httpx.AsyncClient(timeout=10) as http:
        while True:
            ws = None
            try:
                ws = await _ws_handshake(HA_URL, HA_TOKEN)
                await ws.send(json.dumps({"id": 100, "type": "subscribe_events", "event_type": "state_changed"}))
                print("Forwarder subscribed to state_changed events")
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("type") != "event":
                        continue
                    ev = msg.get("event", {})
                    if ev.get("event_type") != "state_changed":
                        continue
                    data = ev.get("data", {})
                    entity_id = data.get("entity_id")
                    if entity_id != "sensor.clock_308":
                        print(f"DEBUG: Received state_changed for {entity_id}") # Debug 1
                    new      = data.get("new_state") or {}
                    state    = new.get("state")
                    attrs    = new.get("attributes", {})
                    tstamp   = ev.get("time_fired")
                    if not entity_id or state in (None, "unknown", "unavailable"):
                        continue
                    area_id = ent_to_area.get(entity_id)
                    if not area_id or (AREAS and area_id not in AREAS):
                        print(f"Dropped event for {entity_id}: area_id={area_id} (Allowed: {AREAS})")
                        continue
                    # Determine artifact name from device name; fallback to object_id
                    entity_meta = ent_by_id.get(entity_id)
                    artifact_label = _entity_display_name(entity_meta, dev_by_id)
                    artifact_name = urllib.parse.quote(artifact_label, safe="")
                    prop = attrs.get("device_class") or "state"
                    value, xtype = _infer_value_and_type(state)
                    artifact_profile = f"{BASE_WS_URI.rstrip('/')}/workspaces/{area_id}/artifacts/{artifact_name}"
                    artifact_uri = f"{artifact_profile}#artifact"
                    property_uri = f"{artifact_profile}/props/{prop}"
                    trigger_uri  = f"{artifact_profile}/actions/read"
                    payload = {
                        "artifactUri": artifact_uri,
                        "propertyUri": property_uri,
                        "value": value,
                        "valueTypeUri": xtype,
                        "timestamp": tstamp,
                        "triggerUri": trigger_uri,
                    }
                    
                    # Suppress noisy debug logs for the hardcoded lab clock sensor
                    if entity_id != "sensor.clock_308":
                        print(f"DEBUG: Forwarding event for {entity_id} to WebSub. Topic: {artifact_uri}") # Debug 2
                    # Distribute to WebSub subscribers
                    await distribute_to_websub_subscribers(http, payload)
            except asyncio.CancelledError:
                print("Forwarder task cancelled; exiting loop")
                break
            except Exception as outer:
                print(f"Event loop error: {outer}; reconnecting in 3s...")
                await asyncio.sleep(3)
            finally:
                if ws is not None:
                    with contextlib.suppress(Exception):
                        await ws.close()



@app.on_event("startup")
async def _startup_forwarder():
    print(f"App startup: AREAS={sorted(AREAS) if AREAS else 'ALL'}, BASE_WS_URI={BASE_WS_URI}")
    
    app.state.forward_task = asyncio.create_task(_event_forwarder_task())
    print("Forwarder task scheduled")

# Simple status endpoint for debugging forwarder
@app.get("/_forwarder/status")
async def forwarder_status():
    task = getattr(app.state, "forward_task", None)
    return {
        "areas": sorted(AREAS) if AREAS else [],
        "baseWsUri": BASE_WS_URI,
        "taskRunning": bool(task) and not task.done(),
    }

@app.get("/workspaces/{workspace_id}", response_class=Response,
         responses={200: {"content": {"text/turtle": {}}}, 404: {"description": "Not found"}})
async def workspace(workspace_id: str, request: Request):
    try:
        areas = await ha_client.get_areas()
        area = next((a for a in areas if a["area_id"] == workspace_id), None)
        if area is None:
            raise HTTPException(status_code=404, detail="Workspace not found")
        devices, _ = await _get_workspace_devices_and_entities(workspace_id)
        rdf = HomeAssistantRDF(str(request.base_url))
        rdf.workspace_to_rdf(area, devices)
        ws_type = _semantic_workspace_type()
        if ws_type:
            ws_uri = URIRef(f"{rdf.base}workspaces/{workspace_id}#workspace")
            rdf.g.add((ws_uri, RDF.type, ws_type))
        for d in devices:
            device_name = d.get("name", d.get("id"))
            art_type = _semantic_artifact_type(device_name)
            if art_type:
                safe_name = urllib.parse.quote(device_name, safe="")
                art_uri = URIRef(f"{rdf.base}workspaces/{workspace_id}/artifacts/{safe_name}#artifact")
                rdf.g.add((art_uri, RDF.type, art_type))
        return Response(rdf.serialize(), media_type="text/turtle")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/workspaces/{workspace_id}/artifacts", response_class=Response,
         responses={200: {"content": {"text/turtle": {}}}, 404: {"description": "Not found"}})
async def list_artifacts(workspace_id: str, request: Request):
    try:
        areas = await ha_client.get_areas()
        area = next((a for a in areas if a["area_id"] == workspace_id), None)
        if area is None:
            raise HTTPException(status_code=404, detail="Workspace not found")
        devices, entities = await _get_workspace_devices_and_entities(workspace_id)
        rdf = HomeAssistantRDF(str(request.base_url))
        aid = area["area_id"]
        ws = URIRef(f"{rdf.base}workspaces/{aid}#workspace")
        art_dir = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/")
        for ent in entities:
            label = ent.get("_artifact_label") or _entity_display_name(ent, {d["id"]: d for d in devices})
            safe_name = ent.get("_artifact_slug") or urllib.parse.quote(label, safe="")
            art = URIRef(f"{art_dir}{safe_name}#artifact")
            rdf.g.add((art, RDF.type, HMAS.Artifact))
            art_type = _semantic_artifact_type(label)
            if art_type:
                rdf.g.add((art, RDF.type, art_type))
            rdf.g.add((ws, HMAS.contains, art))
            rdf.g.add((art, TD.title, Literal(label)))
        return Response(rdf.serialize(), media_type="text/turtle")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/workspaces/{workspace_id}/artifacts/{artifact_name}", response_class=Response,
         responses={200: {"content": {"text/turtle": {}, "application/json": {}}}, 404: {"description": "Not found"}})
async def get_artifact(workspace_id: str, artifact_name: str, request: Request):
    """Return RDF TD (Accept: text/turtle) or JSON snapshot (application/json) for an artifact."""
    try:
        device, device_entities, primary_entity, artifact_label = await _resolve_device_and_entities(workspace_id, artifact_name)
        states = await ha_rest.get_states()
        state_map = {s["entity_id"]: s for s in states}

        #print(states)
        
        rdf = HomeAssistantRDF(str(request.base_url))
        aid = workspace_id
        ws = URIRef(f"{rdf.base}workspaces/{aid}#workspace")
        art_dir = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/")
        safe_name = urllib.parse.quote(artifact_label, safe="")
        art = URIRef(f"{art_dir}{safe_name}#artifact")

        # Build RDF
        rdf.g.add((art, RDF.type, TD.Thing))
        rdf.g.add((art, RDF.type, HMAS.Artifact))
        rdf.g.add((art, TD.title, Literal(artifact_label)))
        domains = {e["entity_id"].split(".")[0] for e in device_entities}
        art_type = _semantic_artifact_type(artifact_label)
        if art_type:
            rdf.g.add((art, RDF.type, art_type))
        sec = BNode()
        rdf.g.add((art, TD.hasSecurityConfiguration, sec))
        rdf.g.add((sec, RDF.type, WOTSEC.NoSecurityScheme))

        # Actions: discover dynamically from HA services for all domains on this artifact
        try:
            services = await ha_rest.get_services()
        except Exception:
            services = []
        # Build map domain -> services dict
        svc_by_domain: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for svc in services:
            dom = svc.get("domain")
            if not dom:
                continue
            svc_by_domain[dom] = svc.get("services", {}) or {}

        for domain in sorted(domains):
            domain_svcs = svc_by_domain.get(domain, {})

            # Get a representative entity for this domain to check capabilities
            domain_entity = _pick_entity(device_entities, domain)
            domain_entity_state = state_map.get(domain_entity, {}) if domain_entity else {}
            domain_entity_attrs = domain_entity_state.get("attributes", {}) if isinstance(domain_entity_state, dict) else {}

            for svc_name, definition in domain_svcs.items():
                legacy_applies = "entity_id" in (definition.get("fields") or {})
                modern_applies = any(
                    domain in (entry.get("domain") or [])
                    for entry in (definition.get("target") or {}).get("entity", [])
                )
                if not (legacy_applies or modern_applies):
                    continue
                # CamelCase action name for HA services, keep URL stable under /ha/{domain}/{service}
                action_name = f"{_camel_token(domain)}{_camel_token(svc_name)}"

                # Filter service fields to only those supported by this entity
                all_service_fields = definition.get("fields", {})
                supported_fields = get_supported_service_fields(domain, domain_entity_attrs, all_service_fields)

                # Skip this service if no fields are supported (after filtering)
                # But keep entity_id-only services
                if not supported_fields and "entity_id" not in all_service_fields:
                    continue

                # Build input schema from supported service fields only
                input_schema = rdf._build_input_schema_from_fields(supported_fields)

                # Get service description
                service_description = definition.get("description")

                rdf._add_action(
                    art,
                    action_name,
                    _semantic_action_type(artifact_label, svc_name),
                    "POST",
                    URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/ha/{urllib.parse.quote(domain, safe='')}/{urllib.parse.quote(svc_name, safe='')}"),
                    "application/json",
                    input_schema=input_schema,
                    description=service_description,
                )

        # Sensors and binary sensors are read-only; their state and attributes
        # are exposed as PropertyAffordance above. No ActionAffordances.

        if "climate" in domains:
            climate_ent = _pick_entity(device_entities, "climate")
            if climate_ent and climate_ent in state_map:
                # Build output schema
                output_schema = rdf._build_climate_output_schema()

                rdf._add_action(
                    art,
                    "getThermostatState",
                    EX.StatusCommand,
                    "POST",
                    URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/getThermostatState"),
                    "application/json",
                    output_schema=output_schema,
                )

        # Property Affordances: Add properties for each entity's state attributes
        for entity in device_entities:
            entity_id = entity.get("entity_id")
            if not entity_id:
                continue

            entity_state = state_map.get(entity_id, {})
            if not entity_state:
                continue

            # Get entity attributes
            entity_attrs = entity_state.get("attributes", {}) if isinstance(entity_state, dict) else {}

            # Add property for the state itself
            state_value = entity_state.get("state")
            if state_value and state_value not in ("unknown", "unavailable"):
                # Get entity domain for type detection
                entity_domain = entity_id.split(".")[0] if "." in entity_id else ""

                # Parse numeric state values for sensor domains or entities with unit_of_measurement
                unit = entity_attrs.get("unit_of_measurement")
                schema_value = state_value

                # Check if this is a sensor domain or has a unit of measurement
                if (entity_domain == "sensor" or unit) and isinstance(state_value, str):
                    # Try to parse as number for schema generation
                    try:
                        if '.' not in state_value:
                            schema_value = int(state_value)
                        else:
                            schema_value = float(state_value)
                    except ValueError:
                        # If parsing fails, keep as string
                        pass

                property_uri = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/properties/state")
                # Pass domain to schema builder for context-aware schema generation
                schema = rdf._build_property_schema("state", schema_value, entity_attrs, entity_domain=entity_domain)
                rdf._add_property(
                    art,
                    "state",
                    property_uri,
                    output_schema=schema,
                    description=f"Current state of {entity_id}",
                    observable=True
                )

            # Add properties for each attribute (filtering out metadata)
            if entity_attrs:
                from ha_utils import get_operational_attributes, get_metadata_attributes
                domain = entity_id.split(".")[0] if "." in entity_id else ""
                operational_attrs = get_operational_attributes(domain, entity_attrs)

                for attr_name, attr_value in operational_attrs.items():
                    # Skip None values
                    if attr_value is None:
                        continue

                    # Build property URI
                    property_uri = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/properties/{urllib.parse.quote(attr_name, safe='')}")

                    # Build schema for this property
                    schema = rdf._build_property_schema(attr_name, attr_value, entity_attrs)

                    # Add the property affordance
                    rdf._add_property(
                        art,
                        attr_name,
                        property_uri,
                        output_schema=schema,
                        description=f"{attr_name} of {entity_id}",
                        observable=True
                    )

                # Add special "metadata" property affordance
                # Include both attributes and state-level metadata (timestamps)
                metadata_attrs = get_metadata_attributes(entity_attrs)
                # Add timestamps from state object
                for ts_field in ("last_changed", "last_reported", "last_updated"):
                    if ts_field in entity_state:
                        metadata_attrs[ts_field] = entity_state[ts_field]

                if metadata_attrs:
                    metadata_property_uri = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/properties/metadata")
                    metadata_schema = rdf._build_metadata_schema(metadata_attrs)
                    rdf._add_property(
                        art,
                        "metadata",
                        metadata_property_uri,
                        output_schema=metadata_schema,
                        description=f"Metadata attributes of {entity_id}",
                        observable=False  # Metadata is typically not observable
                    )

        # Generic Jacamo/WebSub affordances
        rdf._add_action(art, "getArtifactRepresentation", JACAMO.PerceiveArtifact, "GET",
                        URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}"), "application/json")
        rdf._add_action(art, "updateArtifactRepresentation", JACAMO.UpdateArtifact, "PUT",
                        URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}"), "application/json")
        rdf._add_action(art, "deleteArtifactRepresentation", JACAMO.DeleteArtifact, "DELETE",
                        URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}"), "application/json")
        rdf._add_action(art, "focusArtifact", JACAMO.Focus, "POST",
                        URIRef(f"{rdf.base}workspaces/{aid}/focus"), "application/json")
        rdf._add_action(art, "subscribeToArtifact", WEBSUB.subscribeToArtifact, "POST",
                        URIRef(f"{rdf.base}hub/"), "application/json", "websub")
        rdf._add_action(art, "unsubscribeFromArtifact", WEBSUB.unsubscribeFromArtifact, "POST",
                        URIRef(f"{rdf.base}hub/"), "application/json", "websub")

        # Content negotiation
        accept = request.headers.get("accept", "").lower()
        #if "text/turtle" in accept:
        if True:
            rdf.g.add((art, HMAS.isContainedIn, ws))
            rdf.g.add((ws, RDF.type, HMAS.Workspace))
            profile = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}")
            rdf.g.add((profile, RDF.type, HMAS.ResourceProfile))
            rdf.g.add((profile, HMAS.isProfileOf, art))
            return Response(rdf.serialize(), media_type="text/turtle")
        else:
            snapshot = {
                "artifact": artifact_label,
                "workspace": aid,
                "entities": {
                    e["entity_id"]: state_map.get(e["entity_id"], {}) for e in device_entities
                }
            }
            return JSONResponse(snapshot)
    except HTTPException:
        raise
    except Exception as exc:
        print(exc)
        raise HTTPException(status_code=500, detail=str(exc))



async def _ensure_entity(workspace_id: str, artifact_name: str, domain: str) -> str:
    _, device_entities, _, _ = await _resolve_device_and_entities(workspace_id, artifact_name)
    ent = _pick_entity(device_entities, domain)
    if not ent:
        raise HTTPException(status_code=404, detail=f"No {domain} entity on artifact")
    return ent

# Property affordance endpoint - read property value
@app.get("/workspaces/{workspace_id}/artifacts/{artifact_name}/properties/{property_name}")
async def read_property(workspace_id: str, artifact_name: str, property_name: str):
    """Read the current value of a property from a Home Assistant entity."""
    try:
        # Resolve the artifact to get its entities
        _, device_entities, _, _ = await _resolve_device_and_entities(workspace_id, artifact_name)

        # Get current states
        states = await ha_rest.get_states()
        state_map = {s["entity_id"]: s for s in states}

        # Decode the property name
        decoded_property_name = urllib.parse.unquote(property_name)

        # Search through all entities in this artifact for the requested property
        for entity in device_entities:
            entity_id = entity.get("entity_id")
            if not entity_id:
                continue

            entity_state = state_map.get(entity_id, {})
            if not entity_state:
                continue

            entity_attrs = entity_state.get("attributes", {}) if isinstance(entity_state, dict) else {}

            # Check if this is the special "metadata" property
            if decoded_property_name == "metadata":
                from ha_utils import get_metadata_attributes
                metadata_attrs = get_metadata_attributes(entity_attrs)
                # Add timestamps from state object
                for ts_field in ("last_changed", "last_reported", "last_updated"):
                    if ts_field in entity_state:
                        metadata_attrs[ts_field] = entity_state[ts_field]

                if metadata_attrs:
                    # Return the metadata object, matching the ObjectSchema
                    return JSONResponse(metadata_attrs)

            # Check if this is the "state" property
            if decoded_property_name == "state":
                state_value = entity_state.get("state")
                if state_value and state_value not in ("unknown", "unavailable"):
                    # Get entity domain for type detection
                    entity_domain = entity_id.split(".")[0] if "." in entity_id else ""

                    # Parse numeric state values for sensor domains or entities with unit_of_measurement
                    unit = entity_attrs.get("unit_of_measurement")

                    # Check if this is a sensor domain or has a unit of measurement
                    if (entity_domain == "sensor" or unit) and isinstance(state_value, str):
                        # Try to parse as number
                        try:
                            # Try integer first
                            if '.' not in state_value:
                                state_value = int(state_value)
                            else:
                                state_value = float(state_value)
                        except ValueError:
                            # If parsing fails, keep as string
                            pass
                    # Return the value, matching the output schema
                    return JSONResponse(state_value)

            # Check in entity attributes
            if decoded_property_name in entity_attrs:
                attr_value = entity_attrs[decoded_property_name]
                if attr_value is not None:
                    # Return the raw value, matching the output schema
                    return JSONResponse(attr_value)

        # Property not found in any entity
        raise HTTPException(status_code=404, detail=f"Property '{decoded_property_name}' not found on artifact")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# Generic HA service forwarder for dynamically discovered actions
@app.post("/workspaces/{workspace_id}/artifacts/{artifact_name}/ha/{domain}/{service}")
async def action_ha_service(workspace_id: str, artifact_name: str, domain: str, service: str, request: Request):
    # Ensure entity for the requested domain exists on this artifact
    ent = await _ensure_entity(workspace_id, artifact_name, domain)

    # Validate the service exists for the domain
    try:
        services = await ha_rest.get_services()
    except Exception:
        services = []
    svc = next((s for s in services if s.get("domain") == domain), None)
    if not svc or service not in (svc.get("services") or {}):
        raise HTTPException(status_code=404, detail="Service not found for domain")

    payload = {}
    if request.headers.get("content-length") not in (None, "0"):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
    payload = {**payload, "entity_id": ent}

    await ha_rest.call_service(domain, service, payload)
    return Response(content="Action succeeded:")

# Jacamo/WebSub stubs
@app.api_route("/workspaces/{workspace_id}/focus", methods=["POST", "GET"])
async def focus_workspace(workspace_id: str, request: Request):
    """
    Handle Jacamo Focus action by registering a WebSub subscription.
    
    Payload expected:
    {
        "artifactName": "optional, if focusing on specific artifact",
        "callbackUrl": "required, where to send events"
    }
    """
    if request.method == "GET":
        body = dict(request.query_params)
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    callback_url = body.get("callbackUrl")
    # if not callback_url:
    #    raise HTTPException(status_code=400, detail="callbackUrl is required")
        
    artifact_name = body.get("artifactName")
    
    # Determine Topic URI using configured BASE_WS_URI
    base = BASE_WS_URI.rstrip("/")
    
    if artifact_name:
        # Resolve artifact URI
        # Check if artifact exists in the workspace
        try:
            await _resolve_device_and_entities(workspace_id, artifact_name)
        except HTTPException:
             pass

        safe_name = urllib.parse.quote(artifact_name, safe="")
        topic = f"{base}/workspaces/{workspace_id}/artifacts/{safe_name}#artifact"
    else:
        # Workspace focus
        topic = f"{base}/workspaces/{workspace_id}"
        
    if callback_url:
        # Register subscription directly
        subscription_id = f"{topic}-{callback_url}"
        subscriptions[subscription_id] = {
            "topic": topic,
            "callback": callback_url,
            "lease_seconds": None, # Infinite focus until explicit unfocus/unsubscribe
            "timestamp": asyncio.get_event_loop().time(),
            "type": "focus"
        }
        print(f"Agent focused on {topic} -> {callback_url}")
    else:
        print(f"Agent focused on {topic} (no callback)")

    return Response(content="Focus succeeded")

@app.post("/hub/")
async def hub(request: Request):
    """
    Handles WebSub subscribe and unsubscribe requests.
    Implements the hub's side of the WebSub protocol for intent verification.
    """
    form_data = await request.form()
    
    hub_mode = form_data.get("hub.mode")
    hub_topic = form_data.get("hub.topic")
    hub_callback = form_data.get("hub.callback")
    hub_lease_seconds = form_data.get("hub.lease_seconds")
    hub_secret = form_data.get("hub.secret")

    if not all([hub_mode, hub_topic, hub_callback]):
        raise HTTPException(
            status_code=400,
            detail="Missing required WebSub parameters: hub.mode, hub.topic, hub.callback"
        )

    print(f"WebSub request: mode={hub_mode}, topic={hub_topic}, callback={hub_callback}")

    # Intent Verification
    challenge = str(uuid.uuid4())
    verify_params = {
        "hub.mode": hub_mode,
        "hub.topic": hub_topic,
        "hub.callback": hub_callback,
        "hub.challenge": challenge,
    }
    if hub_lease_seconds:
        verify_params["hub.lease_seconds"] = hub_lease_seconds

    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_VERIFY_TIMEOUT) as client:
            print(f"Sending intent verification GET to {hub_callback} with params: {verify_params}")
            verify_response = await client.get(hub_callback, params=verify_params)
            verify_response.raise_for_status()

            if verify_response.text != challenge:
                print(f"Intent verification failed: Challenge mismatch for callback {hub_callback}")
                raise HTTPException(
                    status_code=409, # Conflict
                    detail="Intent verification challenge response mismatch"
                )
            print(f"Intent verification successful for callback {hub_callback}")

    except httpx.RequestError as e:
        print(f"Intent verification failed: Network error connecting to {hub_callback}: {e}")
        raise HTTPException(
            status_code=412, # Precondition Failed
            detail=f"Intent verification failed: Network error connecting to callback URL: {e}"
        )
    except httpx.HTTPStatusError as e:
        print(f"Intent verification failed: HTTP error from {hub_callback}: {e}")
        raise HTTPException(
            status_code=412, # Precondition Failed
            detail=f"Intent verification failed: HTTP error from callback URL: {e}"
        )

    subscription_id = f"{hub_topic}-{hub_callback}" # Simple unique ID for now

    if hub_mode == "subscribe":
        subscriptions[subscription_id] = {
            "topic": hub_topic,
            "callback": hub_callback,
            "lease_seconds": int(hub_lease_seconds) if hub_lease_seconds else None,
            "secret": hub_secret,
            "timestamp": asyncio.get_event_loop().time(), # Store subscription time
        }
        print(f"Subscription added for topic: {hub_topic}, callback: {hub_callback}")
        return Response(status_code=202, content="Subscribed")

    elif hub_mode == "unsubscribe":
        if subscription_id in subscriptions:
            del subscriptions[subscription_id]
            print(f"Unsubscription successful for topic: {hub_topic}, callback: {hub_callback}")
            return Response(status_code=202, content="Unsubscribed")
        else:
            print(f"Unsubscription failed: No active subscription found for topic: {hub_topic}, callback: {hub_callback}")
            raise HTTPException(
                status_code=404,
                detail="No active subscription found for this topic and callback."
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid hub.mode. Must be 'subscribe' or 'unsubscribe'."
        )

# PUT/DELETE artifact representation stubs
@app.put("/workspaces/{workspace_id}/artifacts/{artifact_name}")
async def update_artifact_representation(workspace_id: str, artifact_name: str, request: Request):
    _ = await request.json()
    return Response(content="Action succeeded:")

@app.delete("/workspaces/{workspace_id}/artifacts/{artifact_name}")
async def delete_artifact_representation(workspace_id: str, artifact_name: str):
    return Response(content="Action succeeded:")

