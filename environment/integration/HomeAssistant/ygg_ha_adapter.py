#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import asyncio
import contextlib
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from rdflib import BNode, Graph, Literal, Namespace, RDF, URIRef

from http import HTTPStatus
from ha_utils import (HomeAssistantWS, HomeAssistantRDF, HomeAssistantREST,
                      get_supported_service_fields)

# Namespaces
BASE_FALLBACK = "http://localhost:8080/"
WEBSUB = Namespace("https://purl.org/hmas/websub/")
HCTL   = Namespace("https://www.w3.org/2019/wot/hypermedia#")
JS     = Namespace("https://www.w3.org/2019/wot/json-schema#")
HMAS   = Namespace("https://purl.org/hmas/")
EX     = Namespace("http://example.org/")
WOTSEC = Namespace("https://www.w3.org/2019/wot/security#")
HTV    = Namespace("http://www.w3.org/2011/http#")
JACAMO = Namespace("https://purl.org/hmas/jacamo/")
TD     = Namespace("https://www.w3.org/2019/wot/td#")
SOSA   = Namespace("http://www.w3.org/ns/sosa/")
SSN    = Namespace("http://www.w3.org/ns/ssn/")
QUDT   = Namespace("http://qudt.org/schema/qudt/")
UNIT   = Namespace("http://qudt.org/vocab/unit/")
TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")

# XSD value type URIs for event payloads
XSD_BOOL   = "http://www.w3.org/2001/XMLSchema#boolean"
XSD_INT    = "http://www.w3.org/2001/XMLSchema#integer"
XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"

# ---------------- Config & App -----------------
HA_URL = os.getenv("HA_URL", "ws://localhost:8123/api/websocket")
HA_TOKEN = os.getenv("HA_TOKEN", "").strip()
if not HA_TOKEN:
    raise RuntimeError("HA_TOKEN env var required")
HA_BASE_URL = os.getenv("HA_BASE_URL")
if not HA_BASE_URL:
    HA_BASE_URL = HA_URL.replace("ws://", "http://").replace("wss://", "https://").split("/api/websocket")[0]

# Event forwarder configuration
MONITOR_URL = os.getenv("MONITOR_URL", os.getenv("FORWARD_URL", ""))  # destination to POST event JSON / reset
EXPLORER_URL = os.getenv("EXPLORER_URL", "")  # Environment Explorer base URL for admin reset
AREAS = {a.strip() for a in os.getenv("AREAS", "").split(",") if a.strip()}  # allowed area_ids
BASE_WS_URI = os.getenv("BASE_WS_URI", BASE_FALLBACK)  # e.g., https://example.org/ws/lab


def _load_tdsosa_overrides() -> Dict[str, str]:
    raw = os.getenv("TD_SOSA_ENV_VAR_OVERRIDES", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception as exc:
        print("Invalid TD_SOSA_ENV_VAR_OVERRIDES JSON:", exc)
        return {}
    if not isinstance(data, dict):
        print("TD_SOSA_ENV_VAR_OVERRIDES must be a JSON object")
        return {}
    out: Dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        key = " ".join(k.strip().lower().split())
        val = v.strip()
        if key and val:
            out[key] = val
    return out


TD_SOSA_ENV_VAR_OVERRIDES = _load_tdsosa_overrides()

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
    # Reset external monitors/explorers on shutdown
    try:
        await _post_monitor_reset()
    except Exception as e:
        print("Shutdown monitor reset failed:", e)
    try:
        await _post_explorer_reset()
    except Exception as e:
        print("Shutdown explorer reset failed:", e)
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


def _normalize_override_key(key: str) -> str:
    return " ".join(key.strip().lower().split())


def _to_override_token(value: Optional[str]) -> str:
    return _normalize_override_key((value or "").replace(" ", "_"))


def _resolve_env_var_override(
    *,
    entity: Optional[Dict[str, Any]] = None,
    device: Optional[Dict[str, Any]] = None,
    artifact_label: Optional[str] = None,
    domain: Optional[str] = None,
    signal_name: Optional[str] = None,
    service_name: Optional[str] = None,
) -> Optional[str]:
    if not TD_SOSA_ENV_VAR_OVERRIDES:
        return None
    ent_id = _to_override_token((entity or {}).get("entity_id"))
    dev_id = _to_override_token((device or {}).get("id"))
    dev_name = _to_override_token((device or {}).get("name"))
    art = _to_override_token(artifact_label)
    dom = _to_override_token(domain)
    sig = _to_override_token(signal_name)
    svc = _to_override_token(service_name)
    keys: List[str] = []
    if svc and dom:
        dsvc = f"{dom}.{svc}"
        if ent_id:
            keys.append(f"entity:{ent_id}:action:{dsvc}")
        if dev_id:
            keys.append(f"device_id:{dev_id}:action:{dsvc}")
        if dev_name:
            keys.append(f"device:{dev_name}:action:{dsvc}")
        if art:
            keys.append(f"artifact:{art}:action:{dsvc}")
        keys.append(f"action:{dsvc}")
    if sig:
        if ent_id:
            keys.append(f"entity:{ent_id}:{sig}")
        if dev_id:
            keys.append(f"device_id:{dev_id}:{sig}")
        if dev_name:
            keys.append(f"device:{dev_name}:{sig}")
        if art:
            keys.append(f"artifact:{art}:{sig}")
        if dom:
            keys.append(f"domain:{dom}:{sig}")
    if ent_id:
        keys.append(f"entity:{ent_id}")
    if dev_id:
        keys.append(f"device_id:{dev_id}")
    if dev_name:
        keys.append(f"device:{dev_name}")
    if art:
        keys.append(f"artifact:{art}")
    if dom:
        keys.append(f"domain:{dom}")
    for key in keys:
        match = TD_SOSA_ENV_VAR_OVERRIDES.get(_normalize_override_key(key))
        if match:
            return match
    return None


def _env_var_uri(base: str, workspace_id: str, env_var: str) -> URIRef:
    if env_var.startswith("http://") or env_var.startswith("https://"):
        return URIRef(env_var)
    return URIRef(f"{base}workspaces/{workspace_id}/environment/{urllib.parse.quote(env_var, safe='')}")


def _ambient_var_from_signals(domain: str, device_class: Optional[str], signal_name: Optional[str]) -> Optional[str]:
    text = f"{domain} {(device_class or '')} {(signal_name or '')}".lower()
    if any(tok in text for tok in ("illumin", "lumin", "bright", "light")):
        return "luminosity"
    if any(tok in text for tok in ("temp", "thermal", "heat", "cool")):
        return "thermal_comfort"
    if "humid" in text:
        return "humidity"
    if "press" in text:
        return "pressure"
    if any(tok in text for tok in ("motion", "occup", "presence", "person")):
        return "occupancy_presence"
    if any(tok in text for tok in ("lock", "security", "alarm", "camera")):
        return "security_state"
    if any(tok in text for tok in ("media", "speaker", "vacuum", "coffee", "appliance")):
        return "activity_state"
    return None


def _ambient_var_for_action(domain: str, service_name: str) -> Optional[str]:
    token = f"{domain}.{service_name}".lower()
    if domain == "climate":
        return "thermal_comfort"
    if domain in {"light", "cover"}:
        return "luminosity"
    if domain == "humidifier":
        return "humidity"
    if domain in {"lock", "alarm_control_panel", "camera"}:
        return "security_state"
    if domain in {"media_player", "vacuum"}:
        return "activity_state"
    return _ambient_var_from_signals(domain, None, token)


def _effect_direction(service_name: str) -> Optional[str]:
    s = service_name.lower()
    inc_markers = ("turn_on", "open", "raise", "increase", "up", "unlock", "start")
    dec_markers = ("turn_off", "close", "lower", "decrease", "down", "lock", "stop")
    if any(marker in s for marker in inc_markers):
        return "increase"
    if any(marker in s for marker in dec_markers):
        return "decrease"
    return None


def _domain_is_observer(domain: str) -> bool:
    return domain in {"sensor", "binary_sensor", "weather", "person", "device_tracker"}


def _domain_is_actuator(domain: str) -> bool:
    return domain in {"light", "cover", "climate", "humidifier", "fan", "switch", "media_player", "vacuum", "lock", "alarm_control_panel"}


def _add_tdsosa_property_links(
    rdf: HomeAssistantRDF,
    property_affordance: BNode,
    feature_of_interest: URIRef,
    env_var_uri: URIRef,
    observable: bool,
    actuatable: bool,
) -> None:
    if observable:
        rdf.g.add((property_affordance, RDF.type, TDSOSA.ObservablePropertyAffordance))
        rdf.g.add((env_var_uri, RDF.type, SOSA.ObservableProperty))
    if actuatable:
        rdf.g.add((property_affordance, RDF.type, TDSOSA.ActuatablePropertyAffordance))
        rdf.g.add((env_var_uri, RDF.type, SOSA.ActuatableProperty))
    rdf.g.add((property_affordance, TDSOSA.affordsProperty, env_var_uri))
    rdf.g.add((feature_of_interest, SSN.hasProperty, env_var_uri))
    rdf.g.add((env_var_uri, SSN.isPropertyOf, feature_of_interest))


def _add_tdsosa_action_effect(
    rdf: HomeAssistantRDF,
    action_affordance: BNode,
    feature_of_interest: URIRef,
    env_var_uri: URIRef,
    actuation_uri: URIRef,
    direction: str,
) -> None:
    rdf.g.add((action_affordance, TDSOSA.hasEffectActuation, actuation_uri))
    rdf.g.add((actuation_uri, RDF.type, SOSA.Actuation))
    rdf.g.add((actuation_uri, RDF.type, TDSOSA.IncreasingActuation if direction == "increase" else TDSOSA.DecreasingActuation))
    rdf.g.add((actuation_uri, SOSA.actsOnProperty, env_var_uri))
    rdf.g.add((actuation_uri, TDSOSA.increasesObservableProperty if direction == "increase" else TDSOSA.decreasesObservableProperty, env_var_uri))
    amount = BNode()
    rdf.g.add((actuation_uri, TDSOSA.amount, amount))
    rdf.g.add((amount, RDF.type, QUDT.QuantityValue))
    rdf.g.add((amount, QUDT.numericValue, Literal(1)))
    rdf.g.add((amount, QUDT.unit, UNIT.UNITLESS))
    rdf.g.add((feature_of_interest, SSN.hasProperty, env_var_uri))
    rdf.g.add((env_var_uri, SSN.isPropertyOf, feature_of_interest))
    rdf.g.add((env_var_uri, RDF.type, SOSA.ObservableProperty))
    rdf.g.add((env_var_uri, RDF.type, SOSA.ActuatableProperty))


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
        reason = msg.get("message") or msg.get("type") or "unknown auth error"
        raise RuntimeError(f"Auth failed for HA_URL={url}: {reason}")
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

async def _event_forwarder_task():
    if not MONITOR_URL:
        print("Forwarder disabled: MONITOR_URL is not set")
        return  # forwarding disabled
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
                    new      = data.get("new_state") or {}
                    state    = new.get("state")
                    attrs    = new.get("attributes", {})
                    tstamp   = ev.get("time_fired")
                    if not entity_id or state in (None, "unknown", "unavailable"):
                        continue
                    area_id = ent_to_area.get(entity_id)
                    if not area_id or (AREAS and area_id not in AREAS):
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
                    try:
                        #print("Forwarder posting to", MONITOR_URL, "payload:", payload)
                        r = await http.post(
                            MONITOR_URL,
                            json=payload,
                            headers={
                                "X-Notification-Type": "ArtifactObsPropertyUpdated",
                                "Content-Type": "application/json",
                            },
                        )
                        r.raise_for_status()
                    except Exception as e:
                        print(f"Forwarding failed for {entity_id}: {e}")
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

async def _post_with_retries(url: str, what: str, max_retries: int = 5):
    delays = [0, 1, 2, 4, 8]
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for attempt in range(max_retries):
            try:
                if delays[attempt]:
                    await asyncio.sleep(delays[attempt])
                print(f"POST {what} attempt {attempt+1}/{max_retries} → {url}")
                r = await client.post(url)
                print(f"{what} status: {r.status_code} body: {r.text[:200]}")
                r.raise_for_status()
                return True
            except Exception as e:
                print(f"{what} failed on attempt {attempt+1}: {e}")
        return False

async def _post_monitor_reset():
    if not MONITOR_URL:
        return
    reset_url = MONITOR_URL if MONITOR_URL.rstrip('/').endswith('/reset') else MONITOR_URL.rstrip('/') + '/reset'
    await _post_with_retries(reset_url, "monitor reset")

async def _post_explorer_reset():
    if not EXPLORER_URL:
        return
    reset_url = EXPLORER_URL if EXPLORER_URL.rstrip('/').endswith('/admin/reset') else EXPLORER_URL.rstrip('/') + '/admin/reset'
    await _post_with_retries(reset_url, "explorer reset")

async def _register_known_artifacts_to_monitor():
    """On startup, send current known artifact property values to the monitor.
    Filters by AREAS; uses same payload shape and headers as the forwarder.
    """
    if not MONITOR_URL or not AREAS:
        return
    try:
        ent_to_area, ent_to_device, dev_by_id, ent_by_id = await _build_entity_area_map()
        states = await ha_rest.get_states()
        async with httpx.AsyncClient(timeout=10.0) as client:
            for st in states:
                entity_id = st.get("entity_id")
                area_id = ent_to_area.get(entity_id)
                if not area_id or area_id not in AREAS:
                    continue
                state = st.get("state")
                if state in (None, "unknown", "unavailable"):
                    continue
                attrs = st.get("attributes", {}) or {}
                entity_meta = ent_by_id.get(entity_id)
                artifact_label = _entity_display_name(entity_meta, dev_by_id)
                artifact_name = urllib.parse.quote(artifact_label, safe="")
                prop = attrs.get("device_class") or "state"
                value, xtype = _infer_value_and_type(state)
                artifact_profile = f"{BASE_WS_URI.rstrip('/')}/workspaces/{area_id}/artifacts/{artifact_name}"
                artifact_uri = f"{artifact_profile}#artifact"
                property_uri = f"{artifact_profile}/props/{prop}"
                trigger_uri = f"{artifact_profile}/actions/read"
                payload = {
                    "artifactUri": artifact_uri,
                    "propertyUri": property_uri,
                    "value": value,
                    "valueTypeUri": xtype,
                    "timestamp": st.get("last_changed") or st.get("last_updated") or "",
                    "triggerUri": trigger_uri,
                }
                try:
                    print("Initial monitor register posting to", MONITOR_URL, "payload:", payload)
                    r = await client.post(
                        MONITOR_URL,
                        json=payload,
                        headers={
                            "X-Notification-Type": "ArtifactObsPropertyUpdated",
                            "Content-Type": "application/json",
                        },
                    )
                    r.raise_for_status()
                except Exception as e:
                    print("Initial monitor register failed for", entity_id, "error:", e)
    except Exception as e:
        print("Initial monitor registration failed:", e)

@app.on_event("startup")
async def _startup_forwarder():
    print(
        f"App startup: MONITOR_URL={'set' if MONITOR_URL else 'unset'}, "
        f"EXPLORER_URL={'set' if EXPLORER_URL else 'unset'}, "
        f"AREAS={sorted(AREAS) if AREAS else 'ALL'}, BASE_WS_URI={BASE_WS_URI}, "
        f"TD_SOSA_ENV_VAR_OVERRIDES={len(TD_SOSA_ENV_VAR_OVERRIDES)}"
    )
    # Fire-and-forget reset
    asyncio.create_task(_post_monitor_reset())
    asyncio.create_task(_post_explorer_reset())
    asyncio.create_task(_register_known_artifacts_to_monitor())
    # Fire-and-forget registration for requested areas
    if EXPLORER_URL and AREAS:
        for area_id in AREAS:
            asyncio.create_task(_register_workspace_to_explorer(area_id))
    if MONITOR_URL:
        app.state.forward_task = asyncio.create_task(_event_forwarder_task())
        print("Forwarder task scheduled")
    else:
        print("Forwarder not scheduled: MONITOR_URL is unset")

# Simple status endpoint for debugging forwarder
@app.get("/_forwarder/status")
async def forwarder_status():
    task = getattr(app.state, "forward_task", None)
    return {
        "enabled": bool(MONITOR_URL),
        "areas": sorted(AREAS) if AREAS else [],
        "baseWsUri": BASE_WS_URI,
        "taskRunning": bool(task) and not task.done(),
        "monitorUrl": MONITOR_URL,
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
        artifact_foi = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/environment#foi")

        for prefix, ns in {
            "sosa": SOSA,
            "ssn": SSN,
            "qudt": QUDT,
            "unit": UNIT,
            "tdsosa": TDSOSA,
        }.items():
            rdf.g.bind(prefix, ns)

        # Build RDF
        rdf.g.add((art, RDF.type, TD.Thing))
        rdf.g.add((art, RDF.type, HMAS.Artifact))
        rdf.g.add((art, TD.title, Literal(artifact_label)))
        rdf.g.add((artifact_foi, RDF.type, SOSA.FeatureOfInterest))
        rdf.g.add((artifact_foi, TD.title, Literal(f"{artifact_label} environment")))
        domains = {e["entity_id"].split(".")[0] for e in device_entities}
        if "light" in domains:
            rdf.g.add((art, RDF.type, EX.HueLamp))
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
            domain_entity_meta = next((e for e in device_entities if e.get("entity_id") == domain_entity), None)
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

                action_affordance = rdf._add_action(
                    art,
                    action_name,
                    EX.StatusCommand,
                    "POST",
                    URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/ha/{urllib.parse.quote(domain, safe='')}/{urllib.parse.quote(svc_name, safe='')}"),
                    "application/json",
                    input_schema=input_schema,
                    description=service_description,
                )
                env_var_key = _resolve_env_var_override(
                    entity=domain_entity_meta,
                    device=device,
                    artifact_label=artifact_label,
                    domain=domain,
                    service_name=svc_name,
                ) or _ambient_var_for_action(domain, svc_name)
                direction = _effect_direction(svc_name)
                if action_affordance and env_var_key and direction:
                    env_var_uri = _env_var_uri(rdf.base, aid, env_var_key)
                    actuation_uri = URIRef(
                        f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/actuations/"
                        f"{urllib.parse.quote(domain, safe='')}_{urllib.parse.quote(svc_name, safe='')}_{direction}"
                    )
                    _add_tdsosa_action_effect(
                        rdf=rdf,
                        action_affordance=action_affordance,
                        feature_of_interest=artifact_foi,
                        env_var_uri=env_var_uri,
                        actuation_uri=actuation_uri,
                        direction=direction,
                    )

        # Sensor-specific value action
        if "sensor" in domains:
            sensor_ent = _pick_entity(device_entities, "sensor")
            st = state_map.get(sensor_ent, {}) if sensor_ent else {}
            attrs = st.get("attributes", {}) if isinstance(st, dict) else {}
            device_class = attrs.get("device_class")
            unit = attrs.get("unit_of_measurement")
            action_names = _sensor_action_names(device_class, unit)
            if action_names:
                action_name = action_names[0]

                # Build output schema
                output_schema = rdf._build_sensor_output_schema(device_class, unit)

                rdf._add_action(
                    art,
                    action_name,
                    EX.StatusCommand,
                    "POST",
                    URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/{urllib.parse.quote(action_name, safe='')}"),
                    "application/json",
                    output_schema=output_schema,
                )

        if "binary_sensor" in domains:
            binary_ent = _pick_entity(device_entities, "binary_sensor")
            st = state_map.get(binary_ent, {}) if binary_ent else {}
            attrs = st.get("attributes", {}) if isinstance(st, dict) else {}
            action_names = _binary_sensor_action_names(attrs.get("device_class"))
            if action_names:
                action_name = action_names[0]

                # Build output schema
                output_schema = rdf._build_binary_sensor_output_schema()

                rdf._add_action(
                    art,
                    action_name,
                    EX.StatusCommand,
                    "POST",
                    URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/{urllib.parse.quote(action_name, safe='')}"),
                    "application/json",
                    output_schema=output_schema,
                )

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
                state_prop = rdf._add_property(
                    art,
                    "state",
                    property_uri,
                    output_schema=schema,
                    description=f"Current state of {entity_id}",
                    observable=True
                )
                state_env_key = _resolve_env_var_override(
                    entity=entity,
                    device=device,
                    artifact_label=artifact_label,
                    domain=entity_domain,
                    signal_name="state",
                ) or _ambient_var_from_signals(entity_domain, entity_attrs.get("device_class"), "state")
                if state_prop and state_env_key:
                    env_var_uri = _env_var_uri(rdf.base, aid, state_env_key)
                    _add_tdsosa_property_links(
                        rdf=rdf,
                        property_affordance=state_prop,
                        feature_of_interest=artifact_foi,
                        env_var_uri=env_var_uri,
                        observable=_domain_is_observer(entity_domain),
                        actuatable=_domain_is_actuator(entity_domain),
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
                    op_prop = rdf._add_property(
                        art,
                        attr_name,
                        property_uri,
                        output_schema=schema,
                        description=f"{attr_name} of {entity_id}",
                        observable=True
                    )
                    env_var_key = _resolve_env_var_override(
                        entity=entity,
                        device=device,
                        artifact_label=artifact_label,
                        domain=domain,
                        signal_name=attr_name,
                    ) or _ambient_var_from_signals(domain, entity_attrs.get("device_class"), attr_name)
                    if op_prop and env_var_key:
                        env_var_uri = _env_var_uri(rdf.base, aid, env_var_key)
                        _add_tdsosa_property_links(
                            rdf=rdf,
                            property_affordance=op_prop,
                            feature_of_interest=artifact_foi,
                            env_var_uri=env_var_uri,
                            observable=_domain_is_observer(domain),
                            actuatable=_domain_is_actuator(domain),
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

# -------- Utilities (Explorer registration) --------
async def _register_workspace_to_explorer(area_id: str):
    if not EXPLORER_URL:
        return
    try:
        devices, entities = await _get_workspace_devices_and_entities(area_id)
        base = BASE_WS_URI.rstrip("/")
        artifact_uris = []
        device_map = {d["id"]: d for d in devices}
        for ent in entities:
            label = ent.get("_artifact_label") or _entity_display_name(ent, device_map)
            safe_name = ent.get("_artifact_slug") or urllib.parse.quote(label, safe="")
            artifact_uris.append(f"{base}/workspaces/{area_id}/artifacts/{safe_name}#artifact")
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            for uri in artifact_uris:
                try:
                    print("Explorer register: sending", uri, "to", EXPLORER_URL)
                    r = await client.post(EXPLORER_URL, json={"uri": uri}, headers={"Content-Type": "application/json"})
                    r.raise_for_status()
                except Exception as e:
                    print("Explorer register failed for", uri, "error:", e)
    except Exception as exc:
        print("Explorer registration failed for area", area_id, "error:", exc)

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
@app.post("/workspaces/{workspace_id}/focus")
async def focus_workspace(workspace_id: str, request: Request):
    _ = await request.json()
    return Response(content="Action succeeded:")

@app.post("/hub/")
async def hub(request: Request):
    _ = await request.json()
    return Response(content="Action succeeded:")

# PUT/DELETE artifact representation stubs
@app.put("/workspaces/{workspace_id}/artifacts/{artifact_name}")
async def update_artifact_representation(workspace_id: str, artifact_name: str, request: Request):
    _ = await request.json()
    return Response(content="Action succeeded:")

@app.delete("/workspaces/{workspace_id}/artifacts/{artifact_name}")
async def delete_artifact_representation(workspace_id: str, artifact_name: str):
    return Response(content="Action succeeded:")

# Dynamic sensor/binary sensor actions
@app.post("/workspaces/{workspace_id}/artifacts/{artifact_name}/{action_name}")
async def action_sensor_dynamic(workspace_id: str, artifact_name: str, action_name: str):
    _, device_entities, _, _ = await _resolve_device_and_entities(workspace_id, artifact_name)
    sensor_ent = _pick_entity(device_entities, "sensor")
    binary_ent = _pick_entity(device_entities, "binary_sensor")
    climate_ent = _pick_entity(device_entities, "climate")

    states = await ha_rest.get_states()
    state_map = {s.get("entity_id"): s for s in states}
    sensor_state = state_map.get(sensor_ent) if sensor_ent else None
    binary_state = state_map.get(binary_ent) if binary_ent else None
    climate_state = state_map.get(climate_ent) if climate_ent else None

    sensor_names = _sensor_action_names(
        (sensor_state or {}).get("attributes", {}).get("device_class"),
        (sensor_state or {}).get("attributes", {}).get("unit_of_measurement"),
    ) if sensor_state else []
    binary_names = _binary_sensor_action_names(
        (binary_state or {}).get("attributes", {}).get("device_class")
    ) if binary_state else []

    if action_name in sensor_names:
        return PlainTextResponse(str((sensor_state or {}).get("state", "")))
    if action_name in binary_names:
        return PlainTextResponse(str((binary_state or {}).get("state", "")))

    if action_name == "getThermostatState":
        if not climate_ent:
            raise HTTPException(status_code=404, detail="No climate entity on artifact")
        if not climate_state:
            raise HTTPException(status_code=404, detail="Climate state not found")
        return JSONResponse(_format_climate_state(climate_state))

    # Provide meaningful errors for known patterns
    if action_name.startswith("get") and "In" in action_name[3:]:
        if not sensor_ent:
            raise HTTPException(status_code=404, detail="No sensor entity on artifact")
        if not sensor_state:
            raise HTTPException(status_code=404, detail="Sensor state not found")
        raise HTTPException(status_code=404, detail="Action not applicable to this sensor")

    binary_like = action_name.startswith("get") and action_name.endswith("State")
    if binary_like:
        if not binary_ent:
            raise HTTPException(status_code=404, detail="No binary sensor entity on artifact")
        if not binary_state:
            raise HTTPException(status_code=404, detail="Binary sensor state not found")
        raise HTTPException(status_code=404, detail="Action not applicable to this binary sensor")

    raise HTTPException(status_code=404, detail="Unknown action")
