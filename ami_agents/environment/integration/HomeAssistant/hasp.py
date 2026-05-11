#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import asyncio
import contextlib
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from rdflib import BNode, Graph, Literal, Namespace, RDF, URIRef

from http import HTTPStatus
from hasp_utils import (HomeAssistantWS, HomeAssistantRDF, HomeAssistantREST,
                        get_supported_service_fields, is_service_supported_for_entity)
from hasp_cache import HASPGraphCache

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

WEBHOOK_VERIFY_TIMEOUT = 5.0

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

AREAS = {a.strip() for a in os.getenv("AREAS", "").split(",") if a.strip()}  # allowed area_ids
BASE_WS_URI = os.getenv("BASE_WS_URI", BASE_FALLBACK)  # e.g., https://example.org/ws/lab
GRAPH_SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "graph_snapshots")
subscriptions: Dict[str, Dict[str, Any]] = {}


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

app = FastAPI(title="HASP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ha_client = HomeAssistantWS(HA_URL, HA_TOKEN)
ha_rest = HomeAssistantREST(HA_BASE_URL, HA_TOKEN)


def _get_graph_cache() -> HASPGraphCache:
    cache = getattr(app.state, "graph_cache", None)
    if cache is None:
        raise RuntimeError("Graph cache not initialized")
    return cache


async def _ensure_graph_cache() -> HASPGraphCache:
    cache = getattr(app.state, "graph_cache", None)
    expected_base = BASE_WS_URI.rstrip("/") + "/"
    expected_areas = set(AREAS)
    cache_stale = (
        cache is None
        or getattr(cache, "ws_client", None) is not ha_client
        or getattr(cache, "rest_client", None) is not ha_rest
        or getattr(cache, "base_uri", None) != expected_base
        or getattr(cache, "allowed_workspaces", None) != expected_areas
    )
    if cache_stale:
        cache = HASPGraphCache(
            ws_client=ha_client,
            rest_client=ha_rest,
            base_uri=expected_base,
            allowed_workspaces=expected_areas,
            artifact_builder=_build_cached_artifact_ttl,
        )
        app.state.graph_cache = cache
        await cache.refresh()
    return cache


def _ttl_for_request(ttl: str, request: Request, cache: HASPGraphCache) -> str:
    request_base = str(request.base_url)
    cache_base = getattr(cache, "base_uri", BASE_WS_URI.rstrip("/") + "/")
    if cache_base == request_base:
        return ttl
    normalized_cache_base = cache_base.rstrip("/")
    normalized_request_base = request_base.rstrip("/")
    rewritten = ttl.replace(cache_base, request_base)
    if normalized_cache_base != normalized_request_base:
        rewritten = rewritten.replace(normalized_cache_base, normalized_request_base)
    return rewritten


def _sanitize_service_payload(
    payload: Dict[str, Any],
    service_definition: Optional[Dict[str, Any]],
    *,
    entity_id: str,
) -> Tuple[Dict[str, Any], List[str]]:
    fields = (service_definition or {}).get("fields") or {}
    allowed_fields = {key for key in fields if isinstance(key, str)}
    sanitized = {key: value for key, value in payload.items() if key in allowed_fields}
    sanitized["entity_id"] = entity_id
    dropped = sorted(
        key for key in payload
        if key != "entity_id" and key not in allowed_fields
    )
    return sanitized, dropped

@app.on_event("shutdown")
async def _shutdown():
    task = getattr(app.state, "sync_task", None)
    if task:
        print("Shutting down HA state sync task...")
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


def _artifact_slug_from_entity(entity: Dict[str, Any], fallback_label: str) -> str:
    entity_id = entity.get("entity_id", "")
    if isinstance(entity_id, str) and "." in entity_id:
        object_id = entity_id.split(".", 1)[1].strip()
        if object_id:
            return object_id
    return fallback_label


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


_ORIGINAL_RESOLVE_DEVICE_AND_ENTITIES = _resolve_device_and_entities


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
    slug_counts: Dict[str, int] = {}
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
        base_slug = _artifact_slug_from_entity(ent, label)
        slug_counts[base_slug] = slug_counts.get(base_slug, 0) + 1
        if slug_counts[base_slug] > 1:
            entity_id = ent.get("entity_id", "")
            if isinstance(entity_id, str) and entity_id:
                base_slug = entity_id.replace(".", "_")
            else:
                base_slug = label
        ent["_artifact_slug"] = urllib.parse.quote(base_slug, safe="")
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
        cache = await _ensure_graph_cache()
        ttl = await cache.get_platform_ttl()
        return Response(_ttl_for_request(ttl, request, cache), media_type="text/turtle")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/workspaces", response_class=Response,
         responses={200: {"content": {"text/turtle": {}}}})
async def list_workspaces(request: Request):
    try:
        cache = await _ensure_graph_cache()
        ttl = await cache.get_workspaces_ttl()
        return Response(_ttl_for_request(ttl, request, cache), media_type="text/turtle")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

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


def _workspace_topic(workspace_id: str) -> str:
    return f"{BASE_WS_URI.rstrip('/')}/workspaces/{workspace_id}"


def _artifact_topic(workspace_id: str, artifact_slug: str) -> str:
    return f"{BASE_WS_URI.rstrip('/')}/workspaces/{workspace_id}/artifacts/{artifact_slug}#artifact"


def _subscription_id(topic: str, callback_url: str) -> str:
    return f"{topic}-{callback_url}"


def _subscription_is_active(subscription: Dict[str, Any]) -> bool:
    lease_seconds = subscription.get("lease_seconds")
    timestamp = subscription.get("timestamp")
    if not lease_seconds:
        return True
    if not isinstance(timestamp, (int, float)):
        return False
    return (asyncio.get_event_loop().time() - timestamp) < lease_seconds


def _prune_expired_subscriptions() -> None:
    expired = [
        sub_id for sub_id, sub in subscriptions.items()
        if not _subscription_is_active(sub)
    ]
    for sub_id in expired:
        subscriptions.pop(sub_id, None)


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


async def _verify_callback_intent(
    callback_url: str,
    *,
    mode: str,
    topic: str,
    lease_seconds: Optional[int] = None,
) -> None:
    challenge = str(uuid.uuid4())
    verify_params = {
        "hub.mode": mode,
        "hub.topic": topic,
        "hub.callback": callback_url,
        "hub.challenge": challenge,
    }
    if lease_seconds:
        verify_params["hub.lease_seconds"] = str(lease_seconds)
    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_VERIFY_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(callback_url, params=verify_params)
            response.raise_for_status()
            if response.text != challenge:
                raise HTTPException(status_code=409, detail="Intent verification challenge response mismatch")
    except HTTPException:
        raise
    except httpx.RequestError as exc:
        raise HTTPException(status_code=412, detail=f"Intent verification failed: Network error connecting to callback URL: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=412, detail=f"Intent verification failed: HTTP error from callback URL: {exc}") from exc


async def _register_subscription(
    *,
    topic: str,
    callback_url: str,
    lease_seconds: Optional[int],
    subscription_type: str,
    secret: Optional[str] = None,
) -> str:
    await _verify_callback_intent(
        callback_url,
        mode="subscribe" if subscription_type != "unsubscribe" else "unsubscribe",
        topic=topic,
        lease_seconds=lease_seconds,
    )
    subscription_id = _subscription_id(topic, callback_url)
    subscriptions[subscription_id] = {
        "topic": topic,
        "callback": callback_url,
        "lease_seconds": lease_seconds,
        "secret": secret,
        "timestamp": asyncio.get_event_loop().time(),
        "type": subscription_type,
    }
    return subscription_id


def _extract_hub_request(payload: Dict[str, Any]) -> Tuple[str, str, str, Optional[int], Optional[str]]:
    hub_mode = payload.get("hub.mode") or payload.get("mode")
    hub_topic = payload.get("hub.topic") or payload.get("topic")
    hub_callback = payload.get("hub.callback") or payload.get("callback") or payload.get("callbackUrl")
    raw_lease = payload.get("hub.lease_seconds") or payload.get("lease_seconds")
    hub_secret = payload.get("hub.secret") or payload.get("secret")
    lease_seconds = None
    if raw_lease not in (None, ""):
        try:
            lease_seconds = int(raw_lease)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="hub.lease_seconds must be an integer") from exc
    if not all([hub_mode, hub_topic, hub_callback]):
        raise HTTPException(
            status_code=400,
            detail="Missing required parameters: hub.mode, hub.topic, hub.callback",
        )
    return str(hub_mode), str(hub_topic), str(hub_callback), lease_seconds, hub_secret


async def _iter_notification_targets(entity_id: str) -> List[Dict[str, Any]]:
    cache = _get_graph_cache()
    contexts = await cache.get_entity_contexts(entity_id)
    if not contexts:
        return []
    _prune_expired_subscriptions()
    notifications: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for context in contexts:
        for topic in (context["workspace_topic"], context["artifact_topic"]):
            for sub in subscriptions.values():
                if sub.get("topic") != topic or not _subscription_is_active(sub):
                    continue
                key = (sub["callback"], topic)
                if key in seen:
                    continue
                seen.add(key)
                notifications.append(
                    {
                        "callback": sub["callback"],
                        "topic": topic,
                        "workspace_id": context["workspace_id"],
                        "artifact_uri": context["artifact_uri"],
                        "artifact_title": context["artifact_title"],
                    }
                )
    return notifications


async def _notify_subscribers_for_entity(
    entity_id: str,
    new_state: Optional[Dict[str, Any]],
    timestamp: Optional[str],
) -> None:
    notifications = await _iter_notification_targets(entity_id)
    if not notifications:
        return
    payload = {
        "entityId": entity_id,
        "timestamp": timestamp or "",
        "state": (new_state or {}).get("state"),
        "attributes": (new_state or {}).get("attributes", {}),
    }
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for target in notifications:
            body = {
                **payload,
                "topic": target["topic"],
                "workspaceId": target["workspace_id"],
                "artifactUri": target["artifact_uri"],
                "artifactTitle": target["artifact_title"],
            }
            try:
                response = await client.post(
                    target["callback"],
                    json=body,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
            except Exception as exc:
                print(f"Notification failed for {entity_id} -> {target['callback']}: {exc}")


async def _process_state_changed_event(event: Dict[str, Any]) -> None:
    data = event.get("data", {}) if isinstance(event, dict) else {}
    entity_id = data.get("entity_id")
    if not entity_id:
        return
    new_state = data.get("new_state") or None
    state_value = (new_state or {}).get("state")
    cache = getattr(app.state, "graph_cache", None)
    if cache is None:
        return
    if state_value in (None, "unknown", "unavailable"):
        await cache.apply_state_change(entity_id, None)
        await _notify_subscribers_for_entity(entity_id, None, event.get("time_fired"))
        return
    await cache.apply_state_change(entity_id, new_state)
    await _notify_subscribers_for_entity(entity_id, new_state, event.get("time_fired"))


async def _ha_state_sync_task() -> None:
    print("Starting HA state sync task")
    while True:
        ws = None
        try:
            ws = await _ws_handshake(HA_URL, HA_TOKEN)
            await ws.send(json.dumps({"id": 100, "type": "subscribe_events", "event_type": "state_changed"}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") != "event":
                    continue
                event = msg.get("event", {})
                if event.get("event_type") != "state_changed":
                    continue
                await _process_state_changed_event(event)
        except asyncio.CancelledError:
            print("HA state sync task cancelled")
            break
        except Exception as exc:
            print(f"HA state sync error: {exc}; reconnecting in 3s...")
            await asyncio.sleep(3)
        finally:
            if ws is not None:
                with contextlib.suppress(Exception):
                    await ws.close()


def _build_cached_artifact_ttl(
    workspace_id: str,
    safe_name: str,
    artifact_label: str,
    device_entities: List[Dict[str, Any]],
    state_map: Dict[str, Dict[str, Any]],
    device: Dict[str, Any],
) -> str:
    rdf = HomeAssistantRDF(BASE_WS_URI)
    aid = workspace_id
    ws = URIRef(f"{rdf.base}workspaces/{aid}#workspace")
    art_dir = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/")
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

    rdf.g.add((art, RDF.type, TD.Thing))
    rdf.g.add((art, RDF.type, HMAS.Artifact))
    rdf.g.add((art, TD.title, Literal(artifact_label)))
    rdf.g.add((artifact_foi, RDF.type, SOSA.FeatureOfInterest))
    rdf.g.add((artifact_foi, TD.title, Literal(f"{artifact_label} environment")))
    domains = {e["entity_id"].split(".")[0] for e in device_entities if e.get("entity_id")}
    if "light" in domains:
        rdf.g.add((art, RDF.type, EX.HueLamp))
    sec = BNode()
    rdf.g.add((art, TD.hasSecurityConfiguration, sec))
    rdf.g.add((sec, RDF.type, WOTSEC.NoSecurityScheme))

    svc_by_domain: Dict[str, Dict[str, Dict[str, Any]]] = {}
    graph_cache = getattr(app.state, "graph_cache", None)
    if graph_cache is not None:
        svc_by_domain = {
            dom: (payload.get("services", {}) or {})
            for dom, payload in graph_cache.services_by_domain.items()
        }

    for domain in sorted(domains):
        domain_svcs = svc_by_domain.get(domain, {})
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
            if not is_service_supported_for_entity(domain, svc_name, domain_entity_attrs):
                continue
            action_name = f"{_camel_token(domain)}{_camel_token(svc_name)}"
            all_service_fields = definition.get("fields", {})
            supported_fields = get_supported_service_fields(domain, domain_entity_attrs, all_service_fields)
            # Modern HA services may target entities without requiring any
            # explicit input fields. Those should still be exposed as actions.
            if not supported_fields and not (legacy_applies or modern_applies):
                continue
            input_schema = rdf._build_input_schema_from_fields(supported_fields)
            action_affordance = rdf._add_action(
                art,
                action_name,
                EX.StatusCommand,
                "POST",
                URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/ha/{urllib.parse.quote(domain, safe='')}/{urllib.parse.quote(svc_name, safe='')}"),
                "application/json",
                input_schema=input_schema,
                description=definition.get("description"),
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

    if "sensor" in domains:
        sensor_ent = _pick_entity(device_entities, "sensor")
        st = state_map.get(sensor_ent, {}) if sensor_ent else {}
        attrs = st.get("attributes", {}) if isinstance(st, dict) else {}
        action_names = _sensor_action_names(attrs.get("device_class"), attrs.get("unit_of_measurement"))
        if action_names:
            action_name = action_names[0]
            rdf._add_action(
                art,
                action_name,
                EX.StatusCommand,
                "POST",
                URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/{urllib.parse.quote(action_name, safe='')}"),
                "application/json",
                output_schema=rdf._build_sensor_output_schema(attrs.get("device_class"), attrs.get("unit_of_measurement")),
            )

    if "binary_sensor" in domains:
        binary_ent = _pick_entity(device_entities, "binary_sensor")
        st = state_map.get(binary_ent, {}) if binary_ent else {}
        attrs = st.get("attributes", {}) if isinstance(st, dict) else {}
        action_names = _binary_sensor_action_names(attrs.get("device_class"))
        if action_names:
            rdf._add_action(
                art,
                action_names[0],
                EX.StatusCommand,
                "POST",
                URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/{urllib.parse.quote(action_names[0], safe='')}"),
                "application/json",
                output_schema=rdf._build_binary_sensor_output_schema(),
            )

    if "climate" in domains:
        climate_ent = _pick_entity(device_entities, "climate")
        if climate_ent and climate_ent in state_map:
            rdf._add_action(
                art,
                "getThermostatState",
                EX.StatusCommand,
                "POST",
                URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/getThermostatState"),
                "application/json",
                output_schema=rdf._build_climate_output_schema(),
            )

    for entity in device_entities:
        entity_id = entity.get("entity_id")
        if not entity_id:
            continue
        entity_state = state_map.get(entity_id, {})
        if not entity_state:
            continue
        entity_attrs = entity_state.get("attributes", {}) if isinstance(entity_state, dict) else {}
        state_value = entity_state.get("state")
        if state_value and state_value not in ("unknown", "unavailable"):
            entity_domain = entity_id.split(".")[0] if "." in entity_id else ""
            unit = entity_attrs.get("unit_of_measurement")
            schema_value = state_value
            if (entity_domain == "sensor" or unit) and isinstance(state_value, str):
                try:
                    schema_value = int(state_value) if "." not in state_value else float(state_value)
                except ValueError:
                    pass
            property_uri = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/properties/state")
            schema = rdf._build_property_schema("state", schema_value, entity_attrs, entity_domain=entity_domain)
            state_prop = rdf._add_property(
                art, "state", property_uri, output_schema=schema,
                description=f"Current state of {entity_id}", observable=True
            )
            state_env_key = _resolve_env_var_override(
                entity=entity, device=device, artifact_label=artifact_label,
                domain=entity_domain, signal_name="state",
            ) or _ambient_var_from_signals(entity_domain, entity_attrs.get("device_class"), "state")
            if state_prop and state_env_key:
                env_var_uri = _env_var_uri(rdf.base, aid, state_env_key)
                _add_tdsosa_property_links(
                    rdf=rdf, property_affordance=state_prop, feature_of_interest=artifact_foi,
                    env_var_uri=env_var_uri, observable=_domain_is_observer(entity_domain),
                    actuatable=_domain_is_actuator(entity_domain),
                )

        if entity_attrs:
            from hasp_utils import get_operational_attributes, get_metadata_attributes
            domain = entity_id.split(".")[0] if "." in entity_id else ""
            operational_attrs = get_operational_attributes(domain, entity_attrs)
            for attr_name, attr_value in operational_attrs.items():
                if attr_value is None:
                    continue
                property_uri = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/properties/{urllib.parse.quote(attr_name, safe='')}")
                schema = rdf._build_property_schema(attr_name, attr_value, entity_attrs)
                op_prop = rdf._add_property(
                    art, attr_name, property_uri, output_schema=schema,
                    description=f"{attr_name} of {entity_id}", observable=True
                )
                env_var_key = _resolve_env_var_override(
                    entity=entity, device=device, artifact_label=artifact_label,
                    domain=domain, signal_name=attr_name,
                ) or _ambient_var_from_signals(domain, entity_attrs.get("device_class"), attr_name)
                if op_prop and env_var_key:
                    env_var_uri = _env_var_uri(rdf.base, aid, env_var_key)
                    _add_tdsosa_property_links(
                        rdf=rdf, property_affordance=op_prop, feature_of_interest=artifact_foi,
                        env_var_uri=env_var_uri, observable=_domain_is_observer(domain),
                        actuatable=_domain_is_actuator(domain),
                    )

            metadata_attrs = get_metadata_attributes(entity_attrs)
            for ts_field in ("last_changed", "last_reported", "last_updated"):
                if ts_field in entity_state:
                    metadata_attrs[ts_field] = entity_state[ts_field]
            if metadata_attrs:
                metadata_property_uri = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/properties/metadata")
                metadata_schema = rdf._build_metadata_schema(metadata_attrs)
                rdf._add_property(
                    art, "metadata", metadata_property_uri, output_schema=metadata_schema,
                    description=f"Metadata attributes of {entity_id}", observable=False
                )

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
    rdf.g.add((art, HMAS.isContainedIn, ws))
    rdf.g.add((ws, RDF.type, HMAS.Workspace))
    profile = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}")
    rdf.g.add((profile, RDF.type, HMAS.ResourceProfile))
    rdf.g.add((profile, HMAS.isProfileOf, art))
    return rdf.serialize()

@app.on_event("startup")
async def _startup_cache():
    print(
        f"AREAS={sorted(AREAS) if AREAS else 'ALL'}, BASE_WS_URI={BASE_WS_URI}, "
        f"TD_SOSA_ENV_VAR_OVERRIDES={len(TD_SOSA_ENV_VAR_OVERRIDES)}"
    )
    app.state.graph_cache = HASPGraphCache(
        ws_client=ha_client,
        rest_client=ha_rest,
        base_uri=BASE_WS_URI,
        allowed_workspaces=AREAS,
        artifact_builder=_build_cached_artifact_ttl,
    )
    await app.state.graph_cache.refresh()
    app.state.sync_task = asyncio.create_task(_ha_state_sync_task())


@app.post("/_graph/snapshot")
async def save_graph_snapshot(request: Request):
    payload = await request.json()
    filename = payload.get("filename") if isinstance(payload, dict) else None
    requested_format = payload.get("format") if isinstance(payload, dict) else None
    if not isinstance(filename, str) or not filename.strip():
        raise HTTPException(status_code=400, detail="filename is required")
    filename = filename.strip()
    if filename != os.path.basename(filename) or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="filename must not contain path components")

    format_aliases = {
        None: "turtle",
        "ttl": "turtle",
        "turtle": "turtle",
        "xml": "xml",
        "rdf/xml": "xml",
        "rdfxml": "xml",
    }
    normalized_format = None
    if requested_format is not None:
        if not isinstance(requested_format, str) or not requested_format.strip():
            raise HTTPException(status_code=400, detail="format must be a non-empty string")
        normalized_format = requested_format.strip().lower()
    rdf_format = format_aliases.get(normalized_format)
    if rdf_format is None:
        raise HTTPException(status_code=400, detail="unsupported format")

    cache = await _ensure_graph_cache()
    os.makedirs(GRAPH_SNAPSHOT_DIR, exist_ok=True)
    output_path = os.path.join(GRAPH_SNAPSHOT_DIR, filename)
    graph_ttl = await cache.serialize_full_graph(rdf_format)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(graph_ttl)
    return {"path": output_path, "filename": filename, "format": rdf_format}


@app.post("/_graph/query/actions-affecting-observable-property")
async def query_actions_affecting_observable_property(request: Request):
    payload = await request.json()
    workspace_id = payload.get("workspace_id") if isinstance(payload, dict) else None
    observable_property = payload.get("observable_property") if isinstance(payload, dict) else None
    property_uri = payload.get("property_uri") if isinstance(payload, dict) else None

    if not isinstance(workspace_id, str) or not workspace_id.strip():
        raise HTTPException(status_code=400, detail="workspace_id is required")
    workspace_id = workspace_id.strip()
    if not isinstance(observable_property, str) or not observable_property.strip():
        if not isinstance(property_uri, str) or not property_uri.strip():
            raise HTTPException(status_code=400, detail="observable_property or property_uri is required")
        observable_property = property_uri.strip()
    else:
        observable_property = observable_property.strip()

    cache = await _ensure_graph_cache()
    if not await cache.has_workspace(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")
    matches = await cache.query_actions_affecting_observable_property(workspace_id, observable_property)
    resolved_property_uri = (
        observable_property
        if observable_property.startswith(("http://", "https://"))
        else f"{cache.base_uri}workspaces/{workspace_id}/environment/{urllib.parse.quote(observable_property, safe='')}"
    )
    return {
        "workspace_id": workspace_id,
        "property_uri": resolved_property_uri,
        "actions": matches,
    }

@app.get("/workspaces/{workspace_id}", response_class=Response,
         responses={200: {"content": {"text/turtle": {}}}, 404: {"description": "Not found"}})
async def workspace(workspace_id: str, request: Request):
    try:
        cache = await _ensure_graph_cache()
        if not await cache.has_workspace(workspace_id):
            raise HTTPException(status_code=404, detail="Workspace not found")
        ttl = await cache.get_workspace_ttl(workspace_id)
        return Response(_ttl_for_request(ttl, request, cache), media_type="text/turtle")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/workspaces/{workspace_id}/artifacts", response_class=Response,
         responses={200: {"content": {"text/turtle": {}}}, 404: {"description": "Not found"}})
async def list_artifacts(workspace_id: str, request: Request):
    try:
        cache = await _ensure_graph_cache()
        if not await cache.has_workspace(workspace_id):
            raise HTTPException(status_code=404, detail="Workspace not found")
        ttl = await cache.get_artifacts_ttl(workspace_id)
        return Response(_ttl_for_request(ttl, request, cache), media_type="text/turtle")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/workspaces/{workspace_id}/artifacts/{artifact_name}", response_class=Response,
         responses={200: {"content": {"text/turtle": {}, "application/json": {}}}, 404: {"description": "Not found"}})
async def get_artifact(workspace_id: str, artifact_name: str, request: Request):
    """Return RDF TD (Accept: text/turtle) or JSON snapshot (application/json) for an artifact."""
    try:
        if (
            getattr(app.state, "graph_cache", None) is None
            or _resolve_device_and_entities is not _ORIGINAL_RESOLVE_DEVICE_AND_ENTITIES
        ):
            await _resolve_device_and_entities(workspace_id, artifact_name)
        cache = await _ensure_graph_cache()
        ttl, _ = await cache.get_artifact_ttl(workspace_id, artifact_name)
        return Response(_ttl_for_request(ttl, request, cache), media_type="text/turtle")
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact not found")
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
        cache = await _ensure_graph_cache()
        return JSONResponse(await cache.read_property(workspace_id, artifact_name, property_name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Property '{urllib.parse.unquote(property_name)}' not found on artifact")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# Generic HA service forwarder for dynamically discovered actions
@app.post("/workspaces/{workspace_id}/artifacts/{artifact_name}/ha/{domain}/{service}")
async def action_ha_service(workspace_id: str, artifact_name: str, domain: str, service: str, request: Request):
    # Ensure entity for the requested domain exists on this artifact
    try:
        cache = await _ensure_graph_cache()
        ent = await cache.resolve_artifact_domain_entity(workspace_id, artifact_name, domain)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No {domain} entity on artifact")

    # Validate the service exists for the domain
    svc = await cache.get_service_definition(domain, service)
    if not svc:
        raise HTTPException(status_code=404, detail="Service not found for domain")
    ent_state = cache.states_by_entity_id.get(ent, {}) if hasattr(cache, "states_by_entity_id") else {}
    ent_attrs = ent_state.get("attributes", {}) if isinstance(ent_state, dict) else {}
    if not is_service_supported_for_entity(domain, service, ent_attrs):
        raise HTTPException(status_code=404, detail="Service not supported by this entity")

    payload = {}
    if request.headers.get("content-length") not in (None, "0"):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
    payload, dropped_fields = _sanitize_service_payload(payload, svc, entity_id=ent)
    if dropped_fields:
        print(
            f"HASP dropped unsupported service payload fields: domain={domain} service={service} "
            f"entity_id={ent} dropped={dropped_fields}"
        )

    print(
        f"HASP forwarding service call: workspace={workspace_id} artifact={artifact_name} "
        f"domain={domain} service={service} entity_id={ent} payload={payload}"
    )
    try:
        result = await ha_rest.call_service(domain, service, payload)
    except httpx.HTTPStatusError as exc:
        response_text = ""
        with contextlib.suppress(Exception):
            response_text = exc.response.text
        detail = {
            "error": "home_assistant_service_call_failed",
            "domain": domain,
            "service": service,
            "artifact_name": artifact_name,
            "entity_id": ent,
            "payload": payload,
            "home_assistant_status": exc.response.status_code if exc.response is not None else None,
            "home_assistant_response": response_text,
        }
        print(
            f"HASP service forward failed: domain={domain} service={service} "
            f"entity_id={ent} status={detail['home_assistant_status']} "
            f"response={response_text}"
        )
        raise HTTPException(status_code=502, detail=detail)
    await cache.apply_service_result(result)
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and item.get("entity_id"):
                await _notify_subscribers_for_entity(
                    item["entity_id"],
                    item,
                    item.get("last_updated") or item.get("last_changed"),
                )
    elif isinstance(result, dict) and result.get("entity_id"):
        await _notify_subscribers_for_entity(
            result["entity_id"],
            result,
            result.get("last_updated") or result.get("last_changed"),
        )
    return Response(content="Action succeeded:")

# Jacamo/WebSub
@app.post("/workspaces/{workspace_id}/focus")
async def focus_workspace(workspace_id: str, request: Request):
    body = await request.json()
    callback_url = body.get("callbackUrl") if isinstance(body, dict) else None
    artifact_name = body.get("artifactName") if isinstance(body, dict) else None
    if not callback_url:
        raise HTTPException(status_code=400, detail="callbackUrl is required")
    cache = await _ensure_graph_cache()
    if not await cache.has_workspace(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")
    topic = _workspace_topic(workspace_id)
    if artifact_name:
        contexts = await cache.get_artifact_contexts(workspace_id, artifact_name)
        if not contexts:
            raise HTTPException(status_code=404, detail="Artifact not found")
        topic = contexts[0]["artifact_topic"]
    await _register_subscription(
        topic=topic,
        callback_url=callback_url,
        lease_seconds=None,
        subscription_type="focus",
    )
    return Response(status_code=202, content="Focus succeeded")

@app.post("/hub/")
async def hub(request: Request):
    content_type = request.headers.get("content-type", "")
    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        body = dict(urllib.parse.parse_qsl((await request.body()).decode("utf-8")))
    else:
        body = await request.json()
    hub_mode, hub_topic, hub_callback, lease_seconds, hub_secret = _extract_hub_request(body)
    _prune_expired_subscriptions()
    if hub_mode == "subscribe":
        await _register_subscription(
            topic=hub_topic,
            callback_url=hub_callback,
            lease_seconds=lease_seconds,
            subscription_type="websub",
            secret=hub_secret,
        )
        return Response(status_code=202, content="Subscribed")
    if hub_mode == "unsubscribe":
        await _verify_callback_intent(
            hub_callback,
            mode="unsubscribe",
            topic=hub_topic,
            lease_seconds=lease_seconds,
        )
        removed = subscriptions.pop(_subscription_id(hub_topic, hub_callback), None)
        if removed is None:
            raise HTTPException(status_code=404, detail="No active subscription found for this topic and callback.")
        return Response(status_code=202, content="Unsubscribed")
    raise HTTPException(status_code=400, detail="Invalid hub.mode. Must be 'subscribe' or 'unsubscribe'.")

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
    cache = await _ensure_graph_cache()
    try:
        device_entities, state_map = await cache.get_artifact_snapshot(workspace_id, artifact_name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact not found")
    sensor_ent = _pick_entity(device_entities, "sensor")
    binary_ent = _pick_entity(device_entities, "binary_sensor")
    climate_ent = _pick_entity(device_entities, "climate")
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
