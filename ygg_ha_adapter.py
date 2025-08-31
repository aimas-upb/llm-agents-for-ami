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
from ha_utils import HomeAssistantWS, HomeAssistantRDF, HomeAssistantREST

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
MONITOR_URL = os.getenv("MONITOR_URL", os.getenv("FORWARD_URL", ""))  # destination to POST event JSON / reset
EXPLORER_URL = os.getenv("EXPLORER_URL", "")  # Environment Explorer base URL for admin reset
AREAS = {a.strip() for a in os.getenv("AREAS", "").split(",") if a.strip()}  # allowed area_ids
BASE_WS_URI = os.getenv("BASE_WS_URI", BASE_FALLBACK)  # e.g., https://example.org/ws/lab

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

async def _resolve_device_and_entities(workspace_id: str, artifact_name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    decoded_name = urllib.parse.unquote(artifact_name)
    devices = await ha_client.get_devices(workspace_id)
    device = next((d for d in devices if d.get("name") == decoded_name), None)
    if not device:
        raise HTTPException(status_code=404, detail="Artifact not found")
    entities = await ha_client.get_entities()
    device_entities = [e for e in entities if e.get("device_id") == device["id"]]
    if not device_entities:
        raise HTTPException(status_code=404, detail="No entities for artifact")
    return device, device_entities

def _pick_entity(device_entities: List[Dict[str, Any]], domain: str) -> Optional[str]:
    for e in device_entities:
        if e.get("entity_id", "").startswith(domain + "."):
            return e["entity_id"]
    return None

# ---------------- Endpoints -----------------
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

async def _build_entity_area_map() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, Any]]]:
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
            if area_id:
                ent_to_area[e["entity_id"]] = area_id
            if e.get("device_id"):
                ent_to_device[e["entity_id"]] = e["device_id"]
        return ent_to_area, ent_to_device, dev_by_id
    finally:
        await ws.close()

async def _event_forwarder_task():
    if not MONITOR_URL:
        print("Forwarder disabled: MONITOR_URL is not set")
        return  # forwarding disabled
    ent_to_area, ent_to_device, dev_by_id = await _build_entity_area_map()
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
                    device_name = None
                    dev_id = ent_to_device.get(entity_id)
                    if dev_id:
                        device_name = (dev_by_id.get(dev_id, {}) or {}).get("name")
                    if not device_name:
                        object_id = entity_id.split(".", 1)[-1]
                        device_name = object_id
                    artifact_name = urllib.parse.quote(device_name, safe="")
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
        ent_to_area, ent_to_device, dev_by_id = await _build_entity_area_map()
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
                # Resolve artifact name from device, fallback to object_id
                device_name = None
                dev_id = ent_to_device.get(entity_id)
                if dev_id:
                    device_name = (dev_by_id.get(dev_id, {}) or {}).get("name")
                if not device_name:
                    object_id = entity_id.split(".", 1)[-1]
                    device_name = object_id
                artifact_name = urllib.parse.quote(device_name, safe="")
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
    print(f"App startup: MONITOR_URL={'set' if MONITOR_URL else 'unset'}, EXPLORER_URL={'set' if EXPLORER_URL else 'unset'}, AREAS={sorted(AREAS) if AREAS else 'ALL'}, BASE_WS_URI={BASE_WS_URI}")
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
        devices = await ha_client.get_devices(workspace_id)
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
        devices = await ha_client.get_devices(workspace_id)
        rdf = HomeAssistantRDF(str(request.base_url))
        aid = area["area_id"]
        ws = URIRef(f"{rdf.base}workspaces/{aid}#workspace")
        art_dir = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/")
        for d in devices:
            name = d.get("name", d["id"])  # device label
            safe_name = urllib.parse.quote(name, safe="")
            art = URIRef(f"{art_dir}{safe_name}#artifact")
            rdf.g.add((art, RDF.type, HMAS.Artifact))
            rdf.g.add((ws, HMAS.contains, art))
            rdf.g.add((art, TD.title, Literal(name)))
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
        device, device_entities = await _resolve_device_and_entities(workspace_id, artifact_name)
        states = await ha_rest.get_states()
        state_map = {s["entity_id"]: s for s in states}

        #print(states)
        
        rdf = HomeAssistantRDF(str(request.base_url))
        aid = workspace_id
        ws = URIRef(f"{rdf.base}workspaces/{aid}#workspace")
        art_dir = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/")
        safe_name = urllib.parse.quote(device.get("name"), safe="")
        art = URIRef(f"{art_dir}{safe_name}#artifact")

        # Build RDF
        rdf.g.add((art, RDF.type, TD.Thing))
        rdf.g.add((art, RDF.type, HMAS.Artifact))
        rdf.g.add((art, TD.title, Literal(device.get("name"))))
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
                rdf._add_action(
                    art,
                    action_name,
                    EX.StatusCommand,
                    "POST",
                    URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/ha/{urllib.parse.quote(domain, safe='')}/{urllib.parse.quote(svc_name, safe='')}"),
                    "application/json",
                )

        # Sensor-specific value action
        if any(d in domains for d in ("sensor",)):
            sensor_ent = _pick_entity(device_entities, "sensor")
            st = state_map.get(sensor_ent, {}) if sensor_ent else {}
            attrs = st.get("attributes", {}) if isinstance(st, dict) else {}
            action_name = _sensor_action_name(attrs.get("device_class"), attrs.get("unit_of_measurement"))
            if action_name:
                rdf._add_action(
                    art,
                    action_name,
                    EX.StatusCommand,
                    "POST",
                    URIRef(f"{rdf.base}workspaces/{aid}/artifacts/{safe_name}/{urllib.parse.quote(action_name, safe='')}"),
                    "application/json",
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
                "artifact": device.get("name"),
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
        devices = await ha_client.get_devices(area_id)
        base = BASE_WS_URI.rstrip("/")
        artifact_uris = []
        for d in devices:
            name = d.get("name", d["id"])  # device label
            safe_name = urllib.parse.quote(name, safe="")
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
    device, device_entities = await _resolve_device_and_entities(workspace_id, artifact_name)
    ent = _pick_entity(device_entities, domain)
    if not ent:
        raise HTTPException(status_code=404, detail=f"No {domain} entity on artifact")
    return ent

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

# Dynamic sensor action: get<device_class>in<unit>
@app.post("/workspaces/{workspace_id}/artifacts/{artifact_name}/{action_name}")
async def action_sensor_dynamic(workspace_id: str, artifact_name: str, action_name: str):
    # Only handle actions shaped like get<dc>in<unit>
    if not (action_name.startswith("get") and "In" in action_name[3:]):
        raise HTTPException(status_code=404, detail="Unknown action")

    # Resolve sensor entity
    _, device_entities = await _resolve_device_and_entities(workspace_id, artifact_name)
    ent = _pick_entity(device_entities, "sensor")
    if not ent:
        raise HTTPException(status_code=404, detail="No sensor entity on artifact")

    # Fetch current state and validate the requested action matches this sensor
    states = await ha_rest.get_states()
    st = next((s for s in states if s.get("entity_id") == ent), None)
    if not st:
        raise HTTPException(status_code=404, detail="Sensor state not found")

    attrs = st.get("attributes", {})
    expected_with_dc = _sensor_action_name(attrs.get("device_class"), attrs.get("unit_of_measurement"))
    expected_without_dc = _sensor_action_name(None, attrs.get("unit_of_measurement"))
    if action_name not in {x for x in (expected_with_dc, expected_without_dc) if x}:
        raise HTTPException(status_code=404, detail="Action not applicable to this sensor")

    # Return exact state value as plain text
    return PlainTextResponse(str(st.get("state", "")))
