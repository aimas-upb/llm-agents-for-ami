#!/usr/bin/env python3
"""
Set a property for a Home Assistant artifact (device) by name.

Usage:
  python set-property.py "<artifact_name>" "<property>" "<value>"

Environment:
  - HA_TOKEN: Long-lived access token (required)
  - HA_URL:   WebSocket URL to HA (e.g., ws://localhost:8123/api/websocket)

Behavior:
  - Resolves the device by display name via HA WebSocket registries
  - Picks the first entity belonging to that device
  - If property == "state": updates the entity state
  - Else: updates the entity attributes with {property: value}
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, Optional

import httpx
import websockets


def _ensure_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        print(f"Missing required environment variable: {var}", file=sys.stderr)
        sys.exit(2)
    return val


async def _ws_call(ws, payload: Dict[str, Any]) -> Any:
    _ws_call.counter += 1
    ident = _ws_call.counter
    await ws.send(json.dumps({**payload, "id": ident}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == ident and msg.get("type") == "result":
            if msg.get("success"):
                return msg["result"]
            raise RuntimeError(msg.get("error"))


_ws_call.counter = 0


async def _ws_handshake(url: str, token: str):
    ws = await websockets.connect(url)
    msg = json.loads(await ws.recv())
    if msg.get("type") != "auth_required":
        raise RuntimeError("Unexpected WebSocket handshake")
    await ws.send(json.dumps({"type": "auth", "access_token": token}))
    msg = json.loads(await ws.recv())
    if msg.get("type") != "auth_ok":
        raise RuntimeError("Auth failed (check HA_TOKEN)")
    return ws


def _derive_base_url(ha_url: str) -> str:
    base = ha_url.replace("ws://", "http://").replace("wss://", "https://")
    if base.endswith("/api/websocket"):
        base = base[: -len("/api/websocket")]
    return base.rstrip("/")


def _parse_value(val: str) -> Any:
    # Try JSON parsing first to support numbers/bools/null/objects
    try:
        return json.loads(val)
    except Exception:
        pass
    # Fallbacks: on/off → strings preserved; otherwise raw string
    return val


async def main():
    if len(sys.argv) != 4:
        print("Usage: python set-property.py <artifact_name> <property> <value>")
        sys.exit(2)

    artifact_name, prop, raw_value = sys.argv[1], sys.argv[2], sys.argv[3]

    ha_token = _ensure_env("HA_TOKEN")
    ha_ws_url = _ensure_env("HA_URL")
    base_url = _derive_base_url(ha_ws_url)

    # Resolve device → entity_id via WS registries
    async with websockets.connect(ha_ws_url) as ws:
        # Handshake
        msg = json.loads(await ws.recv())
        if msg.get("type") != "auth_required":
            print("Unexpected WebSocket handshake", file=sys.stderr)
            sys.exit(1)
        await ws.send(json.dumps({"type": "auth", "access_token": ha_token}))
        if json.loads(await ws.recv()).get("type") != "auth_ok":
            print("Auth failed (check HA_TOKEN)", file=sys.stderr)
            sys.exit(1)

        devices = await _ws_call(ws, {"type": "config/device_registry/list"})
        entities = await _ws_call(ws, {"type": "config/entity_registry/list"})

        device = next((d for d in devices if d.get("name") == artifact_name), None)
        if not device:
            print(f"Artifact (device) not found: {artifact_name}", file=sys.stderr)
            sys.exit(1)

        device_entities = [e for e in entities if e.get("device_id") == device.get("id")]
        print("Resolved device:", json.dumps({"id": device.get("id"), "name": device.get("name")}, indent=2))
        print("Device entities:")
        for e in device_entities:
            ent = e.get("entity_id")
            print(" -", ent)
        if not device_entities:
            print(f"No entities found for artifact: {artifact_name}", file=sys.stderr)
            sys.exit(1)

        # Choose the most suitable entity by domain
        def pick_entity(preferred_domains: Optional[set[str]] = None) -> Optional[str]:
            if preferred_domains:
                for e in device_entities:
                    ent = e.get("entity_id") or ""
                    dom = ent.split(".")[0] if "." in ent else ""
                    if dom in preferred_domains:
                        return ent
            # fallback: first entity
            return device_entities[0].get("entity_id")

        preferred: Optional[set[str]] = None
        if sys.argv[2] == "state":
            preferred = {"light", "switch", "fan", "cover", "input_boolean"}
        elif sys.argv[2] in {"brightness", "brightness_pct"}:
            preferred = {"light"}
        entity_id = pick_entity(preferred)
        print("Chosen entity:", entity_id)
        if not entity_id:
            print("Entity without entity_id; aborting", file=sys.stderr)
            sys.exit(1)

    # Try to find an applicable HA service first (preferred over direct state set)
    headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=10.0) as client:
        # Inspect services
        services_resp = await client.get("/api/services")
        services_resp.raise_for_status()
        services = services_resp.json()
        ent_domain = entity_id.split(".")[0]
        value = _parse_value(raw_value)

        def select_service() -> Optional[Dict[str, Any]]:
            # Domain-specific mapping for state on/off style
            if prop == "state":
                desired = str(value).strip().lower()
                on_values = {"on", "true", "1", "open"}
                off_values = {"off", "false", "0", "closed", "close"}
                if ent_domain in {"light", "switch", "fan", "input_boolean"}:
                    svc = "turn_on" if desired in on_values else ("turn_off" if desired in off_values else None)
                    if svc:
                        return {"domain": ent_domain, "name": svc, "fields": {"entity_id": {}}, "score": 100}
                if ent_domain == "cover":
                    svc = "open_cover" if desired in {"open"} else ("close_cover" if desired in {"closed", "close"} else None)
                    if svc:
                        return {"domain": ent_domain, "name": svc, "fields": {"entity_id": {}}, "score": 100}
            candidates = []
            for svc in services:
                dom = svc.get("domain")
                svcs = svc.get("services", {}) or {}
                for name, meta in svcs.items():
                    fields = (meta.get("fields") or {})
                    target = (meta.get("target") or {}).get("entity", [])
                    legacy_applies = "entity_id" in fields
                    modern_applies = any(ent_domain in (t.get("domain") or []) for t in target)
                    if not (legacy_applies or modern_applies):
                        continue
                    score = 0
                    # Prefer virtual domain
                    if dom == "virtual":
                        score += 5
                    # Prefer services that look like setters
                    if "set" in name:
                        score += 3
                    # Prefer ones that have our property or generic 'value'
                    if prop in fields:
                        score += 3
                    if prop == "state" and "value" in fields:
                        score += 2
                    # Prefer domain match
                    if dom == ent_domain:
                        score += 1
                    if score > 0:
                        candidates.append({"domain": dom, "name": name, "fields": fields, "score": score})
            if not candidates:
                return None
            candidates.sort(key=lambda x: x["score"], reverse=True)
            print("Candidate services (top 10):")
            for c in candidates[:10]:
                print(f" - {c['domain']}.{c['name']} score={c['score']} fields={list(c['fields'].keys())}")
            return candidates[0]

        chosen = select_service()
        if chosen is not None:
            payload: Dict[str, Any] = {"entity_id": entity_id}
            fields = chosen["fields"]
            # Map property to appropriate field name
            if prop == "state":
                # For on/off style services (turn_on/turn_off) no extra field is needed.
                # Only pass a value if the service explicitly supports a 'value' field.
                if "value" in fields:
                    payload["value"] = value
            elif prop in fields:
                payload[prop] = value
            elif ent_domain == "light" and prop in {"brightness", "brightness_pct"}:
                payload[prop] = value
            else:
                # Fallback to a common field name
                payload[prop] = value

            url = f"/api/services/{chosen['domain']}/{chosen['name']}"
            print("Calling service:", chosen['domain'] + "." + chosen['name'])
            print("Payload:", json.dumps(payload))
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                print(f"Service call failed ({chosen['domain']}.{chosen['name']}): {resp.status_code} {resp.text}", file=sys.stderr)
                sys.exit(1)
            print(json.dumps({
                "used_service": f"{chosen['domain']}.{chosen['name']}",
                "request": payload,
                "entity_id": entity_id,
                "result": resp.json() if resp.headers.get('content-type','').startswith('application/json') else resp.text,
            }, indent=2))
            return

        # Fallback: direct state set (ephemeral for many entities)
        r = await client.get(f"/api/states/{entity_id}")
        r.raise_for_status()
        current = r.json()
        current_state = current.get("state")
        current_attrs: Dict[str, Any] = current.get("attributes") or {}
        new_state = value if prop == "state" else current_state
        new_attrs = dict(current_attrs) if prop != "state" else current_attrs
        if prop != "state":
            new_attrs[prop] = value
        payload_state = {"state": new_state, "attributes": new_attrs}
        print("FALLBACK direct state set; payload:", json.dumps(payload_state))
        r2 = await client.post(f"/api/states/{entity_id}", json=payload_state)
        if r2.status_code >= 400:
            print(f"Failed to set property: {r2.status_code} {r2.text}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps({"used_service": None, "request": payload_state, "result": r2.json()}, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted")
