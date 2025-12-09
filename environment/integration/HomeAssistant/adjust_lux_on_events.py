#!/usr/bin/env python3
"""
Subscribe to Home Assistant state_changed events and adjust the
internal illuminance when lights turn on/off or blinds open/close.

Triggers
- light.*: state == "on"  → +INCREMENT_LIGHT; state == "off" → -DECREMENT_LIGHT
- cover.*: opening/open or position increased → +INCREMENT_BLINDS
           closing/closed or position decreased → -DECREMENT_BLINDS

Target sensor (configurable via env):
- SENSOR_ENTITY_ID: default "sensor.internal_light_sensing_308"

Environment
- HA_URL   (e.g., ws://localhost:8123/api/websocket)
- HA_TOKEN (long-lived access token)
- INCREMENT_LIGHT   (default: 100)
- DECREMENT_LIGHT   (default: same as INCREMENT_LIGHT)
- INCREMENT_BLINDS  (default: 50)
- DECREMENT_BLINDS  (default: same as INCREMENT_BLINDS)

Notes
- Uses HA WebSocket API for events and HA REST API for setting sensor value.
- Attempts to find an appropriate service accepting entity_id+value; if not
  found, falls back to direct state set on the entity.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

import httpx
import websockets


def _env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


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


def _base_url(ha_url: str) -> str:
    base = ha_url.replace("ws://", "http://").replace("wss://", "https://")
    if base.endswith("/api/websocket"):
        base = base[: -len("/api/websocket")]
    return base.rstrip("/")


async def _get_services(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    r = await client.get("/api/services")
    r.raise_for_status()
    return r.json()


def _pick_set_service(services: list[dict[str, Any]]) -> Optional[tuple[str, str, Dict[str, Any]]]:
    """Pick a service that can set a sensor value. Prefer domain 'virtual' with field 'value'."""
    best: Optional[tuple[str, str, Dict[str, Any], int]] = None
    for svc in services:
        dom = svc.get("domain")
        svcs = svc.get("services", {}) or {}
        for name, meta in svcs.items():
            fields = meta.get("fields") or {}
            target = (meta.get("target") or {}).get("entity", [])
            legacy = "entity_id" in fields
            modern = any("sensor" in (t.get("domain") or []) for t in target)
            if not (legacy or modern):
                continue
            score = 0
            if dom == "virtual":
                score += 5
            if "set" in name or "value" in fields:
                score += 3
            if score > 0 and (best is None or score > best[3]):
                best = (dom, name, fields, score)
    if best:
        dom, name, fields, _ = best
        return dom, name, fields
    return None


async def _get_state(client: httpx.AsyncClient, entity_id: str) -> Dict[str, Any]:
    r = await client.get(f"/api/states/{entity_id}")
    r.raise_for_status()
    return r.json()


async def _set_sensor_value(client: httpx.AsyncClient, services: list[dict[str, Any]], sensor_entity: str, new_value: Any) -> None:
    picked = _pick_set_service(services)
    if picked:
        dom, name, fields = picked
        payload: Dict[str, Any] = {"entity_id": sensor_entity}
        if "value" in fields:
            payload["value"] = new_value
        else:
            # Fallback to generic key
            payload["value"] = new_value
        r = await client.post(f"/api/services/{dom}/{name}", json=payload)
        r.raise_for_status()
        return
    # Fallback: direct state set
    current = await _get_state(client, sensor_entity)
    attrs = current.get("attributes") or {}
    r2 = await client.post(f"/api/states/{sensor_entity}", json={"state": new_value, "attributes": attrs})
    r2.raise_for_status()


def _delta_from_event(ev: Dict[str, Any], inc_light: int, dec_light: int, inc_blinds: int, dec_blinds: int) -> tuple[int, str]:
    data = ev.get("event", {}).get("data", {})
    ent = data.get("entity_id", "")
    new = data.get("new_state") or {}
    old = data.get("old_state") or {}
    state = (new.get("state") or "").lower()
    domain = ent.split(".")[0] if "." in ent else ""
    if domain == "light":
        if state == "on":
            return inc_light, "light"
        if state == "off":
            return -dec_light, "light"
    if domain == "cover":
        if state in ("opening", "open"):
            return inc_blinds, "cover"
        if state in ("closing", "closed"):
            return -dec_blinds, "cover"
        try:
            new_pos = int((new.get("attributes") or {}).get("current_position"))
            old_pos = int((old.get("attributes") or {}).get("current_position")) if old else None
            if old_pos is not None:
                if new_pos > old_pos:
                    return inc_blinds, "cover"
                if new_pos < old_pos:
                    return -dec_blinds, "cover"
        except Exception:
            pass
    return 0, ""


async def main():
    ha_url = _env("HA_URL")
    ha_token = _env("HA_TOKEN")
    sensor_entity = os.getenv("SENSOR_ENTITY_ID", "sensor.internal_light_sensing_308")
    inc_light = int(os.getenv("INCREMENT_LIGHT", "100"))
    dec_light = int(os.getenv("DECREMENT_LIGHT", str(inc_light)))
    inc_blinds = int(os.getenv("INCREMENT_BLINDS", "50"))
    dec_blinds = int(os.getenv("DECREMENT_BLINDS", str(inc_blinds)))

    base_url = _base_url(ha_url)
    headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=10.0) as client:
        services = await _get_services(client)
        ws = await _ws_handshake(ha_url, ha_token)
        try:
            await ws.send(json.dumps({"id": 1, "type": "subscribe_events", "event_type": "state_changed"}))
            print("Subscribed to state_changed events")
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") != "event":
                    continue
                delta, src = _delta_from_event(msg, inc_light, dec_light, inc_blinds, dec_blinds)
                if delta == 0:
                    continue
                try:
                    current = await _get_state(client, sensor_entity)
                    cur_val_raw = current.get("state")
                    try:
                        cur_val = float(cur_val_raw)
                    except Exception:
                        cur_val = 0.0
                    new_val = max(0.0, cur_val + float(delta))
                    await _set_sensor_value(client, services, sensor_entity, new_val)
                    print(f"{sensor_entity} ← {new_val} (from {src}, delta={delta})")
                except Exception as e:
                    print("Adjust failed:", e)
        except KeyboardInterrupt:
            print("Interrupted; closing")
        finally:
            await ws.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted")
