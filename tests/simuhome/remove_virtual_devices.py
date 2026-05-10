#!/usr/bin/env python3
"""
Remove Virtual Devices config entries associated with a Home Assistant area.

Usage:
  python remove_virtual_devices.py <area_name>

Environment:
  - HA_TOKEN: long-lived Home Assistant token
  - HA_URL:   Home Assistant websocket URL, e.g. ws://localhost:8123/api/websocket

Behavior:
  1. Resolves the Home Assistant area by name
  2. Finds devices/entities assigned to that area
  3. Collects Virtual Devices config entries associated with those items
  4. Removes only config entries whose linked devices/entities are scoped entirely
     to the specified area

Notes:
  - This removes config entries from Home Assistant; it does not delete YAML files
    from the host filesystem.
  - If a virtual config entry is linked to items across multiple areas, it is skipped
    to avoid deleting devices outside the requested area.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set

import httpx
import websockets


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _ensure_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        _die(f"Missing required environment variable: {var}")
    return val


def _derive_base_url(ha_url: str) -> str:
    base = ha_url.replace("ws://", "http://").replace("wss://", "https://")
    if base.endswith("/api/websocket"):
        base = base[: -len("/api/websocket")]
    return base.rstrip("/")


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


async def _ws_call(ws, payload: Dict[str, Any]) -> Any:
    _ws_call.counter += 1
    ident = _ws_call.counter
    await ws.send(json.dumps({**payload, "id": ident}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == ident and msg.get("type") == "result":
            if msg.get("success"):
                return msg.get("result")
            raise RuntimeError(str(msg.get("error")))


_ws_call.counter = 0


def _device_area_id(device: Dict[str, Any]) -> Optional[str]:
    area_id = device.get("area_id")
    return area_id if isinstance(area_id, str) and area_id else None


def _entity_effective_area_id(entity: Dict[str, Any], devices_by_id: Dict[str, Dict[str, Any]]) -> Optional[str]:
    area_id = entity.get("area_id")
    if isinstance(area_id, str) and area_id:
        return area_id
    device_id = entity.get("device_id")
    if isinstance(device_id, str):
        return _device_area_id(devices_by_id.get(device_id, {}))
    return None


def _iter_entity_config_entries(entity: Dict[str, Any]) -> Iterable[str]:
    config_entry_id = entity.get("config_entry_id")
    if isinstance(config_entry_id, str) and config_entry_id:
        yield config_entry_id


def _iter_device_config_entries(device: Dict[str, Any]) -> Iterable[str]:
    config_entries = device.get("config_entries")
    if isinstance(config_entries, list):
        for entry_id in config_entries:
            if isinstance(entry_id, str) and entry_id:
                yield entry_id


async def main() -> None:
    if len(sys.argv) != 2:
        _die("Usage: python remove_virtual_devices.py <area_name>")

    area_name = sys.argv[1].strip()
    if not area_name:
        _die("area_name must not be empty")

    ha_token = _ensure_env("HA_TOKEN")
    ha_ws_url = _ensure_env("HA_URL")
    ha_base_url = _derive_base_url(ha_ws_url)

    headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}

    ws = await _ws_handshake(ha_ws_url, ha_token)
    try:
        async with httpx.AsyncClient(base_url=ha_base_url, headers=headers, timeout=20.0) as client:
            areas = await _ws_call(ws, {"type": "config/area_registry/list"})
            area = next((item for item in areas if item.get("name") == area_name), None)
            if not area or not area.get("area_id"):
                _die(f"Area not found: {area_name}", code=1)
            area_id = area["area_id"]

            devices = await _ws_call(ws, {"type": "config/device_registry/list"})
            entities = await _ws_call(ws, {"type": "config/entity_registry/list"})
            devices_by_id = {
                item["id"]: item
                for item in devices
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            }

            entries_resp = await client.get("/api/config/config_entries/entry")
            entries_resp.raise_for_status()
            config_entries = entries_resp.json()
            entries_by_id = {
                item["entry_id"]: item
                for item in config_entries
                if isinstance(item, dict) and isinstance(item.get("entry_id"), str)
            }

            area_entry_ids: Set[str] = set()
            entry_areas: Dict[str, Set[Optional[str]]] = defaultdict(set)
            entry_devices: Dict[str, List[str]] = defaultdict(list)
            entry_entities: Dict[str, List[str]] = defaultdict(list)

            for device in devices:
                if _device_area_id(device) != area_id:
                    continue
                device_name = device.get("name") or device.get("id")
                for entry_id in _iter_device_config_entries(device):
                    area_entry_ids.add(entry_id)
                    entry_areas[entry_id].add(area_id)
                    entry_devices[entry_id].append(str(device_name))

            for entity in entities:
                effective_area = _entity_effective_area_id(entity, devices_by_id)
                if effective_area != area_id:
                    continue
                entity_id = entity.get("entity_id")
                if not isinstance(entity_id, str) or not entity_id:
                    continue
                for entry_id in _iter_entity_config_entries(entity):
                    area_entry_ids.add(entry_id)
                    entry_areas[entry_id].add(area_id)
                    entry_entities[entry_id].append(entity_id)

            for device in devices:
                device_area = _device_area_id(device)
                for entry_id in _iter_device_config_entries(device):
                    if entry_id in area_entry_ids:
                        entry_areas[entry_id].add(device_area)

            for entity in entities:
                effective_area = _entity_effective_area_id(entity, devices_by_id)
                for entry_id in _iter_entity_config_entries(entity):
                    if entry_id in area_entry_ids:
                        entry_areas[entry_id].add(effective_area)

            removed: List[Dict[str, Any]] = []
            skipped: List[Dict[str, Any]] = []

            for entry_id in sorted(area_entry_ids):
                entry = entries_by_id.get(entry_id)
                if not entry:
                    skipped.append(
                        {
                            "entry_id": entry_id,
                            "reason": "config entry not found via REST API",
                        }
                    )
                    continue
                if entry.get("domain") != "virtual":
                    skipped.append(
                        {
                            "entry_id": entry_id,
                            "title": entry.get("title"),
                            "reason": f"entry domain is {entry.get('domain')}, not virtual",
                        }
                    )
                    continue

                linked_areas = {item for item in entry_areas.get(entry_id, set()) if item}
                if linked_areas and linked_areas != {area_id}:
                    skipped.append(
                        {
                            "entry_id": entry_id,
                            "title": entry.get("title"),
                            "reason": "entry spans multiple areas",
                            "linked_area_ids": sorted(linked_areas),
                        }
                    )
                    continue

                resp = await client.delete(f"/api/config/config_entries/entry/{entry_id}")
                if resp.status_code >= 400:
                    skipped.append(
                        {
                            "entry_id": entry_id,
                            "title": entry.get("title"),
                            "reason": f"delete failed: {resp.status_code}",
                            "response": resp.text,
                        }
                    )
                    continue

                removed.append(
                    {
                        "entry_id": entry_id,
                        "title": entry.get("title"),
                        "device_names": sorted(set(entry_devices.get(entry_id, []))),
                        "entity_ids": sorted(set(entry_entities.get(entry_id, []))),
                    }
                )

    finally:
        await ws.close()

    print(
        json.dumps(
            {
                "area_name": area_name,
                "area_id": area_id,
                "removed": removed,
                "skipped": skipped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
