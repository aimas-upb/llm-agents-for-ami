#!/usr/bin/env python3
"""
Install a Virtual Devices YAML into Home Assistant and assign its devices/entities to an area.

Usage:
  python add_virtual_devices.py <yaml_file> <area_name> <virtual_yaml_dir>

Arguments:
  yaml_file:
      Path to the source YAML file to install.
  area_name:
      Home Assistant area name to create/reuse and assign all created devices/entities to.
  virtual_yaml_dir:
      Host filesystem directory that maps to Home Assistant's
      /config/custom_components/virtual directory.

Environment:
  - HA_TOKEN: long-lived Home Assistant token
  - HA_URL:   Home Assistant websocket URL, e.g. ws://localhost:8123/api/websocket

Behavior:
  1. Copies the YAML file into <virtual_yaml_dir>/<yaml_file.name>
  2. Creates or reuses the requested Home Assistant area
  3. Starts the Virtual Devices config flow for domain "virtual"
  4. Submits the YAML file path and group name to the flow
  5. Assigns newly created devices/entities to the target area

Notes:
  - The script assumes the Home Assistant container sees virtual YAMLs at
    /config/custom_components/virtual/<filename>.
  - The exact Virtual Devices config flow schema can vary by version; this script
    inspects the form keys and fills known field names heuristically.
  - The printed "area_id" is the exact Home Assistant registry ID to use in
    HASP's AREAS environment variable.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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


def _snapshot_ids(items: Iterable[Dict[str, Any]], key: str) -> Set[str]:
    out: Set[str] = set()
    for item in items:
        value = item.get(key)
        if isinstance(value, str) and value:
            out.add(value)
    return out


def _extract_schema_keys(flow_resp: Dict[str, Any]) -> List[str]:
    schema = flow_resp.get("data_schema")
    if isinstance(schema, list):
        keys: List[str] = []
        for item in schema:
            if isinstance(item, dict):
                if isinstance(item.get("name"), str):
                    keys.append(item["name"])
                elif isinstance(item.get("key"), str):
                    keys.append(item["key"])
            elif isinstance(item, str):
                keys.append(item)
        return keys
    if isinstance(schema, dict):
        keys = schema.get("fields") if isinstance(schema.get("fields"), list) else schema.keys()
        return [k for k in keys if isinstance(k, str)]
    return []


def _build_flow_input(
    schema_keys: List[str],
    *,
    area_name: str,
    area_id: str,
    group_name: str,
    ha_yaml_path: str,
) -> Dict[str, Any]:
    if not schema_keys:
        return {}

    values: Dict[str, Any] = {}
    for key in schema_keys:
        lk = key.lower()
        if lk in {"name", "group_name", "group", "title", "friendly_name"}:
            values[key] = group_name
        elif lk in {"file", "path", "filename", "file_name", "config_path", "yaml_file"}:
            values[key] = ha_yaml_path
        elif lk in {"area_name", "area"}:
            values[key] = area_name
        elif lk == "area_id":
            values[key] = area_id
        elif lk in {"create_area", "add_area"}:
            values[key] = False
    return values


async def _ensure_area(ws, area_name: str) -> str:
    areas = await _ws_call(ws, {"type": "config/area_registry/list"})
    for area in areas:
        if area.get("name") == area_name and area.get("area_id"):
            return area["area_id"]
    created = await _ws_call(ws, {"type": "config/area_registry/create", "name": area_name})
    area_id = created.get("area_id")
    if not area_id:
        raise RuntimeError(f"Failed to create area '{area_name}'")
    return area_id


async def _run_virtual_flow(
    client: httpx.AsyncClient,
    *,
    area_name: str,
    area_id: str,
    group_name: str,
    ha_yaml_path: str,
) -> Dict[str, Any]:
    start = await client.post(
        "/api/config/config_entries/flow",
        json={"handler": "virtual", "show_advanced_options": False},
    )
    start.raise_for_status()
    flow = start.json()

    for _ in range(6):
        flow_type = flow.get("type")
        if flow_type in {"create_entry", "abort"}:
            return flow
        flow_id = flow.get("flow_id")
        if not flow_id:
            raise RuntimeError(f"Unexpected flow response without flow_id: {flow}")

        payload = _build_flow_input(
            _extract_schema_keys(flow),
            area_name=area_name,
            area_id=area_id,
            group_name=group_name,
            ha_yaml_path=ha_yaml_path,
        )
        resp = await client.post(f"/api/config/config_entries/flow/{flow_id}", json=payload)
        resp.raise_for_status()
        flow = resp.json()

    raise RuntimeError(f"Virtual Devices flow did not complete: {flow}")


def _iter_device_config_entries(device: Dict[str, Any]) -> Iterable[str]:
    config_entries = device.get("config_entries")
    if isinstance(config_entries, list):
        for entry_id in config_entries:
            if isinstance(entry_id, str) and entry_id:
                yield entry_id


async def _assign_area_to_config_entry_items(
    ws,
    *,
    area_id: str,
    config_entry_id: str,
    poll_timeout_s: float = 15.0,
) -> Tuple[List[str], List[str]]:
    deadline = time.monotonic() + poll_timeout_s
    entry_devices: List[Dict[str, Any]] = []
    entry_entities: List[Dict[str, Any]] = []

    while time.monotonic() < deadline:
        devices = await _ws_call(ws, {"type": "config/device_registry/list"})
        entities = await _ws_call(ws, {"type": "config/entity_registry/list"})

        entry_devices = [
            item for item in devices
            if config_entry_id in set(_iter_device_config_entries(item))
        ]
        entry_entities = [
            item for item in entities
            if item.get("config_entry_id") == config_entry_id
        ]

        if entry_devices or entry_entities:
            break
        await asyncio.sleep(1.0)

    updated_devices: List[str] = []
    updated_entities: List[str] = []

    for device in entry_devices:
        device_id = device.get("id")
        if not device_id:
            continue
        await _ws_call(
            ws,
            {"type": "config/device_registry/update", "device_id": device_id, "area_id": area_id},
        )
        updated_devices.append(device.get("name") or device_id)

    entry_device_ids = _snapshot_ids(entry_devices, "id")
    for entity in entry_entities:
        entity_id = entity.get("entity_id")
        if not entity_id:
            continue
        device_id = entity.get("device_id")
        if isinstance(device_id, str) and device_id in entry_device_ids:
            continue
        await _ws_call(
            ws,
            {"type": "config/entity_registry/update", "entity_id": entity_id, "area_id": area_id},
        )
        updated_entities.append(entity_id)

    return updated_devices, updated_entities


async def main() -> None:
    if len(sys.argv) != 4:
        _die(
            "Usage: python add_virtual_devices.py <yaml_file> <area_name> <virtual_yaml_dir>"
        )

    yaml_file = Path(sys.argv[1]).expanduser().resolve()
    area_name = sys.argv[2].strip()
    virtual_yaml_dir = Path(sys.argv[3]).expanduser().resolve()

    if not area_name:
        _die("area_name must not be empty")
    if not yaml_file.is_file():
        _die(f"YAML file not found: {yaml_file}")
    if not virtual_yaml_dir.exists():
        _die(f"virtual_yaml_dir does not exist: {virtual_yaml_dir}")
    if not virtual_yaml_dir.is_dir():
        _die(f"virtual_yaml_dir is not a directory: {virtual_yaml_dir}")

    ha_token = _ensure_env("HA_TOKEN")
    ha_ws_url = _ensure_env("HA_URL")
    ha_base_url = _derive_base_url(ha_ws_url)

    dest_path = virtual_yaml_dir / yaml_file.name
    shutil.copy2(yaml_file, dest_path)

    # Path AS SEEN BY HomeAssistant. In the docker deployment the host
    # virtual_yaml_dir is mounted at /config/custom_components/virtual; for a
    # venv HA Core the host path IS the HA path, so override the prefix:
    #   export HA_VIRTUAL_CONFIG_PREFIX=~/ha_config/custom_components/virtual
    ha_config_prefix = os.getenv(
        "HA_VIRTUAL_CONFIG_PREFIX", "/config/custom_components/virtual"
    ).rstrip("/")
    ha_yaml_path = f"{ha_config_prefix}/{yaml_file.name}"
    group_name = yaml_file.stem

    headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}

    ws = await _ws_handshake(ha_ws_url, ha_token)
    try:
        async with httpx.AsyncClient(
            base_url=ha_base_url,
            headers=headers,
            timeout=20.0,
        ) as client:
            area_id = await _ensure_area(ws, area_name)

            flow_result = await _run_virtual_flow(
                client,
                area_name=area_name,
                area_id=area_id,
                group_name=group_name,
                ha_yaml_path=ha_yaml_path,
            )

            created_entry = flow_result.get("result") if isinstance(flow_result, dict) else None
            created_entry_id = created_entry.get("entry_id") if isinstance(created_entry, dict) else None
            if not isinstance(created_entry_id, str) or not created_entry_id:
                raise RuntimeError(f"Virtual Devices flow did not return an entry_id: {flow_result}")

            updated_devices, updated_entities = await _assign_area_to_config_entry_items(
                ws,
                area_id=area_id,
                config_entry_id=created_entry_id,
            )
    finally:
        await ws.close()

    print(
        json.dumps(
            {
                "copied_yaml": str(dest_path),
                "ha_yaml_path": ha_yaml_path,
                "area_name": area_name,
                "area_id": area_id,
                "hasp_areas_env": area_id,
                "hasp_hint": f"Set AREAS={area_id} for HASP",
                "flow_result_type": flow_result.get("type"),
                "flow_result": flow_result,
                "updated_devices": updated_devices,
                "updated_entities": updated_entities,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
