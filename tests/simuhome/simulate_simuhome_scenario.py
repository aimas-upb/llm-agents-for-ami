#!/usr/bin/env python3
"""
Home Assistant sidecar simulator for one SimuHome benchmark scenario.

The generated SimuHome YAML exposes room state values as Home Assistant sensors.
This sidecar watches the scenario's actuator entities and updates those room
state sensors with SimuHome-style dynamics.

Usage:
  python tests/simuhome/simulate_simuhome_scenario.py \
    /path/to/SimuHome/data/benchmark/qt2_feasible_seed_1.json

Environment:
  - HA_URL    WebSocket URL, e.g. ws://localhost:8123/api/websocket
  - HA_TOKEN  Long-lived Home Assistant token

Notes:
  - This script expects entity names produced by
    initial_home_config_to_homeassistant_yaml.py.
  - It updates only sensors. Actuators remain under Home Assistant / agent
    control.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx
import websockets


ROOM_STATE_UNITS = {
    "temperature": "°C",
    "humidity": "%",
    "illuminance": "lx",
    "pm10": "ug/m3",
}

ROOM_STATE_CLASSES = {
    "temperature": "temperature",
    "humidity": "humidity",
    "illuminance": "illuminance",
    "pm10": "pm10",
}

ACTUATOR_TYPES = {
    "air_conditioner",
    "air_purifier",
    "dehumidifier",
    "dimmable_light",
    "fan",
    "heat_pump",
    "humidifier",
    "on_off_light",
    "window_covering_controller",
}

CONTROL_ATTR_SUFFIXES = {
    ".FanControl.PercentCurrent",
    ".FanControl.PercentSetting",
    ".LevelControl.CurrentLevel",
    ".OnOff.OnOff",
    ".Thermostat.OccupiedCoolingSetpoint",
    ".Thermostat.OccupiedHeatingSetpoint",
    ".Thermostat.SystemMode",
    ".WindowCovering.CurrentPositionLiftPercent100ths",
    ".WindowCovering.TargetPositionLiftPercent100ths",
}


def _env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _base_url(ha_url: str) -> str:
    base = ha_url.replace("ws://", "http://").replace("wss://", "https://")
    if base.endswith("/api/websocket"):
        base = base[: -len("/api/websocket")]
    return base.rstrip("/")


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "item"


def _dotted_token(value: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", value.strip())
    cleaned = [part for part in parts if part]
    return "".join(part[:1].upper() + part[1:] for part in cleaned) or "Item"


def _scenario_token(source_path: Path) -> str:
    return _dotted_token(source_path.stem)


def _room_token(room_id: str) -> str:
    return _dotted_token(room_id)


def _device_token(device_id: str, room_id: str) -> str:
    prefix = f"{room_id}_"
    core = device_id[len(prefix) :] if device_id.startswith(prefix) else device_id
    return _dotted_token(core)


def _entity_slug(name: str) -> str:
    return _slug(name)


def _state_value_for_ha(state_name: str, raw_value: Any) -> float:
    if not isinstance(raw_value, (int, float)):
        return 0.0
    if state_name in {"temperature", "humidity"}:
        return round(float(raw_value) / 100.0, 3)
    return round(float(raw_value), 3)


def _attribute_value_for_ha(attr_path: str, raw_value: Any) -> Any:
    if isinstance(raw_value, bool):
        return "on" if raw_value else "off"
    if raw_value is None:
        return "unknown"
    if isinstance(raw_value, (dict, list)):
        return json.dumps(raw_value, ensure_ascii=True, sort_keys=True)
    if not isinstance(raw_value, (int, float)):
        return raw_value

    parts = attr_path.split(".", 2)
    key = ".".join(parts[1:]) if len(parts) == 3 else attr_path
    if key in {
        "Thermostat.LocalTemperature",
        "Thermostat.OccupiedCoolingSetpoint",
        "Thermostat.OccupiedHeatingSetpoint",
        "TemperatureControl.MaxTemperature",
        "TemperatureControl.MinTemperature",
        "TemperatureControl.TemperatureSetpoint",
        "RelativeHumidityMeasurement.MeasuredValue",
        "RelativeHumidityMeasurement.MaxMeasuredValue",
        "RelativeHumidityMeasurement.MinMeasuredValue",
        "RelativeHumidityMeasurement.Tolerance",
        "WindowCovering.CurrentPositionLiftPercent100ths",
        "WindowCovering.TargetPositionLiftPercent100ths",
    }:
        return round(float(raw_value) / 100.0, 3)
    return round(float(raw_value), 3)


def _domain_for_device_type(device_type: str) -> Optional[str]:
    if device_type in {"on_off_light", "dimmable_light"}:
        return "light"
    if device_type in {"fan", "air_purifier", "humidifier", "dehumidifier"}:
        return "fan"
    if device_type in {"air_conditioner", "heat_pump"}:
        return "climate"
    if device_type == "window_covering_controller":
        return "cover"
    # Cabinets expose a settable target temperature; the rest are started and
    # stopped. Must stay in step with
    # initial_home_config_to_homeassistant_yaml.py::_device_primary_entry.
    if device_type in {"freezer", "refrigerator"}:
        return "number"
    if device_type in {"dishwasher", "laundry_washer", "laundry_dryer", "tv", "rvc"}:
        return "switch"
    return "sensor"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _entity_is_on(state: Dict[str, Any]) -> bool:
    return str((state or {}).get("state", "")).lower() in {
        "on",
        "open",
        "opening",
        "cool",
        "heat",
        "heating",
        "cooling",
    }


def _cover_open_fraction(state: Dict[str, Any]) -> float:
    attrs = (state or {}).get("attributes") or {}
    raw = attrs.get("current_position")
    if raw is not None:
        return _clamp(_as_float(raw) / 100.0, 0.0, 1.0)
    current = str((state or {}).get("state", "")).lower()
    if current == "open":
        return 1.0
    if current == "closed":
        return 0.0
    return 0.5


def _light_level(state: Dict[str, Any]) -> float:
    if not _entity_is_on(state):
        return 0.0
    attrs = (state or {}).get("attributes") or {}
    if "brightness" in attrs:
        return _clamp(_as_float(attrs["brightness"]) / 255.0, 0.0, 1.0)
    if "brightness_pct" in attrs:
        return _clamp(_as_float(attrs["brightness_pct"]) / 100.0, 0.0, 1.0)
    return 1.0


def _fan_percent(state: Dict[str, Any], default: float = 50.0) -> float:
    if not _entity_is_on(state):
        return 0.0
    attrs = (state or {}).get("attributes") or {}
    for key in ("percentage", "percent", "speed"):
        if key in attrs:
            return _clamp(_as_float(attrs[key], default), 0.0, 100.0)
    return default


def _sensor_on_off(state: Dict[str, Any]) -> bool:
    return str((state or {}).get("state", "")).lower() in {"on", "true", "1"}


def _sensor_number(state: Dict[str, Any], default: float = 0.0) -> float:
    return _as_float((state or {}).get("state"), default)


@dataclass
class DeviceBinding:
    room_id: str
    device_id: str
    device_type: str
    primary_entity_id: str
    attr_entity_ids: Dict[str, str] = field(default_factory=dict)


@dataclass
class RoomBinding:
    room_id: str
    env_entity_ids: Dict[str, str]
    baseline: Dict[str, float]
    current: Dict[str, float]
    devices: list[DeviceBinding] = field(default_factory=list)


@dataclass
class ScenarioBinding:
    name: str
    tick_interval: float
    base_time: Optional[str]
    clock_entity_id: Optional[str]
    rooms: Dict[str, RoomBinding]


def _attr_entity_name(scenario: str, room_id: str, device_id: str, attr_path: str) -> str:
    parts = attr_path.split(".", 2)
    if len(parts) == 3:
        _, cluster_id, attribute_id = parts
    else:
        cluster_id = "Attribute"
        attribute_id = attr_path
    return (
        f"{scenario}.{_room_token(room_id)}.{_device_token(device_id, room_id)}."
        f"{_dotted_token(cluster_id)}.{_dotted_token(attribute_id)}"
    )


def _load_scenario(path: Path) -> ScenarioBinding:
    payload = json.loads(path.read_text(encoding="utf-8"))
    initial_home_config = payload.get("initial_home_config")
    if not isinstance(initial_home_config, dict):
        raise ValueError(f"{path} does not contain initial_home_config")

    scenario = _scenario_token(path)
    clock_entity_id = f"sensor.{_entity_slug(f'{scenario}.Clock')}"
    rooms_cfg = initial_home_config.get("rooms")
    if not isinstance(rooms_cfg, dict):
        raise ValueError("initial_home_config.rooms must be an object")

    rooms: Dict[str, RoomBinding] = {}
    for room_id, room_cfg in rooms_cfg.items():
        if not isinstance(room_cfg, dict):
            continue
        state_cfg = room_cfg.get("state") if isinstance(room_cfg.get("state"), dict) else {}
        env_entity_ids: Dict[str, str] = {}
        baseline: Dict[str, float] = {}
        for state_name, raw_value in state_cfg.items():
            if state_name not in ROOM_STATE_UNITS:
                continue
            entity_name = f"{scenario}.{_room_token(room_id)}.{_dotted_token(state_name)}"
            env_entity_ids[state_name] = f"sensor.{_entity_slug(entity_name)}"
            baseline[state_name] = _state_value_for_ha(state_name, raw_value)

        rooms[room_id] = RoomBinding(
            room_id=room_id,
            env_entity_ids=env_entity_ids,
            baseline=dict(baseline),
            current=dict(baseline),
        )

        for device in room_cfg.get("devices") or []:
            if not isinstance(device, dict):
                continue
            device_id = str(device.get("device_id") or "").strip()
            device_type = str(device.get("device_type") or "").strip()
            attrs = device.get("attributes") if isinstance(device.get("attributes"), dict) else {}
            domain = _domain_for_device_type(device_type)
            if not device_id or not device_type or not domain:
                continue
            entity_name = f"{scenario}.{_room_token(room_id)}.{_device_token(device_id, room_id)}"
            attr_entity_ids = {
                attr_path: f"sensor.{_entity_slug(_attr_entity_name(scenario, room_id, device_id, attr_path))}"
                for attr_path in attrs
                if isinstance(attr_path, str)
            }
            rooms[room_id].devices.append(
                DeviceBinding(
                    room_id=room_id,
                    device_id=device_id,
                    device_type=device_type,
                    primary_entity_id=f"{domain}.{_entity_slug(entity_name)}",
                    attr_entity_ids=attr_entity_ids,
                )
            )

    return ScenarioBinding(
        name=path.stem,
        tick_interval=float(initial_home_config.get("tick_interval") or 0.1),
        base_time=initial_home_config.get("base_time"),
        clock_entity_id=clock_entity_id,
        rooms=rooms,
    )


class SimuHomeSidecar:
    def __init__(
        self,
        client: httpx.AsyncClient,
        scenario: ScenarioBinding,
        *,
        tick_seconds: float,
        speed: float,
        ha_sync_seconds: float,
    ):
        self.client = client
        self.scenario = scenario
        self.tick_seconds = tick_seconds
        self.speed = speed
        self.ha_sync_seconds = max(0.0, ha_sync_seconds)
        self.last_update: Optional[dt.datetime] = None
        self.last_ha_sync: Optional[dt.datetime] = None
        self.world_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    async def _get_state(self, entity_id: str) -> Dict[str, Any]:
        response = await self.client.get(f"/api/states/{entity_id}")
        response.raise_for_status()
        return response.json()

    async def _set_entity(
        self,
        entity_id: str,
        state: Any,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Dedupe against the value HA actually holds, not what we last wrote:
        # virtual entities can revert externally-set states, and a stale
        # last-written cache would then suppress the correcting write forever.
        current = await self._get_state(entity_id)
        if str(current.get("state")) == str(state):
            return
        merged_attrs = dict(current.get("attributes") or {})
        if attributes:
            merged_attrs.update(attributes)
        response = await self.client.post(
            f"/api/states/{entity_id}",
            json={"state": state, "attributes": merged_attrs},
        )
        response.raise_for_status()

    async def _read_device(self, device: DeviceBinding) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "primary": {"state": "unknown", "attributes": {}},
            "attrs": {},
        }
        try:
            data["primary"] = await self._get_state(device.primary_entity_id)
        except Exception:
            pass
        for attr_path, entity_id in device.attr_entity_ids.items():
            try:
                data["attrs"][attr_path] = await self._get_state(entity_id)
            except Exception:
                continue
        return data

    async def _read_room_world(self, room: RoomBinding) -> Dict[str, Dict[str, Any]]:
        world: Dict[str, Dict[str, Any]] = {}
        for device in room.devices:
            world[device.device_id] = await self._read_device(device)
        return world

    async def tick_once(self, *, force_ha_sync: bool = False) -> Dict[str, Dict[str, float]]:
        now = dt.datetime.now().astimezone()
        if self.last_update is None:
            elapsed_seconds = self.tick_seconds
        else:
            elapsed_seconds = max(0.0, (now - self.last_update).total_seconds())
        self.last_update = now
        dt_seconds = elapsed_seconds * self.speed
        sync_due = (
            force_ha_sync
            or self.last_ha_sync is None
            or (now - self.last_ha_sync).total_seconds() >= self.ha_sync_seconds
        )

        results: Dict[str, Dict[str, float]] = {}
        for room in self.scenario.rooms.values():
            if sync_due or room.room_id not in self.world_cache:
                self.world_cache[room.room_id] = await self._read_room_world(room)
            world = self.world_cache.get(room.room_id, {})
            room.current.update(self._compute_room_state(room, world, dt_seconds))
            if sync_due:
                await self._write_room_state(room)
            results[room.room_id] = dict(room.current)

        if sync_due:
            self.last_ha_sync = now
        if sync_due and self.scenario.clock_entity_id:
            await self._set_entity(
                self.scenario.clock_entity_id,
                now.isoformat(),
                {"device_class": "timestamp"},
            )
        return results

    def _compute_room_state(
        self,
        room: RoomBinding,
        world: Dict[str, Dict[str, Any]],
        dt_seconds: float,
    ) -> Dict[str, float]:
        out = dict(room.current)
        if "illuminance" in out:
            out["illuminance"] = self._compute_illuminance(room, world)
        if "temperature" in out:
            out["temperature"] = self._step_temperature(room, world, out["temperature"], dt_seconds)
        if "humidity" in out:
            out["humidity"] = self._step_humidity(room, world, out["humidity"], dt_seconds)
        if "pm10" in out:
            out["pm10"] = self._step_pm10(room, world, out["pm10"], dt_seconds)
        return out

    def _compute_illuminance(
        self,
        room: RoomBinding,
        world: Dict[str, Dict[str, Any]],
    ) -> float:
        value = room.baseline.get("illuminance", 0.0)
        for device in room.devices:
            data = world.get(device.device_id) or {}
            primary = data.get("primary") or {}
            if device.device_type == "on_off_light":
                if _entity_is_on(primary):
                    value += 500.0
            elif device.device_type == "dimmable_light":
                value += _light_level(primary) * 500.0
        return round(_clamp(value, 0.0, 5000.0), 3)

    def _step_temperature(
        self,
        room: RoomBinding,
        world: Dict[str, Dict[str, Any]],
        current: float,
        dt_seconds: float,
    ) -> float:
        value = current
        baseline = room.baseline.get("temperature", current)
        for device in room.devices:
            data = world.get(device.device_id) or {}
            primary = data.get("primary") or {}
            attrs = data.get("attrs") or {}
            if device.device_type == "fan":
                fan_intensity = _fan_percent(primary) / 100.0
                target_floor = baseline - 2.0
                if fan_intensity > 0.0 and value > target_floor:
                    value -= min(0.015, max(value - target_floor, 0.0) * 0.015) * fan_intensity * dt_seconds
            elif device.device_type in {"air_conditioner", "heat_pump"}:
                hvac_mode = self._read_climate_hvac_mode(
                    primary,
                    attrs,
                    default="cool" if device.device_type == "air_conditioner" else "heat",
                )
                on = hvac_mode != "off" or _entity_is_on(primary) or _sensor_on_off(primary)
                if hvac_mode == "cool":
                    mode = 3
                elif hvac_mode == "heat":
                    mode = 4
                elif hvac_mode == "off":
                    mode = 0
                else:
                    mode = self._read_system_mode(attrs, default=3 if device.device_type == "air_conditioner" else 4)
                cooling_sp = self._read_numeric_attr_suffix(
                    attrs,
                    ".Thermostat.OccupiedCoolingSetpoint",
                    default=self._read_climate_temperature(primary, default=baseline - 2.0),
                )
                heating_sp = self._read_numeric_attr_suffix(
                    attrs,
                    ".Thermostat.OccupiedHeatingSetpoint",
                    default=self._read_climate_temperature(primary, default=baseline + 2.0),
                )
                fan_intensity = self._read_numeric_attr_suffix(
                    attrs,
                    ".FanControl.PercentSetting",
                    default=self._read_numeric_attr_suffix(attrs, ".FanControl.PercentCurrent", default=70.0),
                ) / 100.0
                if on and fan_intensity <= 0.0:
                    fan_intensity = 0.7
                if on and mode == 3 and value > cooling_sp:
                    value -= min(0.035, max(value - cooling_sp, 0.0) * 0.03) * fan_intensity * dt_seconds
                elif on and mode == 4 and value < heating_sp:
                    value += min(0.035, max(heating_sp - value, 0.0) * 0.03) * fan_intensity * dt_seconds

        value += (baseline - value) * 0.0002 * dt_seconds
        return round(_clamp(value, 5.0, 40.0), 3)

    def _step_humidity(
        self,
        room: RoomBinding,
        world: Dict[str, Dict[str, Any]],
        current: float,
        dt_seconds: float,
    ) -> float:
        value = current
        baseline = room.baseline.get("humidity", current)
        for device in room.devices:
            data = world.get(device.device_id) or {}
            primary = data.get("primary") or {}
            if device.device_type == "humidifier":
                fan_intensity = _fan_percent(primary) / 100.0
                if fan_intensity > 0.0 and value < 90.0:
                    efficiency = max(0.1, (90.0 - value) / 90.0)
                    value += 0.05 * fan_intensity * efficiency * dt_seconds
            elif device.device_type == "dehumidifier":
                fan_intensity = _fan_percent(primary) / 100.0
                if fan_intensity > 0.0 and value > 10.0:
                    efficiency = max(0.1, (value - 10.0) / 90.0)
                    value -= 0.05 * fan_intensity * efficiency * dt_seconds

        value += (baseline - value) * 0.01 * dt_seconds
        return round(_clamp(value, 0.0, 100.0), 3)

    def _step_pm10(
        self,
        room: RoomBinding,
        world: Dict[str, Dict[str, Any]],
        current: float,
        dt_seconds: float,
    ) -> float:
        value = current
        baseline = max(room.baseline.get("pm10", current), 0.001)
        for device in room.devices:
            if device.device_type != "air_purifier":
                continue
            data = world.get(device.device_id) or {}
            fan_intensity = (_fan_percent(data.get("primary") or {}) / 100.0) ** 0.8
            if fan_intensity <= 0.0:
                continue
            concentration_ratio = value / baseline
            pollution_factor = concentration_ratio * 0.8 if concentration_ratio > 2.0 else min(2.0, concentration_ratio)
            value -= 0.05 * fan_intensity * pollution_factor * dt_seconds

        value += (baseline - value) * 0.1 * dt_seconds
        return round(max(0.0, value), 3)

    def _read_system_mode(self, attrs: Dict[str, Dict[str, Any]], *, default: int) -> int:
        return int(round(self._read_numeric_attr_suffix(attrs, ".Thermostat.SystemMode", default=default)))

    def _read_climate_hvac_mode(
        self,
        state: Dict[str, Any],
        attrs: Dict[str, Dict[str, Any]],
        *,
        default: str,
    ) -> str:
        ha_attrs = (state or {}).get("attributes") or {}
        mode = ha_attrs.get("hvac_mode") or (state or {}).get("state")
        if isinstance(mode, str) and mode.lower() in {"off", "cool", "heat", "auto"}:
            return mode.lower()

        system_mode = self._read_numeric_attr_suffix(attrs, ".Thermostat.SystemMode", default=-1)
        if system_mode == 0:
            return "off"
        if system_mode == 3:
            return "cool"
        if system_mode == 4:
            return "heat"
        if system_mode == 1:
            return "auto"
        return default

    def _read_climate_temperature(self, state: Dict[str, Any], *, default: float) -> float:
        attrs = (state or {}).get("attributes") or {}
        for key in ("temperature", "target_temp_low", "target_temp_high"):
            if key in attrs:
                return _as_float(attrs[key], default)
        return default

    def _read_numeric_attr_suffix(
        self,
        attrs: Dict[str, Dict[str, Any]],
        attr_suffix: str,
        *,
        default: float,
    ) -> float:
        for attr_path, state in attrs.items():
            if attr_path.endswith(attr_suffix):
                return _sensor_number(state, default)
        return default

    async def _write_room_state(self, room: RoomBinding) -> None:
        for state_name, entity_id in room.env_entity_ids.items():
            if state_name not in room.current:
                continue
            attrs: Dict[str, Any] = {}
            if state_name in ROOM_STATE_UNITS:
                attrs["unit_of_measurement"] = ROOM_STATE_UNITS[state_name]
            if state_name in ROOM_STATE_CLASSES:
                attrs["device_class"] = ROOM_STATE_CLASSES[state_name]
            await self._set_entity(entity_id, room.current[state_name], attrs)

        for device in room.devices:
            for attr_path, entity_id in device.attr_entity_ids.items():
                if attr_path.endswith(".Thermostat.LocalTemperature") and "temperature" in room.current:
                    await self._set_entity(entity_id, room.current["temperature"], {"unit_of_measurement": "°C"})
                if attr_path.endswith(".RelativeHumidityMeasurement.MeasuredValue") and "humidity" in room.current:
                    await self._set_entity(entity_id, room.current["humidity"], {"unit_of_measurement": "%"})


async def _ws_handshake(url: str, token: str):
    ws = await websockets.connect(url)
    msg = json.loads(await ws.recv())
    if msg.get("type") != "auth_required":
        raise RuntimeError("Unexpected Home Assistant WebSocket handshake")
    await ws.send(json.dumps({"type": "auth", "access_token": token}))
    msg = json.loads(await ws.recv())
    if msg.get("type") != "auth_ok":
        raise RuntimeError("Home Assistant authentication failed")
    return ws


def _relevant_entity_ids(scenario: ScenarioBinding) -> set[str]:
    ids: set[str] = set()
    for room in scenario.rooms.values():
        for device in room.devices:
            if device.device_type not in ACTUATOR_TYPES:
                continue
            ids.add(device.primary_entity_id)
            for attr_path, entity_id in device.attr_entity_ids.items():
                if any(attr_path.endswith(suffix) for suffix in CONTROL_ATTR_SUFFIXES):
                    ids.add(entity_id)
    return ids


async def _event_listener(
    ha_url: str,
    ha_token: str,
    relevant_entity_ids: set[str],
    queue: asyncio.Queue[str],
) -> None:
    ws = await _ws_handshake(ha_url, ha_token)
    try:
        await ws.send(json.dumps({"id": 1, "type": "subscribe_events", "event_type": "state_changed"}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") != "event":
                continue
            data = ((msg.get("event") or {}).get("data") or {})
            entity_id = str(data.get("entity_id") or "")
            if entity_id not in relevant_entity_ids:
                continue
            await queue.put(f"event:{entity_id}")
    finally:
        await ws.close()


async def _periodic_trigger(queue: asyncio.Queue[str], tick_seconds: float) -> None:
    while True:
        await asyncio.sleep(tick_seconds)
        await queue.put("tick")


def _load_json_or_die(path: Path) -> ScenarioBinding:
    if not path.is_file():
        raise FileNotFoundError(f"Scenario JSON not found: {path}")
    return _load_scenario(path)


def _format_summary(results: Dict[str, Dict[str, float]]) -> str:
    parts: list[str] = []
    for room_id, state in sorted(results.items()):
        brief = ",".join(f"{key}={value}" for key, value in sorted(state.items()))
        parts.append(f"{room_id}[{brief}]")
    return " ".join(parts)


async def _run(args: argparse.Namespace) -> None:
    scenario = _load_json_or_die(args.scenario.expanduser().resolve())
    tick_seconds = args.tick_seconds if args.tick_seconds is not None else scenario.tick_interval
    ha_url = _env("HA_URL")
    ha_token = _env("HA_TOKEN")
    base_url = _base_url(ha_url)
    headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}

    relevant = _relevant_entity_ids(scenario)
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=args.http_timeout) as client:
        sim = SimuHomeSidecar(
            client,
            scenario,
            tick_seconds=tick_seconds,
            speed=args.speed,
            ha_sync_seconds=args.ha_sync_seconds,
        )
        queue: asyncio.Queue[str] = asyncio.Queue()
        listener_task = asyncio.create_task(_event_listener(ha_url, ha_token, relevant, queue))
        periodic_task = asyncio.create_task(_periodic_trigger(queue, tick_seconds))
        await queue.put("startup")
        print(
            f"Simulating {scenario.name}; rooms={len(scenario.rooms)} "
            f"relevant_entities={len(relevant)} tick={tick_seconds:.2f}s speed={args.speed:.2f}x"
        )
        try:
            while True:
                reason = await queue.get()
                while not queue.empty():
                    reason = await queue.get()
                results = await sim.tick_once(force_ha_sync=reason.startswith("event:") or reason == "startup")
                print(f"{scenario.name} reason={reason} {_format_summary(results)}")
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            listener_task.cancel()
            periodic_task.cancel()
            await asyncio.gather(listener_task, periodic_task, return_exceptions=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Home Assistant sidecar simulator for one SimuHome benchmark JSON."
    )
    parser.add_argument("scenario", type=Path, help="Path to a SimuHome benchmark JSON file")
    parser.add_argument(
        "--tick-seconds",
        type=float,
        default=float(os.environ["SIMUHOME_TICK_SECONDS"]) if os.getenv("SIMUHOME_TICK_SECONDS") else None,
        help=(
            "Real seconds between periodic updates; default: env SIMUHOME_TICK_SECONDS "
            "or the scenario tick_interval"
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=float(os.getenv("SIMUHOME_SIM_SPEED", "1")),
        help="Simulation seconds per real second; default: env SIMUHOME_SIM_SPEED or 1",
    )
    parser.add_argument(
        "--ha-sync-seconds",
        type=float,
        default=float(os.getenv("SIMUHOME_HA_SYNC_SECONDS", "1")),
        help="Minimum real seconds between Home Assistant state refresh/write cycles; default: env SIMUHOME_HA_SYNC_SECONDS or 1",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=10.0,
        help="Home Assistant HTTP timeout in seconds",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
