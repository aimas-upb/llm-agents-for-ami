#!/usr/bin/env python3
"""
Hardcoded simulator loop for the lab308e virtual workspace.

What it does:
- updates the clock sensor
- computes outdoor light from real time of day
- reads current actuator/device states from Home Assistant
- updates the derived light sensors based on lights, blinds, blackout blinds,
  and projector/display usage
- updates temperature, humidity, and CO2 based on occupancy and HVAC/window state
- recomputes immediately when relevant Home Assistant entities change

Required environment:
- HA_URL    WebSocket URL, e.g. ws://localhost:8123/api/websocket
- HA_TOKEN  Long-lived access token

Optional environment:
- LAB308E_TICK_SECONDS   default 5

Notes:
- This script is intentionally hardcoded to the entity names created by
  lab308e.yaml.
- It updates only sensors/binary sensors/clock-like values; actuators remain
  under Home Assistant / agent control.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
import websockets


ENTITY = {
    "clock": "sensor.clock_308e",
    "ambient_lights": "light.ambient_lights_308e",
    "task_lights": "light.task_lights_308e",
    "desk_lamp": "light.desk_lamp_308e",
    "blinds": "cover.blinds_308e_cover",
    "blackout_blinds": "cover.blackout_blinds_308e_cover",
    "window": "cover.window_308e_cover",
    "projector": "media_player.projector_308e",
    "display_wall": "media_player.display_wall_308e",
    "ceiling_fan": "switch.ceiling_fan_308e",
    "air_conditioner": "climate.air_conditioner_308e",
    "heater": "climate.heater_308e",
    "internal_lux": "sensor.internal_light_sensing_308e",
    "desk_lux": "sensor.desk_light_sensing_308e",
    "glare": "sensor.glare_sensing_308e",
    "external_lux": "sensor.external_light_sensing_308e",
    "temperature": "sensor.temperature_sensing_308e",
    "humidity": "sensor.humidity_sensing_308e",
    "co2": "sensor.co2_sensing_308e",
    "person_counter": "sensor.person_counter_308e",
    "presence": "binary_sensor.presence_sensing_308e",
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


def _iso_utc(timestamp: dt.datetime) -> str:
    return timestamp.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _entity_is_on(state: Dict[str, Any]) -> bool:
    return str((state or {}).get("state", "")).lower() in {"on", "open", "opening", "heat", "cool"}


def _cover_open_fraction(state: Dict[str, Any]) -> float:
    attrs = (state or {}).get("attributes") or {}
    raw = attrs.get("current_position")
    if raw is None:
        current = str((state or {}).get("state", "")).lower()
        if current == "open":
            return 1.0
        if current == "closed":
            return 0.0
        return 0.5
    return _clamp(_as_float(raw, 0.0) / 100.0, 0.0, 1.0)


def _light_level(state: Dict[str, Any], fallback_pct: float = 0.0) -> float:
    if not _entity_is_on(state):
        return 0.0
    attrs = (state or {}).get("attributes") or {}
    if "brightness" in attrs:
        return _clamp(_as_float(attrs["brightness"]) / 255.0, 0.0, 1.0)
    if "brightness_pct" in attrs:
        return _clamp(_as_float(attrs["brightness_pct"]) / 100.0, 0.0, 1.0)
    return _clamp(fallback_pct, 0.0, 1.0)


def _media_is_on(state: Dict[str, Any]) -> bool:
    current = str((state or {}).get("state", "")).lower()
    return current not in {"off", "idle", "unavailable", "unknown", ""}


def _hvac_mode(state: Dict[str, Any]) -> str:
    attrs = (state or {}).get("attributes") or {}
    mode = attrs.get("hvac_mode")
    if mode is not None:
        return str(mode).lower()
    return str((state or {}).get("state", "")).lower()


@dataclass
class Lab308eState:
    temperature_c: float = 26.0
    humidity_pct: float = 62.0
    co2_ppm: float = 980.0
    last_update: Optional[dt.datetime] = None


class Lab308eSimulator:
    def __init__(self, client: httpx.AsyncClient, tick_seconds: float):
        self.client = client
        self.state = Lab308eState()
        self.tick_seconds = tick_seconds

    async def _get_state(self, entity_id: str) -> Dict[str, Any]:
        response = await self.client.get(f"/api/states/{entity_id}")
        response.raise_for_status()
        return response.json()

    async def _set_entity(self, entity_id: str, state: Any, attributes: Optional[Dict[str, Any]] = None) -> None:
        current = await self._get_state(entity_id)
        merged_attrs = dict(current.get("attributes") or {})
        if attributes:
            merged_attrs.update(attributes)
        payload = {"state": state, "attributes": merged_attrs}
        response = await self.client.post(f"/api/states/{entity_id}", json=payload)
        response.raise_for_status()

    async def _read_world(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for key, entity_id in ENTITY.items():
            try:
                result[key] = await self._get_state(entity_id)
            except Exception:
                result[key] = {"state": "unknown", "attributes": {}}
        return result

    def _external_lux_for_time(self, now: dt.datetime) -> float:
        hour = now.hour + (now.minute / 60.0) + (now.second / 3600.0)
        solar = math.sin(((hour - 6.0) / 12.0) * math.pi)
        solar = max(0.0, solar)
        cloud_factor = 0.85 + 0.15 * math.sin(((hour + 1.5) / 24.0) * 2.0 * math.pi)
        return round(150.0 + solar * 8500.0 * cloud_factor, 1)

    def _compute_targets(self, world: Dict[str, Dict[str, Any]], now: dt.datetime) -> Dict[str, Any]:
        external_lux = self._external_lux_for_time(now)

        blinds_open = _cover_open_fraction(world["blinds"])
        blackout_open = _cover_open_fraction(world["blackout_blinds"])
        window_open = _cover_open_fraction(world["window"])

        ambient_level = _light_level(world["ambient_lights"], fallback_pct=0.5)
        task_level = _light_level(world["task_lights"], fallback_pct=0.6)
        desk_level = _light_level(world["desk_lamp"], fallback_pct=0.7)

        projector_on = _media_is_on(world["projector"])
        display_on = _media_is_on(world["display_wall"])

        daylight_room = external_lux * (0.55 * blinds_open + 0.08 * blackout_open)
        daylight_desk = external_lux * (0.35 * blinds_open + 0.05 * blackout_open)

        internal_lux = (
            40.0
            + daylight_room
            + ambient_level * 420.0
            + task_level * 300.0
            + desk_level * 80.0
            - (140.0 if projector_on else 0.0)
        )

        desk_lux = (
            25.0
            + daylight_desk
            + ambient_level * 180.0
            + task_level * 220.0
            + desk_level * 520.0
            - (60.0 if projector_on else 0.0)
        )

        # Model glare as strongly driven by outdoor light and screen use, with
        # aggressive reduction when covers are closed. This makes blackout
        # blinds the dominant anti-glare actuator in the lab308e scenario.
        glare = (
            3.0
            + external_lux / 180.0
            + blinds_open * 18.0
            + blackout_open * 20.0
            + window_open * 14.0
            + (14.0 if display_on else 0.0)
            + (22.0 if projector_on else 0.0)
            - (1.0 - blinds_open) * 12.0
            - (1.0 - blackout_open) * 38.0
            - (1.0 - window_open) * 8.0
            - ambient_level * 6.0
        )

        outside_temp = 19.0 + 8.0 * math.sin((((now.hour + now.minute / 60.0) - 8.0) / 24.0) * 2.0 * math.pi)
        outside_humidity = 66.0 - 12.0 * math.sin((((now.hour + now.minute / 60.0) - 7.0) / 24.0) * 2.0 * math.pi)

        heater_mode = _hvac_mode(world["heater"])
        ac_mode = _hvac_mode(world["air_conditioner"])
        fan_on = _entity_is_on(world["ceiling_fan"])
        people = max(0.0, _as_float(world["person_counter"].get("state"), 0.0))
        presence = str(world["presence"].get("state", "")).lower() == "on"

        # Allow tests to seed the derived environment sensors directly via
        # Home Assistant state updates; the simulator picks up those values as
        # its starting point for the next dynamics step.
        self.state.temperature_c = _as_float(world["temperature"].get("state"), self.state.temperature_c)
        self.state.humidity_pct = _as_float(world["humidity"].get("state"), self.state.humidity_pct)
        self.state.co2_ppm = _as_float(world["co2"].get("state"), self.state.co2_ppm)

        elapsed_seconds = self.tick_seconds
        if self.state.last_update is not None:
            elapsed_seconds = max(0.5, (now - self.state.last_update).total_seconds())
        time_factor = elapsed_seconds / max(self.tick_seconds, 1.0)

        occupancy_heat = people * 0.12 + (0.25 if projector_on else 0.0) + (0.18 if display_on else 0.0)
        temp = self.state.temperature_c
        temp += (outside_temp - temp) * (0.04 + 0.12 * window_open) * time_factor
        if heater_mode in {"heat", "heating"}:
            temp += 0.22 * time_factor
        if ac_mode in {"cool", "cooling"}:
            temp -= 0.28 * time_factor
        if fan_on:
            temp -= 0.08 * time_factor
        temp += occupancy_heat * time_factor
        temp = _clamp(temp, 17.0, 30.0)

        humidity = self.state.humidity_pct
        humidity += (outside_humidity - humidity) * (0.03 + 0.12 * window_open) * time_factor
        humidity += people * 0.35 * time_factor
        if ac_mode in {"cool", "cooling"}:
            humidity -= 0.6 * time_factor
        if heater_mode in {"heat", "heating"}:
            humidity -= 0.25 * time_factor
        humidity = _clamp(humidity, 30.0, 85.0)

        co2 = self.state.co2_ppm
        co2 += people * 18.0 * time_factor if presence else 0.0
        # Ventilation through the motorized window should noticeably improve
        # stale air even with a few occupants present.
        co2 -= (14.0 + 120.0 * window_open) * time_factor
        if fan_on:
            co2 -= 6.0 * time_factor
        co2 = _clamp(co2, 420.0, 2200.0)

        self.state.temperature_c = temp
        self.state.humidity_pct = humidity
        self.state.co2_ppm = co2
        self.state.last_update = now

        return {
            "clock": _iso_utc(now),
            "external_lux": round(_clamp(external_lux, 0.0, 10000.0), 1),
            "internal_lux": round(_clamp(internal_lux, 0.0, 3000.0), 1),
            "desk_lux": round(_clamp(desk_lux, 0.0, 2500.0), 1),
            "glare": round(_clamp(glare, 0.0, 100.0), 1),
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "co2": round(co2, 0),
        }

    async def tick_once(self) -> Dict[str, Any]:
        world = await self._read_world()
        now = dt.datetime.now().astimezone().replace(microsecond=0)
        targets = self._compute_targets(world, now)

        await self._set_entity(ENTITY["clock"], targets["clock"])
        await self._set_entity(ENTITY["external_lux"], targets["external_lux"])
        await self._set_entity(ENTITY["internal_lux"], targets["internal_lux"])
        await self._set_entity(ENTITY["desk_lux"], targets["desk_lux"])
        await self._set_entity(ENTITY["glare"], targets["glare"])
        await self._set_entity(ENTITY["temperature"], targets["temperature"])
        await self._set_entity(ENTITY["humidity"], targets["humidity"])
        await self._set_entity(ENTITY["co2"], targets["co2"])
        return targets


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


def _relevant_entity_ids() -> set[str]:
    return {
        ENTITY["ambient_lights"],
        ENTITY["task_lights"],
        ENTITY["desk_lamp"],
        ENTITY["blinds"],
        ENTITY["blackout_blinds"],
        ENTITY["window"],
        ENTITY["projector"],
        ENTITY["display_wall"],
        ENTITY["ceiling_fan"],
        ENTITY["air_conditioner"],
        ENTITY["heater"],
        ENTITY["person_counter"],
        ENTITY["presence"],
    }


async def _event_listener(ha_url: str, ha_token: str, queue: asyncio.Queue[str]) -> None:
    relevant = _relevant_entity_ids()
    ws = await _ws_handshake(ha_url, ha_token)
    try:
        await ws.send(json.dumps({"id": 1, "type": "subscribe_events", "event_type": "state_changed"}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") != "event":
                continue
            data = ((msg.get("event") or {}).get("data") or {})
            entity_id = str(data.get("entity_id") or "")
            if entity_id not in relevant:
                continue
            await queue.put(f"event:{entity_id}")
    finally:
        await ws.close()


async def _periodic_trigger(queue: asyncio.Queue[str], tick_seconds: float) -> None:
    while True:
        await asyncio.sleep(tick_seconds)
        await queue.put("tick")


async def main() -> None:
    ha_url = _env("HA_URL")
    ha_token = _env("HA_TOKEN")
    base_url = _base_url(ha_url)
    tick_seconds = float(os.getenv("LAB308E_TICK_SECONDS", "5"))

    headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=10.0) as client:
        sim = Lab308eSimulator(client, tick_seconds=tick_seconds)
        queue: asyncio.Queue[str] = asyncio.Queue()
        listener_task = asyncio.create_task(_event_listener(ha_url, ha_token, queue))
        periodic_task = asyncio.create_task(_periodic_trigger(queue, tick_seconds))
        await queue.put("startup")
        print(f"Simulating lab308e in real time; periodic refresh every {tick_seconds:.1f}s. Ctrl+C to stop.")
        try:
            while True:
                reason = await queue.get()
                while not queue.empty():
                    reason = await queue.get()
                targets = await sim.tick_once()
                print(
                    "lab308e "
                    f"reason={reason} "
                    f"clock={targets['clock']} "
                    f"ext_lux={targets['external_lux']} "
                    f"int_lux={targets['internal_lux']} "
                    f"desk_lux={targets['desk_lux']} "
                    f"temp={targets['temperature']} "
                    f"humidity={targets['humidity']} "
                    f"co2={targets['co2']}"
                )
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            listener_task.cancel()
            periodic_task.cancel()
            await asyncio.gather(listener_task, periodic_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
