#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


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


def _group_key(*parts: str) -> str:
    return "_".join(_slug(part) for part in parts if part)


def _room_token(room_id: str) -> str:
    return _dotted_token(room_id)


def _device_token(device_id: str, room_id: str) -> str:
    prefix = f"{room_id}_"
    core = device_id[len(prefix) :] if device_id.startswith(prefix) else device_id
    return _dotted_token(core)


def _room_state_unit(state_name: str) -> str | None:
    return {
        "temperature": "°C",
        "humidity": "%",
        "illuminance": "lx",
        "pm10": "ug/m3",
    }.get(state_name)


def _room_state_class(state_name: str) -> str | None:
    return {
        "temperature": "temperature",
        "humidity": "humidity",
        "illuminance": "illuminance",
        "pm10": "pm10",
    }.get(state_name)


def _room_state_value(state_name: str, raw_value: Any) -> Any:
    if not isinstance(raw_value, (int, float)):
        return raw_value
    if state_name in {"temperature", "humidity"}:
        return round(float(raw_value) / 100.0, 2)
    if state_name == "illuminance":
        return round(float(raw_value), 2)
    if state_name == "pm10":
        return round(float(raw_value), 3)
    return raw_value


def _attribute_unit(cluster_id: str, attribute_id: str) -> str | None:
    key = f"{cluster_id}.{attribute_id}"
    return {
        "Thermostat.LocalTemperature": "°C",
        "Thermostat.OccupiedCoolingSetpoint": "°C",
        "Thermostat.OccupiedHeatingSetpoint": "°C",
        "TemperatureControl.MaxTemperature": "°C",
        "TemperatureControl.MinTemperature": "°C",
        "TemperatureControl.TemperatureSetpoint": "°C",
        "RelativeHumidityMeasurement.MeasuredValue": "%",
        "RelativeHumidityMeasurement.MaxMeasuredValue": "%",
        "RelativeHumidityMeasurement.MinMeasuredValue": "%",
        "RelativeHumidityMeasurement.Tolerance": "%",
        "WindowCovering.CurrentPositionLiftPercent100ths": "%",
        "WindowCovering.TargetPositionLiftPercent100ths": "%",
        "FanControl.PercentCurrent": "%",
        "FanControl.PercentSetting": "%",
    }.get(key)


def _attribute_class(cluster_id: str, attribute_id: str) -> str | None:
    key = f"{cluster_id}.{attribute_id}"
    if "Temperature" in key:
        return "temperature"
    if "Humidity" in key:
        return "humidity"
    if "Illuminance" in key:
        return "illuminance"
    if "Pm10" in key or "PM10" in key:
        return "pm10"
    return None


def _attribute_value(cluster_id: str, attribute_id: str, raw_value: Any) -> Any:
    if not isinstance(raw_value, (int, float)):
        return raw_value

    key = f"{cluster_id}.{attribute_id}"
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
    }:
        return round(float(raw_value) / 100.0, 2)

    if key in {
        "WindowCovering.CurrentPositionLiftPercent100ths",
        "WindowCovering.TargetPositionLiftPercent100ths",
    }:
        return round(float(raw_value) / 100.0, 2)

    if key in {"FanControl.PercentCurrent", "FanControl.PercentSetting"}:
        return round(float(raw_value), 2)

    return raw_value


def _cover_position_from_attrs(attributes: dict[str, Any]) -> Any:
    raw = attributes.get("1.WindowCovering.CurrentPositionLiftPercent100ths")
    if isinstance(raw, (int, float)):
        return round(float(raw) / 100.0, 2)
    return 0


def _light_brightness_from_attrs(attributes: dict[str, Any]) -> int | None:
    if isinstance(attributes.get("1.LevelControl.CurrentLevel"), (int, float)):
        level = float(attributes["1.LevelControl.CurrentLevel"])
        return max(0, min(255, int(round(level))))
    if isinstance(attributes.get("1.FanControl.PercentSetting"), (int, float)):
        pct = float(attributes["1.FanControl.PercentSetting"])
        return max(0, min(255, int(round((pct / 100.0) * 255))))
    return None


def _matter_temp_c(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value) / 100.0, 2)
    return None


def _hvac_mode_from_attrs(attributes: dict[str, Any], *, device_type: str) -> str:
    system_mode = attributes.get("1.Thermostat.SystemMode")
    if device_type == "heat_pump":
        system_mode = attributes.get("4.Thermostat.SystemMode", system_mode)

    if system_mode == 0:
        return "off"
    if system_mode == 3:
        return "cool"
    if system_mode == 4:
        return "heat"
    if system_mode == 1:
        return "auto"
    if attributes.get("1.OnOff.OnOff") is False:
        return "off"
    return "cool" if device_type == "air_conditioner" else "heat"


def _climate_temperature_from_attrs(attributes: dict[str, Any], *, device_type: str) -> float:
    if device_type == "heat_pump":
        mode = _hvac_mode_from_attrs(attributes, device_type=device_type)
        if mode == "cool":
            for key in ("4.Thermostat.OccupiedCoolingSetpoint", "1.Thermostat.OccupiedCoolingSetpoint"):
                value = _matter_temp_c(attributes.get(key))
                if value is not None:
                    return value
        for key in ("4.Thermostat.OccupiedHeatingSetpoint", "1.Thermostat.OccupiedHeatingSetpoint"):
            value = _matter_temp_c(attributes.get(key))
            if value is not None:
                return value
        value = _matter_temp_c(attributes.get("4.Thermostat.LocalTemperature"))
        if value is not None:
            return value
        return 21.0

    mode = _hvac_mode_from_attrs(attributes, device_type=device_type)
    if mode == "heat":
        value = _matter_temp_c(attributes.get("1.Thermostat.OccupiedHeatingSetpoint"))
        if value is not None:
            return value
    value = _matter_temp_c(attributes.get("1.Thermostat.OccupiedCoolingSetpoint"))
    if value is not None:
        return value
    value = _matter_temp_c(attributes.get("1.Thermostat.LocalTemperature"))
    if value is not None:
        return value
    return 22.0


def _climate_limits_from_attrs(attributes: dict[str, Any], *, device_type: str) -> tuple[float, float]:
    candidates: list[float] = []
    keys = [
        "1.Thermostat.OccupiedCoolingSetpoint",
        "1.Thermostat.OccupiedHeatingSetpoint",
        "1.Thermostat.LocalTemperature",
    ]
    if device_type == "heat_pump":
        keys.extend(
            [
                "4.Thermostat.OccupiedCoolingSetpoint",
                "4.Thermostat.OccupiedHeatingSetpoint",
                "4.Thermostat.LocalTemperature",
            ]
        )
    for key in keys:
        value = _matter_temp_c(attributes.get(key))
        if value is not None:
            candidates.append(value)

    if not candidates:
        return (16.0, 30.0)
    low = min(candidates) - 8.0
    high = max(candidates) + 8.0
    return (max(5.0, round(low, 1)), min(40.0, round(high, 1)))


def _bool_to_on_off(value: Any) -> str:
    return "on" if bool(value) else "off"


def _scalar_sensor_entry(
    *,
    name: str,
    initial_value: Any,
    sensor_class: str | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "platform": "sensor",
        "name": name,
        "initial_value": initial_value,
    }
    if sensor_class:
        entry["class"] = sensor_class
    if unit:
        entry["unit_of_measurement"] = unit
    return entry


def _raw_attribute_sensor_entry(
    *,
    entity_name: str,
    cluster_id: str,
    attribute_id: str,
    raw_value: Any,
) -> dict[str, Any]:
    converted = _attribute_value(cluster_id, attribute_id, raw_value)
    if isinstance(converted, (dict, list)):
        converted = json.dumps(converted, ensure_ascii=True, sort_keys=True)
    if converted is None:
        converted = "unknown"
    if isinstance(converted, bool):
        converted = _bool_to_on_off(converted)
    return _scalar_sensor_entry(
        name=entity_name,
        initial_value=converted,
        sensor_class=_attribute_class(cluster_id, attribute_id),
        unit=_attribute_unit(cluster_id, attribute_id),
    )


def _device_primary_entry(
    *,
    device_type: str,
    entity_name: str,
    attributes: dict[str, Any],
) -> dict[str, Any] | None:
    on_off = attributes.get("1.OnOff.OnOff")

    if device_type in {"on_off_light", "dimmable_light"}:
        entry: dict[str, Any] = {
            "platform": "light",
            "name": entity_name,
            "initial_value": _bool_to_on_off(on_off),
        }
        brightness = _light_brightness_from_attrs(attributes)
        if device_type == "dimmable_light":
            entry["support_brightness"] = True
            entry["initial_brightness"] = brightness if brightness is not None else 128
        return entry

    if device_type in {"fan", "air_purifier", "humidifier", "dehumidifier"}:
        # The HA virtual fan platform rejects unknown config keys (such as
        # initial_percentage), which silently kills every fan entity in the
        # group. speed_count enables the set_percentage feature instead.
        return {
            "platform": "fan",
            "name": entity_name,
            "initial_value": _bool_to_on_off(on_off),
            "speed_count": 100,
        }

    if device_type == "window_covering_controller":
        return {
            "platform": "cover",
            "name": entity_name,
            "initial_position": _cover_position_from_attrs(attributes),
        }

    if device_type in {"air_conditioner", "heat_pump"}:
        min_temp, max_temp = _climate_limits_from_attrs(attributes, device_type=device_type)
        return {
            "platform": "climate",
            "name": entity_name,
            "initial_hvac_mode": _hvac_mode_from_attrs(attributes, device_type=device_type),
            "initial_temperature": _climate_temperature_from_attrs(attributes, device_type=device_type),
            "min_temp": min_temp,
            "max_temp": max_temp,
            "target_temp_step": 0.5,
        }

    if device_type in {"freezer", "refrigerator"}:
        return {
            "platform": "sensor",
            "name": entity_name,
            "initial_value": _bool_to_on_off(on_off),
        }

    if device_type in {"dishwasher", "laundry_washer", "laundry_dryer", "tv", "rvc"}:
        return {
            "platform": "sensor",
            "name": entity_name,
            "initial_value": _bool_to_on_off(on_off),
        }

    return None


def _convert_initial_home_config(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    initial_home_config = payload.get("initial_home_config")
    if not isinstance(initial_home_config, dict):
        raise ValueError("JSON file does not contain an object at 'initial_home_config'")

    scenario = _scenario_token(source_path)
    scenario_key = _slug(source_path.stem)
    devices: dict[str, list[dict[str, Any]]] = {}

    base_time = initial_home_config.get("base_time")
    if base_time is not None:
        devices[_group_key(scenario_key, "clock")] = [
            {
                "platform": "sensor",
                "name": f"{scenario}.Clock",
                "class": "timestamp",
                "initial_value": str(base_time),
            }
        ]

    tick_interval = initial_home_config.get("tick_interval")
    if tick_interval is not None:
        devices[_group_key(scenario_key, "simulator")] = [
            _scalar_sensor_entry(
                name=f"{scenario}.Simulator.TickInterval",
                initial_value=tick_interval,
            )
        ]

    rooms = initial_home_config.get("rooms")
    if not isinstance(rooms, dict):
        raise ValueError("'initial_home_config.rooms' must be an object")

    for room_id, room_cfg in rooms.items():
        if not isinstance(room_cfg, dict):
            continue

        room_token = _room_token(room_id)

        state_cfg = room_cfg.get("state")
        if isinstance(state_cfg, dict):
            env_group_key = _group_key(scenario_key, room_id, "environment")
            env_entries = devices.setdefault(env_group_key, [])
            for state_name, raw_value in state_cfg.items():
                env_entries.append(
                    _scalar_sensor_entry(
                        name=f"{scenario}.{room_token}.{_dotted_token(state_name)}",
                        initial_value=_room_state_value(state_name, raw_value),
                        sensor_class=_room_state_class(state_name),
                        unit=_room_state_unit(state_name),
                    )
                )

        device_list = room_cfg.get("devices")
        if not isinstance(device_list, list):
            continue

        for device in device_list:
            if not isinstance(device, dict):
                continue

            device_id = str(device.get("device_id", "")).strip()
            device_type = str(device.get("device_type", "")).strip()
            attributes = device.get("attributes")
            if not device_id or not isinstance(attributes, dict):
                continue

            device_token = _device_token(device_id, room_id)
            entity_name = f"{scenario}.{room_token}.{device_token}"
            group_key = _group_key(scenario_key, room_id, device_id)
            entries = devices.setdefault(group_key, [])

            primary = _device_primary_entry(
                device_type=device_type,
                entity_name=entity_name,
                attributes=attributes,
            )
            if primary is not None:
                entries.append(primary)

            for attr_path, raw_value in attributes.items():
                if not isinstance(attr_path, str):
                    continue
                parts = attr_path.split(".", 2)
                if len(parts) == 3:
                    _, cluster_id, attribute_id = parts
                else:
                    cluster_id = "Attribute"
                    attribute_id = attr_path

                attr_entity_name = (
                    f"{entity_name}.{_dotted_token(cluster_id)}.{_dotted_token(attribute_id)}"
                )
                entries.append(
                    _raw_attribute_sensor_entry(
                        entity_name=attr_entity_name,
                        cluster_id=cluster_id,
                        attribute_id=attribute_id,
                        raw_value=raw_value,
                    )
                )

    # Virtual entities default to persistent: true and restore the previous
    # run's state on re-import, polluting the episode's initial conditions.
    for entries in devices.values():
        for entry in entries:
            entry["persistent"] = False

    return {
        "version": 1,
        "devices": devices,
    }


def _process_json_file(path: Path) -> tuple[bool, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"{path}: failed to read JSON: {exc}"

    if not isinstance(payload, dict):
        return False, f"{path}: top-level JSON value must be an object"

    try:
        ha_yaml = _convert_initial_home_config(payload, path)
    except Exception as exc:
        return False, f"{path}: {exc}"

    out_path = path.with_suffix(".yaml")
    out_path.write_text(
        yaml.safe_dump(
            ha_yaml,
            sort_keys=False,
            allow_unicode=False,
            width=120,
        ),
        encoding="utf-8",
    )
    return True, f"{path} -> {out_path}"


def _walk_json_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SimuHome benchmark JSON files into Home Assistant-style YAML files "
            "using each file's initial_home_config."
        )
    )
    parser.add_argument("directory", type=Path, help="Directory to walk recursively")
    args = parser.parse_args()

    root = args.directory.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"directory not found: {root}")

    json_files = _walk_json_files(root)
    if not json_files:
        print(f"No JSON files found under {root}")
        return 0

    ok_count = 0
    skipped_count = 0
    for path in json_files:
        ok, message = _process_json_file(path)
        print(message)
        if ok:
            ok_count += 1
        else:
            skipped_count += 1

    print(
        f"Finished. Generated {ok_count} YAML file(s); skipped {skipped_count} JSON file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
