#!/usr/bin/env python3
"""
Convert SimuHome benchmark JSON into Home Assistant Virtual Devices YAML.

Matter facts — cluster ids, attribute ids, data types — are resolved from the
vendored CSA data model in matter_model/vendor/<VER>/ rather than hardcoded
here. Units and scaling follow each attribute's declared spec type, so e.g.
`temperature` (centi-degrees) and `percent100ths` (0..10000) are converted from
the type rather than from an attribute-name allowlist.

Run matter_model/bootstrap.py once before using this script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# The vendored Matter registry lives with the SHTD integration package, which is
# its primary consumer; this script is a second one.
_SHTD = (Path(__file__).resolve().parents[2]
         / "ami_agents" / "environment" / "integration" / "SimuHome")
if str(_SHTD) not in sys.path:
    sys.path.insert(0, str(_SHTD))

from matter_model.registry import load_registry  # noqa: E402

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


# Matter spec data types -> (Home Assistant unit, divisor applied to raw values).
# Keyed on the type declared for the attribute in the vendored data model, not
# on attribute names: RelativeHumidityMeasurement.MeasuredValue and
# Thermostat.LocalTemperature scale identically because the spec gives them the
# same underlying type family, and any new attribute of a known type is handled
# without touching this file.
_TYPE_UNITS: dict[str, tuple[str | None, float]] = {
    "temperature": ("°C", 100.0),          # centi-degrees Celsius
    "percent": ("%", 1.0),                 # already 0..100
    "percent100ths": ("%", 100.0),         # 0..10000
    "power-mW": ("mW", 1.0),
    "power-mVA": ("mVA", 1.0),
    "amperage-mA": ("mA", 1.0),
    "voltage-mV": ("mV", 1.0),
    "elapsed-s": ("s", 1.0),
}

# RelativeHumidityMeasurement stores hundredths of a percent; the spec types
# those attributes as plain uint16, so the scale comes from the cluster.
_HUNDREDTHS_PERCENT_CLUSTERS = {"RelativeHumidityMeasurement"}

# Home Assistant device classes, chosen from the Matter type where the type is
# unambiguous and from the cluster otherwise.
_TYPE_DEVICE_CLASSES: dict[str, str] = {
    "temperature": "temperature",
}
_CLUSTER_DEVICE_CLASSES: dict[str, str] = {
    "RelativeHumidityMeasurement": "humidity",
    "TemperatureMeasurement": "temperature",
    "IlluminanceMeasurement": "illuminance",
}


def _attribute_scale(cluster_id: str, attribute_id: str) -> tuple[str | None, float]:
    """Return (unit, divisor) for an attribute, resolved from the vendored spec."""
    registry = load_registry()
    if cluster_id in _HUNDREDTHS_PERCENT_CLUSTERS:
        return ("%", 100.0)
    attr_type = registry.attribute_type(cluster_id, attribute_id)
    if attr_type and attr_type in _TYPE_UNITS:
        return _TYPE_UNITS[attr_type]
    return (None, 1.0)


def _attribute_unit(cluster_id: str, attribute_id: str) -> str | None:
    return _attribute_scale(cluster_id, attribute_id)[0]


def _attribute_class(cluster_id: str, attribute_id: str) -> str | None:
    registry = load_registry()
    if cluster_id in _CLUSTER_DEVICE_CLASSES:
        return _CLUSTER_DEVICE_CLASSES[cluster_id]
    attr_type = registry.attribute_type(cluster_id, attribute_id)
    if attr_type and attr_type in _TYPE_DEVICE_CLASSES:
        return _TYPE_DEVICE_CLASSES[attr_type]
    return None


def _attribute_value(cluster_id: str, attribute_id: str, raw_value: Any) -> Any:
    if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
        return raw_value
    _unit, divisor = _attribute_scale(cluster_id, attribute_id)
    if divisor == 1.0:
        return raw_value
    return round(float(raw_value) / divisor, 2)


def _is_boilerplate_attribute(cluster_id: str, attribute_id: str) -> bool:
    """True for Matter plumbing that carries no device state.

    Global attributes (ClusterRevision, FeatureMap, AttributeList, ...) and the
    descriptive/identity clusters describe the protocol, not the home. Emitting
    them multiplies entity counts — a single air purifier yields 19 entities, of
    which 10 are of this kind — without giving an agent anything to observe or
    act on.
    """
    registry = load_registry()
    if registry.is_global(cluster_id, attribute_id):
        return True
    return cluster_id in _BOILERPLATE_CLUSTERS


_BOILERPLATE_CLUSTERS = {
    "BasicInformation",
    "BridgedDeviceBasicInformation",
    "Descriptor",
    "Identify",
    "PowerTopology",
}


# Attributes a cluster's commands write. Mirrors td/tables.py::_command_targets;
# an attribute that is read-only in the spec but driven by a command is still
# agent-controllable, and needs an entity for the agent to act through.
_COMMAND_TARGETS: dict[str, set[str]] = {
    "OnOff": {"OnOff"},
    "LevelControl": {"CurrentLevel"},
    "WindowCovering": {
        "CurrentPositionLiftPercent100ths", "TargetPositionLiftPercent100ths",
    },
    "Thermostat": {
        "OccupiedCoolingSetpoint", "OccupiedHeatingSetpoint", "SystemMode",
    },
    "TemperatureControl": {"TemperatureSetpoint"},
    "FanControl": {"PercentSetting", "FanMode", "SpeedSetting"},
    "DishwasherMode": {"CurrentMode"},
    "LaundryWasherMode": {"CurrentMode"},
    "RVCRunMode": {"CurrentMode"},
    "RVCCleanMode": {"CurrentMode"},
    "RTCCMode": {"CurrentMode"},
}

# Attributes already carried by the device's primary entity, so a companion
# would duplicate them: a `fan` primary owns its percentage, a `climate` primary
# its setpoints, a `light` its brightness.
_PRIMARY_OWNED: dict[str, set[tuple[str, str]]] = {
    "fan": {("OnOff", "OnOff"), ("FanControl", "PercentSetting"), ("FanControl", "FanMode")},
    "light": {("OnOff", "OnOff"), ("LevelControl", "CurrentLevel")},
    "climate": {
        ("OnOff", "OnOff"), ("Thermostat", "OccupiedCoolingSetpoint"),
        ("Thermostat", "OccupiedHeatingSetpoint"), ("Thermostat", "SystemMode"),
    },
    "cover": {
        ("WindowCovering", "CurrentPositionLiftPercent100ths"),
        ("WindowCovering", "TargetPositionLiftPercent100ths"),
    },
    "switch": {("OnOff", "OnOff")},
    "number": {("TemperatureControl", "TemperatureSetpoint")},
}


# Writable, but writing them changes nothing now: installed capability, the
# state to resume after a power cut, and timing that shapes a future action.
# Mirrors td/tables.py::NO_IMMEDIATE_EFFECT. They stay read-only sensors -- an
# agent has no reason to set them, and exposing them as controls invites it to.
_NO_IMMEDIATE_EFFECT: set[tuple[str, str]] = {
    ("Thermostat", "ControlSequenceOfOperation"),
    ("WindowCovering", "Mode"),
    ("OnOff", "StartUpOnOff"),
    ("LevelControl", "StartUpCurrentLevel"),
    ("OnOff", "OnTime"),
    ("OnOff", "OffWaitTime"),
    ("LevelControl", "OnOffTransitionTime"),
    ("LevelControl", "OnTransitionTime"),
    ("LevelControl", "OffTransitionTime"),
    ("LevelControl", "DefaultMoveRate"),
    ("LevelControl", "OnLevel"),
    ("LevelControl", "Options"),
}


def _is_actuatable(cluster_id: str, attribute_id: str) -> bool:
    """Writable in the spec, or written by one of the cluster's commands."""
    if (cluster_id, attribute_id) in _NO_IMMEDIATE_EFFECT:
        return False
    registry = load_registry()
    if registry.is_writable(cluster_id, attribute_id) is True:
        return True
    return attribute_id in _COMMAND_TARGETS.get(cluster_id, set())


def _numeric_bounds(
    cluster_id: str,
    attribute_id: str,
    value: Any,
    attributes: dict[str, Any],
) -> tuple[float, float]:
    """min/max for a `number` entity. The virtual platform requires both.

    Home Assistant ENFORCES them -- a set_value outside the range returns 500 and
    leaves the state untouched -- so a bound that is too wide lets an agent set a
    value the device would never accept, and one that is too narrow makes a legal
    request fail. Prefer, in order:

      1. the device's own declared bounds, from sibling attributes
         (TemperatureControl.Min/MaxTemperature, LevelControl.Min/MaxLevel);
      2. the number of entries in a sibling SupportedModes / SupportedX list,
         since a mode index is only valid within that list;
      3. the spec type's own range.
    """
    registry = load_registry()

    def sibling(name: str) -> Any:
        for path, raw in attributes.items():
            if isinstance(path, str) and path.endswith(f".{cluster_id}.{name}"):
                return _attribute_value(cluster_id, name, raw)
        return None

    for lo_name, hi_name in (
        ("MinTemperature", "MaxTemperature"),
        ("MinLevel", "MaxLevel"),
        ("MinMeasuredValue", "MaxMeasuredValue"),
    ):
        low, high = sibling(lo_name), sibling(hi_name)
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            return (float(low), float(high))

    # A mode or selection index is bounded by the list that enumerates it.
    for list_name in ("SupportedModes", "SupportedRinses", "SpinSpeeds",
                      "SupportedDrynessLevels"):
        options = sibling(list_name)
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except ValueError:
                options = None
        if isinstance(options, list) and options:
            return (0.0, float(len(options) - 1))

    attr_type = registry.attribute_type(cluster_id, attribute_id)
    if attr_type in {"percent", "percent100ths"}:
        return (0.0, 100.0)
    if attr_type == "uint8":
        return (0.0, 255.0)
    if attr_type == "uint16":
        return (0.0, 65535.0)
    numeric = float(value) if isinstance(value, (int, float)) else 0.0
    return (min(0.0, numeric), max(numeric, numeric + 100.0))


def _companion_entities(
    *,
    entity_name: str,
    attributes: dict[str, Any],
    primary_platform: str | None,
) -> list[dict[str, Any]]:
    """`number` entities for actuatable attributes the primary does not cover.

    Without these an agent can only switch the device on and off: qt4 asks to
    start a washer "on Heavy mode", set a vacuum "to Running state" and a TV "to
    level 70", none of which a switch can express.
    """
    owned = _PRIMARY_OWNED.get(primary_platform or "", set())
    out: list[dict[str, Any]] = []
    for attr_path, raw_value in attributes.items():
        if not isinstance(attr_path, str):
            continue
        parts = attr_path.split(".", 2)
        if len(parts) != 3:
            continue
        _endpoint, cluster_id, attribute_id = parts
        if (cluster_id, attribute_id) in owned:
            continue
        if _is_boilerplate_attribute(cluster_id, attribute_id):
            continue
        if not _is_actuatable(cluster_id, attribute_id):
            continue

        value = _attribute_value(cluster_id, attribute_id, raw_value)
        if isinstance(value, bool):
            continue  # a `number` cannot carry a boolean; that is the primary's job
        if value is None:
            # The control exists even when the episode leaves it unset: a TV's
            # CurrentLevel is null at start, yet qt4 asks to "set TV 1 to level
            # 70". Seed it at the device's own lower bound.
            low, _high = _numeric_bounds(cluster_id, attribute_id, 0, attributes)
            value = low
        if not isinstance(value, (int, float)):
            continue

        low, high = _numeric_bounds(cluster_id, attribute_id, value, attributes)
        entry: dict[str, Any] = {
            "platform": "number",
            "name": f"{entity_name}.{_dotted_token(cluster_id)}.{_dotted_token(attribute_id)}",
            "initial_value": value,
            "min": min(low, float(value)),
            "max": max(high, float(value)),
        }
        unit = _attribute_unit(cluster_id, attribute_id)
        if unit:
            entry["unit_of_measurement"] = unit
        device_class = _attribute_class(cluster_id, attribute_id)
        if device_class:
            entry["class"] = device_class
        out.append(entry)
    return out


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
        # A cabinet is always running; what an agent sets is its target
        # temperature ("set freezer 1 to -23"), so expose a `number` rather than
        # a switch. min/max are REQUIRED by the virtual number platform and come
        # from the device's own TemperatureControl bounds.
        setpoint = _matter_temp_c(attributes.get("1.TemperatureControl.TemperatureSetpoint"))
        low = _matter_temp_c(attributes.get("1.TemperatureControl.MinTemperature"))
        high = _matter_temp_c(attributes.get("1.TemperatureControl.MaxTemperature"))
        if setpoint is None:
            setpoint = low if low is not None else 0.0
        return {
            "platform": "number",
            "name": entity_name,
            "initial_value": setpoint,
            "min": low if low is not None else setpoint,
            "max": high if high is not None else setpoint,
            "unit_of_measurement": "°C",
            "class": "temperature",
        }

    if device_type in {"dishwasher", "laundry_washer", "laundry_dryer", "tv", "rvc"}:
        # These are started and stopped ("start dishwasher 1", "pause it",
        # "start playing TV 1"), so they need an actuable primary. The virtual
        # component offers no media_player or vacuum platform, and `switch` is
        # the one that carries turn_on/turn_off/toggle.
        return {
            "platform": "switch",
            "name": entity_name,
            "initial_value": _bool_to_on_off(on_off),
        }

    return None


def _convert_initial_home_config(
    payload: dict[str, Any],
    source_path: Path,
    *,
    keep_boilerplate: bool = False,
) -> dict[str, Any]:
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

            # An actuatable attribute the primary does not expose gets its own
            # `number`, so an agent can set a washer's mode or a TV's level
            # rather than only switching the device on and off.
            companions = _companion_entities(
                entity_name=entity_name,
                attributes=attributes,
                primary_platform=(primary or {}).get("platform"),
            )
            entries.extend(companions)
            # Those attributes are now writable entities; emitting a read-only
            # sensor for the same value as well would duplicate it under a second
            # entity_id that silently drifts from the one an agent writes.
            companion_names = {entry["name"] for entry in companions}

            for attr_path, raw_value in attributes.items():
                if not isinstance(attr_path, str):
                    continue
                parts = attr_path.split(".", 2)
                if len(parts) == 3:
                    _, cluster_id, attribute_id = parts
                else:
                    cluster_id = "Attribute"
                    attribute_id = attr_path

                if not keep_boilerplate and _is_boilerplate_attribute(cluster_id, attribute_id):
                    continue

                attr_entity_name = (
                    f"{entity_name}.{_dotted_token(cluster_id)}.{_dotted_token(attribute_id)}"
                )
                if attr_entity_name in companion_names:
                    continue  # already emitted as a writable `number`
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


def _process_json_file(path: Path, *, keep_boilerplate: bool = False) -> tuple[bool, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"{path}: failed to read JSON: {exc}"

    if not isinstance(payload, dict):
        return False, f"{path}: top-level JSON value must be an object"

    try:
        ha_yaml = _convert_initial_home_config(payload, path, keep_boilerplate=keep_boilerplate)
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
    parser.add_argument(
        "--keep-boilerplate",
        action="store_true",
        help=(
            "Also emit sensors for Matter protocol plumbing (global attributes plus the "
            "BasicInformation/Descriptor/Identify clusters). Off by default: these describe "
            "the protocol, not the home, and roughly halve the entity count."
        ),
    )
    args = parser.parse_args()

    root = args.directory.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"directory not found: {root}")

    registry = load_registry()
    print(f"Matter data model: {registry.provenance}")

    json_files = _walk_json_files(root)
    if not json_files:
        print(f"No JSON files found under {root}")
        return 0

    ok_count = 0
    skipped_count = 0
    for path in json_files:
        ok, message = _process_json_file(path, keep_boilerplate=args.keep_boilerplate)
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
