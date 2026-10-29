"""Generate the HomeBench mapping tables (``tests/homebench/mappings/*.yaml``).

The tables are the reviewed layer between HomeBench's device model (``state`` plus a
flat attribute map, one ``set_<attribute>`` method per settable attribute) and the
Thing Descriptions the agents read: which homeont class a device type is, what each
attribute is called as an affordance, which homeont property and command class types
it, and which gold call an invocation stands for. ``HomeBench/mappings.py`` loads
them; ``convert.py --mappings`` writes them.

Rows follow the SimuHome convention (``tests/simuhome/td/convert.py --phase-a``): a
generated row carries ``status: auto``; a row a human has set to ``status: approved``
is kept verbatim on regeneration, so a hand edit is never overwritten.

The generator is driven by the dataset: it reads the homes and fails when a home has
a device type, attribute or room that the specification below does not cover, so a
dataset change cannot slip into the TDs unmapped.

Worked example. ``light`` devices occur with ``state``, ``brightness`` and ``color``
attributes, so the generator writes, among others::

    command_map:  key light.onOff       gold_op {"true": "turn_on()", "false": "turn_off()"}
                  key light.brightness  gold_op "set_brightness({value})"
    actuatable_properties:
                  key light.brightness  affordance_name brightness, value_kind integer,
                                        property_type homeont:LevelControlBrightness,
                                        action_class saref:SetAbsoluteLevelCommand

and ``HomeBenchTD`` then serves an action ``brightness`` on ``bathroom_light``; the
gold string of invoking it with 20 is ``bathroom.light.set_brightness(20)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from ami_agents.environment.integration.HomeBench import engine

LOGGER = logging.getLogger(__name__)

# -- constants ------------------------------------------------------------------------

MAPPINGS_DIR = Path(__file__).resolve().parent / "mappings"
GENERATOR = "tests/homebench/convert.py --mappings"
SETTLED = ("approved", "auto")

#: Tables written empty on purpose: HomeBench has no ambient instruments, so no virtual
#: sensor exists to read a room variable. The table exists so ``HomeBenchTD`` reads the
#: same lookups as ``SimuHomeTD`` and an empty answer is a stated fact, not a missing
#: file. (The room variables and the effects are declared, see ``EFFECT_FAMILIES``.)
EMPTY_TABLES = ("ambient_sensors",)

# -- the specification ----------------------------------------------------------------

#: HomeBench room -> (homeont class, title). ``ding_room`` is HomeBench's spelling.
ROOM_SPEC: Dict[str, Tuple[str, str]] = {
    "master_bedroom": ("homeont:MasterBedroom", "Master Bedroom"),
    "guest_bedroom": ("homeont:GuestBedroom", "Guest Bedroom"),
    "living_room": ("homeont:LivingRoom", "Living Room"),
    "ding_room": ("homeont:DiningRoom", "Dining Room"),
    "study_room": ("homeont:StudyRoom", "Study Room"),
    "kitchen": ("homeont:Kitchen", "Kitchen"),
    "bathroom": ("homeont:Bathroom", "Bathroom"),
    "foyer": ("homeont:Foyer", "Foyer"),
    "corridor": ("homeont:Corridor", "Corridor"),
    "balcony": ("homeont:Balcony", "Balcony"),
    "garage": ("homeont:Garage", "Garage"),
    "store_room": ("homeont:StoreRoom", "Store Room"),
    # The vacuum robot has no room in HomeBench; it lives here (decided 2026-10-24).
    engine.HOUSEHOLD_ROOM: ("homeont:BuildingSpace", "Household"),
}

#: device type -> (homeont class, SAREF parent or ``None``, title). A device type with
#: no static parent is typed ``saref:Actuator`` by ``HomeBenchTD`` only when it has an
#: effect on a room variable (``EFFECT_FAMILIES``), so the type says what the device does
#: in the model instead of being a blanket parent.
DEVICE_SPEC: Dict[str, Tuple[str, Optional[str], str]] = {
    "light": ("homeont:OnOffLight", None, "Light"),
    "air_conditioner": ("homeont:AirConditioner", "saref:HVAC", "Air Conditioner"),
    "heating": ("homeont:Heater", "saref:HVAC", "Heater"),
    "fan": ("homeont:Fan", "saref:Appliance", "Fan"),
    "air_purifiers": ("homeont:AirPurifier", "saref:Appliance", "Air Purifier"),
    "humidifier": ("homeont:Humidifier", "saref:Appliance", "Humidifier"),
    "dehumidifiers": ("homeont:Dehumidifier", "saref:Appliance", "Dehumidifier"),
    "aromatherapy": ("homeont:Aromatherapy", "saref:Appliance", "Aromatherapy"),
    "water_heater": ("homeont:WaterHeater", "saref:Appliance", "Water Heater"),
    "curtain": ("homeont:Curtain", None, "Curtain"),
    "blinds": ("homeont:Blinds", None, "Blinds"),
    "garage_door": ("homeont:GarageDoor", None, "Garage Door"),
    "media_player": ("homeont:MediaPlayer", "saref:Appliance", "Media Player"),
    "trash": ("homeont:Trash", "saref:Appliance", "Trash"),
    "vacuum_robot": ("homeont:Rvc", "saref:Appliance", "Vacuum Robot"),
}

#: A light is the most specific class its attributes allow, first match wins:
#: ``color`` -> ColorLight, ``brightness`` -> DimmableLight, else OnOffLight.
LIGHT_VARIANTS: Tuple[Tuple[str, str, str], ...] = (
    ("color", "homeont:ColorLight", "Color Light"),
    ("brightness", "homeont:DimmableLight", "Dimmable Light"),
)

#: What a device's ``state`` attribute is, per device type: the kind selects the
#: affordance name, the value schema and the gold calls (``STATE_KINDS``).
DEVICE_STATE_KIND: Dict[str, Optional[str]] = {
    "light": "onoff", "air_conditioner": "onoff", "heating": "onoff", "fan": "onoff",
    "air_purifiers": "onoff", "humidifier": "onoff", "dehumidifiers": "onoff",
    "aromatherapy": "onoff", "water_heater": "onoff",
    "curtain": "openclose", "blinds": "openclose", "garage_door": "openclose",
    "media_player": "playback", "trash": "pack",
    "vacuum_robot": None,  # reports ``on`` and has no command that changes it
}

#: The ``state`` kinds: affordance name, value kind, enum values, home value shown to
#: a reader (``read_map``), property and command class, and the gold call per value.
STATE_KINDS: Dict[str, Dict[str, Any]] = {
    "onoff": {
        "name": "onOff", "value_kind": "boolean",
        "read_map": {"on": True, "off": False},
        "property_type": "homeont:OnOff", "action_class": "homeont:SetOnOffCommand",
        "gold_op": {"true": "turn_on()", "false": "turn_off()"},
    },
    "openclose": {
        "name": "openClose", "value_kind": "enum", "values": ["open", "closed"],
        "property_type": "homeont:OpenCloseState",
        "action_class": "homeont:SetOpenCloseCommand",
        "gold_op": {"open": "open()", "closed": "close()"},
    },
    "playback": {
        "name": "playback", "value_kind": "enum",
        "values": ["playing", "paused", "stopped"],
        "property_type": "homeont:OperationalState",
        "action_class": "homeont:SetOperationalStateCommand",
        "gold_op": {"playing": "play()", "paused": "pause()", "stopped": "stop()"},
    },
    "pack": {
        "name": "pack", "value_kind": "none",
        "read_values": ["empty", "not_full", "full"],
        "property_type": "homeont:TrashFillState",
        "action_class": "homeont:PackCommand", "gold_op": "pack()",
    },
}

#: The vacuum's read-only ``state``.
VACUUM_STATE = {"name": "state", "property_type": "homeont:VacuumState"}

#: Settable attribute -> affordance name, value kind, property and command class
#: (``set_<attribute>`` is the gold method). ``unit`` is the symbol the TD builder
#: resolves to a QUDT unit. ``PROPERTY_OVERRIDES`` refines a class per device type.
ATTRIBUTE_SPEC: Dict[str, Dict[str, Any]] = {
    "brightness": {"name": "brightness", "value_kind": "integer",
                   "property_type": "homeont:LevelControlBrightness",
                   "action_class": "saref:SetAbsoluteLevelCommand"},
    "color": {"name": "color", "value_kind": "rgb",
              "property_type": "homeont:LightColor",
              "action_class": "homeont:SetColorCommand"},
    "temperature": {"name": "temperature", "value_kind": "integer", "unit": "°C",
                    "property_type": "homeont:TemperatureSetpoint",
                    "action_class": "saref:SetAbsoluteLevelCommand"},
    "mode": {"name": "mode", "value_kind": "enum",
             "property_type": "homeont:DeviceMode",
             "action_class": "homeont:SetModeCommand"},
    "fan_speed": {"name": "fanSpeed", "value_kind": "enum",
                  "property_type": "homeont:FanControlFanMode",
                  "action_class": "homeont:SetModeCommand"},
    "speed": {"name": "speed", "value_kind": "enum",
              "property_type": "homeont:FanControlFanMode",
              "action_class": "homeont:SetModeCommand"},
    "swing": {"name": "swing", "value_kind": "enum",
              "property_type": "homeont:SwingMode",
              "action_class": "homeont:SetModeCommand"},
    "intensity": {"name": "intensity", "value_kind": "integer",
                  "property_type": "homeont:OutputIntensity",
                  "action_class": "saref:SetAbsoluteLevelCommand"},
    "interval": {"name": "interval", "value_kind": "integer",
                 "property_type": "homeont:OperatingInterval",
                 "action_class": "saref:SetAbsoluteLevelCommand"},
    "degree": {"name": "degree", "value_kind": "integer",
               "property_type": "homeont:WindowCoveringPosition",
               "action_class": "saref:SetAbsoluteLevelCommand"},
    "volume": {"name": "volume", "value_kind": "integer",
               "property_type": "homeont:Volume",
               "action_class": "saref:SetAbsoluteLevelCommand"},
    "cleaning_area": {"name": "cleaningArea", "value_kind": "enum",
                      "property_type": "homeont:CleaningArea",
                      "action_class": "homeont:SetModeCommand"},
}
PROPERTY_OVERRIDES: Dict[Tuple[str, str], str] = {
    ("air_conditioner", "mode"): "homeont:ThermostatHvacMode",
    ("heating", "mode"): "homeont:HeaterHvacMode",
    ("heating", "temperature"): "homeont:HeaterHeatingSetpoint",
}

#: Read-only attributes: no ``set_`` method exists, so no action.
READONLY_SPEC: Dict[str, Dict[str, Any]] = {
    "battery": {"name": "battery", "value_kind": "integer",
                "property_type": "homeont:BatteryLevel", "unit": "%"},
    "tank": {"name": "tank", "value_kind": "integer",
             "property_type": "homeont:WaterTankLevel", "unit": "%"},
}

#: Property classes the rows use and how a TD marks them (``kind``): an actuatable
#: property gets ``tdsosa:ActuatablePropertyAffordance`` and an action; an observable
#: or state one ``tdsosa:ObservablePropertyAffordance`` and no action.
OBSERVABLE_CLASSES = ("homeont:BatteryLevel", "homeont:WaterTankLevel",
                      "homeont:VacuumState")

#: Which ambient variable the devices of a type can affect: device type -> the
#: ``sosa_property`` of the variable. Exactly SimuHome's effect table, without the
#: window coverings (curtain, blinds) and the garage door, which the project decided
#: not to model as acting on ``illuminance`` (decision N1, 2026-10-26).
EFFECT_FAMILIES: Dict[str, str] = {
    "air_conditioner": "air_temperature",
    "heating": "air_temperature",
    "humidifier": "relative_humidity",
    "dehumidifiers": "relative_humidity",
    "air_purifiers": "pm10_mass_concentration",
    "light": "illuminance",
}

#: The room variables, keyed by the room-state token the tables and ``HomeBenchTD``
#: use (``room_state.<token>``); the values are SimuHome's approved rows (the same
#: homeont class, QUDT quantity kind and unit), so one variable is the same node on
#: every platform. ``unit`` is the symbol the TD builder resolves to a QUDT unit.
OBSERVABLE_SPEC: Dict[str, Dict[str, str]] = {
    "temperature": {
        "room_state_token": "Temperature", "sosa_property": "air_temperature",
        "homeont_class": "homeont:AirTemperature", "unit": "°C",
        "quantityKind": "quantitykind:ThermodynamicTemperature",
        "qudt_unit": "unit:DEG_C"},
    "humidity": {
        "room_state_token": "Humidity", "sosa_property": "relative_humidity",
        "homeont_class": "homeont:RelativeHumidity", "unit": "%",
        "quantityKind": "quantitykind:RelativeHumidity", "qudt_unit": "unit:PERCENT"},
    "illuminance": {
        "room_state_token": "Illuminance", "sosa_property": "illuminance",
        "homeont_class": "homeont:Illuminance", "unit": "lx",
        "quantityKind": "quantitykind:Illuminance", "qudt_unit": "unit:LUX"},
    "pm10": {
        "room_state_token": "Pm10", "sosa_property": "pm10_mass_concentration",
        "homeont_class": "homeont:Pm10MassConcentration", "unit": "ug/m3",
        "quantityKind": "quantitykind:MassConcentration",
        "qudt_unit": "unit:MicroGM-PER-M3"},
}


class MappingError(ValueError):
    """The dataset holds something the specification does not cover."""


# -- row builders -----------------------------------------------------------------------


def _header(table: str) -> Dict[str, Any]:
    """The ``generator`` block every table opens with."""
    return {"script": GENERATOR, "table": table}


def device_type_rows(types: Iterable[str]) -> List[Dict[str, Any]]:
    """One ``device_type_map`` row per device type seen in the data."""
    rows = []
    for dtype in sorted(types):
        cls, parent, title = DEVICE_SPEC[dtype]
        # The SAREF parent is optional: a type without one gets ``saref:Actuator`` from
        # the builder, and only when it has an effect.
        parents = [parent] if parent else []
        row: Dict[str, Any] = {"key": dtype, "td_types": [cls, *parents],
                               "title": title}
        # A light is typed by what it can do; variants are tried in order.
        if dtype == "light":
            row["variants"] = [
                {"requires": attr, "title": vtitle, "td_types": [vcls, *parents]}
                for attr, vcls, vtitle in LIGHT_VARIANTS]
        row["status"] = "auto"
        rows.append(row)
    return rows


def room_rows(rooms: Iterable[str]) -> List[Dict[str, Any]]:
    """One ``room_map`` row per room seen in the data."""
    return [{"key": room, "homeont_class": ROOM_SPEC[room][0],
             "title": ROOM_SPEC[room][1], "status": "auto"}
            for room in sorted(rooms)]


def _attribute_rows(dtype: str, attrs: Iterable[str]) -> List[Dict[str, Any]]:
    """The affordance rows of one device type: its state, then its attributes."""
    rows: List[Dict[str, Any]] = []
    # 1. The ``state`` attribute.
    kind = DEVICE_STATE_KIND[dtype]
    if kind is not None:
        spec = STATE_KINDS[kind]
        row = {"key": f"{dtype}.{spec['name']}", "device_type": dtype,
               "attribute": "state", "affordance_name": spec["name"],
               "value_kind": spec["value_kind"], "actuatable": True,
               "property_type": spec["property_type"],
               "action_class": spec["action_class"]}
        if "read_map" in spec:
            row["read_map"] = dict(spec["read_map"])
        if "values" in spec:
            row["values"] = list(spec["values"])
        if "read_values" in spec:
            row["read_values"] = list(spec["read_values"])
        rows.append(row)
    else:
        rows.append({"key": f"{dtype}.{VACUUM_STATE['name']}", "device_type": dtype,
                     "attribute": "state", "affordance_name": VACUUM_STATE["name"],
                     "value_kind": "enum", "actuatable": False,
                     "property_type": VACUUM_STATE["property_type"]})
    # 2. Every other attribute: settable ones get an action, the rest only a read.
    for attr in sorted(a for a in attrs if a != "state"):
        if attr in engine.SETTABLE_ATTRIBUTES.get(dtype, ()):
            spec = ATTRIBUTE_SPEC[attr]
            row = {"key": f"{dtype}.{spec['name']}", "device_type": dtype,
                   "attribute": attr, "affordance_name": spec["name"],
                   "value_kind": spec["value_kind"], "actuatable": True,
                   "property_type": PROPERTY_OVERRIDES.get(
                       (dtype, attr), spec["property_type"]),
                   "action_class": spec["action_class"]}
        elif attr in READONLY_SPEC:
            spec = READONLY_SPEC[attr]
            row = {"key": f"{dtype}.{spec['name']}", "device_type": dtype,
                   "attribute": attr, "affordance_name": spec["name"],
                   "value_kind": spec["value_kind"], "actuatable": False,
                   "property_type": spec["property_type"]}
        else:
            raise MappingError(f"{dtype}.{attr}: attribute has no specification")
        if "unit" in spec:
            row["unit"] = spec["unit"]
        rows.append(row)
    for row in rows:
        row["status"] = "auto"
    return rows


def actuatable_rows(seen: Dict[str, set]) -> List[Dict[str, Any]]:
    """The ``actuatable_properties`` table: every affordance row, all device types."""
    return [row for dtype in sorted(seen) for row in _attribute_rows(dtype, seen[dtype])]


def command_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The ``command_map`` table: the gold call each actuatable row stands for.

    ``gold_op`` is the reverse template: a string with ``{value}`` is a setter whose
    argument is the action's value; a string without it is a call with no argument;
    a mapping gives one call per value (``onOff`` true -> ``turn_on()``). The forward
    direction (action -> engine command) is parsed back out of it by
    ``HomeBench/mappings.py``, so the two cannot drift.
    """
    out = []
    for row in rows:
        if not row["actuatable"]:
            continue
        dtype, attr = row["device_type"], row["attribute"]
        if attr == "state":
            gold: Any = STATE_KINDS[DEVICE_STATE_KIND[dtype]]["gold_op"]
            gold = dict(gold) if isinstance(gold, dict) else gold
        else:
            gold = f"set_{attr}({{value}})"
        out.append({"key": row["key"], "device_type": dtype,
                    "action": row["affordance_name"], "gold_op": gold,
                    "status": "auto"})
    return out


def property_class_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The ``property_classes`` table: each homeont property class the rows use."""
    used: Dict[str, str] = {}
    for row in rows:
        kind = "actuatable" if row["actuatable"] else "observable"
        used[row["property_type"]] = kind
    return [{"key": cls, "kind": used[cls], "status": "auto"} for cls in sorted(used)]


def observable_rows() -> List[Dict[str, Any]]:
    """The ``observable_properties`` table: the four room variables.

    One row per ``OBSERVABLE_SPEC`` entry, shaped like SimuHome's
    ``room_state.<token>`` rows. The variables are declared, not observable (no
    HomeBench device reads one and the home has no value for it); the row types them.
    """
    rows = []
    # One row per variable; ``evidence`` says where the shape comes from and that no
    # reader exists, so a reviewer sees why the property is declared but not served.
    for token, spec in OBSERVABLE_SPEC.items():
        rows.append({
            "key": f"room_state.{token}", **spec,
            "sosa_type": "sosa:ObservableProperty", "feature_of_interest": "room",
            "evidence": {"named_after": f"the measured quantity ({spec['quantityKind']})",
                         "mirrors": "tests/simuhome/td/mappings/observable_properties.yaml",
                         "observable": False},
            "status": "auto"})
    return rows


def effect_rows(affordances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The ``actuation_effects`` table: one row per settable attribute of an effect family.

    Every actuatable affordance of a device type in ``EFFECT_FAMILIES`` is claimed to
    affect that family's variable, ``direction`` null: every HomeBench action carries a
    value (``onOff`` included), so invoking it does not say which way the variable moves.
    The rows are ``auto``: a human approves them, and an approved row survives
    regeneration. They mean "this action can move that variable", declared domain
    semantics; the engine does not simulate them.
    """
    rows = []
    for row in affordances:
        # 1. Only settable attributes of an effect family: read-only ones (``tank``)
        #    have no action, so nothing to claim an effect for.
        dtype = row["device_type"]
        if not row["actuatable"] or dtype not in EFFECT_FAMILIES:
            continue
        # 2. One claim per (device type, attribute): the variable it can move.
        attribute = row["attribute"]
        rows.append({
            "key": f"{dtype}:{attribute}", "family": dtype,
            "command": {"mechanism": "attribute_write", "name": attribute},
            "affectsObservableProperty": EFFECT_FAMILIES[dtype],
            "actsOnProperty": None, "direction": None,
            "evidence": {
                "rationale": "value-carrying setter: no inherent direction",
                "seeded_from": "SimuHome actuation_effects (same family, same variable)",
                "simulated": False},
            "status": "auto"})
    return rows


# -- the tables -------------------------------------------------------------------------


def observed(homes: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, set], set]:
    """``({device_type: attributes seen}, rooms seen)`` over converted homes.

    Raises ``MappingError`` for a device type or room the specification lacks.
    """
    seen: Dict[str, set] = {}
    rooms: set = set()
    for home in homes:
        for room, data in home["rooms"].items():
            if room not in ROOM_SPEC:
                raise MappingError(f"room {room!r} has no specification")
            rooms.add(room)
            for device in data["devices"]:
                dtype = device["device_type"]
                if dtype not in DEVICE_SPEC:
                    raise MappingError(f"device type {dtype!r} has no specification")
                seen.setdefault(dtype, set()).update(device["attributes"])
    return seen, rooms


def generate_tables(homes: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Build every table (name -> rows) from the converted homes."""
    seen, rooms = observed(homes)
    affordances = actuatable_rows(seen)
    tables = {
        "device_type_map": device_type_rows(seen),
        "room_map": room_rows(rooms),
        "property_classes": property_class_rows(affordances),
        "actuatable_properties": affordances,
        "command_map": command_rows(affordances),
        "observable_properties": observable_rows(),
        "actuation_effects": effect_rows(affordances),
    }
    for name in EMPTY_TABLES:
        tables[name] = []
    return tables


def _merge(existing: List[Dict[str, Any]], fresh: List[Dict[str, Any]]
           ) -> List[Dict[str, Any]]:
    """Fresh rows, except where a human approved the existing row (kept verbatim).

    An approved row whose key is no longer generated is kept too: removing a
    reviewed decision is the reviewer's call, not the generator's.
    """
    approved = {r["key"]: r for r in existing if r.get("status") == "approved"}
    merged = [approved.pop(row["key"], row) for row in fresh]
    return merged + list(approved.values())


def write_tables(tables: Dict[str, List[Dict[str, Any]]],
                 out_dir: Any = MAPPINGS_DIR) -> Dict[str, int]:
    """Write one YAML file per table, preserving approved rows; return row counts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in tables.items():
        path = out / f"{name}.yaml"
        # 1. Keep what a human approved in the previous file.
        if path.is_file():
            previous = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                        ).get("rows") or []
            rows = _merge(previous, rows)
        # 2. Write the table: a comment banner, the generator block, the rows.
        body = yaml.safe_dump({"version": 1, "generator": _header(name), "rows": rows},
                              sort_keys=False, allow_unicode=True)
        banner = (f"# GENERATED by {GENERATOR}\n"
                  "# Edit freely. Set `status: approved` on a row to protect it from\n"
                  "# regeneration; every other row is rewritten on each run.\n")
        path.write_text(banner + body, encoding="utf-8", newline="\n")
        counts[name] = len(rows)
    return counts
