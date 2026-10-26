"""Load the HomeBench mapping tables and expose them as lookups.

The tables under ``tests/homebench/mappings/`` (written by
``tests/homebench/convert.py --mappings``) are the reviewed layer between HomeBench's
device model and the Thing Descriptions: device type -> homeont class, room -> homeont
space, attribute -> affordance name and homeont property / command class, and the
gold call each action invocation stands for. ``HomeBenchTD`` and ``hbtd.py`` read
them through this module and keep no table of their own.

The class offers the lookups ``SimuHomeTD`` calls on its ``self.m`` (``room``,
``device_family``, ``senses``, ``room_state_property``, ``ambient_sensor``, ...), so
the inherited document builders run unchanged; the observable, ambient and actuation
tables are empty (HomeBench has no physics), and those lookups answer "nothing".

A row is honoured only when ``status`` is ``approved`` or ``auto``; anything else is
counted by ``unsettled()`` instead of being used.

The ``command_map`` row is the single source of both directions. Its ``gold_op`` is
either a call string (``pack()``), a template whose ``{value}`` is the action's value
(``set_brightness({value})``) or a mapping value -> call (``onOff``: true ->
``turn_on()``). The forward direction is parsed out of it, so it cannot drift:

    m = load_mappings()
    m.command_for("light", "onOff", True)             # ("turn_on", None)
    m.command_for("light", "brightness", 20)          # ("set_brightness", 20)
    m.gold_string("bathroom_light", "light", "brightness", 20)
    # "bathroom.light.set_brightness(20)"
    m.action_for_command("light", "turn_off", None)   # ("onOff", False)
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

LOGGER = logging.getLogger(__name__)

# -- constants ------------------------------------------------------------------------

MAPPINGS_DIR = (
    Path(__file__).resolve().parents[4] / "tests" / "homebench" / "mappings"
)

#: The table file format this module understands (``version`` in every file).
TABLE_VERSION = 1

#: Rows in these states are trusted. Anything else still needs a human.
SETTLED = {"approved", "auto"}

#: Every table this module reads (the last three are empty by design).
TABLES = (
    "device_type_map", "room_map", "property_classes", "actuatable_properties",
    "command_map", "observable_properties", "ambient_sensors", "actuation_effects",
)

#: The synthetic room that holds the vacuum robot; the gold string has no room for it.
HOUSEHOLD_ROOM = "household"

_OP = re.compile(r"^(?P<command>\w+)\((?P<arg>.*)\)$")
_VALUE = "{value}"


class MappingTableError(ValueError):
    """A mapping table row is malformed or a lookup names no row."""


# -- loading -----------------------------------------------------------------------------


def _load_table(name: str) -> Dict[str, Any]:
    """Read one table file, or raise ``FileNotFoundError`` naming it."""
    path = MAPPINGS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"mapping table not found: {path}")
    table = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # The only table format this module reads; a newer file must not load silently.
    if table.get("version") != TABLE_VERSION:
        raise MappingTableError(
            f"{path.name}: version {table.get('version')!r}, expected {TABLE_VERSION}")
    return table


def _rows(name: str, settled_only: bool = True) -> List[Dict[str, Any]]:
    """The rows of a table; by default only the settled ones."""
    rows = _load_table(name).get("rows") or []
    if not settled_only:
        return rows
    return [r for r in rows if (r.get("status") or "auto") in SETTLED]


def _value_key(value: Any) -> str:
    """The mapping key of an action value: ``True`` -> ``"true"``, ``"open"`` -> itself."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_arg(value: Any) -> str:
    """A value as the gold writes an argument: ints and words bare, lists comma-joined."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _parse_op(call: str) -> Tuple[str, bool]:
    """Split a ``gold_op`` call into ``(command, takes_value)``."""
    found = _OP.match(call)
    if found is None:
        raise MappingTableError(f"gold_op {call!r} is not command(args)")
    arg = found["arg"]
    if arg not in ("", _VALUE):
        raise MappingTableError(f"gold_op {call!r}: argument must be empty or {_VALUE}")
    return found["command"], arg == _VALUE


class Mappings:
    """Every HomeBench table, indexed for lookup by the TD builder and the server."""

    def __init__(self) -> None:
        """Read the tables and build the indexes."""
        # 1. Devices and rooms, keyed by the HomeBench name.
        self.device_types: Dict[str, Dict[str, Any]] = {
            str(r["key"]): r for r in _rows("device_type_map")}
        self.rooms: Dict[str, Dict[str, Any]] = {
            str(r["key"]): r for r in _rows("room_map")}
        # 2. Property classes, keyed by class CURIE; affordance rows by device type.
        self.property_classes: Dict[str, Dict[str, Any]] = {
            str(r["key"]): r for r in _rows("property_classes")}
        self._affordances: Dict[str, List[Dict[str, Any]]] = {}
        for row in _rows("actuatable_properties"):
            self._affordances.setdefault(str(row["device_type"]), []).append(row)
        # 3. The command table, with each row's forward form parsed once.
        self._commands: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in _rows("command_map"):
            self._commands[(str(row["device_type"]), str(row["action"]))] = \
                self._parse_command_row(row)
        # 4. The tables HomeBench leaves empty: no physics, no ambient sensors.
        self.observable_properties: Dict[str, Dict[str, Any]] = {
            str(r["key"]): r for r in _rows("observable_properties")}
        self.ambient_sensors: Dict[str, Dict[str, Any]] = {
            str(r["key"]): r for r in _rows("ambient_sensors")}
        self.actuation_effects: List[Dict[str, Any]] = _rows("actuation_effects")
        self.families_affecting_environment: Dict[str, set] = {}
        self.sensing: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _parse_command_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Index a ``command_map`` row: ``{value key: (command, takes_value)}``."""
        gold = row["gold_op"]
        if isinstance(gold, dict):
            by_value = {str(k): _parse_op(v) for k, v in gold.items()}
            return {"by_value": by_value, "gold": gold}
        return {"single": _parse_op(gold), "gold": gold}

    # -- SimuHomeTD's lookups ------------------------------------------------------

    def room(self, room_id: str) -> Optional[Dict[str, Any]]:
        """The ``room_map`` row of a room id."""
        return self.rooms.get(room_id)

    def device_family(self, device_type: str,
                      attributes: Optional[Dict[str, Any]] = None
                      ) -> Optional[Dict[str, Any]]:
        """The ``device_type_map`` row of a device type, variant resolved.

        When the row lists ``variants`` (a light), the first one whose required
        attribute the device has replaces ``td_types`` and ``title``. Without
        ``attributes`` the base class is returned.
        """
        row = self.device_types.get(device_type)
        if row is None or not attributes:
            return row
        for variant in row.get("variants") or []:
            if variant["requires"] in attributes:
                return {**row, "td_types": variant["td_types"],
                        "title": variant["title"]}
        return row

    def room_state_property(self, token: str) -> None:
        """No room variables exist in HomeBench."""
        return None

    def ambient_sensor(self, token: str) -> None:
        """No ambient sensors exist in HomeBench."""
        return None

    def senses(self, device_type: str) -> None:
        """No HomeBench device observes a room property."""
        return None

    def effects_for(self, device_type: str, cluster: str) -> List[Dict[str, Any]]:
        """No actuation effects: HomeBench has no physics."""
        return []

    # -- affordances ----------------------------------------------------------------

    def affordances(self, device_type: str, attributes: Optional[Dict[str, Any]] = None
                    ) -> List[Dict[str, Any]]:
        """The affordance rows of a device type, in table order.

        With ``attributes`` only the rows whose home attribute the device has: a light
        without ``brightness`` has no ``brightness`` property or action.
        """
        rows = self._affordances.get(device_type, [])
        if attributes is None:
            return list(rows)
        return [r for r in rows if r["attribute"] in attributes]

    def affordance(self, device_type: str, name: str) -> Optional[Dict[str, Any]]:
        """The affordance row named ``name`` on a device type."""
        for row in self._affordances.get(device_type, []):
            if row["affordance_name"] == name:
                return row
        return None

    def property_kind(self, curie: str) -> str:
        """``actuatable`` or ``observable``: how a TD marks a property of this class."""
        row = self.property_classes.get(curie)
        if row is None:
            raise MappingTableError(f"property class {curie!r} is not in property_classes")
        return str(row["kind"])

    @staticmethod
    def read_value(row: Dict[str, Any], raw: Any) -> Any:
        """A home attribute value as the TD reports it (``on`` -> ``true``)."""
        read_map = row.get("read_map")
        if read_map is None:
            return raw
        if raw not in read_map:
            raise MappingTableError(f"{row['key']}: no read_map entry for {raw!r}")
        return read_map[raw]

    @staticmethod
    def write_attribute(row: Dict[str, Any], value: Any) -> Any:
        """The inverse of ``read_value``: a TD value as the home attribute holds it."""
        read_map = row.get("read_map")
        if read_map is None:
            return value
        for stored, shown in read_map.items():
            if shown == value and type(shown) is type(value):
                return stored
        raise MappingTableError(f"{row['key']}: no home value shows as {value!r}")

    # -- the gold round trip ----------------------------------------------------------

    def command_for(self, device_type: str, action: str, value: Any
                    ) -> Tuple[str, Any]:
        """The engine ``(command, arg)`` an action invocation stands for.

        Raises ``MappingTableError`` for an action or value the table does not know.
        """
        row = self._commands.get((device_type, action))
        if row is None:
            raise MappingTableError(f"{device_type} has no action {action!r}")
        # 1. A mapping row: the value picks the call.
        if "by_value" in row:
            picked = row["by_value"].get(_value_key(value))
            if picked is None:
                raise MappingTableError(
                    f"{device_type}.{action}: no call for value {value!r}")
            return picked[0], None
        # 2. A single row: the call, with the value as argument when it takes one.
        command, takes_value = row["single"]
        return command, (value if takes_value else None)

    def gold_call(self, device_type: str, action: str, value: Any) -> str:
        """The gold call text (``set_brightness(20)``) of an action invocation."""
        row = self._commands.get((device_type, action))
        if row is None:
            raise MappingTableError(f"{device_type} has no action {action!r}")
        # 1. A mapping row holds the finished call per value.
        if "by_value" in row:
            if _value_key(value) not in row["gold"]:
                raise MappingTableError(
                    f"{device_type}.{action}: no call for value {value!r}")
            return str(row["gold"][_value_key(value)])
        # 2. A template: substitute the value (or none).
        return str(row["gold"]).replace(_VALUE, format_arg(value))

    def gold_string(self, device_id: str, device_type: str, action: str,
                    value: Any) -> str:
        """The gold token ``room.device.call`` of an invocation on ``device_id``.

        The room is the device id minus its ``_<device type>`` suffix; the vacuum
        robot (room ``household``) has none: ``vacuum_robot.set_mode(auto)``.
        """
        suffix = f"_{device_type}"
        if not device_id.endswith(suffix):
            raise MappingTableError(f"{device_id!r} does not end with {suffix!r}")
        room = device_id[: -len(suffix)]
        call = self.gold_call(device_type, action, value)
        if room == HOUSEHOLD_ROOM:
            return f"{device_type}.{call}"
        return f"{room}.{device_type}.{call}"

    def action_for_command(self, device_type: str, command: str, arg: Any
                           ) -> Tuple[str, Any]:
        """The reverse: ``(action, value)`` a gold ``command(arg)`` stands for.

        Raises ``MappingTableError`` when no action of the device type issues it.
        """
        for (dtype, action), row in self._commands.items():
            if dtype != device_type:
                continue
            # 1. A mapping row: which value's call names this command.
            if "by_value" in row:
                for key, (name, _takes) in row["by_value"].items():
                    if name == command:
                        return action, _typed_key(key)
            # 2. A single row: the command names the action, the argument is the value.
            elif row["single"][0] == command:
                return action, (arg if row["single"][1] else None)
        raise MappingTableError(f"{device_type}: no action issues {command!r}")

    # -- review status ----------------------------------------------------------------

    def unsettled(self) -> Dict[str, int]:
        """Rows per table still awaiting review (not ``approved`` or ``auto``)."""
        out: Dict[str, int] = {}
        for name in TABLES:
            pending = len(_rows(name, settled_only=False)) - len(_rows(name))
            if pending:
                out[name] = pending
        return out


def _typed_key(key: str) -> Any:
    """A mapping key back to the value it stands for (``"true"`` -> ``True``)."""
    if key == "true":
        return True
    if key == "false":
        return False
    return key


@lru_cache(maxsize=1)
def load_mappings() -> Mappings:
    """The tables, read once per process."""
    return Mappings()
