"""Gold-consistent state engine for HomeBench homes (pure Python, no I/O at import).

HomeBench labels were produced by a generator whose device classes carry
preconditions ("cannot set brightness while the device is off"), but the labels
themselves ignore them: 6,472 gold calls set an attribute on a device that is off.
This engine therefore follows the *gold*, not the generator:

* no preconditions: a ``set_*`` on an ``off`` device succeeds, ``turn_on`` on an
  ``on`` device succeeds;
* validation only where the gold never contradicts it: enum values must be in the
  device's ``options``; numeric values must lie in the device's range, where the
  range of ``heating.temperature`` is corrected to 16-30 (the data declares 30-100
  on 99 of 129 heaters, yet every gold heating value is 16-30);
* no side effects between attributes, no clock, no physics;
* gold vacuum-robot calls the home does not offer (``start``, ``stop``, ``pause``,
  ``charge``: 70 calls) are *reported* as ``unreachable`` and never exposed.

A home is the ``initial_home_config`` dictionary written by
``tests/homebench/convert.py``: ``rooms -> {room: {"state": {}, "devices": [...]}}``,
each device ``{"device_id", "device_type", "attributes", "constraints"}`` with a
flat ``attributes`` map that includes ``state``.

Worked example::

    engine = HomeBenchEngine.from_file("homes/home75.json")
    result = engine.apply("bathroom_light", "set_brightness", 20)
    # ApplyResult(status="applied", attribute="brightness", value=20)
    engine.apply("bathroom_light", "set_brightness", 101)   # raises EngineError
    engine.reset()                                          # back to the home file
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -- constants ------------------------------------------------------------------

#: Synthetic room that holds the vacuum robot (HomeBench gives it no room).
HOUSEHOLD_ROOM = "household"

#: Fixed clock: HomeBench has no time, but the bench's home config carries one.
BASE_TIME = "2025-01-01 08:00:00"
TICK_INTERVAL = 0.1

#: The 12 HomeBench rooms; a vacuum ``set_cleaning_area`` argument is one of these
#: with the underscore dropped (``storeroom``, ``masterbedroom``).
ROOMS: Tuple[str, ...] = (
    "master_bedroom", "guest_bedroom", "living_room", "ding_room", "study_room",
    "kitchen", "bathroom", "foyer", "corridor", "balcony", "garage", "store_room",
)
CLEANING_AREAS: Tuple[str, ...] = tuple(room.replace("_", "") for room in ROOMS)

#: Commands that only set ``state``: command -> new state value.
STATE_COMMANDS: Dict[str, str] = {
    "turn_on": "on", "turn_off": "off",
    "open": "open", "close": "closed",
    "play": "playing", "pause": "paused", "stop": "stopped",
    "pack": "empty",
}

#: Which state commands each device type offers (the home's method list agrees).
DEVICE_STATE_COMMANDS: Dict[str, frozenset] = {
    "light": frozenset({"turn_on", "turn_off"}),
    "air_conditioner": frozenset({"turn_on", "turn_off"}),
    "heating": frozenset({"turn_on", "turn_off"}),
    "fan": frozenset({"turn_on", "turn_off"}),
    "air_purifiers": frozenset({"turn_on", "turn_off"}),
    "humidifier": frozenset({"turn_on", "turn_off"}),
    "dehumidifiers": frozenset({"turn_on", "turn_off"}),
    "aromatherapy": frozenset({"turn_on", "turn_off"}),
    "water_heater": frozenset({"turn_on", "turn_off"}),
    "curtain": frozenset({"open", "close"}),
    "blinds": frozenset({"open", "close"}),
    "garage_door": frozenset({"open", "close"}),
    "media_player": frozenset({"play", "pause", "stop"}),
    "trash": frozenset({"pack"}),
    "vacuum_robot": frozenset(),
}

#: Attributes a ``set_<attribute>`` command can write, per device type. Read-only
#: attributes (``battery``, ``tank``) are absent on purpose: HomeBench has no setter.
#: ``set_song``, ``set_artist`` and ``set_style`` never occur in gold and have no
#: attribute to assert, so they are not modelled (recorded in the mapping notes).
SETTABLE_ATTRIBUTES: Dict[str, frozenset] = {
    "light": frozenset({"brightness", "color"}),
    "air_conditioner": frozenset({"temperature", "mode", "fan_speed", "swing"}),
    "heating": frozenset({"temperature", "mode", "fan_speed"}),
    "fan": frozenset({"speed", "swing"}),
    "air_purifiers": frozenset({"mode", "fan_speed"}),
    "humidifier": frozenset({"mode", "intensity"}),
    "dehumidifiers": frozenset({"mode", "intensity"}),
    "aromatherapy": frozenset({"intensity", "interval"}),
    "water_heater": frozenset({"temperature", "mode"}),
    "curtain": frozenset({"degree"}),
    "media_player": frozenset({"volume"}),
    "vacuum_robot": frozenset({"mode", "cleaning_area"}),
}

#: Gold vacuum calls the home's method list does not contain; reported, not exposed.
UNREACHABLE_COMMANDS: Dict[str, frozenset] = {
    "vacuum_robot": frozenset({"start", "stop", "pause", "charge"}),
}

#: Declared ranges the gold contradicts: (device_type, attribute) -> (low, high).
RANGE_CORRECTIONS: Dict[Tuple[str, str], Tuple[int, int]] = {
    ("heating", "temperature"): (16, 30),
}

STATUS_APPLIED = "applied"
STATUS_UNREACHABLE = "unreachable"


# -- errors and results -----------------------------------------------------------


class EngineError(ValueError):
    """A command the engine refuses: unknown device or command, bad value."""


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of one ``apply``: ``applied`` with the attribute written, or
    ``unreachable`` (a gold call the home does not offer; the state is unchanged)."""

    status: str
    attribute: Optional[str]
    value: Any


# -- pure helpers (shared with the converter) ----------------------------------------


def make_device_id(room: Optional[str], device: str) -> str:
    """Return the device id for a gold ``room.device`` pair: ``<room>_<device>``.

    A ``None`` room (the vacuum robot) maps to the synthetic ``household`` room, so
    the gold string is a pure function of ``(device_id, command, value)``.
    """
    return f"{room or HOUSEHOLD_ROOM}_{device}"


def command_effect(command: str, arg: Any = None) -> Optional[Tuple[str, Any]]:
    """Return ``(attribute, value)`` a successful command writes, or ``None``.

    ``None`` means the command writes nothing the engine models (an unreachable
    vacuum command); the converter marks such gold parts as unreachable.
    """
    # 1. State commands write ``state``; setters write the attribute they name.
    if command in STATE_COMMANDS:
        return "state", STATE_COMMANDS[command]
    if command.startswith("set_"):
        return command[len("set_"):], arg
    return None


def corrected_constraints(
    device_type: str, attributes: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Turn HomeBench's declared attribute blocks into engine constraints.

    Numeric blocks (``lowest``/``highest``, strings or ints) become integer ranges
    with ``RANGE_CORRECTIONS`` applied; enum blocks keep their ``options``; the light
    ``color`` block becomes a 3-channel 0-255 constraint. This is the single place
    the heating correction lives: the converter writes its output into the home file
    and the engine validates against what the file says.
    """
    constraints: Dict[str, Dict[str, Any]] = {}
    for name, block in attributes.items():
        # 1. Enum attribute: the options are the whole constraint.
        if "options" in block:
            constraints[name] = {"options": list(block["options"])}
        # 2. Numeric attribute: parse the (string or int) bounds, then correct.
        elif "lowest" in block:
            low, high = int(float(block["lowest"])), int(float(block["highest"]))
            low, high = RANGE_CORRECTIONS.get((device_type, name), (low, high))
            constraints[name] = {"lowest": low, "highest": high}
        # 3. RGB colour: the only list-valued attribute.
        elif isinstance(block.get("value"), (list, tuple)):
            constraints[name] = {"channels": 3, "lowest": 0, "highest": 255}
    return constraints


def load_home_file(path: Any) -> Dict[str, Any]:
    """Read a ``homes/home<id>.json`` file (an ``initial_home_config``)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# -- the engine -------------------------------------------------------------------


class HomeBenchEngine:
    """One home's mutable state with gold-consistent command semantics.

    Built from an ``initial_home_config``; ``reset`` restores it. Devices are
    addressed by ``device_id`` (``<room>_<device>``), which is what the TD server
    will mint in a later step, so no lookup table sits between a gold call and an
    engine command.
    """

    def __init__(self, home_config: Dict[str, Any]) -> None:
        """Keep a private copy of ``home_config`` as the reset point and load it."""
        self._initial = copy.deepcopy(home_config)
        self._rooms: Dict[str, Any] = {}
        self._devices: Dict[str, Dict[str, Any]] = {}
        self.reset()

    @classmethod
    def from_file(cls, path: Any) -> "HomeBenchEngine":
        """Build an engine from a ``homes/home<id>.json`` file."""
        return cls(load_home_file(path))

    # -- lifecycle and reads ---------------------------------------------------

    def reset(self, home_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Restore the home (or load ``home_config`` as the new home) and return
        its state in the same shape as ``home_state``."""
        # 1. A new config replaces the reset point; none means "back to the file".
        if home_config is not None:
            self._initial = copy.deepcopy(home_config)
        # 2. Rebuild rooms from the pristine copy and index devices by id.
        self._rooms = copy.deepcopy(self._initial["rooms"])
        self._devices = {
            device["device_id"]: device
            for room in self._rooms.values()
            for device in room["devices"]
        }
        return self.home_state()

    def home_state(self) -> Dict[str, Any]:
        """Return the whole state: ``{current_tick, current_time, rooms}``."""
        return {
            "current_tick": 0,
            "current_time": self._initial.get("base_time", BASE_TIME),
            "rooms": copy.deepcopy(self._rooms),
        }

    def device_ids(self) -> List[str]:
        """Return every device id in the home."""
        return list(self._devices)

    def device(self, device_id: str) -> Dict[str, Any]:
        """Return a copy of one device (``device_id``, ``device_type``, ...)."""
        if device_id not in self._devices:
            raise EngineError(f"unknown device {device_id!r}")
        return copy.deepcopy(self._devices[device_id])

    def exposed_commands(self, device_id: str) -> List[str]:
        """Return the commands the home offers on a device.

        State commands of the device type plus ``set_<attribute>`` for each settable
        attribute the device actually has: a light with no ``brightness`` has no
        ``set_brightness``, which is what makes a brightness request on it
        ``unexist_attribute``. Unreachable vacuum commands are not included.
        """
        device = self._devices.get(device_id)
        if device is None:
            raise EngineError(f"unknown device {device_id!r}")
        dtype = device["device_type"]
        commands = sorted(DEVICE_STATE_COMMANDS.get(dtype, ()))
        settable = SETTABLE_ATTRIBUTES.get(dtype, frozenset())
        commands += sorted(f"set_{a}" for a in settable if a in device["attributes"])
        return commands

    # -- commands --------------------------------------------------------------

    def apply(self, device_id: str, command: str, arg: Any = None) -> ApplyResult:
        """Apply one command to one device, or raise ``EngineError``.

        Unreachable vacuum commands return ``status="unreachable"`` and leave the
        state alone. Nothing else is rejected for the device's current state.
        """
        # 1. Resolve the device.
        device = self._devices.get(device_id)
        if device is None:
            raise EngineError(f"unknown device {device_id!r}")
        dtype = device["device_type"]
        # 2. Gold calls the home does not offer are reported, not applied.
        if command in UNREACHABLE_COMMANDS.get(dtype, ()):
            return ApplyResult(STATUS_UNREACHABLE, None, None)
        # 3. The command must be one the home offers on this device.
        if command not in self.exposed_commands(device_id):
            raise EngineError(f"{device_id} offers no command {command!r}")
        effect = command_effect(command, arg)
        if effect is None:  # pragma: no cover - exposed commands always have one
            raise EngineError(f"{command!r} has no modelled effect")
        attribute, value = effect
        # 4. State commands take no argument; setters validate theirs.
        if attribute == "state":
            if arg is not None:
                raise EngineError(f"{command!r} takes no argument, got {arg!r}")
        else:
            value = _validate(device, attribute, arg)
        # 5. Write it. No precondition on the device's current state.
        device["attributes"][attribute] = value
        return ApplyResult(STATUS_APPLIED, attribute, value)


def _validate(device: Dict[str, Any], attribute: str, arg: Any) -> Any:
    """Return ``arg`` as the value to store, or raise ``EngineError``.

    Enum attributes need a member of ``options``; numeric ones an ``int`` (not a
    bool) inside ``[lowest, highest]``; colour a list of three 0-255 ints.
    """
    rule = device.get("constraints", {}).get(attribute)
    where = f"{device['device_id']}.{attribute}"
    if rule is None:
        raise EngineError(f"{where} declares no constraint")
    # 1. Enum: membership in the declared options.
    if "options" in rule:
        if arg not in rule["options"]:
            raise EngineError(f"{where}: {arg!r} not in {rule['options']}")
        return arg
    low, high = rule["lowest"], rule["highest"]
    # 2. RGB colour: a fixed-length list of in-range ints.
    if "channels" in rule:
        ok = isinstance(arg, (list, tuple)) and len(arg) == rule["channels"]
        if not ok or not all(_is_int(c) and low <= c <= high for c in arg):
            raise EngineError(f"{where}: {arg!r} is not an RGB triple")
        return list(arg)
    # 3. Numeric: an int inside the (corrected) range.
    if not _is_int(arg) or not low <= arg <= high:
        raise EngineError(f"{where}: {arg!r} outside [{low}, {high}]")
    return arg


def _is_int(value: Any) -> bool:
    """Return True for a real int (``bool`` is excluded: ``True`` is not a level)."""
    return isinstance(value, int) and not isinstance(value, bool)
