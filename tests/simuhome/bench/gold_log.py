"""What our pipeline did to the home, as HomeBench gold tokens.

`toollog.py` turns the pipeline's requests into SimuHome tool calls; this module is its
HomeBench counterpart. Every action request the plans sent (`execution.actions` of an
episode record, written by `execution.PostRecorder`) that the home accepted is turned
back into the string HomeBench's gold uses, `room.device.op(args)`, through the reverse
mapping of the TD layer (`HomeBench/mappings.py`, the `command_map` table). The result
is the "prediction" the official metrics compare with the gold (`homebench_metrics.py`).

    action request                                             gold token
    POST .../living_room/artifacts/living_room_light/          living_room.light.set_brightness(20)
         actions/brightness   {"value": 20}
    POST .../living_room/artifacts/living_room_light/          living_room.light.turn_off()
         actions/onOff        {"value": false}
    POST .../household/artifacts/household_vacuum_robot/       vacuum_robot.set_mode(auto)
         actions/mode         {"value": "auto"}

Only requests the home accepted (a 2xx answer) count: a refused request changed nothing.
An accepted request that no table row names (a malformed record, or a table that
changed) is not dropped silently; it is listed in `GoldLog.unmapped` with the reason.
The device type is not in the URL: it is the device id without its room prefix
(`living_room_light` in `living_room`), the id convention of the TD layer (plan 3.3).

Worked example::

    log = gold_log(record["execution"]["actions"])
    log.tokens      # ["living_room.light.set_brightness(20)", "kitchen.light.turn_off()"]
    log.unmapped    # [] -- every accepted action had a gold form
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ami_agents.environment.integration.HomeBench.mappings import (
    MappingTableError, Mappings, load_mappings,
)
from tests.simuhome.bench.toollog import parse_url


@dataclass
class GoldLog:
    """The gold tokens of one episode's accepted actions, and what could not be named.

    ``tokens`` keeps the order the requests were sent in, repeats included (the official
    metric counts a repeated call twice). ``unmapped`` holds ``{"url", "reason"}`` for
    each accepted action with no gold form. ``refused`` counts the requests the home
    answered with an error (they are not tokens).
    """

    tokens: List[str] = field(default_factory=list)
    unmapped: List[Dict[str, Any]] = field(default_factory=list)
    refused: int = 0


def action_value(parameters: Optional[Dict[str, Any]]) -> Any:
    """The value an action request carried: ``{"value": v}``, or the aliases HBTD
    accepts (``input``, ``payload``); None when there is none.

    An integral float (``20.0``) is an int, as HBTD's own value check makes it, so
    ``set_brightness(20.0)`` never appears in a token.
    """
    # 1. The same lookup order as the server's `_action_value`.
    body = parameters or {}
    value = body.get("value", body.get("input", body.get("payload")))
    # 2. JSON has one number type; the engine stores an int.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def action_token(action: Dict[str, Any], mappings: Mappings) -> str:
    """The gold token of one recorded action request.

    Raises ``MappingTableError`` when the URL is not an action URL of the TD layer or
    no table row gives the (device type, action, value) a gold form.
    """
    # 1. The room, device id and action name come from the URL (SHTD's layout).
    parts = parse_url(action.get("url")) or {}
    room, device_id, name = parts.get("room"), parts.get("device"), parts.get("name")
    if parts.get("kind") != "actions" or not (room and device_id and name):
        raise MappingTableError(f"not an action URL: {action.get('url')!r}")
    # 2. The device type is the id minus its room prefix.
    prefix = f"{room}_"
    if not device_id.startswith(prefix) or len(device_id) == len(prefix):
        raise MappingTableError(f"{device_id!r} does not start with {prefix!r}")
    device_type = device_id[len(prefix):]
    # 3. The reverse mapping names the token (the vacuum's has no room).
    return mappings.gold_string(device_id, device_type, name,
                                action_value(action.get("parameters")))


def gold_log(actions: Optional[Iterable[Dict[str, Any]]],
             mappings: Optional[Mappings] = None) -> GoldLog:
    """The gold tokens of the accepted actions in ``actions`` (module docstring).

    ``mappings`` defaults to the repo's tables; a test may pass its own.
    """
    mappings = mappings or load_mappings()
    log = GoldLog()
    for action in actions or []:
        # 1. A refused request changed nothing: counted, not a token.
        if not action.get("success"):
            log.refused += 1
            continue
        # 2. An accepted one is named, or listed with the reason it could not be.
        try:
            log.tokens.append(action_token(action, mappings))
        except MappingTableError as exc:
            log.unmapped.append({"url": action.get("url"), "reason": str(exc)})
    return log
