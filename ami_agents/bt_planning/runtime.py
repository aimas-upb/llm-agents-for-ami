"""What a dependency plan keeps fresh around its tree: readings, the time, derived values.

A dependency tree waits on blackboard keys -- "the washer's remaining time is
under 23 minutes", "it is 1:39 PM", "R has at most as long to go as F takes"
(`interaction_solver/utils/dependency_compiler.py`). Its nodes only look at the
keys; something has to keep them current, before every tick, whatever part of
the tree is running. That is this runtime, described by the plan entry's
`runtime` section and run by the plan's executor before each tick:

    "runtime": {
      "reads":   [{"key": "/artifacts/washer_1/countdownTime",
                   "url": ".../properties/countdownTime", "field": null}],
      "derived": [{"key": "/goals/x/G1/due", "op": "timestamp_minus",
                   "inputs": ["/artifacts/washer_1/endTime"], "args": {"seconds": 1380}}],
      "clock":   true
    }

    reads     a property per key, read over HTTP; `field` picks one field of
              an object value. A key holds the last value read.
    clock     CLOCK_KEY gets the configured clock's time (`shared/utils/clock.py`):
              the system time, or the simulator's virtual time.
    derived   a named op (DERIVED_OPS) over other keys, recomputed every tick;
              skipped while any input is unset.

Reads and the clock are fetched on the BT nodes' thread pool, so a tick never
waits on the network: each refresh submits what is not already in flight and
writes what has landed. `synchronous=True` (tests, scripts) fetches in place.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional

import py_trees
from py_trees.common import Access

from ..shared.utils.clock import Clock, parse_moment, to_seconds
from .nodes.async_support import get_executor
from .nodes.dependency_nodes import CLOCK_KEY
from .nodes.http_client import HTTPClient, HTTPClientConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Derived values: named ops, (inputs, args, now) -> value
# --------------------------------------------------------------------------

def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("a boolean is not a number")
    return float(value)


def _to_seconds(inputs: List[Any], args: Dict[str, Any], now: Optional[float]) -> float:
    """A duration in the property's unit -> seconds (`factor`: seconds per unit)."""
    return _number(inputs[0]) * float(args.get("factor", 1.0))


def _timestamp_minus(inputs: List[Any], args: Dict[str, Any], now: Optional[float]) -> float:
    """An event's time minus an offset: when to act ahead of it."""
    moment = parse_moment(inputs[0])
    if moment is None:
        raise ValueError(f"{inputs[0]!r} is not a time")
    return moment - float(args.get("seconds", 0.0))


def _remaining_from_timestamp(inputs: List[Any], args: Dict[str, Any],
                              now: Optional[float]) -> float:
    """Seconds from now until an end time."""
    moment = parse_moment(inputs[0])
    if moment is None or now is None:
        raise ValueError("needs a time and the clock")
    return moment - now


def _remaining_from_duration(inputs: List[Any], args: Dict[str, Any],
                             now: Optional[float]) -> float:
    """Seconds left of a run that takes `inputs[0]` (in its unit) and started at
    `inputs[1]` (a clock time)."""
    if now is None:
        raise ValueError("needs the clock")
    return _number(inputs[0]) * float(args.get("factor", 1.0)) - (now - _number(inputs[1]))


def _abs_deviation(inputs: List[Any], args: Dict[str, Any], now: Optional[float]) -> float:
    """How far a reading is from a target."""
    return abs(_number(inputs[0]) - _number(args["target"]))


DERIVED_OPS: Dict[str, Callable[[List[Any], Dict[str, Any], Optional[float]], Any]] = {
    "to_seconds": _to_seconds,
    "timestamp_minus": _timestamp_minus,
    "remaining_from_timestamp": _remaining_from_timestamp,
    "remaining_from_duration": _remaining_from_duration,
    "abs_deviation": _abs_deviation,
}


# --------------------------------------------------------------------------
# The runtime
# --------------------------------------------------------------------------

class PlanRuntime:
    """Keeps a plan's runtime keys current; call `before_tick()` before each tick."""

    def __init__(self, spec: Optional[Dict[str, Any]], clock: Optional[Clock] = None,
                 http_client: Optional[HTTPClient] = None, synchronous: bool = False) -> None:
        spec = spec or {}
        self.reads: List[Dict[str, Any]] = list(spec.get("reads") or [])
        self.derived: List[Dict[str, Any]] = list(spec.get("derived") or [])
        self.clock = clock if spec.get("clock") else None
        self.http = http_client or HTTPClient(config=HTTPClientConfig())
        self.synchronous = synchronous
        self._in_flight: Dict[str, Future] = {}
        self._clock_warned = False

        keys = ([r["key"] for r in self.reads] + [d["key"] for d in self.derived]
                + ([CLOCK_KEY] if self.clock else []))
        inputs = [k for d in self.derived for k in d.get("inputs") or []]
        self.blackboard = py_trees.blackboard.Client(name="plan-runtime")
        for key in set(keys):
            self.blackboard.register_key(key=key, access=Access.WRITE)
        for key in set(inputs) | {CLOCK_KEY}:
            if key not in keys:
                self.blackboard.register_key(key=key, access=Access.READ)

    @property
    def empty(self) -> bool:
        return not (self.reads or self.derived or self.clock)

    # -- fetching ----------------------------------------------------------

    def _read(self, read: Dict[str, Any]) -> Any:
        response = self.http.get(read["url"])
        if not getattr(response, "is_success", False):
            raise RuntimeError(f"GET {read['url']} -> {getattr(response, 'status_code', '?')}")
        value = response.body
        field = read.get("field")
        if field and field != "NA" and isinstance(value, dict):
            value = value.get(field)
        return value

    def _fetch(self, slot: str, fn: Callable[[], Any], on_value: Callable[[Any], None]) -> None:
        """Run `fn` for `slot` -- in place, or on the pool without waiting -- and
        hand its value to `on_value` once it has one."""
        if self.synchronous:
            try:
                on_value(fn())
            except Exception as exc:  # noqa: BLE001 - a failed read keeps the last value
                self._failed(slot, exc)
            return
        future = self._in_flight.get(slot)
        if future is None:
            self._in_flight[slot] = get_executor().submit(fn)
            return
        if not future.done():
            return
        del self._in_flight[slot]
        try:
            on_value(future.result())
        except Exception as exc:  # noqa: BLE001
            self._failed(slot, exc)
        self._in_flight[slot] = get_executor().submit(fn)     # the next refresh

    def _failed(self, slot: str, exc: Exception) -> None:
        if slot == CLOCK_KEY:
            if not self._clock_warned:
                logger.warning(f"Plan clock unreadable, time-based waits hold: {exc}")
                self._clock_warned = True
        else:
            logger.debug(f"Runtime read {slot} failed: {exc}")

    # -- the refresh -------------------------------------------------------

    def before_tick(self) -> None:
        """Refresh the clock and the readings, then recompute the derived keys."""
        if self.clock is not None:
            self._fetch(CLOCK_KEY, lambda: to_seconds(self.clock.now()),
                        lambda value: self.blackboard.set(CLOCK_KEY, value))
        for read in self.reads:
            self._fetch(read["key"], lambda r=read: self._read(r),
                        lambda value, k=read["key"]: self.blackboard.set(k, value))
        self._derive()

    def _derive(self) -> None:
        now = self._get(CLOCK_KEY)
        for item in self.derived:
            values = [self._get(key) for key in item.get("inputs") or []]
            if any(v is None for v in values):
                continue
            try:
                value = DERIVED_OPS[item["op"]](values, item.get("args") or {}, now)
            except (ValueError, TypeError, KeyError) as exc:
                logger.debug(f"Derived {item['key']} skipped: {exc}")
                continue
            self.blackboard.set(item["key"], value)

    def _get(self, key: str) -> Any:
        try:
            return self.blackboard.get(key)
        except KeyError:
            return None
