"""The few nodes dependency plans need that py_trees does not provide.

Dependency plans (`interaction_solver/utils/dependency_compiler.py`) are built
from py_trees built-ins (checks, waits, Repeat, Condition, Inverter, ...). Three
things are missing from them:

    CompareBlackboardKeys   compare two blackboard keys (py_trees 2.3 has no
                            CompareBlackboardVariables)
    ClockTimer              wait a span of time -- of the plan's clock
    WaitUntilTime           wait for a time of day -- of the plan's clock
    StampTime               record the plan clock's current time in a key

The clock nodes never read a clock themselves: the plan's runtime writes the
current time to CLOCK_KEY before every tick (`bt_planning/runtime.py`), from the
configured clock -- the system time, or the simulator's virtual time
(`shared/utils/clock.py`). So one tick sees one time, and a plan waits in
whatever time its environment runs on. Without a time on the blackboard they
stay RUNNING: better late than firing on a wrong time.
"""

from __future__ import annotations

import operator
from datetime import timedelta
from typing import Any, Callable, Optional

import py_trees
from py_trees.common import Access, Status

from ...shared.utils.clock import from_seconds, to_seconds

# Where the plan's runtime writes the current time (seconds, `clock.to_seconds`).
CLOCK_KEY = "/clock/now"

# Comparison operators by name: how they travel in the JSON IR.
OPERATORS = {"eq": operator.eq, "ne": operator.ne, "gt": operator.gt,
             "ge": operator.ge, "lt": operator.lt, "le": operator.le}
COMBINERS = {"and_": operator.and_, "or_": operator.or_}


class _ClockReader(py_trees.behaviour.Behaviour):
    """A node that reads CLOCK_KEY."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client(name=name)
        self.blackboard.register_key(key=CLOCK_KEY, access=Access.READ)

    def _now(self) -> Optional[float]:
        try:
            return float(self.blackboard.get(CLOCK_KEY))
        except (KeyError, TypeError, ValueError):
            return None


class ClockTimer(_ClockReader):
    """RUNNING for `duration` seconds of the plan's clock, then SUCCESS. The
    span counts from the node's first tick in each run."""

    def __init__(self, name: str, duration: float) -> None:
        super().__init__(name)
        self.duration = max(0.0, float(duration))
        self._start: Optional[float] = None

    def initialise(self) -> None:
        self._start = None

    def update(self) -> Status:
        now = self._now()
        if now is None:
            return Status.RUNNING
        if self._start is None:
            self._start = now
        elapsed = now - self._start
        self.feedback_message = f"{elapsed:.0f}s of {self.duration:.0f}s"
        return Status.SUCCESS if elapsed >= self.duration else Status.RUNNING


class WaitUntilTime(_ClockReader):
    """RUNNING until the plan's clock reaches a time of day ("HH:MM[:SS]"),
    resolved on the first tick: today, or tomorrow if it has already passed."""

    def __init__(self, name: str, time_of_day: str) -> None:
        super().__init__(name)
        parts = [int(p) for p in str(time_of_day).split(":")]
        self.hour, self.minute = parts[0], parts[1]
        self.second = parts[2] if len(parts) > 2 else 0
        self._due: Optional[float] = None

    def initialise(self) -> None:
        self._due = None

    def update(self) -> Status:
        now = self._now()
        if now is None:
            return Status.RUNNING
        if self._due is None:
            today = from_seconds(now).replace(hour=self.hour, minute=self.minute,
                                              second=self.second, microsecond=0)
            due = to_seconds(today)
            self._due = due if due >= now else to_seconds(today + timedelta(days=1))
        self.feedback_message = f"due at {from_seconds(self._due)}"
        return Status.SUCCESS if now >= self._due else Status.RUNNING


class StampTime(_ClockReader):
    """Write the plan clock's current time to `key` (e.g. when a goal starts)."""

    def __init__(self, name: str, key: str) -> None:
        super().__init__(name)
        self.key = key
        self.blackboard.register_key(key=key, access=Access.WRITE)

    def update(self) -> Status:
        now = self._now()
        if now is None:
            return Status.RUNNING
        self.blackboard.set(self.key, now)
        return Status.SUCCESS


class CompareBlackboardKeys(py_trees.behaviour.Behaviour):
    """SUCCESS if `key1 <op> key2` holds, FAILURE if not or if either is unset."""

    def __init__(self, name: str, key1: str, key2: str,
                 op: Callable[[Any, Any], bool]) -> None:
        super().__init__(name=name)
        self.key1, self.key2, self.op = key1, key2, op
        self.blackboard = self.attach_blackboard_client(name=name)
        for key in (key1, key2):
            self.blackboard.register_key(key=key, access=Access.READ)

    def update(self) -> Status:
        try:
            left, right = self.blackboard.get(self.key1), self.blackboard.get(self.key2)
        except KeyError:
            return Status.FAILURE
        try:
            return Status.SUCCESS if self.op(left, right) else Status.FAILURE
        except TypeError:
            return Status.FAILURE
