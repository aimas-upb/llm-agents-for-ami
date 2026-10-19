"""Executing an episode's plans against the simulator, through SHTD.

All the episode's trees run together, as the UA runs its plans side by side.

    immediate   no dependency plan among them: the UA's loop, one tick of every
                tree, then `min_tick_yield`, until all have concluded or
                `timeout` real seconds pass. The UA's tick budget is not applied
                (the bench is slower per tick than the agent loop); exceeding
                it is reported as `over_tick_budget`.

    lockstep    some plan waits on time or events (dependency plans): the bench
                drives the simulator's clock instead of waiting for it --

                    refresh the runtimes; tick every tree
                    wait until no action is in flight      (its POST was applied)
                    fast_forward_to(now + step)            (step = 10 s virtual)

                so a 40-minute plan runs in seconds, actions fire at most one
                step late and never early, and device dynamics evolve inside
                each jump as in real time. It stops when every tree has
                concluded and the last observation point is passed, or at the
                horizon. Tests only: the agents always run on their clock.

Observation points are where the home's state is recorded for the evaluator:
every step that falls inside an eval goal's window, the window's edges and its
`at_tick` exactly. They decide where states are kept, never what a plan does.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import py_trees
from py_trees.common import Status

from ami_agents.bt_planning.nodes.affordance_nodes import ActionAffordanceNode
from ami_agents.bt_planning.nodes.http_client import HTTPClient, HTTPError
from ami_agents.bt_planning.nodes.registry import compile_node
from ami_agents.bt_planning.runtime import PlanRuntime

from .toollog import ToolLog

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# Ticks at one virtual moment while requests are pending (a sequence of
# actions completes within one moment; this bounds a tree that never settles).
MAX_TICKS_PER_MOMENT = 20


def _compile(spec: Dict[str, Any]):
    return compile_node(spec, _compile)


@dataclass
class Plan:
    """One plan entry being executed."""

    index: int
    entry: Dict[str, Any]
    tree: Any = None
    runtime: Optional[PlanRuntime] = None
    status: str = "RUNNING"
    ticks: int = 0
    error: Optional[str] = None


@dataclass
class Execution:
    """What execution amounted to."""

    mode: str                                   # immediate | lockstep | scheduled | none
    plans: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    snapshots: List[Dict[str, Any]] = field(default_factory=list)   # lockstep
    # Lockstep: every change of the attributes the eval goals check, over the
    # whole run -- [{tick, values: {"device|attribute": value}}], first entry
    # at the start. Tells a value reached too early or late from one never set.
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    real_seconds: float = 0.0
    virtual_seconds: Optional[float] = None
    steps: int = 0
    horizon_tick: Optional[int] = None
    end_tick: Optional[int] = None
    # Mode-specific facts (scheduled: the workflows, syncs, real reads).
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "plans": self.plans, "actions": self.actions,
                "real_seconds": self.real_seconds, "virtual_seconds": self.virtual_seconds,
                "steps": self.steps, "horizon_tick": self.horizon_tick,
                "end_tick": self.end_tick, "snapshot_ticks": [s["tick"] for s in self.snapshots],
                **({"details": self.details} if self.details else {})}


class StateClock:
    """The simulator's time as of the last state the bench saw (lockstep):
    the plans' clock reads the same moment the step ended on."""

    def __init__(self) -> None:
        self.moment: Optional[datetime] = None

    def update(self, state: Dict[str, Any]) -> None:
        if state.get("current_time"):
            self.moment = datetime.strptime(state["current_time"], TIME_FORMAT)

    def now(self) -> datetime:
        if self.moment is None:
            from ami_agents.shared.utils.clock import ClockError
            raise ClockError("no simulator state yet")
        return self.moment


# --------------------------------------------------------------------------
# Common parts
# --------------------------------------------------------------------------

def _prepare(entries: Sequence[Dict[str, Any]], clock, tools: ToolLog) -> List[Plan]:
    """Compile every entry with a tree; log the reads its tree and runtime make."""
    py_trees.blackboard.Blackboard.clear()
    plans = []
    for index, entry in enumerate(entries):
        if not isinstance(entry.get("tree"), dict) or not entry["tree"]:
            continue
        plan = Plan(index=index, entry=entry)
        try:
            plan.tree = _compile(entry["tree"])
            plan.tree.setup_with_descendants()
            if entry.get("runtime"):
                plan.runtime = PlanRuntime(entry["runtime"], clock=clock, synchronous=True)
                for read in entry["runtime"].get("reads") or []:
                    tools.read(read.get("url"), None, "plan_runtime")
            for node in _walk(entry["tree"]):
                if node.get("property_url"):
                    tools.read(node["property_url"], None, "plan")
        except Exception as exc:  # noqa: BLE001 - a tree that does not compile fails
            plan.status, plan.error = "FAILURE", f"compile: {exc}"
        plans.append(plan)
    return plans


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for child in node.get("children") or []:
            yield from _walk(child)


def _tick(plans: List[Plan]) -> None:
    for plan in plans:
        if plan.status != "RUNNING" or plan.tree is None:
            continue
        try:
            if plan.runtime is not None:
                plan.runtime.before_tick()
            plan.tree.tick_once()
            plan.ticks += 1
            plan.status = plan.tree.status.name
        except Exception as exc:  # noqa: BLE001 - one plan's crash fails that plan
            plan.status, plan.error = "FAILURE", f"tick: {exc}"


class PostRecorder:
    """Every action request the plans send, recorded where it leaves for the
    home: `HTTPClient.post`, the call every action node makes.

    Reading results off the nodes misses some: an action whose guard is
    re-checked after its request went out (a selector "already off? else turn
    off") is skipped once the guard holds, and its answer is never consumed --
    yet the device was switched. What counts is what the home received.
    Each record carries the moment it was sent (`moment`, kept current by the
    loop: the tick and virtual time in lockstep)."""

    def __init__(self, plans: List[Plan], tools: ToolLog) -> None:
        self.tools = tools
        self.names: Dict[str, Tuple[int, str]] = {}
        for plan in plans:
            if plan.tree is not None:
                for node in plan.tree.iterate():
                    if isinstance(node, ActionAffordanceNode):
                        self.names.setdefault(node.action_url, (plan.index, node.name))
        self.moment: Dict[str, Any] = {}
        self.actions: List[Dict[str, Any]] = []
        self.started = time.perf_counter()
        self._lock = threading.Lock()

    @contextmanager
    def installed(self) -> Iterator["PostRecorder"]:
        original = HTTPClient.post
        recorder = self

        def post(client, url, payload=None, headers=None):
            sent = {**recorder.moment, "real_s": time.perf_counter() - recorder.started}
            status, error = None, None
            try:
                response = original(client, url, payload=payload, headers=headers)
                status = response.status_code
                return response
            except HTTPError as exc:
                status, error = exc.status_code, exc.message
                raise
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                error = str(exc)
                raise
            finally:
                recorder._record(url, payload, status, error, sent)

        HTTPClient.post = post
        try:
            yield self
        finally:
            HTTPClient.post = original

    def _record(self, url, payload, status, error, sent) -> None:
        plan, name = self.names.get(url, (None, None))
        success = status is not None and 200 <= int(status) < 300
        with self._lock:
            self.tools.action(url, success)
            self.actions.append({"plan": plan, "name": name, "url": url,
                                 "parameters": dict(payload or {}), "status_code": status,
                                 "success": success, **({"error": error} if error else {}),
                                 **sent})


def _requests_out(plans: List[Plan]) -> bool:
    """Some action request has been sent and not yet answered."""
    return any(getattr(node, "_io_in_flight", False)
               for plan in plans if plan.tree is not None
               for node in plan.tree.iterate() if isinstance(node, ActionAffordanceNode))


def _finish(plans: List[Plan], execution: Execution) -> None:
    for plan in plans:
        if plan.tree is not None:
            try:
                plan.tree.shutdown()
            except Exception:  # noqa: BLE001
                pass
        execution.plans.append({"plan": plan.index, "status": plan.status,
                                "ticks": plan.ticks, "error": plan.error,
                                "source": plan.entry.get("source")})


# --------------------------------------------------------------------------
# The two modes
# --------------------------------------------------------------------------

def needs_lockstep(entries: Sequence[Dict[str, Any]]) -> bool:
    return any(e.get("runtime") or e.get("execution_mode") == "dependency"
               for e in entries if isinstance(e.get("tree"), dict) and e["tree"])


def run_immediate(entries, tools: ToolLog, *, tick_period: float, tick_budget: int,
                  timeout: float) -> Execution:
    """The UA's loop, in real time."""
    execution = Execution(mode="immediate")
    plans = _prepare(entries, None, tools)
    recorder = PostRecorder(plans, tools)
    started = time.perf_counter()
    with recorder.installed():
        while any(p.status == "RUNNING" for p in plans):
            _tick(plans)
            if time.perf_counter() - started > timeout:
                for plan in plans:
                    if plan.status == "RUNNING":
                        plan.status, plan.error = "FAILURE", f"timeout after {timeout}s"
                break
            time.sleep(tick_period)
        # A request still out when the trees concluded is recorded when it lands.
        while _requests_out(plans) and time.perf_counter() - started < timeout:
            time.sleep(0.005)
    execution.actions = recorder.actions
    execution.real_seconds = time.perf_counter() - started
    _finish(plans, execution)
    for summary, plan in zip(execution.plans, plans):
        summary["over_tick_budget"] = plan.ticks > tick_budget
    return execution


def watched_attributes(goals: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """(device, attribute) of every assert in the eval goals."""
    out = []
    for goal in goals or []:
        for target in (goal.get("targets") or []) if isinstance(goal, dict) else []:
            for check in target.get("asserts") or []:
                pair = (target.get("device_id"), check.get("attribute"))
                if pair not in out:
                    out.append(pair)
    return out


def attribute_values(state: Dict[str, Any], watched) -> Dict[str, Any]:
    """{"device|attribute": value} for the watched pairs in one home state."""
    devices = {d.get("device_id"): d.get("attributes") or {}
               for room in (state.get("rooms") or {}).values()
               for d in room.get("devices") or []}
    return {f"{device}|{attribute}": devices.get(device, {}).get(attribute)
            for device, attribute in watched}


def observation_points(goals: Sequence[Dict[str, Any]]) -> List[Tuple[int, int, int]]:
    """(start, at, end) tick of each timed eval goal's window."""
    windows = []
    for goal in goals or []:
        when = goal.get("when") if isinstance(goal, dict) else None
        if isinstance(when, dict) and "at_tick" in when:
            at, tol = int(when["at_tick"]), int(when.get("tolerance_ticks", 0))
            windows.append((max(0, at - tol), at, at + tol))
    return windows


def run_lockstep(entries, tools: ToolLog, sim, *, step_seconds: float,
                 tick_interval: float, eval_goals: Sequence[Dict[str, Any]],
                 default_horizon_s: float = 1800.0, margin_s: float = 60.0,
                 settle_timeout: float = 30.0,
                 on_step: Optional[Callable[[int, Dict[str, Any]], None]] = None) -> Execution:
    """Run the plans with the simulator's clock driven step by step.

    ``sim`` is the simulator client (`fast_forward_to`, `home_state`)."""
    execution = Execution(mode="lockstep")
    clock = StateClock()
    state = sim.home_state()
    clock.update(state)
    start_tick = int(state["current_tick"])
    step_ticks = max(1, int(round(step_seconds / tick_interval)))
    windows = observation_points(eval_goals)
    margin = int(round(margin_s / tick_interval))
    horizon = (max(end for _, _, end in windows) + margin) if windows else \
        start_tick + int(round(default_horizon_s / tick_interval))
    marks = sorted({t for window in windows for t in window})
    execution.horizon_tick = horizon

    plans = _prepare(entries, clock, tools)
    recorder = PostRecorder(plans, tools)
    started = time.perf_counter()
    tick = start_tick
    execution.snapshots.append({"tick": tick, "state": state})
    watched = watched_attributes(eval_goals)

    def follow(at_tick: int, at_state: Dict[str, Any]) -> None:
        values = attribute_values(at_state, watched)
        if not execution.timeline or execution.timeline[-1]["values"] != values:
            execution.timeline.append({"tick": at_tick, "values": values})

    follow(tick, state)
    with recorder.installed():
        while True:
            # Tick at this moment until no request is out: a POST is answered
            # once the simulator has applied it, and the next tick consumes the
            # answer -- and may send the sequence's next action, at the same
            # virtual time. Only then may the clock move on.
            recorder.moment = {"tick": tick, "time": state.get("current_time")}
            for _ in range(MAX_TICKS_PER_MOMENT):
                _tick(plans)
                waited = time.perf_counter()
                pending = _requests_out(plans)
                while _requests_out(plans) and time.perf_counter() - waited < settle_timeout:
                    time.sleep(0.005)
                if not pending:
                    break

            done = not any(p.status == "RUNNING" for p in plans)
            last_mark = marks[-1] if marks else start_tick
            if (done and tick >= last_mark) or tick >= horizon:
                break

            # The next stop: the next step on a grid from the start tick (a
            # plan's waits run from its start, so they come due on the grid),
            # or the next window edge / at_tick if sooner.
            target = start_tick + ((tick - start_tick) // step_ticks + 1) * step_ticks
            upcoming = [m for m in marks if m > tick]
            if upcoming and upcoming[0] < target:
                target = upcoming[0]
            envelope = sim.fast_forward_to(target)
            state = (envelope or {}).get("data") or sim.home_state()
            tick = int(state.get("current_tick", target))
            clock.update(state)
            follow(tick, state)
            execution.steps += 1
            if any(lo <= tick <= hi for lo, _, hi in windows) or tick in marks:
                execution.snapshots.append({"tick": tick, "state": state})
            if on_step is not None:
                on_step(tick, state)

    execution.actions = recorder.actions
    if execution.snapshots[-1]["tick"] != tick:
        execution.snapshots.append({"tick": tick, "state": state})
    execution.real_seconds = time.perf_counter() - started
    execution.virtual_seconds = (tick - start_tick) * tick_interval
    execution.end_tick = tick
    _finish(plans, execution)
    return execution
