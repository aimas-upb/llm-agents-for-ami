"""Executing an episode's plans as SimuHome scheduled workflows, from a dry run of the trees.

Lockstep (`execution.run_lockstep`) ticks every tree and fast-forwards the
simulator 10 virtual seconds at a time, so a plan that waits 44 minutes for a
dryer costs some 270 simulator round trips that only move the clock. SimuHome's
own method is to submit the actions as timed workflows
(`schedule_workflow(start_time, steps)`) and let the simulator run them.

Here the same compiled trees are ticked in memory -- a **dry run** -- against a
virtual clock that steps one second at a time:

    action requests (HTTPClient.post)   recorded at their virtual time, applied to
                                        the home model, not sent
    reads (HTTPClient.get)              answered by the home model
    the plan clock, settling waits      the virtual clock
    node I/O                            inline: every request resolves in its tick

The home model (`DryRunHome`) holds only what needs no simulator:

    our actions            a property an action acts upon holds the value sent
    appliance cycles       Matter CountdownTime, as the simulator runs it: a
                           Running cycle ends at its start + countdown (state
                           Stopped, countdown null); Paused freezes it

Anything else -- a room temperature, a cycle's countdown just after we started
it, every value at the start -- is read from the **real simulator at that
virtual moment** (`SimSync.read`). The simulator is deterministic and only our
actions intervene, so this is exact: the actions recorded so far are submitted
(sent now, or scheduled as workflows at their times) and the simulator is
advanced to the moment first. A real reading is reused for `real_read_seconds`.

The simulator is advanced only to: the moments a real reading is needed, the
eval marks (window edges, `at_tick`), one tick after each of our actions and
after each cycle end the model predicts -- where a snapshot is recorded for the
evaluator, which replays them as it does lockstep's. Everything else is one jump.

Worked example -- qt4-3 seed 45, "start washer 1 on Heavy mode now and pause it
when dryer 1 finishes":

    t=0       the trees switch the washer on, set Heavy, start it: three actions
              at the start, sent for real. The dependency tree's runtime reads
              dryer 1's operational state: a cycle, so its state and countdown are
              read once from the simulator (Running, about 2640 s left).
    t=1..2639 the dry run ticks in memory; the runtime's reads of the dryer are
              answered by the cycle rule -- no simulator call
    t=2640    the model ends the dryer's cycle (Stopped); the wait succeeds; the
              pause is recorded then (about 14:10:31) -- lockstep sent it up to
              one 10 s step later
    end       the pause becomes one workflow at 14:10:31; the simulator jumps to
              the eval marks, the pause and the dryer's end, recording snapshots

A plan waiting on a room temperature, by contrast, reads it from the simulator
every `real_read_seconds` of virtual time -- lockstep's fidelity, no worse.

    run_scheduled(entries, tools, sim, ...)  -> Execution(mode="scheduled")
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from rdflib import Graph
from rdflib.namespace import RDF

from ami_agents.agents.interaction_solver.utils.dependency_context import artifact_of_url
from ami_agents.agents.interaction_solver.utils.specific_capability_context import _form
from ami_agents.bt_planning.nodes import affordance_nodes, async_support
from ami_agents.bt_planning.nodes.http_client import HTTPClient, HTTPError, HTTPResponse
from ami_agents.shared.utils.namespaces import HOMEONT, SAREF, TD
from ami_agents.shared.utils.vocabulary import close_types

from .execution import (
    TIME_FORMAT,
    Execution,
    Plan,
    _finish,
    _prepare,
    _tick,
    attribute_values,
    observation_points,
    watched_attributes,
)
from .toollog import ToolLog

# Matter OperationalStateEnum values the cycle rule uses.
STOPPED, RUNNING, PAUSED = 0, 1, 2
# Ticks at one virtual moment while plans keep progressing (bounds a tree that
# never settles at one moment).
MAX_TICKS_PER_MOMENT = 20


# --------------------------------------------------------------------------
# The virtual clock and inline I/O
# --------------------------------------------------------------------------

class VirtualClock:
    """The dry run's time: seconds since the episode's start, and the
    datetime the plans' clock reads (`now()`, as `execution.StateClock`)."""

    def __init__(self, start: datetime) -> None:
        self.start = start
        self.seconds = 0.0

    def now(self) -> datetime:
        return self.start + timedelta(seconds=self.seconds)

    def text(self, seconds: Optional[float] = None) -> str:
        """A virtual moment as SimuHome writes it ("YYYY-MM-DD HH:MM:SS")."""
        moment = self.start + timedelta(seconds=self.seconds if seconds is None else seconds)
        return moment.strftime(TIME_FORMAT)


class InlineExecutor:
    """Runs submitted work at once and hands back a finished future, so a
    node's request resolves within the tick that made it."""

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - handed to the node, as a pool would
            future.set_exception(exc)
        return future


def _response(url: str, body: Any, status: int = 200) -> HTTPResponse:
    """An HTTP response as `HTTPClient` returns one."""
    return HTTPResponse(status_code=status, body=body, headers={}, url=url, elapsed_time=0.0)


# --------------------------------------------------------------------------
# Recorded actions
# --------------------------------------------------------------------------

@dataclass
class Recorded:
    """One action the dry run sent, and what became of it."""

    plan: Optional[int]
    name: Optional[str]
    url: str
    parameters: Dict[str, Any]
    seconds: float                       # virtual seconds since the start
    time: str                            # the same, as SimuHome writes it
    current_state: Optional[int] = None  # the device's operational state just before
    submitted: bool = False
    how: Optional[str] = None            # "sent" | "scheduled"
    status_code: Optional[int] = None
    workflow_id: Optional[str] = None
    success: Optional[bool] = None
    error: Optional[str] = None

    def as_dict(self, tick: int) -> Dict[str, Any]:
        """As the record's `actions` lists them (lockstep's fields, plus how)."""
        return {"plan": self.plan, "name": self.name, "url": self.url,
                "parameters": self.parameters, "time": self.time, "tick": tick,
                "how": self.how, "status_code": self.status_code,
                **({"workflow_id": self.workflow_id} if self.workflow_id else {}),
                "success": bool(self.success), **({"error": self.error} if self.error else {})}


# --------------------------------------------------------------------------
# The home model
# --------------------------------------------------------------------------

@dataclass
class Cycle:
    """An appliance's cycle as Matter's OperationalState / CountdownTime run it."""

    state_url: str
    countdown_url: Optional[str]
    state: Optional[int] = None
    remaining: Optional[float] = None    # seconds left at `as_of`; None: none / unknown
    as_of: float = 0.0
    known: bool = False                  # state and remaining read or derived
    ends: List[float] = field(default_factory=list)   # predicted cycle ends

    def at(self, seconds: float) -> None:
        """Move the cycle to virtual time ``seconds``: a Running cycle with a
        known countdown ends when it runs out (Stopped, countdown null)."""
        if self.state == RUNNING and self.remaining is not None:
            elapsed = seconds - self.as_of
            if elapsed >= self.remaining:
                end = self.as_of + self.remaining
                if end not in self.ends:
                    self.ends.append(end)
                self.state, self.remaining = STOPPED, None
            else:
                self.remaining -= elapsed
        self.as_of = seconds


class DryRunHome:
    """The property values the dry run reads, by URL (see the module docstring).

    ``real`` is `SimSync.read`: the value of a property URL from the real
    simulator at a virtual moment."""

    def __init__(self, graph: Graph, clock: VirtualClock,
                 real: Callable[[str, float], Any], real_read_seconds: float) -> None:
        self.graph = graph
        self.clock = clock
        self.real = real
        self.real_read_seconds = real_read_seconds
        self.written: Dict[str, Any] = {}                 # property URL -> our value
        self.cache: Dict[str, Tuple[Any, float]] = {}     # property URL -> (value, read at)
        self.cycles: Dict[Any, Cycle] = {}                # artifact -> its cycle
        self.real_reads = 0

    # -- the TD graph ------------------------------------------------------

    def _affordance(self, artifact: Any, predicate, url: str) -> Any:
        """The device's affordance under ``predicate`` posting to / read at ``url``."""
        return next((a for a in self.graph.objects(artifact, predicate)
                     if _form(self.graph, a)[0] == url), None)

    def _property_of_class(self, artifact: Any, cls) -> Optional[str]:
        """The URL of the device's property of class ``cls``, or None."""
        for prop in self.graph.objects(artifact, TD.hasPropertyAffordance):
            if (prop, RDF.type, cls) in self.graph:
                return _form(self.graph, prop)[0] or None
        return None

    def cycle_of(self, url: str) -> Optional[Cycle]:
        """The cycle whose state or countdown property ``url`` is, or whose
        device ``url`` (an action or property) belongs to -- None when the
        device has no operational state."""
        artifact = artifact_of_url(self.graph, url)
        if artifact is None:
            return None
        if artifact not in self.cycles:
            state_url = self._property_of_class(artifact, HOMEONT.OperationalState)
            if state_url is None:
                return None
            self.cycles[artifact] = Cycle(
                state_url=state_url,
                countdown_url=self._property_of_class(artifact, HOMEONT.RemainingTime))
        return self.cycles[artifact]

    def acted_upon(self, action_url: str) -> List[str]:
        """The property URLs an action writes (`saref:actsUpon`)."""
        artifact = artifact_of_url(self.graph, action_url)
        action = self._affordance(artifact, TD.hasActionAffordance, action_url) \
            if artifact is not None else None
        if action is None:
            return []
        return [u for p in self.graph.objects(action, SAREF.actsUpon) if (u := _form(self.graph, p)[0])]

    # -- cycles ------------------------------------------------------------

    def _know(self, cycle: Cycle) -> None:
        """Read a cycle's state and countdown from the simulator, now."""
        now = self.clock.seconds
        cycle.state = _int(self._real(cycle.state_url, now))
        countdown = self._real(cycle.countdown_url, now) if cycle.countdown_url else None
        cycle.remaining = float(countdown) if _number(countdown) else None
        cycle.as_of, cycle.known = now, True

    def _cycle_now(self, cycle: Cycle) -> Cycle:
        """The cycle at the current virtual moment, read from the simulator
        when the model does not know it (first use, or a cycle we started)."""
        if not cycle.known or (cycle.state == RUNNING and cycle.remaining is None):
            self._know(cycle)
        else:
            cycle.at(self.clock.seconds)
        return cycle

    # -- reads and writes --------------------------------------------------

    def _real(self, url: str, seconds: float) -> Any:
        """A property's value from the simulator at ``seconds``."""
        self.real_reads += 1
        value = self.real(url, seconds)
        self.cache[url] = (value, seconds)
        return value

    def read(self, url: str) -> Any:
        """The value a plan reads at ``url`` now."""
        now = self.clock.seconds
        # 1. A cycle's state or countdown: the cycle rule.
        cycle = self.cycle_of(url)
        if cycle is not None and url in (cycle.state_url, cycle.countdown_url):
            cycle = self._cycle_now(cycle)
            if url == cycle.state_url:
                return cycle.state
            return None if cycle.remaining is None else int(round(cycle.remaining))
        # 2. A property our actions set.
        if url in self.written:
            return self.written[url]
        # 3. Anything else: the simulator, reused while fresh.
        cached = self.cache.get(url)
        if cached is not None and now - cached[1] < self.real_read_seconds:
            return cached[0]
        return self._real(url, now)

    def current_state(self, action_url: str) -> Optional[int]:
        """The operational state the action's device is in just before it
        runs, for an action that sets that state (it picks Start or Resume);
        None for any other action. A cycle's length is not needed for this,
        so a known cycle is not read again."""
        cycle = self.cycle_of(action_url)
        if cycle is None or cycle.state_url not in self.acted_upon(action_url):
            return None
        if not cycle.known:
            self._know(cycle)
        cycle.at(self.clock.seconds)
        return cycle.state

    def write(self, action_url: str, parameters: Dict[str, Any]) -> None:
        """Apply one of our actions: the properties it acts upon take the value
        sent; an operational-state change moves the cycle."""
        value = (parameters or {}).get("value")
        cycle = self.cycle_of(action_url)
        for url in self.acted_upon(action_url):
            if cycle is not None and url == cycle.state_url:
                self._set_state(cycle, _int(value))
            else:
                self.written[url] = value

    def _set_state(self, cycle: Cycle, value: Optional[int]) -> None:
        """Our command to a cycle: pause freezes it, resume continues it, start
        begins one whose length only the simulator knows, stop ends it."""
        cycle = self._cycle_now(cycle)
        if value == PAUSED and cycle.state == RUNNING:
            cycle.state = PAUSED
        elif value == RUNNING and cycle.state == PAUSED:
            cycle.state = RUNNING
        elif value == RUNNING:
            cycle.state, cycle.remaining = RUNNING, None       # read once it has started
        elif value == STOPPED:
            cycle.state, cycle.remaining = STOPPED, None
        cycle.as_of = self.clock.seconds

    def cycle_ends(self) -> List[float]:
        """Every cycle end the model has predicted, in virtual seconds."""
        return sorted(end for cycle in self.cycles.values() for end in cycle.ends)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Keeping the simulator in step
# --------------------------------------------------------------------------

class SimSync:
    """The real simulator, trailing the dry run (see the module docstring).

    ``post`` / ``get`` are the real HTTP calls (the dry run patches
    `HTTPClient`'s; these are the originals)."""

    def __init__(self, sim, clock: VirtualClock, state: Dict[str, Any], tick_interval: float,
                 marks: Sequence[int], watched, execution: Execution, tools: ToolLog,
                 post: Callable, get: Callable) -> None:
        self.sim, self.clock, self.tick_interval = sim, clock, tick_interval
        self.start_tick = int(state["current_tick"])
        self.tick = self.start_tick
        self.marks = sorted(set(marks))
        self.watched, self.execution, self.tools = watched, execution, tools
        self.post, self.get = post, get
        self.http = HTTPClient()
        self.recorded: List[Recorded] = []
        self.workflows: List[Dict[str, Any]] = []
        self.stops: set = set()              # ticks a snapshot is taken at, on the way
        self.home: Optional[DryRunHome] = None
        self.ticking: Optional[int] = None   # the plan being ticked (credits its actions)
        self.syncs = 0
        self._record_state(self.tick, state)

    # -- time --------------------------------------------------------------

    def tick_at(self, seconds: float) -> int:
        """The simulator tick of a virtual moment."""
        return self.start_tick + int(round(seconds / self.tick_interval))

    # -- observing ---------------------------------------------------------

    def _record_state(self, tick: int, state: Dict[str, Any]) -> None:
        """A snapshot, and the watched attributes' timeline when they changed."""
        self.execution.snapshots.append({"tick": tick, "state": state})
        values = attribute_values(state, self.watched)
        if not self.execution.timeline or self.execution.timeline[-1]["values"] != values:
            self.execution.timeline.append({"tick": tick, "values": values})

    def advance(self, to_tick: int) -> None:
        """Move the simulator to ``to_tick``, stopping at every observation point
        on the way: eval marks, a tick after our actions, a tick after cycle ends."""
        if to_tick <= self.tick:
            return
        # 1. Where to stop: what the evaluator looks at, and where the checked
        #    state can change.
        ends = {self.tick_at(s) + 1 for s in (self.home.cycle_ends() if self.home else [])}
        stops = sorted(t for t in (set(self.marks) | self.stops | ends) if self.tick < t < to_tick)
        # 2. One jump per stop; a snapshot at each.
        for target in stops + [to_tick]:
            envelope = self.sim.fast_forward_to(target)
            state = (envelope or {}).get("data") or self.sim.home_state()
            self.tick = int(state.get("current_tick", target))
            self._record_state(self.tick, state)
        self.execution.steps += 1

    # -- submitting --------------------------------------------------------

    def record(self, plan: Optional[int], name: Optional[str], url: str,
               parameters: Dict[str, Any], current_state: Optional[int]) -> None:
        """One action of the dry run, at its virtual moment."""
        self.recorded.append(Recorded(plan=plan, name=name, url=url, parameters=dict(parameters),
                                      seconds=self.clock.seconds, time=self.clock.text(),
                                      current_state=current_state))

    def _send(self, action: Recorded) -> None:
        """An action at the simulator's current moment: sent through SHTD, as
        the agents send it."""
        action.how = "sent"
        try:
            response = self.post(self.http, action.url, payload=action.parameters)
            action.status_code, action.success = response.status_code, response.is_success
        except HTTPError as exc:
            action.status_code, action.success, action.error = exc.status_code, False, exc.message
        except Exception as exc:  # noqa: BLE001 - recorded on the action
            action.success, action.error = False, str(exc)
        self.tools.action(action.url, bool(action.success))

    def _schedule(self, group: List[Recorded]) -> None:
        """Later actions of one plan at one moment: one SimuHome workflow, each
        step translated by SHTD for the state predicted then."""
        steps = []
        for action in group:
            action.how = "scheduled"
            body = {"value": action.parameters.get("value"),
                    **({"current": {"operational_state": action.current_state}}
                       if action.current_state is not None else {})}
            try:
                steps.append(self.post(self.http, action.url.rstrip("/") + "/dispatch",
                                       payload=body).body)
            except Exception as exc:  # noqa: BLE001 - this action cannot be scheduled
                action.success, action.error = False, f"dispatch: {exc}"
        steps = [s for s in steps if isinstance(s, dict) and s.get("tool")]
        if not steps:
            return
        try:
            created = self.sim.schedule_workflow(group[0].time, steps,
                                                 description=f"plan {group[0].plan}")
            workflow_id = (created or {}).get("workflow_id")
        except Exception as exc:  # noqa: BLE001 - recorded on its actions
            for action in group:
                if action.error is None:
                    action.success, action.error = False, f"schedule: {exc}"
            return
        self.workflows.append({"workflow_id": workflow_id, "plan": group[0].plan,
                               "start_time": group[0].time, "steps": steps})
        for action in group:
            if action.error is None:
                action.workflow_id = workflow_id
        self.stops.add(self.tick_at(group[0].seconds) + 1)

    def _catch_up(self) -> None:
        """Take the simulator's own tick: between our calls it runs on in real
        time (0.1 s a tick), so it may be past where we left it. A workflow
        must start after it, so anything due by then is sent instead."""
        try:
            state = self.sim.home_state()
            self.tick = max(self.tick, int(state.get("current_tick", self.tick)))
        except Exception:  # noqa: BLE001 - keep our own count
            pass

    def submit(self) -> None:
        """Every action recorded since the last submit: those due by the
        simulator's moment sent, later ones scheduled, one workflow per
        (moment, plan)."""
        pending = [a for a in self.recorded if not a.submitted]
        if not pending:
            return
        self._catch_up()
        groups: Dict[Tuple[float, Optional[int]], List[Recorded]] = {}
        for action in pending:
            action.submitted = True
            if self.tick_at(action.seconds) <= self.tick:
                self._send(action)
                self.stops.add(self.tick + 1)
            else:
                groups.setdefault((action.seconds, action.plan), []).append(action)
        for group in groups.values():
            self._schedule(group)

    def sync(self, seconds: float) -> None:
        """Bring the simulator to virtual moment ``seconds``: our actions so far
        submitted, then advanced -- one tick past the moment when an action of
        ours falls on it, so a read there sees it."""
        self.submit()
        target = self.tick_at(seconds)
        if any(a.how == "scheduled" and self.tick_at(a.seconds) == target for a in self.recorded):
            target += 1
        self.advance(target)
        self.syncs += 1

    def read(self, url: str, seconds: float) -> Any:
        """A property's value in the simulator at virtual moment ``seconds``."""
        self.sync(seconds)
        try:
            return self.get(self.http, url).body
        except Exception:  # noqa: BLE001 - an unreadable value reads as None
            return None

    def finish(self, horizon: int) -> None:
        """The last actions submitted, the simulator advanced to the horizon,
        and each scheduled action's outcome read from its workflow."""
        self.submit()
        self.advance(horizon)
        try:
            statuses = {w["workflow_id"]: w for w in self.sim.workflow_list() or []}
        except Exception:  # noqa: BLE001 - outcomes unknown
            statuses = {}
        for workflow in self.workflows:
            status = (statuses.get(workflow["workflow_id"]) or {}).get("status")
            workflow["status"] = status
        for action in self.recorded:
            if action.how == "scheduled" and action.workflow_id:
                status = (statuses.get(action.workflow_id) or {}).get("status")
                action.success = status == "completed"
                if not action.success and action.error is None:
                    action.error = f"workflow {status or 'unknown'}"
                self.tools.action(action.url, bool(action.success))


# --------------------------------------------------------------------------
# The dry run
# --------------------------------------------------------------------------

@contextmanager
def dry_run(home: DryRunHome, sync: SimSync, clock: VirtualClock,
            names: Dict[Tuple[Any, ...], str]) -> Iterator[None]:
    """Replace the outside world for the trees (see the module docstring):
    reads from the home model, actions recorded and applied, the clock and
    node I/O virtual and inline."""
    original_get, original_post = HTTPClient.get, HTTPClient.post
    original_executor, original_time = async_support._executor, affordance_nodes.time

    def get(client, url, headers=None):
        """A plan's read, answered by the home model."""
        return _response(url, home.read(url))

    def post(client, url, payload=None, headers=None):
        """A plan's action, recorded at this moment -- for the plan being
        ticked -- and applied to the model."""
        plan = sync.ticking
        name = names.get((plan, url, _sent(payload))) or names.get((plan, url))
        sync.record(plan, name, url, payload or {}, home.current_state(url))
        home.write(url, payload or {})
        return _response(url, {"ok": True, "dry_run": True})

    HTTPClient.get, HTTPClient.post = get, post
    async_support._executor = InlineExecutor()
    affordance_nodes.time = SimpleNamespace(monotonic=lambda: clock.seconds,
                                            time=time.time, sleep=time.sleep,
                                            perf_counter=time.perf_counter)
    try:
        yield
    finally:
        HTTPClient.get, HTTPClient.post = original_get, original_post
        async_support._executor = original_executor
        affordance_nodes.time = original_time


def _sent(parameters: Any) -> str:
    """The parameters an action sends, as a comparable key."""
    return json.dumps(parameters or {}, sort_keys=True, default=str)


def _action_names(plans: List[Plan]) -> Dict[Tuple[Any, ...], str]:
    """Node names for the action records: (plan index, action URL, the
    parameters it sends) -> name, so two actions on one URL -- a guard's pause
    and a goal's start of the same cycle -- keep their own names; and (plan
    index, action URL) -> the first such name, for an action whose parameters
    are filled in from the blackboard when it runs."""
    names: Dict[Tuple[Any, ...], str] = {}
    for plan in plans:
        if plan.tree is not None:
            for node in plan.tree.iterate():
                url = getattr(node, "action_url", None)
                if url:
                    sent = _sent(getattr(node, "parameters", None))
                    names.setdefault((plan.index, url, sent), node.name)
                    names.setdefault((plan.index, url), node.name)
    return names


def _tick_each(plans: List[Plan], sync: "SimSync") -> None:
    """One tick of every running plan (`execution._tick`), one plan at a time,
    so each action is credited to the plan that sent it -- two plans may act
    on the same affordance."""
    for plan in plans:
        sync.ticking = plan.index
        _tick([plan])
    sync.ticking = None


def run_scheduled(entries, tools: ToolLog, sim, *, turtle: str, tick_interval: float,
                  eval_goals: Sequence[Dict[str, Any]], dry_step_seconds: float = 1.0,
                  real_read_seconds: float = 10.0, default_horizon_s: float = 1800.0,
                  margin_s: float = 60.0) -> Execution:
    """Run the plans as a dry run turned into scheduled workflows (see the
    module docstring). ``turtle`` is the home's TD graph; ``sim`` the
    simulator client."""
    execution = Execution(mode="scheduled")
    started = time.perf_counter()

    # 1. Where the run starts and ends: the simulator's moment, the eval marks
    #    and the horizon, as lockstep computes them.
    state = sim.home_state()
    clock = VirtualClock(datetime.strptime(state["current_time"], TIME_FORMAT))
    windows = observation_points(eval_goals)
    marks = sorted({t for window in windows for t in window})
    start_tick = int(state["current_tick"])
    horizon = (max(end for _, _, end in windows) + int(round(margin_s / tick_interval))) \
        if windows else start_tick + int(round(default_horizon_s / tick_interval))
    horizon_seconds = (horizon - start_tick) * tick_interval
    execution.horizon_tick = horizon

    # 2. The graph, the simulator link, the home model, the compiled plans.
    graph = Graph().parse(data=turtle, format="turtle")
    close_types(graph)
    sync = SimSync(sim, clock, state, tick_interval, marks, watched_attributes(eval_goals),
                   execution, tools, post=HTTPClient.post, get=HTTPClient.get)
    home = DryRunHome(graph, clock, sync.read, real_read_seconds)
    sync.home = home
    plans = _prepare(entries, clock, tools)

    # 3. The dry run: tick at each moment while plans progress, then one step.
    with dry_run(home, sync, clock, _action_names(plans)):
        last_mark_seconds = ((marks[-1] - start_tick) * tick_interval) if marks else 0.0
        while True:
            for _ in range(MAX_TICKS_PER_MOMENT):
                before = (len(sync.recorded), [p.status for p in plans])
                _tick_each(plans, sync)
                if (len(sync.recorded), [p.status for p in plans]) == before:
                    break
            done = not any(p.status == "RUNNING" for p in plans)
            if (done and clock.seconds >= last_mark_seconds) or clock.seconds >= horizon_seconds:
                break
            clock.seconds += dry_step_seconds

    # 4. The simulator to the horizon -- or, with no eval window (a qt1-qt3
    #    episode, whose final state is what is checked), just past where the
    #    dry run concluded -- and each action's real outcome.
    if not windows:
        horizon = max(sync.tick, sync.tick_at(clock.seconds)) + 1
        execution.horizon_tick = horizon
    sync.finish(horizon)
    _finish(plans, execution)
    failed = {a.plan for a in sync.recorded if a.success is False}
    for summary in execution.plans:
        if summary["plan"] in failed and summary["status"] != "FAILURE":
            errors = [a.error or f"HTTP {a.status_code}" for a in sync.recorded
                      if a.plan == summary["plan"] and a.success is False]
            summary["status"], summary["error"] = "FAILURE", "; ".join(errors)

    # 5. The record.
    execution.actions = [a.as_dict(sync.tick_at(a.seconds)) for a in sync.recorded]
    execution.details = {"workflows": sync.workflows, "syncs": sync.syncs,
                         "real_reads": home.real_reads, "dry_seconds": clock.seconds}
    execution.real_seconds = time.perf_counter() - started
    execution.virtual_seconds = (sync.tick - start_tick) * tick_interval
    execution.end_tick = sync.tick
    return execution
