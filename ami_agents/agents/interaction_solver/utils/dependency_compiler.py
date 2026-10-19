"""Compile a profiled dependency structure into one behaviour tree, by templates.

Every construct of a dependency `GoalStructure` maps to a fixed pattern
(`docs/plans/dependency_goal_planner.md`, after the user's bt_templates.md):

    predicates      a check over the keys of the properties the predicate reads
                    (`dependency_context.resolve_predicate`); any -> or_, all -> and_;
                    connective and / or between predicates of one role
    goal body       achievement: achieve(G) (the achievement planners' tree);
                    filter: per device, skip / act on the first match;
                    maintenance: native / unobserved -> act, then stay alive;
                    regulated -> repeat [in band, or act again], every poll period
    ends            Parallel(SuccessOnOne)[stop conditions..., body]:
                    ends duration -> timer, at_time -> wait_until,
                    termination X -> wait_for X, scope X -> wait_for not X;
                    an end condition already holding at the start (X true /
                    scope false) skips the goal: it need not run
    guard           Selector[not X, unit] -- a false guard skips the goal
    start           now / at_time / after_delay / on_trigger (trigger, offset
                    after or -- profiled -- before; recurrence)
    placement       after_goal / with_goal followers, and goals that complete
                    with a goal and have no start of their own, hang off their
                    referenced goal; heads run side by side under the root;
                    a goal that completes_with a PREDICATE (finish when an event
                    does) and has no start of its own is a head: it starts once
                    the event's remaining time is no more than its own run
                    (completion_timing); a goal that completes_with something
                    AND has a start the user gave is placed by that start --
                    `completes_with` was checked before planning and adds nothing
                    to the tree; a with_goal follower on the SAME device runs
                    right after its goal instead of beside it (one device, one
                    setting at a time)
    checks          a trigger predicate that none of its goals waits for
                    (on_trigger, or completes_with it) -> CompileError; a time
                    text with no time in it ("soon") -> VagueTime, which the
                    planner turns into a question for the user

The tree is JSON IR (it travels from the InteractionSolver to the UserAssistant):
each template node is an IR type compiling to a py_trees built-in
(`bt_planning/nodes/registry.py`). What the tree waits on lives in blackboard
keys, kept current by the plan's runtime (`bt_planning/runtime.py`), which this
compiler describes alongside the tree:

    /artifacts/<device>/<property>[/<field>]   readings (runtime.reads)
    /clock/now                                 the plan's clock (runtime.clock)
    /goals/<uid>/<G>/...                       derived values (runtime.derived)

`compile_dependency(...)` returns the tree, its runtime and its notes, or raises
CompileError with the reason -- there are no partial dependency plans.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Graph

from ....shared.models.goal_structure import GoalSpec, GoalStructure, PredicateSpec
from ....bt_planning.nodes.dependency_nodes import CLOCK_KEY
from .dependency_context import (
    PredicateTarget,
    PropertyInfo,
    _property_on,
    _artifact_by_title,
    property_info,
)
from .time_text import TimeTextError, span_seconds, time_of_day

# The structurer's comparisons, as the IR's operator names; and their negations.
OP_NAMES = {"==": "eq", "!=": "ne", ">": "gt", ">=": "ge", "<": "lt", "<=": "le"}
NEGATED = {"eq": "ne", "ne": "eq", "gt": "le", "le": "gt", "lt": "ge", "ge": "lt"}

# Defaults for `interaction_solver.planning.dependency` in agents.yaml.
DEFAULT_POLL_SECONDS = 30.0
DEFAULT_TOLERANCE = 0.5

HEAD_STARTS = (None, "now", "at_time", "after_delay", "on_trigger")


class CompileError(ValueError):
    """The structure cannot be compiled; the message says what is missing."""


class VagueTime(CompileError):
    """A goal's time is words with no time in them ("soon", "later"): only the
    user can say what they mean, so the planner asks instead of failing."""

    def __init__(self, gid: str, text: str, reason: str) -> None:
        super().__init__(f"{gid}: {reason}")
        self.gid = gid
        self.text = text


@dataclass
class CompiledPlan:
    tree: Dict[str, Any]
    runtime: Dict[str, Any]
    long_running: bool = False
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Small IR builders
# --------------------------------------------------------------------------

def _seq(name, children, memory=True):
    return {"type": "sequence", "name": name, "memory": memory, "children": children}


def _sel(name, children, memory=False):
    return {"type": "selector", "name": name, "memory": memory, "children": children}


def _one(kind, name, child):
    return {"type": kind, "name": name, "children": [child]}


def _par_one(name, children):
    return {"type": "parallel", "name": name, "policy": "success_on_one", "children": children}


def _par_all(name, children):
    return {"type": "parallel", "name": name, "policy": "success_on_all",
            "synchronise": True, "children": children}


def _checks_in(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every comparison in a check node, or in a sequence / selector of them."""
    out = list(node.get("checks") or [])
    for child in node.get("children") or []:
        out.extend(_checks_in(child))
    return out


def _timer(name, seconds):
    return {"type": "timer", "name": name, "seconds": float(seconds)}


def _key_part(title: str) -> str:
    """A title as a blackboard key segment: py_trees reads "." as attribute access."""
    return str(title).replace(".", "_").replace(" ", "_")


def _typed(text: Any, schema: Dict[str, Any]) -> Any:
    """A determined value as the property reports it (bool / number / text)."""
    kind = (schema or {}).get("type")
    value = str(text)
    if kind == "boolean":
        return value.strip().lower() in ("true", "1", "on", "yes")
    if kind in ("integer", "number"):
        number = float(value)
        return int(number) if kind == "integer" and number.is_integer() else number
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


# --------------------------------------------------------------------------
# The compiler
# --------------------------------------------------------------------------

class _Compiler:
    def __init__(self, graph: Graph, structure: GoalStructure,
                 subtrees: Dict[str, Dict[str, Any]],
                 per_device: Dict[str, List[Tuple[str, Dict[str, Any]]]],
                 predicates: Dict[str, PredicateTarget],
                 profile: Dict[str, Dict[str, Any]],
                 poll_seconds: float, tolerance: float) -> None:
        self.graph, self.structure = graph, structure
        self.goals, self.preds = structure.goals, structure.predicates
        self.subtrees, self.per_device = subtrees, per_device
        self.targets, self.profile = predicates, profile
        self.poll_seconds, self.tolerance = poll_seconds, tolerance
        self.uid = uuid.uuid4().hex[:8]
        self.reads: Dict[str, Dict[str, Any]] = {}
        self.derived: List[Dict[str, Any]] = []
        self.clock = False
        self.long_running = False
        self.notes: List[str] = []
        self.started_keys: Dict[str, str] = {}      # goal -> key of its start time
        self.placed: set = set()

    # -- runtime -----------------------------------------------------------

    def _read(self, info: PropertyInfo, prop_field: Optional[str] = None) -> str:
        """The key a property is read into (registered once)."""
        key = f"/artifacts/{_key_part(info.artifact)}/{_key_part(info.prop)}"
        if prop_field and prop_field != "NA":
            key += f"/{_key_part(prop_field)}"
        self.reads.setdefault(key, {"key": key, "url": info.url,
                                    "field": None if prop_field in (None, "NA") else prop_field})
        return key

    def _derive(self, key: str, op: str, inputs: List[str], **args) -> str:
        self.derived.append({"key": key, "op": op, "inputs": inputs, "args": args})
        return key

    def _goal_key(self, gid: str, name: str) -> str:
        return f"/goals/{self.uid}/{gid}/{name}"

    def _ref_info(self, ref: Dict[str, Any]) -> PropertyInfo:
        info = property_info(self.graph, ref.get("artifact_name"), ref.get("property_name"))
        if info is None:
            raise CompileError(f"{ref.get('artifact_name')}.{ref.get('property_name')} "
                               f"is not a readable property")
        return info

    def _answer(self, question: str, target: str) -> Optional[Dict[str, Any]]:
        for item in self.profile.values():
            if item["question"] == question and item["target"] == target:
                return item["answer"]
        return None

    # -- predicates (§2) ---------------------------------------------------

    def _checks(self, pid: str, negate: bool = False,
                only: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
        """The predicate's comparisons, one per property it reads, and how they
        combine. `only` restricts them to one device (a filter on a device)."""
        pred = self.preds[pid]
        target = self.targets.get(pid)
        infos = list(target.properties) if target else []
        if only is not None:
            infos = [i for i in infos if i.artifact == only]
            if not infos:
                artifact = _artifact_by_title(self.graph, only)
                info = (_property_on(self.graph, artifact, pred.property_name,
                                     pred.property_class) if artifact is not None else None)
                infos = [info] if info else []
        if not infos:
            raise CompileError(f"{pid} ({pred.predicate_text!r}) reads no property"
                               + (f" on {only}" if only else ""))
        if pred.target_value_determined is None:
            raise CompileError(f"{pid} ({pred.predicate_text!r}) has no value to compare against")
        op = OP_NAMES.get(pred.comparison or "==")
        if op is None:
            raise CompileError(f"{pid}: unknown comparison {pred.comparison!r}")
        combine = "and_" if (pred.quantifier == "all") else "or_"
        if negate:
            op, combine = NEGATED[op], ("or_" if combine == "and_" else "and_")
        checks = [{"key": self._read(info, pred.property_field), "op": op,
                   "value": _typed(pred.target_value_determined, info.schema)}
                  for info in infos]
        return checks, combine

    def _check(self, pid: str, negate: bool = False, only: Optional[str] = None,
               kind: str = "check") -> Dict[str, Any]:
        checks, combine = self._checks(pid, negate, only)
        node = {"type": kind, "name": f"{'not ' if negate else ''}{pid}", "checks": checks}
        if len(checks) > 1:
            node["combine"] = combine
        return node

    def _role(self, gid: str, role: str) -> List[str]:
        return [pid for pid, p in self.preds.items() if p.role == role and gid in p.applies_to]

    def _joined(self, pids: List[str], negate: bool = False) -> Dict[str, Any]:
        """Several predicates of one role: and -> sequence, or -> selector."""
        checks = [self._check(pid, negate) for pid in pids]
        if len(checks) == 1:
            return checks[0]
        orred = any(self.preds[pid].connective == "or" for pid in pids)
        if negate:
            orred = not orred
        return (_sel if orred else _seq)("+".join(pids), checks, memory=False)

    def _wait(self, pids: List[str], negate: bool = False) -> Dict[str, Any]:
        """RUNNING until the (joined) predicates hold."""
        if len(pids) == 1:
            return self._check(pids[0], negate, kind="wait_for")
        return _one("await", f"wait {'+'.join(pids)}", self._joined(pids, negate))

    # -- goal body (§3) ----------------------------------------------------

    def _achieve(self, gid: str) -> Dict[str, Any]:
        tree = self.subtrees.get(gid)
        if not tree:
            raise CompileError(f"{gid} has no plan for its action")
        return tree

    def _body(self, gid: str) -> Dict[str, Any]:
        goal = self.goals[gid]
        if goal.goal_kind == "maintenance":
            return self._maintenance(gid, goal)
        filters = self._role(gid, "filter")
        if not filters:
            return self._achieve(gid)
        (pid,) = filters[:1]
        devices = self.per_device.get(gid) or []
        if not devices:
            raise CompileError(f"{gid}: no devices to filter by {pid}")
        if goal.quantifier == "one":
            return _sel(f"{gid}: first matching", [
                _seq(title, [self._check(pid, only=title), tree]) for title, tree in devices],
                memory=True)
        return _seq(f"{gid}: every matching", [
            _sel(title, [_one("inverter", f"not {pid}", self._check(pid, only=title)), tree])
            for title, tree in devices])

    def _maintenance(self, gid: str, goal: GoalSpec) -> Dict[str, Any]:
        answer = self._answer("maintenance_profile", gid) or {"mode": "unobserved"}
        act = self._achieve(gid)
        if answer.get("mode") != "regulated":
            # native / unobserved: act once, then stay alive until an end stops it.
            return _seq(f"{gid}: hold", [act, {"type": "running", "name": f"{gid}: holding"}])
        observed = answer.get("observed_property")
        if not observed or goal.target_value_determined is None:
            self.notes.append(f"{gid}: regulated without an observed value -> acted on once")
            return _seq(f"{gid}: hold", [act, {"type": "running", "name": f"{gid}: holding"}])
        in_band = self._in_band(gid, goal, observed, answer.get("tolerance_property"))
        return _one("repeat", f"{gid}: regulate", _seq(f"{gid}: poll", [
            _sel(f"{gid}: in band or act", [
                in_band, _one("ignore_failure", f"{gid}: retry next poll", act)]),
            _timer(f"{gid}: poll period", self.poll_seconds)]))

    def _in_band(self, gid: str, goal: GoalSpec, observed: Dict[str, Any],
                 tolerance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        info = self._ref_info(observed)
        key = self._read(info, observed.get("property_field"))
        target = _typed(goal.target_value_determined, info.schema)
        if not isinstance(target, (int, float)) or isinstance(target, bool):
            return {"type": "check", "name": f"{gid}: holds",
                    "checks": [{"key": key, "op": "eq", "value": target}]}
        deviation = self._derive(self._goal_key(gid, "deviation"), "abs_deviation", [key],
                                 target=target)
        if tolerance:
            tol_info = self._ref_info(tolerance)
            tol_key = self._read(tol_info, tolerance.get("property_field"))
            return {"type": "compare", "name": f"{gid}: in band", "key1": deviation,
                    "key2": tol_key, "op": "le"}
        return {"type": "check", "name": f"{gid}: in band",
                "checks": [{"key": deviation, "op": "le", "value": self.tolerance}]}

    # -- ends (§4), guard (§5) --------------------------------------------

    def _stops(self, gid: str) -> List[Dict[str, Any]]:
        goal = self.goals[gid]
        stops = []
        timing = goal.timing
        if timing and timing.ends_kind == "duration":
            stops.append(_timer(f"{gid}: for {timing.ends_text}", self._span(timing.ends_text, gid)))
        elif timing and timing.ends_kind == "at_time":
            self.clock = True
            stops.append({"type": "wait_until", "name": f"{gid}: until {timing.ends_text}",
                          "time_of_day": self._time(timing.ends_text, gid)})
        terminations = self._role(gid, "termination")
        if terminations:
            stops.append(self._wait(terminations))
        scopes = self._role(gid, "scope")
        if scopes:
            stops.append(self._wait(scopes, negate=True))
        return stops

    def _ends(self, gid: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Ends(B): the goal's body, stopped by whichever of its ends comes
        first (a duration, a clock time, a termination, a scope turning
        false) -- for a maintenance goal only. Returns the body unchanged when
        there is nothing to wrap."""
        # 1. No ends: nothing to wait for.
        stops = self._stops(gid)
        if not stops:
            return body
        # 2. An achievement: its ends are left out, with a note.
        if self.goals[gid].goal_kind != "maintenance":
            # An achievement's body is its command(s): done once the device
            # accepts them, before any end could stop it -- and stopping the
            # branch would not stop the device. Its ends are left out.
            ends = [n["name"] for n in stops]
            self.notes.append(f"{gid}: {', '.join(ends)} has no effect on a one-time "
                              f"action -> left out")
            return body
        # 3. A maintenance goal: run beside its ends until one succeeds, and
        #    not at all if one already holds when it starts.
        return self._unless_ended(gid, _par_one(f"{gid}: until it ends", stops + [body]))

    def _unless_ended(self, gid: str, ended_by: Dict[str, Any]) -> Dict[str, Any]:
        """Skip the goal when an end condition already holds at its start (a
        termination true, a scope false): it need not run. Without this the
        body would be ticked once beside the end wait, and its action sent.

        Decided once, after the conditions' properties have been read: a value
        not read yet is not taken to mean the condition does not hold."""
        ended = []
        terminations = self._role(gid, "termination")
        if terminations:
            ended.append(self._joined(terminations))
        scopes = self._role(gid, "scope")
        if scopes:
            ended.append(self._joined(scopes, negate=True))
        if not ended:
            return ended_by
        keys = sorted({c["key"] for node in ended for c in _checks_in(node)})
        known = {"type": "wait_for", "name": f"{gid}: end conditions read",
                 "checks": [{"key": key, "op": "ne", "value": None} for key in keys]}
        if len(keys) > 1:
            known["combine"] = "and_"
        return _seq(f"{gid}: unless already ended", [
            known, _sel(f"{gid}: ended already, or run", ended + [ended_by], memory=True)])

    def _guard(self, gid: str, unit: Dict[str, Any]) -> Dict[str, Any]:
        guards = self._role(gid, "guard")
        if not guards:
            return unit
        return _sel(f"{gid}: if {'+'.join(guards)}",
                    [_one("inverter", "guard false: skip", self._joined(guards)), unit],
                    memory=True)

    def _unit(self, gid: str) -> Dict[str, Any]:
        """U(G) = Guard(Ends(Body(G)))."""
        return self._guard(gid, self._ends(gid, self._body(gid)))

    # -- start (§6) -------------------------------------------------------

    def _span(self, text: Optional[str], gid: str) -> float:
        """Seconds in a span of time ("13 minutes"); VagueTime when the words
        hold none ("soon")."""
        try:
            return span_seconds(text or "")
        except TimeTextError as exc:
            raise VagueTime(gid, text or "", str(exc))

    def _time(self, text: Optional[str], gid: str) -> str:
        """A clock time ("1:39 PM") as HH:MM[:SS]; VagueTime when the words
        hold none."""
        try:
            return time_of_day(text or "")
        except TimeTextError as exc:
            raise VagueTime(gid, text or "", str(exc))

    def _started(self, gid: str, unit: Dict[str, Any]) -> Dict[str, Any]:
        """Record the goal's start time first, when a completes_with follower
        counts down from it."""
        key = self.started_keys.get(gid)
        if key is None:
            return unit
        return _seq(f"{gid}: started", [{"type": "stamp", "name": f"{gid}: stamp start",
                                         "key": key}, unit])

    def _start(self, gid: str, inner: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The head's start wrapper around `inner`; None leaves the goal out."""
        timing = self.goals[gid].timing
        starts = timing.starts if timing else None
        if starts == "at_time":
            self.clock = True
            return _seq(f"{gid}: at {timing.time_text}", [
                {"type": "wait_until", "name": f"{gid}: wait for {timing.time_text}",
                 "time_of_day": self._time(timing.time_text, gid)}, inner])
        if starts == "after_delay":
            self.clock = True
            return _seq(f"{gid}: in {timing.offset_text}", [
                _timer(f"{gid}: wait {timing.offset_text}", self._span(timing.offset_text, gid)),
                inner])
        if starts == "on_trigger":
            return self._triggered(gid, inner)
        if self._finishes_with_event(gid):
            # Finish together with an event (dishwasher 1 finishing), the start
            # left to us: start once the event's remaining time is no more than
            # this goal's own run.
            ref = timing.completes_with
            wait = self._completes_wait(ref, gid)
            return _seq(f"{gid}: finish with {ref}", [wait, inner]) if wait else inner
        return inner

    def _finishes_with_event(self, gid: str) -> bool:
        """G completes with a predicate, not a goal, and has no start of its
        own: nothing else places it, so it heads its own group."""
        timing = self.goals[gid].timing
        return bool(timing and timing.start_chosen and timing.completes_with in self.preds)

    def _triggered(self, gid: str, inner: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        triggers = self._role(gid, "trigger")
        if not triggers and self._role(gid, "recurrence"):
            return inner                    # the recurrence loop waits itself
        if not triggers:
            raise CompileError(f"{gid} starts on a trigger, but no trigger applies to it")
        pid = triggers[0]
        pred = self.preds[pid]
        if pred.offset_direction == "before" and pred.offset_text:
            wait = self._before(gid, pid, pred)
            if wait is None:
                return None
            return _seq(f"{gid}: {pred.offset_text} before {pid}", [wait, inner])
        steps = [self._wait(triggers)]
        if pred.offset_text:
            self.clock = True
            steps.append(_timer(f"{gid}: {pred.offset_text} after {pid}",
                                self._span(pred.offset_text, gid)))
        return _seq(f"{gid}: when {pid}", steps + [inner])

    def _before(self, gid: str, pid: str, pred: PredicateSpec) -> Optional[Dict[str, Any]]:
        answer = self._answer("time_to_event", pid) or {"fallback": "at_event"}
        offset = self._span(pred.offset_text, gid)
        if "time_to_event" not in answer:
            if answer.get("fallback") == "skip":
                self.notes.append(f"{gid} left out: nothing reports {pid} ahead of time")
                return None
            self.notes.append(f"{gid}: nothing reports {pid} ahead of time -> "
                              f"acting when it happens")
            return self._wait([pid])
        source = answer["time_to_event"]
        info = self._ref_info(source["ref"])
        key = self._read(info, source["ref"].get("property_field"))
        if source.get("reads_as") == "timestamp":
            self.clock = True
            due = self._derive(self._goal_key(gid, "due"), "timestamp_minus", [key],
                               seconds=offset)
            return _one("await", f"{gid}: due", {"type": "compare", "name": f"{gid}: now >= due",
                                                  "key1": CLOCK_KEY, "key2": due, "op": "ge"})
        # A countdown, in the property's own unit.
        return {"type": "wait_for", "name": f"{gid}: {pid} within {pred.offset_text}",
                "checks": [{"key": key, "op": "le", "value": offset / info.seconds_per_unit}]}

    def _recurring(self, gid: str) -> Dict[str, Any]:
        """Whenever X: act each time X becomes true, until an end stops it."""
        pids = self._role(gid, "recurrence")
        self.long_running = True
        loop = _one("repeat", f"{gid}: whenever {'+'.join(pids)}", _seq(f"{gid}: once", [
            self._wait(pids),
            _one("ignore_failure", f"{gid}: a failed run does not stop later ones",
                 self._guard(gid, self._body(gid))),
            self._wait(pids, negate=True)]))
        stops = self._stops(gid)
        if not stops:
            return loop
        return self._unless_ended(gid, _par_one(f"{gid}: until it ends", stops + [loop]))

    # -- placement and assembly (§7) ---------------------------------------

    def _followers(self, gid: str, starts: str) -> List[str]:
        """The goals placed by goal ``gid``: those starting ``starts``
        ("after_goal" / "with_goal") it, or, for "completes_with", those
        completing with it whose start is left to us."""
        out = []
        for fid, goal in self.goals.items():
            timing = goal.timing
            if timing is None:
                continue
            if starts == "completes_with":
                follows = timing.start_chosen and timing.completes_with == gid
            else:
                follows = timing.starts == starts and timing.ref_goal == gid
            if follows:
                out.append(fid)
        return out

    def _after_chain(self, ref: str, fid: str) -> Dict[str, Any]:
        """[anchor wait?, timer(offset)?, group(F)] for a goal after `ref`."""
        timing = self.goals[fid].timing
        steps = []
        if self.goals[ref].goal_kind == "achievement":
            answer = self._answer("dependency_anchor", fid) or {"anchor": "goal_success"}
            if answer.get("anchor") == "condition" and answer.get("condition"):
                steps.append(self._condition_wait(fid, answer["condition"]))
        if timing.offset_text:
            self.clock = True
            steps.append(_timer(f"{fid}: {timing.offset_text} after {ref}",
                                self._span(timing.offset_text, fid)))
        group = self._group(fid, with_start=False)
        if group is None:
            return None
        return _seq(f"{fid}: after {ref}", steps + [group]) if steps else group

    def _condition_wait(self, fid: str, condition: Dict[str, Any]) -> Dict[str, Any]:
        info = self._ref_info(condition["ref"])
        op = OP_NAMES.get(condition.get("comparison") or "==")
        if op is None:
            raise CompileError(f"{fid}: unknown comparison in its anchor condition")
        return {"type": "wait_for", "name": f"{fid}: until {info.artifact}.{info.prop} "
                                            f"{condition.get('comparison')} "
                                            f"{condition.get('target_value_text')}",
                "checks": [{"key": self._read(info, condition["ref"].get("property_field")),
                            "op": op,
                            "value": _typed(condition.get("target_value_text"), info.schema)}]}

    def _completes_wait(self, ref: str, cid: str) -> Optional[Dict[str, Any]]:
        """Start C once R's remaining time is no more than C's duration. The
        planner asks the user before compiling when the durations cannot be
        known (`time_consistency.check_completion`), so None -- C starts with
        R (a goal) or now (R an event's predicate), noted -- is reached only
        when the compiler is called without that check."""
        # 1. C's own run length: nothing to time against without it.
        answer = self._answer("completion_timing", cid) or {}
        if not answer.get("target_duration"):
            self.notes.append(f"{cid}: no timing to finish with {ref} -> "
                              f"{'starts now' if ref in self.preds else 'starts with it'}")
            return None
        # 2. C's duration, read when the tree runs.
        self.clock = True
        target = answer["target_duration"]
        t_info = self._ref_info(target["ref"])
        duration = self._derive(self._goal_key(cid, "duration"), "to_seconds",
                                [self._read(t_info, target["ref"].get("property_field"))],
                                factor=t_info.seconds_per_unit)
        # 3. R's time left: read directly, or counted down from R's start.
        remaining_key = self._goal_key(ref, "remaining")
        if answer.get("ref_remaining"):
            source = answer["ref_remaining"]
            info = self._ref_info(source["ref"])
            key = self._read(info, source["ref"].get("property_field"))
            if source.get("reads_as") == "timestamp":
                self._derive(remaining_key, "remaining_from_timestamp", [key])
            else:
                self._derive(remaining_key, "to_seconds", [key], factor=info.seconds_per_unit)
        elif answer.get("ref_duration") and ref in self.preds:
            # An event's run has no start this plan stamps, so a duration alone
            # does not say how long is left.
            self.notes.append(f"{cid}: only {ref}'s total duration is known, not its "
                              f"remaining time -> starts now")
            return None
        elif answer.get("ref_duration"):
            source = answer["ref_duration"]
            info = self._ref_info(source["ref"])
            started = self.started_keys.setdefault(ref, self._goal_key(ref, "started_at"))
            self._derive(remaining_key, "remaining_from_duration",
                         [self._read(info, source["ref"].get("property_field")), started],
                         factor=info.seconds_per_unit)
        else:
            self.notes.append(f"{cid}: nothing gives {ref}'s remaining time -> "
                              f"{'starts now' if ref in self.preds else 'starts with it'}")
            return None
        # 4. Wait until R's time left is no more than C's run.
        return _one("await", f"{cid}: finish with {ref}", {
            "type": "compare", "name": f"{cid}: {ref} remaining <= {cid} duration",
            "key1": remaining_key, "key2": duration, "op": "le"})

    def _group(self, gid: str, with_start: bool = True) -> Optional[Dict[str, Any]]:
        """group(G): its unit, its after-followers in sequence, its with- and
        completes-followers alongside -- under its start wrapper."""
        self.placed.add(gid)
        if self._role(gid, "recurrence"):
            core = self._recurring(gid)
        else:
            if self.goals[gid].goal_kind == "maintenance" and not self._stops(gid):
                self.long_running = True
            core = self._unit(gid)

        # Completes-followers first: they may ask for G's start to be recorded.
        side = []
        for cid in self._followers(gid, "completes_with"):
            wait = self._completes_wait(gid, cid)
            group = self._group(cid, with_start=False)
            side.append(_seq(f"{cid}: timed", [wait, group]) if wait else group)
        # With-followers on the same device run right after G, in order: one
        # device cannot take two settings at the same instant (a fan's speed
        # sent with its power-on reaches it while it is still off). Others run
        # alongside G.
        withs = self._followers(gid, "with_goal")
        same = [wid for wid in withs if self._same_device(gid, wid)]
        side = [self._group(wid, with_start=False)
                for wid in withs if wid not in same] + side
        core = self._started(gid, core)

        tail = []
        if same:
            groups = [self._group(wid, with_start=False) for wid in same]
            tail.append(groups[0] if len(groups) == 1 else _seq(f"{gid} then on its device", groups))
        afters = [a for a in (self._after_chain(gid, fid)
                              for fid in self._followers(gid, "after_goal")) if a is not None]
        if afters:
            tail.append(afters[0] if len(afters) == 1 else _par_all(f"after {gid}", afters))
        if tail:
            # The same-device steps and G's after-followers both follow G; one
            # does not wait for the other.
            core = _seq(f"{gid} then", [core, tail[0] if len(tail) == 1
                                        else _par_all(f"after {gid}", tail)])
        if side:
            core = _par_all(f"{gid} with", [core] + side)
        return self._start(gid, core) if with_start else core

    def _same_device(self, gid: str, other: str) -> bool:
        """Both goals name the same device."""
        name = self.goals[gid].artifact_name
        return bool(name) and name == self.goals[other].artifact_name

    def _unused_triggers(self) -> None:
        """CompileError for a trigger predicate none of its goals waits for --
        neither starting on_trigger nor completing with it: the structure lost
        the link between the event and the goal, and running the goal without
        it would ignore part of the request."""
        for pid, pred in self.preds.items():
            if pred.role != "trigger":
                continue
            goals = [self.goals[g] for g in pred.applies_to or [] if g in self.goals]
            if goals and not any(g.timing and (g.timing.starts == "on_trigger"
                                               or g.timing.completes_with == pid)
                                 for g in goals):
                raise CompileError(f"{pid} is a trigger of {', '.join(pred.applies_to)}, "
                                   f"but none of them waits for it")

    def compile(self) -> CompiledPlan:
        """The tree, its runtime and notes (module docstring); CompileError
        when the structure cannot be built."""
        # 1. The structure's own consistency, before anything is built.
        self._unused_triggers()
        # 2. The heads: goals that start on their own. A goal that completes
        #    with another goal, its start left to us, is placed by that goal.
        heads = [gid for gid, goal in self.goals.items()
                 if ((goal.timing.starts if goal.timing else None) in HEAD_STARTS
                     and not (goal.timing and goal.timing.start_chosen))
                 or self._finishes_with_event(gid)]
        if not heads:
            raise CompileError("no goal starts on its own (every goal waits for another)")
        # 3. Each head's group, then the tree and the runtime that keeps its
        #    keys current.
        groups = [g for g in (self._group(gid) for gid in heads) if g is not None]
        if not groups:
            raise CompileError("every goal was left out")
        unplaced = sorted(set(self.goals) - self.placed)
        if unplaced:
            raise CompileError(f"{', '.join(unplaced)} wait for a goal that is not in the request")
        tree = groups[0] if len(groups) == 1 else _par_all("every goal", groups)
        runtime: Dict[str, Any] = {"reads": list(self.reads.values()), "derived": self.derived,
                                   "clock": self.clock or _uses_clock(tree, self.derived)}
        return CompiledPlan(tree=tree, runtime=runtime, long_running=self.long_running,
                            notes=self.notes)


_CLOCK_NODES = {"timer", "wait_until", "stamp"}
_CLOCK_OPS = {"remaining_from_timestamp", "remaining_from_duration"}


def _uses_clock(tree: Dict[str, Any], derived: List[Dict[str, Any]]) -> bool:
    """Whether anything in the plan needs the plan's clock."""
    if any(d["op"] in _CLOCK_OPS for d in derived):
        return True

    def walk(node: Any) -> bool:
        if not isinstance(node, dict):
            return False
        if node.get("type") in _CLOCK_NODES:
            return True
        if node.get("type") == "compare" and CLOCK_KEY in (node.get("key1"), node.get("key2")):
            return True
        return any(walk(child) for child in node.get("children") or [])

    return walk(tree)


def compile_dependency(graph: Graph, structure: GoalStructure,
                       subtrees: Dict[str, Dict[str, Any]],
                       predicates: Dict[str, PredicateTarget],
                       profile: Dict[str, Dict[str, Any]],
                       per_device: Optional[Dict[str, List[Tuple[str, Dict[str, Any]]]]] = None,
                       poll_seconds: float = DEFAULT_POLL_SECONDS,
                       tolerance: float = DEFAULT_TOLERANCE) -> CompiledPlan:
    """The tree for a profiled dependency structure, or CompileError."""
    return _Compiler(graph, structure, subtrees, per_device or {}, predicates, profile,
                     poll_seconds, tolerance).compile()
