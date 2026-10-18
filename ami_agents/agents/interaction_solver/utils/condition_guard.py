"""Condition guards: check a device's state before a command, establish it when it does not hold.

A condition of `config/planning_guidelines.yaml` ("the operational state must
not be Running before the spin speed changes") becomes, in the plan, a guard
around the step it concerns:

    selector  "S1 washer_1.spinSpeedCurrent: <condition id> condition guard"
      sequence "holds"      condition  operationalState != 1        (not running)
                            the step   spinSpeedCurrent 1200
      sequence "establish"  condition  operationalState == 1        (running)
                            guard      operationalState 2           (pause)
                            the step   spinSpeedCurrent 1200
                            restore    operationalState 1           (resume)

The negated condition opens the second branch, so a step that fails although
the state held does not trigger the guard. Which check and which action guard
a step is the model's answer (`condition_check.py`) or a remembered decision
(`GuardStore`); everything here is deterministic, against the TD graph:

    resolve_guard     the answer's property and action found on the step's own
                      device, by title and class; values typed by their schema.
                      Not found / malformed -> no guard, with a note.
    restore_action    a guard that pauses, stops or switches off is followed by
                      the matching resume, start or switch-on: the same combined
                      action with the opposite value (homeont:SetOperationalState
                      Command -> Running, homeont:SetOnOffCommand -> true), else
                      the device's one separate saref:StartCommand / OnCommand
                      (a tie broken by name). Not found -> the step is not
                      guarded at all: a pause with no resume would leave the
                      cycle stopped.
    wrap_steps        each guarded step wrapped where it stands in the tree
    drop_satisfied    a guard is removed when the plan, earlier in execution
                      order, already establishes its check (`plan_order.py`);
                      e.g. "pause the washer, then set the spin" needs none

Worked example -- washer_1, a LaundryWasher; operationalState 0 Stopped,
1 Running, 2 Paused; the condition "spin changes need operationalState != Running":

    1. "set spin to 800", memory empty: the model answers check
       operationalState != 1, guard operationalState = 2. The restore is the
       same action with 1. Nothing earlier sets operationalState: the guard is
       kept. The washer was running: pause, spin, resume. The UA records the
       decision (without the restore).
    2. "set spin to 1200", remembered: the property and the action are found
       again on today's TD (`apply_remembered`), the restore derived again.
       Wrapped, no model call. The washer is stopped: the "holds" branch runs,
       spin only.
    3. "pause the washer, then set spin to 1200", remembered: G1 sets
       operationalState = 2 before G2, which satisfies != 1: the guard is
       dropped, with a note.

    resolve_guard(graph, step, answer)          (Guard | None, note | None)
    apply_remembered(graph, step, record)       Decision | None (None: a miss)
    restore_action(graph, artifact, action, value)
                                                (Invocation | None, required)
    wrap_steps(tree, steps, decisions)          the tree, guarded steps wrapped
    drop_satisfied(graph, tree)                 (the tree, notes)
    guards_wire(decisions, tree)                the plan entry's `guards`
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Graph, URIRef
from rdflib.namespace import RDF

from ....shared.utils.namespaces import HOMEONT, SAREF, TD, expand, shorten
from .dependency_compiler import _typed
from .dependency_context import artifact_of_url
from .plan_order import Facts, satisfies, same_value, walk_in_order
from .planning_conditions import CONDITION_GUARD_NAME, Step
from .specific_capability_context import (
    DIRECT_VALUE_KEY,
    _form,
    _input_schema,
    _title,
    json_schema,
)

# Matter OperationalStateEnum: the values of an appliance's operational state
# (also SimuHome's `shtd._OPERATIONAL_STATES`).
OPERATIONAL_STATES = {"stopped": 0, "running": 1, "paused": 2}
RUNNING = OPERATIONAL_STATES["running"]

# The comparisons a check may use, and the one that negates each.
NEGATED = {"==": "!=", "!=": "==", ">": "<=", "<=": ">", "<": ">=", ">=": "<"}

# A guard's sources, on its decision.
FROM_MODEL = "model"
FROM_MEMORY = "memory"

# Why a step has no guard.
NOT_APPLICABLE = "not_applicable"           # the condition does not concern it
ESTABLISHED_BY_PLAN = "established_by_plan"  # the plan sets the state before it
WHY = (NOT_APPLICABLE, ESTABLISHED_BY_PLAN)
# No guard because the check itself gave nothing usable (the call failed, the
# answer did not resolve): never the model's choice, never recorded.
CHECK_FAILED = "check_failed"


# -- what a guard is ---------------------------------------------------------

@dataclass
class Invocation:
    """One action to post: its affordance, title, class, URL and parameters."""

    affordance: Any
    title: str
    cls: Optional[str]
    url: str
    parameters: Dict[str, Any]

    @property
    def value(self) -> Any:
        """The single value it sends; None for a command that takes no input."""
        return self.parameters.get(DIRECT_VALUE_KEY)

    def to_wire_dict(self) -> Dict[str, Any]:
        """As the guard memory stores it (`GuardStore` record `action`)."""
        return {"title": self.title, "class": self.cls, "target": self.url,
                "value": self.value}


@dataclass
class Check:
    """One property comparison: what the guard reads, and what must hold."""

    affordance: Any
    title: str
    cls: Optional[str]
    url: str
    operator: str
    value: Any

    def to_wire_dict(self) -> Dict[str, Any]:
        """As the guard memory stores it (`GuardStore` record `check`)."""
        return {"title": self.title, "class": self.cls, "target": self.url,
                "operator": self.operator, "value": self.value}


@dataclass
class Guard:
    """A resolved guard: the check, the action that makes it hold, and the
    restore that follows the step (None when none is needed)."""

    check: Check
    action: Invocation
    restore: Optional[Invocation] = None


@dataclass
class Decision:
    """What was decided for one step: a guard, or none and why."""

    step: Step
    source: str                          # FROM_MODEL | FROM_MEMORY
    guard: Optional[Guard] = None
    why: Optional[str] = None            # with no guard: one of WHY
    reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """The guard's selector name: unique per step, ends with
        CONDITION_GUARD_NAME (how a guard is found again in the tree)."""
        ids = "+".join(c.id for c in self.step.conditions)
        return f"{self.step.id} {self.step.key.get('device_name')}." \
               f"{self.step.key.get('affordance_name')}: {ids} {CONDITION_GUARD_NAME}"

    def to_wire_dict(self) -> Dict[str, Any]:
        """The plan entry's `guards` item -- what the UA records after a
        successful run (`user_assistant/guards.py`)."""
        out: Dict[str, Any] = {"step": self.step.id, "goal": self.step.goal,
                               "key": self.step.key, "protection": self.guard is not None,
                               "source": self.source,
                               "conditions": [c.id for c in self.step.conditions]}
        if self.guard is not None:
            out.update({"name": self.name, "check": self.guard.check.to_wire_dict(),
                        "action": self.guard.action.to_wire_dict()})
            if self.guard.restore is not None:
                out["restore"] = self.guard.restore.to_wire_dict()
        else:
            out["why"] = self.why
        if self.reason:
            out["reason"] = self.reason
        return out


# -- the TD graph ------------------------------------------------------------

def _has_class(graph: Graph, node: Any, cls: Optional[str]) -> bool:
    """True when ``node`` is of class ``cls`` (a CURIE); a node of a subclass
    is too, the graph being closed under rdfs:subClassOf."""
    return bool(cls) and (node, RDF.type, URIRef(expand(cls))) in graph


def _affordance_by_title(graph: Graph, artifact: Any, predicate, title: Any,
                         cls: Any) -> Tuple[Any, Optional[str]]:
    """(the device's affordance under ``predicate`` with td:title ``title``,
    None) or (None, why not). When ``cls`` is given the affordance must be of
    that class: a title copied from the wrong affordance is caught here."""
    found = [a for a in graph.objects(artifact, predicate)
             if (_title(graph, a) or "") == str(title or "")]
    if len(found) != 1:
        return None, (f"no affordance titled {title!r}" if not found
                      else f"{len(found)} affordances titled {title!r}")
    if cls and not _has_class(graph, found[0], str(cls)):
        return None, f"{title!r} is not a {cls}"
    return found[0], None


def _value_schema(graph: Graph, affordance: Any, output: bool) -> Dict[str, Any]:
    """The schema of the single value a property reads (output) or an action
    takes (input; a direct-value input is wrapped in {"value": ...})."""
    if output:
        schema = graph.value(affordance, TD.hasOutputSchema)
        return json_schema(graph, schema) if schema is not None else {}
    wrapped = _input_schema(graph, graph.value(affordance, TD.hasInputSchema)) or {}
    return (wrapped.get("properties") or {}).get(DIRECT_VALUE_KEY) or {}


def _typed_value(text: Any, schema: Dict[str, Any], operational: bool) -> Any:
    """``text`` as the schema types it. An operational state may be named
    ("Paused") where its schema is a number: Matter's enum gives the number.
    Raises ValueError when the value cannot be read."""
    # 1. A named operational state, on a numeric schema.
    named = str(text).strip().lower()
    if operational and named in OPERATIONAL_STATES and schema.get("type") in ("integer", "number"):
        return OPERATIONAL_STATES[named]
    # 2. Everything else, as the dependency compiler types determined values.
    if text is None:
        raise ValueError("no value")
    return _typed(text, schema)


def _is_operational(graph: Graph, affordance: Any) -> bool:
    """True for an operational-state property or command."""
    return ((affordance, RDF.type, HOMEONT.OperationalState) in graph
            or (affordance, RDF.type, HOMEONT.SetOperationalStateCommand) in graph)


def _most_specific_class(graph: Graph, affordance: Any, given: Optional[str]) -> Optional[str]:
    """The class to remember an affordance under: the one the answer named
    (it was checked), else its first homeont / SAREF type."""
    if given:
        return str(given)
    types = sorted(shorten(t) for t in graph.objects(affordance, RDF.type)
                   if str(t).startswith((str(HOMEONT), str(SAREF))))
    return types[0] if types else None


# -- the restore -------------------------------------------------------------

def _invocation(graph: Graph, affordance: Any, value: Any = None) -> Invocation:
    """An invocation of ``affordance``, sending ``value`` (none: no parameters)."""
    return Invocation(affordance=affordance, title=_title(graph, affordance) or "",
                      cls=_most_specific_class(graph, affordance, None),
                      url=_form(graph, affordance)[0],
                      parameters={} if value is None else {DIRECT_VALUE_KEY: value})


def _separate(graph: Graph, artifact: Any, cls, words: Tuple[str, ...]) -> Any:
    """The device's one separate action of ``cls`` (not a combined command,
    which is of the opposite class too); several -> the first whose title
    contains one of ``words``, in that order; else None."""
    combined = (HOMEONT.SetOnOffCommand, HOMEONT.SetOperationalStateCommand)
    found = [a for a in graph.objects(artifact, TD.hasActionAffordance)
             if (a, RDF.type, cls) in graph
             and not any((a, RDF.type, c) in graph for c in combined)]
    if len(found) == 1:
        return found[0]
    for word in words:
        named = [a for a in found if word in (_title(graph, a) or "").lower()]
        if len(named) == 1:
            return named[0]
    return None


def restore_action(graph: Graph, artifact: Any, action: Any, value: Any
                   ) -> Tuple[Optional[Invocation], bool]:
    """(the invocation that undoes a guard action, whether one is required).

    A guard that switches off, pauses or stops must be followed by the switch-
    on, resume or start (required = True); any other guard needs none. Found
    deterministically: the same combined action with the opposite value, else
    the device's one separate command of the opposite kind. Required but not
    found -> (None, True): the caller leaves the step unguarded."""
    # 1. A combined on/off command sending false: the same, sending true.
    if (action, RDF.type, HOMEONT.SetOnOffCommand) in graph:
        return ((_invocation(graph, action, True), True) if same_value(value, False)
                else (None, False))
    # 2. A combined operational-state command pausing or stopping: the same,
    #    sending Running (a resume from Paused, a start from Stopped).
    if (action, RDF.type, HOMEONT.SetOperationalStateCommand) in graph:
        stops = same_value(value, OPERATIONAL_STATES["paused"]) or \
            same_value(value, OPERATIONAL_STATES["stopped"])
        return (_invocation(graph, action, RUNNING), True) if stops else (None, False)
    # 3. A separate off / pause / stop: the device's separate on / start.
    for undone, undoing, words in ((SAREF.OffCommand, SAREF.OnCommand, ("on",)),
                                   (SAREF.PauseCommand, SAREF.StartCommand, ("resume", "start")),
                                   (SAREF.StopCommand, SAREF.StartCommand, ("start", "resume"))):
        if (action, RDF.type, undone) in graph:
            found = _separate(graph, artifact, undoing, words)
            return (_invocation(graph, found) if found is not None else None), True
    # 4. Anything else changes a setting: nothing to undo.
    return None, False


# -- resolving an answer -----------------------------------------------------

def resolve_guard(graph: Graph, step: Step, answer: Dict[str, Any]
                  ) -> Tuple[Optional[Guard], Optional[str]]:
    """(the guard ``answer`` names for ``step``, None) or (None, why not).

    ``answer`` has `check: {property, property_class, operator, value}` and
    `action: {action, action_class, value}` (the model's shape; a remembered
    record is converted to it by `apply_remembered`). Both are looked up on the
    step's own device by title, their class checked; values typed by the
    property's output / the action's input schema."""
    check, action = answer.get("check"), answer.get("action")
    if not isinstance(check, dict) or not isinstance(action, dict):
        return None, "the answer has no check or no action"

    # 1. The property the guard reads.
    prop, why = _affordance_by_title(graph, step.artifact, TD.hasPropertyAffordance,
                                     check.get("property"), check.get("property_class"))
    if prop is None:
        return None, f"check property: {why}"
    operator = str(check.get("operator") or "==").strip()
    if operator not in NEGATED:
        return None, f"check operator {operator!r} is not a comparison"

    # 2. The action that makes it hold: never the step's own.
    act, why = _affordance_by_title(graph, step.artifact, TD.hasActionAffordance,
                                    action.get("action"), action.get("action_class"))
    if act is None:
        return None, f"guard action: {why}"
    if act == step.action and same_value(action.get("value"), step.value):
        return None, "the guard action is the step itself"

    # 3. The values, typed; their URLs. A separate command (a pause, an
    #    "on") takes no input: it is sent with no parameters.
    takes_input = graph.value(act, TD.hasInputSchema) is not None
    try:
        check_value = _typed_value(check.get("value"), _value_schema(graph, prop, True),
                                   _is_operational(graph, prop))
        action_value = _typed_value(action.get("value"), _value_schema(graph, act, False),
                                    _is_operational(graph, act)) if takes_input else None
    except (TypeError, ValueError) as exc:
        return None, f"unreadable value: {exc}"
    prop_url, act_url = _form(graph, prop)[0], _form(graph, act)[0]
    if not prop_url or not act_url:
        return None, "the check property or the guard action has no URL"

    # 4. The restore a pause / stop / switch-off needs; without it, no guard.
    restore, required = restore_action(graph, step.artifact, act, action_value)
    if required and restore is None:
        return None, f"{_title(graph, act)} {action_value!r} has no matching resume / start / on"

    return Guard(
        check=Check(affordance=prop, title=_title(graph, prop) or "",
                    cls=_most_specific_class(graph, prop, check.get("property_class")),
                    url=prop_url, operator=operator, value=check_value),
        action=Invocation(affordance=act, title=_title(graph, act) or "",
                          cls=_most_specific_class(graph, act, action.get("action_class")),
                          url=act_url,
                          parameters={DIRECT_VALUE_KEY: action_value} if takes_input else {}),
        restore=restore), None


def apply_remembered(graph: Graph, step: Step, record: Dict[str, Any]) -> Optional[Decision]:
    """The decision a remembered record makes for ``step``, or None (a miss).

    The store only looked the key up; nothing in the record is trusted as it
    is. A guard is resolved again on today's TD graph, by the titles and
    classes it stored (not its URLs), and its restore derived again. When that
    fails -- the TD changed, the device was replaced -- the hit is a miss and
    the step goes to the model. Whether this plan already establishes the
    check is decided later, for every guard alike (`drop_satisfied`)."""
    # 1. A remembered "no guard": only a device-level reason is ever stored.
    if record.get("protection") is False:
        return Decision(step=step, source=FROM_MEMORY,
                        why=record.get("why") or NOT_APPLICABLE)

    # 2. A remembered guard, in the model's shape, resolved again.
    check, action = record.get("check") or {}, record.get("action") or {}
    answer = {"check": {"property": check.get("title"), "property_class": check.get("class"),
                        "operator": check.get("operator") or "==", "value": check.get("value")},
              "action": {"action": action.get("title"), "action_class": action.get("class"),
                         "value": action.get("value")}}
    guard, _ = resolve_guard(graph, step, answer)
    return Decision(step=step, source=FROM_MEMORY, guard=guard) if guard is not None else None


# -- the tree ----------------------------------------------------------------

def _condition(name: str, check: Check, operator: str) -> Dict[str, Any]:
    """A condition node reading the check's property."""
    return {"type": "condition", "name": name, "property_url": check.url,
            "expected_value": check.value, "operator": operator}


def _action(name: str, invocation: Invocation) -> Dict[str, Any]:
    """An action node posting the invocation."""
    return {"type": "action", "name": name, "action_url": invocation.url,
            "parameters": dict(invocation.parameters)}


def wrapped(decision: Decision, node: Dict[str, Any]) -> Dict[str, Any]:
    """``node`` behind ``decision``'s guard -- the module docstring's tree."""
    guard, label = decision.guard, decision.name.rsplit(":", 1)[0]
    check = guard.check
    # 1. The state holds: the step alone.
    holds = {"type": "sequence", "name": f"{label}: holds", "memory": True, "children": [
        _condition(f"{label}: {check.title} {check.operator} {check.value!r}",
                   check, check.operator), node]}
    # 2. It does not: establish it, the step, then undo what the guard did.
    establish = [_condition(f"{label}: {check.title} {NEGATED[check.operator]} {check.value!r}",
                            check, NEGATED[check.operator]),
                 _action(f"{label}: guard {guard.action.title} {guard.action.value!r}",
                         guard.action),
                 node]
    if guard.restore is not None:
        establish.append(_action(f"{label}: restore {guard.restore.title} "
                                 f"{guard.restore.value!r}", guard.restore))
    return {"type": "selector", "name": decision.name, "memory": True, "children": [
        holds, {"type": "sequence", "name": f"{label}: establish", "memory": True,
                "children": establish}]}


def wrap_steps(tree: Dict[str, Any], steps: List[Step],
               decisions: Dict[str, Decision]) -> Dict[str, Any]:
    """A copy of ``tree`` with every step that has a guard wrapped where it
    stands (by its path). Steps are leaves, so wrapping one moves no other."""
    out = copy.deepcopy(tree)
    for step in steps:
        decision = decisions.get(step.id)
        if decision is None or decision.guard is None:
            continue
        # 1. The root itself is the step.
        if not step.path:
            out = wrapped(decision, out)
            continue
        # 2. Else its parent's child at the last index.
        parent = out
        for index in step.path[:-1]:
            parent = parent["children"][index]
        parent["children"][step.path[-1]] = wrapped(decision, parent["children"][step.path[-1]])
    return out


def is_condition_guard(node: Any) -> bool:
    """True for a selector built by `wrapped`."""
    return isinstance(node, dict) and node.get("type") == "selector" and \
        str(node.get("name", "")).endswith(CONDITION_GUARD_NAME)


def guard_names(tree: Any) -> List[str]:
    """The names of the condition guards in ``tree``, in tree order."""
    out: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if is_condition_guard(node):
                out.append(node["name"])
            for child in node.get("children") or []:
                walk(child)

    walk(tree)
    return out


def drop_satisfied(graph: Graph, tree: Any) -> Tuple[Any, List[str]]:
    """(``tree`` without the condition guards the plan already satisfies,
    notes on the guards dropped).

    Walked in execution order (`plan_order.walk_in_order`); the facts are
    keyed by property URL:

        an action             sets the property it `saref:actsUpon` to the
                              value it sends
        a condition           ==/!= its value holds after it (in a sequence)
        a condition guard     its check is satisfied by an earlier fact ->
                              replaced by its step; else kept, its step walked
                              from the check (which holds there either way),
                              and its property uncertain afterwards when a
                              restore runs on one branch only

    Run on the whole plan -- for a dependency plan on the compiled tree, where
    "earlier" is the real order across goals."""
    notes: List[str] = []

    def acted_upon(node: Dict[str, Any]) -> List[str]:
        """The property URLs an action node writes."""
        artifact = artifact_of_url(graph, node["action_url"])
        if artifact is None:
            return []
        for action in graph.objects(artifact, TD.hasActionAffordance):
            if _form(graph, action)[0] == node["action_url"]:
                return [u for p in graph.objects(action, SAREF.actsUpon)
                        if (u := _form(graph, p)[0])]
        return []

    def writes(node: Dict[str, Any]):
        """The properties an action writes; None for every other node."""
        if node.get("type") == "action" and node.get("action_url"):
            return set(acted_upon(node))
        return None

    def leaf(node: Dict[str, Any], facts: Facts, path):
        """The walk's step: guards kept or dropped, facts recorded."""
        # 1. A condition guard: drop it when an earlier fact satisfies it.
        if is_condition_guard(node):
            holds = node["children"][0]
            check, step = holds["children"][0], holds["children"][1]
            url, op, value = check["property_url"], check.get("operator") or "==", \
                check["expected_value"]
            if satisfies(facts.get(url), op, value):
                notes.append(f"{node['name'].rsplit(':', 1)[0]}: not guarded, the plan "
                             f"sets {url} before it")
                return walk_in_order(step, leaf, writes, facts)
            inner, after = walk_in_order(step, leaf, writes, {**facts, url: (op, value)})
            restores = len(node["children"][1]["children"]) > 3
            kept = copy.deepcopy(node)
            kept["children"][0]["children"][1] = inner
            kept["children"][1]["children"][2] = inner
            return kept, ({k: v for k, v in after.items() if k != url} if restores else after)

        # 2. An action: the properties it writes now hold its value.
        if node.get("type") == "action" and node.get("action_url"):
            value = (node.get("parameters") or {}).get(DIRECT_VALUE_KEY)
            written = acted_upon(node)
            if value is None:
                return node, {k: v for k, v in facts.items() if k not in written}
            return node, {**facts, **{url: ("==", value) for url in written}}

        # 3. A condition (or a wait for one): once passed, its comparison holds.
        if node.get("type") in ("condition", "wait_condition") and node.get("property_url"):
            op = node.get("operator") or "=="
            if op in ("==", "!="):
                return node, {**facts, node["property_url"]: (op, node.get("expected_value"))}
            return node, facts
        return None

    out, _ = walk_in_order(tree, leaf, writes)
    return out, notes


def guards_wire(decisions: Dict[str, Decision], tree: Any) -> List[Dict[str, Any]]:
    """The plan entry's `guards`: every decision, in step order, a guard that
    `drop_satisfied` removed from ``tree`` marked `dropped` (it is not
    recorded: it was not needed by this plan, which says nothing about the
    device)."""
    present = set(guard_names(tree))
    out = []
    for step_id in sorted(decisions, key=lambda s: int(s[1:]) if s[1:].isdigit() else 0):
        wire = decisions[step_id].to_wire_dict()
        if wire["protection"] and wire["name"] not in present:
            wire["dropped"] = True
        out.append(wire)
    return out
