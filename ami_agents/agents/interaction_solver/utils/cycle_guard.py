"""Cycle checks: send a start, pause, resume or stop only from a state that allows it.

Matter's Operational State cluster fixes which commands a cycle accepts in
which state (R1.3, Table 3 Pause and Table 4 Resume: only in Running or
Paused, else CommandInvalidInState; Start "from an operational state from
which it can be started"; never from Error). No action can put a cycle into a
state it is not in without inventing one -- pausing a stopped washer would mean
starting it first -- so these are **check-only** guards: the command is sent
when the state allows it, skipped as already done when the cycle is already
there, and the step fails, with the reason in its node names, otherwise. No
model is involved: the rule and the states are Matter's.

Which actions: every action of class `homeont:SetOperationalStateCommand` --
one action that starts, pauses, resumes or stops a cycle by the operational
state it sends (Matter `OperationalStateEnum`: 0 Stopped, 1 Running, 2 Paused,
3 Error) -- on any device, read through the operational-state property it
`saref:actsUpon`. In the home ontology that is the appliances with a cycle:
for example laundry washers, laundry dryers and dishwashers (Matter's
Operational State cluster; ovens and microwaves use it too). Not covered: a
device whose lifecycle commands are separate actions (a Home Assistant
vacuum's `start` / `pause`), whose state values are not Matter's.

    sends        the check                                  the tree
    2 Paused     already Paused: done; Running: pause;      selector[ ==2, sequence[ ==1, pause ] ]
                 Stopped / Error: fail
    1 Running    already Running: done; Stopped (start) or  selector[ ==1, sequence[ !=3, start ] ]
                 Paused (resume): send; Error: fail
    0 Stopped    already Stopped: done; else send           selector[ ==0, stop ]

Example -- "pause washer 1 when dryer 1 finishes", the pause guarded:

    selector "G4 utility_room_laundry_washer_1.operationalState Paused: cycle check"
      condition  operationalState == 2   "already paused"
      sequence
        condition  operationalState == 1 "running, so it can be paused"
        action     operationalState {"value": 2}

If washer 1 is stopped when dryer 1 finishes, nothing is sent and the step
fails on "running, so it can be paused".

Applied with the power-on guard, before it, to every plan
(`PowerOnGuardBehaviour`, the dependency planner's step 4b), so a stopped
device is not switched on only to be refused.

    guard_cycles(graph, tree)   (the tree with every cycle command checked, notes)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Graph
from rdflib.namespace import RDF

from ....shared.utils.namespaces import HOMEONT, SAREF, TD
from .condition_guard import OPERATIONAL_STATES
from .dependency_context import artifact_of_url
from .specific_capability_context import DIRECT_VALUE_KEY, _form, _title

# The name every cycle check's selector ends with; also how one already in the
# tree (a remembered tree being replayed) is recognised.
CHECK_NAME = "cycle check"

STOPPED, RUNNING, PAUSED, ERROR = 0, 1, 2, 3
STATE_NAMES = {STOPPED: "Stopped", RUNNING: "Running", PAUSED: "Paused", ERROR: "Error"}


# -- the TD graph ------------------------------------------------------------

def _cycle_action(graph: Graph, url: str) -> Tuple[Any, Optional[str]]:
    """(the device, the URL of the operational-state property its action at
    ``url`` sets) for an operational-state command; (None, None) otherwise."""
    artifact = artifact_of_url(graph, url)
    if artifact is None:
        return None, None
    action = next((a for a in graph.objects(artifact, TD.hasActionAffordance)
                   if _form(graph, a)[0] == url), None)
    if action is None or (action, RDF.type, HOMEONT.SetOperationalStateCommand) not in graph:
        return None, None
    state = next((p for p in graph.objects(action, SAREF.actsUpon)
                  if (p, RDF.type, HOMEONT.OperationalState) in graph), None)
    return artifact, (_form(graph, state)[0] or None) if state is not None else None


def _state_value(value: Any) -> Optional[int]:
    """The operational state an action sends: a number, or a Matter name."""
    named = str(value).strip().lower()
    if named in OPERATIONAL_STATES:
        return OPERATIONAL_STATES[named]
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number in STATE_NAMES else None


# -- the tree ----------------------------------------------------------------

def _condition(name: str, url: str, operator: str, value: int) -> Dict[str, Any]:
    return {"type": "condition", "name": name, "property_url": url,
            "expected_value": value, "operator": operator}


def checked(label: str, node: Dict[str, Any], state_url: str, sends: int) -> Dict[str, Any]:
    """``node`` (sending ``sends``) behind its cycle check -- the module
    docstring's table."""
    name = f"{label} {STATE_NAMES[sends]}: {CHECK_NAME}"
    # 1. Already in the state the command asks for: nothing to send.
    done = _condition(f"{label}: already {STATE_NAMES[sends].lower()}", state_url, "==", sends)
    # 2. The state the command may be sent from.
    if sends == PAUSED:
        allowed = [_condition(f"{label}: running, so it can be paused", state_url, "==", RUNNING),
                   node]
    elif sends == RUNNING:
        allowed = [_condition(f"{label}: not in error, so it can be started or resumed",
                              state_url, "!=", ERROR), node]
    else:
        allowed = [node]
    send = allowed[0] if len(allowed) == 1 else \
        {"type": "sequence", "name": f"{label}: allowed, send", "memory": True,
         "children": allowed}
    return {"type": "selector", "name": name, "memory": True, "children": [done, send]}


def is_cycle_check(node: Any) -> bool:
    """True for a selector built by `checked`."""
    return isinstance(node, dict) and node.get("type") == "selector" and \
        str(node.get("name", "")).endswith(CHECK_NAME)


def guard_cycles(graph: Graph, tree: Any, goal: Optional[str] = None
                 ) -> Tuple[Any, List[str]]:
    """(a copy of ``tree`` with every operational-state command behind its
    cycle check, notes on those left unchecked and why). A check already in
    the tree is kept as it is. ``goal`` prefixes the check's node names."""
    notes: List[str] = []

    def visit(node: Any) -> Any:
        """The node, its cycle commands checked."""
        if not isinstance(node, dict) or is_cycle_check(node):
            return node
        # 1. A cycle command: checked, when its state can be read and its value is one.
        if node.get("type") == "action" and node.get("action_url"):
            artifact, state_url = _cycle_action(graph, node["action_url"])
            if artifact is None:
                return node
            label = " ".join(p for p in (goal, f"{_title(graph, artifact)}.operationalState") if p)
            sends = _state_value((node.get("parameters") or {}).get(DIRECT_VALUE_KEY))
            if state_url is None or sends is None or sends == ERROR:
                notes.append(f"{label}: not cycle-checked ("
                             + ("no operational-state property it acts upon" if state_url is None
                                else "it does not send Stopped, Running or Paused") + ")")
                return node
            return checked(label, node, state_url, sends)
        # 2. Anything else: its children.
        if node.get("children"):
            return {**node, "children": [visit(child) for child in node["children"]]}
        return node

    return visit(copy.deepcopy(tree)), notes
