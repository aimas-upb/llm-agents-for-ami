"""The device conditions of `config/planning_guidelines.yaml`, and the plan steps they concern.

A condition is a device state that must hold before a command is sent ("the
operational state must not be Running before the spin speed changes"). This
module reads them and, deterministically, finds the steps of a plan a
condition applies to -- the questions the condition-guard check asks
(`condition_check.py`) and the guards it builds (`condition_guard.py`).

Example -- a simple plan for "set washer 1's spin to 1200":

    tree       action .../washer_1/actions/spinSpeedCurrent {"value": 1200}
    the action a saref:SetAbsoluteLevelCommand, acting upon a property of
               class homeont:LaundryWasherSpinSpeedCurrent (a subclass of
               homeont:LaundryWasherControlsSpinSpeedCurrent)
    washer 1   a homeont:LaundryWasher
    matches    laundry_washer_settings_not_running: its class, one of its
               `commands`, one of its `acts_upon`
    step       Step(id="S1", path=(), goal=None, node=<the action>,
                    conditions=[that condition], key={affordance_name:
                    "spinSpeedCurrent", ..., invocation_value: None})

`power_on_first` is not loaded here: the power-on guard applies it, with no
model involved (`power_on_guard.py`).

    load_conditions(path)                 every other condition, read once
    matching(graph, artifact, action, conditions)
                                          the conditions one action is subject to
    conditioned_steps(graph, tree, conditions, goal, first)
                                          the plan's steps that have conditions
    guard_key(graph, artifact, action, parameters)
                                          what a decision is remembered under
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rdflib import Graph, URIRef
from rdflib.namespace import RDF

from ....shared.utils.namespaces import HOMEONT, SAREF, TD, expand
from .dependency_context import artifact_of_url
from .power_on_guard import (
    DEFAULT_GUIDELINES_PATH,
    GUARD_NAME as POWER_ON_GUARD_NAME,
    POWER_ON_ENTRY,
    applies,
)
from .specific_capability_context import (
    DIRECT_VALUE_KEY,
    _command_types,
    _form,
    _title,
    device_class,
    device_location,
)

# The name a condition guard's selector ends with; also how a step already
# guarded (a remembered tree being replayed) is recognised.
CONDITION_GUARD_NAME = "condition guard"

# Commands whose value says which command they are: one operationalState
# action starts, pauses or stops; one onOff action switches on or off. The
# value is part of their key, so a pause and a start are decided apart.
VALUE_SELECTS_COMMAND = (HOMEONT.SetOperationalStateCommand, HOMEONT.SetOnOffCommand)


# -- the file ----------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """One entry of the file's `conditions:` list."""

    id: str
    applies_to: Dict[str, Any]
    commands: Tuple[str, ...]
    acts_upon: Tuple[str, ...]          # empty: any property
    text: str

    def to_wire_dict(self) -> Dict[str, Any]:
        """What the condition-guard model reads of it."""
        return {"id": self.id, "condition": self.text}


@lru_cache(maxsize=8)
def load_conditions(path: Optional[str] = None) -> Tuple[Condition, ...]:
    """Every condition in the file at ``path`` (the repo file when None)
    except `power_on_first`; empty when the file is missing. Read once per
    path. An entry without an id, commands or text is skipped: it could match
    nothing or say nothing."""
    # 1. The file; a missing one means no conditions.
    file = Path(path or DEFAULT_GUIDELINES_PATH)
    if not file.is_file():
        return ()
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}

    # 2. Each usable entry, power-on excluded.
    out: List[Condition] = []
    for entry in data.get("conditions") or []:
        if not isinstance(entry, dict) or entry.get("id") == POWER_ON_ENTRY:
            continue
        if not (entry.get("id") and entry.get("commands") and entry.get("text")):
            continue
        out.append(Condition(
            id=str(entry["id"]), applies_to=entry.get("applies_to") or {},
            commands=tuple(entry["commands"]), acts_upon=tuple(entry.get("acts_upon") or ()),
            text=" ".join(str(entry["text"]).split())))
    return tuple(out)


# -- matching ----------------------------------------------------------------

def _acted_upon(graph: Graph, action: Any) -> List[Any]:
    """The property affordances an action writes (`saref:actsUpon`)."""
    return list(graph.objects(action, SAREF.actsUpon))


def matching(graph: Graph, artifact: Any, action: Any,
             conditions: Tuple[Condition, ...]) -> List[Condition]:
    """The conditions the action ``action`` of device ``artifact`` is subject to.

    A condition matches when it applies to the device (class or exact
    manufacturer, `power_on_guard.applies`), the action is of one of its
    `commands`, and -- when it lists `acts_upon` -- the action writes a
    property of one of those classes. The graph is closed under
    rdfs:subClassOf, so subclasses match through their materialised types."""
    found = []
    props = None
    for condition in conditions:
        # 1. The device.
        if not applies(graph, artifact, condition.applies_to):
            continue
        # 2. The command.
        if not any((action, RDF.type, URIRef(expand(c))) in graph for c in condition.commands):
            continue
        # 3. What it writes, when the condition narrows to some properties.
        if condition.acts_upon:
            props = _acted_upon(graph, action) if props is None else props
            if not any((p, RDF.type, URIRef(expand(c))) in graph
                       for p in props for c in condition.acts_upon):
                continue
        found.append(condition)
    return found


# -- steps -------------------------------------------------------------------

def guard_key(graph: Graph, artifact: Any, action: Any,
              parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The key a decision about this action of this device is remembered
    under (`GuardStore.KEY_FIELDS`): the action's title and command class, the
    device's title and class, its room's class -- and the value sent, for the
    commands whose value says which command they are."""
    commands = _command_types(graph, action)
    selects = any((action, RDF.type, c) in graph for c in VALUE_SELECTS_COMMAND)
    value = (parameters or {}).get(DIRECT_VALUE_KEY) if selects else None
    return {"affordance_name": _title(graph, action),
            "affordance_class": commands[0] if commands else None,
            "device_name": _title(graph, artifact),
            "device_class": device_class(graph, artifact),
            "room_class": device_location(graph, artifact),
            "invocation_value": None if value is None else str(value)}


@dataclass
class Step:
    """One action invocation of a plan that has conditions."""

    id: str                             # S1, S2, ... across one check
    path: Tuple[int, ...]               # child indices from the tree's root
    goal: Optional[str]                 # the goal it achieves (dependency plans)
    node: Dict[str, Any]                # the action node, as planned
    artifact: Any
    action: Any                         # its action affordance in the TD graph
    conditions: List[Condition]
    key: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> Any:
        """The value the step sends, when it sends a single one."""
        return (self.node.get("parameters") or {}).get(DIRECT_VALUE_KEY)


def _is_guard(node: Dict[str, Any]) -> bool:
    """True for a guard already in the tree -- power-on or condition: the
    actions inside it were checked when it was built."""
    name = str(node.get("name", ""))
    return name.endswith(POWER_ON_GUARD_NAME) or name.endswith(CONDITION_GUARD_NAME)


def _action_of(graph: Graph, artifact: Any, url: str) -> Any:
    """The device's action affordance posting to ``url``, or None."""
    return next((a for a in graph.objects(artifact, TD.hasActionAffordance)
                 if _form(graph, a)[0] == url), None)


def conditioned_steps(graph: Graph, tree: Any, conditions: Tuple[Condition, ...],
                      goal: Optional[str] = None, first: int = 1) -> List[Step]:
    """Every action of ``tree`` subject to at least one condition, in tree
    order, numbered S<first>, S<first+1>, ... Actions inside a guard already
    there are skipped. ``goal`` is recorded on each step (a dependency
    structure's goal id)."""
    steps: List[Step] = []

    def walk(node: Any, path: Tuple[int, ...]) -> None:
        """Depth-first, children in order -- the order `walk_in_order` visits."""
        if not isinstance(node, dict) or _is_guard(node):
            return
        # 1. An action: its device, its affordance, its conditions.
        if node.get("type") == "action" and node.get("action_url"):
            artifact = artifact_of_url(graph, node["action_url"])
            action = _action_of(graph, artifact, node["action_url"]) if artifact else None
            found = matching(graph, artifact, action, conditions) if action is not None else []
            if found:
                steps.append(Step(
                    id=f"S{first + len(steps)}", path=path, goal=goal, node=node,
                    artifact=artifact, action=action, conditions=found,
                    key=guard_key(graph, artifact, action, node.get("parameters"))))
            return
        # 2. Anything else: its children.
        for index, child in enumerate(node.get("children") or []):
            walk(child, path + (index,))

    walk(tree, ())
    return steps
