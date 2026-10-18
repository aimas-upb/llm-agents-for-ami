"""The "power on first" guard: switch a device on before setting it, deterministically.

The `power_on_first` condition of `config/planning_guidelines.yaml` says: before
any command other than switching the device on or off, check that the device is
switched on; if not, switch it on first. For plans that are not part of a
dependency structure, this module puts that check in front of every mode, level
or operational-state command on a device the entry applies to -- found in the TD
graph, no model involved.

Example -- "set dryer 1 to Normal dryness" (dryer 1 is a homeont:LaundryDryer,
listed in the entry's `applies_to.classes`):

    planned   action  .../dryer_1/actions/selectedDrynessLevel  {"value": 1}
              (a saref:SetAbsoluteLevelCommand -> guarded)
    lookup    dryer 1's homeont:SetOnOffCommand action .../dryer_1/actions/onOff,
              which saref:actsUpon the property .../dryer_1/properties/onOff
    guarded   sequence "G1: power on first"
                selector
                  condition  .../properties/onOff == true      (on already: done)
                  action     .../actions/onOff {"value": true}  (else switch on)
                action  .../actions/selectedDrynessLevel {"value": 1}

When the device has no on/off command (or more than one, or the command acts
upon no property, or either has no URL), the action is left unguarded and a
note says why. An action already behind this guard -- a remembered tree being
replayed -- is not wrapped again.

Used on every plan: by `PowerOnGuardBehaviour` for plans outside dependency
structures, and by the dependency planner on its compiled tree. Either way an
action is not guarded when the plan switches its device on before it, in
execution order (see `guard_tree`). Example -- qt4-3 seed 41, a dependency plan:

    G1  dryer 1 onOff true            switches the dryer on
    G2  dryer 1 operationalState 1    after G1 -> already on, no guard
    G3  dryer 1 dryness level 1       after G2 -> already on, no guard

Without G1, G2 would be guarded, and G3 still not: G2's guard switches it on.

    load_applies_to(path)                 the entry's applies_to, read once
    guidelines_path(config)               `planning.guidelines.path`, if set
    applies(graph, artifact, applies_to)  by class, or by exact manufacturer
    power_on_affordances(graph, artifact) (on/off property URL, on/off action URL)
    guard_tree(graph, tree, applies_to, goal)
                                          (the guarded copy of the tree, notes)
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rdflib import Graph, URIRef
from rdflib.namespace import RDF

from ....shared.utils.namespaces import HOMEONT, SAREF, TD, expand
from .dependency_context import artifact_of_url
from .plan_order import satisfies, walk_in_order
from .specific_capability_context import DIRECT_VALUE_KEY, _form, _title

# The guidelines entry this module implements.
POWER_ON_ENTRY = "power_on_first"

# The repo's guidelines file, used when `planning.guidelines.path` is not set.
DEFAULT_GUIDELINES_PATH = str(Path(__file__).resolve().parents[3] / "config"
                              / "planning_guidelines.yaml")

# The commands guarded: settings and lifecycle commands, the ones a device that
# is switched off may refuse (Matter "dead front").
GUARDED_COMMANDS = (HOMEONT.SetModeCommand, SAREF.SetAbsoluteLevelCommand,
                    HOMEONT.SetOperationalStateCommand)

# The device's on/off command, whose `saref:actsUpon` is its on/off state.
SET_ON_OFF = HOMEONT.SetOnOffCommand

# The TD annotation naming a device's manufacturer.
SCHEMA_MANUFACTURER = URIRef("https://schema.org/manufacturer")

# The name of a guard's sequence; also how an already guarded action is recognised.
GUARD_NAME = "power on first"


# -- configuration -----------------------------------------------------------

@lru_cache(maxsize=8)
def load_applies_to(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The `applies_to` of the `power_on_first` condition in the guidelines file at
    ``path`` (the repo file when None); None when the file or the entry is
    missing, which turns the guard off. Read once per path."""
    # 1. The file; a missing one means no guard.
    file = Path(path or DEFAULT_GUIDELINES_PATH)
    if not file.is_file():
        return None
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    # 2. The entry, by its id.
    for entry in data.get("conditions") or []:
        if isinstance(entry, dict) and entry.get("id") == POWER_ON_ENTRY:
            return entry.get("applies_to") or {}
    return None


def guidelines_path(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """`planning.guidelines.path` from the InteractionSolver's config; None
    (also for an empty string, the yaml default) means the repo file."""
    planning = (config or {}).get("planning") or {}
    return ((planning.get("guidelines") or {}).get("path")) or None


# -- the graph ---------------------------------------------------------------

def applies(graph: Graph, artifact: Any, applies_to: Dict[str, Any]) -> bool:
    """True when the entry applies to the device ``artifact``: one of its
    classes is in `applies_to.classes`, or its schema:manufacturer is exactly
    one of `applies_to.manufacturers` (a literal match, case included).

    The discovered graph is closed under rdfs:subClassOf (`close_types`), so a
    device of a subclass carries the listed class among its rdf:types."""
    # 1. By class.
    types = set(graph.objects(artifact, RDF.type))
    if any(URIRef(expand(c)) in types for c in applies_to.get("classes") or []):
        return True
    # 2. By manufacturer, literally.
    makers = {str(m) for m in graph.objects(artifact, SCHEMA_MANUFACTURER)}
    return any(str(m) in makers for m in applies_to.get("manufacturers") or [])


def power_on_affordances(graph: Graph, artifact: Any
                         ) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
    """((on/off property URL, on/off action URL), None), or (None, why not).

    The device's one action typed homeont:SetOnOffCommand, and the property it
    `saref:actsUpon` (the on/off state the guard reads). No such action, more
    than one, no property, or a missing URL: no guard."""
    # 1. Exactly one on/off command: with two, which one switches it on is a guess.
    commands = [a for a in graph.objects(artifact, TD.hasActionAffordance)
                if (a, RDF.type, SET_ON_OFF) in graph]
    if len(commands) != 1:
        return None, (f"{len(commands)} on/off commands" if commands
                      else "no on/off command")
    command = commands[0]
    # 2. The on/off state it writes: exactly one property.
    states = list(graph.objects(command, SAREF.actsUpon))
    if len(states) != 1:
        return None, "the on/off command acts upon no single property"
    # 3. Both URLs: where the guard reads, where it posts.
    property_url, _ = _form(graph, states[0])
    action_url, _ = _form(graph, command)
    if not property_url or not action_url:
        return None, "the on/off state or command has no URL"
    return (property_url, action_url), None


# -- the tree ----------------------------------------------------------------

def _guarded(label: str, node: Dict[str, Any], property_url: str,
             action_url: str) -> Dict[str, Any]:
    """sequence[selector[on already?, switch on], node] -- see the module example."""
    return {"type": "sequence", "name": f"{label}: {GUARD_NAME}", "memory": True, "children": [
        {"type": "selector", "name": f"{label}: on, or switch on", "memory": True, "children": [
            {"type": "condition", "name": f"{label}: switched on",
             "property_url": property_url, "expected_value": True, "operator": "=="},
            {"type": "action", "name": f"{label}: switch on", "action_url": action_url,
             "parameters": {DIRECT_VALUE_KEY: True}}]},
        node]}


def _is_guard(node: Dict[str, Any]) -> bool:
    """True for a power-on guard already in the tree (built by `_guarded`)."""
    return node.get("type") == "sequence" and str(node.get("name", "")).endswith(GUARD_NAME)


def _affordance(graph: Graph, artifact: Any, url: str) -> Any:
    """The device's action affordance posting to ``url``, or None."""
    return next((a for a in graph.objects(artifact, TD.hasActionAffordance)
                 if _form(graph, a)[0] == url), None)


def _true(value: Any) -> bool:
    """True for the values an on/off command means "on" by."""
    return value is True or value == 1 or str(value).strip().lower() == "true"


def _on_off(graph: Graph, node: Dict[str, Any]) -> Optional[Tuple[Any, bool]]:
    """(the device, on?) when ``node`` sets a device's on/off state, else None.

    An action of a homeont:SetOnOffCommand sets it to the value it sends; a
    power-on guard already in the tree switches its device on (through its
    own switch-on action, which runs whenever the device was off)."""
    # 1. An existing guard: its switch-on action is in its selector.
    if _is_guard(node):
        selector = (node.get("children") or [{}])[0]
        actions = [c for c in selector.get("children") or [] if c.get("type") == "action"]
        artifact = artifact_of_url(graph, actions[0]["action_url"]) if actions else None
        return (artifact, True) if artifact is not None else None
    # 2. An on/off command, with the value it sends.
    if node.get("type") != "action" or not node.get("action_url"):
        return None
    artifact = artifact_of_url(graph, node["action_url"])
    affordance = _affordance(graph, artifact, node["action_url"]) if artifact else None
    if affordance is None or (affordance, RDF.type, SET_ON_OFF) not in graph:
        return None
    return artifact, _true((node.get("parameters") or {}).get(DIRECT_VALUE_KEY))


def guard_tree(graph: Graph, tree: Dict[str, Any], applies_to: Dict[str, Any],
               goal: Optional[str] = "G1") -> Tuple[Dict[str, Any], List[str]]:
    """(a copy of ``tree`` with each guarded action behind the power-on guard,
    notes on the actions left unguarded and why).

    An action is guarded when its affordance is one of GUARDED_COMMANDS, the
    entry applies to its device, and the plan has not already switched that
    device on before it. "Before" follows execution order, as
    `plan_order.walk_in_order` carries it: the fact is "this device is on",
    set by a switch-on (or a guard already there -- a replayed tree), unset by
    a switch-off; a sequence carries it from child to child, a selector's
    branch, a parallel child or a decorated child does not carry it past the
    node.

    Conservative: a switch-on that may not have run does not count, so the
    worst case is one guard too many -- which only reads the state once.
    ``goal`` names the guard's nodes; None names them after the device (a
    dependency tree holds many goals)."""
    devices: Dict[Any, Tuple[Optional[Tuple[str, str]], Optional[str]]] = {}
    notes: List[str] = []

    def lookup(artifact: Any):
        """The device's on/off affordances, resolved once per device."""
        if artifact not in devices:
            devices[artifact] = power_on_affordances(graph, artifact)
        return devices[artifact]

    def writes(node: Dict[str, Any]):
        """The device whose on/off state ``node`` sets, if any."""
        found = _on_off(graph, node)
        return {found[0]} if found is not None else None

    def leaf(node: Dict[str, Any], on: Dict[Any, Any], path):
        """The walk's step for one node: switches recorded, guarded commands
        wrapped; None for every node the guard does not concern."""
        # 1. Something that switches a device on or off (an on/off command, or
        #    a guard already there): kept as it is, and remembered.
        found = _on_off(graph, node)
        if found is not None:
            artifact, state = found
            return node, {**on, artifact: ("==", state)}
        if node.get("type") != "action" or not node.get("action_url"):
            return None

        # 2. An action: a guarded command, on a device the entry applies to,
        #    not switched on earlier in the plan.
        url = node["action_url"]
        artifact = artifact_of_url(graph, url)
        if artifact is None or satisfies(on.get(artifact), "==", True) \
                or not applies(graph, artifact, applies_to):
            return node, on
        affordance = _affordance(graph, artifact, url)
        if affordance is None or not any((affordance, RDF.type, c) in graph
                                         for c in GUARDED_COMMANDS):
            return node, on

        # 3. Wrap it -- the device is on after the guard -- or say why not.
        found_affordances, why = lookup(artifact)
        if found_affordances is None:
            notes.append(f"{goal or 'plan'}: {_title(graph, artifact)}."
                         f"{_title(graph, affordance)} not guarded by power on first: {why}")
            return node, on
        label = goal or _title(graph, artifact) or "device"
        return _guarded(label, node, *found_affordances), {**on, artifact: ("==", True)}

    guarded, _ = walk_in_order(copy.deepcopy(tree), leaf, writes)
    return guarded, notes
