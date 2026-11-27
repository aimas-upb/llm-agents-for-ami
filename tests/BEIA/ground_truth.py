"""Scoring a plan against a BEIA case's ground truth -- statically, without running it.

A plan can be right in several ways: one reads the Netatmo and another the OPTIX,
one nests the steps differently. So the ground truth is not a tree to diff; it is
a list of FACTS the plan must have, each checked on its own:

    reads      the plan reads one of these properties, compared like this
               ("garage temperature > 26": any of three sensors, op >, value 26)
    acts       every step of at least one alternative appears, with these
               parameters (in order, when the alternative says so)
    alerts     the plan raises an alert saying something like this
    forbidden  no action of the plan matches any of these
    shape      the plan recurs ("whenever"), re-arms after acting, waits for a
               condition to HOLD for a span

The flow, for one plan (the UA's `plan_json`: {"plans": [entry, ...]}):

    extract_facts(plan)
      each entry's tree is walked: action nodes give (ref, params) in tree
      order; condition nodes and blackboard checks give (ref, op, value) reads
      -- a check's key is mapped back to its property through the entry's
      runtime `reads` and the tree's `read` nodes; alert nodes give texts;
      the shape flags come from the composites (`repeat`, a `parallel` racing
      a `timer`, ...)
    score(plan, ground_truth) -> one row
      one column per fact (`reads_ok`, `acts_ok`, ...), `missing` saying what
      is not there, and `static_pass` when every fact holds

A "ref" is the HASP name of an affordance: `workspace/artifact/affordance`
(`garaj/lg_garaj_hvac/climate_set_hvac_mode`), whatever host the URL had.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -- names -----------------------------------------------------------------------------

REF_RE = re.compile(r"/workspaces/([^/]+)/artifacts/([^/#?]+)/(?:actions|properties)/([^/?#]+)")

# Every spelling of a comparison, to one name. The dependency compiler writes
# eq/ne/gt/ge/lt/le; condition nodes and ground truths write == != > >= < <=.
OPS = {"==": "eq", "!=": "ne", ">": "gt", ">=": "ge", "<": "lt", "<=": "le",
       "eq": "eq", "ne": "ne", "gt": "gt", "ge": "ge", "lt": "lt", "le": "le"}
# "Above 26" may be written > 26 or >= 26: a threshold's side is what matters.
SIDE = {"eq": "eq", "ne": "ne", "gt": "above", "ge": "above", "lt": "below", "le": "below"}
# A check that waits for its negation ("wait_for not (> 25)") reads <= 25.
WAIT_TYPES = {"wait_for", "await", "wait_condition", "check", "condition"}


def short_ref(url: Any) -> str:
    """`workspace/artifact/affordance` for a HASP URL; the text itself otherwise."""
    match = REF_RE.search(str(url or ""))
    return "/".join(match.groups()) if match else str(url or "")


def side(op: Any) -> Optional[str]:
    """Which side of a threshold a comparison asks for (above / below / eq / ne)."""
    return SIDE.get(OPS.get(str(op or "").strip(), ""))


# -- what a plan says --------------------------------------------------------------------

@dataclass
class Read:
    """The plan compares a property with a value."""

    ref: str
    op: str
    value: Any


@dataclass
class Action:
    """The plan invokes an affordance with parameters; `order` is tree order."""

    ref: str
    params: Dict[str, Any]
    order: int


@dataclass
class PlanFacts:
    """Everything the scorer reads off one plan."""

    reads: List[Read] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    recurring: bool = False
    rearm: bool = False
    hold: bool = False


def plan_entries(plan: Any) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """(tree, runtime) per plan entry, whatever form the plan comes in: the UA's
    canonical JSON string, {"plans": [...]}, one entry, or a bare tree."""
    # 1. Text: the UA's canonical JSON.
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except ValueError:
            return []
    if not isinstance(plan, dict):
        return []
    # 2. Several entries, one entry, or a tree.
    entries = plan.get("plans") if isinstance(plan.get("plans"), list) else [plan]
    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tree = entry.get("tree") if isinstance(entry.get("tree"), dict) else entry
        if isinstance(tree.get("tree"), dict):        # {"tree": {"tree": ...}} envelopes
            tree = tree["tree"]
        out.append((tree, entry.get("runtime") or {}))
    return out


def _children(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A node's child nodes (decorators keep their one child in `children` too)."""
    return [c for c in node.get("children") or [] if isinstance(c, dict)]


def _walk(node: Dict[str, Any], parent: Optional[Dict[str, Any]] = None):
    """Every node with its parent, depth first, in tree (execution) order."""
    yield node, parent
    for child in _children(node):
        yield from _walk(child, node)


def _contains(node: Dict[str, Any], types: Iterable[str]) -> bool:
    """Whether any node at or below `node` is of one of these types."""
    wanted = set(types)
    return any(n.get("type") in wanted for n, _ in _walk(node))


def _is_retry(node: Dict[str, Any], parent: Optional[Dict[str, Any]]) -> bool:
    """The middle `repeat` of a held block: inverter(repeat(inverter(...)))."""
    children = _children(node)
    return (node.get("type") == "repeat" and parent is not None
            and parent.get("type") == "inverter"
            and len(children) == 1 and children[0].get("type") == "inverter")


def _first_sequence(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The first sequence at or below a node, through one-child decorators."""
    while node is not None:
        if node.get("type") == "sequence":
            return node
        children = _children(node)
        node = children[0] if len(children) == 1 else None
    return None


def _rearms(loop: Dict[str, Any]) -> bool:
    """After its last action or alert, the loop's sequence waits for something
    again -- so one long excursion acts once, not once per pass."""
    sequence = _first_sequence(_children(loop)[0]) if _children(loop) else None
    if sequence is None:
        return False
    steps = _children(sequence)
    acting = [i for i, step in enumerate(steps) if _contains(step, ("action", "alert"))]
    if not acting:
        return False
    return any(_contains(step, WAIT_TYPES) for step in steps[acting[-1] + 1:])


def extract_facts(plan: Any) -> PlanFacts:
    """Reads, actions, alerts and shape flags of every entry of a plan."""
    facts = PlanFacts()
    order = 0
    for tree, runtime in plan_entries(plan):
        # 1. Which property each blackboard key holds: the runtime's reads, and
        #    the tree's own read nodes.
        key_refs = {str(r.get("key")): short_ref(r.get("url"))
                    for r in runtime.get("reads") or [] if isinstance(r, dict)}
        key_refs.update({str(n.get("output")): short_ref(n.get("property_url"))
                         for n, _ in _walk(tree) if n.get("type") == "read"})
        # 2. Each node, in execution order.
        for node, parent in _walk(tree):
            kind = node.get("type")
            if kind == "action":
                facts.actions.append(Action(short_ref(node.get("action_url")),
                                            dict(node.get("parameters") or {}), order))
                order += 1
            elif kind in ("condition", "wait_condition"):
                facts.reads.append(Read(short_ref(node.get("property_url")),
                                        OPS.get(str(node.get("operator") or "=="), "eq"),
                                        node.get("expected_value")))
            elif kind in ("wait_for", "check"):
                for check in node.get("checks") or []:
                    ref = key_refs.get(str(check.get("key")))
                    if ref:
                        facts.reads.append(Read(ref, OPS.get(str(check.get("op")), ""),
                                                check.get("value")))
            elif kind == "alert":
                facts.alerts.append(str(node.get("text") or ""))
            elif kind == "parallel":
                # 3. A timer raced against an inverted wait: "held for a span".
                types = [c.get("type") for c in _children(node)]
                if "timer" in types and "inverter" in types:
                    facts.hold = True
            elif kind == "repeat" and not _is_retry(node, parent):
                # 4. A loop around acting: "whenever"; and whether it re-arms.
                if _contains(node, ("action", "alert")):
                    facts.recurring = True
                    facts.rearm = facts.rearm or _rearms(node)
    return facts


# -- matching ----------------------------------------------------------------------------

def _number(value: Any) -> Optional[float]:
    """A value as a number, or None (a boolean is no number)."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def same_value(actual: Any, expected: Any, tolerance: float = 0.0) -> bool:
    """Whether a plan's value is the ground truth's: one of a list, inside a
    {"min", "max"} range, a number within `tolerance`, or the same text
    (case-insensitive; true / "true" alike)."""
    # 1. Alternatives and ranges.
    if isinstance(expected, list):
        return any(same_value(actual, e, tolerance) for e in expected)
    if isinstance(expected, dict) and ({"min", "max"} & set(expected)):
        number = _number(actual)
        return (number is not None
                and number >= float(expected.get("min", float("-inf")))
                and number <= float(expected.get("max", float("inf"))))
    # 2. Numbers, then text.
    a, e = _number(actual), _number(expected)
    if a is not None and e is not None:
        return abs(a - e) <= tolerance
    return str(actual).strip().lower() == str(expected).strip().lower()


def find_param(params: Any, name: str) -> List[Any]:
    """Every value under `name`, at any depth of an action's parameters (a
    planner may nest them, for example {"data": {"hvac_mode": "cool"}})."""
    found: List[Any] = []
    if isinstance(params, dict):
        for key, value in params.items():
            if key == name:
                found.append(value)
            found.extend(find_param(value, name))
    elif isinstance(params, list):
        for item in params:
            found.extend(find_param(item, name))
    return found


def _step_matches(action: Action, step: Dict[str, Any]) -> bool:
    """One plan action against one ground-truth step: its ref, and each param."""
    if action.ref != step.get("ref"):
        return False
    for name, expected in (step.get("params") or {}).items():
        if not any(same_value(v, expected) for v in find_param(action.params, name)):
            return False
    return True


def _alternative_matches(actions: List[Action], alternative: Dict[str, Any]) -> bool:
    """Every step of the alternative is in the plan -- in order when it says so."""
    steps = alternative.get("steps") or []
    if not alternative.get("ordered"):
        return all(any(_step_matches(a, s) for a in actions) for s in steps)
    # Ordered: a subsequence of the plan's actions, in tree order.
    position = -1
    for step in steps:
        later = [a for a in actions if a.order > position and _step_matches(a, step)]
        if not later:
            return False
        position = min(a.order for a in later)
    return True


def _read_matches(read: Read, wanted: Dict[str, Any]) -> bool:
    """One plan read against one ground-truth read: property, side, value."""
    return (read.ref in (wanted.get("any_of") or [])
            and side(read.op) == side(wanted.get("op"))
            and same_value(read.value, wanted.get("value"),
                           float(wanted.get("tolerance", 0.0))))


def _forbidden_hits(actions: List[Action], forbidden: List[Dict[str, Any]]) -> List[str]:
    """The plan's actions that match a forbidden entry: a ref (with params), a
    ref prefix (a whole room), or any action at all."""
    hits = []
    for action in actions:
        for rule in forbidden:
            if rule.get("any_action"):
                hit = True
            elif "ref_prefix" in rule:
                hit = action.ref.startswith(str(rule["ref_prefix"]))
            else:
                hit = _step_matches(action, rule)
            if hit:
                hits.append(f"{action.ref} {json.dumps(action.params, sort_keys=True)}")
                break
    return hits


# -- the row -----------------------------------------------------------------------------

def score(plan: Any, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """One row: a column per fact, what is missing, and `static_pass`.

    A ground truth may be several whole solutions (`{"any_of": [gt, gt]}`):
    keeping a humidity band by a rule that reads the sensor, or by the
    device's own setpoint -- each with its own reads, acts and shape. The plan
    passes when it is one of them; the row is that solution's (or, failing
    all, the first's), with `alternative` saying which.
    """
    # 1. Several solutions: the first that passes, else the first.
    if ground_truth.get("any_of"):
        rows = [score(plan, gt) for gt in ground_truth["any_of"]]
        best = next((i for i, r in enumerate(rows) if r["static_pass"]), 0)
        return {**rows[best], "alternative": best}
    return _score_one(plan, ground_truth)


def _score_one(plan: Any, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """`score` for one solution."""
    facts = extract_facts(plan)
    missing: List[str] = []

    # 1. Reads: each ground-truth read is met by some read of the plan.
    reads_ok = True
    for wanted in ground_truth.get("reads") or []:
        options = wanted.get("alternatives") or [wanted]
        if not any(_read_matches(r, w) for w in options for r in facts.reads):
            reads_ok = False
            missing.append(f"read {json.dumps(wanted, sort_keys=True)}")

    # 2. Acts: one alternative fully present (no acts given: nothing required).
    acts = ground_truth.get("acts") or {}
    alternatives = acts.get("any_of") or []
    acts_ok = (not alternatives) or any(_alternative_matches(facts.actions, alt)
                                        for alt in alternatives)
    if not acts_ok:
        missing.append("no alternative of `acts` is complete")

    # 3. Alerts: each wanted alert raised, saying something like it.
    alerts_ok = True
    for wanted in ground_truth.get("alerts") or []:
        words = [w.lower() for w in wanted.get("contains_any") or []]
        if not any(not words or any(w in text.lower() for w in words)
                   for text in facts.alerts):
            alerts_ok = False
            missing.append(f"alert {json.dumps(wanted, sort_keys=True)}")

    # 4. Forbidden actions.
    hits = _forbidden_hits(facts.actions, ground_truth.get("forbidden") or [])
    missing.extend(f"forbidden: {h}" for h in hits)

    # 5. Shape: each feature the ground truth asks for.
    shape = ground_truth.get("shape") or {}
    shape_ok = {name: (not shape.get(name)) or getattr(facts, name)
                for name in ("recurring", "rearm", "hold")}
    missing.extend(f"shape: no {name}" for name, ok in shape_ok.items() if not ok)

    return {
        "reads_ok": reads_ok,
        "acts_ok": acts_ok,
        "alerts_ok": alerts_ok,
        "forbidden_hits": hits,
        "recurring_ok": shape_ok["recurring"],
        "rearm_ok": shape_ok["rearm"],
        "hold_ok": shape_ok["hold"],
        "static_pass": (reads_ok and acts_ok and alerts_ok and not hits
                        and all(shape_ok.values())),
        "missing": missing,
    }
