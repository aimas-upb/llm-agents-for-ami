"""The plan's execution order: what is certainly true at each node of a tree.

Both guards ask the same question of a plan: before this action runs, has the
plan itself already put the device in the state the guard would establish?
The power-on guard asks it of the on/off state (`power_on_guard.py`), the
condition guard of any property (`condition_guard.py`). This module answers it
once, for both, by walking the BT IR in execution order and carrying *facts*:
what holds, for certain, when a node starts.

A fact maps a key (whatever the caller tracks -- a property URL, a device) to
`(op, value)`: `("==", 2)` "holds 2", `("!=", 1)` "holds anything but 1".

How each node carries the facts (`walk_in_order`):

    a node the caller handles   the caller's `leaf` decides (an action, a guard
                                already in the tree, a condition)
    sequence                    children in order: each starts from what the
                                previous one left; the sequence ends with what
                                the last one left
    any other node with         each child starts from the facts before the
    children                    node; afterwards the facts before it, minus
    (selector, parallel,        every key a child writes -- a selector's branch
    inverter, ignore_failure,   may not have run, parallel children run in no
    await, ...)                 order, a decorated child may have stopped
                                partway
    repeat                      the same, and its body also starts without the
                                keys it writes: from the second pass on they
                                hold the previous pass's values
    anything else (timer,       no change: the plan is trusted, nothing outside
    wait_until, read, ...)      it is assumed to change a device while it waits

Example -- "pause washer 1, wait 10 minutes, then set its spin to 1200":

    sequence                                      facts {}
      action operationalState {"value": 2}        -> {opState: ("==", 2)}
      timer 600                                   -> unchanged
      action spinSpeedCurrent {"value": 1200}     starts with {opState: ("==", 2)}:
                                                  a guard "opState != 1" is
                                                  already satisfied

Trusting the plan through waits is a decision: a cycle finishing on its own, a
button press or another plan running in parallel during the wait is not seen.

    walk_in_order(tree, leaf, writes, facts)
                                        (the tree, as the leaves rewrote it;
                                         the facts after it)
    satisfies(fact, op, value)          whether a fact makes a check hold
    same_value(a, b)                    loose equality of two device values
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Set, Tuple

# A fact: (op, value), op "==" or "!=".
Fact = Tuple[str, Any]
Facts = Dict[Any, Fact]

# leaf(node, facts, path) -> (node, facts after it), or None for "not mine:
# carry the facts through it by its type".
Leaf = Callable[[Dict[str, Any], Facts, Tuple[int, ...]], Optional[Tuple[Any, Facts]]]
# writes(node) -> the keys this one node writes, or None for "not mine: look
# at its children".
Writes = Callable[[Dict[str, Any]], Optional[Set[Any]]]

# The one composite whose children run one after the other, each after the
# previous one succeeded.
ORDERED = "sequence"
# Its body runs again and again.
LOOP = "repeat"


# -- values ------------------------------------------------------------------

def same_value(a: Any, b: Any) -> bool:
    """True when two device values are the same: numbers by value (1 == 1.0),
    booleans as booleans, anything else by its text, case ignored."""
    # 1. Booleans first: True must not equal the number 1 by accident of
    #    Python, only another boolean (or its text).
    if isinstance(a, bool) or isinstance(b, bool):
        return str(a).strip().lower() == str(b).strip().lower()
    # 2. Numbers, by value.
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        pass
    # 3. Text.
    return str(a).strip().lower() == str(b).strip().lower()


def satisfies(fact: Optional[Fact], op: str, value: Any) -> bool:
    """True when ``fact`` makes the check "``op`` ``value``" hold for certain.

    "== v" decides every comparison; "!= v" only decides "!= v" itself (it
    says nothing about which other value is there)."""
    if fact is None:
        return False
    fact_op, fact_value = fact
    # 1. A known value: compare it.
    if fact_op == "==":
        if op == "==":
            return same_value(fact_value, value)
        if op == "!=":
            return not same_value(fact_value, value)
        try:
            a, b = float(fact_value), float(value)
        except (TypeError, ValueError):
            return False
        return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}.get(op, False)
    # 2. A known "not": only the same "not" follows from it.
    return fact_op == "!=" and op == "!=" and same_value(fact_value, value)


# -- the walk ----------------------------------------------------------------

def _written(node: Any, writes: Writes) -> Set[Any]:
    """Every key a subtree writes: the caller's answer for the nodes it knows,
    the union of the children's for the others."""
    if not isinstance(node, dict):
        return set()
    own = writes(node)
    if own is not None:
        return set(own)
    out: Set[Any] = set()
    for child in node.get("children") or []:
        out |= _written(child, writes)
    return out


def _without(facts: Facts, keys: Set[Any]) -> Facts:
    """The facts, minus those about ``keys``."""
    return {k: v for k, v in facts.items() if k not in keys}


def walk_in_order(tree: Any, leaf: Leaf, writes: Writes,
                  facts: Optional[Facts] = None) -> Tuple[Any, Facts]:
    """(``tree`` as ``leaf`` rewrote its nodes, the facts after it).

    ``leaf`` is offered every node first and may rewrite it (wrap an action in
    a guard, unwrap a guard); ``writes`` names the keys a node writes, for the
    nodes after which those keys are no longer certain. See the module
    docstring for the rules. ``facts`` is what holds before the tree (none, by
    default; a guard walks its inner node from its own check). The tree is not
    modified; rewritten composites are new dicts."""

    def visit(node: Any, facts: Facts, path: Tuple[int, ...]) -> Tuple[Any, Facts]:
        """(the node, rewritten; the facts after it), given the facts before."""
        if not isinstance(node, dict):
            return node, facts

        # 1. The caller's nodes: it decides what they are and what they leave.
        handled = leaf(node, facts, path)
        if handled is not None:
            return handled

        children = node.get("children")
        if not children:
            # 2. A leaf the caller does not track: changes nothing.
            return node, facts

        # 3. A sequence: each child after the previous one.
        if node.get("type") == ORDERED:
            out, after = [], facts
            for index, child in enumerate(children):
                child_out, after = visit(child, after, path + (index,))
                out.append(child_out)
            return {**node, "children": out}, after

        # 4. Anything else with children: each child from the facts before the
        #    node (a loop's body also without what it writes); afterwards,
        #    nothing a child writes is certain.
        written = set().union(*(_written(c, writes) for c in children))
        start = _without(facts, written) if node.get("type") == LOOP else facts
        out = [visit(child, start, path + (index,))[0]
               for index, child in enumerate(children)]
        return {**node, "children": out}, _without(facts, written)

    return visit(tree, dict(facts or {}), ())
