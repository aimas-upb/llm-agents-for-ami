"""Deterministic repairs of a planner's behaviour tree before it is executed.

    goal_condition_first   `sequence[condition on the goal's variable, steps...]`:
                           the planner put the goal's own test first (humidity
                           above 38%, then "raise the humidity"), so the sequence
                           fails at once whenever the goal is not yet met and
                           nothing runs. The tree becomes
                           `selector[condition, sequence[steps...]]`: the goal
                           already met is left alone, otherwise the steps run.
                           Only a condition on one of the goal's implied
                           environment variables (its property class or name) is
                           taken as the goal's own test; a condition on a device
                           (is the fan on?) is a step and stays where it is.

`repair_tree(tree, goal)` returns (tree, notes); the tree is a new dict when a
repair applied, else the one given.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ....shared.models.goal_structure import GoalSpec


def _variable_names(goal: GoalSpec) -> set:
    names = set()
    for var in goal.implied_environment_vars or []:
        for value in (var.property_class, var.property_name):
            if value:
                names.add(str(value).lower())
                names.add(str(value).split(":")[-1].lower())
    return names


def _on_goal_variable(node: Dict[str, Any], names: set) -> bool:
    if node.get("type") != "condition" or not names:
        return False
    for key in ("affordance_class", "affordance", "property", "property_class"):
        value = node.get(key)
        if value and str(value).lower() in names:
            return True
    url = str(node.get("property_url") or "").lower()
    return bool(url) and any(f"/properties/{n}" in url for n in names)


def goal_condition_first(tree: Dict[str, Any], goal: GoalSpec) -> Optional[Dict[str, Any]]:
    """The rewritten tree, or None when the pattern is not there."""
    if not isinstance(tree, dict) or tree.get("type") != "sequence":
        return None
    children = tree.get("children") or []
    if len(children) < 2 or not _on_goal_variable(children[0], _variable_names(goal)):
        return None
    condition, steps = children[0], children[1:]
    body = steps[0] if len(steps) == 1 and steps[0].get("type") == "sequence" else {
        "type": "sequence", "name": f"{tree.get('name') or 'goal'}: steps",
        "memory": tree.get("memory", True), "children": steps}
    return {"type": "selector", "name": tree.get("name") or "goal", "memory": False,
            "children": [condition, body]}


def repair_tree(tree: Dict[str, Any], goal: GoalSpec) -> Tuple[Dict[str, Any], List[str]]:
    """Every tree repair, with a note per repair applied."""
    notes: List[str] = []
    rewritten = goal_condition_first(tree, goal)
    if rewritten is not None:
        name = (tree.get("children") or [{}])[0].get("name") or "the goal's condition"
        notes.append(f"the plan tested \"{name}\" before acting; it now acts unless that already holds")
        tree = rewritten
    return tree, notes
