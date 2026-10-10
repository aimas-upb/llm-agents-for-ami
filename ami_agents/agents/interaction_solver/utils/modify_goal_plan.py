"""Deterministic plans for "modify" goals: read the value, change it, set it.

A modify goal states a relative change ("dim the living room light by 20%",
"lower the cooling setpoint by 2 degrees") as a signed amount in
`target_value_determined`, flagged `target_value_is_percentage` when it is a
percentage. The device cannot apply a change, only a value, so the plan reads
the current value, computes the new one and sets it:

    sequence
      read     the property the action acts upon     -> blackboard .../current
      compute  change_clamped(current)               -> blackboard .../new
      action   the goal's action, value from .../new

What the change means is decided here, from the TD graph, when the plan is
built:

    not a percentage                         add:   new = current + amount
    a percentage, property in unit:PERCENT   add    (percentage points)
    a percentage, any other unit             scale: new = current + amount% of current

and the result is kept within the property's range (else the action input's):
below the minimum it is the minimum, above the maximum the maximum.

The flow, for one device:

    modify_recipe(graph, goal)     the action, the property it acts upon, the
                                   mode and range -> a ModifyRecipe, or why not
    modify_tree(recipe)            the three-node tree
    plan_modify_goal(graph, goal)  both, as a plan entry -- or None, which hands
                                   the goal to the small model for a recipe
    recipe_from_answer(...)        the small model's recipe, checked against the
                                   TDs before it is built into the same tree

Pure functions over the InteractionSolver's closed TD graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from rdflib import Graph

from ....shared.models.goal_structure import NA, GoalSpec
from ....shared.utils.namespaces import HCTL, QUDT, SAREF, TD, shorten
from .explicit_goal_plan import find_actions
from .specific_capability_context import DIRECT_VALUE_KEY, find_artifacts, json_schema

SOURCE = "modify_deterministic"
PERCENT_UNIT = "unit:PERCENT"


@dataclass
class ModifyRecipe:
    """Everything the read-compute-set tree needs, for one device's action."""

    device: str                 # the artifact's title
    action: str                 # the action's title
    action_url: str
    parameter: str              # where the new value goes in the request body
    property_url: str           # where the current value is read
    amount: float               # the signed change
    mode: str                   # "add" or "scale"
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    integer: bool = False


# --------------------------------------------------------------------------
# Reading the TD graph
# --------------------------------------------------------------------------

def _form_url(graph: Graph, affordance: Any) -> Optional[str]:
    """The URL an affordance is used at: its form's target (read, or invoke)."""
    for form in graph.objects(affordance, TD.hasForm):
        target = graph.value(form, HCTL.hasTarget)
        if target is not None:
            return str(target)
    return None


def _property_unit(graph: Graph, prop: Any) -> Optional[str]:
    """The property's unit: on the affordance itself, else on its output schema."""
    unit = graph.value(prop, QUDT.unit)
    if unit is None:
        schema = graph.value(prop, TD.hasOutputSchema)
        unit = graph.value(schema, QUDT.unit) if schema is not None else None
    return shorten(unit) if unit is not None else None


def _bounds(schema: Dict[str, Any]) -> tuple:
    return schema.get("minimum"), schema.get("maximum")


def _amount(goal: GoalSpec) -> Optional[float]:
    """The signed change as a number, or None when it does not parse."""
    try:
        return float(goal.target_value_determined)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# The recipe
# --------------------------------------------------------------------------

def modify_recipe(graph: Graph, goal: GoalSpec) -> Union[ModifyRecipe, str]:
    """The recipe for one explicit modify goal, or the reason it cannot be
    built deterministically."""
    # 1. The amount must be a number: "-20", "2.5".
    amount = _amount(goal)
    if amount is None:
        return f"the change {goal.target_value_determined!r} is not a number"

    # 2. The action: the goal names it, so exactly one must match.
    actions = find_actions(graph, goal)
    if len({a["affordance"] for a in actions}) != 1:
        return f"{len(actions)} actions match {goal.artifact_name}.{goal.affordance_name}"
    action = actions[0]
    input_schema = json_schema(graph, action["schema"]) if action["schema"] is not None else {}

    # 3. Where the new value goes: the whole body for a direct value, else the
    #    goal's parameter.
    if input_schema.get("type") == "object":
        if goal.parameter_name in (None, NA) or goal.parameter_name not in input_schema.get(
                "properties", {}):
            return "the action takes named parameters and the goal names none of them"
        parameter = goal.parameter_name
        value_schema = input_schema["properties"][parameter]
    elif input_schema:
        parameter, value_schema = DIRECT_VALUE_KEY, input_schema
    else:
        return "the action takes no input"
    if value_schema.get("type") not in ("integer", "number"):
        return "the action's input is not a number"

    # 4. The property the action changes, read to know the current value.
    props = list(graph.objects(action["affordance"], SAREF.actsUpon))
    if len(props) != 1:
        return f"the action acts upon {len(props)} properties (saref:actsUpon)"
    (prop,) = props
    property_url = _form_url(graph, prop)
    if property_url is None:
        return "the property it acts upon cannot be read"

    # 5. Add or scale: a percentage scales, unless the property is itself a
    #    percentage -- then it is percentage points, added.
    mode = "add"
    if goal.target_value_is_percentage and _property_unit(graph, prop) != PERCENT_UNIT:
        mode = "scale"

    # 6. The range: the property's own, else the action input's.
    output = graph.value(prop, TD.hasOutputSchema)
    minimum, maximum = _bounds(json_schema(graph, output)) if output is not None else (None, None)
    if minimum is None and maximum is None:
        minimum, maximum = _bounds(value_schema)

    return ModifyRecipe(
        device=goal.artifact_name, action=goal.affordance_name,
        action_url=action["target"], parameter=parameter, property_url=property_url,
        amount=amount, mode=mode, minimum=minimum, maximum=maximum,
        integer=value_schema.get("type") == "integer")


# --------------------------------------------------------------------------
# The tree and the plan entry
# --------------------------------------------------------------------------

def modify_tree(recipe: ModifyRecipe) -> Dict[str, Any]:
    """read -> compute -> action, with blackboard keys of this device's action
    so that several devices' trees never share one. (Keys are "/"-separated: a
    "." in a py_trees key means attribute access.)"""
    where = f"{recipe.device}.{recipe.action}"
    # Titles may contain dots (Home Assistant's "light.kitchen"): replaced.
    base = "modify/" + "/".join(t.replace(".", "_") for t in (recipe.device, recipe.action))
    current, new = f"{base}/current", f"{base}/new"
    args: Dict[str, Any] = {"amount": recipe.amount, "mode": recipe.mode,
                            "integer": recipe.integer}
    if recipe.minimum is not None:
        args["min"] = recipe.minimum
    if recipe.maximum is not None:
        args["max"] = recipe.maximum
    return {"type": "sequence", "name": f"{where}: change", "children": [
        {"type": "read", "name": f"{where}: read", "property_url": recipe.property_url,
         "output": current},
        {"type": "compute", "name": f"{where}: compute", "op": "change_clamped",
         "inputs": [current], "output": new, "args": args},
        {"type": "action", "name": where, "action_url": recipe.action_url,
         "parameter_keys": {recipe.parameter: new}},
    ]}


def describe(recipe: ModifyRecipe) -> str:
    """The plan in words, for its explanation."""
    change = (f"{recipe.amount:+g}% of the current value" if recipe.mode == "scale"
              else f"{recipe.amount:+g}")
    bounds = ""
    if recipe.minimum is not None or recipe.maximum is not None:
        bounds = f", kept within [{recipe.minimum}, {recipe.maximum}]"
    return f"Read {recipe.device}.{recipe.action}'s value, change it by {change}{bounds}, set it."


def plan_modify_goal(graph: Graph, goal: GoalSpec) -> Optional[Dict[str, Any]]:
    """A plan entry for one explicit modify goal, or None when no deterministic
    recipe can be built -- the caller then asks the small model for one."""
    recipe = modify_recipe(graph, goal)
    if isinstance(recipe, str):
        return None
    return {"tree": modify_tree(recipe), "explanation": describe(recipe), "source": SOURCE}


# --------------------------------------------------------------------------
# The small model's recipe
# --------------------------------------------------------------------------

def _by_title(graph: Graph, artifact: Any, predicate: Any, title: Any) -> Optional[Any]:
    """The device's affordance (action or property) with this title."""
    for affordance in graph.objects(artifact, predicate):
        if str(graph.value(affordance, TD.title)) == str(title):
            return affordance
    return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def recipe_from_answer(graph: Graph, goal: GoalSpec,
                       answer: Dict[str, Any]) -> Union[ModifyRecipe, str]:
    """The model's answer as a recipe, every name checked against the device's
    TD -- or why it cannot be used. The URLs come from the graph, never from
    the model."""
    # 1. The device the goal names.
    devices = find_artifacts(graph, goal.artifact_name, goal.artifact_class,
                             goal.location_class)
    if len(devices) != 1:
        return f"no single device {goal.artifact_name!r}"
    (device,) = devices

    # 2. The action and the property, by the names the model gave.
    action = _by_title(graph, device, TD.hasActionAffordance, answer.get("action"))
    if action is None:
        return f"the device has no action {answer.get('action')!r}"
    prop = _by_title(graph, device, TD.hasPropertyAffordance, answer.get("property"))
    if prop is None:
        return f"the device has no property {answer.get('property')!r}"
    action_url, property_url = _form_url(graph, action), _form_url(graph, prop)
    if not action_url or not property_url:
        return "the action or the property has no URL"

    # 3. Where the value goes, and its type, from the action's input schema.
    schema = graph.value(action, TD.hasInputSchema)
    facts = json_schema(graph, schema) if schema is not None else {}
    parameter = answer.get("parameter") or DIRECT_VALUE_KEY
    if facts.get("type") == "object":
        value_schema = facts.get("properties", {}).get(parameter)
        if value_schema is None:
            return f"the action has no parameter {parameter!r}"
    elif facts:
        parameter, value_schema = DIRECT_VALUE_KEY, facts
    else:
        return "the action takes no input"
    if value_schema.get("type") not in ("integer", "number"):
        return "the action's input is not a number"

    # 4. The numbers.
    amount = _number(answer.get("amount"))
    if amount is None:
        return f"the amount {answer.get('amount')!r} is not a number"
    mode = answer.get("mode")
    if mode not in ("add", "scale"):
        return f"the mode {mode!r} is neither add nor scale"
    return ModifyRecipe(
        device=goal.artifact_name, action=str(answer["action"]), action_url=action_url,
        parameter=parameter, property_url=property_url, amount=amount, mode=mode,
        minimum=_number(answer.get("min")), maximum=_number(answer.get("max")),
        integer=value_schema.get("type") == "integer")
