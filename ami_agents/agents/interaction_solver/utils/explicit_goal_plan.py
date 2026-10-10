"""Deterministic plans for explicit, achievement goals.

A goal the UA structured as explicit names its room, device and action by
class and title, and -- when the action takes an input -- states the value the
action's schema makes of the user's words (`target_value_determined`). That is
enough to plan without an LLM:

    one action in the TD graph matches   -> a one-node behaviour tree
    none matches                         -> impossible, with the reason
    several match                        -> impossible: the names should have
                                            singled one out, so something is
                                            inconsistent and is reported, not guessed
    it takes an input, no value given    -> not deterministic: back to the
                                            planning workflow

A `modify` goal needs the current value read first: it is planned by
`modify_goal_plan.py`, as read -> compute -> set. A `set` goal flagged
`target_value_is_percentage` is mapped onto the input's range here first.

The flow, for "set the bathroom air conditioner 1 to cooling mode":

    deterministic_goal(structure)   does this goal qualify?           -> G1
    find_actions(graph, G1)         which action in the TD graph is   -> the
                                    "hvacMode" of the                    hvacMode
                                    AirConditioner titled                affordance,
                                    bathroom_air_conditioner_1, a        its URL and
                                    SetModeCommand, in a Bathroom?       input schema
    action_payload(graph, G1, ...)  what body does that action take   -> {"value": 3}
                                    for target_value_determined "3"?
    plan_explicit_goal(graph, G1)   all of the above, as a plan entry -> {"tree": {...}}

Pure functions over an rdflib graph; the behaviour that fetches the graph and
replies is `behaviours/explicit_goal_planning.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from ....shared.models.goal_structure import NA, GoalSpec, GoalStructure
from ....shared.utils.namespaces import JS, expand, shorten
from .specific_capability_context import DIRECT_VALUE_KEY, json_schema

# A direct-value input is sent wrapped as {DIRECT_VALUE_KEY: value}; a
# parameterized call sends its parameter by name. DIRECT_VALUE_KEY is defined in
# `specific_capability_context`, so the deterministic plans here and the
# LLM-planned ones follow one convention.

# Marks plan entries produced here, so a log or a summary can tell them from
# the LLM planner's.
SOURCE = "explicit_deterministic"


# --------------------------------------------------------------------------
# Which goals qualify
# --------------------------------------------------------------------------

def deterministic_goal(structure: GoalStructure) -> Optional[GoalSpec]:
    """The goal, if this structure qualifies for a deterministic plan.

    Qualifies: a simple structure holding exactly one goal that is explicit,
    an achievement, a `set` or a `modify`, and names its device and action.
    Anything else -- dependencies, maintenance, incomplete or ambiguous goals --
    returns None and is left to the planning workflow (incomplete ones have
    their own path, `incomplete_goal_plan.py`).

    A determined value is not required here: many actions take no input at all
    (Home Assistant's `switch/turn_on`, `cover/open_cover` -- the action itself
    is the value). Whether one is needed is only known once the action is
    found, so `plan_explicit_goal` decides it.
    """
    if structure.structure != "simple" or structure.predicates or len(structure.goals) != 1:
        return None
    (goal,) = structure.goals.values()
    qualifies = (goal.goal_specificity == "explicit"
                 and goal.goal_kind == "achievement"
                 and goal.goal_effect in ("set", "modify")
                 and goal.artifact_name and goal.affordance_name)
    return goal if qualifies else None


# --------------------------------------------------------------------------
# Finding the action in the TD graph
# --------------------------------------------------------------------------

# The core pattern: the artifact with this title, its action with this title,
# and the URL the action is invoked at (its form's `hctl:hasTarget`). The input
# schema is optional -- an action may take no input at all. A title the goal
# does not give is left unbound, so an incomplete goal finds every device (or
# action) its classes and room allow.
#
# The three class blocks below are pasted into the query text only when the
# goal states the class they test, so a goal is never filtered on a slot it
# left empty. Once pasted they are ordinary required patterns, not SPARQL
# OPTIONALs: a class that does not fit removes the row. (Only the input schema
# is OPTIONAL -- an action may take no input.)
_QUERY = """
PREFIX td:      <https://www.w3.org/2019/wot/td#>
PREFIX hctl:    <https://www.w3.org/2019/wot/hypermedia#>
PREFIX hmas:    <https://purl.org/hmas/>
PREFIX homeont: <http://example.org/homeont/>
SELECT DISTINCT ?artifact ?artifactTitle ?aff ?target ?schema WHERE {
  ?artifact td:title ?artifactTitle ;
            td:hasActionAffordance ?aff .
  ?aff td:title ?affordanceTitle ;
       td:hasForm ?form .
  ?form hctl:hasTarget ?target .
  OPTIONAL { ?aff td:hasInputSchema ?schema }
  %(artifact_block)s
  %(class_block)s
  %(location_block)s
}
"""

# The device is of the goal's kind: a homeont:DimmableLight is not a
# homeont:ColorLight, whatever either is titled.
#
# All three class tests are a plain `a`, and still accept a class more generic
# than the one the TD asserts (saref:Device, saref:OnCommand,
# homeont:BuildingSpace): the graph has been closed under subclass inference up
# to the class roots when it was fetched (`environment_graph`), so the
# ancestors are there as triples. No property paths, so no duplicate rows.
_ARTIFACT_BLOCK = "?artifact a ?artifactClass ."

# The action does what the goal says: it performs the goal's command class
# (homeont:SetModeCommand), or -- after the type closure -- a specialisation of
# it. A consistency check: the titles already identify the action, so this only
# rejects a goal whose class was copied from a different action than its title.
_CLASS_BLOCK = "?aff a ?affordanceClass ."

# The device stands in the right kind of room: its workspace's space is of the
# goal's location class.
_LOCATION_BLOCK = ("?workspace hmas:contains ?artifact ; homeont:hasSpace ?space . "
                   "?space a ?locationClass .")


def find_actions(graph: Graph, goal: GoalSpec) -> List[Dict[str, Any]]:
    """The action affordances matching the goal's names and classes.

    Returns one dict per match: the artifact node and title, the affordance
    node, its target URL, and its input schema node (None when the action takes
    no input). Titles are unique per artifact, so for an explicit goal a
    consistent graph yields at most one; an incomplete goal, missing a title,
    yields one per candidate.

    The goal's names and classes are passed as query bindings, never pasted
    into the query text, so a title containing a quote cannot break the query.
    """
    bindings: Dict[str, Any] = {}
    if goal.artifact_name:
        bindings["artifactTitle"] = Literal(goal.artifact_name)
    if goal.affordance_name:
        bindings["affordanceTitle"] = Literal(goal.affordance_name)
    artifact_block = class_block = location_block = ""
    if goal.artifact_class:
        artifact_block = _ARTIFACT_BLOCK
        bindings["artifactClass"] = URIRef(expand(goal.artifact_class))
    if goal.affordance_class:
        class_block = _CLASS_BLOCK
        bindings["affordanceClass"] = URIRef(expand(goal.affordance_class))
    if goal.location_class:
        location_block = _LOCATION_BLOCK
        bindings["locationClass"] = URIRef(expand(goal.location_class))
    query = _QUERY % {"artifact_block": artifact_block, "class_block": class_block,
                      "location_block": location_block}
    rows = graph.query(query, initBindings=bindings)
    return [{"artifact": row.artifact, "artifact_title": str(row.artifactTitle),
             "affordance": row.aff, "target": str(row.target), "schema": row.schema}
            for row in rows]


# --------------------------------------------------------------------------
# Turning the determined value into the action's request body
# --------------------------------------------------------------------------

class SchemaMismatch(ValueError):
    """The determined value does not fit the action's input schema."""


class NoPercentageScale(ValueError):
    """A percentage for an input that is neither in percent nor bounded: nothing
    says which value it is."""


def _from_percentage(graph: Graph, schema: Any, text: str) -> str:
    """A "set" percentage ("40% brightness" -> "40") as the input's own value.

    An input in unit:PERCENT takes it as it is. Any other input takes the point
    that far along its range: 40% of [1, 254] is 1 + 0.40 * 253 = 102.2, rounded
    for an integer input. Without a range there is no such point.
    """
    facts = json_schema(graph, schema)
    if facts.get("unit") == "unit:PERCENT":
        return text
    low, high = facts.get("minimum"), facts.get("maximum")
    if low is None or high is None:
        raise NoPercentageScale("the input states no percent unit and no range")
    try:
        value = low + float(text) / 100.0 * (high - low)
    except ValueError:
        raise SchemaMismatch(f"{text!r} is not a percentage")
    return str(int(round(value))) if facts.get("type") == "integer" else str(value)


def _schema_type(graph: Graph, schema: Any) -> Optional[str]:
    """The JSON Schema type of a schema node, e.g. "js:IntegerSchema"."""
    types = sorted(shorten(t) for t in graph.objects(schema, RDF.type)
                   if str(t).startswith(str(JS)))
    return types[0] if types else None


def _typed(graph: Graph, schema: Any, text: str) -> Any:
    """`text` as the JSON value the schema asks for, checked against its enum
    and bounds.

    The determined value arrives as a string ("3", "true", "-16.0"); the action
    needs a JSON value of the schema's type (3, True, -16.0). A value that
    cannot be converted, is not one of the schema's enum members, or lies
    outside its minimum/maximum raises SchemaMismatch -- better an impossible
    plan with a reason than a request the device rejects.
    """
    # Convert the string to the schema's type. A schema with no recognised
    # type (or a string schema) keeps the text as it is.
    kind = _schema_type(graph, schema)
    try:
        if kind == "js:IntegerSchema":
            number = float(text)
            if not number.is_integer():
                raise SchemaMismatch(f"{text!r} is not an integer")
            value: Any = int(number)
        elif kind == "js:NumberSchema":
            value = float(text)
        elif kind == "js:BooleanSchema":
            lowered = text.strip().lower()
            if lowered not in ("true", "false"):
                raise SchemaMismatch(f"{text!r} is not a boolean")
            value = lowered == "true"
        else:
            value = text
    except SchemaMismatch:
        raise
    except ValueError as exc:   # float() on something that is not a number
        raise SchemaMismatch(f"{text!r} is not a {kind}") from exc

    # An enum restricts the value to its listed members ("3 = Cool" is one of
    # hvacMode's 0..9 minus 2).
    enum = [v.toPython() for v in graph.objects(schema, JS.enum)]
    if enum and value not in enum:
        raise SchemaMismatch(f"{value!r} is not one of the permitted values {sorted(enum)}")

    # Numeric bounds, where the schema states them.
    for bound, ok in ((JS.minimum, lambda b: value >= b), (JS.maximum, lambda b: value <= b)):
        limit = graph.value(schema, bound)
        if limit is not None and isinstance(value, (int, float)) and not ok(limit.toPython()):
            raise SchemaMismatch(f"{value!r} is outside the permitted range")
    return value


def action_payload(graph: Graph, goal: GoalSpec, schema: Any) -> Dict[str, Any]:
    """The request body for the action, from the goal's determined value.

    Three shapes, decided by the action's input schema:

        no input schema        {}                          (nothing to send)
        a direct value         {"value": <typed value>}    (DIRECT_VALUE_KEY)
        an object schema       {<parameter_name>: <typed value>}

    The goal's `parameter_name` must agree with the shape: "NA" (or null) for a
    direct value, the name of one of the object's properties otherwise. A
    disagreement means the structurer read the action wrong, and raises
    SchemaMismatch rather than sending a body the device would reject.
    """
    if schema is None:
        return {}
    text = goal.target_value_determined
    is_object = (schema, RDF.type, JS.ObjectSchema) in graph

    # A direct value: the schema itself types the value.
    if not is_object:
        if goal.parameter_name not in (None, NA):
            raise SchemaMismatch(f"the action takes a direct value, not parameter "
                                 f"{goal.parameter_name!r}")
        if goal.target_value_is_percentage:
            text = _from_percentage(graph, schema, text)
        return {DIRECT_VALUE_KEY: _typed(graph, schema, text)}

    # A parameterized call: the named property's own schema types the value.
    if goal.parameter_name in (None, NA):
        raise SchemaMismatch("the action takes named parameters, and the goal names none")
    for member in graph.objects(schema, JS.properties):
        if str(graph.value(member, JS.propertyName)) == goal.parameter_name:
            if goal.target_value_is_percentage:
                text = _from_percentage(graph, member, text)
            return {goal.parameter_name: _typed(graph, member, text)}
    raise SchemaMismatch(f"the action has no parameter {goal.parameter_name!r}")


# --------------------------------------------------------------------------
# The plan entry
# --------------------------------------------------------------------------

def plan_explicit_goal(graph: Graph, goal: GoalSpec) -> Optional[Dict[str, Any]]:
    """A plan entry for one explicit goal: a one-action tree, impossible -- or
    None when the goal cannot be planned deterministically after all.

    The entry has the shape every plan entry in a multi-plan envelope has
    (`tree`, `explanation`, and `impossible` when there is no tree), so the UA
    summarises and executes it like any other.

    None means: the action was found, it takes an input, and the goal does not
    determine the value to send. That is not "impossible" -- the value may
    still be worked out from the user's words -- so the caller hands the goal
    to the planning workflow instead.
    """
    where = f"{goal.artifact_name}.{goal.affordance_name}"

    # 1. Find the action. Zero or several matches end here, with the reason.
    actions = find_actions(graph, goal)
    distinct = {a["affordance"] for a in actions}
    if not distinct:
        return _impossible(
            f"No action {goal.affordance_name!r} on {goal.artifact_name!r}"
            + (f" (a {goal.artifact_class})" if goal.artifact_class else "")
            + (f" in a {goal.location_class}" if goal.location_class else "")
            + (f" that performs {goal.affordance_class}" if goal.affordance_class else "")
            + ".")
    if len(distinct) > 1:
        return _impossible(f"{len(distinct)} actions match {where}; the goal's names "
                           f"should single one out.")

    # 2. Build its request body. An action without an input is invoked with an
    #    empty body -- the action itself carries the meaning ("turn on"). An
    #    action with an input needs the determined value: without one this is
    #    not deterministic (None, see above); with one the schema does not
    #    accept, it is impossible.
    action = actions[0]
    if action["schema"] is not None and goal.target_value_determined is None:
        return None
    try:
        payload = action_payload(graph, goal, action["schema"])
    except NoPercentageScale:
        # Not impossible: the small model may still read the scale off the
        # device's description.
        return None
    except SchemaMismatch as exc:
        return _impossible(f"Cannot invoke {where} with "
                           f"{goal.target_value_determined!r}: {exc}.")

    # 3. The tree: a single action node, as the executor's `action` node type
    #    expects it (`bt_planning/nodes/registry.py`).
    node: Dict[str, Any] = {"type": "action", "name": where,
                            "action_url": action["target"]}
    if payload:
        node["parameters"] = payload
    words = f" (\"{goal.target_value_text}\")" if goal.target_value_text else ""
    return {
        "tree": node,
        "explanation": f"Invoke {where} with {payload or 'no input'}{words}.",
        "source": SOURCE,
    }


def _impossible(reason: str) -> Dict[str, Any]:
    """A tree-less plan entry: the UA does not execute it, and the reason
    reaches the user through the plan summary."""
    return {"tree": None, "impossible": True, "explanation": reason, "source": SOURCE}
