"""Specific capability contexts: just the part of the environment one goal is about.

The general BT planner sees the whole home. A structured goal usually pins down
far more than that -- an explicit goal names its device -- so it can be planned
against a context cut to what it names, by a much smaller model.

This module builds that context from the InteractionSolver's TD graph (closed
under subclass inference when fetched, see `environment_graph`), in the
affordance-dict format `AsyncBTPlanner` already reads
(`bt_planning/planning/prompts.py`): the planner's prompt, short affordance ids,
parameter validation and retries are reused as they are.

    build_specific_context(graph, goal)        the entry point: dispatches on the
                                               goal's kind and specificity
      -> _explicit_achievement(graph, goal)    one helper per goal kind
      -> _incomplete_achievement(graph, goal)
           -> find_artifacts(...)              goal-independent primitives:
           -> artifact_affordances(...)        graph -> planner affordance dicts
      -> SpecificCapabilityContext(affordances, artifacts, goal_brief)

For "set the bathroom air conditioner 1 to cooling mode", with no value
determined, the context is every typed action and readable property of
`bathroom_air_conditioner_1` -- and nothing else in the home -- plus a brief
saying the goal is its `hvacMode`, with the words "cooling mode".

An incomplete goal that names no action ("make the hallway light blue") gets
every device its class and room allow, with all their affordances, and a brief
warning that the request may be impossible with them.

An ambiguous goal ("the bathroom is so damp") names environment variables, not
devices. Its context starts from each variable's room and hangs everything off
the variable (`_ambiguous_achievement`), grouped by variable for the planner.

The other helpers the dispatch table is meant to grow (not implemented):

    maintenance              the actuator, plus the properties that observe the
                             maintained variable
    predicates               the properties their pools name
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from ....shared.models.goal_structure import GoalSpec
from ....shared.utils.vocabulary import vocabulary
from ....shared.utils.namespaces import (
    HCTL,
    HMAS,
    HOMEONT,
    JS,
    QUDT,
    SAREF,
    SOSA,
    SSN,
    TD,
    TDSOSA,
    action_types,
    expand,
    in_namespace,
    shorten,
)

# A direct-value input is sent wrapped: SHTD reads `{"value": ...}` (see
# SimuHome/shtd.py `invoke_action`; verified end-to-end). The deterministic
# planner sends it this way, and the LLM is shown the input as an object with
# this one field so it does the same.
DIRECT_VALUE_KEY = "value"


@dataclass
class SpecificCapabilityContext:
    """What a goal is planned against.

    `affordances` are planner affordance dicts; `artifacts` the titles of the
    devices they belong to; `goal_brief` the structured goal as text for the
    planner's user message.
    """

    affordances: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    goal_brief: str = ""
    # For an ambiguous goal: the affordances grouped by the environment variable
    # they affect, in the planner's "Observable Property Effects" format
    # (`bt_planning/planning/prompts.py format_observable_property_hints`).
    observable_property_hints: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------
# Primitives: finding devices
# --------------------------------------------------------------------------

_ARTIFACT_QUERY = """
PREFIX td:      <https://www.w3.org/2019/wot/td#>
PREFIX hmas:    <https://purl.org/hmas/>
PREFIX homeont: <http://example.org/homeont/>
SELECT DISTINCT ?artifact WHERE {
  ?artifact a hmas:Artifact .
  %(name_block)s
  %(class_block)s
  %(location_block)s
}
"""


def find_artifacts(graph: Graph, name: Optional[str] = None,
                   cls: Optional[str] = None,
                   location_cls: Optional[str] = None) -> List[URIRef]:
    """The devices matching whichever of title, class and room class are given.

    Each filter is pasted into the query only when given, and is then a
    required pattern. Classes match at any level up to their root, because the
    graph was closed under subclass inference when fetched. Values are passed
    as query bindings, never pasted into the query text.
    """
    bindings: Dict[str, Any] = {}
    blocks = {"name_block": "", "class_block": "", "location_block": ""}
    if name:
        blocks["name_block"] = "?artifact td:title ?name ."
        bindings["name"] = Literal(name)
    if cls:
        blocks["class_block"] = "?artifact a ?cls ."
        bindings["cls"] = URIRef(expand(cls))
    if location_cls:
        blocks["location_block"] = ("?workspace hmas:contains ?artifact ; "
                                    "homeont:hasSpace ?space . ?space a ?locationCls .")
        bindings["locationCls"] = URIRef(expand(location_cls))
    rows = graph.query(_ARTIFACT_QUERY % blocks, initBindings=bindings)
    return sorted({row.artifact for row in rows}, key=str)


# --------------------------------------------------------------------------
# Primitives: affordances as the planner reads them
# --------------------------------------------------------------------------

def _title(graph: Graph, node: Any) -> Optional[str]:
    value = graph.value(node, TD.title)
    return str(value) if value is not None else None


def _most_specific(classes) -> List[URIRef]:
    """The classes no other class in the set specialises.

    The graph is closed under subclass inference, so a node carries its
    ancestors as types too. Shown to the model, those are noise -- and picking
    "the" class of a node by sort order would return a root. Only the most
    specific types are kept.
    """
    vocab = vocabulary()
    classes = {c for c in classes if isinstance(c, URIRef)}
    ancestors = set()
    for cls in classes:
        ancestors |= set(vocab.transitive_objects(cls, RDFS.subClassOf)) - {cls}
    return sorted(classes - ancestors, key=str)


def _homeont_types(graph: Graph, node: Any) -> List[str]:
    """The node's most specific homeont types, as CURIEs."""
    return [shorten(t) for t in _most_specific(graph.objects(node, RDF.type))
            if in_namespace(t, HOMEONT)]


def device_class(graph: Graph, artifact: Any) -> Optional[str]:
    """The device's most specific homeont class, as a CURIE."""
    types = _homeont_types(graph, artifact)
    return types[0] if types else None


def device_location(graph: Graph, artifact: Any) -> Optional[str]:
    """The most specific homeont class of the space the device's workspace
    stands for (homeont:Kitchen), as a CURIE."""
    for workspace in sorted(graph.subjects(HMAS.contains, artifact), key=str):
        for space in sorted(graph.objects(workspace, HOMEONT.hasSpace), key=str):
            types = _homeont_types(graph, space)
            if types:
                return types[0]
    return None


def _command_types(graph: Graph, action: Any) -> List[str]:
    """The action's most specific command classes (homeont or SAREF)."""
    return action_types(_most_specific(graph.objects(action, RDF.type)))


def _form(graph: Graph, affordance: Any) -> Tuple[str, Optional[str]]:
    """The affordance's target URL and HTTP method, from its first form."""
    for form in graph.objects(affordance, TD.hasForm):
        target = graph.value(form, HCTL.hasTarget)
        method = graph.value(form, URIRef("http://www.w3.org/2011/http#methodName"))
        if target is not None:
            return str(target), (str(method) if method is not None else None)
    return "", None


_JSON_TYPES = {
    "js:IntegerSchema": "integer", "js:NumberSchema": "number",
    "js:BooleanSchema": "boolean", "js:StringSchema": "string",
    "js:ArraySchema": "array", "js:ObjectSchema": "object",
}


def json_schema(graph: Graph, schema: Any) -> Dict[str, Any]:
    """A TD data schema as a JSON-schema dict: type, enum, bounds, unit,
    description; an object schema with its named properties."""
    types = [shorten(t) for t in graph.objects(schema, RDF.type)]
    kind = next((_JSON_TYPES[t] for t in types if t in _JSON_TYPES), None)
    out: Dict[str, Any] = {"type": kind} if kind else {}

    if kind == "object":
        properties: Dict[str, Any] = {}
        for member in graph.objects(schema, JS.properties):
            name = graph.value(member, JS.propertyName)
            if name is not None:
                properties[str(name)] = json_schema(graph, member)
        out["properties"] = dict(sorted(properties.items()))
        required = sorted(str(r) for r in graph.objects(schema, JS.required))
        if required:
            out["required"] = required
        return out

    enum = sorted((v.toPython() for v in graph.objects(schema, JS.enum)),
                  key=lambda v: (str(type(v)), v))
    if enum:
        out["enum"] = enum
    for key, predicate in (("minimum", JS.minimum), ("maximum", JS.maximum)):
        value = graph.value(schema, predicate)
        if value is not None:
            out[key] = value.toPython()
    unit = graph.value(schema, QUDT.unit)
    if unit is not None:
        out["unit"] = shorten(unit)
    description = graph.value(schema, JS.description)
    if description is not None:
        out["description"] = str(description)
    return out


def _input_schema(graph: Graph, schema: Any) -> Optional[Dict[str, Any]]:
    """An action's input as the planner should fill it.

    The planner sends `parameters` as a JSON object, so a direct-value input is
    presented as an object with one field, DIRECT_VALUE_KEY -- the body the
    device server accepts. An object input is shown as it is.
    """
    if schema is None:
        return None
    converted = json_schema(graph, schema)
    if converted.get("type") == "object":
        return converted
    return {"type": "object", "properties": {DIRECT_VALUE_KEY: converted},
            "required": [DIRECT_VALUE_KEY]}


def _action_description(graph: Graph, action: Any) -> str:
    """What the action changes: the property it acts upon, and the room
    variables it moves -- the facts that let the planner pick it."""
    parts = []
    for prop in graph.objects(action, SAREF.actsUpon):
        classes = _homeont_types(graph, prop)
        parts.append(f"acts upon \"{_title(graph, prop)}\""
                     + (f" ({classes[0]})" if classes else ""))
    for actuation in graph.objects(action, TDSOSA.hasEffectActuation):
        for var in graph.objects(actuation, SOSA.actsOnProperty):
            environment = graph.value(var, SSN.isPropertyOf)
            label = graph.value(environment, RDFS.label) if environment is not None else None
            parts.append(f"affects \"{_title(graph, var)}\""
                         + (f" of \"{label}\"" if label is not None else ""))
    return "; ".join(parts)


def action_affordance(graph: Graph, artifact: Any, action: Any) -> Dict[str, Any]:
    """One action of a device, as a planner affordance dict."""
    target, method = _form(graph, action)
    aff: Dict[str, Any] = {
        "name": _title(graph, action),
        "type": "action_affordance",
        "target": target,
        "form": {"href": target, "method": method or "POST"},
        "artifact_id": str(artifact),
        "artifact_name": _title(graph, artifact),
        "semantic_types": _command_types(graph, action),
    }
    description = _action_description(graph, action)
    if description:
        aff["description"] = description
    input_schema = _input_schema(graph, graph.value(action, TD.hasInputSchema))
    if input_schema:
        aff["input_schema"] = input_schema
    return aff


def property_affordance(graph: Graph, artifact: Any, prop: Any) -> Dict[str, Any]:
    """One readable property of a device, as a planner affordance dict."""
    target, method = _form(graph, prop)
    aff: Dict[str, Any] = {
        "name": _title(graph, prop),
        "type": "property_affordance",
        "target": target,
        "form": {"href": target, "method": method or "GET"},
        "artifact_id": str(artifact),
        "artifact_name": _title(graph, artifact),
        "semantic_types": _homeont_types(graph, prop),
    }
    output = graph.value(prop, TD.hasOutputSchema)
    if output is not None:
        aff["output_schema"] = json_schema(graph, output)
        # The planner's prompt prints a description but not an output schema;
        # the value's shape is what a read-back condition is written against.
        aff["description"] = f"reads {json.dumps(aff['output_schema'])}"
    return aff


def artifact_affordances(graph: Graph, artifact: Any) -> List[Dict[str, Any]]:
    """Every typed action and readable property of one device.

    Protocol actions (WebSub, CRUD) carry no homeont/SAREF command class and
    are left out, as are properties with no homeont class.
    """
    actions = [a for a in graph.objects(artifact, TD.hasActionAffordance)
               if action_types(graph.objects(a, RDF.type))]
    props = [p for p in graph.objects(artifact, TD.hasPropertyAffordance)
             if _homeont_types(graph, p)]

    def by_title(node: Any) -> str:
        return _title(graph, node) or ""

    return ([action_affordance(graph, artifact, a) for a in sorted(actions, key=by_title)]
            + [property_affordance(graph, artifact, p) for p in sorted(props, key=by_title)])


# --------------------------------------------------------------------------
# The goal, as text for the planner
# --------------------------------------------------------------------------

# Told to the model when the goal names no action: the devices it sees were
# chosen by class and room alone, and may not be able to do what is asked.
IMPOSSIBLE_WARNING = ("- no action was identified for this request: choose from the "
                      "devices' affordances. It may be impossible with these devices; "
                      "if none of their actions can achieve it, answer that it is "
                      "impossible rather than approximate it.")

_QUANTIFIER_WORDS = {"all": "every one of them", "one": "one of them is enough"}


def goal_brief(goal: GoalSpec) -> str:
    """The structured goal in a few lines: which device, which action, and the
    user's words for the value -- so the model maps the words to a value
    against the action's schema, rather than re-reading the sentence."""
    # The device: named, or -- for an incomplete goal -- described by its class
    # and room, with how many of the matching devices the request means.
    if goal.artifact_name:
        lines = [f"- device: \"{goal.artifact_name}\""
                 + (f" ({goal.artifact_class})" if goal.artifact_class else "")]
    else:
        lines = ["- devices: " + (goal.artifact_class or "any device")
                 + (f" in a {goal.location_class}" if goal.location_class else "")
                 + (f" -- {_QUANTIFIER_WORDS[goal.quantifier]}"
                    if goal.quantifier in _QUANTIFIER_WORDS else "")]
    if goal.affordance_name:
        lines.append(f"- action: \"{goal.affordance_name}\""
                     + (f" ({goal.affordance_class})" if goal.affordance_class else ""))
    elif not goal.affordance_class:
        lines.append(IMPOSSIBLE_WARNING)
    if goal.parameter_name and goal.parameter_name != "NA":
        lines.append(f"- parameter: \"{goal.parameter_name}\"")
    if goal.target_value_text:
        lines.append(f"- value, in the user's words: \"{goal.target_value_text}\"")
    if goal.goal_effect:
        lines.append(f"- effect: {goal.goal_effect}")
    # A modify goal's change, as the structurer read it: signed, and either a
    # plain amount or a percentage.
    if goal.goal_effect == "modify" and goal.target_value_determined is not None:
        kind = "a percentage" if goal.target_value_is_percentage else "a plain amount"
        lines.append(f"- change: {goal.target_value_determined} ({kind})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Goal-kind helpers and the dispatch
# --------------------------------------------------------------------------

def _explicit_achievement(graph: Graph, goal: GoalSpec) -> Optional[SpecificCapabilityContext]:
    """The one device an explicit goal names, with all of its affordances.

    None unless exactly one device matches the goal's title, class and room:
    an explicit goal that names no device, or names one ambiguously, has no
    specific context.
    """
    if not goal.artifact_name:
        return None
    devices = find_artifacts(graph, goal.artifact_name, goal.artifact_class,
                             goal.location_class)
    if len(devices) != 1:
        return None
    (device,) = devices
    return SpecificCapabilityContext(
        affordances=artifact_affordances(graph, device),
        artifacts=[_title(graph, device) or str(device)],
        goal_brief=goal_brief(goal),
    )


def _incomplete_achievement(graph: Graph, goal: GoalSpec) -> Optional[SpecificCapabilityContext]:
    """Every device an incomplete goal's class and room allow, with all of
    their affordances -- the context for a goal that names no action.

    None when no device fits: there is nothing to plan against.
    """
    devices = find_artifacts(graph, goal.artifact_name, goal.artifact_class,
                             goal.location_class)
    if not devices:
        return None
    affordances: List[Dict[str, Any]] = []
    for device in devices:
        affordances += artifact_affordances(graph, device)
    return SpecificCapabilityContext(
        affordances=affordances,
        artifacts=[_title(graph, d) or str(d) for d in devices],
        goal_brief=goal_brief(goal),
    )


# --------------------------------------------------------------------------
# Ambiguous goals: everything hangs off the room's environment variable
# --------------------------------------------------------------------------

# The variable, starting from the hmas workspace: the workspace's space is of the
# goal's room class, its environment has the variable, the variable is of the
# goal's class. (Closed graph: subclasses count, as plain `a`.)
_VARIABLE_QUERY = """
PREFIX homeont: <http://example.org/homeont/>
PREFIX ssn:     <http://www.w3.org/ns/ssn/>
SELECT DISTINCT ?workspace ?var WHERE {
  ?workspace homeont:hasSpace ?space .
  ?space a ?spaceClass ;
         homeont:hasEnvironment ?env .
  ?env ssn:hasProperty ?var .
  ?var a ?varClass .
}
"""

# The actions acting on that variable, of artifacts the workspace contains. Any
# effect actuation counts: most only say they influence it, not which way.
_AFFECTING_QUERY = """
PREFIX hmas:   <https://purl.org/hmas/>
PREFIX td:     <https://www.w3.org/2019/wot/td#>
PREFIX sosa:   <http://www.w3.org/ns/sosa/>
PREFIX tdsosa: <https://example.org/hmas/td-sosa-ext#>
SELECT DISTINCT ?artifact ?action ?actuation WHERE {
  ?workspace hmas:contains ?artifact .
  ?artifact td:hasActionAffordance ?action .
  ?action tdsosa:hasEffectActuation ?actuation .
  ?actuation sosa:actsOnProperty ?var .
}
"""

# The readings of that variable, of artifacts the workspace contains: property
# affordances of the variable's class (a device-specific subclass counts).
_READING_QUERY = """
PREFIX hmas: <https://purl.org/hmas/>
PREFIX td:   <https://www.w3.org/2019/wot/td#>
SELECT DISTINCT ?artifact ?prop WHERE {
  ?workspace hmas:contains ?artifact .
  ?artifact td:hasPropertyAffordance ?prop .
  ?prop a ?varClass .
}
"""

# Told to the planner with every ambiguous goal: the direction is its call.
DIRECTION_INSTRUCTIONS = (
    "For each variable above, decide from the request whether the user wants it to "
    "increase or decrease (\"so damp\" -> humidity decrease; \"too dark\" -> "
    "illuminance increase). Then choose actions, and their values, that move it that "
    "way. Most actions only say they influence the variable: decide from the action "
    "and its input which value moves it the wanted way (turn a dehumidifier on; lower "
    "a cooling setpoint). Read the current value first when the value to set depends "
    "on it. If nothing can move a variable the wanted way, answer impossible.")


def _direction(graph: Graph, actuation: Any) -> str:
    """The direction the TD states for an effect, or "affects" when it states none."""
    if (actuation, RDF.type, TDSOSA.IncreasingActuation) in graph:
        return "increase"
    if (actuation, RDF.type, TDSOSA.DecreasingActuation) in graph:
        return "decrease"
    return "affects"


def _room_variables(graph: Graph, var: Any) -> List[Tuple[Any, Any]]:
    """(workspace, variable) for one implied variable: its room, then the
    room's variable of its class. A `space_name` matching a workspace title
    narrows rooms of one class (two bedrooms); otherwise every room of the
    class counts."""
    if not var.property_class or not var.space_class:
        return []
    rows = graph.query(_VARIABLE_QUERY, initBindings={
        "spaceClass": URIRef(expand(var.space_class)),
        "varClass": URIRef(expand(var.property_class))})
    found = sorted({(row.workspace, row.var) for row in rows}, key=str)
    if var.space_name:
        named = [(ws, v) for ws, v in found
                 if (_title(graph, ws) or "").lower() == var.space_name.lower()]
        found = named or found
    return found


def ambiguous_brief(goal: GoalSpec) -> str:
    """The ambiguous goal for the planner: the variables, their rooms, and the
    instructions to decide the direction."""
    lines = ["- environment variables the request is about:"]
    for var in goal.implied_environment_vars:
        room = var.space_name or var.space_class or "?"
        lines.append(f"  - \"{var.property_name}\" ({var.property_class}) in {room}")
    lines.append(f"- {DIRECTION_INSTRUCTIONS}")
    return "\n".join(lines)


def _ambiguous_achievement(graph: Graph, goal: GoalSpec) -> Optional[SpecificCapabilityContext]:
    """The actions that act on each implied variable of the goal's rooms, the
    properties they act upon, and the readings of the variable -- grouped by
    variable.

    None when no implied variable has an action acting on it: there is nothing
    to plan with.
    """
    affordances: Dict[str, Dict[str, Any]] = {}       # by target URL, deduped
    artifacts: List[str] = []
    results: List[Dict[str, Any]] = []

    def add(affordance: Dict[str, Any], artifact: Any) -> None:
        affordances.setdefault(affordance["target"], affordance)
        title = _title(graph, artifact) or str(artifact)
        if title not in artifacts:
            artifacts.append(title)

    for implied in goal.implied_environment_vars:
        for workspace, var in _room_variables(graph, implied):
            room = _title(graph, workspace) or str(workspace)
            bindings = {"workspace": workspace, "var": var}

            # 1. The actions acting on this room's variable, with their stated
            #    direction, and the properties they act upon (current values).
            actions = []
            for row in graph.query(_AFFECTING_QUERY, initBindings=bindings):
                action = action_affordance(graph, row.artifact, row.action)
                add(action, row.artifact)
                actions.append({"artifact_title": _title(graph, row.artifact),
                                "action_name": action["name"],
                                "action_target": action["target"],
                                "direction": _direction(graph, row.actuation)})
                for prop in graph.objects(row.action, SAREF.actsUpon):
                    add(property_affordance(graph, row.artifact, prop), row.artifact)
            if not actions:
                continue

            # 2. The readings of the variable in that room, to check the effect.
            readings = []
            for row in graph.query(_READING_QUERY, initBindings={
                    "workspace": workspace,
                    "varClass": URIRef(expand(implied.property_class))}):
                reading = property_affordance(graph, row.artifact, row.prop)
                add(reading, row.artifact)
                readings.append(reading["target"])

            results.append({
                "property_uri": (f"\"{_title(graph, var) or implied.property_name}\" of "
                                 f"{room} ({implied.property_class})"),
                "actions": sorted(actions, key=lambda a: (a["artifact_title"] or "",
                                                          a["action_name"] or "")),
                "readable_property_urls": sorted(set(readings)),
            })

    if not results:
        return None
    return SpecificCapabilityContext(
        affordances=list(affordances.values()),
        artifacts=artifacts,
        goal_brief=ambiguous_brief(goal),
        observable_property_hints={"results": results},
    )


# (goal_kind, goal_specificity) -> helper. Grows as other goal kinds are given
# specific contexts (see the module docstring).
_HELPERS: Dict[Tuple[str, str], Callable[[Graph, GoalSpec],
                                         Optional[SpecificCapabilityContext]]] = {
    ("achievement", "explicit"): _explicit_achievement,
    ("achievement", "incomplete"): _incomplete_achievement,
    ("achievement", "ambiguous"): _ambiguous_achievement,
}


def build_specific_context(graph: Graph, goal: GoalSpec) -> Optional[SpecificCapabilityContext]:
    """The specific capability context for one goal, or None when its kind has
    no helper yet or the helper finds nothing to narrow to."""
    helper = _HELPERS.get((goal.goal_kind, goal.goal_specificity))
    return helper(graph, goal) if helper else None
