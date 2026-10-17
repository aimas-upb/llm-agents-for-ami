"""What a dependency structure is about, per goal and per predicate.

The dependency planner plans each goal's action first (the achievement
planners); those plans name the devices each goal acts on. Predicates have no
planner, so they are resolved here, deterministically:

    device predicate         the devices its names allow (find_artifacts), and
                             on each the property it reads -- by title, else by
                             class
    environment predicate    the readings of its variable in its room(s)
    ambiguous predicate      not resolvable: a qualitative condition ("while
                             it is stuffy") has no value to compare against

    resolve_predicate(graph, pred)      -> PredicateTarget | error
    profile_context(graph, artifacts)   -> {"G1": {"artifacts": [...]}, "P1": ...}
    property_info(graph, art, prop)     -> where the plan reads a PropertyRef

`profile_context` is the `specific_goal_context` of the profiling prompt: per
id, the same affordance dicts the achievement planners get
(`specific_capability_context`), grouped per artifact with its title, class
and room class -- what the model copies a PropertyRef from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from ....shared.models.goal_structure import GoalStructure, PredicateSpec
from ....shared.utils.namespaces import HCTL, HOMEONT, QUDT, SSN, TD, expand, shorten
from .specific_capability_context import (
    _READING_QUERY,
    _room_variables,
    artifact_affordances,
    device_class,
    device_location,
    find_artifacts,
    json_schema,
)

# Seconds per unit, for the units a time property may report in.
SECONDS_PER_UNIT = {"unit:SEC": 1.0, "unit:MilliSEC": 0.001, "unit:MIN": 60.0,
                    "unit:HR": 3600.0, "unit:DAY": 86400.0}


@dataclass
class PropertyInfo:
    """Where the plan reads one property, and what it reads."""

    artifact: str          # td:title
    prop: str              # td:title
    url: str
    schema: Dict[str, Any] = field(default_factory=dict)
    unit: Optional[str] = None

    @property
    def seconds_per_unit(self) -> float:
        return SECONDS_PER_UNIT.get(self.unit or "", 1.0)


@dataclass
class PredicateTarget:
    """The properties a predicate reads: one per device or reading it covers."""

    properties: List[PropertyInfo] = field(default_factory=list)
    artifacts: List[URIRef] = field(default_factory=list)


# --------------------------------------------------------------------------
# Reading the TD graph
# --------------------------------------------------------------------------

def _title(graph: Graph, node: Any) -> Optional[str]:
    value = graph.value(node, TD.title)
    return str(value) if value is not None else None


def _form_url(graph: Graph, affordance: Any) -> Optional[str]:
    for form in graph.objects(affordance, TD.hasForm):
        target = graph.value(form, HCTL.hasTarget)
        if target is not None:
            return str(target)
    return None


def _info(graph: Graph, artifact: Any, prop: Any) -> Optional[PropertyInfo]:
    url = _form_url(graph, prop)
    if not url:
        return None
    output = graph.value(prop, TD.hasOutputSchema)
    schema = json_schema(graph, output) if output is not None else {}
    unit = graph.value(prop, QUDT.unit)
    return PropertyInfo(artifact=_title(graph, artifact) or str(artifact),
                        prop=_title(graph, prop) or str(prop), url=url, schema=schema,
                        unit=shorten(unit) if unit is not None else schema.get("unit"))


def _artifact_by_title(graph: Graph, title: str) -> Optional[URIRef]:
    for artifact in graph.subjects(TD.title, Literal(title)):
        if graph.value(artifact, TD.hasPropertyAffordance) is not None \
                or graph.value(artifact, TD.hasActionAffordance) is not None:
            return artifact
    return None


def property_info(graph: Graph, artifact_name: str,
                  property_name: str) -> Optional[PropertyInfo]:
    """The property `property_name` of the device titled `artifact_name`."""
    artifact = _artifact_by_title(graph, artifact_name)
    if artifact is None:
        return None
    for prop in graph.objects(artifact, TD.hasPropertyAffordance):
        if _title(graph, prop) == property_name:
            return _info(graph, artifact, prop)
    return None


def _property_on(graph: Graph, artifact: Any, name: Optional[str],
                 cls: Optional[str]) -> Optional[PropertyInfo]:
    """The device's property: by title, else the one of the class."""
    props = list(graph.objects(artifact, TD.hasPropertyAffordance))
    if name:
        for prop in props:
            if _title(graph, prop) == name:
                return _info(graph, artifact, prop)
    if cls:
        wanted = URIRef(expand(cls))
        for prop in props:
            if (prop, RDF.type, wanted) in graph:
                return _info(graph, artifact, prop)
    return None


def artifact_of_url(graph: Graph, url: str) -> Optional[URIRef]:
    """The device an action or property URL belongs to."""
    for form in graph.subjects(HCTL.hasTarget, URIRef(url)):
        for affordance in graph.subjects(TD.hasForm, form):
            for predicate in (TD.hasActionAffordance, TD.hasPropertyAffordance):
                for artifact in graph.subjects(predicate, affordance):
                    return artifact
    return None


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------

def resolve_predicate(graph: Graph, pred: PredicateSpec) -> Union[PredicateTarget, str]:
    """The properties a predicate reads, or why it cannot be read."""
    if pred.predicate_specificity == "ambiguous":
        return ("a qualitative condition (" + str(pred.qualitative_condition or pred.predicate_text)
                + ") has no value to compare against")
    if pred.target_value_determined is None and pred.role != "filter":
        return f"no determined value for {pred.predicate_text!r}"

    # A device's own property.
    if pred.predicate_subject == "device_property":
        artifacts = find_artifacts(graph, pred.artifact_name, pred.artifact_class,
                                   pred.location_class)
        target = PredicateTarget(artifacts=artifacts)
        for artifact in artifacts:
            info = _property_on(graph, artifact, pred.property_name, pred.property_class)
            if info is not None:
                target.properties.append(info)
        if pred.role == "filter" and not (pred.artifact_name or pred.artifact_class
                                          or pred.location_class):
            return target          # read on the goal's own devices, at compile time
        if not target.properties:
            return f"no device has the property read by {pred.predicate_text!r}"
        return target

    # A room's environment variable, through the readings that observe it.
    var = pred.environment_var
    if var is None or not var.property_class:
        return f"no environment variable named by {pred.predicate_text!r}"
    target = PredicateTarget()
    for workspace, _ in _room_variables(graph, var) or _all_rooms(graph, var.property_class):
        for row in graph.query(_READING_QUERY, initBindings={
                "workspace": workspace, "varClass": URIRef(expand(var.property_class))}):
            info = _info(graph, row.artifact, row.prop)
            if info is not None:
                target.properties.append(info)
                target.artifacts.append(row.artifact)
    if not target.properties:
        return f"nothing in this home reads {var.property_class} for {pred.predicate_text!r}"
    return target


def _all_rooms(graph: Graph, property_class: str) -> List[Tuple[Any, Any]]:
    """Every (workspace, variable) of the class, for a predicate naming no room."""
    wanted = URIRef(expand(property_class))
    out = []
    for workspace, space in graph.subject_objects(HOMEONT.hasSpace):
        for env in graph.objects(space, HOMEONT.hasEnvironment):
            for var in graph.objects(env, SSN.hasProperty):
                if (var, RDF.type, wanted) in graph:
                    out.append((workspace, var))
    return sorted(out, key=str)


# --------------------------------------------------------------------------
# The profiling context
# --------------------------------------------------------------------------

def artifact_context(graph: Graph, artifact: Any) -> Dict[str, Any]:
    """One device as the profiling model sees it."""
    return {"name": _title(graph, artifact), "class": device_class(graph, artifact),
            "location_class": device_location(graph, artifact),
            "affordances": artifact_affordances(graph, artifact)}


def profile_context(graph: Graph, artifacts_by_id: Dict[str, Iterable[Any]]) -> Dict[str, Any]:
    """`specific_goal_context`: per goal / predicate id, its devices."""
    out: Dict[str, Any] = {}
    for item_id, artifacts in artifacts_by_id.items():
        seen, entries = set(), []
        for artifact in artifacts:
            if artifact is None or artifact in seen:
                continue
            seen.add(artifact)
            entries.append(artifact_context(graph, artifact))
        out[item_id] = {"artifacts": entries}
    return out


def known_properties(context: Dict[str, Any]) -> Dict[str, set]:
    """Per id, the (artifact, property) pairs its context shows."""
    return {item_id: {(a["name"], aff["name"]) for a in entry["artifacts"]
                      for aff in a["affordances"] if aff["type"] == "property_affordance"}
            for item_id, entry in context.items()}

# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def dependency_structure(structure: GoalStructure) -> Optional[GoalStructure]:
    """The structure, if the dependency planner takes it: a dependency structure,
    or a simple maintenance goal."""
    if not structure.goals:
        return None
    if structure.structure == "dependency":
        return structure
    if len(structure.goals) == 1 and \
            next(iter(structure.goals.values())).goal_kind == "maintenance":
        return structure
    return None

