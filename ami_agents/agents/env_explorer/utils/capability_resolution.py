"""Resolve ENV_CAPABILITIES class identifiers to the affordances that answer them.

The UA's structuring stage hands over ontology classes -- a location, a device
kind, a device property or an environment variable, and a command -- and this
finds what in the discovered environment *can* do or report that. Nothing is
read: a capability is a fact about the Thing Descriptions, so the answer comes
from the graph alone, including the values a property or action admits
(`js:enum` / `js:minimum` / `js:maximum` on its schema).

Every class is matched with `rdfs:subClassOf*`. TDs are typed with
device-specific leaves (`homeont:DimmableLightBrightness`), while a parser
answering a question that names no device correctly picks the general class
(`homeont:LevelControlBrightness`). Same rule as `state_resolution.build_query`.

Which query runs depends on which slots are filled:

    device_property (metadata)        does the device state its make / model
    device_property                   property affordances of that class
    environment_variable + command    actions whose effect acts on that variable
    environment_variable              property affordances sensing it
    command                           action affordances performing it
    location and/or device only       inventory of everything in scope
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

from .state_resolution import (
    METADATA_LINES,
    SPARQL_PREFIXES,
    build_artifact_query,
    curie,
    device_class_of,
    scope_lines,
)

HOMEONT = Namespace("http://example.org/homeont/")
HMAS = Namespace("https://purl.org/hmas/")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
TD = Namespace("https://www.w3.org/2019/wot/td#")
JS = Namespace("https://www.w3.org/2019/wot/json-schema#")
TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")
SAREF = Namespace("https://saref.etsi.org/core/")

_PREFIXES = SPARQL_PREFIXES + """PREFIX sosa:    <http://www.w3.org/ns/sosa/>
PREFIX tdsosa:  <https://example.org/hmas/td-sosa-ext#>
"""

# The root a matched property class descends from says what kind of capability
# it is, which is what the answer has to say: changeable, reported, supported.
_PROPERTY_BRANCHES = (
    (HOMEONT.ActuatableDeviceProperty, "actuatable"),
    (HOMEONT.DeviceStateProperty, "state"),
    (HOMEONT.DeviceCapabilityProperty, "capability"),
    (SOSA.ObservableProperty, "measurement"),
)

# Thing-level facts, stated on the TD rather than served as affordances.
METADATA_PROPERTIES = ("schema:manufacturer", "schema:model")

# Types every affordance carries, which say nothing about what it does.
_DOMAIN_NAMESPACES = ("http://example.org/homeont/", "https://saref.etsi.org/core/")


class CapabilityOutcome(str, Enum):
    """What a capability resolution found.

    `DEVICE_LACKS` is the case where the named device is there but nothing on
    it matches: "can the purifier dim?" is answered *no, it cannot*, which is a
    different sentence from *there is no purifier*.
    """

    FOUND = "found"
    DEVICE_LACKS = "device_lacks_capability"
    NONE = "none"
    INDETERMINATE = "indeterminate"


@dataclass
class CapabilityEntry:
    """One device, with the affordance that provides the capability.

    The affordance half is absent when the entry is the device alone -- an
    inventory row for a device with nothing typed, a metadata answer, or a
    device that lacks the capability asked about.
    """

    artifact: str
    artifact_name: str
    artifact_type: Optional[str] = None
    workspace_name: Optional[str] = None
    # The room kind (`homeont:Kitchen`), so a caller can fill a location class.
    workspace_class: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    affordance_name: Optional[str] = None
    affordance_kind: Optional[str] = None      # "property" | "action"
    affordance_type: Optional[str] = None
    # For a property: actuatable / state / capability / measurement / environment.
    property_branch: Optional[str] = None
    # From the affordance's schema: enum (+ its gloss), minimum, maximum.
    permitted_values: Optional[Dict[str, Any]] = None
    # For an effect: the variable acted on and, when stated, the direction.
    effect_on: Optional[str] = None
    effect_direction: Optional[str] = None
    # For an action: the properties it changes (`saref:actsUpon`), each as
    # {"affordance_name", "affordance_type"}.
    acts_upon: Optional[List[Dict[str, Any]]] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "artifact_name": self.artifact_name,
            "artifact_type": self.artifact_type,
            "workspace_name": self.workspace_name,
            "artifact": self.artifact,
        }
        # Omitted rather than null, as in ENV_STATE: an entry without an
        # affordance is a device, not a broken affordance.
        for key in ("workspace_class", "manufacturer", "model", "affordance_name",
                    "affordance_kind", "affordance_type", "property_branch",
                    "permitted_values", "effect_on", "effect_direction",
                    "acts_upon"):
            value = getattr(self, key)
            if value:
                out[key] = value
        return out


@dataclass
class CapabilityResolution:
    """The outcome, plus what was asked and what was found."""

    outcome: CapabilityOutcome
    entries: List[CapabilityEntry] = field(default_factory=list)
    query: Optional[Dict[str, Any]] = None
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "entries": [e.as_dict() for e in self.entries],
            "query": self.query,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

def _finish(lines: List[str], projection: str, location_class: Optional[str]) -> str:
    lines.extend(METADATA_LINES)
    lines.append("  FILTER EXISTS { ?artifact a/rdfs:subClassOf* saref:Device }")
    if location_class:
        projection += " ?spaceLabel"
    body = "\n".join(lines)
    return f"{_PREFIXES}\nSELECT DISTINCT {projection} WHERE {{\n{body}\n}}"


def build_property_query(location_class: Optional[str],
                         device_class: Optional[str],
                         property_class: str) -> str:
    """Property affordances whose class is `property_class` or below it."""
    lines = scope_lines(location_class, device_class)
    lines.append("  ?artifact td:title ?artTitle ; td:hasPropertyAffordance ?aff .")
    lines.append("  ?aff a ?ac ; td:name ?name .")
    lines.append(f"  ?ac rdfs:subClassOf* {property_class} .")
    return _finish(lines, "?artifact ?artTitle ?aff ?ac ?name ?manufacturer ?model",
                   location_class)


def build_action_query(location_class: Optional[str],
                       device_class: Optional[str],
                       command_class: str) -> str:
    """Action affordances performing `command_class` or a specialisation of it.

    `homeont:SetOnOffCommand` specialises On, Off and Toggle, so a request for
    `saref:OnCommand` reaches it through the path.
    """
    lines = scope_lines(location_class, device_class)
    lines.append("  ?artifact td:title ?artTitle ; td:hasActionAffordance ?aff .")
    lines.append("  ?aff a ?ac ; td:name ?name .")
    lines.append(f"  ?ac rdfs:subClassOf* {command_class} .")
    return _finish(lines, "?artifact ?artTitle ?aff ?ac ?name ?manufacturer ?model",
                   location_class)


def build_acts_upon_query(location_class: Optional[str],
                          device_class: Optional[str],
                          command_class: str,
                          property_class: str) -> str:
    """Actions performing `command_class` that change a `property_class` property.

    The link is the action's own `saref:actsUpon` triple to the property
    affordance it changes, never a shared name.
    """
    lines = scope_lines(location_class, device_class)
    lines.append("  ?artifact td:title ?artTitle ; td:hasActionAffordance ?aff .")
    lines.append("  ?aff a ?ac ; td:name ?name ; saref:actsUpon ?target .")
    lines.append(f"  ?ac rdfs:subClassOf* {command_class} .")
    lines.append(f"  ?target a/rdfs:subClassOf* {property_class} .")
    return _finish(lines, "?artifact ?artTitle ?aff ?ac ?name ?manufacturer ?model",
                   location_class)


def build_effect_query(location_class: Optional[str],
                       device_class: Optional[str],
                       environment_class: str,
                       command_class: Optional[str] = None) -> str:
    """Actions that state an effect on a variable of `environment_class`.

    The effect target is the room's own observable property, typed with its
    homeont class (SimuHome's TD builder). `command_class` narrows the actions
    further; `saref:Command` (any action) adds nothing and is skipped.
    """
    lines = scope_lines(location_class, device_class)
    lines.append("  ?artifact td:title ?artTitle ; td:hasActionAffordance ?aff .")
    lines.append("  ?aff td:name ?name ; tdsosa:hasEffectActuation ?actuation .")
    lines.append("  ?actuation sosa:actsOnProperty ?obs .")
    lines.append(f"  ?obs a/rdfs:subClassOf* {environment_class} .")
    lines.append("  ?aff a ?ac .")
    lines.append("  ?ac rdfs:subClassOf* saref:Command .")
    if command_class and command_class != "saref:Command":
        lines.append(f"  ?ac rdfs:subClassOf* {command_class} .")
    return _finish(
        lines,
        "?artifact ?artTitle ?aff ?ac ?name ?actuation ?manufacturer ?model",
        location_class)


def build_inventory_query(location_class: Optional[str],
                          device_class: Optional[str]) -> str:
    """Every typed property and action affordance of the devices in scope."""
    lines = scope_lines(location_class, device_class)
    lines.append("  ?artifact td:title ?artTitle .")
    lines.append("  { ?artifact td:hasPropertyAffordance ?aff . BIND(\"property\" AS ?kind) }")
    lines.append("  UNION { ?artifact td:hasActionAffordance ?aff . BIND(\"action\" AS ?kind) }")
    lines.append("  ?aff a ?ac ; td:name ?name .")
    filters = " || ".join(f'STRSTARTS(STR(?ac), "{ns}")' for ns in _DOMAIN_NAMESPACES)
    lines.append(f"  FILTER({filters})")
    return _finish(
        lines, "?artifact ?artTitle ?kind ?aff ?ac ?name ?manufacturer ?model",
        location_class)


# --------------------------------------------------------------------------
# Rows -> entries
# --------------------------------------------------------------------------

def _literal(value: Any) -> Any:
    return value.toPython() if hasattr(value, "toPython") else str(value)


def permitted_values(graph: Graph, schema: Any) -> Optional[Dict[str, Any]]:
    """What a schema admits: `enum` (with its gloss), `minimum`, `maximum`.

    The enum members are the wire values, usually integers; the schema's
    `js:description` is what says "0 = Off, 1 = Low, ..." and travels with it.
    """
    if schema is None:
        return None
    out: Dict[str, Any] = {}
    enum = sorted((_literal(v) for v in graph.objects(schema, JS.enum)),
                  key=lambda v: (str(type(v)), v))
    if enum:
        out["enum"] = enum
        gloss = graph.value(schema, JS.description)
        if gloss is not None:
            out["meaning"] = str(gloss)
    for key, predicate in (("minimum", JS.minimum), ("maximum", JS.maximum)):
        value = graph.value(schema, predicate)
        if value is not None:
            out[key] = _literal(value)
    return out or None


_CURIE_NAMESPACES = {
    "homeont": str(HOMEONT),
    "saref": str(SAREF),
    "sosa": str(SOSA),
    "schema": "https://schema.org/",
}


def expand_curie(value: Optional[str]) -> Optional[URIRef]:
    """`homeont:OnOff` -> its IRI; a full IRI is returned as is."""
    if not value:
        return None
    prefix, _, local = str(value).partition(":")
    namespace = _CURIE_NAMESPACES.get(prefix)
    return URIRef(namespace + local) if namespace else URIRef(str(value))


def property_branch(graph: Graph, cls: Any) -> Optional[str]:
    """Which root a property class descends from (or `environment`)."""
    uri = URIRef(str(cls))
    if (uri, SSN.isPropertyOf, HOMEONT.Environment) in graph:
        return "environment"
    ancestors = set(graph.transitive_objects(uri, RDFS.subClassOf))
    for root, name in _PROPERTY_BRANCHES:
        if root in ancestors:
            return name
    return None


def acts_upon(graph: Graph, action: Any) -> List[Dict[str, Any]]:
    """The property affordances an action states it changes."""
    targets = []
    for target in graph.objects(action, SAREF.actsUpon):
        name = graph.value(target, TD.name)
        types = sorted(curie(t) for t in graph.objects(target, RDF.type)
                       if str(t).startswith(str(HOMEONT)))
        targets.append({"affordance_name": str(name) if name else None,
                        "affordance_type": types[0] if types else None})
    return sorted(targets, key=lambda t: str(t["affordance_name"]))


def _effect_direction(graph: Graph, actuation: Any) -> Optional[str]:
    if (actuation, RDF.type, TDSOSA.IncreasingActuation) in graph:
        return "increase"
    if (actuation, RDF.type, TDSOSA.DecreasingActuation) in graph:
        return "decrease"
    return None


def workspace_label(graph: Graph, artifact: Any) -> Optional[str]:
    """The room an artifact is in, by the label of the space its workspace is.

    The scoped queries project it only when a location was asked for; an answer
    naming devices needs it either way ("the living-room light").
    """
    for workspace in graph.subjects(HMAS.contains, URIRef(str(artifact))):
        for space in graph.subjects(HOMEONT.isSpaceOfWorkspace, workspace):
            label = graph.value(space, RDFS.label)
            if label is not None:
                return str(label)
    return None


def workspace_class(graph: Graph, artifact: Any) -> Optional[str]:
    """The room kind an artifact is in (`homeont:Kitchen`), not just its name."""
    for workspace in graph.subjects(HMAS.contains, URIRef(str(artifact))):
        for space in graph.subjects(HOMEONT.isSpaceOfWorkspace, workspace):
            for cls in sorted(graph.objects(space, RDF.type), key=str):
                if str(cls).startswith(str(HOMEONT)) and cls != HOMEONT.BuildingSpace:
                    return curie(cls)
    return None


def _optional(row: Any, name: str) -> Optional[str]:
    value = getattr(row, name, None)
    return str(value) if value is not None else None


def _entry(graph: Graph, row: Any, kind: Optional[str] = None) -> CapabilityEntry:
    """One SPARQL row as an entry, with schema and branch looked up beside it."""
    entry = CapabilityEntry(
        artifact=str(row.artifact),
        artifact_name=str(row.artTitle),
        artifact_type=device_class_of(graph, row.artifact),
        workspace_name=(_optional(row, "spaceLabel")
                        or workspace_label(graph, row.artifact)),
        workspace_class=workspace_class(graph, row.artifact),
        manufacturer=_optional(row, "manufacturer"),
        model=_optional(row, "model"),
    )
    aff = getattr(row, "aff", None)
    if aff is None:
        return entry

    kind = kind or str(getattr(row, "kind", "") or "") or None
    entry.affordance_name = str(row.name)
    entry.affordance_kind = kind
    entry.affordance_type = curie(row.ac)
    if kind == "property":
        entry.property_branch = property_branch(graph, row.ac)
        schema = graph.value(aff, TD.hasOutputSchema)
    else:
        schema = graph.value(aff, TD.hasInputSchema)
        entry.acts_upon = acts_upon(graph, aff) or None
    entry.permitted_values = permitted_values(graph, schema)
    return entry


def _dedupe(entries: List[CapabilityEntry]) -> List[CapabilityEntry]:
    """One entry per (device, affordance), in a stable order.

    An affordance typed with two domain classes on one path would otherwise
    come back once per class.
    """
    seen = set()
    out: List[CapabilityEntry] = []
    for entry in entries:
        key = (entry.artifact, entry.affordance_kind, entry.affordance_name,
               entry.effect_on)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    out.sort(key=lambda e: (e.workspace_name or "", e.artifact_name,
                            e.affordance_kind or "", e.affordance_name or ""))
    return out


def _devices(graph: Graph, location_class: Optional[str],
             device_class: Optional[str]) -> List[CapabilityEntry]:
    """The devices in scope, each as an entry with no affordance attached."""
    rows = graph.query(build_artifact_query(location_class, device_class))
    return _dedupe([_entry(graph, row) for row in rows])


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def _class_of(slot: Any) -> Optional[str]:
    if isinstance(slot, dict) and slot.get("class"):
        return str(slot["class"])
    return None


def resolve_capability_request(
    graph: Graph,
    location_class: Optional[str] = None,
    device_class: Optional[str] = None,
    device_property: Optional[Dict[str, Any]] = None,
    environment_variable: Optional[Dict[str, Any]] = None,
    command: Optional[Dict[str, Any]] = None,
    artifact_name: Optional[str] = None,
) -> CapabilityResolution:
    """Find what in the environment provides one structured capability.

    `graph` must already hold the discovered Thing Descriptions and the
    vocabulary (see `state_resolution.discovered_graph`).

    `artifact_name` narrows the answer to the device with that `td:title` --
    used when a goal names its device but not its room.
    """
    resolution = _resolve(graph, location_class, device_class, device_property,
                          environment_variable, command)
    if not artifact_name:
        return resolution

    resolution.query = dict(resolution.query or {}, artifact_name=artifact_name)
    named = [e for e in resolution.entries if e.artifact_name == artifact_name]
    if named:
        resolution.entries = named
        return resolution
    return CapabilityResolution(
        outcome=CapabilityOutcome.NONE,
        query=resolution.query,
        detail=f"no device titled {artifact_name!r} matches: {resolution.detail}",
    )


def _resolve(
    graph: Graph,
    location_class: Optional[str],
    device_class: Optional[str],
    device_property: Optional[Dict[str, Any]],
    environment_variable: Optional[Dict[str, Any]],
    command: Optional[Dict[str, Any]],
) -> CapabilityResolution:
    property_class = _class_of(device_property)
    environment_class = _class_of(environment_variable)
    command_class = _class_of(command)

    asked = {
        "location_class": location_class,
        "device_class": device_class,
        "property_class": property_class,
        "environment_variable": environment_class,
        "command_class": command_class,
    }
    where = location_class or "this environment"

    if property_class in METADATA_PROPERTIES:
        return _resolve_metadata(graph, location_class, device_class,
                                 property_class, asked, where)

    # A capability can be modelled as a changeable property, as a command, or
    # both; the vocabulary decides which slots the parser fills.
    #   command + actuatable property  -> union of the property and command routes;
    #   command + any other property   -> only the commands that `saref:actsUpon`
    #                                     a property of that class.
    branch = (property_branch(graph, expand_curie(property_class))
              if property_class else None)
    asked["property_branch"] = branch
    acts_upon_only = bool(property_class and command_class and branch != "actuatable")

    entries: List[CapabilityEntry] = []
    asked_for: List[str] = []

    if acts_upon_only:
        rows = graph.query(build_acts_upon_query(
            location_class, device_class, command_class, property_class))
        entries += [_entry(graph, row, "action") for row in rows]
        asked_for.append(f"{command_class} acting on {property_class}")
    elif property_class:
        rows = graph.query(build_property_query(
            location_class, device_class, property_class))
        entries += [_entry(graph, row, "property") for row in rows]
        asked_for.append(property_class)

    if environment_class and command_class:
        rows = graph.query(build_effect_query(
            location_class, device_class, environment_class, command_class))
        for row in rows:
            entry = _entry(graph, row, "action")
            entry.effect_on = environment_class
            entry.effect_direction = _effect_direction(graph, row.actuation)
            entries.append(entry)
        asked_for.append(f"an action affecting {environment_class}")
    elif environment_class:
        rows = graph.query(build_property_query(
            location_class, device_class, environment_class))
        entries += [_entry(graph, row, "property") for row in rows]
        asked_for.append(environment_class)
    elif command_class and not acts_upon_only:
        rows = graph.query(build_action_query(
            location_class, device_class, command_class))
        entries += [_entry(graph, row, "action") for row in rows]
        asked_for.append(command_class)

    if not asked_for:
        if not (location_class or device_class):
            return CapabilityResolution(
                outcome=CapabilityOutcome.INDETERMINATE,
                query=asked,
                detail="the request named no location, device, property, "
                       "environment variable or command",
            )
        rows = graph.query(build_inventory_query(location_class, device_class))
        entries = [_entry(graph, row) for row in rows]
        asked_for.append("anything")
    what = " or ".join(asked_for)

    entries = _dedupe(entries)
    if entries:
        devices = len({e.artifact for e in entries})
        return CapabilityResolution(
            outcome=CapabilityOutcome.FOUND,
            entries=entries,
            query=asked,
            detail=f"{len(entries)} affordance(s) on {devices} device(s) in "
                   f"{where} provide {what}",
        )

    # Only a named device can *lack* something: "can anything dim in the
    # kitchen" with nothing dimmable is simply no, not a device inventory.
    devices = _devices(graph, location_class, device_class) if device_class else []
    if devices:
        return CapabilityResolution(
            outcome=CapabilityOutcome.DEVICE_LACKS,
            entries=devices,
            query=asked,
            detail=f"{len(devices)} {device_class} device(s) in {where}; none "
                   f"provides {what}",
        )
    return CapabilityResolution(
        outcome=CapabilityOutcome.NONE,
        query=asked,
        detail=f"nothing in {where} provides {what}"
               + (f" ({device_class} not found)" if device_class else ""),
    )


def _resolve_metadata(graph: Graph, location_class: Optional[str],
                      device_class: Optional[str], property_class: str,
                      asked: Dict[str, Any], where: str) -> CapabilityResolution:
    """Does the device state its make / model? Answered from the Thing."""
    devices = _devices(graph, location_class, device_class)
    attribute = property_class.split(":", 1)[1]
    stating = [d for d in devices if getattr(d, attribute)]
    if stating:
        return CapabilityResolution(
            outcome=CapabilityOutcome.FOUND,
            entries=stating,
            query=asked,
            detail=f"{len(stating)} device(s) in {where} state {property_class}",
        )
    if devices:
        return CapabilityResolution(
            outcome=CapabilityOutcome.DEVICE_LACKS,
            entries=devices,
            query=asked,
            detail=f"{len(devices)} device(s) in {where}; none states "
                   f"{property_class}",
        )
    return CapabilityResolution(
        outcome=CapabilityOutcome.NONE,
        query=asked,
        detail=f"no {device_class or 'device'} in {where}",
    )
