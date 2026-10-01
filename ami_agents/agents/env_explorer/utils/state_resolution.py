"""Resolve ENV_STATE class identifiers to readable property affordances.

The UA's structuring stage hands over ontology classes -- a location, a device
kind, a device property or an environment variable -- and this turns them into
the affordances that can answer the question, each with the URL its value is
read from.

The resolution is a SPARQL query over the discovered Thing Descriptions plus the
two ontologies that define the vocabulary. It is deliberately *not* string
matching on names: a property affordance in the graph is typed with its homeont
class and carries its own read target, so a class-level match lands directly on
the thing to read.

Values come only from devices that sense them. A room workspace may also expose
a property of its environment, but in a deployable system no such thing exists:
a room is not a sensor. Every query here binds an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdflib import Graph, URIRef
from rdflib.namespace import RDF

from ....shared.utils.namespaces import domain_types

ONTOLOGY_DIR = (
    Path(__file__).resolve().parents[3] / "shared" / "ontologies"
)
HOMEONT_PATH = ONTOLOGY_DIR / "homeont.ttl"
SAREF_PATH = ONTOLOGY_DIR / "saref.rdf"

# Rendering IRIs back as CURIEs keeps the response speaking the same vocabulary
# the request arrived in -- a caller that sent `homeont:AirTemperature` should
# not have to recognise the expanded IRI in the reply.
_CURIE_PREFIXES = (
    ("homeont:", "http://example.org/homeont/"),
    ("sosa:", "http://www.w3.org/ns/sosa/"),
    ("saref:", "https://saref.etsi.org/core/"),
)


def curie(uri: Any) -> str:
    text = str(uri)
    for prefix, namespace in _CURIE_PREFIXES:
        if text.startswith(namespace):
            return prefix + text[len(namespace):]
    return text


SPARQL_PREFIXES = """
PREFIX td:      <https://www.w3.org/2019/wot/td#>
PREFIX hctl:    <https://www.w3.org/2019/wot/hypermedia#>
PREFIX hmas:    <https://purl.org/hmas/>
PREFIX homeont: <http://example.org/homeont/>
PREFIX saref:   <https://saref.etsi.org/core/>
PREFIX schema:  <https://schema.org/>
PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>
"""


class StateOutcome(str, Enum):
    """What a resolution attempt found.

    The distinction that matters is the *semantic type* of the matched
    affordances, not how many rows came back. Several affordances of one type
    are one question answered by several sensors; several types are several
    questions and cannot be answered together.

    `RESOLVED_ARTIFACT` is the case where the device was found but no affordance
    reports what was asked. The artifact is still the answer to questions about
    the device itself -- its make and model are stated on the Thing, not read
    from a property -- so it is returned rather than discarded.
    """

    RESOLVED = "resolved_affordance"
    RESOLVED_ARTIFACT = "resolved_artifact"
    NONE = "no_affordance"
    INDETERMINATE = "indeterminate_affordance"
    MISMATCHED = "mismatched_affordance"


@dataclass
class ResolvedAffordance:
    """One resolved device, with the affordance that answers the question.

    The field names are the ones the retrieval task speaks:
    `<artifact_name, artifact_type, workspace_name, affordance_name,
    affordance_type, parameter_name>`, plus the URI and read target behind them.

    The affordance half is optional. When a device is found but no affordance
    reports what was asked, the same entry carries the device alone and
    `affordance_name` / `affordance_type` / `target` stay None -- the question
    may still be answerable from `manufacturer` and `model`, which are stated on
    the Thing rather than read from a property.

    `parameter_name` names a field inside an object-valued property (the hue of
    a colour reading). Nothing selects one yet -- the parser has no slot for it
    -- so it stays None and such a property is read whole, its value arriving as
    the dictionary the device returns.
    """

    artifact: str
    artifact_name: str
    artifact_type: Optional[str] = None
    workspace_name: Optional[str] = None
    # Thing-level facts about the device: what it is, rather than what it
    # senses. Present on every entry, because "what brand is the freezer" is
    # answerable from the same row that carries its temperature.
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    # Absent when the device resolved but no affordance matched the request.
    affordance_name: Optional[str] = None
    affordance_type: Optional[str] = None
    target: Optional[str] = None
    parameter_name: Optional[str] = None
    # Set once read; absent until then, so an unread affordance is visibly
    # unread rather than indistinguishable from one whose value is null.
    value: Any = None
    has_value: bool = False
    detail: Optional[str] = None

    @property
    def is_readable(self) -> bool:
        """Is there something to dereference, or only the device itself?"""
        return self.target is not None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "artifact_name": self.artifact_name,
            "artifact_type": self.artifact_type,
            "workspace_name": self.workspace_name,
            "artifact": self.artifact,
        }
        if self.manufacturer:
            out["manufacturer"] = self.manufacturer
        if self.model:
            out["model"] = self.model
        # Omitted rather than emitted as null: an entry without an affordance is
        # a device, and saying `"affordance_name": null` three times would
        # describe it as a broken affordance instead.
        if self.is_readable:
            out["affordance_name"] = self.affordance_name
            out["affordance_type"] = self.affordance_type
            out["parameter_name"] = self.parameter_name
            out["target"] = self.target
        if self.has_value:
            out["value"] = self.value
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass
class StateResolution:
    """The outcome, plus what was asked and what was found."""

    outcome: StateOutcome
    affordances: List[ResolvedAffordance] = field(default_factory=list)
    property_classes: List[str] = field(default_factory=list)
    query: Optional[Dict[str, Any]] = None
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "affordances": [a.as_dict() for a in self.affordances],
            "property_classes": self.property_classes,
            "query": self.query,
            "detail": self.detail,
        }


def load_vocabulary(graph: Optional[Graph] = None) -> Graph:
    """Add homeont and SAREF to `graph` (or a new one).

    Both are required, for different reasons. homeont defines the property and
    space classes; SAREF defines `saref:Device` and the five subclasses every
    homeont device family descends from. Without SAREF the path from
    `homeont:AirConditioner` up to `saref:Device` does not close, and a query
    constrained on device-ness silently returns nothing.
    """
    graph = graph if graph is not None else Graph()
    graph.parse(str(HOMEONT_PATH), format="turtle")
    if SAREF_PATH.exists():
        graph.parse(str(SAREF_PATH), format="xml")
    return graph


def discovered_graph(artifacts, workspaces, logger=None) -> Graph:
    """The discovered Thing Descriptions and workspaces, plus the vocabulary.

    Rebuilt per request: artifacts come and go, and a state query is rare enough
    that a stale graph would be a worse trade than the parse cost.
    `rdfs:subClassOf*` only closes if homeont and SAREF are in here too, which
    `load_vocabulary` adds.

    An unparseable Thing Description is skipped rather than fatal: one bad
    artifact should not make the rest of the environment unanswerable.
    """
    graph = Graph()
    for artifact in artifacts:
        rdf = getattr(getattr(artifact, "thing_description", None), "rdf", None)
        if not rdf:
            continue
        try:
            graph.parse(data=rdf, format="turtle")
        except Exception as exc:
            if logger:
                logger.warning("Skipping unparseable TD for %s: %s",
                               getattr(artifact, "name", "?"), exc)
    for workspace in workspaces:
        rdf = getattr(workspace, "rdf", None)
        if not rdf:
            continue
        try:
            graph.parse(data=rdf, format="turtle")
        except Exception as exc:
            if logger:
                logger.warning("Skipping unparseable workspace RDF for %s: %s",
                               getattr(workspace, "name", "?"), exc)
    return load_vocabulary(graph)


def scope_lines(location_class: Optional[str],
                 device_class: Optional[str]) -> List[str]:
    """The location and device constraints both queries share."""
    lines: List[str] = []
    if location_class:
        lines.append(f"  ?space a {location_class} ; homeont:isSpaceOfWorkspace ?ws .")
        lines.append("  ?space rdfs:label ?spaceLabel .")
        lines.append("  ?ws hmas:contains ?artifact .")
    if device_class:
        # A path, so an intermediate class (`saref:Appliance`) still reaches its
        # subclasses instead of matching only an exact assertion.
        lines.append(f"  ?artifact a/rdfs:subClassOf* {device_class} .")
    return lines


# Thing-level facts about the device. OPTIONAL, unlike the names: a Thing
# Description may legitimately state neither, and a device that does not say who
# made it must still be found.
METADATA_LINES = [
    "  OPTIONAL { ?artifact schema:manufacturer ?manufacturer }",
    "  OPTIONAL { ?artifact schema:model ?model }",
]


def build_query(
    location_class: Optional[str] = None,
    device_class: Optional[str] = None,
    property_class: Optional[str] = None,
) -> str:
    """Compose the SPARQL for whichever slots the parser determined.

    Device-ness is asked as `FILTER EXISTS` -- *is there a path* to
    `saref:Device` -- rather than by binding a `?devClass` variable. Binding it
    makes SPARQL return every value that satisfies the constraint, one row per
    ancestor, so an air conditioner comes back as both `homeont:AirConditioner`
    and `saref:HVAC` and one sensor looks like an aggregation. The device class
    itself is read from the artifact's own asserted types, not projected here.

    Names are required, not OPTIONAL: both belong to the retrieval tuple, so a
    Thing Description that states neither is malformed and should drop out
    rather than yield a row with a silently missing field. The make and model
    are OPTIONAL for the opposite reason -- they are facts a TD may omit.
    """
    lines = scope_lines(location_class, device_class)
    lines.append("  ?artifact td:title ?artTitle ; td:hasPropertyAffordance ?prop .")
    lines.append("  ?prop a ?pc ; td:name ?name ; td:hasForm [ hctl:hasTarget ?target ] .")
    lines.append(f"  ?pc rdfs:subClassOf* {property_class} .")
    lines.extend(METADATA_LINES)
    lines.append("  FILTER EXISTS { ?artifact a/rdfs:subClassOf* saref:Device }")
    body = "\n".join(lines)
    projection = "?artifact ?artTitle ?pc ?name ?target ?manufacturer ?model"
    if location_class:
        projection += " ?spaceLabel"
    return (
        f"{SPARQL_PREFIXES}\n"
        f"SELECT DISTINCT {projection} WHERE {{\n{body}\n}}"
    )


def build_artifact_query(
    location_class: Optional[str] = None,
    device_class: Optional[str] = None,
) -> str:
    """The same scope, without requiring an affordance.

    Answers "is this device here at all, and what is it?" -- which is a
    different question from "what can it tell me", and the one left when no
    affordance reports what was asked. A device's make and model are stated on
    the Thing, so this reaches them where the affordance query cannot.
    """
    lines = scope_lines(location_class, device_class)
    lines.append("  ?artifact td:title ?artTitle .")
    lines.extend(METADATA_LINES)
    lines.append("  FILTER EXISTS { ?artifact a/rdfs:subClassOf* saref:Device }")
    body = "\n".join(lines)
    projection = "?artifact ?artTitle ?manufacturer ?model"
    if location_class:
        projection += " ?spaceLabel"
    return (
        f"{SPARQL_PREFIXES}\n"
        f"SELECT DISTINCT {projection} WHERE {{\n{body}\n}}"
    )


def device_class_of(graph: Graph, artifact: Any) -> Optional[str]:
    """The artifact's own device family, from the types it asserts.

    Read from the graph rather than projected by the query, so the ancestor
    fan-out that `?devClass` caused cannot come back. `domain_types` keeps the
    home-ontology terms, which after the artifact-scoped extraction is exactly
    the device family.
    """
    asserted = domain_types(graph.objects(URIRef(str(artifact)), RDF.type))
    return asserted[0] if asserted else None


def _artifacts(graph: Graph, location_class: Optional[str],
               device_class: Optional[str]) -> List[ResolvedAffordance]:
    """The devices in scope, each as an entry with no affordance attached."""
    rows = list(graph.query(build_artifact_query(location_class, device_class)))
    return [
        ResolvedAffordance(
            artifact=str(row.artifact),
            artifact_name=str(row.artTitle),
            artifact_type=device_class_of(graph, row.artifact),
            workspace_name=(str(row.spaceLabel)
                            if getattr(row, "spaceLabel", None) else None),
            manufacturer=(str(row.manufacturer)
                          if getattr(row, "manufacturer", None) else None),
            model=str(row.model) if getattr(row, "model", None) else None,
        )
        for row in rows
    ]


def resolve_state_request(
    graph: Graph,
    location_class: Optional[str] = None,
    device_class: Optional[str] = None,
    device_property: Optional[Dict[str, Any]] = None,
    environment_variable: Optional[Dict[str, Any]] = None,
) -> StateResolution:
    """Find what can answer one structured state request.

    Usually that is a property affordance. When the request names a device but
    no affordance reports what was asked, the device itself is the answer --
    "what make is the freezer" is answered from the Thing, not from a reading --
    so the artifact is returned instead of the request failing.

    `graph` must already hold the discovered Thing Descriptions and the
    vocabulary (see `load_vocabulary`).
    """
    property_class = None
    for candidate in (device_property, environment_variable):
        if isinstance(candidate, dict) and candidate.get("class"):
            property_class = str(candidate["class"])
            break

    asked = {
        "location_class": location_class,
        "device_class": device_class,
        "property_class": property_class,
    }

    if not property_class and not device_class:
        # Neither a property nor a device: nothing to look for, and nothing to
        # return. A location alone does not qualify -- "how is the kitchen?"
        # would otherwise be answered with an inventory of every device in it,
        # which is not what was asked. Decided before any query runs.
        return StateResolution(
            outcome=StateOutcome.INDETERMINATE,
            query=asked,
            detail=("the request named neither a device property nor an "
                    "environment variable"),
        )

    affordances: List[ResolvedAffordance] = []
    if property_class:
        rows = list(graph.query(
            build_query(location_class, device_class, property_class)))
        affordances = [
            ResolvedAffordance(
                artifact=str(row.artifact),
                artifact_name=str(row.artTitle),
                artifact_type=device_class_of(graph, row.artifact),
                workspace_name=(str(row.spaceLabel)
                                if getattr(row, "spaceLabel", None) else None),
                manufacturer=(str(row.manufacturer)
                              if getattr(row, "manufacturer", None) else None),
                model=str(row.model) if getattr(row, "model", None) else None,
                affordance_name=str(row.name),
                affordance_type=curie(row.pc),
                target=str(row.target),
            )
            for row in rows
        ]

    if affordances:
        property_classes = sorted({a.affordance_type for a in affordances})
        if len(property_classes) > 1:
            # Usually the parser answered with a class too high in the taxonomy:
            # one generic class matches many unrelated properties.
            return StateResolution(
                outcome=StateOutcome.MISMATCHED,
                affordances=affordances,
                property_classes=property_classes,
                query=asked,
                detail=(f"{property_class} matched {len(property_classes)} "
                        "different property types; the request does not "
                        "identify one"),
            )
        return StateResolution(
            outcome=StateOutcome.RESOLVED,
            affordances=affordances,
            property_classes=property_classes,
            query=asked,
            detail=(f"{len(affordances)} affordance(s) of {property_classes[0]}"),
        )

    # No affordance answered. If the request named a device, that device may
    # still be there and may still answer a question about itself.
    #
    # Only when a device was named: "how bright is the kitchen" names a property
    # nothing senses, and answering it with every device in the kitchen would be
    # an inventory, not an answer. The device is what the question is *about*;
    # a room is only where to look.
    where = location_class or "this environment"
    artifacts = (_artifacts(graph, location_class, device_class)
                 if device_class else [])

    if artifacts:
        asked_for = property_class or "that"
        return StateResolution(
            outcome=StateOutcome.RESOLVED_ARTIFACT,
            affordances=artifacts,
            query=asked,
            detail=(f"{len(artifacts)} device(s) found in {where}; none "
                    f"reports {asked_for}"),
        )

    missing = device_class or property_class
    return StateResolution(
        outcome=StateOutcome.NONE,
        query=asked,
        detail=f"nothing in {where} matches {missing}",
    )
