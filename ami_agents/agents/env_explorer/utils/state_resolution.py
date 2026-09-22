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


def _curie(uri: Any) -> str:
    text = str(uri)
    for prefix, namespace in _CURIE_PREFIXES:
        if text.startswith(namespace):
            return prefix + text[len(namespace):]
    return text


_PREFIXES = """
PREFIX td:      <https://www.w3.org/2019/wot/td#>
PREFIX hctl:    <https://www.w3.org/2019/wot/hypermedia#>
PREFIX hmas:    <https://purl.org/hmas/>
PREFIX homeont: <http://example.org/homeont/>
PREFIX saref:   <https://saref.etsi.org/core/>
PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>
"""


class StateOutcome(str, Enum):
    """What a resolution attempt found.

    The distinction that matters is the *semantic type* of the matched
    affordances, not how many rows came back. Several affordances of one type
    are one question answered by several sensors; several types are several
    questions and cannot be answered together.
    """

    RESOLVED = "resolved_affordance"
    NONE = "no_affordance"
    INDETERMINATE = "indeterminate_affordance"
    MISMATCHED = "mismatched_affordance"


@dataclass
class ResolvedAffordance:
    """One readable property affordance, as a value-retrieval tuple.

    The field names are the ones the retrieval task speaks:
    `<artifact_name, artifact_type, workspace_name, affordance_name,
    affordance_type, parameter_name>`, plus the URI and read target behind them.

    `parameter_name` names a field inside an object-valued property (the hue of
    a colour reading). Nothing selects one yet -- the parser has no slot for it
    -- so it stays None and such a property is read whole, its value arriving as
    the dictionary the device returns.
    """

    artifact: str
    artifact_name: str
    artifact_type: Optional[str]
    workspace_name: Optional[str]
    affordance_name: str
    affordance_type: str
    target: str
    parameter_name: Optional[str] = None
    # Set once read; absent until then, so an unread affordance is visibly
    # unread rather than indistinguishable from one whose value is null.
    value: Any = None
    has_value: bool = False
    detail: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "artifact_name": self.artifact_name,
            "artifact_type": self.artifact_type,
            "workspace_name": self.workspace_name,
            "affordance_name": self.affordance_name,
            "affordance_type": self.affordance_type,
            "parameter_name": self.parameter_name,
            "artifact": self.artifact,
            "target": self.target,
        }
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
    rather than yield a row with a silently missing field.
    """
    lines: List[str] = []
    if location_class:
        lines.append(f"  ?space a {location_class} ; homeont:isSpaceOfWorkspace ?ws .")
        lines.append("  ?space rdfs:label ?spaceLabel .")
        lines.append("  ?ws hmas:contains ?artifact .")
    if device_class:
        # A path, so an intermediate class (`saref:Appliance`) still reaches its
        # subclasses instead of matching only an exact assertion.
        lines.append(f"  ?artifact a/rdfs:subClassOf* {device_class} .")
    lines.append("  ?artifact td:title ?artTitle ; td:hasPropertyAffordance ?prop .")
    lines.append("  ?prop a ?pc ; td:name ?name ; td:hasForm [ hctl:hasTarget ?target ] .")
    lines.append(f"  ?pc rdfs:subClassOf* {property_class} .")
    lines.append("  FILTER EXISTS { ?artifact a/rdfs:subClassOf* saref:Device }")
    body = "\n".join(lines)
    projection = "?artifact ?artTitle ?pc ?name ?target"
    if location_class:
        projection += " ?spaceLabel"
    return (
        f"{_PREFIXES}\n"
        f"SELECT DISTINCT {projection} WHERE {{\n{body}\n}}"
    )


def _device_class(graph: Graph, artifact: Any) -> Optional[str]:
    """The artifact's own device family, from the types it asserts.

    Read from the graph rather than projected by the query, so the ancestor
    fan-out that `?devClass` caused cannot come back. `domain_types` keeps the
    home-ontology terms, which after the artifact-scoped extraction is exactly
    the device family.
    """
    asserted = domain_types(graph.objects(URIRef(str(artifact)), RDF.type))
    return asserted[0] if asserted else None


def resolve_state_request(
    graph: Graph,
    location_class: Optional[str] = None,
    device_class: Optional[str] = None,
    device_property: Optional[Dict[str, Any]] = None,
    environment_variable: Optional[Dict[str, Any]] = None,
) -> StateResolution:
    """Find the affordances that can answer one structured state request.

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

    if not property_class:
        # Nothing readable was named, so there is nothing to look for. Decided
        # before any query runs -- this is a parse outcome, not a query result.
        return StateResolution(
            outcome=StateOutcome.INDETERMINATE,
            query=asked,
            detail=("the request named neither a device property nor an "
                    "environment variable"),
        )

    query = build_query(location_class, device_class, property_class)
    rows = list(graph.query(query))

    affordances = [
        ResolvedAffordance(
            artifact=str(row.artifact),
            artifact_name=str(row.artTitle),
            artifact_type=_device_class(graph, row.artifact),
            workspace_name=(str(row.spaceLabel)
                            if getattr(row, "spaceLabel", None) else None),
            affordance_name=str(row.name),
            affordance_type=_curie(row.pc),
            target=str(row.target),
        )
        for row in rows
    ]
    property_classes = sorted({a.affordance_type for a in affordances})

    if not affordances:
        where = location_class or "this environment"
        return StateResolution(
            outcome=StateOutcome.NONE,
            query=asked,
            detail=f"nothing in {where} senses {property_class}",
        )

    if len(property_classes) > 1:
        # Usually the parser answered with a class too high in the taxonomy:
        # one generic class matches many unrelated properties.
        return StateResolution(
            outcome=StateOutcome.MISMATCHED,
            affordances=affordances,
            property_classes=property_classes,
            query=asked,
            detail=(f"{property_class} matched {len(property_classes)} different "
                    f"property types; the request does not identify one"),
        )

    return StateResolution(
        outcome=StateOutcome.RESOLVED,
        affordances=affordances,
        property_classes=property_classes,
        query=asked,
        detail=(f"{len(affordances)} affordance(s) of {property_classes[0]}"),
    )
