"""The capabilities context: `homeont.ttl` projected into structured JSON.

The intent parsers need the home vocabulary, not the home ontology. Handing an
LLM the raw Turtle means ~17k tokens of Matter provenance, object properties and
design commentary it cannot act on, and leaves it to infer the taxonomy from
scattered `rdfs:subClassOf` lines.

This module answers the one question a parser asks -- *which classes may I
choose from, and what does each mean* -- as a deterministic projection:

    locations                        where a request can be scoped
    device_types                     what kind of thing it targets
    device_properties.capabilities   what a device CAN do
    device_properties.states         what condition it is in
    device_properties.actuatable     what can be changed (and usually read back)
    device_properties.measurements   what it measures about itself
    environment_variables            what a space's environment exposes

No LLM call, no network, no discovered-environment dependency: the same
ontology the graph is typed against, rendered for a prompt.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

HOMEONT = Namespace("http://example.org/homeont/")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SAREF = Namespace("https://saref.etsi.org/core/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
SSN_SYSTEM = Namespace("http://www.w3.org/ns/ssn/systems/")
QUDT = Namespace("http://qudt.org/schema/qudt/")

# .../ami_agents/agents/user_assistant/utils/ -> .../ami_agents/
ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[3] / "shared" / "ontologies" / "homeont.ttl"
)

# Prefixes the context renders identifiers with. The parser quotes these back
# verbatim and the resolver puts them straight into SPARQL, so they must match
# the prefixes the served graphs use.
_PREFIXES = (
    ("homeont:", str(HOMEONT)),
    ("sosa:", str(SOSA)),
    ("saref:", str(SAREF)),
    ("ssn-system:", str(SSN_SYSTEM)),
    ("ssn:", str(SSN)),
    ("quantitykind:", "http://qudt.org/vocab/quantitykind/"),
    ("unit:", "http://qudt.org/vocab/unit/"),
)

# The SAREF classes homeont hangs its device families from. SAREF defines
# exactly these five subclasses of saref:Device and homeont adds no others.
_DEVICE_ROOTS = (SAREF.HVAC, SAREF.Actuator, SAREF.Appliance,
                 SAREF.Sensor, SAREF.Meter)


def _curie(uri: Any) -> str:
    """`http://example.org/homeont/Kitchen` -> `homeont:Kitchen`."""
    text = str(uri)
    for prefix, namespace in _PREFIXES:
        if text.startswith(namespace):
            return prefix + text[len(namespace):]
    return text


def _children(graph: Graph) -> Dict[str, List[URIRef]]:
    """Superclass IRI -> its direct subclasses, for one pass over the graph."""
    index: Dict[str, List[URIRef]] = {}
    for cls in graph.subjects(RDF.type, OWL.Class):
        for parent in graph.objects(cls, RDFS.subClassOf):
            index.setdefault(str(parent), []).append(cls)
    return index


def _descendants(index: Dict[str, List[URIRef]], root: Any) -> List[URIRef]:
    """Every class under `root`, transitively.

    The property branch is three levels deep -- AirConditionerFanMode is a
    FanControlFanMode is an ActuatableDeviceProperty -- and all three levels
    belong in the context: the parser picks the most specific class the request
    supports, which it can only do if the specific ones are visible.
    """
    seen: set = set()
    stack = [str(root)]
    while stack:
        for child in index.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                stack.append(str(child))
    return sorted(seen, key=str)


def _entry(graph: Graph, cls: URIRef, parent: Optional[Any] = None) -> Dict[str, Any]:
    """One class, as the parser sees it.

    `description` falls back to `rdfs:label` where a class carries no
    `rdfs:comment` -- 19 of them do not, including every device type and three
    of the four environment variables. A label is a poorer description than a
    comment but an honest one; an empty string would just look like a bug.
    """
    comment = graph.value(cls, RDFS.comment)
    label = graph.value(cls, RDFS.label)
    description = str(comment or label or "").replace("\n", " ").strip()

    if parent is None:
        # The immediate superclass, so the context carries the taxonomy rather
        # than flattening it to a root the parser must never answer with.
        parents = sorted(graph.objects(cls, RDFS.subClassOf), key=str)
        parent = parents[0] if parents else None

    entry: Dict[str, Any] = {
        "class": _curie(cls),
        "description": description,
        "parent_class": _curie(parent) if parent is not None else None,
    }

    # Measurement keys appear only where the ontology states them: a missing
    # quantity kind is an absent key, not a null.
    quantity = graph.value(cls, QUDT.hasQuantityKind)
    if quantity is not None:
        entry["measurement_quantity"] = _curie(quantity)
    unit = graph.value(cls, QUDT.unit)
    if unit is not None:
        entry["measurement_unit"] = _curie(unit)
    return entry


def build_capabilities_context(
    ontology_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Project the ontology into the capabilities context. Uncached."""
    graph = Graph()
    graph.parse(str(ontology_path or ONTOLOGY_PATH), format="turtle")
    index = _children(graph)

    device_classes: List[URIRef] = []
    for root in _DEVICE_ROOTS:
        device_classes.extend(index.get(str(root), []))

    observable = _descendants(index, SOSA.ObservableProperty)
    # A property of a space's environment says so with ssn:isPropertyOf; every
    # other observable property belongs to the device that reports it. Marking
    # only the smaller set keeps the ontology free of a class for "the inside
    # of an appliance", which nothing else needs.
    environment = [
        cls for cls in observable
        if (cls, SSN.isPropertyOf, HOMEONT.Environment) in graph
    ]
    measurements = [cls for cls in observable if cls not in set(environment)]

    return {
        "locations": [
            _entry(graph, cls) for cls in _descendants(index, HOMEONT.BuildingSpace)
        ],
        "device_types": [
            _entry(graph, cls) for cls in sorted(device_classes, key=str)
        ],
        "device_properties": {
            "capabilities": [
                _entry(graph, cls)
                for cls in _descendants(index, HOMEONT.DeviceCapabilityProperty)
            ],
            "states": [
                _entry(graph, cls)
                for cls in _descendants(index, HOMEONT.DeviceStateProperty)
            ],
            "actuatable": [
                _entry(graph, cls)
                for cls in _descendants(index, HOMEONT.ActuatableDeviceProperty)
            ],
            "measurements": [_entry(graph, cls) for cls in measurements],
        },
        "environment_variables": [_entry(graph, cls) for cls in environment],
    }


@lru_cache(maxsize=1)
def get_capabilities_context() -> Dict[str, Any]:
    """The capabilities context, parsed once per process.

    Callers must treat the result as read-only; it is shared.
    """
    return build_capabilities_context()


@lru_cache(maxsize=1)
def get_capabilities_context_json() -> str:
    """The context as the indented JSON a prompt embeds."""
    import json

    return json.dumps(get_capabilities_context(), indent=2, ensure_ascii=False)
