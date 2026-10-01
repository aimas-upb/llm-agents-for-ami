"""The ontology context: `homeont.ttl` projected into structured JSON.

The intent parsers need the home vocabulary, not the home ontology. Handing an
LLM the raw Turtle means ~17k tokens of Matter provenance, object properties and
design commentary it cannot act on, and leaves it to infer the taxonomy from
scattered `rdfs:subClassOf` lines.

This module answers the one question a parser asks -- *which classes may I
choose from, and what does each mean* -- as a deterministic projection. Every
section is a tree keyed by its root class, so choosing "the most specific class
the request supports" is a walk down the tree rather than a reconstruction of
it from parent pointers:

    locations              homeont:BuildingSpace     where a request is scoped
    device_types           saref:Device              what kind of thing it targets
    device_properties      homeont:ActuatableDeviceProperty   what can be changed
                           homeont:DeviceStateProperty        what condition it is in
                           homeont:DeviceCapabilityProperty   what it supports
                           sosa:ObservableProperty            what it measures about itself
    environment_variables  sosa:ObservableProperty   what a space's environment exposes

The capabilities view adds two sections the ENV_STATE parser has no slot for:

    commands               saref:Command             what an action does
    device_metadata        schema:manufacturer, schema:model

No LLM call, no network, no discovered-environment dependency: the same
ontologies the graph is typed against, rendered for a prompt.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

HOMEONT = Namespace("http://example.org/homeont/")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SAREF = Namespace("https://saref.etsi.org/core/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
SSN_SYSTEM = Namespace("http://www.w3.org/ns/ssn/systems/")
QUDT = Namespace("http://qudt.org/schema/qudt/")

# .../ami_agents/agents/user_assistant/utils/ -> .../ami_agents/
ONTOLOGY_DIR = Path(__file__).resolve().parents[3] / "shared" / "ontologies"
ONTOLOGY_PATH = ONTOLOGY_DIR / "homeont.ttl"
# SAREF supplies the device families' and the commands' own definitions, and
# the command classes homeont does not redefine (SetAbsoluteLevelCommand).
SAREF_PATH = ONTOLOGY_DIR / "saref.rdf"

# Prefixes the context renders identifiers with. The parser quotes these back
# verbatim and the resolver puts them straight into SPARQL, so they must match
# the prefixes the served graphs use.
_PREFIXES = (
    ("homeont:", str(HOMEONT)),
    ("sosa:", str(SOSA)),
    ("saref:", str(SAREF)),
    ("ssn-system:", str(SSN_SYSTEM)),
    ("ssn:", str(SSN)),
    ("schema:", "https://schema.org/"),
    ("quantitykind:", "http://qudt.org/vocab/quantitykind/"),
    ("unit:", "http://qudt.org/vocab/unit/"),
)

# The SAREF classes homeont hangs its device families from. SAREF defines
# exactly these five subclasses of saref:Device.
_DEVICE_ROOTS = (SAREF.HVAC, SAREF.Actuator, SAREF.Appliance,
                 SAREF.Sensor, SAREF.Meter)

# SOSA is not vendored, so its root carries the W3C definition verbatim.
_SOSA_OBSERVABLE_PROPERTY = (
    "An observable quality (property, characteristic) of a FeatureOfInterest."
)

# Facts stated on the Thing rather than served as affordances (see the
# "Matter BasicInformation" block in homeont.ttl). Not classes: the parser
# names one of these as a device property to ask about the device itself.
DEVICE_METADATA = (
    {"property": "schema:manufacturer",
     "description": "The company that made the device."},
    {"property": "schema:model",
     "description": "The manufacturer's product name for the device."},
)


def _curie(uri: Any) -> str:
    """`http://example.org/homeont/Kitchen` -> `homeont:Kitchen`."""
    text = str(uri)
    for prefix, namespace in _PREFIXES:
        if text.startswith(namespace):
            return prefix + text[len(namespace):]
    return text


def _parents(graph: Graph, cls: URIRef) -> List[URIRef]:
    """Named superclasses, sorted. OWL restrictions (blank nodes) are not."""
    return sorted((p for p in graph.objects(cls, RDFS.subClassOf)
                   if isinstance(p, URIRef)), key=str)


def _children(graph: Graph) -> Dict[str, List[URIRef]]:
    """Primary parent IRI -> its subclasses, for one pass over the graph.

    A class with several parents is filed under the first (sorted) only, so the
    tree never shows it twice; `_node` records the others.
    """
    index: Dict[str, List[URIRef]] = {}
    for cls in graph.subjects(RDF.type, OWL.Class):
        if not isinstance(cls, URIRef):
            continue
        parents = _parents(graph, cls)
        if parents:
            index.setdefault(str(parents[0]), []).append(cls)
    for children in index.values():
        children.sort(key=str)
    return index


def _description(graph: Graph, cls: URIRef) -> str:
    """`rdfs:comment`, falling back to `rdfs:label`.

    19 homeont classes carry no comment, including every device type and three
    of the four environment variables. A label is a poorer description than a
    comment but an honest one; an empty string would just look like a bug.
    """
    comment = graph.value(cls, RDFS.comment)
    label = graph.value(cls, RDFS.label)
    return str(comment or label or "").replace("\n", " ").strip()


def _node(graph: Graph, cls: URIRef) -> Dict[str, Any]:
    """One class, as the parser sees it -- without its subclasses."""
    node: Dict[str, Any] = {
        "class": _curie(cls),
        "description": _description(graph, cls),
    }
    # Measurement keys appear only where the ontology states them: a missing
    # quantity kind is an absent key, not a null.
    quantity = graph.value(cls, QUDT.hasQuantityKind)
    if quantity is not None:
        node["measurement_quantity"] = _curie(quantity)
    unit = graph.value(cls, QUDT.unit)
    if unit is not None:
        node["measurement_unit"] = _curie(unit)
    others = _parents(graph, cls)[1:]
    if others:
        node["also_subclass_of"] = [_curie(p) for p in others]
    return node


def _subtree(graph: Graph, index: Dict[str, List[URIRef]], cls: URIRef,
             keep: Callable[[URIRef], bool]) -> List[Dict[str, Any]]:
    """The kept subclasses of `cls`, each with its own subtree.

    A class `keep` rejects is spliced out rather than pruned: its kept
    descendants attach to the nearest kept ancestor, so filtering a section
    never silently loses a class below the filtered one.
    """
    nodes: List[Dict[str, Any]] = []
    for child in index.get(str(cls), []):
        below = _subtree(graph, index, child, keep)
        if not keep(child):
            nodes.extend(below)
            continue
        node = _node(graph, child)
        if below:
            node["subclasses"] = below
        nodes.append(node)
    return nodes


def _section(graph: Graph, index: Dict[str, List[URIRef]], root: URIRef,
             keep: Callable[[URIRef], bool] = lambda cls: True,
             description: Optional[str] = None) -> Dict[str, Any]:
    """`{root: {description, subclasses}}` -- one tree, keyed by its root."""
    node = _node(graph, root)
    node.pop("class")
    # What the root itself specialises (saref:Device is an s4syst:System) is
    # outside the section and nothing the parser may answer with.
    node.pop("also_subclass_of", None)
    if description is not None:
        node["description"] = description
    node["subclasses"] = _subtree(graph, index, root, keep)
    return {_curie(root): node}


def _load(ontology_path: Optional[Path]) -> Graph:
    graph = Graph()
    graph.parse(str(ontology_path or ONTOLOGY_PATH), format="turtle")
    if SAREF_PATH.exists():
        graph.parse(str(SAREF_PATH), format="xml")
    return graph


def build_ontology_context(
    ontology_path: Optional[Path] = None,
    view: str = "state",
) -> Dict[str, Any]:
    """Project the ontologies into the context for one parser. Uncached.

    `view="state"` is what the ENV_STATE parser reads; `view="capabilities"`
    adds `commands` and `device_metadata`.
    """
    if view not in ("state", "capabilities"):
        raise ValueError(f"unknown ontology context view: {view!r}")

    graph = _load(ontology_path)
    index = _children(graph)

    def is_environment(cls: URIRef) -> bool:
        # A property of a space's environment says so with ssn:isPropertyOf;
        # every other observable property belongs to the device reporting it.
        return (cls, SSN.isPropertyOf, HOMEONT.Environment) in graph

    def is_device_type(cls: URIRef) -> bool:
        # The five SAREF families, and homeont's own classes beneath them.
        # SAREF's other devices (saref:Switch, saref:SmokeSensor) type no Thing
        # in any served graph, and offering them would invite answers that
        # match nothing.
        return cls in _DEVICE_ROOTS or str(cls).startswith(str(HOMEONT))

    device_types = _section(graph, index, SAREF.Device, keep=is_device_type)

    context: Dict[str, Any] = {
        "locations": _section(graph, index, HOMEONT.BuildingSpace),
        "device_types": device_types,
        "device_properties": {
            **_section(graph, index, HOMEONT.ActuatableDeviceProperty),
            **_section(graph, index, HOMEONT.DeviceStateProperty),
            **_section(graph, index, HOMEONT.DeviceCapabilityProperty),
            **_section(graph, index, SOSA.ObservableProperty,
                       keep=lambda cls: not is_environment(cls),
                       description=_SOSA_OBSERVABLE_PROPERTY),
        },
        "environment_variables": _section(
            graph, index, SOSA.ObservableProperty, keep=is_environment,
            description=_SOSA_OBSERVABLE_PROPERTY),
    }
    if view == "capabilities":
        context["commands"] = _section(graph, index, SAREF.Command)
        context["device_metadata"] = [dict(entry) for entry in DEVICE_METADATA]
    return context


def iter_classes(context: Dict[str, Any]) -> Iterator[Tuple[str, Optional[str], Dict[str, Any]]]:
    """Every class in a context, as `(class, parent_class, node)`.

    Roots come first with a None parent. `sosa:ObservableProperty` roots two
    sections (device measurements, environment variables) and is yielded once
    per section. `device_metadata` is skipped: those are properties stated on
    a Thing, not classes.
    """
    def walk(nodes: List[Dict[str, Any]], parent: str):
        for node in nodes:
            yield node["class"], parent, node
            yield from walk(node.get("subclasses") or [], node["class"])

    def sections():
        for key in ("locations", "device_types", "environment_variables",
                    "commands"):
            if key in context:
                yield context[key]
        yield context["device_properties"]

    for section in sections():
        for root, node in section.items():
            yield root, None, node
            yield from walk(node.get("subclasses") or [], root)


@lru_cache(maxsize=1)
def get_ontology_context() -> Dict[str, Any]:
    """The ENV_STATE context, parsed once per process.

    Callers must treat the result as read-only; it is shared.
    """
    return build_ontology_context()


@lru_cache(maxsize=1)
def get_capability_ontology_context() -> Dict[str, Any]:
    """The ENV_CAPABILITIES context, parsed once per process. Read-only."""
    return build_ontology_context(view="capabilities")


@lru_cache(maxsize=1)
def get_ontology_context_json() -> str:
    """The ENV_STATE context as the indented JSON a prompt embeds."""
    import json

    return json.dumps(get_ontology_context(), indent=2, ensure_ascii=False)


@lru_cache(maxsize=1)
def get_capability_ontology_context_json() -> str:
    """The ENV_CAPABILITIES context as the indented JSON a prompt embeds."""
    import json

    return json.dumps(get_capability_ontology_context(), indent=2,
                      ensure_ascii=False)
