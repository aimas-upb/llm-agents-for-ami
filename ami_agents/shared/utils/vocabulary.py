"""The home vocabulary -- homeont plus SAREF -- parsed once per process.

Every consumer of the vocabulary reads it from here: the prefix bindings
(`namespaces`), the class labels (`class_labels`), the parsers' class trees
(`user_assistant/utils/ontology_context`) and the graph the EnvExplorer types
its SPARQL against (`env_explorer/utils/state_resolution.load_vocabulary`).
One file, one parse, many projections.

The graphs returned are shared. Treat them as read-only: a caller that needs to
add triples copies them into a graph of its own (`graph += vocabulary()`).

Not to be confused with `shared/ontologies/loader.py`, which loads the HMAS, TD
and protocol ontologies with owlready2 -- a different library, and one that
cannot read Turtle.

A file that is missing or malformed is skipped with a warning rather than
raised: a broken ontology degrades labels and class trees, and must not stop an
agent from starting.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Set

from rdflib import BNode, Graph, URIRef
from rdflib.namespace import RDF, RDFS

logger = logging.getLogger(__name__)

# .../ami_agents/shared/utils/ -> .../ami_agents/shared/ontologies/
ONTOLOGY_DIR = Path(__file__).resolve().parents[1] / "ontologies"
HOMEONT_PATH = ONTOLOGY_DIR / "homeont.ttl"
# SAREF supplies the device families' parents, the command classes homeont does
# not redefine (SetAbsoluteLevelCommand) and their definitions.
SAREF_PATH = ONTOLOGY_DIR / "saref.rdf"


def parse_into(graph: Graph, path: Path, fmt: str) -> Graph:
    """Parse `path` into `graph`, skipping it if it is missing or malformed."""
    if not path.is_file():
        logger.warning("Ontology file not found: %s", path)
        return graph
    try:
        graph.parse(str(path), format=fmt)
    except Exception as exc:
        logger.warning("Failed to parse ontology %s: %s", path, exc)
    return graph


@lru_cache(maxsize=1)
def homeont() -> Graph:
    """The home ontology alone."""
    return parse_into(Graph(), HOMEONT_PATH, "turtle")


@lru_cache(maxsize=1)
def saref() -> Graph:
    """SAREF core alone."""
    return parse_into(Graph(), SAREF_PATH, "xml")


@lru_cache(maxsize=1)
def vocabulary() -> Graph:
    """homeont and SAREF together.

    Both are needed for `rdfs:subClassOf*` to close: SAREF defines
    `saref:Device` and the five classes every homeont device family descends
    from, so without it the path from `homeont:AirConditioner` up to
    `saref:Device` stops short.
    """
    graph = Graph()
    graph += homeont()
    graph += saref()
    for prefix, namespace in homeont().namespaces():
        graph.bind(prefix, namespace, override=False)
    return graph


# --------------------------------------------------------------------------
# Class roots
# --------------------------------------------------------------------------
#
# The top of every class tree an agent chooses from. The parsers' class trees
# (`user_assistant/utils/ontology_context.py`) are cut at these, so no class
# above them ever reaches a structured goal; the type closure below stops at
# them for the same reason. Plain IRIs: `namespaces` imports this module.

_HOMEONT = "http://example.org/homeont/"
_SAREF = "https://saref.etsi.org/core/"
_SOSA = "http://www.w3.org/ns/sosa/"

BUILDING_SPACE_ROOT = URIRef(_HOMEONT + "BuildingSpace")           # rooms
DEVICE_ROOT = URIRef(_SAREF + "Device")                            # devices
COMMAND_ROOT = URIRef(_SAREF + "Command")                          # what an action does
ACTUATABLE_PROPERTY_ROOT = URIRef(_HOMEONT + "ActuatableDeviceProperty")
STATE_PROPERTY_ROOT = URIRef(_HOMEONT + "DeviceStateProperty")
CAPABILITY_PROPERTY_ROOT = URIRef(_HOMEONT + "DeviceCapabilityProperty")
OBSERVABLE_PROPERTY_ROOT = URIRef(_SOSA + "ObservableProperty")    # measurements, environment

CLASS_ROOTS: FrozenSet[URIRef] = frozenset({
    BUILDING_SPACE_ROOT, DEVICE_ROOT, COMMAND_ROOT,
    ACTUATABLE_PROPERTY_ROOT, STATE_PROPERTY_ROOT, CAPABILITY_PROPERTY_ROOT,
    OBSERVABLE_PROPERTY_ROOT,
})


# --------------------------------------------------------------------------
# Type closure, bounded at the roots
# --------------------------------------------------------------------------

def _ancestors_to_roots(cls: URIRef, vocab: Graph, roots: FrozenSet[URIRef],
                        cache: Dict[URIRef, Set[URIRef]],
                        visiting: FrozenSet[URIRef] = frozenset()) -> Set[URIRef]:
    """The named classes on `cls`'s `rdfs:subClassOf` chains that lead to a root,
    `cls` and the root included -- or the empty set if no chain does.

    A chain stops at the first root it reaches: what lies above a root
    (`saref:Device rdfs:subClassOf s4syst:System`) is never added. Anonymous
    superclasses -- SAREF's `owl:Restriction` nodes -- are not followed, and
    a chain that never reaches a root contributes nothing.
    """
    if cls in cache:
        return cache[cls]
    if cls in roots:
        cache[cls] = {cls}
        return cache[cls]
    found: Set[URIRef] = set()
    for parent in vocab.objects(cls, RDFS.subClassOf):
        if isinstance(parent, BNode) or parent in visiting:
            continue
        found |= _ancestors_to_roots(parent, vocab, roots, cache, visiting | {cls})
    result = found | {cls} if found else set()
    cache[cls] = result
    return result


def close_types(graph: Graph, vocab: Graph = None,
                roots: Iterable[URIRef] = CLASS_ROOTS) -> int:
    """Add, for every typed node, the superclasses of its types up to the roots.

    Materialised RDFS subclass inference (rdfs9 with rdfs11's transitivity),
    restricted to the class trees agents choose from: afterwards a device typed
    `homeont:ColorLight` is also plainly `a saref:Actuator` and `a saref:Device`,
    so a query filters on whatever class it was given with a plain
    `?x a ?class` -- no property paths, no duplicate rows. Nothing else is
    inferred: no domain, range or subproperty rules.

    Returns the number of triples added.
    """
    vocab = vocab if vocab is not None else vocabulary()
    roots = frozenset(roots)
    cache: Dict[URIRef, Set[URIRef]] = {}
    added = 0
    for node, cls in list(graph.subject_objects(RDF.type)):
        if not isinstance(cls, URIRef):
            continue
        for ancestor in _ancestors_to_roots(cls, vocab, roots, cache):
            if (node, RDF.type, ancestor) not in graph:
                graph.add((node, RDF.type, ancestor))
                added += 1
    return added
