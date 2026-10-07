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

from rdflib import Graph

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
