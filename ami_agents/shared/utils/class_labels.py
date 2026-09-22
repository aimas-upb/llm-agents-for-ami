"""Human-readable names for ontology classes.

A user asking about the kitchen light is answered about the "On/Off Light", not
about `homeont:OnOffLightOnOff`. The ontology already states how each class is
meant to read, in `rdfs:label` -- "PM10 Mass Concentration", "Compartment
Temperature", "TV" -- so the label is the name, and nothing here derives one
from the identifier.

Distinct from the `description` in the capabilities context, which prefers
`rdfs:comment` and is therefore a sentence ("A Kitchen is a kind of
BuildingSpace: a room used for cooking...") rather than a name.

Lives beside `namespaces` and shares its ontology directory: naming a class is
not the business of any one agent.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from rdflib import Graph, RDFS, URIRef

from .namespaces import ONTOLOGY_DIR, expand, local_name

HOMEONT_PATH = ONTOLOGY_DIR / "homeont.ttl"


@lru_cache(maxsize=1)
def _labels() -> dict:
    """Every `rdfs:label` in the home ontology, read once per process.

    A plain dict rather than a live graph: this is a lookup table consulted once
    per answered question, and keeping the graph would hold the whole ontology
    for the sake of one predicate.
    """
    table: dict = {}
    if not HOMEONT_PATH.is_file():
        return table
    graph = Graph()
    try:
        graph.parse(str(HOMEONT_PATH), format="turtle")
    except Exception:
        # A malformed ontology degrades the phrasing of an answer; it must not
        # stop the agent from answering. Callers fall back to the local name.
        return table
    for subject, label in graph.subject_objects(RDFS.label):
        table.setdefault(str(subject), str(label))
    return table


def label_for(identifier: Optional[str]) -> str:
    """The `rdfs:label` of a class, or its local name if it states none.

        >>> label_for("homeont:OnOffLight")
        'On/Off Light'
        >>> label_for("homeont:Pm10MassConcentration")
        'PM10 Mass Concentration'

    An empty identifier gives an empty string, so a caller can interpolate a
    slot the request never filled without guarding every use.
    """
    if not identifier:
        return ""
    label = _labels().get(expand(identifier))
    return label if label else local_name(identifier)
