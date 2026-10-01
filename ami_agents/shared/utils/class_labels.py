"""Human-readable names for ontology classes.

A user asking about the kitchen light is answered about the "On/Off Light", not
about `homeont:OnOffLightOnOff`. The ontology already states how each class is
meant to read, in `rdfs:label` -- "PM10 Mass Concentration", "Compartment
Temperature", "TV" -- so the label is the name, and nothing here derives one
from the identifier.

Distinct from the `description` in the ontology context, which prefers
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
# SAREF names the commands an action performs ("On command") and the device
# families homeont hangs its devices from; homeont's own label wins where both
# state one.
SAREF_PATH = ONTOLOGY_DIR / "saref.rdf"


@lru_cache(maxsize=1)
def _labels() -> dict:
    """Every `rdfs:label` in the home ontology and SAREF, read once per process.

    A plain dict rather than a live graph: this is a lookup table consulted once
    per answered question, and keeping the graph would hold the whole ontology
    for the sake of one predicate.
    """
    table: dict = {}
    for path, fmt in ((HOMEONT_PATH, "turtle"), (SAREF_PATH, "xml")):
        if not path.is_file():
            continue
        graph = Graph()
        try:
            graph.parse(str(path), format=fmt)
        except Exception:
            # A malformed ontology degrades the phrasing of an answer; it must
            # not stop the agent from answering. Callers fall back to the local
            # name.
            continue
        for subject, label in graph.subject_objects(RDFS.label):
            # Several labels may be stated (one per language); the first read
            # wins, and homeont is read first.
            if getattr(label, "language", None) not in (None, "en"):
                continue
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
