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

from rdflib import RDFS

from .namespaces import expand, local_name
from .vocabulary import homeont, saref


@lru_cache(maxsize=1)
def _labels() -> dict:
    """Every `rdfs:label` in the home ontology and SAREF, built once per process.

    SAREF names the commands an action performs ("On command") and the device
    families homeont hangs its devices from; homeont is read first, so its own
    label wins where both state one. A missing or malformed ontology yields no
    labels, and callers fall back to the local name.
    """
    table: dict = {}
    for graph in (homeont(), saref()):
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
