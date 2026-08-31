"""Ontology namespaces and CURIE shortening for semantic types.

The agents reason over *semantic types* -- `homeont:Kitchen`, `saref:Appliance`,
`sosa:Sensor` -- rather than plain-text device or room names. Those types arrive
from the environment as full IRIs and have to be shortened before they go into
an LLM prompt or a matching rule, and the shortening must preserve which
vocabulary a term came from.

Previously the shortening was done by taking the local name and stamping a
hardcoded ``ex:`` on it, in eight separate inlined f-strings. That flattened five
distinct vocabularies onto one prefix, so a freezer typed

    homeont:Freezer, sosa:Sensor, hmas:Artifact, saref:Appliance, td:Thing

became ``ex:Freezer, ex:Sensor, ex:Artifact, ex:Appliance, ex:Thing`` -- three of
which name nothing that exists. It also predates HomeOnt: the environment now
serves ``homeont:``, while the agent layer still matched on ``ex:``.

`shorten` resolves against the real namespace and leaves an unknown IRI intact
rather than mislabelling it.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

# The vocabularies the environment serves. Order matters only for readability;
# lookup is by exact namespace match on the IRI prefix.
NAMESPACES: Dict[str, str] = {
    # Home ontology -- building spaces, device families, environment properties.
    # This is the vocabulary the agents primarily reason over.
    "homeont": "http://example.org/homeont/",
    # W3C Web of Things
    "td": "https://www.w3.org/2019/wot/td#",
    "js": "https://www.w3.org/2019/wot/json-schema#",
    "hctl": "https://www.w3.org/2019/wot/hypermedia#",
    # Hypermedia MAS
    "hmas": "https://purl.org/hmas/",
    "websub": "https://purl.org/hmas/websub/",
    # Sensing and actuation
    "sosa": "http://www.w3.org/ns/sosa/",
    "ssn": "http://www.w3.org/ns/ssn/",
    "tdsosa": "https://example.org/hmas/td-sosa-ext#",
    # Devices and buildings
    "saref": "https://saref.etsi.org/core/",
    "s4bldg": "https://saref.etsi.org/saref4bldg/",
    # Quantities and units
    "qudt": "http://qudt.org/schema/qudt/",
    "unit": "http://qudt.org/vocab/unit/",
    "quantitykind": "http://qudt.org/vocab/quantitykind/",
    # Core RDF vocabularies
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

# Longest-first, so `https://purl.org/hmas/websub/` wins over `https://purl.org/hmas/`.
_BY_LENGTH = sorted(NAMESPACES.items(), key=lambda kv: len(kv[1]), reverse=True)

# The vocabulary that describes the home itself: rooms, device families, and the
# properties of a room's environment. Matching rules that ask "what kind of room
# is this?" or "what kind of device is this?" filter on this prefix.
DOMAIN_PREFIX = "homeont"


def shorten(iri: str) -> str:
    """Full IRI -> CURIE, preserving the vocabulary it came from.

    Returns the input unchanged when no known namespace matches, so an unknown
    term is visibly unknown rather than silently relabelled.

        >>> shorten("http://example.org/homeont/Kitchen")
        'homeont:Kitchen'
        >>> shorten("https://saref.etsi.org/core/Appliance")
        'saref:Appliance'
    """
    if not isinstance(iri, str) or not iri:
        return iri
    for prefix, namespace in _BY_LENGTH:
        if iri.startswith(namespace):
            local = iri[len(namespace):]
            if local and "/" not in local and "#" not in local:
                return f"{prefix}:{local}"
    return iri


def shorten_all(iris: Iterable) -> List[str]:
    """Shorten many IRIs, dropping non-strings and duplicates, order preserved."""
    out: List[str] = []
    for iri in iris or []:
        curie = shorten(str(iri))
        if curie and curie not in out:
            out.append(curie)
    return out


def expand(curie: str) -> str:
    """CURIE -> full IRI. Returns the input unchanged if the prefix is unknown."""
    if not isinstance(curie, str) or ":" not in curie:
        return curie
    prefix, _, local = curie.partition(":")
    namespace = NAMESPACES.get(prefix)
    return f"{namespace}{local}" if namespace else curie


def is_domain_type(curie: str) -> bool:
    """Is this a term from the home ontology (a room kind, device family, ...)?

    Use instead of `startswith("ex:")` when filtering semantic types down to the
    ones that describe the home rather than the protocol.
    """
    return isinstance(curie, str) and curie.startswith(f"{DOMAIN_PREFIX}:")


def domain_types(curies: Iterable) -> List[str]:
    """Keep only the home-ontology terms from a list of semantic types."""
    return [c for c in (curies or []) if is_domain_type(c)]


def local_name(curie_or_iri: Optional[str]) -> str:
    """The bare local name, from either a CURIE or a full IRI ('' if empty)."""
    if not curie_or_iri:
        return ""
    text = str(curie_or_iri)
    for sep in ("#", "/"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text.rpartition(":")[2] if ":" in text else text
