"""Ontology namespaces and CURIE shortening for semantic types.

The agents reason over *semantic types* -- `homeont:Kitchen`, `saref:Appliance`,
`sosa:Sensor` -- rather than plain-text device or room names. Those types arrive
from the environment as IRIs and have to be shortened before they go into an LLM
prompt or a matching rule, and the shortening must preserve which vocabulary a
term came from.

An IRI is a structured identifier, so this module treats it as one: terms are
`rdflib.URIRef`, vocabularies are `rdflib.Namespace`, and the prefix mapping is
an `rdflib.namespace.NamespaceManager`. Nothing here decides what a term means by
testing how its string starts. `is_domain_type` compares the term's *namespace*
against the HomeOnt namespace, so it is unaffected by whether the caller happens
to hold a CURIE or a full IRI -- the bug that made it silently answer `False` for
every artifact when `semantic_types` stored full IRIs.

Where the prefixes come from
----------------------------
Every vocabulary the agents use is bound on `NAMESPACE_MANAGER`; the two sources
below only differ in what supplies the binding.

`homeont.ttl` is read at import, so a prefix it declares needs no declaration
here: homeont, saref, td, js, hmas, sosa, ssn, qudt, unit and quantitykind all
arrive from the file (rdflib adds its own default set -- dcterms, foaf and the
rest -- which is harmless). Changing a namespace there changes it here.

The remaining four -- hctl, websub, s4bldg and tdsosa -- are declared below
because nothing we ship states them in a form rdflib can harvest. `hctl.owl`
defines the vocabulary but leaves its own namespace as the unprefixed default,
`saref.rdf` binds no prefix of interest, and `hmas.owl` does not parse under
rdflib at all (owlready2 loads it). The other declarations below duplicate what
`homeont.ttl` already gives and are kept so the set the agents rely on is
readable in one place, and so an unparseable ontology cannot silently remove one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import NamespaceManager

# Anything that can name a term: a URIRef, a full IRI, or an already-short CURIE.
TermLike = Union[URIRef, str, None]

ONTOLOGY_DIR = Path(__file__).resolve().parents[1] / "ontologies"

# The vocabularies the environment serves, as namespace objects rather than
# strings so callers can build terms (`HOMEONT.Kitchen`) and compare membership
# without string surgery.
HOMEONT = Namespace("http://example.org/homeont/")
TD = Namespace("https://www.w3.org/2019/wot/td#")
JS = Namespace("https://www.w3.org/2019/wot/json-schema#")
HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
HMAS = Namespace("https://purl.org/hmas/")
WEBSUB = Namespace("https://purl.org/hmas/websub/")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")
SAREF = Namespace("https://saref.etsi.org/core/")
S4BLDG = Namespace("https://saref.etsi.org/saref4bldg/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
QUANTITYKIND = Namespace("http://qudt.org/vocab/quantitykind/")

# The vocabulary that describes the home itself: rooms, device families, and the
# properties of a room's environment. Matching rules that ask "what kind of room
# is this?" or "what kind of device is this?" filter on this namespace.
DOMAIN_NAMESPACE = HOMEONT

_DECLARED = {
    "homeont": HOMEONT,
    "td": TD,
    "js": JS,
    "hctl": HCTL,
    "hmas": HMAS,
    "websub": WEBSUB,
    "sosa": SOSA,
    "ssn": SSN,
    "tdsosa": TDSOSA,
    "saref": SAREF,
    "s4bldg": S4BLDG,
    "qudt": QUDT,
    "unit": UNIT,
    "quantitykind": QUANTITYKIND,
}


def _build_manager() -> NamespaceManager:
    """Prefix bindings, taken from the ontologies where they state them.

    rdflib binds rdf/rdfs/owl/xsd itself. `override=False` keeps a prefix we
    declared above from being renamed by a vocabulary that binds the same
    namespace under a different name.
    """
    manager = NamespaceManager(Graph())
    for prefix, namespace in _DECLARED.items():
        manager.bind(prefix, namespace, override=True, replace=True)

    homeont_file = ONTOLOGY_DIR / "homeont.ttl"
    if homeont_file.is_file():
        source = Graph()
        try:
            source.parse(str(homeont_file), format="turtle")
        except Exception:
            # A malformed ontology must not stop the agents from starting; the
            # declarations above already cover every vocabulary in use.
            return manager
        for prefix, namespace in source.namespaces():
            if prefix:
                manager.bind(prefix, namespace, override=False, replace=False)
    return manager


NAMESPACE_MANAGER = _build_manager()

# Namespace -> prefix, for deciding whether a term is in a vocabulary we know.
# Built once; `namespaces()` yields (prefix, URIRef) pairs.
_KNOWN = {str(ns): prefix for prefix, ns in NAMESPACE_MANAGER.namespaces()}


def as_term(value: TermLike) -> Optional[URIRef]:
    """A `URIRef` for whatever the caller holds, or None if it is not a term.

    Accepts a `URIRef`, a full IRI, or a CURIE (`homeont:Kitchen`), so callers
    at the edge of the system do not each have to know which form they were
    handed. A CURIE with an unknown prefix stays unresolved and returns None.
    """
    if value is None:
        return None
    if isinstance(value, URIRef):
        return value
    if not isinstance(value, str) or not value:
        return None
    if value.startswith(("http://", "https://", "urn:")):
        return URIRef(value)
    prefix, sep, local = value.partition(":")
    if not sep or not local:
        return None
    namespace = _DECLARED.get(prefix)
    if namespace is None:
        namespace = next(
            (ns for ns, p in _KNOWN.items() if p == prefix), None)
        return URIRef(f"{namespace}{local}") if namespace else None
    return namespace[local]


def split(value: TermLike) -> Optional[tuple]:
    """(prefix, namespace, local_name) for a term in a vocabulary we know.

    None when the term is unknown -- either not a term at all, or in a namespace
    nothing has bound. Callers use that to leave a term visibly unknown instead
    of relabelling it.
    """
    term = as_term(value)
    if term is None:
        return None
    text = str(term)
    for separator in ("#", "/"):
        head, sep, local = text.rpartition(separator)
        if sep and local:
            namespace = head + sep
            prefix = _KNOWN.get(namespace)
            if prefix is not None:
                return prefix, Namespace(namespace), local
            break
    return None


def shorten(value: TermLike) -> str:
    """Term -> CURIE, preserving the vocabulary it came from.

    Returns the input unchanged when the namespace is not one we know, so an
    unknown term is visibly unknown rather than silently relabelled. (rdflib's
    own `normalizeUri` would invent a `ns1:` prefix here, which is exactly the
    mislabelling this avoids.)

        >>> shorten("http://example.org/homeont/Kitchen")
        'homeont:Kitchen'
        >>> shorten("https://saref.etsi.org/core/Appliance")
        'saref:Appliance'
    """
    parts = split(value)
    if parts is None:
        return value if isinstance(value, str) else str(value)
    prefix, _, local = parts
    return f"{prefix}:{local}"


def shorten_all(values: Iterable) -> List[str]:
    """Shorten many terms, dropping empties and duplicates, order preserved."""
    out: List[str] = []
    for value in values or []:
        curie = shorten(value)
        if curie and curie not in out:
            out.append(curie)
    return out


def expand(value: TermLike) -> str:
    """CURIE -> full IRI. Returns the input unchanged if it cannot be resolved."""
    term = as_term(value)
    if term is None:
        return value if isinstance(value, str) else str(value)
    return str(term)


def in_namespace(value: TermLike, namespace: Namespace) -> bool:
    """Is this term defined in `namespace`?

    Compares the term's namespace, so it answers the same for
    `homeont:Kitchen` and `http://example.org/homeont/Kitchen`.
    """
    parts = split(value)
    return parts is not None and str(parts[1]) == str(namespace)


def is_domain_type(value: TermLike) -> bool:
    """Is this a term from the home ontology (a room kind, device family, ...)?

    Use when filtering semantic types down to the ones that describe the home
    rather than the protocol.
    """
    return in_namespace(value, DOMAIN_NAMESPACE)


def domain_types(values: Iterable) -> List[str]:
    """The home-ontology terms from a list of semantic types, as CURIEs.

    Shortens as it filters, so a caller holding full IRIs gets back the CURIEs
    the rest of the system matches on.
    """
    return [shorten(v) for v in (values or []) if is_domain_type(v)]


def local_name(value: TermLike) -> str:
    """The bare local name of a term ('' if there is none)."""
    parts = split(value)
    if parts is not None:
        return parts[2]
    if not value:
        return ""
    # An unknown namespace still has a local name worth reporting.
    text = str(value)
    for separator in ("#", "/"):
        if separator in text:
            text = text.rsplit(separator, 1)[-1]
    return text.rpartition(":")[2] if ":" in text else text
