"""Semantic terms are identified by namespace, not by how their string starts.

The module these tests cover replaced string prefix tests with rdflib namespace
comparison. The bug that motivated it: `is_domain_type` asked whether a string
started with `"homeont:"`, while `semantic_types` stored full IRIs, so it
answered False for every artifact and the legacy type filter rejected them all.
"""

from rdflib import URIRef

from ami_agents.shared.utils.namespaces import (
    HMAS,
    HOMEONT,
    QUANTITYKIND,
    SAREF,
    SOSA,
    SSN,
    SSN_SYSTEM,
    TD,
    UNIT,
    WEBSUB,
    domain_types,
    expand,
    in_namespace,
    is_domain_type,
    local_name,
    shorten,
    shorten_all,
)


class TestShorten:
    def test_known_vocabularies(self):
        assert shorten(f"{HOMEONT}Kitchen") == "homeont:Kitchen"
        assert shorten(f"{SAREF}Appliance") == "saref:Appliance"
        assert shorten(f"{TD}Thing") == "td:Thing"
        assert shorten(f"{SOSA}Sensor") == "sosa:Sensor"

    def test_longest_namespace_wins(self):
        """`websub/` is nested under `hmas/`; the more specific one must win."""
        assert shorten(
            f"{WEBSUB}WebSubSubscription"
        ) == "websub:WebSubSubscription"
        assert shorten(f"{HMAS}Artifact") == "hmas:Artifact"

    def test_unknown_namespace_is_left_intact(self):
        """rdflib's normalizeUri would invent `ns1:` here and mislabel it."""
        unknown = "http://nowhere.example/Foo"
        assert shorten(unknown) == unknown

    def test_accepts_uriref_and_curie(self):
        assert shorten(URIRef(f"{HOMEONT}Kitchen")) == "homeont:Kitchen"
        assert shorten("homeont:Kitchen") == "homeont:Kitchen"

    def test_shorten_all_dedupes_preserving_order(self):
        assert shorten_all([
            f"{HOMEONT}Freezer",
            f"{SAREF}Appliance",
            f"{HOMEONT}Freezer",
        ]) == ["homeont:Freezer", "saref:Appliance"]


class TestIsDomainType:
    def test_same_answer_for_iri_and_curie(self):
        """The regression: an IRI used to answer False."""
        assert is_domain_type(f"{HOMEONT}AirConditioner")
        assert is_domain_type("homeont:AirConditioner")
        assert is_domain_type(URIRef(f"{HOMEONT}AirConditioner"))

    def test_other_vocabularies_are_not_domain_types(self):
        assert not is_domain_type(f"{SAREF}HVAC")
        assert not is_domain_type("td:Thing")
        assert not is_domain_type("http://nowhere.example/Foo")

    def test_non_terms(self):
        assert not is_domain_type(None)
        assert not is_domain_type("")

    def test_in_namespace_is_general(self):
        assert in_namespace("saref:HVAC", SAREF)
        assert in_namespace(f"{HOMEONT}Kitchen", HOMEONT)
        assert not in_namespace("saref:HVAC", HOMEONT)


class TestDomainTypes:
    def test_filters_and_shortens_raw_iris(self):
        """Used to return [] for exactly this input."""
        assert domain_types([
            f"{HOMEONT}AirConditioner",
            f"{SAREF}HVAC",
            f"{TD}Thing",
        ]) == ["homeont:AirConditioner"]

    def test_accepts_curies_too(self):
        assert domain_types(
            ["homeont:Freezer", "saref:Appliance"]) == ["homeont:Freezer"]

    def test_empty_input(self):
        assert domain_types([]) == []
        assert domain_types(None) == []


class TestRoundTrip:
    def test_expand_and_local_name(self):
        assert expand("homeont:Kitchen") == f"{HOMEONT}Kitchen"
        assert local_name("homeont:Kitchen") == "Kitchen"
        assert local_name(f"{HOMEONT}Kitchen") == "Kitchen"
        assert local_name(f"{TD}Thing") == "Thing"

    def test_expand_leaves_unknown_prefix_alone(self):
        assert expand("nosuch:Thing") == "nosuch:Thing"

    def test_local_name_of_empty(self):
        assert local_name(None) == ""
        assert local_name("") == ""


class TestOntologyDerivedBindings:
    def test_prefixes_declared_by_homeont_are_bound(self):
        """homeont.ttl is authoritative for the vocabularies it binds."""
        assert shorten(f"{UNIT}LUX") == "unit:LUX"
        assert shorten(
            f"{QUANTITYKIND}Illuminance"
        ) == "quantitykind:Illuminance"

    def test_a_nested_namespace_does_not_shadow_its_parent(self):
        """`ssn-system:` sits under `ssn:`; both are bound and stay distinct."""
        assert shorten(f"{SSN}isPropertyOf") == "ssn:isPropertyOf"
        assert shorten(
            f"{SSN_SYSTEM}SystemCapability"
        ) == "ssn-system:SystemCapability"


# --------------------------------------------------------------------------
# One source of IRIs
# --------------------------------------------------------------------------
#
# `namespaces.py` is the only place the agents spell out a vocabulary IRI. The
# two adapters cannot import it and keep one copy each; these tests fail when a
# copy drifts, when a prefix helper misrenders, or when an IRI is hard-coded
# again anywhere else.

import importlib.util
import io
import re
import tokenize
from pathlib import Path

import pytest
from rdflib import Graph, Namespace

from ami_agents.shared.utils import namespaces as central
from ami_agents.shared.utils.namespaces import sparql_prefixes, turtle_prefixes

REPO = Path(__file__).resolve().parents[2]
INTEGRATION = REPO / "ami_agents" / "environment" / "integration"
ADAPTER_COPIES = [
    INTEGRATION / "HomeAssistant" / "hasp_namespaces.py",
    INTEGRATION / "SimuHome" / "shtd_namespaces.py",
]


def _load(path: Path):
    """The module at `path`, loaded by file so the adapter's directory need not
    be importable (it is not a package the tests can reach by name)."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAdapterCopies:
    @pytest.mark.parametrize("path", ADAPTER_COPIES, ids=lambda p: p.name)
    def test_every_namespace_equals_the_central_one(self, path):
        """Each namespace an adapter serves has the same name and IRI centrally."""
        # 1. Every Namespace the adapter module defines.
        copy = {name: value for name, value in vars(_load(path)).items()
                if isinstance(value, Namespace)}
        assert copy, f"{path.name} defines no namespaces"

        # 2. Each must exist in namespaces.py under the same name, same IRI.
        for name, value in copy.items():
            assert hasattr(central, name), f"{path.name}: {name} unknown centrally"
            assert str(value) == str(getattr(central, name)), \
                f"{path.name}: {name} = {value}, central = {getattr(central, name)}"


class TestPrefixLines:
    def test_sparql_prefixes_render_the_central_iris(self):
        assert sparql_prefixes("homeont", "rdfs") == (
            f"PREFIX homeont: <{HOMEONT}>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n")

    def test_a_query_built_from_them_runs(self):
        """The prefix lines are valid SPARQL: a query using them parses and matches."""
        graph = Graph()
        graph.parse(data=turtle_prefixes("homeont", "saref", "rdfs")
                    + "homeont:Kitchen rdfs:label \"Kitchen\" .", format="turtle")
        rows = list(graph.query(sparql_prefixes("homeont", "rdfs")
                                + "SELECT ?l WHERE { homeont:Kitchen rdfs:label ?l }"))
        assert [str(r.l) for r in rows] == ["Kitchen"]

    def test_an_unknown_prefix_is_an_error_not_a_guess(self):
        with pytest.raises(KeyError):
            sparql_prefixes("homeonte")


# Our vocabularies' IRIs. A string literal containing one, outside the three
# namespace modules, is a copy that will drift.
_VOCABULARY_IRI = re.compile("|".join(re.escape(str(ns)) for ns in (
    central.HOMEONT, central.TDSOSA, central.TD, central.JS, central.HCTL,
    central.WOTSEC, central.HTV, central.HMAS, central.SOSA, central.SSN,
    central.SAREF, central.QUDT, central.UNIT, central.QUANTITYKIND, central.SCHEMA)))

# Where the agents' code and the adapters live, and what is exempt: the three
# namespace modules themselves, and the legacy pre-HASP adapter (out of scope).
_SCANNED = [REPO / "ami_agents" / "agents", REPO / "ami_agents" / "shared",
            INTEGRATION / "integration_engine.py",
            INTEGRATION / "HomeAssistant", INTEGRATION / "SimuHome"]
_EXEMPT = {"namespaces.py", "hasp_namespaces.py", "shtd_namespaces.py",
           "ygg_ha_adapter.py", "ha_utils.py"}


def _hardcoded_iris(path: Path):
    """(line, text) of every non-docstring string literal in `path` that spells
    out one of our vocabulary IRIs. Comments and docstrings may name an IRI."""
    tokens = list(tokenize.generate_tokens(io.StringIO(path.read_text()).readline))
    skip = (tokenize.NL, tokenize.COMMENT)
    for i, token in enumerate(tokens):
        # 1. String literals only (f-string parts included), naming our IRIs.
        if token.type not in (tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)):
            continue
        if not _VOCABULARY_IRI.search(token.string):
            continue
        # 2. A docstring is a string that is a statement on its own.
        before = next((t for t in reversed(tokens[:i]) if t.type not in skip), None)
        after = next((t for t in tokens[i + 1:] if t.type not in skip), None)
        # (A module docstring has no token before it at all.)
        if ((before is None or before.type in (tokenize.INDENT, tokenize.DEDENT,
                                               tokenize.NEWLINE, tokenize.ENCODING))
                and after is not None and after.type == tokenize.NEWLINE):
            continue
        yield token.start[0], token.string.strip()[:80]


class TestNoHardcodedIris:
    def test_no_module_spells_out_a_vocabulary_iri(self):
        """Every IRI comes from a namespace module, so changing one is one edit."""
        # 1. Collect offending literals across the agents and the adapters.
        offenders = []
        for root in _SCANNED:
            files = [root] if root.is_file() else sorted(root.rglob("*.py"))
            for path in files:
                if path.name in _EXEMPT or "tests" in path.parts:
                    continue
                offenders += [f"{path.relative_to(REPO)}:{line}: {text}"
                              for line, text in _hardcoded_iris(path)]
        # 2. None may remain.
        assert not offenders, "hard-coded IRIs:\n" + "\n".join(offenders)
