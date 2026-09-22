"""Semantic terms are identified by namespace, not by how their string starts.

The module these tests cover replaced string prefix tests with rdflib namespace
comparison. The bug that motivated it: `is_domain_type` asked whether a string
started with `"homeont:"`, while `semantic_types` stored full IRIs, so it
answered False for every artifact and the legacy type filter rejected them all.
"""

from rdflib import URIRef

from ami_agents.shared.utils.namespaces import (
    HOMEONT,
    SAREF,
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
        assert shorten("http://example.org/homeont/Kitchen") == "homeont:Kitchen"
        assert shorten("https://saref.etsi.org/core/Appliance") == "saref:Appliance"
        assert shorten("https://www.w3.org/2019/wot/td#Thing") == "td:Thing"
        assert shorten("http://www.w3.org/ns/sosa/Sensor") == "sosa:Sensor"

    def test_longest_namespace_wins(self):
        """`websub/` is nested under `hmas/`; the more specific one must win."""
        assert shorten(
            "https://purl.org/hmas/websub/WebSubSubscription"
        ) == "websub:WebSubSubscription"
        assert shorten("https://purl.org/hmas/Artifact") == "hmas:Artifact"

    def test_unknown_namespace_is_left_intact(self):
        """rdflib's normalizeUri would invent `ns1:` here and mislabel it."""
        unknown = "http://nowhere.example/Foo"
        assert shorten(unknown) == unknown

    def test_accepts_uriref_and_curie(self):
        assert shorten(URIRef("http://example.org/homeont/Kitchen")) == "homeont:Kitchen"
        assert shorten("homeont:Kitchen") == "homeont:Kitchen"

    def test_shorten_all_dedupes_preserving_order(self):
        assert shorten_all([
            "http://example.org/homeont/Freezer",
            "https://saref.etsi.org/core/Appliance",
            "http://example.org/homeont/Freezer",
        ]) == ["homeont:Freezer", "saref:Appliance"]


class TestIsDomainType:
    def test_same_answer_for_iri_and_curie(self):
        """The regression: an IRI used to answer False."""
        assert is_domain_type("http://example.org/homeont/AirConditioner")
        assert is_domain_type("homeont:AirConditioner")
        assert is_domain_type(URIRef("http://example.org/homeont/AirConditioner"))

    def test_other_vocabularies_are_not_domain_types(self):
        assert not is_domain_type("https://saref.etsi.org/core/HVAC")
        assert not is_domain_type("td:Thing")
        assert not is_domain_type("http://nowhere.example/Foo")

    def test_non_terms(self):
        assert not is_domain_type(None)
        assert not is_domain_type("")

    def test_in_namespace_is_general(self):
        assert in_namespace("saref:HVAC", SAREF)
        assert in_namespace("http://example.org/homeont/Kitchen", HOMEONT)
        assert not in_namespace("saref:HVAC", HOMEONT)


class TestDomainTypes:
    def test_filters_and_shortens_raw_iris(self):
        """Used to return [] for exactly this input."""
        assert domain_types([
            "http://example.org/homeont/AirConditioner",
            "https://saref.etsi.org/core/HVAC",
            "https://www.w3.org/2019/wot/td#Thing",
        ]) == ["homeont:AirConditioner"]

    def test_accepts_curies_too(self):
        assert domain_types(
            ["homeont:Freezer", "saref:Appliance"]) == ["homeont:Freezer"]

    def test_empty_input(self):
        assert domain_types([]) == []
        assert domain_types(None) == []


class TestRoundTrip:
    def test_expand_and_local_name(self):
        assert expand("homeont:Kitchen") == "http://example.org/homeont/Kitchen"
        assert local_name("homeont:Kitchen") == "Kitchen"
        assert local_name("http://example.org/homeont/Kitchen") == "Kitchen"
        assert local_name("https://www.w3.org/2019/wot/td#Thing") == "Thing"

    def test_expand_leaves_unknown_prefix_alone(self):
        assert expand("nosuch:Thing") == "nosuch:Thing"

    def test_local_name_of_empty(self):
        assert local_name(None) == ""
        assert local_name("") == ""


class TestOntologyDerivedBindings:
    def test_prefixes_declared_by_homeont_are_bound(self):
        """homeont.ttl is authoritative for the vocabularies it binds."""
        assert shorten("http://qudt.org/vocab/unit/LUX") == "unit:LUX"
        assert shorten(
            "http://qudt.org/vocab/quantitykind/Illuminance"
        ) == "quantitykind:Illuminance"

    def test_a_nested_namespace_does_not_shadow_its_parent(self):
        """`ssn-system:` sits under `ssn:`; both are bound and stay distinct."""
        assert shorten("http://www.w3.org/ns/ssn/isPropertyOf") == "ssn:isPropertyOf"
        assert shorten(
            "http://www.w3.org/ns/ssn/systems/SystemCapability"
        ) == "ssn-system:SystemCapability"
