"""Subclass inference bounded at the class roots (`vocabulary.close_types`)."""

from rdflib import BNode, Graph, URIRef
from rdflib.namespace import RDF, RDFS

from ami_agents.shared.utils.vocabulary import (
    CLASS_ROOTS,
    DEVICE_ROOT,
    close_types,
    vocabulary,
)

HOMEONT = "http://example.org/homeont/"
SAREF = "https://saref.etsi.org/core/"
TD = "https://www.w3.org/2019/wot/td#"

LIGHT = URIRef("http://h/light#artifact")
AC = URIRef("http://h/ac#artifact")
ROOM = URIRef("http://h/bathroom#place")
ACTION = URIRef("http://h/light#onOff")


def _graph():
    g = Graph()
    g.add((LIGHT, RDF.type, URIRef(HOMEONT + "ColorLight")))
    g.add((LIGHT, RDF.type, URIRef(TD + "Thing")))              # outside every tree
    g.add((AC, RDF.type, URIRef(HOMEONT + "AirConditioner")))
    g.add((AC, RDF.type, URIRef(SAREF + "HVAC")))               # two routes to saref:Device
    g.add((ROOM, RDF.type, URIRef(HOMEONT + "Bathroom")))
    g.add((ACTION, RDF.type, URIRef(HOMEONT + "SetOnOffCommand")))
    close_types(g)
    return g


def _types(g, node):
    return set(g.objects(node, RDF.type))


def test_a_device_becomes_an_instance_of_its_ancestors_up_to_the_root():
    types = _types(_graph(), LIGHT)
    assert URIRef(SAREF + "Actuator") in types
    assert DEVICE_ROOT in types


def test_nothing_above_a_root_is_added():
    above_device = {p for p in vocabulary().objects(DEVICE_ROOT, RDFS.subClassOf)
                    if isinstance(p, URIRef)}
    assert above_device, "saref:Device has named superclasses to stop short of"
    assert not _types(_graph(), LIGHT) & above_device


def test_restrictions_are_not_followed():
    g = _graph()
    assert not any(isinstance(t, BNode) for _, t in g.subject_objects(RDF.type))


def test_a_type_outside_every_tree_adds_nothing():
    # td:Thing leads to no root; the light gains only the ColorLight chain.
    types = _types(_graph(), LIGHT)
    assert all(str(t).startswith((HOMEONT, SAREF)) or t == URIRef(TD + "Thing")
               for t in types)


def test_rooms_and_commands_close_too():
    g = _graph()
    assert URIRef(HOMEONT + "BuildingSpace") in _types(g, ROOM)
    assert URIRef(SAREF + "OnCommand") in _types(g, ACTION)
    assert URIRef(SAREF + "Command") in _types(g, ACTION)


def test_a_generic_class_matches_each_device_once():
    rows = list(_graph().query(
        "SELECT ?x WHERE { ?x a <https://saref.etsi.org/core/Device> }"))
    assert sorted(str(r.x) for r in rows) == sorted([str(LIGHT), str(AC)])


def test_a_sibling_class_is_not_inferred():
    assert URIRef(HOMEONT + "OnOffLight") not in _types(_graph(), LIGHT)


def test_a_colour_light_is_a_dimmable_light_as_homeont_says():
    # homeont: `:ColorLight rdfs:subClassOf :DimmableLight`.
    assert URIRef(HOMEONT + "DimmableLight") in _types(_graph(), LIGHT)


def test_closing_twice_adds_nothing():
    g = _graph()
    assert close_types(g) == 0


def test_every_root_is_in_the_vocabulary():
    vocab = vocabulary()
    for root in CLASS_ROOTS:
        assert (root, None, None) in vocab or (None, RDFS.subClassOf, root) in vocab, root
