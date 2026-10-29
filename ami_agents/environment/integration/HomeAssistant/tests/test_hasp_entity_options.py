"""A service field takes the values the entity itself lists, as an enum and a description.

A Home Assistant service definition is shared by every entity of its domain,
so it cannot say which modes one climate unit or one washer's mode `select`
has; the entity says so in its state attributes (`hvac_modes`, `options`,
...). Those values go into the action's input schema as `js:enum`, and into
its description, which is what the agents read.

    entity_field_options       which fields an entity lists values for
    the input schema           enum + "Values on this device: ..." after the
                               service's own description; static select
                               options get the same
"""

from rdflib import Literal
from rdflib.namespace import RDF

from hasp_utils import JS, HomeAssistantRDF, entity_field_options

HVAC_FIELD = {"hvac_mode": {"description": "HVAC operation mode",
                            "selector": {"state": {"hvac_mode": True}}}}


def _field(rdf, schema, name):
    """The member of an object schema named ``name``."""
    return next(m for m in rdf.g.objects(schema, JS.properties)
                if rdf.g.value(m, JS.propertyName) == Literal(name))


def test_entity_options_are_read_from_the_entity_state():
    attrs = {"hvac_modes": ["off", "cool", "heat"], "preset_modes": []}
    assert entity_field_options("climate", "set_hvac_mode", attrs) == \
        {"hvac_mode": ["off", "cool", "heat"]}
    # An empty list, or another service, lists nothing.
    assert entity_field_options("climate", "set_preset_mode", attrs) == {}
    assert entity_field_options("select", "select_option", {"options": ["Normal", "Heavy"]}) \
        == {"option": ["Normal", "Heavy"]}


def test_the_entitys_values_become_an_enum_and_a_description():
    rdf = HomeAssistantRDF("http://h/")
    schema = rdf._build_input_schema_from_fields(HVAC_FIELD, {"hvac_mode": ["off", "cool"]})
    field = _field(rdf, schema, "hvac_mode")
    assert (field, RDF.type, JS.StringSchema) in rdf.g
    assert sorted(str(v) for v in rdf.g.objects(field, JS.enum)) == ["cool", "off"]
    assert str(rdf.g.value(field, JS.description)) == \
        "HVAC operation mode. Values on this device: off, cool."


def test_static_select_options_are_described_too():
    rdf = HomeAssistantRDF("http://h/")
    schema = rdf._build_input_schema_from_fields(
        {"mode": {"selector": {"select": {"options": ["eco", {"value": "boost"}]}}}})
    field = _field(rdf, schema, "mode")
    assert str(rdf.g.value(field, JS.description)) == "Values on this device: eco, boost."


def test_a_field_without_listed_values_is_unchanged():
    rdf = HomeAssistantRDF("http://h/")
    schema = rdf._build_input_schema_from_fields(
        {"brightness": {"description": "Brightness", "selector": {"number": {"min": 0}}}})
    field = _field(rdf, schema, "brightness")
    assert list(rdf.g.objects(field, JS.enum)) == []
    assert str(rdf.g.value(field, JS.description)) == "Brightness."


# -- which way a scale runs (SCALE_MEANINGS) -----------------------------------

COVER_POSITION = "Percent open: 0 is fully closed, 100 is fully open."


def test_a_cover_position_field_says_it_counts_open():
    """set_cover_position's `position` says 100 is open: Home Assistant counts
    the reverse of Matter, so the TD states its own direction."""
    # 1. The service field, as Home Assistant defines it, for a cover.
    rdf = HomeAssistantRDF("http://h/")
    schema = rdf._build_input_schema_from_fields(
        {"position": {"description": "Target position",
                      "selector": {"number": {"min": 0, "max": 100}}}}, domain="cover")
    # 2. The direction follows the service's own description.
    assert str(rdf.g.value(_field(rdf, schema, "position"), JS.description)) == \
        "Target position. " + COVER_POSITION


def test_the_same_field_name_in_another_domain_gets_no_direction():
    """The table is keyed by domain: a `position` elsewhere is not a cover's."""
    rdf = HomeAssistantRDF("http://h/")
    schema = rdf._build_input_schema_from_fields(
        {"position": {"selector": {"number": {}}}}, domain="valve")
    assert rdf.g.value(_field(rdf, schema, "position"), JS.description) is None


def test_a_cover_current_position_says_it_counts_open():
    """The attribute the field sets reads the same way; a cover's other
    numbers, and a fan's percentage, stay undescribed."""
    # 1. The cover's position attribute.
    rdf = HomeAssistantRDF("http://h/")
    position = rdf._build_property_schema("current_position", 40, {}, entity_domain="cover")
    assert str(rdf.g.value(position, JS.description)) == COVER_POSITION
    # 2. Another domain's percentage.
    percentage = rdf._build_property_schema("percentage", 40, {}, entity_domain="fan")
    assert rdf.g.value(percentage, JS.description) is None
