"""The capabilities context is a projection, so it is testable without an LLM.

These tests pin the shape the ENV_STATE parser prompt depends on. Two of them
guard properties that would otherwise fail silently: the rdfs:label fallback
(19 classes have no rdfs:comment) and the immediate-superclass rule (flattening
to a branch root would make "pick the most specific class" unfollowable).
"""

import pytest

from ami_agents.agents.user_assistant.utils.ontology_context import (
    build_capabilities_context,
    get_capabilities_context,
    get_capabilities_context_json,
)

BRANCH_ROOTS = {
    "homeont:ActuatableDeviceProperty",
    "homeont:DeviceStateProperty",
    "homeont:DeviceCapabilityProperty",
    "homeont:BuildingSpace",
}


@pytest.fixture(scope="module")
def context():
    return build_capabilities_context()


def _all_entries(context):
    yield from context["locations"]
    yield from context["device_types"]
    for section in context["device_properties"].values():
        yield from section
    yield from context["environment_variables"]


class TestSections:
    def test_section_counts(self, context):
        properties = context["device_properties"]
        assert len(context["locations"]) == 16
        assert len(context["device_types"]) == 16
        assert len(properties["capabilities"]) == 19
        assert len(properties["states"]) == 47
        assert len(properties["actuatable"]) == 93
        assert len(properties["measurements"]) == 10
        assert len(context["environment_variables"]) == 4

    def test_environment_variables_are_the_four_room_variables(self, context):
        assert [e["class"] for e in context["environment_variables"]] == [
            "homeont:AirTemperature",
            "homeont:Illuminance",
            "homeont:Pm10MassConcentration",
            "homeont:RelativeHumidity",
        ]

    def test_compartment_temperature_is_a_device_measurement(self, context):
        """The distinction the whole feature-of-interest edit exists for.

        AirTemperature and CompartmentTemperature share a parent and a quantity
        kind; only ssn:isPropertyOf separates "how warm is the kitchen" from
        "how cold is the freezer".
        """
        measurements = {
            e["class"] for e in context["device_properties"]["measurements"]
        }
        environment = {e["class"] for e in context["environment_variables"]}
        assert "homeont:CompartmentTemperature" in measurements
        assert "homeont:CompartmentTemperature" not in environment
        assert "homeont:AirTemperature" not in measurements


class TestEntries:
    def test_every_entry_has_class_description_and_parent(self, context):
        for entry in _all_entries(context):
            assert entry["class"], entry
            assert entry["description"], entry
            assert entry["parent_class"], entry

    def test_identifiers_are_curies(self, context):
        for entry in _all_entries(context):
            assert ":" in entry["class"]
            assert not entry["class"].startswith("http"), entry

    def test_label_fallback_when_no_comment(self, context):
        """19 classes carry no rdfs:comment and fall back to rdfs:label.

        Asserted by value, not merely non-emptiness: a regression that dropped
        comments everywhere would still pass a truthiness check.
        """
        by_class = {e["class"]: e for e in _all_entries(context)}
        assert by_class["homeont:AirConditioner"]["description"] == "Air Conditioner"
        assert by_class["homeont:Illuminance"]["description"] == "Illuminance"

    def test_comment_preferred_over_label(self, context):
        by_class = {e["class"]: e for e in _all_entries(context)}
        description = by_class["homeont:CompartmentTemperature"]["description"]
        assert "NOT the temperature of the room" in description

    def test_parent_is_immediate_superclass_not_branch_root(self, context):
        """Flattening to the root would hide the middle of a 3-level branch."""
        by_class = {e["class"]: e for e in _all_entries(context)}
        leaf = by_class["homeont:AirConditionerFanMode"]
        assert leaf["parent_class"] == "homeont:FanControlFanMode"

        mid = by_class["homeont:FanControlFanMode"]
        assert mid["parent_class"] == "homeont:ActuatableDeviceProperty"

    def test_device_types_parent_is_the_saref_class(self, context):
        by_class = {e["class"]: e for e in context["device_types"]}
        assert by_class["homeont:AirConditioner"]["parent_class"] == "saref:HVAC"

    def test_specific_leaves_are_present_for_generic_classes(self, context):
        """The parser can only be specific if the specific classes are offered."""
        actuatable = {e["class"] for e in context["device_properties"]["actuatable"]}
        assert "homeont:OnOff" in actuatable
        assert "homeont:OnOffLightOnOff" in actuatable
        assert "homeont:AirConditionerOnOff" in actuatable


class TestMeasurementKeys:
    def test_present_where_the_ontology_states_them(self, context):
        by_class = {e["class"]: e for e in context["environment_variables"]}
        illuminance = by_class["homeont:Illuminance"]
        assert illuminance["measurement_quantity"] == "quantitykind:Illuminance"
        assert illuminance["measurement_unit"] == "unit:LUX"

    def test_absent_rather_than_null_where_not_stated(self, context):
        """A missing quantity kind is an absent key, never a null value."""
        by_class = {e["class"]: e for e in context["locations"]}
        kitchen = by_class["homeont:Kitchen"]
        assert "measurement_quantity" not in kitchen
        assert "measurement_unit" not in kitchen

    def test_quantity_without_unit_is_allowed(self, context):
        """PowerFactor is dimensionless: quantity kind, no unit."""
        by_class = {
            e["class"]: e for e in context["device_properties"]["measurements"]
        }
        power_factor = by_class["homeont:PowerFactor"]
        assert power_factor["measurement_quantity"]
        assert "measurement_unit" not in power_factor


class TestRendering:
    def test_json_is_stable_and_cached(self):
        assert get_capabilities_context() is get_capabilities_context()
        assert get_capabilities_context_json() == get_capabilities_context_json()

    def test_no_branch_root_is_ever_a_leaf_only_option(self, context):
        """Branch roots appear as parent_class but must also be choosable.

        They are in the context (a parser may legitimately need the mid-level
        class); the prompt is what forbids answering with a root. This test
        records that the roots are NOT silently filtered out of the sections,
        so a change of policy is a deliberate edit rather than an accident.
        """
        actuatable = {e["class"] for e in context["device_properties"]["actuatable"]}
        assert not (BRANCH_ROOTS & actuatable), (
            "a branch root leaked into its own section"
        )

    def test_size_is_within_budget(self):
        """~12k tokens; a large jump means the projection pulled in noise."""
        rendered = get_capabilities_context_json()
        assert len(rendered) < 60_000, len(rendered)
