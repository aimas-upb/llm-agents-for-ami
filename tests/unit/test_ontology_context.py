"""The ontology context is a projection, so it is testable without an LLM.

These tests pin the shape the ENV_STATE and ENV_CAPABILITIES parser prompts
depend on: every section a tree keyed by its root class. Two of them guard
properties that would otherwise fail silently: the rdfs:label fallback (19
classes have no rdfs:comment) and the nesting itself (flattening to a branch
root would make "pick the most specific class" unfollowable).
"""

import pytest

from ami_agents.agents.user_assistant.utils.ontology_context import (
    build_ontology_context,
    get_capability_ontology_context,
    get_capability_ontology_context_json,
    get_ontology_context,
    get_ontology_context_json,
    iter_classes,
)

PROPERTY_ROOTS = [
    "homeont:ActuatableDeviceProperty",
    "homeont:DeviceStateProperty",
    "homeont:DeviceCapabilityProperty",
    "sosa:ObservableProperty",
]


@pytest.fixture(scope="module")
def context():
    return build_ontology_context()


@pytest.fixture(scope="module")
def capability_context():
    return build_ontology_context(view="capabilities")


def _nodes(context):
    """class -> node, roots excluded (they carry no `class` key)."""
    return {cls: node for cls, parent, node in iter_classes(context)
            if parent is not None}


def _parents(context):
    return {cls: parent for cls, parent, _ in iter_classes(context)
            if parent is not None}


def _descendants(node):
    for child in node.get("subclasses") or []:
        yield child["class"]
        yield from _descendants(child)


class TestSections:
    def test_sections_are_keyed_by_their_root_class(self, context):
        assert list(context["locations"]) == ["homeont:BuildingSpace"]
        assert list(context["device_types"]) == ["saref:Device"]
        assert list(context["device_properties"]) == PROPERTY_ROOTS
        assert list(context["environment_variables"]) == ["sosa:ObservableProperty"]

    def test_section_counts(self, context):
        properties = context["device_properties"]
        assert len(list(_descendants(context["locations"]["homeont:BuildingSpace"]))) == 16
        # 16 SimuHome device families + the 4 virtual ambient sensors SHTD
        # mints + 6 Home Assistant (lab308e) devices, under the five SAREF
        # families.
        devices = context["device_types"]["saref:Device"]
        assert len(devices["subclasses"]) == 5
        assert len(list(_descendants(devices))) == 5 + 20 + 6
        assert len(list(_descendants(properties["homeont:DeviceCapabilityProperty"]))) == 19
        assert len(list(_descendants(properties["homeont:DeviceStateProperty"]))) == 47 + 1
        assert len(list(_descendants(properties["homeont:ActuatableDeviceProperty"]))) == 93 + 5
        assert len(list(_descendants(properties["sosa:ObservableProperty"]))) == 10
        assert len(list(_descendants(
            context["environment_variables"]["sosa:ObservableProperty"]))) == 8

    def test_the_ambient_sensors_are_saref_sensors(self, context):
        """The parent is what closes the path to saref:Device for a resolver."""
        parents = _parents(context)
        sensors = {cls: parents[cls] for cls in parents
                   if cls.startswith("homeont:") and cls.endswith("Sensor")}
        assert sensors == {
            "homeont:TemperatureSensor": "saref:Sensor",
            "homeont:HumiditySensor": "saref:Sensor",
            "homeont:LightSensor": "saref:Sensor",
            "homeont:AirQualitySensor": "saref:Sensor",
            # Home Assistant (lab308e)
            "homeont:CarbonDioxideSensor": "saref:Sensor",
            "homeont:OccupancySensor": "saref:Sensor",
            "homeont:GlareSensor": "saref:Sensor",
        }

    def test_saref_devices_no_graph_uses_are_left_out(self, context):
        devices = set(_descendants(context["device_types"]["saref:Device"]))
        assert "saref:Switch" not in devices
        assert "saref:SmokeSensor" not in devices

    def test_environment_variables_are_the_room_variables(self, context):
        root = context["environment_variables"]["sosa:ObservableProperty"]
        assert [n["class"] for n in root["subclasses"]] == [
            "homeont:AirTemperature",
            "homeont:CarbonDioxideConcentration",
            "homeont:Glare",
            "homeont:Illuminance",
            "homeont:Occupancy",
            "homeont:OccupantCount",
            "homeont:Pm10MassConcentration",
            "homeont:RelativeHumidity",
        ]

    def test_compartment_temperature_is_a_device_measurement(self, context):
        """The distinction the whole feature-of-interest edit exists for.

        AirTemperature and CompartmentTemperature share a parent and a quantity
        kind; only ssn:isPropertyOf separates "how warm is the kitchen" from
        "how cold is the freezer".
        """
        measurements = set(_descendants(
            context["device_properties"]["sosa:ObservableProperty"]))
        environment = set(_descendants(
            context["environment_variables"]["sosa:ObservableProperty"]))
        assert "homeont:CompartmentTemperature" in measurements
        assert "homeont:CompartmentTemperature" not in environment
        assert "homeont:AirTemperature" not in measurements

    def test_state_view_has_no_capability_sections(self, context):
        assert "commands" not in context
        assert "device_metadata" not in context


class TestNodes:
    def test_every_node_has_class_and_description(self, context):
        for cls, parent, node in iter_classes(context):
            assert node["description"], cls
            if parent is not None:
                assert node["class"] == cls

    def test_identifiers_are_curies(self, context):
        for cls, _, _ in iter_classes(context):
            assert ":" in cls
            assert not cls.startswith("http"), cls

    def test_label_fallback_when_no_comment(self, context):
        """19 classes carry no rdfs:comment and fall back to rdfs:label.

        Asserted by value, not merely non-emptiness: a regression that dropped
        comments everywhere would still pass a truthiness check.
        """
        nodes = _nodes(context)
        assert nodes["homeont:AirConditioner"]["description"] == "Air Conditioner"
        assert nodes["homeont:Illuminance"]["description"] == "Illuminance"

    def test_comment_preferred_over_label(self, context):
        description = _nodes(context)["homeont:CompartmentTemperature"]["description"]
        assert "NOT the temperature of the room" in description

    def test_nesting_follows_the_immediate_superclass(self, context):
        """Flattening to the root would hide the middle of a 3-level branch."""
        parents = _parents(context)
        assert parents["homeont:AirConditionerFanMode"] == "homeont:FanControlFanMode"
        assert parents["homeont:FanControlFanMode"] == "homeont:ActuatableDeviceProperty"
        assert parents["homeont:DimmableLightBrightness"] == "homeont:LevelControlBrightness"
        assert parents["homeont:GuestBedroom"] == "homeont:Bedroom"

    def test_device_types_nest_under_their_saref_family(self, context):
        assert _parents(context)["homeont:AirConditioner"] == "saref:HVAC"

    def test_specific_leaves_are_present_for_generic_classes(self, context):
        """The parser can only be specific if the specific classes are offered."""
        actuatable = set(_descendants(
            context["device_properties"]["homeont:ActuatableDeviceProperty"]))
        assert {"homeont:OnOff", "homeont:OnOffLightOnOff",
                "homeont:AirConditionerOnOff"} <= actuatable

    def test_a_root_is_never_its_own_descendant(self, context):
        for root, node in context["device_properties"].items():
            assert root not in set(_descendants(node))

    def test_no_class_appears_twice_in_a_section(self, context):
        for root, node in context["device_properties"].items():
            classes = list(_descendants(node))
            assert len(classes) == len(set(classes)), root


class TestMeasurementKeys:
    def test_present_where_the_ontology_states_them(self, context):
        illuminance = _nodes(context)["homeont:Illuminance"]
        assert illuminance["measurement_quantity"] == "quantitykind:Illuminance"
        assert illuminance["measurement_unit"] == "unit:LUX"

    def test_absent_rather_than_null_where_not_stated(self, context):
        """A missing quantity kind is an absent key, never a null value."""
        kitchen = _nodes(context)["homeont:Kitchen"]
        assert "measurement_quantity" not in kitchen
        assert "measurement_unit" not in kitchen

    def test_quantity_without_unit_is_allowed(self, context):
        """PowerFactor is dimensionless: quantity kind, no unit."""
        power_factor = _nodes(context)["homeont:PowerFactor"]
        assert power_factor["measurement_quantity"]
        assert "measurement_unit" not in power_factor


class TestCapabilityView:
    def test_adds_commands_and_device_metadata(self, capability_context):
        assert list(capability_context["commands"]) == ["saref:Command"]
        assert [m["property"] for m in capability_context["device_metadata"]] == [
            "schema:manufacturer", "schema:model"]

    def test_shares_every_state_section(self, context, capability_context):
        for key in context:
            assert capability_context[key] == context[key], key

    def test_commands_come_from_homeont_and_saref(self, capability_context):
        commands = set(_descendants(capability_context["commands"]["saref:Command"]))
        assert {"homeont:SetModeCommand", "homeont:SetOnOffCommand",
                "saref:SetAbsoluteLevelCommand", "saref:OnCommand"} <= commands

    def test_a_class_with_several_parents_appears_once(self, capability_context):
        """SetOnOffCommand specialises On, Off and Toggle: filed under one,
        the others recorded, never duplicated."""
        placements = [(parent, node) for cls, parent, node
                      in iter_classes(capability_context)
                      if cls == "homeont:SetOnOffCommand"]
        assert len(placements) == 1
        parent, node = placements[0]
        assert parent == "saref:OffCommand"
        assert node["also_subclass_of"] == ["saref:OnCommand", "saref:ToggleCommand"]

    def test_a_root_carries_no_parent_of_its_own(self, capability_context):
        device = capability_context["device_types"]["saref:Device"]
        assert "also_subclass_of" not in device


class TestRendering:
    def test_json_is_stable_and_cached(self):
        assert get_ontology_context() is get_ontology_context()
        assert get_ontology_context_json() == get_ontology_context_json()
        assert get_capability_ontology_context() is get_capability_ontology_context()

    def test_unknown_view_is_rejected(self):
        with pytest.raises(ValueError):
            build_ontology_context(view="everything")

    def test_size_is_within_budget(self):
        """~12k tokens; a large jump means the projection pulled in noise."""
        assert len(get_ontology_context_json()) < 60_000
        assert len(get_capability_ontology_context_json()) < 60_000
