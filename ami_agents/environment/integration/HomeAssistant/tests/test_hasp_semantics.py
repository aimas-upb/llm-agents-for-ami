"""The HA -> homeont mapping, on its own: defaults, overrides, gaps."""

from pathlib import Path

import pytest

from hasp_semantics import Entity, light_supports, load_semantics, snake_case

LAB308E = Path(__file__).resolve().parents[1] / "lab308e.yaml"


@pytest.fixture(scope="module")
def defaults():
    return load_semantics(home_config="")


@pytest.fixture(scope="module")
def lab308e():
    return load_semantics(home_config=str(LAB308E))


def light(modes):
    return Entity("light", {"supported_color_modes": modes})


class TestDevices:
    @pytest.mark.parametrize("entities, expected", [
        ([light(["hs"])], "homeont:ColorLight"),
        ([light(["brightness"])], "homeont:DimmableLight"),
        ([light(["onoff"])], "homeont:OnOffLight"),
        ([Entity("climate", {"hvac_modes": ["off", "cool"]})], "homeont:AirConditioner"),
        ([Entity("climate", {"hvac_modes": ["off", "heat"]})], "homeont:Heater"),
        ([Entity("cover")], "homeont:WindowCoveringController"),
        ([Entity("media_player")], "homeont:MediaPlayer"),
        ([Entity("sensor", {"device_class": "humidity"})], "homeont:HumiditySensor"),
        ([Entity("sensor", {"unit_of_measurement": "ppm"})], "homeont:CarbonDioxideSensor"),
        ([Entity("binary_sensor", {"device_class": "occupancy"})], "homeont:OccupancySensor"),
    ])
    def test_defaults(self, defaults, entities, expected):
        assert defaults.device_class("x", entities) == expected

    def test_the_primary_entity_decides(self, defaults):
        """A climate device that also exposes a sensor is a climate device."""
        entities = [Entity("sensor", {"device_class": "temperature"}),
                    Entity("climate", {"hvac_modes": ["cool"]})]
        assert defaults.device_class("x", entities) == "homeont:AirConditioner"

    def test_unmapped_is_none(self, defaults):
        assert defaults.device_class("x", [Entity("switch")]) is None
        assert defaults.device_class("x", [Entity("sensor", {"unit_of_measurement": "%"})]) is None

    def test_a_home_override_wins(self, lab308e):
        assert lab308e.device_class("ceiling_fan_308e", [Entity("switch")]) == "homeont:Fan"
        assert lab308e.device_class(
            "glare_sensing_308e", [Entity("sensor", {"unit_of_measurement": "%"})]
        ) == "homeont:GlareSensor"


class TestProperties:
    def test_family_leaf_before_generic(self, defaults):
        entity = light(["brightness"])
        assert defaults.property_class(
            "x", "homeont:DimmableLight", entity, "brightness") == "homeont:DimmableLightBrightness"
        assert defaults.property_class(
            "x", None, entity, "brightness") == "homeont:LevelControlBrightness"

    def test_a_climate_state_is_its_hvac_mode(self, defaults):
        entity = Entity("climate", {"hvac_modes": ["off", "heat"]})
        assert defaults.property_class(
            "x", "homeont:Heater", entity, "state") == "homeont:HeaterHvacMode"

    def test_sensor_state_is_the_room_variable(self, defaults):
        assert defaults.property_class(
            "x", "homeont:LightSensor", Entity("sensor", {"device_class": "illuminance"}),
            "state") == "homeont:Illuminance"

    def test_brightness_is_declared_while_the_light_is_off(self, defaults):
        rules = defaults.declared_properties(
            "x", "homeont:ColorLight", light(["hs"]))
        assert [r["signal"] for r in rules] == ["brightness"]

    def test_a_property_override(self, lab308e):
        entity = Entity("sensor", {"unit_of_measurement": "%"})
        assert lab308e.property_class(
            "glare_sensing_308e", "homeont:GlareSensor", entity, "state") == "homeont:Glare"


class TestCommands:
    @pytest.mark.parametrize("domain, service, expected", [
        ("light", "turn_on", "saref:OnCommand"),
        ("cover", "open_cover", "saref:OpenCommand"),
        ("climate", "set_temperature", "saref:SetAbsoluteLevelCommand"),
        ("climate", "set_hvac_mode", "homeont:SetModeCommand"),
        ("media_player", "media_pause", "saref:PauseCommand"),
    ])
    def test_defaults(self, defaults, domain, service, expected):
        assert defaults.command_class("x", domain, service) == expected

    def test_unmapped_is_none(self, defaults):
        assert defaults.command_class("x", "media_player", "join") is None


class TestRooms:
    def test_an_area_named_like_a_room(self, defaults):
        assert defaults.location_class({"area_id": "living_room",
                                        "name": "Living Room"}) == "homeont:LivingRoom"

    def test_an_unknown_area(self, defaults):
        assert defaults.location_class({"area_id": "lab308e", "name": "lab308e"}) is None

    def test_the_home_declares_its_room(self, lab308e):
        assert lab308e.location_class({"area_id": "lab308e"}) == "homeont:StudyRoom"


# -- environment variables ------------------------------------------------------

class TestEnvironmentVariables:
    """Which variables exist, what they are called, which devices observe and
    move them -- derived from homeont and HA's vocabulary, not listed."""

    def test_the_variables_are_homeonts_environment_classes(self, defaults):
        names = {c.name for c in defaults.environment_classes.values()}
        assert {"air_temperature", "relative_humidity", "illuminance",
                "pm10_mass_concentration", "carbon_dioxide_concentration",
                "occupancy", "occupant_count", "glare"} <= names
        assert defaults.environment_class("homeont:DimmableLightOnOff") is None

    def test_quantity_kinds_and_units_come_from_homeont(self, defaults):
        co2 = defaults.environment_class("homeont:CarbonDioxideConcentration")
        assert str(co2.quantity_kind).endswith("quantitykind/MoleFraction")
        assert str(co2.unit).endswith("unit/PPM")
        assert str(defaults.environment_class("homeont:Glare").quantity_kind).endswith(
            "quantitykind/DimensionlessRatio")
        assert str(defaults.environment_class("homeont:OccupantCount").quantity_kind).endswith(
            "quantitykind/Count")

    @pytest.mark.parametrize("local, name", [
        ("AirTemperature", "air_temperature"),
        ("Pm10MassConcentration", "pm10_mass_concentration"),
        ("Illuminance", "illuminance"),
    ])
    def test_names_are_simuhomes(self, local, name):
        assert snake_case(local) == name

    def test_a_sensor_without_device_class_is_typed_by_its_unit(self, defaults):
        lux = Entity("sensor", {"unit_of_measurement": "lx"})
        assert defaults.property_class("x", None, lux, "state") == "homeont:Illuminance"
        # "%" is humidity and glare alike: no guess.
        percent = Entity("sensor", {"unit_of_measurement": "%"})
        assert defaults.property_class("x", None, percent, "state") is None

    def test_zone_and_environment_name_the_variable(self, lab308e):
        assert lab308e.variable("internal_light_sensing_308e",
                                "homeont:Illuminance").name == "illuminance"
        assert lab308e.variable("desk_light_sensing_308e",
                                "homeont:Illuminance").name == "desk_illuminance"
        outdoor = lab308e.variable("external_light_sensing_308e", "homeont:Illuminance")
        assert (outdoor.name, outdoor.environment) == ("outdoor_illuminance", "outdoor")

    @pytest.mark.parametrize("device, domain, service, expected", [
        ("lamp", "light", "turn_on", [("illuminance", "increase")]),
        ("blinds", "cover", "close_cover", [("illuminance", "decrease")]),
        ("blinds", "cover", "set_cover_position", [("illuminance", None)]),
        ("blinds", "cover", "stop_cover", []),
        ("ac", "climate", "turn_on", [("air_temperature", None)]),
        ("projector", "media_player", "turn_on", []),
    ])
    def test_default_effects(self, defaults, device, domain, service, expected):
        got = defaults.effects(device, domain, service)
        assert [(v.name, d) for v, d in got] == expected

    @pytest.mark.parametrize("device, domain, service, expected", [
        ("desk_lamp_308e", "light", "turn_on", [("desk_illuminance", "increase")]),
        ("window_308e", "cover", "open_cover", [("carbon_dioxide_concentration", "decrease")]),
        ("blackout_blinds_308e", "cover", "close_cover", [("glare", "decrease")]),
        ("ceiling_fan_308e", "switch", "turn_on", [("air_temperature", "decrease")]),
    ])
    def test_a_homes_effects_replace_the_defaults(self, lab308e, device, domain,
                                                  service, expected):
        got = lab308e.effects(device, domain, service)
        assert [(v.name, d) for v, d in got] == expected

    def test_ranges_and_older_names(self, lab308e):
        assert lab308e.target_range("desk_illuminance") == {"min": 300, "unit": "lx"}
        assert lab308e.variable_name("luminosity") == "illuminance"
        assert lab308e.variable_name("thermal_comfort") == "air_temperature"
        assert lab308e.variable_name("illuminance") == "illuminance"


def test_light_supports():
    assert light_supports({"supported_color_modes": ["onoff"]}) == set()
    assert light_supports({"supported_color_modes": ["hs"]}) == {"brightness", "color"}


def test_class_labels_come_from_homeont(defaults):
    assert defaults.label("homeont:ColorLight") == "Color Light"
    assert defaults.label("homeont:WindowCoveringController")
    assert defaults.label(None) is None



class TestActsUpon:
    def test_fields_named_like_properties(self, defaults):
        assert defaults.acts_upon("climate", "set_temperature", ["temperature"]) == {"temperature"}

    def test_rules_add_what_no_field_names(self, defaults):
        assert defaults.acts_upon("cover", "set_cover_position", ["position"]) >= {
            "state", "current_position"}

    def test_brightness_pct_changes_brightness(self, defaults):
        assert "brightness" in defaults.acts_upon("light", "turn_on", ["brightness_pct"])
        assert "brightness" not in defaults.acts_upon("light", "turn_on", ["effect"])
