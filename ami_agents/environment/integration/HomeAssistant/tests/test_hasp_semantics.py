"""The HA -> homeont mapping, on its own: defaults, overrides, gaps."""

from pathlib import Path

import pytest

from hasp_semantics import Entity, light_supports, load_semantics, snake_case

LAB308E = Path(__file__).resolve().parents[1] / "lab308e.yaml"
HOMELAB = Path(__file__).resolve().parents[1] / "homelab.yaml"


@pytest.fixture(scope="module")
def defaults():
    return load_semantics(home_config="")


@pytest.fixture(scope="module")
def lab308e():
    return load_semantics(home_config=str(LAB308E))


@pytest.fixture(scope="module")
def homelab():
    """The BEIA lab home: a kitchen, a garage, three device groups."""
    return load_semantics(home_config=str(HOMELAB))


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
        # A motorised window is no window covering.
        ([Entity("cover", {"device_class": "window"})], "homeont:WindowController"),
        ([Entity("switch", {"device_class": "outlet"})], "homeont:SmartPlug"),
        ([Entity("binary_sensor", {"device_class": "moisture"}),
          Entity("sensor", {"device_class": "temperature"})], "homeont:WaterLeakSensor"),
    ])
    def test_defaults(self, defaults, entities, expected):
        """What HA's domain and device_class alone make a device."""
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


# -- a device of several classes (a 2-in-1) -------------------------------------

DEHUMIDIFIER = Entity("humidifier", {"device_class": "dehumidifier"})
PURIFIER_FAN = Entity("fan", {"preset_modes": ["auto", "night"]})


class TestSeveralClasses:
    def test_the_defaults_give_one_class(self, defaults):
        """Without a home override a device is one thing: its primary entity's."""
        assert defaults.device_classes("x", [DEHUMIDIFIER]) == ["homeont:Dehumidifier"]
        assert defaults.device_classes("x", [Entity("humidifier")]) == ["homeont:Humidifier"]
        assert defaults.device_classes("x", [Entity("switch")]) == []

    def test_a_home_lists_every_class_of_a_2_in_1(self, homelab):
        """The Blue Air is both; the first class is its primary one."""
        entities = [PURIFIER_FAN, DEHUMIDIFIER]
        assert homelab.device_classes("Blue Air Garage", entities) == [
            "homeont:AirPurifier", "homeont:Dehumidifier"]
        assert homelab.device_class("Blue Air Garage", entities) == "homeont:AirPurifier"

    def test_a_device_rule_matches_any_class(self, defaults):
        """Each half of a 2-in-1 is typed by the rules of its own class."""
        both = ["homeont:AirPurifier", "homeont:Dehumidifier"]
        assert defaults.property_class(
            "x", both, DEHUMIDIFIER, "state") == "homeont:DehumidifierOnOff"
        got = defaults.effects("x", "humidifier", "turn_on", both)
        assert [(v.name, d) for v, d in got] == [("relative_humidity", "decrease")]

    @pytest.mark.parametrize("device_class, service, expected", [
        ("homeont:Dehumidifier", "turn_on", [("relative_humidity", "decrease")]),
        ("homeont:Dehumidifier", "turn_off", [("relative_humidity", "increase")]),
        ("homeont:Dehumidifier", "set_humidity", [("relative_humidity", None)]),
        ("homeont:Humidifier", "turn_on", [("relative_humidity", "increase")]),
        ("homeont:Humidifier", "turn_off", [("relative_humidity", "decrease")]),
    ])
    def test_which_way_a_humidifier_moves_humidity(self, defaults, device_class,
                                                   service, expected):
        """A dehumidifier dries the air, a humidifier moistens it; a setpoint
        moves it either way."""
        got = defaults.effects("x", "humidifier", service, device_class)
        assert [(v.name, d) for v, d in got] == expected

    def test_a_plain_humidifier_entity_is_a_humidifier(self, defaults):
        assert defaults.property_class(
            "x", None, Entity("humidifier"), "state") == "homeont:HumidifierOnOff"


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

    @pytest.mark.parametrize("entity, signal, expected", [
        (Entity("sensor", {"device_class": "pm25"}), "state", "homeont:Pm25MassConcentration"),
        (Entity("sensor", {"device_class": "pm10"}), "state", "homeont:Pm10MassConcentration"),
        (Entity("sensor", {"device_class": "atmospheric_pressure"}), "state", "homeont:AtmosphericPressure"),
        (Entity("sensor", {"device_class": "sound_pressure"}), "state", "homeont:SoundPressureLevel"),
        (Entity("sensor", {"device_class": "power"}), "state", "homeont:ActivePower"),
        (Entity("sensor", {"device_class": "reactive_power"}), "state", "homeont:ReactivePower"),
        (Entity("binary_sensor", {"device_class": "moisture"}), "state", "homeont:WaterLeak"),
        (Entity("binary_sensor", {"device_class": "connectivity"}), "state", "homeont:Connectivity"),
        (Entity("select"), "state", "homeont:DeviceMode"),
        (Entity("cover", {"device_class": "window"}), "state", "homeont:OpenCloseState"),
    ])
    def test_new_defaults(self, defaults, entity, signal, expected):
        """Readings and states typed by HA's device_class alone."""
        assert defaults.property_class("x", None, entity, signal) == expected

    def test_an_override_by_name_types_one_entity(self, homelab):
        """A thermal camera's three °C readings: the affordance name says which
        is which; a reading the home names nothing for keeps the default."""
        celsius = Entity("sensor", {"device_class": "temperature"})
        assert homelab.property_class(
            "Camera termica 1", "homeont:ThermalCamera", celsius, "state",
            name="max_temperature") == "homeont:SurfaceTemperatureMax"
        assert homelab.property_class(
            "Camera termica 2", "homeont:ThermalCamera", celsius, "state",
            name="average_temperature") == "homeont:SurfaceTemperatureAverage"
        assert homelab.property_class(
            "Camera termica 1", "homeont:ThermalCamera", celsius, "state",
            name="other_reading") == "homeont:AirTemperature"

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

    def test_choosing_and_setting(self, defaults):
        """An option is a mode; a number a level; a text value neither (a
        gateway sends it as a code)."""
        assert defaults.command_class("x", "select", "select_option") == "homeont:SetModeCommand"
        assert defaults.command_class("x", "number", "set_value") == "saref:SetAbsoluteLevelCommand"
        assert defaults.command_class("x", "text", "set_value") is None

    def test_an_override_by_action_name(self, homelab):
        """The Broadlink's text action sends learned codes."""
        assert homelab.command_class(
            "Broadlink RM4 pro Garaj", "text", "set_value",
            name="send_command_set_value") == "homeont:SendCodeCommand"


class TestRooms:
    def test_an_area_named_like_a_room(self, defaults):
        assert defaults.location_class({"area_id": "living_room",
                                        "name": "Living Room"}) == "homeont:LivingRoom"

    def test_an_unknown_area(self, defaults):
        assert defaults.location_class({"area_id": "lab308e", "name": "lab308e"}) is None

    def test_the_home_declares_its_room(self, lab308e):
        assert lab308e.location_class({"area_id": "lab308e"}) == "homeont:StudyRoom"
        assert lab308e.group_class({"area_id": "lab308e"}) is None

    def test_each_area_its_own_room(self, homelab):
        """`areas:` types each area; "Garaj" matches no homeont label itself."""
        assert homelab.location_class({"area_id": "peroni_22", "name": "Peroni 22"}) == "homeont:Kitchen"
        assert homelab.location_class({"area_id": "garaj", "name": "Garaj"}) == "homeont:Garage"

    def test_a_device_group_is_no_room(self, homelab):
        """An energy system stands for a device group, never a place."""
        area = {"area_id": "invertoare", "name": "Invertoare"}
        assert homelab.group_class(area) == "homeont:DeviceGroup"
        assert homelab.location_class(area) is None


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

    def test_units_mqtt_devices_send(self, defaults):
        """hPa and dB name one variable each; μg/m³ (Greek mu, as MQTT devices
        send it) names PM10 and PM2.5 alike, so it types nothing on its own."""
        hpa = Entity("sensor", {"unit_of_measurement": "hPa"})
        assert defaults.property_class("x", None, hpa, "state") == "homeont:AtmosphericPressure"
        db = Entity("sensor", {"unit_of_measurement": "dB"})
        assert defaults.property_class("x", None, db, "state") == "homeont:SoundPressureLevel"
        for micro in ("μg/m³", "µg/m³"):
            pm = Entity("sensor", {"unit_of_measurement": micro})
            assert defaults.property_class("x", None, pm, "state") is None

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

    @pytest.mark.parametrize("device_class, service, expected", [
        ("homeont:WindowController", "open_cover", [("carbon_dioxide_concentration", "decrease")]),
        ("homeont:WindowController", "close_cover", [("carbon_dioxide_concentration", "increase")]),
        ("homeont:WindowCoveringController", "close_cover", [("illuminance", "decrease")]),
    ])
    def test_a_window_ventilates_instead_of_dimming(self, defaults, device_class,
                                                    service, expected):
        """Rules for a device class replace the domain's general rules."""
        got = defaults.effects("x", "cover", service, device_class)
        assert [(v.name, d) for v, d in got] == expected

    @pytest.mark.parametrize("device, domain, service, expected", [
        ("desk_lamp_308e", "light", "turn_on", [("desk_illuminance", "increase")]),
        ("window_308e", "cover", "open_cover", [("carbon_dioxide_concentration", "decrease")]),
        ("blackout_blinds_308e", "cover", "close_cover",
         [("illuminance", "decrease"), ("glare", "decrease")]),
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
