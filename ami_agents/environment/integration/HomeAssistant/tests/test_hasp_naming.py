"""How HASP names an artifact's affordances (`hasp_utils.affordance_keys`).

One artifact is one HA device; each of its entities gets a key, and its
affordances are named from the key (`<key>`, `<key>_<attribute>`,
`<key>_<service>`). The key is the entity's object id with the device prefix
stripped; the domain when nothing is left; the domain in front only on a
clash. The cases below are real devices of a Home Assistant deployment.
"""
from hasp_utils import affordance_keys, slugify


# -- slugs ----------------------------------------------------------------------

def test_slugify_lowercases_and_joins_with_underscores():
    """Device names become the slug HA derives object ids from."""
    assert slugify("Blue Air Garage") == "blue_air_garage"
    assert slugify("Deschidere/Inchidere geam Telecontact") == "deschidere_inchidere_geam_telecontact"
    assert slugify("Weather station 70:ee:50:2e:da:88") == "weather_station_70_ee_50_2e_da_88"
    assert slugify(None) == ""


# -- the device prefix ----------------------------------------------------------

def test_the_device_name_is_stripped():
    """The Effekta inverter: every object id starts with the device's slug."""
    keys = affordance_keys("SolarAssistant Effekta", [
        "sensor.solarassistant_effekta_battery_state_of_charge",
        "select.solarassistant_effekta_charger_source_priority",
        "select.solarassistant_effekta_output_source_priority",
    ])
    assert keys == {
        "sensor.solarassistant_effekta_battery_state_of_charge": "battery_state_of_charge",
        "select.solarassistant_effekta_charger_source_priority": "charger_source_priority",
        "select.solarassistant_effekta_output_source_priority": "output_source_priority",
    }


def test_a_prefix_the_entities_share_is_stripped_when_the_device_name_differs():
    """The Netatmo station is named "Weather station 70:ee:..." but its
    entities `netatmo_*`: the prefix they share is stripped."""
    keys = affordance_keys("Weather station 70:ee:50:2e:da:88", [
        "sensor.netatmo_temperature",
        "sensor.netatmo_co2",
        "binary_sensor.netatmo_connectivity",
    ])
    assert keys == {
        "sensor.netatmo_temperature": "temperature",
        "sensor.netatmo_co2": "co2",
        "binary_sensor.netatmo_connectivity": "connectivity",
    }


# -- nothing left: the domain -----------------------------------------------------

def test_the_entity_named_like_its_device_is_keyed_by_its_domain():
    """The Blueair purifier: the fan entity is the device name itself."""
    keys = affordance_keys("Blue Air Garage", [
        "fan.blue_air_garage",
        "sensor.blue_air_garage_pm25",
    ])
    assert keys == {"fan.blue_air_garage": "fan", "sensor.blue_air_garage_pm25": "pm25"}


def test_a_one_entity_device_is_keyed_by_its_domain():
    """A lone entity is all device: lab308e's lamp, and a window opener whose
    object id does not even start with its device name."""
    assert affordance_keys("desk_lamp_308e", ["light.desk_lamp_308e"]) == {
        "light.desk_lamp_308e": "light"}
    assert affordance_keys("Deschidere/Inchidere geam Telecontact",
                           ["cover.geam_telecontact"]) == {"cover.geam_telecontact": "cover"}


def test_an_entity_with_no_device_is_keyed_by_its_domain():
    """A helper or template entity stands alone: same rule, no device name."""
    assert affordance_keys(None, ["sensor.total_power"]) == {"sensor.total_power": "sensor"}


# -- clashes ------------------------------------------------------------------------

def test_the_domain_tells_two_entities_with_one_key_apart():
    """Two entities left with the same key get their domain in front; the
    others keep theirs."""
    keys = affordance_keys("Blue Air", [
        "fan.blue_air_garage",
        "sensor.blue_air_garage",
        "sensor.blue_air_pm25",
    ])
    assert keys == {
        "fan.blue_air_garage": "fan_garage",
        "sensor.blue_air_garage": "sensor_garage",
        "sensor.blue_air_pm25": "pm25",
    }


def test_keys_are_unique_even_when_the_domain_does_not_help():
    """A domain-prefixed key equal to another entity's key falls back to the
    entity id: no two entities ever share a key."""
    keys = affordance_keys("dev", ["fan.dev_x", "sensor.dev_x", "sensor.dev_fan_x"])
    assert len(set(keys.values())) == 3
