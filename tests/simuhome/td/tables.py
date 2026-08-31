#!/usr/bin/env python3
"""
The eight Phase A mapping tables, plus the status-gated idempotent merge.

Every row carries:
  key       stable identity (decimal cluster.attribute, family token, ...) —
            never derived from ordering, so reruns are byte-identical.
  status    auto     fully determined by the registry or by unambiguous input
            review   a defensible heuristic value that encodes judgement
            todo     could not be decided; must be filled before Phase B
            approved set by a human; the generator never overwrites it
  evidence  the counts/sources that justify an auto value

Merge rules on rerun (see merge_rows):
  approved -> preserved verbatim; material drift is recorded, never applied
  auto     -> regenerated wholesale (machine-owned)
  review   -> machine fields refreshed, human-supplied values preserved
  todo     -> same as review
  vanished -> kept and flagged _orphaned, never silently deleted
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from . import registry_access, scan
from .scan import Corpus

AUTO, REVIEW, TODO, APPROVED = "auto", "review", "todo", "approved"

# The complete set of saref:Device subclasses in SAREF core v3.2.1. Checked
# against the published ontology, not assumed: there is no `saref:Lighting`.
SAREF_DEVICE_SUBCLASSES = {
    "saref:Actuator", "saref:Appliance", "saref:HVAC", "saref:Meter", "saref:Sensor",
}

# Fields the generator owns even on review/todo rows. Anything else a human
# writes into such a row survives regeneration.
MACHINE_FIELDS = {
    "cluster", "attribute", "command", "matter", "ha_domain", "platform",
    "occurrences", "instances", "episodes", "used_by_families", "clusters",
    "writable", "access", "evidence", "affordance", "mechanism", "actuated_by",
    "title", "rooms", "state_tokens", "unit", "samples", "role", "global",
    "spec_name", "spec_id", "spec_id_hex", "type",
    # Derived from the cluster/attribute pair, not curated. It must stay machine
    # owned: `td:name` has to be unique within a Thing, and a preserved older
    # value silently reintroduces a collision the generator has just fixed.
    "affordance_name",
}

# --- semantic seeds -------------------------------------------------------
#
# The environmental vocabulary is not derivable from Matter, so these are named
# after the physical quantity the sensor actually measures, matching the QUDT
# quantity kind in each row.
#
# The lab308e runtime (hasp.py::_ambient_var_from_signals and the
# tdsosa-env-overrides.*.json files) uses looser names for the same variables --
# `luminosity`, `thermal_comfort`, `air_quality`. Those are kept as aliases so a
# deployment configured against the old vocabulary still resolves, but they are
# not the property names: luminosity is emitted light, not the illuminance a room
# sensor receives; thermal comfort is a derived index (ASHRAE-style, involving
# humidity and air speed), not the dry-bulb temperature actually reported; and
# `air_quality` implies a composite index where the benchmark models only PM10.
ROOM_STATE_PROPERTIES = {
    "Temperature": (
        "air_temperature", "°C", "quantitykind:ThermodynamicTemperature", "unit:DEG_C",
        ["thermal_comfort"],
    ),
    "Humidity": (
        "relative_humidity", "%", "quantitykind:RelativeHumidity", "unit:PERCENT",
        ["humidity"],
    ),
    "Illuminance": (
        "illuminance", "lx", "quantitykind:Illuminance", "unit:LUX",
        ["luminosity"],
    ),
    "Pm10": (
        "pm10_mass_concentration", "ug/m3", "quantitykind:MassConcentration",
        "unit:MicroGM-PER-M3", ["air_quality"],
    ),
}

# Device family -> (observable property it affects, direction when switched ON).
# Seeded from run_simuhome_e2e.py::_scenario_tdsosa_hints, which is hand-curated
# and benchmark-validated -- notably "purifier ON lowers pm10", which service-name
# heuristics get backwards.
FAMILY_EFFECTS: Dict[str, Tuple[str, Optional[str]]] = {
    "AirPurifier": ("pm10_mass_concentration", "decrease"),
    "Humidifier": ("relative_humidity", "increase"),
    "Dehumidifier": ("relative_humidity", "decrease"),
    "OnOffLight": ("illuminance", "increase"),
    "DimmableLight": ("illuminance", "increase"),
    # Fan and WindowCoveringController are deliberately absent: SimuHome does not
    # simulate them affecting any environmental variable, and claiming an effect
    # the simulator will not produce would make a plan look correct while the
    # measured value never moves. See docs/effects-vs-simulator.md.
    # AC and heat pump are the exception: their On/Off has no direction either,
    # because the device drives toward a SETPOINT that spans the current room
    # temperature. Every instance permits max_temp 36-40C, so switching one on
    # can warm or cool depending on a target set elsewhere.
    "AirConditioner": ("air_temperature", None),
    "HeatPump": ("air_temperature", None),
}

# `actsOnProperty` / `affectsObservableProperty` state what a command CAN affect
# in general -- the device's capability -- not what it will achieve under current
# conditions. Opening a blind affects illuminance even at night; running a fan
# affects air temperature even in a room that will not cool. Whether the effect
# materialises here and now is for a planning agent to determine from the
# environment, not something a Thing Description should hedge about.
#
# Direction is a property of the COMMAND, not of the device:
#   On / Off        -> directional. On starts the effect, Off stops it.
#   percentage/level-> directionless. Setting a humidifier from 40 to 20 LOWERS
#                      the room humidity; 40 to 80 raises it. The value decides,
#                      so the affordance itself asserts no direction.
# DIRECTIONAL_COMMANDS and NON_DIRECTIONAL_COMMANDS below encode exactly that.

# SAREF parents, checked against saref.ttl v3.2.1 rather than assumed.
#
# SAREF core declares exactly five subclasses of saref:Device -- Actuator,
# Appliance, HVAC, Meter, Sensor. There is no `saref:Lighting`: the only
# light-related terms are saref:Light and saref:LightSwitch, both DEPRECATED and
# both modelling a *Property*, not a device. So lamps take saref:Actuator, whose
# definition -- "designed to control one or more properties or states of one or
# more features of interest" -- is exactly what a lamp does to room illuminance.
#
# saref:Appliance is "designed to accomplish a particular task for occupant use;
# it consumes, produces, or stores some commodity", which covers the air
# treatment devices and the task appliances alike.
FAMILY_SAREF_PARENT = {
    "AirConditioner": "saref:HVAC",
    "HeatPump": "saref:HVAC",
    "AirPurifier": "saref:Appliance",
    "Dehumidifier": "saref:Appliance",
    "Humidifier": "saref:Appliance",
    "Fan": "saref:Appliance",
    "OnOffLight": "saref:Actuator",
    "DimmableLight": "saref:Actuator",
    "WindowCoveringController": "saref:Actuator",
    "Dishwasher": "saref:Appliance",
    "LaundryWasher": "saref:Appliance",
    "LaundryDryer": "saref:Appliance",
    "Freezer": "saref:Appliance",
    "Refrigerator": "saref:Appliance",
    "Tv": "saref:Appliance",
    "Rvc": "saref:Appliance",
}

# Commands whose direction is inherent, per the spec's own semantics.
DIRECTIONAL_COMMANDS = {
    ("OnOff", "On"): "increase",
    ("OnOff", "Off"): "decrease",
    ("OnOff", "OffWithEffect"): "decrease",
    ("OnOff", "OnWithRecallGlobalScene"): "increase",
    ("OnOff", "OnWithTimedOff"): "increase",
    ("WindowCovering", "UpOrOpen"): "increase",
    ("WindowCovering", "DownOrClose"): "decrease",
}

# Non-directional by construction: toggles flip, setters go either way.
NON_DIRECTIONAL_COMMANDS = {
    ("OnOff", "Toggle"),
    ("LevelControl", "MoveToLevel"),
    ("LevelControl", "MoveToLevelWithOnOff"),
    ("LevelControl", "Move"),
    ("LevelControl", "MoveWithOnOff"),
    ("LevelControl", "Step"),
    ("LevelControl", "StepWithOnOff"),
    ("LevelControl", "Stop"),
    ("LevelControl", "StopWithOnOff"),
    # FanControl.Step omitted: see UNMODELLED_COMMANDS below.
    ("WindowCovering", "GoToLiftPercentage"),
    ("WindowCovering", "GoToLiftValue"),
    ("WindowCovering", "StopMotion"),
    ("Thermostat", "SetpointRaiseLower"),
    ("TemperatureControl", "SetTemperature"),
}

# Commands a simulator may implement but that carry no generic meaning, so no
# command -> attribute link is curated for them and no effect is claimed.
#
# `FanControl.Step` is the case that named this set. SimuHome does implement it
# (`fan_control.py:_step`), and it works by calling `_update_percent_setting` --
# so in THIS simulator it writes PercentSetting. But that is an implementation
# choice, not protocol: the Matter spec declares no command -> attribute relation
# for it, and a different stack could implement Step against FanMode, or not at
# all. Curating the row would encode one simulator's internals as though they
# were Matter semantics.
#
# Nothing is lost by the omission. `control_rules` lists Step as *optional* for
# air_purifier and dehumidifier -- only `OnOff.On` is required -- the real
# actuation path is the PercentSetting/FanMode attributes, which are modelled,
# and no benchmark query asks for a relative fan change.
UNMODELLED_COMMANDS = {
    ("FanControl", "Step"),
}

# Clusters that describe the protocol rather than the home.
PLUMBING_CLUSTERS = {"Descriptor", "Identify", "PowerTopology", "BasicInformation"}

# --- quantities, results and sensor metadata ------------------------------
#
# Matter attribute names label fields in the protocol, not concepts in the
# world. `RelativeHumidityMeasurement.MeasuredValue` is not a property: the
# property is *relative humidity*, and MeasuredValue is where a reading of it
# arrives. Minting `homeont:RelativeHumidityMeasurementMeasuredValue` would invert
# SOSA -- a result masquerading as the quantity it results from -- and would
# also split one quantity across two nodes, since the room-level observation
# already declares `relative_humidity`.
#
# So each attribute gets a `sosa_role`:
#   quantity_result   a reading of QUANTITY            -> affords the shared property
#   quantity_target   a settable target for QUANTITY   -> affords the shared property
#   sensor_metadata   range/accuracy of the sensor     -> ssn:MeasurementRange / Accuracy
#   device_state      genuine device-local state       -> its own property
#   drop              protocol plumbing
#
# ATTRIBUTE_QUANTITY maps (cluster, attribute) -> the quantity it observes or
# targets. The quantity names are those declared in observable_properties.yaml,
# so device and room observations resolve to the SAME property node.
ATTRIBUTE_QUANTITY: Dict[Tuple[str, str], str] = {
    ("RelativeHumidityMeasurement", "MeasuredValue"): "relative_humidity",
    ("TemperatureMeasurement", "MeasuredValue"): "air_temperature",
    ("Thermostat", "LocalTemperature"): "air_temperature",
    ("Thermostat", "OccupiedCoolingSetpoint"): "air_temperature",
    ("Thermostat", "OccupiedHeatingSetpoint"): "air_temperature",
    # NOT TemperatureControl.TemperatureSetpoint: that cluster sets the
    # temperature *inside* an appliance cabinet -- a freezer targets -22 C, a
    # laundry washer 90 C -- which is a different property from the room air a
    # thermostat conditions. It stays a per-family leaf.
}

# Attributes that describe the measuring device rather than the world: the
# range it can report, and how accurate it is.
SENSOR_METADATA_ATTRIBUTES = {
    ("RelativeHumidityMeasurement", "MinMeasuredValue"),
    ("RelativeHumidityMeasurement", "MaxMeasuredValue"),
    ("RelativeHumidityMeasurement", "Tolerance"),
    ("TemperatureMeasurement", "MinMeasuredValue"),
    ("TemperatureMeasurement", "MaxMeasuredValue"),
    ("TemperatureControl", "MinTemperature"),
    ("TemperatureControl", "MaxTemperature"),
    ("ElectricalEnergyMeasurement", "Accuracy"),
    ("ElectricalPowerMeasurement", "Accuracy"),
    ("ElectricalPowerMeasurement", "NumberOfMeasurementTypes"),
    ("LevelControl", "MinLevel"),
    ("LevelControl", "MaxLevel"),
    ("LevelControl", "MinFrequency"),
    ("LevelControl", "MaxFrequency"),
}

# Human-facing affordance names. A TD affordance is named after what it gives
# you, never after the Matter field it happens to arrive in.
AFFORDANCE_NAMES: Dict[Tuple[str, str], str] = {
    ("RelativeHumidityMeasurement", "MeasuredValue"): "humidity",
    ("TemperatureMeasurement", "MeasuredValue"): "temperature",
    ("Thermostat", "LocalTemperature"): "temperature",
    ("Thermostat", "OccupiedCoolingSetpoint"): "coolingSetpoint",
    ("Thermostat", "OccupiedHeatingSetpoint"): "heatingSetpoint",
    ("Thermostat", "SystemMode"): "hvacMode",
    ("TemperatureControl", "TemperatureSetpoint"): "temperatureSetpoint",
    ("FanControl", "PercentSetting"): "fanSpeed",
    ("FanControl", "PercentCurrent"): "currentFanSpeed",
    ("FanControl", "FanMode"): "fanMode",
    ("OnOff", "OnOff"): "onOff",
    ("LevelControl", "CurrentLevel"): "brightness",
    ("WindowCovering", "TargetPositionLiftPercent100ths"): "targetPosition",
    ("WindowCovering", "CurrentPositionLiftPercent100ths"): "position",
    ("MediaPlayback", "CurrentState"): "playbackState",
    ("OperationalState", "OperationalState"): "operationalState",
    ("RVCOperationalState", "OperationalState"): "operationalState",
}

# Writable attributes whose write does NOT change the world now. They stay
# actuatable -- Matter really does let a client write them, and a TD says so with
# `readOnly: false` -- but Phase B must emit no `tdsosa:hasEffectActuation` for
# them, because claiming an environmental effect would be false.
#
# Three kinds, all sharing "no present-tense effect":
#   capability    what the installed equipment can do, or how it is mounted
#   startup       the state to resume after power is restored
#   parameter     timing that shapes how a *future* action behaves
#
# Note this is not derivable from Matter's access model: `SystemMode` carries the
# same `manage` write privilege as `ControlSequenceOfOperation` yet is the
# primary heat/cool control, while the StartUp* pair are mere `operate`.
NO_IMMEDIATE_EFFECT: Dict[Tuple[str, str], str] = {
    ("Thermostat", "ControlSequenceOfOperation"): "capability",
    ("WindowCovering", "Mode"): "capability",
    ("OnOff", "StartUpOnOff"): "startup",
    ("LevelControl", "StartUpCurrentLevel"): "startup",
    ("OnOff", "OnTime"): "parameter",
    ("OnOff", "OffWaitTime"): "parameter",
    ("LevelControl", "OnOffTransitionTime"): "parameter",
    ("LevelControl", "OnTransitionTime"): "parameter",
    ("LevelControl", "OffTransitionTime"): "parameter",
    ("LevelControl", "DefaultMoveRate"): "parameter",
    ("LevelControl", "OnLevel"): "parameter",
    ("LevelControl", "Options"): "parameter",
}


# Matter field names that must never surface as an affordance or property name.
FORBIDDEN_NAME_TOKENS = (
    "MeasuredValue", "MinMeasured", "MaxMeasured", "Tolerance",
    "Percent100ths", "Setting", "Struct",
)


def _lower_camel(value: str) -> str:
    """`CurrentLevel` -> `currentLevel`, for TD affordance names."""
    camel = _camel(value)
    return camel[:1].lower() + camel[1:] if camel else camel


def _command_affordance_name(affordance_name: Optional[str], family: str) -> Optional[str]:
    """Family-qualified TD affordance name.

        brightness + Tv -> tvBrightness

    A bare `brightness` collides the moment two families expose the same Matter
    process, and 31 of the 102 affordance attributes are shared across families
    (OnOff.OnOff alone spans 11). The name must say *whose* brightness it is.
    """
    if not affordance_name:
        return None
    return _lower_camel(f"{family}_{affordance_name}")


def family_property_iri(family: str, cluster_property_iri: str) -> str:
    """The family-specific leaf under a cluster-level property.

        homeont:LevelControlBrightness + Tv -> homeont:TvBrightness

    Matter clusters name *processes* -- `LevelControl` is "the numeric-knob
    process", `OnOff` is "the switching process" -- so a cluster-level property
    generalises across every family that happens to use that process. A TV's
    brightness and a lamp's brightness are both LevelControl.CurrentLevel, but
    they are different properties with different ranges and different effects on
    the room. Constraints therefore belong on the leaf, and the cluster-level
    class exists only to say what the leaves have in common.
    """
    local = cluster_property_iri.split(":", 1)[1]
    for cluster in sorted(_CLUSTER_PREFIXES, key=len, reverse=True):
        if local.startswith(cluster):
            suffix = local[len(cluster):]
            # Keep the cluster term when nothing else remains, or the leaf would
            # collapse onto the bare family name and collide with the device
            # class: homeont:AirPurifier (a saref:Device) vs its on/off property.
            return f"homeont:{_camel(family)}{suffix or cluster}"
    return f"homeont:{_camel(family)}{local}"


# Cluster-name prefixes stripped when building a family leaf, so
# `homeont:LevelControlBrightness` becomes `homeont:TvBrightness`, not
# `homeont:TvLevelControlBrightness`.
_CLUSTER_PREFIXES = {
    "LevelControl", "OnOff", "FanControl", "WindowCovering", "Thermostat",
    "TemperatureControl", "OperationalState", "RVCOperationalState",
    "MediaPlayback", "Channel", "PowerSource", "LaundryWasherControls",
    "LaundryDryerControls", "DishwasherAlarm", "DeviceEnergyManagement",
    "ElectricalPowerMeasurement", "ElectricalEnergyMeasurement",
    "RelativeHumidityMeasurement", "TemperatureMeasurement",
    "DishwasherMode", "LaundryWasherMode", "RTCCMode", "RVCCleanMode", "RVCRunMode",
}


def _device_property_iri(cluster_name: str, affordance_name: str) -> str:
    """Cluster-qualified IRI for a device-local property.

    Qualification is load-bearing: 12 of the benchmark's attribute names occur in
    more than one cluster (`MeasuredValue` in 15 registry-wide, `CurrentMode` in
    11), so dropping the cluster would fuse distinct properties onto one node.

    The exception is an exact duplicate -- cluster `On/Off` with attribute
    `OnOff` -- where qualifying adds nothing but a stutter. Collapsing it is safe
    only because the attribute name is unique across the registry; the general
    rule stays cluster-qualified.
    """
    cluster = _camel(cluster_name)
    local = _camel(affordance_name)
    return f"homeont:{local}" if cluster == local else f"homeont:{cluster}{local}"


# A TD property name must be unique WITHIN one Thing, because `td:name` is how
# the read route resolves back to a Matter attribute. Three names would
# otherwise be ambiguous on a single device, verified across all 14 distinct
# device shapes in the 600-episode corpus:
#
#   RVC          RVCRunMode.CurrentMode    vs RVCCleanMode.CurrentMode
#   RVC          RVCRunMode.SupportedModes vs RVCCleanMode.SupportedModes
#   HeatPump     ElectricalPowerMeasurement.Accuracy
#                                          vs ElectricalEnergyMeasurement.Accuracy
#
# Names shared across clusters that never co-occur on one device (DishwasherMode
# vs LaundryWasherMode) are left alone -- they are unambiguous where it counts,
# and qualifying them would only lengthen the planner's vocabulary.
_NAME_QUALIFIER = {
    "RVCRunMode": "run",
    "RVCCleanMode": "clean",
    "ElectricalPowerMeasurement": "power",
    "ElectricalEnergyMeasurement": "energy",
}
_QUALIFIED_ATTRIBUTES = {"CurrentMode", "SupportedModes", "Accuracy"}


def _affordance_name(cluster_token: str, attribute_token: str) -> str:
    """The TD `td:name` for an attribute, unique within any one device."""
    explicit = AFFORDANCE_NAMES.get((cluster_token, attribute_token))
    if explicit:
        return explicit
    qualifier = _NAME_QUALIFIER.get(cluster_token)
    if qualifier and attribute_token in _QUALIFIED_ATTRIBUTES:
        return _lower_camel(f"{qualifier}_{attribute_token}")
    return _lower_camel(attribute_token)


def classify_attribute(cluster_token: str, attribute_token: str, *, role: str) -> Dict[str, Any]:
    """Decide the SOSA role, the quantity, and the human-facing name."""
    pair = (cluster_token, attribute_token)
    if role == "drop":
        return {"sosa_role": "drop", "quantity": None, "affordance_name": None}
    if pair in SENSOR_METADATA_ATTRIBUTES:
        # Metadata still reaches the Thing Description as a property affordance,
        # so it needs a name like any other. Leaving it null pushed naming onto
        # a fallback and made `MinLevel`/`MinTemperature` indistinguishable.
        return {
            "sosa_role": "sensor_metadata",
            "quantity": ATTRIBUTE_QUANTITY.get(pair),
            "affordance_name": _affordance_name(cluster_token, attribute_token),
        }
    quantity = ATTRIBUTE_QUANTITY.get(pair)
    name = _affordance_name(cluster_token, attribute_token)
    if quantity:
        is_target = any(
            token in attribute_token for token in ("Setpoint", "Target", "Setting")
        )
        return {
            "sosa_role": "quantity_target" if is_target else "quantity_result",
            "quantity": quantity,
            "affordance_name": name,
        }
    return {"sosa_role": "device_state", "quantity": None, "affordance_name": name}

# Matter attribute types that denote a device-internal reading rather than an
# environmental one (energy/power telemetry, battery, run state).
INTERNAL_CLUSTERS = {
    "PowerSource", "ElectricalEnergyMeasurement", "ElectricalPowerMeasurement",
    "DeviceEnergyManagement", "OperationalState", "RVCOperationalState",
    "DishwasherAlarm",
}

# Matter command -> Home Assistant service, per HA domain. Judgement (review).
HA_SERVICE_MAP: Dict[Tuple[str, str, str], Dict[str, Any]] = {
    ("fan", "OnOff", "On"): {"service": "fan.turn_on", "payload": {}},
    ("fan", "OnOff", "Off"): {"service": "fan.turn_off", "payload": {}},
    ("fan", "OnOff", "Toggle"): {"service": "fan.toggle", "payload": {}},
    ("fan", "FanControl", "PercentSetting"): {
        "service": "fan.set_percentage", "payload": {"percentage": "{value}"}
    },
    ("fan", "FanControl", "FanMode"): {
        "service": "fan.set_preset_mode",
        "payload": {"preset_mode": "{value}"},
        # Matter's FanModeEnum minus the values no device here implements.
        # `medium` is included deliberately: every device in the benchmark
        # declares FanModeSequence=OffLowHigh, which excludes Medium, yet 1398
        # devices sit at Medium and 12 queries ask for a "medium" fan. Honouring
        # the capability declaration would reject both the live state and the
        # user's request, so the enum follows what the corpus actually uses.
        "enum": ["off", "low", "medium", "high"],
    },
    ("light", "OnOff", "On"): {"service": "light.turn_on", "payload": {}},
    ("light", "OnOff", "Off"): {"service": "light.turn_off", "payload": {}},
    ("light", "OnOff", "Toggle"): {"service": "light.toggle", "payload": {}},
    ("light", "LevelControl", "CurrentLevel"): {
        "service": "light.turn_on", "payload": {"brightness": "{value}"}
    },
    ("cover", "WindowCovering", "UpOrOpen"): {"service": "cover.open_cover", "payload": {}},
    ("cover", "WindowCovering", "DownOrClose"): {"service": "cover.close_cover", "payload": {}},
    ("cover", "WindowCovering", "StopMotion"): {"service": "cover.stop_cover", "payload": {}},
    ("cover", "WindowCovering", "GoToLiftPercentage"): {
        "service": "cover.set_cover_position", "payload": {"position": "{value}"}
    },
    ("climate", "Thermostat", "OccupiedCoolingSetpoint"): {
        "service": "climate.set_temperature", "payload": {"temperature": "{value}"}
    },
    ("climate", "Thermostat", "OccupiedHeatingSetpoint"): {
        "service": "climate.set_temperature", "payload": {"temperature": "{value}"}
    },
    ("climate", "Thermostat", "SystemMode"): {
        "service": "climate.set_hvac_mode", "payload": {"hvac_mode": "{value}"}
    },
    # Start/stop appliances. qt4 asks to "start dishwasher 1", "pause it when
    # ... finishes", "start playing TV 1", "start robot vacuum 1 in Running
    # state" -- all on/off in Home Assistant terms, since the virtual component
    # provides no media_player or vacuum platform.
    ("switch", "OnOff", "On"): {"service": "switch.turn_on", "payload": {}},
    ("switch", "OnOff", "Off"): {"service": "switch.turn_off", "payload": {}},
    ("switch", "OnOff", "Toggle"): {"service": "switch.toggle", "payload": {}},
    # Cabinet target temperature: "set freezer 1 in the kitchen to -23".
    ("number", "TemperatureControl", "TemperatureSetpoint"): {
        "service": "number.set_value", "payload": {"value": "{value}"}
    },
}

# Services the repo already treats as direction-neutral setpoints
# (hasp.py::_SETPOINT_SERVICES).
SETPOINT_SERVICES = {
    "set_temperature", "set_hvac_mode", "set_percentage", "set_speed",
    "set_humidity", "set_cover_position", "set_position", "set_value",
}


def _camel(value: str) -> str:
    """Build a syntactically valid CURIE local name.

    A CURIE local name must be an NCName: it may not contain `/`, spaces or other
    delimiters, or the reference becomes ambiguous once expanded. Matter cluster
    names carry exactly such characters -- "On/Off", "Fan Control", "RVC Run
    Mode" -- so split on every non-alphanumeric run and CamelCase the parts:
    "On/Off" + "OnOff" -> `OnOffOnOff`, never `On/OffOnOff`.
    """
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", value) if part]
    local = "".join(part[:1].upper() + part[1:] for part in parts)
    if local and local[0].isdigit():
        local = f"_{local}"  # NCName may not start with a digit
    return local or "Item"


def _row(key: str, status: str, **fields: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {"key": key}
    row.update(fields)
    row["status"] = status
    return row


def _matter_ref(cluster_token: str, attribute_token: str) -> Optional[Dict[str, Any]]:
    reg = registry_access.get_registry()
    cluster = reg.cluster_by_name(cluster_token)
    attribute = reg.attribute(cluster_token, attribute_token)
    if cluster is None or attribute is None:
        return None
    return {
        "cluster_id": cluster["id"],
        "cluster_id_hex": cluster["id_hex"],
        "cluster_name": cluster["name"],
        "attribute_id": attribute["id"],
        "attribute_id_hex": attribute["id_hex"],
        "attribute_name": attribute["name"],
        "type": attribute.get("type"),
    }


def _attribute_key(cluster_token: str, attribute_token: str) -> str:
    ref = _matter_ref(cluster_token, attribute_token)
    if ref is None:
        return f"unresolved:{cluster_token}.{attribute_token}"
    return f"{ref['cluster_id']}.{ref['attribute_id']}"


# Which attribute each cluster command writes.
#
# The Matter XML never states this: there is no `writesAttribute` relation, and
# the link lives in the spec's prose. Deriving it from field types and name
# similarity is unreliable -- that heuristic pairs GoToTiltPercentage with the
# *Lift* attributes (both percent100ths) and SetpointRaiseLower with SystemMode
# (spurious "Mode" substring). So the links are curated here, and the evidence
# that supports each is derived from the XML and attached to the row, so a
# reviewer can check the claim rather than take it on trust.
#
# `approved` marks links verified against the vendored command definitions.
COMMAND_TARGETS: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("TemperatureControl", "SetTemperature"): {
        "attribute": "TemperatureSetpoint",
        "status": APPROVED,
        "evidence": "only command on the cluster; field TargetTemperature shares "
                    "the attribute's type `temperature`",
    },
    ("OnOff", "On"): {
        "attribute": "OnOff", "status": APPROVED,
        "evidence": "no input fields; the command name is the attribute's value",
    },
    ("OnOff", "Off"): {
        "attribute": "OnOff", "status": APPROVED,
        "evidence": "no input fields; the command name is the attribute's value",
    },
    ("OnOff", "Toggle"): {
        "attribute": "OnOff", "status": APPROVED,
        "evidence": "no input fields; inverts the attribute it is named after",
    },
    ("LevelControl", "MoveToLevel"): {
        "attribute": "CurrentLevel", "status": APPROVED,
        "evidence": "field `Level` shares the attribute's type `uint8`",
    },
    ("LevelControl", "MoveToLevelWithOnOff"): {
        "attribute": "CurrentLevel", "status": APPROVED,
        "evidence": "field `Level` shares the attribute's type `uint8`",
    },
    ("WindowCovering", "GoToLiftPercentage"): {
        "attribute": "TargetPositionLiftPercent100ths", "status": APPROVED,
        "evidence": "field `LiftPercent100thsValue` shares the attribute's type "
                    "`percent100ths`; both name the Lift axis",
    },
    ("WindowCovering", "UpOrOpen"): {
        "attribute": "TargetPositionLiftPercent100ths", "status": APPROVED,
        "evidence": "no input fields; drives the lift to its open limit, writing the "
                    "same target attribute GoToLiftPercentage writes at an arbitrary "
                    "value (CurrentPosition… is the read-back). Bound to "
                    "cover.open_cover, and requested in the benchmark as "
                    "\"fully open the blinds\" with no percentage",
    },
    ("WindowCovering", "DownOrClose"): {
        "attribute": "TargetPositionLiftPercent100ths", "status": APPROVED,
        "evidence": "no input fields; drives the lift to its closed limit, writing the "
                    "same target attribute as GoToLiftPercentage. Bound to "
                    "cover.close_cover, and the most common cover request in the "
                    "benchmark (\"fully close the blinds\", no percentage)",
    },
    # The Mode Base family. Every *Mode cluster derives from Mode Base, whose
    # ChangeToMode(NewMode: uint8) writes CurrentMode -- the same shape as
    # TemperatureControl.SetTemperature -> TemperatureSetpoint. qt4 exercises
    # these directly: "start washer 1 on Heavy mode", "set robot vacuum 1 to
    # Running state". The human-readable labels live in the sibling
    # SupportedModes attribute and must be surfaced with the affordance, since
    # `CurrentMode: 3` is meaningless on its own.
    ("DishwasherMode", "ChangeToMode"): {
        "attribute": "CurrentMode", "status": APPROVED,
        "evidence": "Mode Base defines ChangeToMode(NewMode: uint8) writing "
                    "CurrentMode (uint8, read-only); labels come from SupportedModes",
    },
    ("LaundryWasherMode", "ChangeToMode"): {
        "attribute": "CurrentMode", "status": APPROVED,
        "evidence": "as Mode Base; qt4 asks to start the washer on a named mode",
    },
    ("RVCRunMode", "ChangeToMode"): {
        "attribute": "CurrentMode", "status": APPROVED,
        "evidence": "as Mode Base; qt4 asks to set the vacuum to Running state",
    },
    ("RVCCleanMode", "ChangeToMode"): {
        "attribute": "CurrentMode", "status": APPROVED,
        "evidence": "as Mode Base; selects the cleaning mode",
    },
    ("RTCCMode", "ChangeToMode"): {
        "attribute": "CurrentMode", "status": APPROVED,
        "evidence": "as Mode Base; refrigerator/freezer cabinet mode",
    },
    # FanControl.Step is deliberately absent. Home Assistant's `fan` domain
    # exposes no step service, so no TD form could invoke it, and no query in the
    # 600-episode benchmark asks for a relative fan change: of 174 fan-mentioning
    # queries, 106 give an absolute percentage and 21 name a mode. Keeping it
    # would assert an environmental effect nothing can trigger.
}


def build_command_targets(corpus: Corpus) -> List[Dict[str, Any]]:
    """Which attribute each command writes, with the evidence for the claim."""
    reg = registry_access.get_registry()
    used_clusters = set(corpus.clusters)
    rows: List[Dict[str, Any]] = []
    for (cluster_token, command_name), spec in sorted(COMMAND_TARGETS.items()):
        if cluster_token not in used_clusters:
            continue
        cluster = reg.cluster_by_name(cluster_token)
        attribute = reg.attribute(cluster_token, spec["attribute"])
        command = next(
            (c for c in reg.commands(cluster_token) if c["name"] == command_name), None
        )
        if cluster is None or attribute is None or command is None:
            continue
        rows.append(
            _row(
                f"{cluster['id']}.{command['id']}",
                spec["status"],
                cluster=cluster["name"],
                command=command_name,
                writes_attribute=spec["attribute"],
                attribute_type=attribute.get("type"),
                attribute_writable=attribute.get("writable"),
                evidence={
                    "why": spec["evidence"],
                    "note": "the Matter XML states no command->attribute relation; "
                            "this link is curated and the evidence derived",
                },
            )
        )
    return rows


def _command_targets(cluster_token: str) -> Set[str]:
    """Attributes a cluster's commands plausibly drive.

    Matter actuates through commands, so an attribute can be agent-controllable
    while being read-only. This is a curated link because the XML does not state
    which attribute a command writes.
    """
    return {
        "OnOff": {"OnOff"},
        "LevelControl": {"CurrentLevel"},
        "WindowCovering": {
            "CurrentPositionLiftPercent100ths", "TargetPositionLiftPercent100ths"
        },
        "Thermostat": {
            "OccupiedCoolingSetpoint", "OccupiedHeatingSetpoint", "SystemMode"
        },
        "TemperatureControl": {"TemperatureSetpoint"},
        "FanControl": {"PercentSetting", "FanMode", "SpeedSetting"},
        # Mode Base derivatives: ChangeToMode writes CurrentMode.
        "DishwasherMode": {"CurrentMode"},
        "LaundryWasherMode": {"CurrentMode"},
        "RVCRunMode": {"CurrentMode"},
        "RVCCleanMode": {"CurrentMode"},
        "RTCCMode": {"CurrentMode"},
    }.get(cluster_token, set())


def build_attribute_map(corpus: Corpus) -> List[Dict[str, Any]]:
    """One row per distinct cluster.attribute. Pure registry ∩ corpus: all auto."""
    reg = registry_access.get_registry()
    rows: List[Dict[str, Any]] = []
    for cluster_token, attribute_token in scan.sorted_attributes(corpus):
        record = corpus.attributes[(cluster_token, attribute_token)]
        ref = _matter_ref(cluster_token, attribute_token)
        writable = reg.is_writable(cluster_token, attribute_token)
        is_global = reg.is_global(cluster_token, attribute_token)
        command_driven = attribute_token in _command_targets(cluster_token)

        if cluster_token in PLUMBING_CLUSTERS or is_global:
            role = "drop"
        elif cluster_token == "BasicInformation":
            role = "thing_metadata"
        else:
            role = "affordance"

        if writable is True or command_driven:
            affordance = "actuatable"
            mechanism = "attribute_write" if writable is True else "command"
        else:
            affordance = "observable"
            mechanism = "read"

        units = sorted(record["units"])
        classification = classify_attribute(cluster_token, attribute_token, role=role)
        rows.append(
            _row(
                _attribute_key(cluster_token, attribute_token),
                AUTO,
                yaml_path=f"{cluster_token}.{attribute_token}",
                # The Matter identity is provenance: it says where the value comes
                # from, never what the property is called.
                matter=ref,
                writable=writable,
                affordance=affordance,
                mechanism=mechanism,
                role=role,
                sosa_role=classification["sosa_role"],
                quantity=classification["quantity"],
                affordance_name=classification["affordance_name"],
                unit=units[0] if len(units) == 1 else (units or None),
                used_by_families=sorted(record["families"]),
                occurrences=record["occurrences"],
                evidence={
                    "samples": record["samples"][:3],
                    "device_classes": sorted(record["device_classes"]),
                },
            )
        )
    return rows


def build_device_type_map(corpus: Corpus) -> List[Dict[str, Any]]:
    """16 rows keyed on the family token. ha_domain/title auto; ontology review."""
    rows: List[Dict[str, Any]] = []
    for family in scan.sorted_families(corpus):
        record = corpus.families[family]
        platform = scan.family_platform(corpus, family)
        title = corpus.titles.get(family)
        rows.append(
            _row(
                family,
                REVIEW,
                ha_domain=platform,
                # A TD Thing carries BOTH: `homeont:` names the exact device kind,
                # `saref:` places it in the standard taxonomy for consumers that
                # do not load ex.ttl. This is a deliberate exception to the
                # one-type-per-node rule, which applies to property nodes.
                homeont_class=f"homeont:{_camel(family)}",
                td_types=[f"homeont:{_camel(family)}", FAMILY_SAREF_PARENT.get(family)],
                subClassOf=FAMILY_SAREF_PARENT.get(family),
                title=title,
                clusters=sorted(record["clusters"]),
                instances=record["instances"],
                episodes=len(record["episodes"]),
                rooms=sorted(record["rooms"]),
                evidence={
                    "platform_source": "SimuHome YAML `platform`",
                    "title_source": "BasicInformation.ProductName" if title else None,
                },
            )
        )
    return rows


def build_room_map(corpus: Corpus) -> List[Dict[str, Any]]:
    """One row per room token. foi_iri is todo: it needs a workspace base URI."""
    rows: List[Dict[str, Any]] = []
    for room in scan.sorted_rooms(corpus):
        record = corpus.rooms[room]
        rows.append(
            _row(
                room,
                REVIEW,
                homeont_class=f"homeont:{_camel(room)}",
                subClassOf="s4bldg:BuildingSpace",
                foi_iri=None,
                foi_iri_reason="needs the deployment workspace base IRI; unknown at Phase A",
                state_tokens=sorted(record["state_tokens"]),
                instances=record["instances"],
                episodes=len(record["episodes"]),
            )
        )
    return rows


def build_observable_properties(corpus: Corpus) -> List[Dict[str, Any]]:
    """Environmental variables. Not derivable from Matter, so all review."""
    rows: List[Dict[str, Any]] = []
    for token in sorted(corpus.room_states):
        record = corpus.room_states[token]
        name, unit, quantity_kind, qudt_unit, aliases = ROOM_STATE_PROPERTIES[token]
        rows.append(
            _row(
                f"room_state.{token.lower()}",
                REVIEW,
                room_state_token=token,
                sosa_property=name,
                sosa_type="sosa:ObservableProperty",
                # A TD must be self-contained, so this is the TYPE an anonymous
                # property node is typed by inside each TD -- never a shared IRI
                # that TDs point at from outside. Two TDs referring to the air
                # temperature of the same room say so by both typing their own
                # node `homeont:AirTemperature` and naming the same feature of
                # interest, not by sharing a node.
                homeont_class=f"homeont:{_camel(name)}",
                property_node="anonymous",
                unit=unit,
                quantityKind=quantity_kind,
                qudt_unit=qudt_unit,
                feature_of_interest="room",
                legacy_aliases=aliases,
                occurrences=record["occurrences"],
                evidence={
                    "named_after": f"the measured quantity ({quantity_kind})",
                    "aliases_used_by": (
                        "hasp.py::_ambient_var_from_signals and "
                        "tdsosa-env-overrides.*.json (lab308e runtime)"
                    ),
                    "device_classes": sorted(record["device_classes"]),
                },
            )
        )
    return rows


def build_property_classes(
    corpus: Corpus, attribute_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """The cluster-level property classes that family leaves specialise.

    This is the table to review: approving one class here promotes every family
    leaf beneath it, so 5 decisions clear 18 leaf rows rather than 18 decisions
    clearing 18. The classes are what actually encode judgement -- what a Matter
    process cluster *means* as a property -- while a leaf merely says "the Tv's
    version of that".
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for row in attribute_rows:
        if row["role"] != "affordance" or row["sosa_role"] == "sensor_metadata":
            continue
        if row["affordance"] != "actuatable":
            continue  # only actuatable attributes emit family leaves
        if row["quantity"]:
            continue  # shared environmental quantity; declared in observable_properties
        cluster_iri = _device_property_iri(
            row["matter"]["cluster_name"],
            row["affordance_name"] or row["matter"]["attribute_name"],
        )
        entry = seen.setdefault(
            cluster_iri,
            {
                "families": set(),
                "yaml_path": row["yaml_path"],
                "matter": row["matter"],
                "unit": row["unit"],
                "affordance_name": row["affordance_name"],
                "mechanisms": set(),
            },
        )
        entry["families"].update(row["used_by_families"])
        entry["mechanisms"].add(row["mechanism"])

    rows: List[Dict[str, Any]] = []
    for iri in sorted(seen):
        entry = seen[iri]
        # A class is mechanical when nothing is left to decide: either its leaves
        # are spec-writable, or the command->attribute links that drive them have
        # been verified in command_targets.yaml. Asking about the class was only
        # ever a proxy for asking about those links.
        unverified = [
            f"{cluster}.{command}"
            for (cluster, command), spec in COMMAND_TARGETS.items()
            if spec["attribute"] == entry["matter"]["attribute_name"]
            and cluster == entry["yaml_path"].split(".", 1)[0]
            and spec["status"] != APPROVED
        ]
        blocks_leaves = "command" in entry["mechanisms"] and bool(unverified)
        rows.append(
            _row(
                iri,
                REVIEW if blocks_leaves else AUTO,
                homeont_class=iri,
                subClassOf="sosa:ActuatableProperty",
                affordance_name=entry["affordance_name"],
                yaml_path=entry["yaml_path"],
                matter=entry["matter"],
                # The unit is a property of the quantity and so is safe on the
                # class. Ranges are NOT: they vary per deployed instance and are
                # emitted on the anonymous property node inside each TD.
                unit=entry["unit"],
                carries_constraints=False,
                specialised_by=sorted(
                    family_property_iri(family, iri) for family in entry["families"]
                ),
                evidence={
                    "rationale": (
                        "Matter clusters name processes, not things; this class is what "
                        f"{len(entry['families'])} device famil"
                        f"{'ies' if len(entry['families']) != 1 else 'y'} have in common"
                    ),
                    "approving_this": "promotes every leaf below it to `auto`",
                },
            )
        )
    return rows


def build_actuatable_properties(
    corpus: Corpus,
    attribute_rows: List[Dict[str, Any]],
    approved_classes: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Settable knobs, one leaf per family under a cluster-level class.

    A leaf inherits its superclass's approval: once you have decided what
    `homeont:LevelControlBrightness` means, "the Tv's version of it" follows
    mechanically and is marked `auto`.
    """
    approved_classes = approved_classes or set()
    rows: List[Dict[str, Any]] = []
    for row in attribute_rows:
        if row["affordance"] != "actuatable" or row["role"] != "affordance":
            continue
        if row["sosa_role"] == "sensor_metadata":
            continue  # describes the device, not a property of the world
        mechanism = row["mechanism"]
        quantity = row["quantity"]
        cluster_token, attribute_token = row["yaml_path"].split(".", 1)
        effect_kind = NO_IMMEDIATE_EFFECT.get((cluster_token, attribute_token))
        # An attribute targeting a known environmental quantity affords the
        # SHARED property node -- a thermostat setpoint really is air
        # temperature, whichever device carries it. Everything else is
        # device-local state, and gets one leaf per family under a cluster-level
        # superclass.
        cluster_iri = (
            f"homeont:{_camel(quantity)}" if quantity
            else _device_property_iri(
                row["matter"]["cluster_name"],
                row["affordance_name"] or row["matter"]["attribute_name"],
            )
        )
        for family in row["used_by_families"]:
            # The property TYPE. What `tdsosa:affordsProperty` points at is an
            # anonymous INSTANCE of this type, minted per Thing -- see
            # `property_instance` below.
            leaf = cluster_iri if quantity else family_property_iri(family, cluster_iri)
            # A leaf is mechanical once its superclass is settled: the judgement
            # lives in the class, not in "the Tv's version of it".
            verified_link = mechanism == "command" and not any(
                spec["attribute"] == row["matter"]["attribute_name"]
                and cluster == cluster_token
                and spec["status"] != APPROVED
                for (cluster, _cmd), spec in COMMAND_TARGETS.items()
            )
            inherited = bool(quantity) or cluster_iri in approved_classes or verified_link
            if inherited or mechanism == "attribute_write":
                status = AUTO
            else:
                status = REVIEW
            rows.append(
                _row(
                    f"{family}:{row['key']}",
                    status,
                    family=family,
                    yaml_path=row["yaml_path"],
                    affordance_name=row["affordance_name"],
                    td_name=_command_affordance_name(row["affordance_name"], family),
                    sosa_type="sosa:ActuatableProperty",
                    sosa_role=row["sosa_role"],
                    # tdsosa:affordsProperty ranges over property INSTANCES, not
                    # classes. Home Assistant does not fix a range per device
                    # type -- climate entities in this corpus alone carry 42
                    # distinct min_temp and 70 distinct max_temp values -- so
                    # limits are a fact about one deployed device, not about
                    # "TV brightness" in general. Phase B therefore emits a blank
                    # node typed by `property_type`, carrying that Thing's own
                    # constraints, and keeps every TD self-contained.
                    property_node="anonymous",
                    property_type=leaf,
                    subClassOf=None if quantity else cluster_iri,
                    constraints_source="per-instance, from the device's HA entity",
                    # Writable, but does writing it change anything now? If not,
                    # Phase B emits the property affordance with readOnly:false
                    # and NO tdsosa:hasEffectActuation.
                    has_immediate_effect=effect_kind is None,
                    no_effect_reason=effect_kind,
                    quantity=quantity,
                    mechanism=mechanism,
                    writable=row["writable"],
                    matter=row["matter"],
                    unit=row["unit"],
                    evidence={
                        "rationale": (
                            "spec declares write access"
                            if mechanism == "attribute_write"
                            else "read-only in the spec but driven by cluster commands"
                        ),
                        "hierarchy": (
                            "shared environmental quantity; no family leaf"
                            if quantity
                            else f"family leaf under {cluster_iri}, which generalises "
                                 "the Matter process cluster across families"
                        ),
                        "status_source": (
                            f"inherited from approved {cluster_iri}"
                            if inherited and not quantity
                            else "own classification"
                        ),
                    },
                )
            )
    return rows


def build_internal_properties(
    corpus: Corpus, attribute_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Device-internal observables and protocol plumbing."""
    rows: List[Dict[str, Any]] = []
    for row in attribute_rows:
        cluster_token = row["yaml_path"].split(".", 1)[0]
        if row["role"] == "drop":
            category = "matter_plumbing"
        elif row["role"] == "thing_metadata":
            category = "thing_metadata"
        elif row["sosa_role"] == "sensor_metadata":
            category = "sensor_capability"
        elif cluster_token in INTERNAL_CLUSTERS and row["affordance"] == "observable":
            category = "device_internal"
        else:
            continue

        # Sensor range and accuracy describe the measuring device. SSN models
        # them on the sensor, not as observable properties in the world.
        ssn_predicate = None
        if category == "sensor_capability":
            name = row["matter"]["attribute_name"]
            if "Tolerance" in name or "Accuracy" in name:
                ssn_predicate = "ssn-system:hasSystemCapability/ssn-system:Accuracy"
            else:
                ssn_predicate = "ssn-system:hasSystemCapability/ssn-system:MeasurementRange"

        rows.append(
            _row(
                row["key"],
                AUTO,
                yaml_path=row["yaml_path"],
                matter=row["matter"],
                category=category,
                sosa_role=row["sosa_role"],
                quantity=row["quantity"],
                ssn_predicate=ssn_predicate,
                sosa_type=(
                    "sosa:ObservableProperty" if category == "device_internal" else None
                ),
                emit_in_td=category != "matter_plumbing",
                used_by_families=row["used_by_families"],
                unit=row["unit"],
            )
        )
    return rows



# Which families SENSE which observable property, and whose property it is.
#
# SimuHome ships no standalone sensors -- all 16 device types are appliances or
# actuators -- but six families carry a measurement cluster. Measured over all
# 600 episodes by comparing each reading against its room's state, the split is
# absolute, with zero exceptions either way:
#
#   AirConditioner  Thermostat.LocalTemperature                1046 match /    0 differ
#   HeatPump        Thermostat.LocalTemperature                 942 match /    0 differ
#   Dehumidifier    RelativeHumidityMeasurement.MeasuredValue  1140 match /    0 differ
#   Humidifier      RelativeHumidityMeasurement.MeasuredValue   392 match /    0 differ
#   Freezer         TemperatureMeasurement.MeasuredValue          0 match /  641 differ
#   Refrigerator    TemperatureMeasurement.MeasuredValue          0 match /  634 differ
#
# The first four observe the ROOM's air: a heat pump reads 23.69 C where its room
# is 23.69 C. The last two observe their own COMPARTMENT: a freezer reads -15 C
# inside a 23.6 C kitchen. Both are real observations; they simply have different
# features of interest, and conflating them would tell a planner that a freezer
# reports the kitchen temperature.
#
# `illuminance` and `pm10_mass_concentration` have NO sensor anywhere in the
# corpus. For those the room's environmental property is the only source.
FAMILY_SENSING: Dict[str, Dict[str, Any]] = {
    "AirConditioner": {
        "attribute": "Thermostat.LocalTemperature",
        "sosa_property": "air_temperature",
        "feature_of_interest": "room",
        "instances": 1046,
    },
    "HeatPump": {
        "attribute": "Thermostat.LocalTemperature",
        "sosa_property": "air_temperature",
        "feature_of_interest": "room",
        "instances": 942,
    },
    "Dehumidifier": {
        "attribute": "RelativeHumidityMeasurement.MeasuredValue",
        "sosa_property": "relative_humidity",
        "feature_of_interest": "room",
        "instances": 1140,
    },
    "Humidifier": {
        "attribute": "RelativeHumidityMeasurement.MeasuredValue",
        "sosa_property": "relative_humidity",
        "feature_of_interest": "room",
        "instances": 392,
    },
    "Freezer": {
        "attribute": "TemperatureMeasurement.MeasuredValue",
        "sosa_property": "air_temperature",
        "feature_of_interest": "appliance_interior",
        "instances": 641,
    },
    "Refrigerator": {
        "attribute": "TemperatureMeasurement.MeasuredValue",
        "sosa_property": "air_temperature",
        "feature_of_interest": "appliance_interior",
        "instances": 634,
    },
}


def build_sensing(corpus: Corpus) -> List[Dict[str, Any]]:
    """Which families observe which property, and of WHAT.

    Emitted so a planner can answer "who can tell me this room's temperature?"
    from the graph. Rows are pre-approved: the evidence is a full-corpus
    measurement, not a judgement call, and it is recorded per row.
    """
    rows: List[Dict[str, Any]] = []
    for family in scan.sorted_families(corpus):
        spec = FAMILY_SENSING.get(family)
        if spec is None:
            continue
        record = corpus.families[family]
        cluster_token, attribute_token = str(spec["attribute"]).split(".", 1)
        room_scoped = spec["feature_of_interest"] == "room"
        rows.append({
            "key": f"{family}:{spec['attribute']}",
            "family": family,
            "matter": _matter_ref(cluster_token, attribute_token),
            "sosa_property": spec["sosa_property"],
            "featureOfInterest": spec["feature_of_interest"],
            "sensesRoomProperty": room_scoped,
            "affordance_name": _affordance_name(cluster_token, attribute_token),
            "instances": record["instances"],
            "evidence": {
                "measured_over": "600 episodes, every instance",
                "readings_matching_room_state": spec["instances"] if room_scoped else 0,
                "readings_differing_from_room": 0 if room_scoped else spec["instances"],
                "verdict": (
                    "observes the room's air: readings equal the room state on "
                    "every instance"
                    if room_scoped else
                    "observes its own compartment: a freezer reads -15C inside a "
                    "23.6C kitchen, so this is NOT an observation of the room"
                ),
            },
            "status": APPROVED,
        })
    return rows



# Attributes whose WRITE participates in changing an environmental variable.
#
# `actuation_effects` enumerates cluster COMMANDS, so attribute writes were
# never candidates -- yet the simulator's own control_rules list several as
# required or optional participants. Read from
# GET /api/environment/control_rules/{state}, which is the authority:
#
#   air_conditioner  required: Thermostat.SystemMode, Thermostat.OccupiedCoolingSetpoint
#                    optional: FanControl.PercentSetting, FanControl.FanMode
#   heat_pump        required: Thermostat.SystemMode, Thermostat.OccupiedHeatingSetpoint
#   humidifier       optional: FanControl.PercentSetting, FanControl.FanMode
#   dehumidifier     optional: FanControl.PercentSetting, FanControl.FanMode
#   air_purifier     optional: FanControl.PercentSetting, FanControl.FanMode
#
# All are value-carrying setters, so all are DIRECTIONLESS by the rule already
# settled: writing FanMode=0 stops the fan while FanMode=3 runs it hard, so the
# direction follows the value, not the act of writing.
#
# Being listed as "optional" does not weaken the claim. The effect rows are
# hints -- "this affordance can influence that variable" -- and the simulator's
# note is explicit that at least one optional action must be executed for the
# device to work at all.
EFFECTFUL_ATTRIBUTE_WRITES: Dict[str, List[Tuple[str, str, str]]] = {
    "AirConditioner": [
        ("Thermostat", "SystemMode", "required"),
        ("Thermostat", "OccupiedCoolingSetpoint", "required"),
        ("FanControl", "PercentSetting", "optional"),
        ("FanControl", "FanMode", "optional"),
    ],
    "HeatPump": [
        ("Thermostat", "SystemMode", "required"),
        ("Thermostat", "OccupiedHeatingSetpoint", "required"),
    ],
    "Humidifier": [
        ("FanControl", "PercentSetting", "optional"),
        ("FanControl", "FanMode", "optional"),
    ],
    "Dehumidifier": [
        ("FanControl", "PercentSetting", "optional"),
        ("FanControl", "FanMode", "optional"),
    ],
    "AirPurifier": [
        ("FanControl", "PercentSetting", "optional"),
        ("FanControl", "FanMode", "optional"),
    ],
}


def build_actuation_effects(corpus: Corpus) -> List[Dict[str, Any]]:
    """How each command influences the environment. Highest-judgement table."""
    reg = registry_access.get_registry()
    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for family in scan.sorted_families(corpus):
        record = corpus.families[family]
        effect = FAMILY_EFFECTS.get(family)
        for cluster_token in sorted(record["clusters"]):
            cluster = reg.cluster_by_name(cluster_token)
            if cluster is None:
                continue
            for command in reg.commands(cluster_token):
                pair = (cluster_token, command["name"])
                if pair not in DIRECTIONAL_COMMANDS and pair not in NON_DIRECTIONAL_COMMANDS:
                    continue  # no plausible environmental effect; omit per spec
                key = f"{family}:{cluster['id']}.{command['id']}"
                if key in seen:
                    continue
                seen.add(key)

                if effect is None:
                    affects, family_direction = None, None
                else:
                    affects, family_direction = effect

                if pair in NON_DIRECTIONAL_COMMANDS:
                    monotonic, direction = False, None
                    reason = "toggle or level/percent setter: no inherent direction"
                else:
                    base = DIRECTIONAL_COMMANDS[pair]
                    # A purifier switched ON lowers pm10, so the family's own
                    # direction wins over the generic on=increase reading.
                    if family_direction is None:
                        monotonic, direction = False, None
                        reason = "setpoint-style device: direction depends on target value"
                    elif base == "increase":
                        monotonic, direction = True, family_direction
                        reason = f"command turns the device on; {family} {family_direction}s {affects}"
                    else:
                        inverse = "increase" if family_direction == "decrease" else "decrease"
                        monotonic, direction = True, inverse
                        reason = f"command turns the device off; reverses {family}'s effect"

                rows.append(
                    _row(
                        key,
                        REVIEW,
                        family=family,
                        command={
                            "cluster_id": cluster["id"],
                            "cluster_name": cluster["name"],
                            "id": command["id"],
                            "name": command["name"],
                        },
                        # Names the property TYPE. Phase B emits an anonymous
                        # node of that type inside the TD; the actuation points
                        # at that node, so the TD stays self-contained.
                        affectsObservableProperty=affects,
                        actsOnProperty=None,
                        monotonic=monotonic,
                        direction=direction,
                        settling_time_seconds=None,
                        evidence={
                            "rationale": reason,
                            "seeded_from": "run_simuhome_e2e.py::_scenario_tdsosa_hints",
                        },
                    )
                    if affects
                    else _row(
                        key,
                        TODO,
                        family=family,
                        command={
                            "cluster_id": cluster["id"],
                            "cluster_name": cluster["name"],
                            "id": command["id"],
                            "name": command["name"],
                        },
                        affectsObservableProperty=None,
                        reason=(
                            f"no established environmental effect for {family}; "
                            "device acts on internal state only, or the effect is unmodelled"
                        ),
                        monotonic=monotonic,
                        direction=direction,
                    )
                )
        # Attribute writes that participate in the same effect. Emitted with the
        # same shape as a command row so nothing downstream has to special-case
        # them; `command` names the ATTRIBUTE written.
        if effect is not None:
            affects, _family_direction = effect
            for cluster_token, attribute, necessity in EFFECTFUL_ATTRIBUTE_WRITES.get(
                    family, []):
                if cluster_token not in record["clusters"]:
                    continue
                cluster = reg.cluster_by_name(cluster_token)
                resolved = reg.attribute(cluster_token, attribute)
                if cluster is None or resolved is None:
                    continue
                key = f"{family}:{cluster['id']}.attr{resolved['id']}"
                if key in seen:
                    continue
                seen.add(key)
                rows.append(_row(
                    key,
                    APPROVED,
                    family=family,
                    command={
                        "cluster_id": cluster["id"],
                        "cluster_name": cluster["name"],
                        "id": resolved["id"],
                        "name": attribute,
                        "mechanism": "attribute_write",
                    },
                    affectsObservableProperty=affects,
                    actsOnProperty=None,
                    monotonic=False,
                    direction=None,
                    settling_time_seconds=None,
                    evidence={
                        "rationale": (
                            "value-carrying setter: the direction follows the "
                            "value written, not the act of writing"),
                        "seeded_from": (
                            "GET /api/environment/control_rules -- listed as "
                            f"{necessity} for {family}"),
                        "approved_because": (
                            "the simulator itself lists this write as a "
                            f"{necessity} participant; omitting it under-claimed "
                            "the affordance and hid it from a planner searching "
                            f"for ways to move {affects}"),
                    },
                ))

    return sorted(rows, key=lambda item: item["key"])


def build_ha_binding(corpus: Corpus) -> Dict[str, Any]:
    """The invocation layer: entity_id derivation + command -> HA service."""
    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for family in scan.sorted_families(corpus):
        platform = scan.family_platform(corpus, family)
        record = corpus.families[family]
        if platform is None:
            continue

        for cluster_token in sorted(record["clusters"]):
            reg = registry_access.get_registry()
            cluster = reg.cluster_by_name(cluster_token)
            if cluster is None:
                continue

            # Command-driven services.
            for command in reg.commands(cluster_token):
                mapping = HA_SERVICE_MAP.get((platform, cluster_token, command["name"]))
                key = f"{family}:{cluster['id']}.{command['id']}"
                if key in seen:
                    continue
                if mapping is None:
                    continue
                seen.add(key)
                service = mapping["service"]
                rows.append(
                    _row(
                        key,
                        REVIEW,
                        family=family,
                        ha_domain=platform,
                        matter_command=f"{cluster_token}.{command['name']}",
                        # Actions are named per family for the same reason
                        # properties are: `turnOn` alone is ambiguous across the
                        # 11 families that expose On/Off.
                        td_name=_lower_camel(f"{family}_{command['name']}"),
                        service=service,
                        payload=mapping["payload"],
                        enum=mapping.get("enum"),
                        setpoint=service.split(".", 1)[1] in SETPOINT_SERVICES,
                    )
                )

            # Attribute-write services (setpoints exposed as HA services).
            for _cluster_tok, attribute_tok in sorted(record["attributes"]):
                if _cluster_tok != cluster_token:
                    continue
                mapping = HA_SERVICE_MAP.get((platform, cluster_token, attribute_tok))
                if mapping is None:
                    continue
                attribute = reg.attribute(cluster_token, attribute_tok)
                if attribute is None:
                    continue
                key = f"{family}:{cluster['id']}.attr{attribute['id']}"
                if key in seen:
                    continue
                seen.add(key)
                service = mapping["service"]
                rows.append(
                    _row(
                        key,
                        REVIEW,
                        family=family,
                        ha_domain=platform,
                        matter_attribute=f"{cluster_token}.{attribute_tok}",
                        td_name=_lower_camel(f"set_{family}_{attribute_tok}"),
                        service=service,
                        payload=mapping["payload"],
                        enum=mapping.get("enum"),
                        setpoint=service.split(".", 1)[1] in SETPOINT_SERVICES,
                    )
                )

    # Any family still emitted as a bare `sensor` has no actuation surface: HA
    # exposes no service on a sensor entity. The seven that used to land here
    # (dishwasher, laundry washer/dryer, tv, rvc, freezer, refrigerator) are now
    # `switch` or `number`, because qt4 actuates them directly.
    for family in scan.sorted_families(corpus):
        platform = scan.family_platform(corpus, family)
        if platform != "sensor":
            continue
        key = f"{family}:none"
        rows.append(
            _row(
                key,
                TODO,
                family=family,
                ha_domain="sensor",
                service=None,
                reason=(
                    "the converter emits this family as a read-only `sensor` primary, so "
                    "Home Assistant exposes no service to actuate it"
                ),
            )
        )

    return {
        "entity_id_derivation": {
            "device_primary": {
                "pattern": "{platform}.{slug(Scope.Room.DeviceToken)}",
                "slug_rule": "lowercase; [^a-z0-9]+ -> _; collapse repeats; strip edges",
                "source": "initial_home_config_to_homeassistant_yaml.py::_slug",
                "status": AUTO,
            },
            "room_state": {
                "pattern": "sensor.{slug(Scope.Room.StateToken)}",
                "status": AUTO,
            },
            "attribute_sensor": {
                "pattern": "sensor.{slug(Scope.Room.Device.Cluster.Attribute)}",
                "status": AUTO,
            },
            "property_read": {
                "pattern": "GET /api/states/{entity_id}",
                "status": AUTO,
            },
            "action_invoke": {
                "pattern": "POST /api/services/{domain}/{service}",
                "body": {"entity_id": "{entity_id}"},
                "status": AUTO,
            },
        },
        "rows": sorted(rows, key=lambda item: item["key"]),
    }


# --- merge ----------------------------------------------------------------

def merge_rows(
    existing: Optional[List[Dict[str, Any]]],
    generated: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Fold newly generated rows into whatever a human has already edited."""
    stats = {"approved_kept": 0, "drift": 0, "preserved_fields": 0, "orphaned": 0, "new": 0}
    if not existing:
        stats["new"] = len(generated)
        return generated, stats

    by_key = {row.get("key"): row for row in existing if isinstance(row, dict)}
    out: List[Dict[str, Any]] = []
    generated_keys: Set[str] = set()

    for row in generated:
        key = row["key"]
        generated_keys.add(key)
        prior = by_key.get(key)
        if prior is None:
            stats["new"] += 1
            out.append(row)
            continue

        status = prior.get("status")
        if status == APPROVED:
            stats["approved_kept"] += 1
            merged = dict(prior)
            drift = {
                field: [prior.get(field), row.get(field)]
                for field in MACHINE_FIELDS
                if field in row and field in prior and prior.get(field) != row.get(field)
            }
            if drift:
                stats["drift"] += 1
                merged["_drift"] = drift
            else:
                merged.pop("_drift", None)
            out.append(merged)
            continue

        # review / todo / auto: refresh machine fields, keep human values.
        merged = dict(row)
        for field, value in prior.items():
            if field in {"key", "status"} or field in MACHINE_FIELDS:
                continue
            # A human may override a SAREF parent, but not to a class SAREF does
            # not define. `saref:Lighting` was invented in an earlier pass and
            # survived a correction here precisely because this field is
            # human-editable; preserving an undefined class is never right.
            if (
                field == "subClassOf"
                and str(value).startswith("saref:")
                and value not in SAREF_DEVICE_SUBCLASSES
            ):
                stats["rejected_unknown_class"] = stats.get("rejected_unknown_class", 0) + 1
                continue
            if value is not None and value != row.get(field):
                merged[field] = value
                stats["preserved_fields"] += 1
        # Keep the human's status, EXCEPT when the generator has promoted the row
        # to `auto` -- that happens when a leaf's superclass was approved, and
        # restoring the old `review` would make approving a class have no effect.
        if status in {REVIEW, TODO} and row["status"] != AUTO:
            merged["status"] = prior["status"]
        out.append(merged)

    for key, prior in by_key.items():
        if key in generated_keys:
            continue
        stats["orphaned"] += 1
        orphan = dict(prior)
        orphan["_orphaned"] = True
        out.append(orphan)

    return out, stats


def envelope(
    rows: Any,
    *,
    table: str,
    corpus: Corpus,
    include_timestamp: bool,
) -> Dict[str, Any]:
    reg = registry_access.get_registry()
    meta: Dict[str, Any] = {
        "script": "tests/simuhome/td/convert.py",
        "phase": "A",
        "table": table,
        "matter_registry": reg.provenance,
        "episodes_scanned": corpus.episodes,
    }
    if include_timestamp:
        meta["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    payload: Dict[str, Any] = {"version": 1, "generator": meta}
    if isinstance(rows, dict):
        payload.update(rows)
    else:
        payload["rows"] = rows
    return payload


def count_statuses(rows: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {AUTO: 0, REVIEW: 0, TODO: 0, APPROVED: 0}
    items = rows.get("rows", []) if isinstance(rows, dict) else rows
    for row in items:
        status = row.get("status")
        if status in counts:
            counts[status] += 1
    return counts
