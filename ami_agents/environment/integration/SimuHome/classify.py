#!/usr/bin/env python3
"""
Classify a SimuHome device attribute: is it observable, actuatable, or plumbing?

This is the single source of truth shared by the inspector and the TD generator,
so what the inspector shows as settable is exactly what gets an actuatable
affordance in the Thing Description.

Every Matter fact comes from the vendored registry
(`matter_model/vendor/1.5.1/`, connectedhomeip v1.5.1.0), never
from memory. Two rules that are easy to get wrong and are therefore encoded here
once:

  * **Writable is not the same as actuatable.** Matter actuates through cluster
    *commands* as well as attribute writes. `OnOff.OnOff` is read-only in the
    spec, yet it is the most-actuated attribute in the benchmark -- On/Off/Toggle
    drive it. So an attribute is actuatable if it is spec-writable OR is the
    target of one of its cluster's commands.
  * **Writable is not the same as effectful.** Installed capability
    (ControlSequenceOfOperation), power-restoration defaults (StartUp*) and
    timing parameters (transition times) are writable but change nothing now.
    They stay observable so an agent is not invited to set them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# `matter_model` is a sibling package here, but this module is also imported by
# scripts run directly (not as a package), so make the parent directory
# importable rather than relying on a relative import.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from matter_model.registry import load_registry  # noqa: E402

# Which attribute each cluster's commands write. The Matter XML states no
# command->attribute relation, so this mirrors the curated, evidence-backed
# table in tests/simuhome/td/mappings/command_targets.yaml (all 14 approved).
COMMAND_TARGETS: Dict[str, set] = {
    "OnOff": {"OnOff"},
    "LevelControl": {"CurrentLevel"},
    "WindowCovering": {
        "CurrentPositionLiftPercent100ths", "TargetPositionLiftPercent100ths",
    },
    "Thermostat": {
        "OccupiedCoolingSetpoint", "OccupiedHeatingSetpoint", "SystemMode",
    },
    "TemperatureControl": {"TemperatureSetpoint"},
    "FanControl": {"PercentSetting", "FanMode", "SpeedSetting"},
    "DishwasherMode": {"CurrentMode"},
    "LaundryWasherMode": {"CurrentMode"},
    "RVCRunMode": {"CurrentMode"},
    "RVCCleanMode": {"CurrentMode"},
    "RTCCMode": {"CurrentMode"},
}

# Writable, but writing them has no present-tense effect on the world.
NO_IMMEDIATE_EFFECT = {
    ("Thermostat", "ControlSequenceOfOperation"),
    ("WindowCovering", "Mode"),
    ("OnOff", "StartUpOnOff"),
    ("LevelControl", "StartUpCurrentLevel"),
    ("OnOff", "OnTime"),
    ("OnOff", "OffWaitTime"),
    ("LevelControl", "OnOffTransitionTime"),
    ("LevelControl", "OnTransitionTime"),
    ("LevelControl", "OffTransitionTime"),
    ("LevelControl", "DefaultMoveRate"),
    ("LevelControl", "OnLevel"),
    ("LevelControl", "Options"),
}

# Attributes that state a BOUND on a sibling attribute rather than a value of
# their own. They are constraints, so they belong in the sibling's schema as
# `js:minimum`/`js:maximum` (see td_builder._add_limits), not as readable
# properties -- exposing `MinTemperature` as its own affordance would model a
# constraint as if it were state, and state the same fact twice.
#
# Verified against the corpus: every one of these pairs with exactly one sibling
# in the same cluster, and nothing else reads them.
BOUND_ATTRIBUTES = {
    ("TemperatureControl", "MinTemperature"),
    ("TemperatureControl", "MaxTemperature"),
    ("LevelControl", "MinLevel"),
    ("LevelControl", "MaxLevel"),
    ("TemperatureMeasurement", "MinMeasuredValue"),
    ("TemperatureMeasurement", "MaxMeasuredValue"),
    ("RelativeHumidityMeasurement", "MinMeasuredValue"),
    ("RelativeHumidityMeasurement", "MaxMeasuredValue"),
}

# Device telemetry self-description that no planner acts on: measurement
# accuracy (a nested list of AccuracyRanges structs), how many measurement
# types the meter supports, and LevelControl's frequency bounds, which are
# degenerate (0/0) because the feature is not implemented in this simulator.
UNACTIONABLE_METADATA = {
    ("ElectricalPowerMeasurement", "Accuracy"),
    ("ElectricalEnergyMeasurement", "Accuracy"),
    ("ElectricalPowerMeasurement", "NumberOfMeasurementTypes"),
    ("LevelControl", "MinFrequency"),
    ("LevelControl", "MaxFrequency"),
}

# Clusters describing the protocol rather than the home.
PLUMBING_CLUSTERS = {
    "Descriptor", "Identify", "PowerTopology", "BridgedDeviceBasicInformation",
}
METADATA_CLUSTERS = {"BasicInformation"}


def split_attribute_path(path: str) -> Optional[Tuple[int, str, str]]:
    """`"1.FanControl.PercentSetting"` -> `(1, "FanControl", "PercentSetting")`."""
    parts = path.split(".", 2)
    if len(parts) != 3:
        return None
    endpoint, cluster, attribute = parts
    try:
        return int(endpoint), cluster, attribute
    except ValueError:
        return None


def classify(cluster: str, attribute: str) -> Dict[str, Any]:
    """Return the role of one attribute.

    role:       drop | bound | thing_metadata | affordance
    affordance: observable | actuatable
    mechanism:  read | attribute_write | command
    """
    registry = load_registry()
    resolved = registry.attribute(cluster, attribute)
    writable = registry.is_writable(cluster, attribute)
    is_global = registry.is_global(cluster, attribute)

    if cluster in METADATA_CLUSTERS:
        role = "thing_metadata"
    elif cluster in PLUMBING_CLUSTERS or is_global:
        role = "drop"
    elif (cluster, attribute) in BOUND_ATTRIBUTES:
        # Kept out of the affordance set, but still read by the TD generator
        # from the raw device attributes to populate js:minimum/js:maximum.
        role = "bound"
    elif (cluster, attribute) in UNACTIONABLE_METADATA:
        role = "drop"
    else:
        role = "affordance"

    command_driven = attribute in COMMAND_TARGETS.get(cluster, set())
    effectful = (cluster, attribute) not in NO_IMMEDIATE_EFFECT

    if role == "affordance" and effectful and (writable is True or command_driven):
        affordance = "actuatable"
        mechanism = "attribute_write" if writable is True else "command"
    else:
        affordance = "observable"
        mechanism = "read"

    return {
        "role": role,
        "affordance": affordance,
        "mechanism": mechanism,
        "writable": writable,
        "type": (resolved or {}).get("type"),
        "resolved": resolved is not None,
        "cluster": cluster,
        "attribute": attribute,
    }


def classify_device(attributes: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Classify every attribute of a device, keyed by its dotted path.

    `attributes` is what SimuHome returns: {"1.FanControl.PercentSetting": 60}.
    Unparseable or unresolvable paths are reported rather than dropped, so a
    registry gap is visible instead of silent.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for path, value in attributes.items():
        parsed = split_attribute_path(str(path))
        if parsed is None:
            out[str(path)] = {
                "role": "drop", "affordance": "observable", "mechanism": "read",
                "resolved": False, "value": value, "endpoint": None,
                "cluster": None, "attribute": str(path), "writable": None,
                "type": None,
            }
            continue
        endpoint, cluster, attribute = parsed
        record = classify(cluster, attribute)
        record["value"] = value
        record["endpoint"] = endpoint
        out[str(path)] = record
    return out
