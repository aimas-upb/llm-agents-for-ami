#!/usr/bin/env python3
"""
Self-test for the vendored Matter registry and the converter mappings built on it.

Runs offline against matter_model/vendor/<VER>/. Exits non-zero on failure.

  python ami_agents/environment/integration/SimuHome/matter_model/selftest.py
  python ami_agents/environment/integration/SimuHome/matter_model/selftest.py --benchmark /path/to/SimuHome/data/benchmark
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The Home Assistant YAML converter stayed under tests/simuhome/ when this
# package moved: it is an HA-path script, while the registry is now owned by
# the SHTD integration. The end-to-end assertions below still exercise it.
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "tests" / "simuhome"))

from registry import CLUSTER_NAME_ALIASES, load_registry  # noqa: E402

import initial_home_config_to_homeassistant_yaml as conv  # noqa: E402

FAILURES: list[str] = []


def check(label: str, expected, actual) -> None:
    ok = expected == actual
    print(f"{'PASS' if ok else 'FAIL':4}  {label:<58} {actual!r}")
    if not ok:
        FAILURES.append(f"{label}: expected {expected!r}, got {actual!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-test the vendored Matter registry.")
    parser.add_argument("--benchmark", help="SimuHome benchmark dir; enables full-corpus resolution check")
    args = parser.parse_args()

    reg = load_registry()
    print(f"registry: {reg.provenance}\n")

    print("-- name resolution (spec punctuation vs device-dump spelling) --")
    check("cluster 'OnOff' resolves (spec spells it 'On/Off')", 6, reg.cluster_by_name("OnOff")["id"])
    check("cluster 'On/Off' resolves", 6, reg.cluster_by_name("On/Off")["id"])
    check("cluster 'FanControl' resolves", 514, reg.cluster_by_name("FanControl")["id"])
    check("cluster 'RVCOperationalState' resolves", 97, reg.cluster_by_name("RVCOperationalState")["id"])

    print("\n-- SimuHome aliases --")
    for alias, canonical in CLUSTER_NAME_ALIASES.items():
        record = reg.cluster_by_name(alias)
        check(f"alias {alias!r} -> {canonical!r}", canonical, record["name"] if record else None)

    print("\n-- inherited attributes (derived clusters omit what they inherit) --")
    check(
        "RVCOperationalState.CountdownTime inherited from Operational State",
        "elapsed-s",
        reg.attribute_type("RVCOperationalState", "CountdownTime"),
    )
    check(
        "DishwasherAlarm.State inherited from Alarm Base",
        "AlarmBitmap",
        reg.attribute_type("DishwasherAlarm", "State"),
    )

    print("\n-- attribute writability (tri-state) --")
    check("FanControl.PercentSetting writable", True, reg.is_writable("FanControl", "PercentSetting"))
    check("FanControl.PercentCurrent read-only", False, reg.is_writable("FanControl", "PercentCurrent"))
    check("Thermostat.OccupiedCoolingSetpoint writable", True,
          reg.is_writable("Thermostat", "OccupiedCoolingSetpoint"))
    # Matter actuates via commands, not attribute writes: OnOff.OnOff is read-only
    # yet is the most-actuated property in the benchmark.
    check("OnOff.OnOff read-only but command-driven", False, reg.is_writable("OnOff", "OnOff"))
    onoff_commands = {c["name"] for c in reg.commands("OnOff")}
    check("OnOff exposes On/Off/Toggle commands", set(),
          {"On", "Off", "Toggle"} - onoff_commands)
    check("unknown attribute -> None", None, reg.is_writable("FanControl", "NoSuchAttribute"))
    writable_total = sum(
        1
        for record in reg.clusters.values()
        for attr in record["attributes"]
        if attr.get("writable") is True
    )
    check("writable attributes across registry", 159, writable_total)
    # Derived clusters restate inherited attributes with only id+name, so the
    # merge must fill type/access from the base rather than replacing the record.
    check("inherited attributes keep their base type", "uint8",
          reg.attribute_type("DishwasherMode", "CurrentMode"))
    check("inherited attributes keep their base access", True,
          reg.is_writable("DishwasherMode", "StartUpMode"))
    check("CurrentMode is read-only (driven by ChangeToMode)", False,
          reg.is_writable("RVCRunMode", "CurrentMode"))

    print("\n-- global attributes exist on every cluster --")
    check("Identify.ClusterRevision is global", True, reg.is_global("Identify", "ClusterRevision"))
    check("FanControl.FeatureMap is global", True, reg.is_global("FanControl", "FeatureMap"))
    check("FanControl.PercentSetting is NOT global", False, reg.is_global("FanControl", "PercentSetting"))

    print("\n-- converter scaling, derived from spec types --")
    # temperature: centi-degrees Celsius
    check("Thermostat.LocalTemperature 2450 -> 24.5 C",
          24.5, conv._attribute_value("Thermostat", "LocalTemperature", 2450))
    check("Thermostat unit", "°C", conv._attribute_unit("Thermostat", "LocalTemperature"))
    # freezers are legitimately negative
    check("TemperatureControl.TemperatureSetpoint -2200 -> -22.0 C",
          -22.0, conv._attribute_value("TemperatureControl", "TemperatureSetpoint", -2200))
    # percent: already 0..100, must NOT be divided
    check("FanControl.PercentSetting 90 -> 90 (unscaled)",
          90, conv._attribute_value("FanControl", "PercentSetting", 90))
    check("FanControl unit", "%", conv._attribute_unit("FanControl", "PercentSetting"))
    # percent100ths: 0..10000
    check("WindowCovering.CurrentPositionLiftPercent100ths 10000 -> 100.0",
          100.0, conv._attribute_value("WindowCovering", "CurrentPositionLiftPercent100ths", 10000))
    # hundredths-of-a-percent humidity
    check("RelativeHumidityMeasurement.MeasuredValue 5500 -> 55.0",
          55.0, conv._attribute_value("RelativeHumidityMeasurement", "MeasuredValue", 5500))
    check("RelativeHumidityMeasurement device_class", "humidity",
          conv._attribute_class("RelativeHumidityMeasurement", "MeasuredValue"))
    # uint8 level must not be treated as a percent
    check("LevelControl.CurrentLevel 180 -> 180 (uint8, unscaled)",
          180, conv._attribute_value("LevelControl", "CurrentLevel", 180))
    check("LevelControl.CurrentLevel has no unit", None,
          conv._attribute_unit("LevelControl", "CurrentLevel"))
    # booleans must survive untouched
    check("OnOff.OnOff True stays boolean", True, conv._attribute_value("OnOff", "OnOff", True))

    print("\n-- boilerplate filter --")
    for cluster, attribute in (
        ("BasicInformation", "VendorName"),
        ("Descriptor", "ServerList"),
        ("Identify", "IdentifyType"),
        ("FanControl", "FeatureMap"),
    ):
        check(f"{cluster}.{attribute} is boilerplate", True,
              conv._is_boilerplate_attribute(cluster, attribute))
    for cluster, attribute in (
        ("FanControl", "PercentSetting"),
        ("OnOff", "OnOff"),
        ("Thermostat", "SystemMode"),
    ):
        check(f"{cluster}.{attribute} is kept", False,
              conv._is_boilerplate_attribute(cluster, attribute))

    if args.benchmark:
        print("\n-- full-corpus resolution --")
        unresolved: dict[str, int] = {}
        total = 0
        for path in sorted(glob.glob(os.path.join(args.benchmark, "*.json"))):
            with open(path, "r", encoding="utf-8") as handle:
                episode = json.load(handle)
            rooms = ((episode.get("initial_home_config") or {}).get("rooms")) or {}
            for room in rooms.values():
                for device in (room.get("devices") or []):
                    for attr_path in (device.get("attributes") or {}):
                        parts = attr_path.split(".", 2)
                        if len(parts) != 3:
                            continue
                        total += 1
                        if reg.attribute(parts[1], parts[2]) is None:
                            unresolved[f"{parts[1]}.{parts[2]}"] = unresolved.get(f"{parts[1]}.{parts[2]}", 0) + 1
        check(f"all {total} attribute paths resolve", 0, len(unresolved))
        for key, count in sorted(unresolved.items())[:10]:
            print(f"        unresolved: {key} x{count}")

    print()
    if FAILURES:
        print(f"SELFTEST FAILED ({len(FAILURES)}):", file=sys.stderr)
        for failure in FAILURES:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
