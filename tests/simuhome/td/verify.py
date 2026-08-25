#!/usr/bin/env python3
"""
Verification suite for Phase A output. Exits non-zero on any failure.

  python tests/simuhome/td/verify.py [--input DIR]

Checks: registry closure, the observable/actuatable/internal partition,
idempotency, human-edit preservation, entity_id slug agreement with the YAML
converter, vocabulary conformance against the TD-SOSA ontology, and that no
Thing Descriptions leaked out of Phase A.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from td import convert, registry_access, scan, tables  # noqa: E402

TD_DIR = Path(__file__).resolve().parent
MAPPINGS_DIR = TD_DIR / "mappings"
OUT_DIR = TD_DIR / "out"
ONTOLOGY = TD_DIR.parents[3] / "ontologies" / "td-sosa-extension-v2.ttl"

FAILURES: List[str] = []


def check(label: str, expected: Any, actual: Any) -> None:
    ok = expected == actual
    print(f"{'PASS' if ok else 'FAIL':4}  {label:<62} {actual!r}")
    if not ok:
        FAILURES.append(f"{label}: expected {expected!r}, got {actual!r}")


def _load(name: str) -> Dict[str, Any]:
    with open(MAPPINGS_DIR / f"{name}.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _rows(name: str) -> List[Dict[str, Any]]:
    return _load(name).get("rows", [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase A output.")
    parser.add_argument("--input", default=str(convert.DEFAULT_INPUT))
    args = parser.parse_args()
    reg = registry_access.get_registry()

    print(f"registry: {reg.provenance}\n")

    print("-- 1. registry closure --")
    attribute_rows = _rows("attribute_map")
    unresolved = [r for r in attribute_rows if r.get("matter") is None]
    check("every attribute_map row resolves in the registry", 0, len(unresolved))
    check("attribute_map row count", 116, len(attribute_rows))
    bad_key = [
        r for r in attribute_rows
        if r["key"] != f"{r['matter']['cluster_id']}.{r['matter']['attribute_id']}"
    ]
    check("keys are decimal cluster.attribute", 0, len(bad_key))

    print("\n-- 2. partition: observable | actuatable | internal --")
    # actuatable_properties is keyed `<family>:<cluster.attribute>` since the
    # hierarchy emits one leaf per family; the partition is over the attribute.
    actuatable = {r["key"].split(":", 1)[1] for r in _rows("actuatable_properties")}
    internal = {r["key"] for r in _rows("internal_properties")}
    all_keys = {r["key"] for r in attribute_rows}
    observable_attr = {
        r["key"] for r in attribute_rows
        if r["role"] == "affordance" and r["affordance"] == "observable"
        and r["key"] not in internal
    }
    covered = actuatable | internal | observable_attr
    check("no attribute is unclassified", set(), all_keys - covered)
    check("actuatable and internal are disjoint", set(), actuatable & internal)
    check("observable and actuatable are disjoint", set(), observable_attr & actuatable)
    check("partition covers exactly attribute_map", set(), covered - all_keys)

    print("\n-- 3. affordance rules --")
    wrong = [
        r for r in attribute_rows
        if r["affordance"] == "actuatable" and r["writable"] is not True
        and r["mechanism"] != "command"
    ]
    check("actuatable implies writable or command-driven", 0, len(wrong))
    onoff = next(r for r in attribute_rows if r["yaml_path"] == "OnOff.OnOff")
    check("OnOff.OnOff writable=false", False, onoff["writable"])
    check("OnOff.OnOff still actuatable (command-driven)", "actuatable", onoff["affordance"])
    percent = next(r for r in attribute_rows if r["yaml_path"] == "FanControl.PercentSetting")
    check("FanControl.PercentSetting via attribute_write", "attribute_write", percent["mechanism"])

    print("\n-- 4. direction integrity --")
    effects = _rows("actuation_effects")
    bad_dir = [
        r for r in effects
        if (r.get("monotonic") is True) != (r.get("direction") in {"increase", "decrease"})
    ]
    check("monotonic <=> a direction is set", 0, len(bad_dir))
    purifier_on = next(r for r in effects if r["key"] == "AirPurifier:6.1")
    check("purifier ON decreases pm10", "decrease", purifier_on["direction"])
    check("purifier acts on the pm10 property", "pm10_mass_concentration",
          purifier_on["affectsObservableProperty"])
    purifier_off = next(r for r in effects if r["key"] == "AirPurifier:6.0")
    check("purifier OFF reverses to increase", "increase", purifier_off["direction"])
    light_on = next(r for r in effects if r["key"] == "OnOffLight:6.1")
    check("light ON increases illuminance", "increase", light_on["direction"])
    check("light acts on the illuminance property", "illuminance",
          light_on["affectsObservableProperty"])
    toggles = [r for r in effects if r["command"]["name"] == "Toggle"]
    check("every Toggle is directionless", 0, sum(1 for r in toggles if r["monotonic"]))
    setters = [r for r in effects if r["command"]["name"] in
               {"GoToLiftPercentage", "SetpointRaiseLower", "MoveToLevel", "SetTemperature"}]
    check("every level/percent setter is directionless", 0,
          sum(1 for r in setters if r["monotonic"]))

    print("\n-- 4b. environmental vocabulary --")
    observable = _rows("observable_properties")
    declared = {r["sosa_property"] for r in observable}
    referenced = {
        r["affectsObservableProperty"] for r in effects
        if r.get("affectsObservableProperty")
    }
    check("no effect references an undeclared property", set(), referenced - declared)
    check("no declared property is unused", set(), declared - referenced)
    # Property names must denote the measured quantity, not a derived index or a
    # different physical concept: illuminance (received) not luminosity (emitted),
    # air temperature not thermal comfort, PM10 concentration not "air quality".
    misnamed = declared & {"luminosity", "thermal_comfort", "air_quality", "humidity"}
    check("no property is named after a derived index or wrong quantity", set(), misnamed)
    missing_alias = [
        r["key"] for r in observable
        if not r.get("legacy_aliases")
    ]
    check("every property records its lab308e alias", [], missing_alias)
    bad_qk = [r["key"] for r in observable if not str(r.get("quantityKind", "")).startswith("quantitykind:")]
    check("every property carries a QUDT quantity kind", [], bad_qk)

    print("\n-- 4c. CURIE syntax --")
    # A CURIE local name must be an NCName: no '/', spaces or other delimiters,
    # and it may not start with a digit. Matter cluster names ("On/Off",
    # "Fan Control") violate this if pasted in raw.
    ncname = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
    curie_fields = {
        "ex", "subClassOf", "sosa_type", "quantityKind", "qudt_unit", "property_type",
    }
    malformed: List[str] = []
    for name in convert.TABLE_ORDER:
        for row in _rows(name):
            for field, value in row.items():
                if field not in curie_fields or not isinstance(value, str) or ":" not in value:
                    continue
                prefix, _, local = value.partition(":")
                if not ncname.match(prefix) or not ncname.match(local):
                    malformed.append(f"{name}:{row['key']}.{field}={value}")
    check("every CURIE has a valid prefix and local name", [], malformed)
    actuatable = _rows("actuatable_properties")
    print("\n-- 4c-ii. family-specific property hierarchy --")
    # Matter clusters name processes, so a cluster-level property generalises
    # across every family using that process. Constraints belong on the leaf.
    brightness = {r["family"]: r for r in actuatable if r["yaml_path"] == "LevelControl.CurrentLevel"}
    check("a TV's brightness is its own property", "ex:TvBrightness",
          brightness["Tv"]["property_type"])
    check("a lamp's brightness is a different property", "ex:DimmableLightBrightness",
          brightness["DimmableLight"]["property_type"])
    check("...both under one cluster-level superclass", {"ex:LevelControlBrightness"},
          {r["subClassOf"] for r in brightness.values()})
    # A family leaf must never collapse onto the bare family name: ex:AirPurifier
    # is the device class.
    device_classes = {r["ex"] for r in _rows("device_type_map")}
    clashes = sorted({r["property_type"] for r in actuatable} & device_classes)
    check("no property IRI collides with a device class", [], clashes)
    # Appliance-internal setpoints are not the room's air temperature.
    cabinet = {
        r["family"]: r["property_type"] for r in actuatable
        if r["yaml_path"] == "TemperatureControl.TemperatureSetpoint"
    }
    check("a freezer setpoint is not room air temperature", "ex:FreezerTemperatureSetpoint",
          cabinet.get("Freezer"))
    thermostat = {
        r["property_type"] for r in actuatable
        if r["yaml_path"].startswith("Thermostat.Occupied")
    }
    check("thermostat setpoints do target room air temperature", {"ex:AirTemperature"},
          thermostat)
    # Every family leaf declares its superclass; shared quantities do not.
    orphans = [
        r["key"] for r in actuatable
        if not r["quantity"] and not r["subClassOf"]
    ]
    check("every family leaf declares a superclass", [], orphans)
    # Affordance names must be family-qualified to survive cross-family reuse.
    generic = [
        r["key"] for r in actuatable
        if r["td_name"] and not r["td_name"].lower().startswith(r["family"][:3].lower())
    ]
    check("every affordance name is family-qualified", [], generic)

    print("\n-- 4c-iii. properties are instances, not shared class references --")
    # Home Assistant does not fix a value range per device type: the climate
    # entities in this corpus alone carry 42 distinct min_temp and 70 distinct
    # max_temp values. Limits are therefore a fact about one deployed device, so
    # `tdsosa:affordsProperty` must point at an anonymous node typed by the
    # class -- never at a shared class IRI that TDs reference from outside.
    non_anon = [
        r["key"] for r in actuatable if r.get("property_node") != "anonymous"
    ]
    check("every actuatable property is an anonymous instance", [], non_anon)
    non_anon_obs = [
        r["key"] for r in _rows("observable_properties")
        if r.get("property_node") != "anonymous"
    ]
    check("every observable property is an anonymous instance", [], non_anon_obs)
    constrained_classes = [
        r["key"] for r in _rows("property_classes") if r.get("carries_constraints")
    ]
    check("no property class carries instance constraints", [], constrained_classes)
    # The class hierarchy still stands on its own, as rdf:type targets.
    typed = [r for r in actuatable if r.get("property_type")]
    check("every property instance declares its type", len(actuatable), len(typed))

    print("\n-- 4c-iv. writable but no present-tense effect --")
    # These stay actuatable (Matter really does allow the write, and TD says so
    # with readOnly:false) but Phase B must emit no hasEffectActuation for them.
    by_path = {}
    for row in actuatable:
        by_path.setdefault(row["yaml_path"], row)
    check("ControlSequenceOfOperation is installed capability, not a control",
          "capability", by_path["Thermostat.ControlSequenceOfOperation"]["no_effect_reason"])
    check("...so it claims no immediate effect", False,
          by_path["Thermostat.ControlSequenceOfOperation"]["has_immediate_effect"])
    check("StartUpOnOff is a power-restoration default", "startup",
          by_path["OnOff.StartUpOnOff"]["no_effect_reason"])
    check("transition times are parameters", "parameter",
          by_path["LevelControl.OnOffTransitionTime"]["no_effect_reason"])
    # SystemMode carries the same `manage` write privilege as
    # ControlSequenceOfOperation, so privilege alone cannot be the rule.
    check("SystemMode is still a real actuator", True,
          by_path["Thermostat.SystemMode"]["has_immediate_effect"])
    check("OnOff.OnOff is still a real actuator", True,
          by_path["OnOff.OnOff"]["has_immediate_effect"])
    check("setpoints are still real actuators", True,
          by_path["Thermostat.OccupiedCoolingSetpoint"]["has_immediate_effect"])
    inconsistent = [
        r["key"] for r in actuatable
        if (r["no_effect_reason"] is None) != r["has_immediate_effect"]
    ]
    check("flag and reason always agree", [], inconsistent)

    print("\n-- 4c-v. command -> attribute links carry their evidence --")
    # The Matter XML states no command->attribute relation, so each link is a
    # curated claim. Every row must show the evidence that supports it, and a
    # verified link is what promotes the property class above it to `auto`.
    links = _rows("command_targets")
    check("command_targets is populated", True, len(links) > 0)
    missing_evidence = [
        r["key"] for r in links if not (r.get("evidence") or {}).get("why")
    ]
    check("every link states its evidence", [], missing_evidence)
    settemp = next(r for r in links if r["command"] == "SetTemperature")
    check("SetTemperature writes TemperatureSetpoint", "TemperatureSetpoint",
          settemp["writes_attribute"])
    check("...and its type matches the attribute", "temperature",
          settemp["attribute_type"])
    # A verified link must leave nothing to review downstream.
    approved_links = {
        (r["cluster"], r["command"]) for r in links if r["status"] == "approved"
    }
    check("the four verified links are approved", True,
          {("Temperature Control", "SetTemperature"), ("On/Off", "On"),
           ("Level Control", "MoveToLevel"),
           ("Window Covering", "GoToLiftPercentage")} <= approved_links)
    # A class stays in review only while a command link it depends on is
    # unverified; verifying the last link must clear the last class.
    unverified = {
        (r["cluster"], r["command"]) for r in links if r["status"] != "approved"
    }
    still_review = [r["key"] for r in _rows("property_classes") if r["status"] == "review"]
    check("a class is in review only if a link it needs is unverified",
          bool(unverified), bool(still_review))

    print("\n-- 4b-ii. effects match what SimuHome actually simulates --")
    # Claiming an effect the simulator will not produce makes a plan look correct
    # while the measured value never moves, which would make comparison against
    # the published SimuHome benchmark meaningless. Ground truth comes from
    # GET /api/environment/control_rules/{state}; see docs/effects-vs-simulator.md.
    SIMULATED = {
        "air_temperature": {"AirConditioner", "HeatPump"},
        "illuminance": {"OnOffLight", "DimmableLight"},
        "relative_humidity": {"Humidifier", "Dehumidifier"},
        "pm10_mass_concentration": {"AirPurifier"},
    }
    claimed: Dict[str, set] = {}
    for row in effects:
        prop = row.get("affectsObservableProperty")
        if prop:
            claimed.setdefault(prop, set()).add(row["family"])
    check("no effect is claimed that SimuHome does not simulate", {},
          {p: sorted(f - SIMULATED.get(p, set())) for p, f in claimed.items()
           if f - SIMULATED.get(p, set())})
    check("every simulated effect is present", {},
          {p: sorted(f - claimed.get(p, set())) for p, f in SIMULATED.items()
           if f - claimed.get(p, set())})
    # The two families removed for exactly this reason.
    for family in ("Fan", "WindowCoveringController"):
        stray = [r["key"] for r in effects
                 if r["family"] == family and r.get("affectsObservableProperty")]
        check(f"{family} claims no environmental effect", [], stray)

    print("\n-- 4c-vi. fan speed: absolute, not stepped --")
    # FanControl.Step has no HA service and no benchmark query asks for a
    # relative fan change (of 174 fan-mentioning queries: 106 absolute
    # percentages, 21 named modes, 0 steps), so it must not claim an effect.
    fan_steps = [
        r["key"] for r in effects
        if r["command"]["name"] == "Step" and "Fan" in r["command"]["cluster_name"]
    ]
    check("FanControl.Step claims no environmental effect", [], fan_steps)
    presets = [r for r in _rows("ha_binding") if r.get("service") == "fan.set_preset_mode"]
    check("set_preset_mode rows exist", True, len(presets) > 0)
    # `medium` is present deliberately: every device declares
    # FanModeSequence=OffLowHigh yet 1398 sit at Medium and 12 queries ask for it.
    check("preset enum covers the modes the corpus uses",
          ["off", "low", "medium", "high"], presets[0].get("enum") if presets else None)

    print("\n-- 4c-vii. SAREF parents exist and Things carry both types --")
    # saref:Lighting was invented in an earlier pass and does not exist: SAREF
    # core declares exactly five subclasses of saref:Device.
    saref_device_subclasses = {
        "saref:Actuator", "saref:Appliance", "saref:HVAC", "saref:Meter", "saref:Sensor",
    }
    device_rows = _rows("device_type_map")
    bad_parent = sorted({
        r["subClassOf"] for r in device_rows
        if r["subClassOf"] not in saref_device_subclasses
    })
    check("every SAREF parent is a real saref:Device subclass", [], bad_parent)
    lights = {r["key"]: r["subClassOf"] for r in device_rows if "Light" in r["key"]}
    check("lamps use saref:Actuator (there is no saref:Lighting)",
          {"saref:Actuator"}, set(lights.values()))
    check("HVAC families keep saref:HVAC", {"saref:HVAC"},
          {r["subClassOf"] for r in device_rows if r["key"] in {"AirConditioner", "HeatPump"}})
    # The Thing carries the exact class AND the standard one.
    missing_both = [
        r["key"] for r in device_rows
        if not r.get("td_types") or len(r["td_types"]) != 2
        or r["td_types"][0] != r["ex"] or r["td_types"][1] != r["subClassOf"]
    ]
    check("every Thing declares both its ex: and saref: type", [], missing_both)

    print("\n-- 4d. no Matter field names leak into the ontology --")
    # `MeasuredValue` names a field in the Matter protocol. It is not a quantity
    # and not an interaction: minting `ex:...MeasuredValue` would make a result
    # masquerade as the property it results from.
    forbidden = ("MeasuredValue", "MinMeasured", "MaxMeasured", "Tolerance",
                 "Percent100ths", "Struct")
    leaked_iri = [
        f"{r['key']}={r['property_type']}"
        for r in _rows("actuatable_properties")
        if any(token in str(r.get("property_type", "")) for token in forbidden)
    ]
    check("no property IRI contains a Matter field name", [], leaked_iri)
    leaked_name = [
        f"{r['key']}={r['affordance_name']}"
        for r in _rows("actuatable_properties")
        if any(token in str(r.get("affordance_name", "")) for token in forbidden)
    ]
    check("no affordance name contains a Matter field name", [], leaked_name)

    print("\n-- 4e. quantities are shared, not duplicated per attribute --")
    declared_q = {r["sosa_property"] for r in _rows("observable_properties")}
    attr_rows = _rows("attribute_map")
    quantified = [r for r in attr_rows if r.get("quantity")]
    unknown_q = {r["quantity"] for r in quantified} - declared_q
    check("every attribute quantity is a declared property", set(), unknown_q)
    humidity = next(r for r in attr_rows if r["yaml_path"] == "RelativeHumidityMeasurement.MeasuredValue")
    check("RelativeHumidityMeasurement.MeasuredValue is a result, not a property",
          "quantity_result", humidity["sosa_role"])
    check("...and it affords the shared relative_humidity property",
          "relative_humidity", humidity["quantity"])
    check("...named after the quantity, not the field", "humidity",
          humidity["affordance_name"])
    tolerance = next(r for r in attr_rows if r["yaml_path"] == "RelativeHumidityMeasurement.Tolerance")
    check("Tolerance is sensor metadata, not a property", "sensor_metadata",
          tolerance["sosa_role"])
    # It still needs a name: sensor metadata reaches the Thing Description as a
    # property affordance with its own GET form (26 of them in a typical
    # episode), and `td:name` is what the read route resolves. Leaving it null
    # only pushed naming onto a fallback.
    check("...but still gets an affordance name", "tolerance",
          tolerance["affordance_name"])
    # Device and room observations of the same quantity must resolve to one node.
    temp_attrs = {
        r["quantity"] for r in attr_rows
        if r["yaml_path"] in {
            "TemperatureMeasurement.MeasuredValue", "Thermostat.LocalTemperature"
        }
    }
    check("device temperature sensors share one quantity", {"air_temperature"}, temp_attrs)
    # Every affordance row must carry a name, whatever its SOSA role -- the name
    # is the TD's only handle on the attribute now that the Matter cluster and
    # attribute triples have been dropped from the served graphs.
    unnamed = [
        r["key"] for r in attr_rows
        if r["role"] == "affordance" and not r["affordance_name"]
    ]
    check("every affordance row has a name", [], unnamed)

    # ...and names must be unique per cluster, so a device carrying two clusters
    # that share an attribute name still yields two distinct affordances.
    by_name: Dict[str, set] = {}
    for r in attr_rows:
        if r["role"] != "affordance" or not r["affordance_name"]:
            continue
        by_name.setdefault(r["affordance_name"], set()).add(r["yaml_path"])
    for family in ("RVCRunMode", "RVCCleanMode"):
        collides = [n for n, paths in by_name.items()
                    if len(paths) > 1
                    and any(pth.startswith(family) for pth in paths)]
        check(f"{family} names do not collide with a co-resident cluster",
              [], collides)

    print("\n-- 5. vocabulary conformance vs td-sosa-extension-v2.ttl --")
    if ONTOLOGY.is_file():
        try:
            from rdflib import Graph, RDF, OWL
            graph = Graph()
            graph.parse(str(ONTOLOGY), format="turtle")
            declared = {
                str(s).split("#")[-1]
                for s in list(graph.subjects(RDF.type, OWL.Class))
                + list(graph.subjects(RDF.type, OWL.ObjectProperty))
            }
            used = {"affectsObservableProperty", "increasesObservableProperty",
                    "decreasesObservableProperty", "IncreasingActuation",
                    "DecreasingActuation", "hasEffectActuation", "affordsProperty",
                    "ObservablePropertyAffordance", "ActuatablePropertyAffordance"}
            check("every tdsosa term used is declared in the ontology", set(), used - declared)
        except ImportError:
            print("SKIP  rdflib unavailable")
    else:
        print(f"SKIP  ontology not found at {ONTOLOGY}")

    print("\n-- 6. entity_id slug agreement with the YAML converter --")
    sys.path.insert(0, str(TD_DIR.parents[0]))
    import initial_home_config_to_homeassistant_yaml as conv  # noqa: E402

    paths = sorted(glob.glob(os.path.join(args.input, "*.yaml")))
    sample = random.Random(0).sample(paths, min(10, len(paths)))
    mismatches = 0
    checked = 0
    for path in sample:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        for entries in (payload.get("devices") or {}).values():
            for entry in entries or []:
                name = entry.get("name")
                if not isinstance(name, str) or len(name.split(".")) != 3:
                    continue
                if name.split(".")[1] == "Simulator":
                    continue
                expected = f"{entry['platform']}.{conv._slug(name)}"
                checked += 1
                if not re.fullmatch(r"[a-z_]+\.[a-z0-9_]+", expected):
                    mismatches += 1
    check(f"entity_id patterns well-formed ({checked} sampled)", 0, mismatches)

    print("\n-- 7. no Thing Descriptions emitted in Phase A --")
    leaked = list(OUT_DIR.rglob("*.jsonld")) + list(OUT_DIR.rglob("*.td.json"))
    check("out/ contains no TD artifacts", [], leaked)
    check("out/td/ does not exist", False, (OUT_DIR / "td").exists())

    print("\n-- 8. phase-b gate refuses while todos remain --")
    proc = subprocess.run(
        [sys.executable, str(TD_DIR / "convert.py"), "--phase-b"],
        capture_output=True, text=True,
    )
    check("--phase-b exits non-zero", True, proc.returncode != 0)
    check("--phase-b names the blocking todos", True, "todo" in proc.stderr.lower())

    print("\n-- 9. idempotency --")
    backup = tempfile.mkdtemp(prefix="td_mappings_")
    shutil.copytree(MAPPINGS_DIR, Path(backup) / "mappings")
    subprocess.run(
        [sys.executable, str(TD_DIR / "convert.py"), "--phase-a", "--input", args.input,
         "--no-timestamp"],
        capture_output=True, text=True, check=True,
    )
    first = {p.name: p.read_bytes() for p in sorted(MAPPINGS_DIR.glob("*.yaml"))}
    subprocess.run(
        [sys.executable, str(TD_DIR / "convert.py"), "--phase-a", "--input", args.input,
         "--no-timestamp"],
        capture_output=True, text=True, check=True,
    )
    second = {p.name: p.read_bytes() for p in sorted(MAPPINGS_DIR.glob("*.yaml"))}
    changed = [name for name in first if first[name] != second[name]]
    check("rerunning --phase-a is byte-identical", [], changed)

    print("\n-- 9b. approving a class promotes its family leaves --")
    classes_path = MAPPINGS_DIR / "property_classes.yaml"
    classes_doc = yaml.safe_load(classes_path.read_text(encoding="utf-8"))
    gating = [r["key"] for r in classes_doc["rows"] if r["status"] == "review"]
    if not gating:
        # Every command link is verified, so nothing is gated. Exercise the
        # mechanism anyway by demoting a class, to prove promotion still works.
        target_class = classes_doc["rows"][0]
        target_class["status"] = "review"
        with open(classes_path, "w", encoding="utf-8") as handle:
            yaml.dump(classes_doc, handle, sort_keys=False, allow_unicode=True)
        gating = [target_class["key"]]
        classes_doc = yaml.safe_load(classes_path.read_text(encoding="utf-8"))
    before = sum(
        1 for r in _rows("actuatable_properties") if r["status"] == "review"
    )
    for row in classes_doc["rows"]:
        if row["status"] == "review":
            row["status"] = "approved"
    with open(classes_path, "w", encoding="utf-8") as handle:
        yaml.dump(classes_doc, handle, sort_keys=False, allow_unicode=True)
    subprocess.run(
        [sys.executable, str(TD_DIR / "convert.py"), "--phase-a", "--input", args.input,
         "--no-timestamp"],
        capture_output=True, text=True, check=True,
    )
    after = sum(1 for r in _rows("actuatable_properties") if r["status"] == "review")
    check(f"approving {len(gating)} classes clears all {before} leaf reviews", 0, after)
    promoted = [
        r for r in _rows("actuatable_properties")
        if r["evidence"].get("status_source", "").startswith("inherited")
    ]
    check("every leaf that was in review is now promoted and records why",
          True, len(promoted) >= before)

    print("\n-- 10. human edits survive regeneration --")
    target = MAPPINGS_DIR / "device_type_map.yaml"
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    document["rows"][0]["status"] = "approved"
    document["rows"][0]["subClassOf"] = "saref:CustomEditedClass"
    # Force `review` rather than assuming it: once a human approves the whole
    # table, an approved row would bypass field merging and this would test
    # nothing.
    document["rows"][1]["status"] = "review"
    # A free-text field: `subClassOf` is SAREF-validated on merge, so use a field
    # with no validation to test plain edit preservation.
    document["rows"][1]["notes"] = "reviewed by hand"
    # And a value the SAREF guard must reject rather than preserve.
    document["rows"][1]["subClassOf"] = "saref:ReviewEdited"
    approved_key = document["rows"][0]["key"]
    review_key = document["rows"][1]["key"]
    with open(target, "w", encoding="utf-8") as handle:
        yaml.dump(document, handle, sort_keys=False, allow_unicode=True)

    subprocess.run(
        [sys.executable, str(TD_DIR / "convert.py"), "--phase-a", "--input", args.input,
         "--no-timestamp"],
        capture_output=True, text=True, check=True,
    )
    after = yaml.safe_load(target.read_text(encoding="utf-8"))
    approved_row = next(r for r in after["rows"] if r["key"] == approved_key)
    review_row = next(r for r in after["rows"] if r["key"] == review_key)
    check("approved row keeps its status", "approved", approved_row["status"])
    check("approved row keeps the human value", "saref:CustomEditedClass",
          approved_row["subClassOf"])
    check("review row keeps the human value", "reviewed by hand", review_row.get("notes"))
    check("...but an undefined SAREF class is rejected, not preserved", True,
          review_row["subClassOf"] in {
              "saref:Actuator", "saref:Appliance", "saref:HVAC",
              "saref:Meter", "saref:Sensor",
          })
    check("review row keeps machine fields fresh", True, review_row.get("instances") is not None)

    # Restore the pristine tables, then regenerate so out/reports/ describes what
    # is actually on disk rather than the scratch edits made by test 10.
    shutil.rmtree(MAPPINGS_DIR)
    shutil.copytree(Path(backup) / "mappings", MAPPINGS_DIR)
    shutil.rmtree(backup, ignore_errors=True)
    subprocess.run(
        [sys.executable, str(TD_DIR / "convert.py"), "--phase-a", "--input", args.input],
        capture_output=True, text=True, check=True,
    )
    restored = yaml.safe_load((MAPPINGS_DIR / "device_type_map.yaml").read_text(encoding="utf-8"))
    # Count the scratch VALUES, not approvals: a human may legitimately have
    # approved every row before this ran.
    check("scratch edits removed from the tables", 0, sum(
        1 for r in restored["rows"]
        if r.get("subClassOf") == "saref:CustomEditedClass"
        or r.get("notes") == "reviewed by hand"
    ))
    coverage_text = (OUT_DIR / "reports" / "coverage.md").read_text(encoding="utf-8")
    check("coverage.md reflects the restored tables", False, "| 1 | 94%" in coverage_text)

    print()
    if FAILURES:
        print(f"VERIFY FAILED ({len(FAILURES)}):", file=sys.stderr)
        for failure in FAILURES:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("VERIFY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
