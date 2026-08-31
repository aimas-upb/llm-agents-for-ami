#!/usr/bin/env python3
"""
Verify SHTD phase 4 -- TD-SOSA semantics: what senses and what actuates what.

Three parts:

  A. Sensing      -- families that read the room's air are `sosa:Sensor`s
                     observing the ROOM's observable property; families that
                     read their own compartment observe their own interior. A
                     freezer reading -15 C inside a 23.6 C kitchen must not be
                     recorded as an observer of the kitchen.
  B. Actuation    -- the approved effect rows appear on the action affordances
                     phase 3 created, with the settled direction rule.
  C. Reconcile    -- what the graph claims matches the simulator's own causal
                     model at GET /api/environment/control_rules/{state}.

    python ami_agents/environment/integration/SimuHome/verify_phase4.py \\
        --shtd http://127.0.0.1:8097 --sim http://127.0.0.1:44275/api
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from rdflib import Graph, Namespace, RDF, URIRef

try:
    from .mappings import load_mappings
    from .sim_client import SimuHomeClient
except ImportError:  # pragma: no cover
    from mappings import load_mappings  # type: ignore
    from sim_client import SimuHomeClient  # type: ignore

TD = Namespace("https://www.w3.org/2019/wot/td#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")
HOME = Namespace("http://example.org/homeont/")

PASS, FAIL = "PASS", "FAIL"
_results: List[tuple] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SHTD phase 4.")
    parser.add_argument("--shtd", default="http://127.0.0.1:8097")
    parser.add_argument("--sim", default="http://127.0.0.1:44275/api")
    args = parser.parse_args()
    base = args.shtd.rstrip("/")

    sim = SimuHomeClient(args.sim)
    state = sim.home_state()
    rooms = state.get("rooms") or {}
    mappings = load_mappings()

    import json as _json
    import urllib.request

    def status() -> Dict[str, Any]:
        with urllib.request.urlopen(f"{base}/_shtd/status", timeout=20) as r:
            return _json.load(r)

    home = status()["home"]

    graphs: Dict[str, Graph] = {}
    device_rooms: Dict[str, str] = {}
    device_types: Dict[str, str] = {}
    for room_id, room in rooms.items():
        for device in room.get("devices") or []:
            device_id = str(device.get("device_id"))
            uri = f"{base}/workspaces/{home}/{room_id}/artifacts/{device_id}"
            g = Graph()
            g.parse(uri, format="turtle")
            graphs[device_id] = g
            device_rooms[device_id] = room_id
            device_types[device_id] = str(device.get("device_type"))

    def art(device_id: str) -> URIRef:
        room_id = device_rooms[device_id]
        return URIRef(f"{base}/workspaces/{home}/{room_id}/artifacts/{device_id}#artifact")

    print("A. Sensing\n")
    sensors_expected = {
        d for d, t in device_types.items() if mappings.senses(t) is not None}
    sensors_typed = {
        d for d, g in graphs.items() if (art(d), RDF.type, SOSA.Sensor) in g}
    check("every sensing family is typed sosa:Sensor",
          sensors_expected == sensors_typed,
          f"{len(sensors_typed)} typed, {len(sensors_expected)} expected")

    wrong_target, missing_inverse = [], []
    for device_id in sorted(sensors_expected):
        g = graphs[device_id]
        row = mappings.senses(device_types[device_id]) or {}
        room_id = device_rooms[device_id]
        observed = list(g.objects(art(device_id), SOSA.observes))
        if len(observed) != 1:
            wrong_target.append(f"{device_id}: {len(observed)} observes")
            continue
        target = str(observed[0])
        room_prop = f"{base}/workspaces/{home}/{room_id}#{row['sosa_property']}"
        if row.get("sensesRoomProperty"):
            if target != room_prop:
                wrong_target.append(f"{device_id}: {target}")
        else:
            # A compartment sensor must NOT claim the room's property.
            if target == room_prop:
                wrong_target.append(f"{device_id}: claims the room's property")
            elif "#interior" not in str(
                    next(g.objects(art(device_id), SOSA.hasFeatureOfInterest), "")):
                wrong_target.append(f"{device_id}: no interior feature-of-interest")
        if (URIRef(target), SOSA.isObservedBy, art(device_id)) not in g:
            missing_inverse.append(device_id)
    check("each sensor observes the right property", not wrong_target,
          f"{len(wrong_target)}: {wrong_target[:3]}")
    check("the inverse sosa:isObservedBy is stated", not missing_inverse,
          f"{len(missing_inverse)} missing")

    # The headline claim: a freezer is not an observer of its kitchen.
    compartment = [
        d for d, t in device_types.items()
        if (mappings.senses(t) or {}).get("sensesRoomProperty") is False]
    leaked = []
    for device_id in compartment:
        g = graphs[device_id]
        room_id = device_rooms[device_id]
        room_env = URIRef(f"{base}/workspaces/{home}/{room_id}#environment")
        if (art(device_id), SOSA.hasFeatureOfInterest, room_env) in g:
            leaked.append(device_id)
    check("compartment sensors do not observe the room", not leaked,
          f"{len(leaked)} leaked: {leaked[:3]}"
          if leaked else f"{len(compartment)} checked")

    print("\nB. Actuation effects\n")
    effect_actions, no_effect_actions = 0, 0
    contradictory = []
    for device_id, g in graphs.items():
        for action in g.objects(art(device_id), TD.hasActionAffordance):
            actuations = list(g.objects(action, TDSOSA.hasEffectActuation))
            if not actuations:
                no_effect_actions += 1
                continue
            effect_actions += 1
            for actuation in actuations:
                kinds = set(g.objects(actuation, RDF.type))
                if {TDSOSA.IncreasingActuation, TDSOSA.DecreasingActuation} <= kinds:
                    contradictory.append(f"{device_id}: both directions")
            # One claim per property: an action must not say a variable both
            # rises and falls.
            per_property: Dict[str, Set[str]] = {}
            for actuation in actuations:
                for prop in g.objects(actuation, SOSA.actsOnProperty):
                    kinds = {str(k).split("#")[-1] for k in g.objects(actuation, RDF.type)}
                    per_property.setdefault(str(prop), set()).update(
                        kinds & {"IncreasingActuation", "DecreasingActuation"})
            for prop, kinds in per_property.items():
                if len(kinds) > 1:
                    contradictory.append(f"{device_id}: {prop.split('#')[-1]} {kinds}")
    check("actions carry effect actuations", effect_actions > 0,
          f"{effect_actions} with an effect, {no_effect_actions} without")
    check("no action claims a variable both rises and falls",
          not contradictory, f"{len(contradictory)}: {contradictory[:3]}")

    # Every actuation must point at a property the room actually declares.
    dangling = []
    for device_id, g in graphs.items():
        room_id = device_rooms[device_id]
        room_graph = Graph()
        room_graph.parse(f"{base}/workspaces/{home}/{room_id}", format="turtle")
        declared = {str(p) for p in room_graph.objects(None, SSN.hasProperty)}
        for action in g.objects(art(device_id), TD.hasActionAffordance):
            for actuation in g.objects(action, TDSOSA.hasEffectActuation):
                for prop in g.objects(actuation, SOSA.actsOnProperty):
                    if str(prop) not in declared:
                        dangling.append(f"{device_id}: {prop}")
    check("every actuation targets a property its room declares",
          not dangling, f"{len(dangling)}: {dangling[:2]}")

    print("\nC. Reconciliation with the simulator's own causal model\n")
    _STATE_TO_PROPERTY = {
        "temperature": "air_temperature",
        "humidity": "relative_humidity",
        "illuminance": "illuminance",
        "air_quality": "pm10_mass_concentration",
    }
    mismatches = []
    for sim_state, sosa_property in sorted(_STATE_TO_PROPERTY.items()):
        rules = sim.control_rules(sim_state)
        truth = {str(entry["device_type"]) for entry in rules}
        claimed = set()
        for device_id, g in graphs.items():
            room_id = device_rooms[device_id]
            target = URIRef(
                f"{base}/workspaces/{home}/{room_id}#{sosa_property}")
            for action in g.objects(art(device_id), TD.hasActionAffordance):
                for actuation in g.objects(action, TDSOSA.hasEffectActuation):
                    if (actuation, SOSA.actsOnProperty, target) in g:
                        claimed.add(device_types[device_id])
        # Only device types present in THIS episode can be claimed.
        present = {t for t in device_types.values()}
        expected = truth & present
        if claimed != expected:
            mismatches.append(
                f"{sim_state}: claimed {sorted(claimed)} vs simulator {sorted(expected)}")
        print(f"    {sim_state:14s} simulator={sorted(expected)} graph={sorted(claimed)}")
    check("graph claims match the simulator's control_rules",
          not mismatches, "; ".join(mismatches))

    # Device-type granularity is too coarse on its own: it passes as soon as ONE
    # action of a family claims the effect, which is how five of seven families
    # went on under-claiming. `FanControl.FanMode` participates in changing PM10
    # per the simulator, yet carried no claim. So check EVERY action the
    # simulator names.
    print("\n   per-action coverage:")
    from mappings import load_mappings as _lm
    _m = _lm()

    # Commands the tables deliberately leave unmodelled carry no effect claim by
    # design, so they must not count as under-claimed. See tables.UNMODELLED_COMMANDS.
    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "_shtd_tables",
            Path(__file__).resolve().parents[3] / "tests" / "simuhome" / "td" / "tables.py")
        _tables = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_tables)  # type: ignore[union-attr]
        unmodelled = set(getattr(_tables, "UNMODELLED_COMMANDS", ()))
    except Exception:  # noqa: BLE001 - the tables are a convenience here, not a dependency
        unmodelled = {("FanControl", "Step")}

    underclaimed, skipped = [], []
    for sim_state, sosa_property in sorted(_STATE_TO_PROPERTY.items()):
        for entry in sim.control_rules(sim_state):
            device_type = str(entry["device_type"])
            participants = set()
            for action in (entry["actions"]["required"]
                           + entry["actions"]["optional"]):
                cluster = str(action["cluster_id"])
                target = str(action.get("command_id") or action.get("attribute_id"))
                participants.add((cluster, target))
            for device_id, g in graphs.items():
                if device_types[device_id] != device_type:
                    continue
                room_id = device_rooms[device_id]
                prop = URIRef(f"{base}/workspaces/{home}/{room_id}#{sosa_property}")
                claimed_names = set()
                for act in g.objects(art(device_id), TD.hasActionAffordance):
                    for actuation in g.objects(act, TDSOSA.hasEffectActuation):
                        if (actuation, SOSA.actsOnProperty, prop) in g:
                            claimed_names.add(str(next(g.objects(act, TD.name), "")))
                for cluster, target in sorted(participants):
                    if (cluster, target) in unmodelled:
                        skipped.append(f"{device_type}/{cluster}.{target}")
                        continue
                    # A command participant is covered by the affordance for the
                    # attribute that command drives; an attribute participant by
                    # the affordance of that attribute.
                    name = _m.affordance_name(cluster, target)
                    driven = _m.command_targets_for(cluster)
                    if target not in driven:
                        candidates = {_m.affordance_name(cluster, d) for d in driven}
                    else:
                        candidates = {name}
                    candidates.add(name)
                    if not (candidates & claimed_names):
                        underclaimed.append(
                            f"{device_type}/{cluster}.{target} -> {sosa_property}")
                break
    check("every action the simulator names carries the effect",
          not underclaimed,
          f"{len(underclaimed)}: {sorted(set(underclaimed))[:4]}"
          if underclaimed
          else ("all participants claimed"
                + (f"; {len(set(skipped))} unmodelled by design: "
                   f"{sorted(set(skipped))[:3]}" if skipped else "")))

    failures = [r for r in _results if r[0] == FAIL]
    print(f"\n{'=' * 60}")
    print(f"{len(_results) - len(failures)}/{len(_results)} checks passed")
    for _, name, detail in failures:
        print(f"  FAIL {name}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
