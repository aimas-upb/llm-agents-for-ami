#!/usr/bin/env python3
"""
Verify SHTD phase 2 against a live SHTD + simulator pair.

Crawls the way the agent's `integration_engine` crawls -- platform -> hosts ->
contains -> contains -- using the same two type guards it applies
(`_process_workspace_recursive` line 1075, `_map_artifacts` line 1138), then
checks the result against the simulator's own `/api/home/state`.

The headline assertion is the plan's **recovery check**: every device the
simulator reports must appear as an artifact. The Home Assistant path silently
dropped 3,480 devices corpus-wide (all climate and cover); this is what proves
SHTD does not.

    python ami_agents/environment/integration/SimuHome/verify_phase2.py \\
        --shtd http://127.0.0.1:8097 --sim http://127.0.0.1:44275/api
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Set

from rdflib import Graph, Namespace, RDF, URIRef

try:
    from .classify import classify_device
    from .sim_client import SimuHomeClient
except ImportError:  # pragma: no cover
    from classify import classify_device  # type: ignore
    from sim_client import SimuHomeClient  # type: ignore

HMAS = Namespace("https://purl.org/hmas/")
TD = Namespace("https://www.w3.org/2019/wot/td#")
HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
# All SHTD vocabulary lives in one ontology (homeont.ttl).
HOME = Namespace("http://example.org/homeont/")
JS = Namespace("https://www.w3.org/2019/wot/json-schema#")

PASS, FAIL = "PASS", "FAIL"
_results: List[tuple] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def fetch(uri: str) -> Graph:
    g = Graph()
    g.parse(uri, format="turtle")
    return g


def crawl(base: str) -> Dict[str, object]:
    """Walk the platform exactly as the agent crawler does."""
    platform_graph = fetch(base + "/")
    platform = URIRef(f"{base}/#platform")

    workspaces: Dict[str, Graph] = {}
    artifacts: Dict[str, Graph] = {}

    def walk(ws_uri: URIRef, parent_graph: Graph) -> None:
        # Guard 1: the crawler only recurses when the PARENT graph types it.
        if (ws_uri, RDF.type, HMAS.Workspace) not in parent_graph:
            return
        key = str(ws_uri)
        if key in workspaces:
            return
        g = fetch(str(ws_uri).split("#")[0])
        workspaces[key] = g
        for sub in g.objects(ws_uri, HMAS.contains):
            if (sub, RDF.type, HMAS.Workspace) in g:
                walk(sub, g)
            # Guard 2: artifacts must be typed in the WORKSPACE's graph.
            elif (sub, RDF.type, HMAS.Artifact) in g:
                if str(sub) not in artifacts:
                    artifacts[str(sub)] = fetch(str(sub).split("#")[0])

    for root in platform_graph.objects(platform, HMAS.hosts):
        walk(root, platform_graph)

    return {"platform": platform_graph, "workspaces": workspaces,
            "artifacts": artifacts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SHTD phase 2.")
    parser.add_argument("--shtd", default="http://127.0.0.1:8097")
    parser.add_argument("--sim", default="http://127.0.0.1:44275/api")
    args = parser.parse_args()
    base = args.shtd.rstrip("/")

    print("Crawling SHTD as the agent's integration engine would...")
    found = crawl(base)
    workspaces: Dict[str, Graph] = found["workspaces"]  # type: ignore
    artifacts: Dict[str, Graph] = found["artifacts"]  # type: ignore
    print(f"  reached {len(workspaces)} workspaces, {len(artifacts)} artifacts\n")

    state = SimuHomeClient(args.sim).home_state()
    rooms = state.get("rooms") or {}
    sim_devices: Set[str] = {
        str(d.get("device_id"))
        for r in rooms.values() for d in (r.get("devices") or [])
    }

    by_device = {
        str(d.get("device_id")): (d.get("attributes") or {})
        for r in rooms.values() for d in (r.get("devices") or [])
    }

    print("1. Recovery -- every simulated device is an artifact")
    art_titles = {
        str(t) for g in artifacts.values()
        for s in g.subjects(RDF.type, HMAS.Artifact) for t in g.objects(s, TD.title)
    }
    # The TD title is overwritten by ProductName where present, so compare on
    # the artifact IRI's own last path segment instead -- that is the device id.
    art_ids = {str(u).split("#")[0].rsplit("/", 1)[-1] for u in artifacts}
    check("artifact count == device count",
          len(artifacts) == len(sim_devices),
          f"{len(artifacts)} artifacts vs {len(sim_devices)} devices")
    missing = sim_devices - art_ids
    check("no device missing", not missing, f"missing: {sorted(missing)}" if missing else "")

    print("\n2. Containment -- home + one workspace per room")
    check("workspace count == 1 home + rooms",
          len(workspaces) == 1 + len(rooms),
          f"{len(workspaces)} workspaces, {len(rooms)} rooms")

    print("\n3. Every artifact is a td:Thing, located in a place")
    # Location and feature-of-interest are different relations. EVERY device is
    # located in its room; only devices that perceive or affect one of the
    # room's environmental variables observe its environment. A freezer sits in
    # the kitchen but reads -15 C inside a 23.6 C room -- it is not an observer
    # of the kitchen's air.
    from mappings import load_mappings as _lm  # type: ignore
    _mp = _lm()
    device_types = {
        str(d.get("device_id")): str(d.get("device_type"))
        for r in rooms.values() for d in (r.get("devices") or [])
    }
    untyped, unplaced, wrong_foi = [], [], []
    for uri, g in artifacts.items():
        subj = URIRef(uri)
        device_id = uri.split("#")[0].rsplit("/", 1)[-1]
        if (subj, RDF.type, TD.Thing) not in g:
            untyped.append(uri)
        if not list(g.objects(subj, HOME.isLocatedIn)):
            unplaced.append(uri)
        family = (_mp.device_family(device_types.get(device_id, "")) or {})
        should = str(family.get("key") or "") in _mp.families_affecting_environment
        # Only the ROOM's environment counts. A compartment sensor legitimately
        # has a feature of interest of its own (a freezer's interior), which is
        # not an observation of the room.
        room_env = URIRef(uri.split("/artifacts/")[0] + "#environment")
        has = (subj, SOSA.hasFeatureOfInterest, room_env) in g
        if has != should:
            wrong_foi.append(f"{device_id}: foi={has}, expected={should}")
    check("all typed td:Thing", not untyped, f"{len(untyped)} untyped")
    check("all located in a place", not unplaced, f"{len(unplaced)} unplaced")
    check("environment observed only by devices that touch it",
          not wrong_foi, f"{len(wrong_foi)}: {wrong_foi[:3]}")

    print("\n4. Property coverage -- every non-plumbing attribute is an affordance")
    expected = 0
    for room in rooms.values():
        for device in room.get("devices") or []:
            for record in classify_device(device.get("attributes") or {}).values():
                if record["role"] == "affordance":
                    expected += 1
    emitted = sum(
        len(list(g.objects(URIRef(uri), TD.hasPropertyAffordance)))
        for uri, g in artifacts.items()
    )
    check("affordance count matches classifier", emitted == expected,
          f"{emitted} emitted vs {expected} expected")

    print("\n5. Every property affordance has a resolvable GET form")
    import urllib.request
    forms, bad = 0, []
    for uri, g in artifacts.items():
        for prop in g.objects(URIRef(uri), TD.hasPropertyAffordance):
            for form in g.objects(prop, TD.hasForm):
                for target in g.objects(form, HCTL.hasTarget):
                    forms += 1
                    if not str(target).startswith(base):
                        bad.append(str(target))
    check("all targets are on this server", not bad, f"{len(bad)} foreign")
    # Sample rather than hammer: one form per artifact is enough to prove the
    # naming round-trips, and a broken name would break them all identically.
    sampled, failed = 0, []
    for uri, g in list(artifacts.items()):
        for prop in g.objects(URIRef(uri), TD.hasPropertyAffordance):
            target = next(g.objects(next(g.objects(prop, TD.hasForm)), HCTL.hasTarget), None)
            if target is None:
                continue
            sampled += 1
            try:
                with urllib.request.urlopen(str(target), timeout=10) as r:
                    if r.status != 200:
                        failed.append(str(target))
            except Exception as exc:
                failed.append(f"{target}: {exc}")
            break
    check("sampled property reads return 200", not failed,
          f"{sampled} sampled, {len(failed)} failed")

    print("\n6. Containment is stated in both directions, in every document")
    # `hmas:contains` / `hmas:isContainedIn` are `owl:inverseOf` in the HMAS
    # ontology, so a reasoner would infer one from the other -- but a plain
    # crawler holding one document does not reason. Every document that states
    # an edge must therefore state both directions of it.
    asymmetric = []
    for uri, g in list(workspaces.items()) + list(artifacts.items()):
        label = uri.rsplit("/", 1)[-1]
        for parent, child in g.subject_objects(HMAS.contains):
            if (child, HMAS.isContainedIn, parent) not in g:
                asymmetric.append(f"{label}: contains {child} without inverse")
        for child, parent in g.subject_objects(HMAS.isContainedIn):
            if (parent, HMAS.contains, child) not in g:
                asymmetric.append(f"{label}: isContainedIn {parent} without inverse")
    check("every contains has its isContainedIn (and vice versa)",
          not asymmetric, "; ".join(asymmetric[:3]))

    hosts_bad = []
    for uri, g in [("platform", found["platform"])] + list(workspaces.items()):
        for p, w in g.subject_objects(HMAS.hosts):
            if (w, HMAS.isHostedOn, p) not in g:
                hosts_bad.append(f"{uri}: hosts without isHostedOn")
        for w, p in g.subject_objects(HMAS.isHostedOn):
            if (p, HMAS.hosts, w) not in g:
                hosts_bad.append(f"{uri}: isHostedOn without hosts")
    check("every hosts has its isHostedOn (and vice versa)",
          not hosts_bad, "; ".join(hosts_bad[:3]))

    print("\n7. Upward navigation works from any single document")
    # Pick one artifact and walk back up using only what its own TD states.
    sample_uri = sorted(artifacts)[0]
    sg = artifacts[sample_uri]
    room = next(sg.objects(URIRef(sample_uri), HMAS.isContainedIn), None)
    check("artifact names its room", room is not None, str(room or ""))
    home = next(sg.objects(room, HMAS.isContainedIn), None) if room else None
    check("that room names its home", home is not None, str(home or ""))
    if home is not None:
        hg = fetch(str(home).split("#")[0])
        plat = next(hg.objects(home, HMAS.isHostedOn), None)
        check("the home names its platform", plat is not None, str(plat or ""))

    print("\n8. Schemas describe shape, never a reading")
    # A js:*Schema is a static description: type, unit, scale. A value belongs
    # to a dereference of the property's form. Embedding one would freeze an
    # instant of a continuously ticking simulation into a cacheable document.
    leaked = []
    for uri, g in list(artifacts.items()) + list(workspaces.items()):
        for pred in (HOME.currentValue, HOME.matterRawValue):
            for s, _, o in g.triples((None, pred, None)):
                leaked.append(f"{uri.rsplit('/', 1)[-1]}: {pred.split('/')[-1]}={o}")
    check("no value embedded in any schema", not leaked,
          f"{len(leaked)} leaked: " + "; ".join(leaked[:3]) if leaked else "")

    # ...and the shape itself must still be there, or we have thrown out the
    # baby with the bathwater.
    typeless = []
    for uri, g in artifacts.items():
        for prop in g.objects(URIRef(uri), TD.hasPropertyAffordance):
            for schema in g.objects(prop, TD.hasOutputSchema):
                if not list(g.objects(schema, RDF.type)):
                    typeless.append(uri)
    check("every output schema still declares a type", not typeless,
          f"{len(typeless)} typeless")

    print("\n9. td:name is unique within each Thing, and resolves to one attribute")
    # This is what lets the Matter cluster/attribute triples stay out of the TD:
    # `td:name` alone must identify the attribute, so the read route can resolve
    # it. Three names would otherwise collide on a single device (RVC run/clean
    # modes, heat-pump power/energy accuracy) -- qualified in attribute_map.yaml.
    from mappings import load_mappings as _load_mappings  # type: ignore
    _m = _load_mappings()
    duplicate_names = []
    for uri, g in artifacts.items():
        device_id = uri.split("#")[0].rsplit("/", 1)[-1]
        names = [str(n) for prop in g.objects(URIRef(uri), TD.hasPropertyAffordance)
                 for n in g.objects(prop, TD.name)]
        for name, count in __import__("collections").Counter(names).items():
            if count > 1:
                duplicate_names.append(f"{device_id}: {name} x{count}")
    check("no duplicate td:name on any Thing", not duplicate_names,
          f"{len(duplicate_names)}: {duplicate_names[:3]}")

    print("\n10. Enum-typed values are explained, not left as bare integers")
    import sys as _sys
    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from matter_model.registry import load_registry  # type: ignore
    reg = load_registry()

    unexplained, mistyped = [], []
    for uri, g in artifacts.items():
        device_id = uri.split("#")[0].rsplit("/", 1)[-1]
        by_name = {
            _m.affordance_name(str(r["cluster"]), str(r["attribute"])):
                (str(r["cluster"]), str(r["attribute"]))
            for r in classify_device(by_device.get(device_id, {})).values()
            if r["role"] == "affordance"
        }
        for prop in g.objects(URIRef(uri), TD.hasPropertyAffordance):
            name = next(g.objects(prop, TD.name), None)
            resolved = by_name.get(str(name))
            if resolved is None:
                continue
            cluster, attribute = resolved
            if not reg.enum_items(cluster, attribute):
                continue
            for schema in g.objects(prop, TD.hasOutputSchema):
                if not list(g.objects(schema, JS.description)):
                    unexplained.append(f"{cluster}.{attribute}")
                # The wire value is the integer, whatever the enum is named.
                if (schema, RDF.type, JS.IntegerSchema) not in g:
                    mistyped.append(f"{cluster}.{attribute}")
    check("every enum-typed property has a js:description",
          not unexplained, f"{len(unexplained)} bare: {sorted(set(unexplained))[:3]}")
    check("every enum-typed property is js:IntegerSchema",
          not mistyped, f"{len(mistyped)} mistyped: {sorted(set(mistyped))[:3]}")

    print("\n11. Declared schema type matches the actual value's type")
    # Matter names its composite types ("ChannelInfoStruct", "AlarmBitmap")
    # instead of declaring a primitive, so anything not resolved against the
    # registry silently degrades to StringSchema -- telling a planner "string"
    # about a value that is really an object or an integer.
    _EXPECTED = {
        dict: "ObjectSchema", list: "ArraySchema", bool: "BooleanSchema",
        str: "StringSchema",
    }
    mismatched = []
    for uri, g in artifacts.items():
        device_id = uri.split("#")[0].rsplit("/", 1)[-1]
        values = by_device.get(device_id, {})
        by_name = {
            _m.affordance_name(str(r["cluster"]), str(r["attribute"])):
                (str(r["cluster"]), str(r["attribute"]))
            for r in classify_device(values).values()
            if r["role"] == "affordance"
        }
        for prop in g.objects(URIRef(uri), TD.hasPropertyAffordance):
            name = next(g.objects(prop, TD.name), None)
            resolved = by_name.get(str(name))
            schema = next(g.objects(prop, TD.hasOutputSchema), None)
            if schema is None or resolved is None:
                continue
            cluster, attribute = resolved
            declared = str(next(g.objects(schema, RDF.type), "")).split("#")[-1]
            value = next((v for p, v in values.items()
                          if p.endswith(f"{cluster}.{attribute}")), None)
            if value is None:
                continue
            if isinstance(value, bool):
                want = "BooleanSchema"
            elif isinstance(value, (int, float)):
                want = ("IntegerSchema", "NumberSchema")
            else:
                want = _EXPECTED.get(type(value), declared)
            if declared not in (want if isinstance(want, tuple) else (want,)):
                mismatched.append(f"{cluster}.{attribute}: "
                                  f"{type(value).__name__} typed {declared}")
    check("no schema misdeclares its value's type", not mismatched,
          f"{len(mismatched)}: {mismatched[:3]}")

    print("\n15. Every homeont term emitted is defined in the ontology")
    # Minting vocabulary without defining it leaves a consumer with no way to
    # know what a predicate means. ami_agents/shared/ontologies/homeont.ttl is the
    # definition; this asserts the served graphs never outrun it.
    from pathlib import Path as _Path
    onto_path = (_Path(__file__).resolve().parents[3]
                 / "shared" / "ontologies" / "homeont.ttl")
    if not onto_path.is_file():
        check("homeont.ttl exists", False, str(onto_path))
    else:
        onto = Graph()
        onto.parse(str(onto_path), format="turtle")
        OWL = Namespace("http://www.w3.org/2002/07/owl#")
        defined = {
            str(s) for kind in (OWL.Class, OWL.DatatypeProperty, OWL.ObjectProperty)
            for s in onto.subjects(RDF.type, kind)
        }
        emitted = set()
        for _uri, g in list(artifacts.items()) + list(workspaces.items()) \
                + [("platform", found["platform"])]:
            for _s, p, o in g:
                if str(p).startswith(str(HOME)):
                    emitted.add(str(p))
                if p == RDF.type and str(o).startswith(str(HOME)):
                    emitted.add(str(o))
        undefined = sorted(t.rsplit("/", 1)[-1] for t in emitted if t not in defined)
        check("no undefined homeont vocabulary", not undefined,
              f"{len(undefined)}: {undefined[:5]}" if undefined
              else f"{len(emitted)} terms, all defined")

    print("\n13. Dereferencing a form yields a value matching its schema")
    # `td:hasOutputSchema` describes the VALUE, so a js:BooleanSchema must
    # dereference to `false`, not to an object wrapping it. An envelope makes
    # the declared schema unvalidatable and duplicates TD metadata into every
    # read.
    _SCHEMA_PY = {
        "BooleanSchema": bool, "IntegerSchema": (int, float),
        "NumberSchema": (int, float), "StringSchema": str,
        "ArraySchema": list, "ObjectSchema": dict,
    }
    mismatched_reads = []
    for uri, g in list(artifacts.items())[:6]:
        for prop in g.objects(URIRef(uri), TD.hasPropertyAffordance):
            schema = next(g.objects(prop, TD.hasOutputSchema), None)
            form = next(g.objects(prop, TD.hasForm), None)
            if schema is None or form is None:
                continue
            declared = str(next(g.objects(schema, RDF.type), "")).split("#")[-1]
            target = next(g.objects(form, HCTL.hasTarget), None)
            expected = _SCHEMA_PY.get(declared)
            if target is None or expected is None:
                continue
            try:
                with urllib.request.urlopen(str(target), timeout=10) as response:
                    value = __import__("json").loads(response.read() or b"null")
            except Exception as exc:
                mismatched_reads.append(f"{target}: {exc}")
                continue
            if value is None:
                continue
            # bool is an int subclass; keep them distinct.
            if declared != "BooleanSchema" and isinstance(value, bool):
                mismatched_reads.append(f"{declared} got bool")
            elif not isinstance(value, expected):
                mismatched_reads.append(
                    f"{declared} got {type(value).__name__}")
    check("read values match their declared schema", not mismatched_reads,
          f"{len(mismatched_reads)}: {mismatched_reads[:3]}")

    print("\n14. Bounds are schema restrictions, not properties")
    # A Min/Max attribute constrains a sibling's value, so it belongs in that
    # sibling's schema as js:minimum/js:maximum -- machine-checkable standard
    # vocabulary -- never as a readable property of its own, which would model
    # a constraint as state and state the same fact twice.
    from classify import BOUND_ATTRIBUTES  # type: ignore
    leaked_bounds, bounded = [], 0
    for uri, g in artifacts.items():
        device_id = uri.split("#")[0].rsplit("/", 1)[-1]
        values = by_device.get(device_id, {})
        expect = {
            (str(r["cluster"]), str(r["attribute"]))
            for r in classify_device(values).values()
            if (str(r["cluster"]), str(r["attribute"])) in BOUND_ATTRIBUTES
        }
        names = {str(n) for prop in g.objects(URIRef(uri), TD.hasPropertyAffordance)
                 for n in g.objects(prop, TD.name)}
        for _cluster, attribute in expect:
            lower = attribute[:1].lower() + attribute[1:]
            if lower in names:
                leaked_bounds.append(f"{device_id}: {attribute}")
        if expect:
            bounded += len(list(g.subjects(JS.minimum, None)))
    check("no bound attribute is exposed as a property", not leaked_bounds,
          f"{len(leaked_bounds)}: {leaked_bounds[:3]}")
    check("devices declaring bounds carry js:minimum", bounded > 0,
          f"{bounded} schemas bounded")

    print("\n14. Rooms carry the four environmental properties")
    env_bad = []
    for uri, g in workspaces.items():
        foi = next(g.subjects(RDF.type, SOSA.FeatureOfInterest), None)
        if foi is None:
            continue  # the home workspace has none, by design
        names = {str(n) for p in g.objects(foi, SSN.hasProperty)
                 for n in g.objects(p, TD.name)}
        if names != {"air_temperature", "relative_humidity", "illuminance",
                     "pm10_mass_concentration"}:
            env_bad.append(f"{uri}: {sorted(names)}")
    check("every room has all four", not env_bad, "; ".join(env_bad))

    failures = [r for r in _results if r[0] == FAIL]
    print(f"\n{'=' * 60}")
    print(f"{len(_results) - len(failures)}/{len(_results)} checks passed")
    if failures:
        for _, name, detail in failures:
            print(f"  FAIL {name}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
