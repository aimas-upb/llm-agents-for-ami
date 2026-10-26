#!/usr/bin/env python3
"""Verify the HomeBench environment (step P2) against a live engine + HBTD pair.

SimuHome's ``verify_phase2.py`` for HomeBench. It starts (or is pointed at) the two
processes of a worker, loads one home, and checks the gate of the plan's section 4:

1. **Crawl.** The EnvExplorer's own code path (``YggdrasilIntegration``:
   ``initialize`` -> ``explore_hmas_environment``) finds **every device** of the
   engine's state, one workspace per room (plus the home), and reads every initial
   property value the engine holds.
2. **Gold round trip.** Every action of every crawled device is invoked through its
   TD form with values drawn from its own input schema; the engine ends in the
   state the value says, and the gold token ``room.device.call(arg)`` the invocation
   stands for, parsed back and applied on a *separate* engine, gives the same
   device. The mapping tables also round-trip every gold call of the dataset
   (``gold_round_trip``), and the actions together offer exactly the commands the
   home offers (``exposed_commands``).
3. **Strictness.** A missing or wrongly typed value is HTTP 400 and changes nothing
   (SHTD would power the device off); a well-typed value the device refuses is 422.
4. **WebSub.** Through ``subscribe_to_artifact`` and the notification listener, a
   change arrives with ``propertyUri`` (the property's form URL) and ``value``.
5. **Shape.** Typed devices, no environment resource, no TD-SOSA effects, unique
   ``td:name``, every ``homeont`` term emitted is defined in the ontology, curtains
   and blinds are told apart, the vacuum sits in the ``household`` workspace,
   ``/_shtd/time`` returns ``base_time``.

    python ami_agents/environment/integration/HomeBench/verify_phase2.py \\
        --data tests/homebench/data            # spawns engine + HBTD on free ports
    python .../verify_phase2.py --engine http://127.0.0.1:8099/api \\
        --hbtd http://127.0.0.1:8097 --home-id 75   # an already running pair
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

if __package__ in {None, ""}:  # allow `python .../HomeBench/verify_phase2.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from rdflib import Graph, Namespace, RDF, URIRef

from ami_agents.environment.integration.HomeBench import engine as engine_module
from ami_agents.environment.integration.HomeBench.engine import HomeBenchEngine
from ami_agents.environment.integration.HomeBench.mappings import (
    MappingTableError, load_mappings,
)
from ami_agents.environment.integration.HomeBench.sim_client import HomeBenchClient

LOGGER = logging.getLogger("verify_phase2")

# -- constants ------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEFAULT_DATA = REPO / "tests" / "homebench" / "data"
ONTOLOGY = REPO / "ami_agents" / "shared" / "ontologies" / "homeont.ttl"

HMAS = Namespace("https://purl.org/hmas/")
TD = Namespace("https://www.w3.org/2019/wot/td#")
HOME = Namespace("http://example.org/homeont/")
TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
JS = "https://www.w3.org/2019/wot/json-schema#"

STARTUP_TIMEOUT = 60.0
EVENT_TIMEOUT = 15.0
POLL_INTERVAL = 0.2
PASS, FAIL = "PASS", "FAIL"
_results: List[Tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record and log one check; returns ``ok``."""
    _results.append((PASS if ok else FAIL, name, detail))
    LOGGER.info("  [%s] %s%s", PASS if ok else FAIL, name, f" -- {detail}" if detail else "")
    return ok


# -- processes ------------------------------------------------------------------------------


def free_port() -> int:
    """Ask the OS for an unused port."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(probe, what: str, timeout: float = STARTUP_TIMEOUT) -> None:
    """Poll ``probe()`` until it is true or raise ``TimeoutError`` naming ``what``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe():
            return
        time.sleep(0.2)
    raise TimeoutError(f"{what} did not come up within {timeout:.0f}s")


def _http_ok(url: str) -> bool:
    """True when a GET of ``url`` answers 200."""
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


@contextlib.contextmanager
def spawn_pair(home_id: int, home_config: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    """Start an engine and an HBTD on free ports, load the home, yield their URLs.

    The two processes are the ones a worker starts; both are reaped on exit.
    """
    sim_port, hbtd_port = free_port(), free_port()
    sim_url = f"http://127.0.0.1:{sim_port}/api"
    logs = [tempfile.TemporaryFile(), tempfile.TemporaryFile()]
    procs: List[subprocess.Popen] = []
    try:
        # 1. The engine, then the home.
        procs.append(subprocess.Popen(
            [sys.executable, str(HERE / "simulator.py"), "--port", str(sim_port)],
            stdout=logs[0], stderr=subprocess.STDOUT))
        client = HomeBenchClient(sim_url)
        _wait(client.health, "engine")
        client.reset(home_config)
        # 2. HBTD, bound to that engine and home.
        procs.append(subprocess.Popen(
            [sys.executable, str(HERE / "hbtd.py"), "--sim", sim_url,
             "--home", f"home{home_id}", "--port", str(hbtd_port),
             "--poll-interval", str(POLL_INTERVAL)],
            stdout=logs[1], stderr=subprocess.STDOUT))
        _wait(lambda: _http_ok(f"http://127.0.0.1:{hbtd_port}/_shtd/status"), "HBTD")
        yield sim_url, f"http://127.0.0.1:{hbtd_port}"
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
        for log in logs:
            log.close()


# -- the data -----------------------------------------------------------------------------------


def pick_home(data: Path) -> int:
    """The home with the most distinct device types (lowest id on a tie), so one
    home exercises as many device kinds as the data allows."""
    best, best_id = -1, 0
    for path in sorted((data / "homes").glob("home*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        kinds = {d["device_type"] for r in config["rooms"].values() for d in r["devices"]}
        if len(kinds) > best:
            best, best_id = len(kinds), int(path.stem[len("home"):])
    return best_id


def episode_gold_rows(data: Path) -> Iterator[Tuple[str, bool]]:
    """``(gold output, is_multi)`` of every converted episode under ``data``."""
    for path in sorted((data / "episodes").glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
        yield meta["gold"], meta["query_type"] != "hb1"


def gold_round_trip(rows: Iterable[Tuple[str, bool]]) -> Dict[str, Any]:
    """Round-trip every gold call of ``rows`` through the mapping tables.

    ``rows`` are ``(gold output, is_multi)`` pairs. A gold call
    ``room.device.command(arg)`` is mapped to ``(action, value)`` by
    ``action_for_command`` and back to a gold token by ``gold_string``; the token must
    equal the call as written. Calls the home does not offer (the vacuum's ``start``,
    ``stop``, ``pause``, ``charge``) have no action: they are counted as unreachable,
    not as failures. Returns ``{calls, unreachable, failures, ops}``; ``ops`` lists the
    distinct ``device.command`` pairs seen.
    """
    from tests.homebench.convert import parse_gold

    mappings = load_mappings()
    calls = unreachable = 0
    failures: List[str] = []
    ops: set = set()
    for output, multi in rows:
        for call in parse_gold(output, multi):
            if call is None:
                continue
            calls += 1
            ops.add(f"{call.device}.{call.command}")
            written = ((call.room + ".") if call.room else "") + call.device + "."                 + call.command + "(" + ("" if call.arg is None else str(call.arg)) + ")"
            try:
                action, value = mappings.action_for_command(
                    call.device, call.command, call.arg)
            except MappingTableError:
                if call.command in engine_module.UNREACHABLE_COMMANDS.get(call.device, ()):
                    unreachable += 1
                else:
                    failures.append(f"{written}: no action")
                continue
            got = mappings.gold_string(call.device_id, call.device, action, value)
            if got != written:
                failures.append(f"{written} -> {got}")
    return {"calls": calls, "unreachable": unreachable, "failures": failures,
            "ops": sorted(ops)}


# -- HTTP helpers ---------------------------------------------------------------------------------


def post(url: str, payload: Any) -> Tuple[int, str]:
    """POST JSON; return ``(status, body)`` without raising on 4xx."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def get_json(url: str) -> Any:
    """GET a JSON document."""
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read() or b"null")


def device_map(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{device_id: device}`` over a home state."""
    return {d["device_id"]: d for r in state["rooms"].values() for d in r["devices"]}


# -- 1. crawl ---------------------------------------------------------------------------------------


async def crawl(base: str):
    """Crawl HBTD with the EnvExplorer's own integration engine."""
    from ami_agents.environment.integration.integration_engine import YggdrasilIntegration

    integration = YggdrasilIntegration(base + "/")
    ok = await integration.initialize({})
    if not ok:
        raise RuntimeError("YggdrasilIntegration.initialize failed on HBTD")
    await integration.explore_hmas_environment()
    return integration


def device_id_of(iri: str) -> str:
    """The device id: the last path segment of an artifact IRI."""
    return iri.split("#")[0].rsplit("/", 1)[-1]


def check_crawl(integration, state: Dict[str, Any], home_id: int) -> None:
    """Check 1: every device found, one workspace per room, initial values read."""
    mappings = load_mappings()
    devices = device_map(state)
    found = {device_id_of(iri): art for iri, art in integration.artifact_map.items()}
    LOGGER.info("1. Crawl (YggdrasilIntegration) -- every device is found")
    check("artifact count == device count", len(found) == len(devices),
          f"{len(found)} artifacts vs {len(devices)} devices")
    missing = sorted(set(devices) - set(found))
    check("no device missing", not missing, f"missing: {missing[:5]}" if missing else "")
    rooms = state["rooms"]
    check("workspaces == 1 home + rooms", len(integration.workspace_map) == 1 + len(rooms),
          f"{len(integration.workspace_map)} workspaces, {len(rooms)} rooms")
    # Initial values, read through each property form by the crawler.
    wrong: List[str] = []
    for device_id, device in devices.items():
        art = found.get(device_id)
        if art is None:
            continue
        base_path = art.artifact_id.split("#")[0]
        for row in mappings.affordances(device["device_type"], device["attributes"]):
            key = f"{base_path}/properties/{row['affordance_name']}"
            expected = mappings.read_value(row, device["attributes"][row["attribute"]])
            if art.current_state.get(key, "<absent>") != expected:
                wrong.append(f"{device_id}.{row['affordance_name']}: "
                             f"{art.current_state.get(key, '<absent>')!r} != {expected!r}")
    check("initial state read equals the engine's", not wrong,
          f"{len(wrong)} differ: {wrong[:3]}" if wrong else "")
    # The vacuum robot lives in `household`, not in a HomeBench room.
    vacuum = [d for d in devices if d.endswith("_vacuum_robot")]
    if vacuum:
        iri = next(i for i in integration.artifact_map if device_id_of(i) == vacuum[0])
        ws = integration.artifact_map[iri].workspace_id
        check("vacuum robot is in the household workspace",
              ws.split("#")[0].endswith("/household"), ws)


# -- 2. gold round trip -------------------------------------------------------------------------------


def sample_values(aff) -> List[Any]:
    """Values to try on an action, drawn from the input schema the crawler parsed."""
    schema = aff.input_schema
    if schema is None:
        return [None]
    kind = str(schema.get("type", "")).rsplit("#", 1)[-1]
    if kind == "BooleanSchema":
        return [True, False]
    if schema.get("enum"):
        return list(schema["enum"])
    if kind == "IntegerSchema":
        low, high = int(schema.get("minimum", 0)), int(schema.get("maximum", 100))
        return sorted({low, (low + high) // 2, high})
    if kind == "ArraySchema":
        return [[10, 20, 30]]
    return []


def expected_attribute(row: Dict[str, Any], value: Any) -> Any:
    """The home attribute value an invocation with ``value`` leaves behind
    (``pack`` leaves ``empty``)."""
    if row["value_kind"] == "none":
        return "empty"
    return load_mappings().write_attribute(row, value)


async def check_round_trip(integration, client: HomeBenchClient, home_config) -> None:
    """Check 2: every action, through its form, round-trips to the gold string."""
    from tests.homebench.convert import ConversionError, _parse_token

    mappings = load_mappings()
    LOGGER.info("\n2. Gold round trip -- every action, through its TD form")
    twin = HomeBenchEngine(home_config)
    invoked, bad_state, bad_gold, bad_twin = 0, [], [], []
    seen_ops: set = set()
    off_grammar: set = set()
    for iri, art in integration.artifact_map.items():
        device_id = device_id_of(iri)
        dtype = twin.device(device_id)["device_type"]
        for aff_id in art.thing_description.actions:
            aff = integration.affordance_map[aff_id]
            row = mappings.affordance(dtype, aff.name)
            for value in sample_values(aff):
                invoked += 1
                payload = {} if value is None else {"value": value}
                if await integration.execute_affordance(aff_id, payload) is None:
                    bad_state.append(f"{device_id}.{aff.name}={value!r}: refused")
                    continue
                # (a) the engine holds what the value says.
                live = client.device_attributes(device_id)
                want = expected_attribute(row, value)
                if live.get(row["attribute"]) != want:
                    bad_state.append(f"{device_id}.{aff.name}={value!r}: "
                                     f"{live.get(row['attribute'])!r} != {want!r}")
                # (b) the gold token it stands for, parsed and applied on a twin.
                token = mappings.gold_string(device_id, dtype, aff.name, value)
                try:
                    call = _parse_token(token, token)
                    command, arg, target = call.command, call.arg, call.device_id
                except ConversionError:
                    # A call the gold grammar has no form for (set_color takes a
                    # triple; no gold row uses it): applied from the table instead.
                    off_grammar.add(f"{dtype}.{aff.name}")
                    command, arg = mappings.command_for(dtype, aff.name, value)
                    target = device_id
                if target != device_id:
                    bad_gold.append(token)
                    continue
                seen_ops.add(f"{dtype}.{command}")
                twin.apply(target, command, arg)
                if twin.device(device_id)["attributes"] != live:
                    bad_twin.append(f"{token}: {twin.device(device_id)['attributes']} "
                                    f"!= {live}")
    check("every sampled invocation is accepted and moves the engine",
          not bad_state, f"{invoked} invoked; {len(bad_state)} bad: {bad_state[:3]}")
    check("every invocation's gold token parses back to its device", not bad_gold,
          f"{len(bad_gold)}: {bad_gold[:3]}")
    check("gold token applied on a twin engine == the live device", not bad_twin,
          f"{len(bad_twin)}: {bad_twin[:3]}")
    check("ops exercised live", bool(seen_ops),
          f"{len(seen_ops)} distinct device.command; outside the gold grammar: "
          f"{sorted(off_grammar) or 'none'}")
    # The actions together offer exactly the commands the home offers.
    mismatched = []
    for iri, art in integration.artifact_map.items():
        device_id = device_id_of(iri)
        dtype = twin.device(device_id)["device_type"]
        offered: set = set()
        for aff_id in art.thing_description.actions:
            aff = integration.affordance_map[aff_id]
            for value in sample_values(aff):
                offered.add(mappings.command_for(dtype, aff.name, value)[0])
        if offered != set(twin.exposed_commands(device_id)):
            mismatched.append(f"{device_id}: {sorted(offered ^ set(twin.exposed_commands(device_id)))}")
    check("actions reach exactly the commands the home offers", not mismatched,
          f"{len(mismatched)}: {mismatched[:3]}")
    client.reset(home_config)


def check_gold_dataset(data: Path) -> None:
    """Check 2b: every gold call of the dataset round-trips through the tables."""
    LOGGER.info("\n2b. Gold round trip over the whole dataset")
    report = gold_round_trip(episode_gold_rows(data))
    check("every gold call round-trips (device, action, value) <-> gold string",
          not report["failures"],
          f"{report['calls']} calls, {report['unreachable']} unreachable by "
          f"construction, {len(report['failures'])} failures {report['failures'][:3]}")
    check("the ops seen are covered", len(report["ops"]) > 0,
          f"{len(report['ops'])} distinct device.command pairs")


# -- 3. strictness -----------------------------------------------------------------------------------------


def action_url(base: str, home_id: int, state: Dict[str, Any], dtype: str,
               action: str) -> Tuple[str, str, str]:
    """``(url, room, device_id)`` of the first device of ``dtype`` that has ``action``."""
    mappings = load_mappings()
    for room, data in state["rooms"].items():
        for device in data["devices"]:
            if device["device_type"] == dtype and mappings.affordance(dtype, action) and \
                    mappings.affordance(dtype, action)["attribute"] in device["attributes"]:
                return (f"{base}/workspaces/home{home_id}/{room}/artifacts/"
                        f"{device['device_id']}/actions/{action}", room, device["device_id"])
    raise LookupError(f"no {dtype} with action {action} in this home")


def check_strictness(base: str, home_id: int, client: HomeBenchClient) -> None:
    """Check 3: 400 for a missing or wrongly typed value, 422 for a refused one, and
    nothing moves."""
    LOGGER.info("\n3. Strictness -- missing / wrongly typed value is 400, state untouched")
    state = client.home_state()
    cases = [
        # (device type, action, body, expected status)
        ("light", "onOff", {}, 400),
        ("light", "onOff", {"value": None}, 400),
        ("light", "onOff", {"value": "on"}, 400),
        ("light", "onOff", {"value": 1}, 400),
        ("light", "brightness", {}, 400),
        ("light", "brightness", {"value": "20"}, 400),
        ("light", "brightness", {"value": True}, 400),
        ("light", "brightness", {"value": 20.5}, 400),
        ("light", "brightness", {"value": 1000}, 422),
        ("curtain", "openClose", {"value": 5}, 400),
        ("curtain", "openClose", {"value": "ajar"}, 422),
        ("trash", "pack", {"value": True}, 400),
        ("air_conditioner", "mode", {}, 400),
        ("air_conditioner", "mode", {"value": "frost"}, 422),
        ("heating", "temperature", {"value": 80}, 422),
    ]
    failures: List[str] = []
    ran = 0
    for dtype, action, body, want in cases:
        try:
            url, _room, _device = action_url(base, home_id, state, dtype, action)
        except LookupError:
            continue  # this home has no such device; the others cover the rule
        ran += 1
        code, text = post(url, body)
        if code != want:
            failures.append(f"{dtype}.{action} {body}: {code} (want {want}): {text[:80]}")
    check("each bad invocation gets its status", not failures and ran >= len(cases) // 2,
          f"{ran}/{len(cases)} cases ran; {len(failures)} wrong: {failures[:3]}")
    check("no bad invocation changed the home", client.home_state() == state)
    # The SHTD regression: a missing value on onOff must not power the device off.
    url, _room, device_id = action_url(base, home_id, state, "light", "onOff")
    before = client.device_attributes(device_id)["state"]
    post(url, {})
    check("a missing onOff value does not switch the device off",
          client.device_attributes(device_id)["state"] == before)
    code, _ = post(f"{base}/workspaces/home{home_id}/nowhere/artifacts/x/actions/onOff",
                   {"value": True})
    check("an unknown device is 404", code == 404, str(code))
    code, _ = post(url.rsplit("/", 1)[0] + "/nothing", {"value": 1})
    check("an unknown action is 404", code == 404, str(code))


# -- 4. WebSub ---------------------------------------------------------------------------------------------


async def check_websub(integration, base: str, client: HomeBenchClient) -> None:
    """Check 4: a change reaches the listener with ``propertyUri`` and ``value``."""
    LOGGER.info("\n4. WebSub -- the payload carries propertyUri and value")
    mappings = load_mappings()
    port = free_port()
    await integration.start_notification_listener(port)
    try:
        # 1. Subscribe to a light the way the EnvExplorer does.
        iri = next(i for i, a in integration.artifact_map.items()
                   if "light" in device_id_of(i) and a.thing_description.actions)
        device_id = device_id_of(iri)
        subscribed = await integration.subscribe_to_artifact(iri)
        check("subscribe_to_artifact succeeds (hub verifies the callback)", subscribed)
        subs = get_json(f"{base}/_shtd/subscriptions")
        check("the hub holds the subscription", subs["count"] == 1, str(subs["count"]))
        # 2. Change it through the TD and wait for the push.
        art = integration.artifact_map[iri]
        on_off = next(integration.affordance_map[a] for a in art.thing_description.actions
                      if integration.affordance_map[a].name == "onOff")
        current = client.device_attributes(device_id)["state"]
        flipped = current != "on"
        await integration.execute_affordance(on_off.affordance_id, {"value": flipped})
        payload = await asyncio.wait_for(integration.event_queue.get(), EVENT_TIMEOUT)
        # 3. The fields the event loop reads.
        prop_href = next(integration.affordance_map[p].form.href
                         for p in art.thing_description.properties
                         if integration.affordance_map[p].name == "onOff")
        check("payload has artifactUri that the crawler knows",
              payload.get("artifactUri") in integration.artifact_map,
              str(payload.get("artifactUri")))
        check("payload has propertyUri == the property's form URL",
              payload.get("propertyUri") == prop_href, str(payload.get("propertyUri")))
        check("payload has value == the reported value", payload.get("value") is flipped,
              repr(payload.get("value")))
        check("payload keeps SHTD's fields", all(k in payload for k in (
            "deviceId", "property", "state", "previous", "attributes", "timestamp")))
        # 4. Cleanup is what a worker relies on between episodes.
        await integration.unsubscribe_all_artifacts()
        check("unsubscribe leaves no subscription",
              get_json(f"{base}/_shtd/subscriptions")["count"] == 0)
        code, _ = post(f"{base}/hub/", {
            "hub.mode": "subscribe", "hub.topic": f"{base}/workspaces/nothing",
            "hub.callback": f"http://localhost:{port}/webhook"})
        check("a topic that is not served is refused", code == 404, str(code))
        del mappings
    finally:
        await integration.stop_notification_listener()


# -- 5. shape -------------------------------------------------------------------------------------------------


def check_shape(integration, base: str, state: Dict[str, Any]) -> None:
    """Check 5: typing, no environment, no effects, unique names, defined vocabulary."""
    LOGGER.info("\n5. Shape")
    mappings = load_mappings()
    devices = device_map(state)
    graphs: Dict[str, Graph] = {}
    for iri, art in integration.artifact_map.items():
        graph = Graph()
        graph.parse(data=art.thing_description.rdf, format="turtle")
        graphs[iri] = graph
    # Typing: the class the table gives, a light by its attributes.
    untyped, dup_names = [], []
    for iri, graph in graphs.items():
        device = devices[device_id_of(iri)]
        family = mappings.device_family(device["device_type"], device["attributes"])
        subject = URIRef(iri)
        for curie in family["td_types"]:
            if (subject, RDF.type, URIRef(str(HOME) + curie.split(":", 1)[1])) not in graph \
                    and curie.startswith("homeont:"):
                untyped.append(f"{device_id_of(iri)} lacks {curie}")
        names = [str(n) for p in graph.objects(subject, TD.hasPropertyAffordance)
                 for n in graph.objects(p, TD.name)]
        if len(names) != len(set(names)):
            dup_names.append(device_id_of(iri))
    check("every device carries its table class", not untyped, f"{untyped[:3]}")
    check("td:name is unique among each Thing's properties", not dup_names, str(dup_names[:3]))
    # No physics claimed.
    effects = sum(len(list(g.subject_objects(TDSOSA.hasEffectActuation)))
                  for g in graphs.values())
    check("no TD-SOSA effect on any action", effects == 0, f"{effects} found")
    env = 0
    for ws_iri, ws in integration.workspace_map.items():
        graph = Graph()
        graph.parse(data=ws.rdf, format="turtle")
        env += len(list(graph.subjects(RDF.type, HOME.Environment)))
        env += len(list(graph.subjects(RDF.type, SOSA.FeatureOfInterest)))
    check("no environment resource in any workspace", env == 0, f"{env} found")
    # Every homeont term emitted is defined.
    onto = Graph()
    onto.parse(str(ONTOLOGY), format="turtle")
    defined = {str(s) for kind in (OWL.Class, OWL.DatatypeProperty, OWL.ObjectProperty)
               for s in onto.subjects(RDF.type, kind)}
    emitted = set()
    for graph in graphs.values():
        for _s, p, o in graph:
            if str(p).startswith(str(HOME)):
                emitted.add(str(p))
            if p == RDF.type and str(o).startswith(str(HOME)):
                emitted.add(str(o))
    undefined = sorted(t.rsplit("/", 1)[-1] for t in emitted if t not in defined)
    check("every homeont term emitted is defined in the ontology", not undefined,
          f"{len(emitted)} terms" if not undefined else str(undefined[:5]))
    # Curtains and blinds stay apart after the type closure.
    from ami_agents.shared.utils.vocabulary import close_types
    curtains = [i for i in graphs if device_id_of(i).endswith("_curtain")]
    blinds = [i for i in graphs if device_id_of(i).endswith("_blinds")]
    if curtains and blinds:
        for graph in graphs.values():
            close_types(graph)
        is_curtain = lambda i: (URIRef(i), RDF.type, HOME.Curtain) in graphs[i]
        check("a blinds device is not a Curtain, a curtain is not Blinds",
              not any(is_curtain(i) for i in blinds)
              and all(is_curtain(i) for i in curtains)
              and not any((URIRef(i), RDF.type, HOME.Blinds) in graphs[i] for i in curtains))
    # Tables fully reviewed, clock fixed.
    check("no mapping row awaits review", not mappings.unsettled(), str(mappings.unsettled()))
    now = get_json(f"{base}/_shtd/time").get("now")
    check("/_shtd/time returns base_time", now == state["current_time"], str(now))
    status = get_json(f"{base}/_shtd/status")
    check("/_shtd/status counts the devices", status["devices"] == len(devices),
          str(status["devices"]))


# -- main ---------------------------------------------------------------------------------------------------------


async def run(base: str, sim_url: str, home_id: int, home_config: Dict[str, Any],
              data: Optional[Path]) -> None:
    """Run every check against a live pair."""
    client = HomeBenchClient(sim_url)
    state = client.home_state()
    LOGGER.info("Crawling HBTD as the EnvExplorer would (home%d: %d rooms, %d devices)",
                home_id, len(state["rooms"]), len(device_map(state)))
    integration = await crawl(base)
    check_crawl(integration, state, home_id)
    await check_round_trip(integration, client, home_config)
    if data is not None and (data / "episodes").is_dir():
        check_gold_dataset(data)
    check_strictness(base, home_id, client)
    await check_websub(integration, base, client)
    check_shape(integration, base, client.home_state())


def main() -> int:
    """Parse arguments, bring up (or attach to) the pair, run the checks, report."""
    parser = argparse.ArgumentParser(description="Verify HomeBench phase 2.")
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="converted data (homes/, episodes/) from convert.py")
    parser.add_argument("--home-id", type=int, default=None,
                        help="home to load (default: the home with most device types)")
    parser.add_argument("--engine", default=None,
                        help="URL of a running engine API; with --hbtd, spawn nothing")
    parser.add_argument("--hbtd", default=None, help="URL of a running HBTD")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("ami_agents").setLevel(logging.WARNING)

    data = Path(args.data)
    home_id = args.home_id if args.home_id is not None else pick_home(data)
    home_config = json.loads((data / "homes" / f"home{home_id}.json").read_text(encoding="utf-8"))
    if args.engine and args.hbtd:
        asyncio.run(run(args.hbtd.rstrip("/"), args.engine, home_id, home_config, data))
    else:
        with spawn_pair(home_id, home_config) as (sim_url, base):
            asyncio.run(run(base, sim_url, home_id, home_config, data))

    failures = [r for r in _results if r[0] == FAIL]
    LOGGER.info("\n%s\n%d/%d checks passed", "=" * 60,
                len(_results) - len(failures), len(_results))
    for _, name, detail in failures:
        LOGGER.info("  FAIL %s: %s", name, detail)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
