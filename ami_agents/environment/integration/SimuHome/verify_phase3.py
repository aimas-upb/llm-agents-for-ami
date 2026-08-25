#!/usr/bin/env python3
"""
Verify SHTD phase 3 -- actions and state-change notification -- against a live
SHTD + simulator pair.

Covers the plan's verification points 3, 6 and 7:

  3. Invocation      -- a BT action POSTs a form href and the SimuHome attribute
                        changes; read back through the TD.
  6. Verification    -- a plan actuates, then READS BACK the affected property
     round trip         through its own affordance. This replaces settling-time
                        modelling entirely.
  7. Liveness        -- subscribe through `focus`, actuate, and assert the
                        callback fires. Then the negative: an environmental
                        property moving on its own also produces a push, so a
                        long-running plan is never reasoning about stale state.

Run with a subscriber port that is free; the script starts its own subscriber.

    python ami_agents/environment/integration/SimuHome/verify_phase3.py \\
        --shtd http://127.0.0.1:8097 --sim http://127.0.0.1:44275/api
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from rdflib import Graph, Namespace, RDF, URIRef

try:
    from .sim_client import SimuHomeClient
except ImportError:  # pragma: no cover
    from sim_client import SimuHomeClient  # type: ignore

TD = Namespace("https://www.w3.org/2019/wot/td#")
HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
HTV = Namespace("http://www.w3.org/2011/http#")
JS = Namespace("https://www.w3.org/2019/wot/json-schema#")
HOME = Namespace("http://example.org/homeont/")

PASS, FAIL = "PASS", "FAIL"
_results: List[tuple] = []
_pushes: List[Dict[str, Any]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


# -- a subscriber that answers the WebSub challenge -------------------------

class _Subscriber(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        challenge = (parse_qs(urlparse(self.path).query)
                     .get("hub.challenge") or [""])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(challenge.encode())

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            _pushes.append(json.loads(self.rfile.read(length) or b"{}"))
        except Exception:
            pass
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_args):
        pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def post(url: str, payload: Dict[str, Any]) -> Tuple[int, Any]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    def _decode(raw: bytes) -> Any:
        # `focus` and `/hub/` answer with plain text, not JSON.
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return raw.decode("utf-8", errors="replace")

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, _decode(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read())


def get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read() or b"{}")


def wait_for_push(predicate, timeout: float = 12.0) -> Optional[Dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for push in list(_pushes):
            if predicate(push):
                return push
        time.sleep(0.3)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SHTD phase 3.")
    parser.add_argument("--shtd", default="http://127.0.0.1:8097")
    parser.add_argument("--sim", default="http://127.0.0.1:44275/api")
    args = parser.parse_args()
    base = args.shtd.rstrip("/")

    status = get(f"{base}/_shtd/status")
    home = status["home"]
    sim = SimuHomeClient(args.sim)
    state = sim.home_state()

    # Pick a light: it is the one device whose actuation has a visible
    # environmental consequence within a tick or two.
    target = None
    for room_id, room in (state.get("rooms") or {}).items():
        for device in room.get("devices") or []:
            if str(device.get("device_type")) == "on_off_light":
                target = (room_id, str(device.get("device_id")))
                break
        if target:
            break
    if target is None:
        print("no on_off_light in the loaded episode", file=sys.stderr)
        return 2
    room_id, device_id = target
    artifact = f"{base}/workspaces/{home}/{room_id}/artifacts/{device_id}"
    print(f"Target: {device_id} in {room_id}\n")

    print("1. Actuatable properties carry an action affordance")
    g = Graph()
    g.parse(artifact, format="turtle")
    art = URIRef(f"{artifact}#artifact")
    # An artifact also carries subscribeToArtifact / unsubscribeFromArtifact,
    # which target /hub/ rather than an /actions/ route. Those are subscription
    # affordances, not actuation; count only the latter here.
    WEBSUB = Namespace("https://purl.org/hmas/websub/")
    _SUBSCRIPTION_TYPES = {WEBSUB.subscribeToArtifact,
                           WEBSUB.unsubscribeFromArtifact,
                           WEBSUB.subscribeToWorkspace,
                           WEBSUB.unsubscribeFromWorkspace}
    actions = [
        a for a in g.objects(art, TD.hasActionAffordance)
        if not (_SUBSCRIPTION_TYPES & set(g.objects(a, RDF.type)))
    ]
    actuatable = [
        p for p in g.objects(art, TD.hasPropertyAffordance)
        if (p, RDF.type,
            URIRef("https://example.org/hmas/td-sosa-ext#ActuatablePropertyAffordance")) in g
    ]
    check("one action per actuatable property",
          len(actions) == len(actuatable),
          f"{len(actions)} actions vs {len(actuatable)} actuatable properties")

    bad_forms = []
    for action in actions:
        form = next(g.objects(action, TD.hasForm), None)
        if form is None:
            bad_forms.append("no form")
            continue
        if (form, HTV.methodName, None) not in g:
            bad_forms.append("no method")
        method = str(next(g.objects(form, HTV.methodName), ""))
        op = next(g.objects(form, HCTL.hasOperationType), None)
        target_uri = str(next(g.objects(form, HCTL.hasTarget), ""))
        if method != "POST" or op != TD.invokeAction:
            bad_forms.append(f"{method}/{op}")
        if "/actions/" not in target_uri:
            bad_forms.append(target_uri)
    check("every action form is a POST invokeAction on /actions/",
          not bad_forms, f"{len(bad_forms)}: {bad_forms[:3]}")

    missing_input = [a for a in actions if not list(g.objects(a, TD.hasInputSchema))]
    check("every action declares an input schema",
          not missing_input, f"{len(missing_input)} without")

    print("\n2. Invocation changes the world (plan check 3)")
    onoff_action = next(
        (a for a in actions if str(next(g.objects(a, TD.name), "")) == "onOff"), None)
    if onoff_action is None:
        check("light exposes an onOff action", False)
        return 1
    form = next(g.objects(onoff_action, TD.hasForm))
    href = str(next(g.objects(form, HCTL.hasTarget)))

    before = get(f"{artifact}/properties/onOff")
    code, ack = post(href, {"value": not before})
    check("POST to the form href succeeds", code == 200, f"HTTP {code}")
    check("response is an ack, not a state claim",
          isinstance(ack, dict) and ack.get("ok") is True and "note" in ack,
          str(ack.get("note", ""))[:60] if isinstance(ack, dict) else "")

    print("\n3. Verification round trip (plan check 6)")
    # Read the property back through its own affordance rather than trusting
    # the response -- this is what replaces settling-time modelling.
    settled = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        settled = get(f"{artifact}/properties/onOff")
        if settled == (not before):
            break
        time.sleep(0.4)
    check("reading the property back confirms the change",
          settled == (not before), f"{before} -> {settled}")

    print("\n4. Notification liveness (plan check 7)")
    port = free_port()
    server = HTTPServer(("127.0.0.1", port), _Subscriber)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    callback = f"http://127.0.0.1:{port}/cb"

    # Subscribe the way a TD says to: POST /hub/ naming the resource in
    # hub.topic. The room workspace topic yields the room's devices AND its
    # environment.
    room_topic = f"{base}/workspaces/{home}/{room_id}"
    code, _ = post(f"{base}/hub/", {
        "hub.mode": "subscribe", "hub.topic": room_topic,
        "hub.callback": callback})
    check("subscribeToWorkspace accepted (intent verified)",
          code == 202, f"HTTP {code}")

    # Flip from whatever the value is NOW. Section 2 already changed it, and the
    # poller has since re-based, so flipping to a remembered older value can be
    # a no-op that produces nothing to notify about.
    _pushes.clear()
    current = get(f"{artifact}/properties/onOff")
    time.sleep(1.5)          # let the poller take `current` as its baseline
    _pushes.clear()
    post(href, {"value": not current})

    push = wait_for_push(lambda p: p.get("property") == "onOff")
    check("actuation produces a push", push is not None,
          f"{push.get('previous')} -> {push.get('state')}" if push else "none arrived")
    if push:
        check("push identifies the artifact and carries both values",
              push.get("artifactUri", "").endswith("#artifact")
              and "previous" in push and "state" in push,
              push.get("artifactUri", "")[-52:])

    # The negative: the room's own environmental state moves as a consequence,
    # and that must be pushed too -- otherwise a long-running plan reasons
    # about a stale room.
    env = wait_for_push(lambda p: p.get("property") == "illuminance")
    check("environmental consequence is pushed against the feature-of-interest",
          env is not None and str(env.get("featureOfInterest", "")).endswith("#environment"),
          f"{env.get('previous')} -> {env.get('state')} lx" if env else "none arrived")

    print("\n5. Artifact subscription is scoped to that artifact alone")
    art_port = free_port()
    art_server = HTTPServer(("127.0.0.1", art_port), _Subscriber)
    threading.Thread(target=art_server.serve_forever, daemon=True).start()
    art_callback = f"http://127.0.0.1:{art_port}/cb"
    art_topic = f"{artifact}#artifact"

    code, _ = post(f"{base}/hub/", {
        "hub.mode": "subscribe", "hub.topic": art_topic,
        "hub.callback": art_callback})
    check("subscribeToArtifact accepted", code == 202, f"HTTP {code}")

    code, _ = post(f"{base}/hub/", {
        "hub.mode": "subscribe", "hub.topic": "http://elsewhere/nope",
        "hub.callback": art_callback})
    check("a topic this server does not serve is rejected",
          code == 404, f"HTTP {code}")

    _pushes.clear()
    now = get(f"{artifact}/properties/onOff")
    time.sleep(1.5)
    _pushes.clear()
    post(href, {"value": not now})

    scoped = wait_for_push(lambda p: p.get("topic") == art_topic)
    check("artifact subscriber is told about its own device",
          scoped is not None and scoped.get("property") == "onOff",
          f"{scoped.get('previous')} -> {scoped.get('state')}" if scoped else "none")
    # ...and NOT about the room's environment, which is not its topic.
    leaked = [p for p in _pushes
              if p.get("topic") == art_topic and p.get("featureOfInterest")]
    check("artifact subscriber is NOT told about room environment",
          not leaked, f"{len(leaked)} leaked")

    code, _ = post(f"{base}/hub/", {
        "hub.mode": "unsubscribe", "hub.topic": art_topic,
        "hub.callback": art_callback})
    check("unsubscribeFromArtifact accepted", code == 202, f"HTTP {code}")

    art_server.shutdown()
    server.shutdown()

    failures = [r for r in _results if r[0] == FAIL]
    print(f"\n{'=' * 60}")
    print(f"{len(_results) - len(failures)}/{len(_results)} checks passed")
    for _, name, detail in failures:
        print(f"  FAIL {name}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
