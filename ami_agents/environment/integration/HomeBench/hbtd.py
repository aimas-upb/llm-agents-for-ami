#!/usr/bin/env python3
"""HBTD: serve a loaded HomeBench home as Thing Descriptions.

The HomeBench twin of SHTD (``SimuHome/shtd.py``): the same CLI, the same routes, the
same WebSub hub, projecting whichever home its paired engine (``simulator.py``)
currently holds. It is a **stateless projection** like SHTD: it keeps no world of its
own and re-reads ``GET /api/home/state`` per request, so ``POST /api/simulation/reset``
on the engine changes the episode and HBTD follows on the next request.

    GET  /                                              platform
    GET  /workspaces                                     hosted workspaces
    GET  /workspaces/{home}                              home workspace
    GET  /workspaces/{home}/{room}                       room workspace
    GET  /workspaces/{home}/{room}/artifacts             artifact directory
    GET  /workspaces/{home}/{room}/artifacts/{device}    the Thing Description
    GET  .../artifacts/{device}/properties/{name}        one property value (bare JSON)
    POST .../artifacts/{device}/actions/{name}           invoke ({"value": v})
    POST .../artifacts/{device}/actions/{name}/dispatch  the engine call, not executed
    POST /hub/                                           WebSub hub
    GET  /_shtd/subscriptions   /_shtd/status   /_shtd/time

Where it differs from SHTD, on purpose (the plan's section 2.3 "gotchas"):

* **A missing or wrongly typed value is HTTP 400.** SHTD reads a missing value on an
  on/off action as ``false`` and silently powers the device off; here ``onOff`` needs a
  JSON boolean, a number needs an integer, ``pack`` takes no value. A value of the
  right type the device refuses (out of range, not an option) is 422, as in SHTD.
* **The WebSub payload carries ``propertyUri`` and ``value``**, which the EnvExplorer's
  event loop needs to update its state; SHTD's payload has neither and the update is
  skipped silently.
* ``/_shtd/time`` returns the home's fixed ``base_time``; ``/_shtd/status`` states no
  phase and reads no Matter table. Nothing here is Matter-specific.

Worked example (engine on 8099 holding ``home75``)::

    python HomeBench/hbtd.py --sim http://127.0.0.1:8099/api --home home75 --port 8097
    POST /workspaces/home75/bathroom/artifacts/bathroom_light/actions/brightness
         {"value": 20}      -> 200 {"ok": true, "dispatched": {"command": "set_brightness"...}}
    POST ... {}             -> 400 action 'brightness' requires a value
    POST ... {"value": 101} -> 422 bathroom_light.brightness: 101 outside [0, 100]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

if __package__ in {None, ""}:  # allow `python .../HomeBench/hbtd.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from ami_agents.environment.integration.HomeBench.mappings import (
    MappingTableError, load_mappings,
)
from ami_agents.environment.integration.HomeBench.sim_client import (
    HomeBenchClient, SimuHomeError,
)
from ami_agents.environment.integration.HomeBench.td_builder import HomeBenchTD
from ami_agents.environment.integration.SimuHome import notifications as notify
from ami_agents.environment.integration.SimuHome.td_builder import serialize

LOGGER = logging.getLogger(__name__)

# -- constants and process state ------------------------------------------------------------

TURTLE = "text/turtle"
SIM_ENV_VAR = "SIMULATOR_API_BASE_URL"  # the variable SHTD reads; one worker env serves both
HOME_ENV_VAR = "SHTD_HOME"
BASE_ENV_VAR = "SHTD_BASE_URL"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8097
DEFAULT_POLL_INTERVAL = 1.0

_client: Optional[HomeBenchClient] = None
_home_id: str = os.getenv(HOME_ENV_VAR, "home")
_base_url: str = os.getenv(BASE_ENV_VAR, f"http://{DEFAULT_HOST}:{DEFAULT_PORT}")
_poll_interval: float = DEFAULT_POLL_INTERVAL
_watcher: Optional["notify.StateWatcher"] = None
_stop = asyncio.Event()


def get_client() -> HomeBenchClient:
    """The engine client, built from ``SIMULATOR_API_BASE_URL`` on first use."""
    global _client
    if _client is None:
        _client = HomeBenchClient(os.getenv(SIM_ENV_VAR))
    return _client


def build(request: Request) -> HomeBenchTD:
    """A fresh projection of the engine's current home.

    The base URI is taken from the request so the IRIs minted are ones the caller
    can dereference, whatever host and port it reached us on.
    """
    try:
        state = get_client().home_state()
    except SimuHomeError as exc:
        raise HTTPException(status_code=502, detail=f"engine unreachable: {exc}")
    return HomeBenchTD(str(request.base_url).rstrip("/"), _home_id, state)


def ttl(graph) -> Response:
    """A graph as a Turtle response."""
    return Response(serialize(graph), media_type=TURTLE)


def _check_home(td: HomeBenchTD, home: str) -> None:
    """One HBTD serves one home: the one its engine holds."""
    if home != td.home:
        raise HTTPException(
            status_code=404, detail=f"this HBTD serves home '{td.home}', not '{home}'")


def _check_room(td: HomeBenchTD, room: str) -> None:
    """404 for a room the home does not have."""
    if room not in td.rooms:
        raise HTTPException(status_code=404, detail=f"no room '{room}' in {td.home}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run the state poller for the life of the server."""
    _stop.clear()
    global _watcher
    # 1. The watcher diffs the engine's flat attribute map (no value scaling here).
    _watcher = notify.StateWatcher(lambda: get_client().home_state())
    task = asyncio.create_task(
        notify.poll_loop(_watcher, _dispatch_changes, _poll_interval, _stop))
    LOGGER.info("state poller started (every %ss)", _poll_interval)
    yield
    # 2. Stop it on shutdown.
    _stop.set()
    await task


app = FastAPI(title="HBTD - HomeBench Thing Descriptions", lifespan=lifespan)


# -- discovery ---------------------------------------------------------------------------------


@app.get("/", responses={200: {"content": {TURTLE: {}}}})
def platform(request: Request) -> Response:
    """The platform document."""
    return ttl(build(request).platform())


@app.get("/workspaces", responses={200: {"content": {TURTLE: {}}}})
def workspaces(request: Request) -> Response:
    """The directory of hosted workspaces: just the home."""
    return ttl(build(request).workspaces())


@app.get("/workspaces/{home}", responses={200: {"content": {TURTLE: {}}}})
def home_workspace(home: str, request: Request) -> Response:
    """The home workspace: one contained workspace per room."""
    td = build(request)
    _check_home(td, home)
    return ttl(td.home_workspace())


@app.get("/workspaces/{home}/{room}", responses={200: {"content": {TURTLE: {}}}})
def room_workspace(home: str, room: str, request: Request) -> Response:
    """A room workspace: its devices as contained artifacts."""
    td = build(request)
    _check_home(td, home)
    _check_room(td, room)
    return ttl(td.room_workspace(room))


@app.get("/workspaces/{home}/{room}/artifacts",
         responses={200: {"content": {TURTLE: {}}}})
def artifacts(home: str, room: str, request: Request) -> Response:
    """The artifacts of a room, nothing else."""
    td = build(request)
    _check_home(td, home)
    _check_room(td, room)
    return ttl(td.artifacts_directory(room))


@app.get("/workspaces/{home}/{room}/artifacts/{device}",
         responses={200: {"content": {TURTLE: {}}}})
def artifact(home: str, room: str, device: str, request: Request) -> Response:
    """One device's Thing Description."""
    td = build(request)
    _check_home(td, home)
    graph = td.artifact(room, device)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"no device '{device}' in {room}")
    return ttl(graph)


# -- property reads -------------------------------------------------------------------------------


@app.get("/workspaces/{home}/{room}/artifacts/{device}/properties/{name}")
def read_property(home: str, room: str, device: str, name: str,
                  request: Request) -> JSONResponse:
    """One property's value: the bare JSON value, in the form the TD's schema says."""
    td = build(request)
    _check_home(td, home)
    dev, row = _resolve(td, room, device, name)
    return JSONResponse(td.m.read_value(row, dev["attributes"][row["attribute"]]))


def _resolve(td: HomeBenchTD, room: str, device: str, name: str
             ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(device, affordance row)`` for a TD name, or 404.

    The row must be one the device really has: a light without ``brightness`` has no
    ``brightness`` property or action, whatever the table lists for lights.
    """
    dev = td.find_device(room, device)
    if dev is None:
        raise HTTPException(status_code=404, detail=f"no device '{device}' in {room}")
    row = td.m.affordance(str(dev.get("device_type")), name)
    if row is None or row["attribute"] not in (dev.get("attributes") or {}):
        raise HTTPException(status_code=404, detail=f"no '{name}' on {device}")
    return dev, row


# -- actuation ----------------------------------------------------------------------------------------


async def _action_value(request: Request) -> Any:
    """The value of an invocation body: ``{"value": v}`` (``input`` and ``payload`` are
    accepted as SHTD does). ``None`` when the body carries none."""
    try:
        body = await request.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    return body.get("value", body.get("input", body.get("payload")))


def check_value(td: HomeBenchTD, dev: Dict[str, Any], row: Dict[str, Any],
                value: Any) -> Any:
    """The value to send, or HTTP 400 (missing or wrongly typed) / 422 (not allowed).

    The type is what the action's input schema declares: a JSON boolean for
    ``boolean``, an integer for ``integer`` (a whole float is taken as that integer),
    a string for ``enum``, three integers for ``rgb``, nothing for ``none``. Whether a
    well-typed value is permitted (range, options) is the device's rule: 422.
    """
    name, kind = row["affordance_name"], row["value_kind"]
    # 1. An action without input takes none.
    if kind == "none":
        if value is not None:
            raise HTTPException(status_code=400,
                                detail=f"action '{name}' takes no value")
        return None
    # 2. Every other kind needs one.
    if value is None:
        raise HTTPException(status_code=400, detail=f"action '{name}' requires a value")
    # 3. Type check against the declared schema.
    if kind == "boolean" and not isinstance(value, bool):
        raise HTTPException(status_code=400,
                            detail=f"action '{name}' needs a boolean, got {value!r}")
    if kind == "integer":
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool):
            raise HTTPException(status_code=400,
                                detail=f"action '{name}' needs an integer, got {value!r}")
    if kind == "enum" and not isinstance(value, str):
        raise HTTPException(status_code=400,
                            detail=f"action '{name}' needs a string, got {value!r}")
    if kind == "rgb" and not (isinstance(value, list) and len(value) == 3 and all(
            isinstance(c, int) and not isinstance(c, bool) for c in value)):
        raise HTTPException(status_code=400,
                            detail=f"action '{name}' needs [red, green, blue] integers")
    # 4. Enums of a state kind list their members in the table; settable enums are
    #    checked by the engine against the device's own options.
    if kind == "enum" and row.get("values") and value not in row["values"]:
        raise HTTPException(status_code=422,
                            detail=f"{name}: {value!r} not in {row['values']}")
    return value


def _call(td: HomeBenchTD, room: str, device: str, name: str, value: Any
          ) -> Dict[str, Any]:
    """The engine call an invocation becomes, without executing it.

    ``{"tool": "execute_command", "args": {"device_id", "command_id", "args"}}``.
    Raises HTTPException: 404 unknown action, 400 bad value, 422 value not allowed.
    """
    dev, row = _resolve(td, room, device, name)
    if not row["actuatable"]:
        raise HTTPException(status_code=404, detail=f"no action '{name}' on {device}")
    value = check_value(td, dev, row, value)
    try:
        command, arg = td.m.command_for(str(dev["device_type"]), name, value)
    except MappingTableError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"tool": "execute_command",
            "args": {"device_id": device, "command_id": command,
                     "args": {} if arg is None else {"value": arg}}}


@app.post("/workspaces/{home}/{room}/artifacts/{device}/actions/{name}")
async def invoke_action(home: str, room: str, device: str, name: str,
                        request: Request) -> JSONResponse:
    """Invoke an action. Body: ``{"value": ...}``.

    Returns an acknowledgement and the call that was dispatched, never a claim about
    the resulting state: a plan confirms by reading the property back.
    """
    # 1. The call, from the home as it is now (400 / 404 / 422 raised here).
    td = build(request)
    _check_home(td, home)
    value = await _action_value(request)
    call = _call(td, room, device, name, value)
    args = call["args"]
    # 2. Execute it; a refusal by the engine is the device's rule, so 422.
    try:
        result = get_client().execute_command(device, args["command_id"], args["args"])
    except SimuHomeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return JSONResponse({
        "ok": True, "action": name, "device": device, "accepted": value,
        "dispatched": {"call": "execute_command", "command": args["command_id"],
                       "args": args["args"]},
        "result": result,
        "note": "acknowledgement only; read the property back to confirm",
    })


@app.post("/workspaces/{home}/{room}/artifacts/{device}/actions/{name}/dispatch")
async def dispatch_action(home: str, room: str, device: str, name: str,
                          request: Request) -> JSONResponse:
    """The engine call invoking this action would make, not executed."""
    td = build(request)
    _check_home(td, home)
    return JSONResponse(_call(td, room, device, name, await _action_value(request)))


# -- subscriptions ----------------------------------------------------------------------------------


def _canonical_topic(topic: str) -> str:
    """Rewrite a topic's origin to this server's canonical base.

    IRIs are minted from the incoming request, but notifications are matched against
    topics minted from ``_base_url``; a client that subscribed via ``localhost`` would
    otherwise never match a push about ``127.0.0.1`` and every push would be dropped
    in silence. Only the origin changes; the path is left to ``_check_topic``.
    """
    parsed = urlparse(topic)
    if not parsed.scheme or not parsed.netloc:
        return topic
    base = urlparse(_base_url)
    if (parsed.scheme, parsed.netloc) == (base.scheme, base.netloc):
        return topic
    return urlunparse((base.scheme, base.netloc, parsed.path,
                       parsed.params, parsed.query, parsed.fragment))


def _check_topic(td: HomeBenchTD, topic: str) -> None:
    """Reject a subscription to something not served: valid topics are the home
    workspace, a room workspace, and an artifact (``...#artifact``)."""
    if topic == td.home_path():
        return
    for room_id in td.room_ids():
        if topic == td.room_path(room_id):
            return
        for device in (td.rooms.get(room_id) or {}).get("devices") or []:
            device_id = str(device.get("device_id") or "")
            if topic == f"{td.artifact_path(room_id, device_id)}#artifact":
                return
    raise HTTPException(
        status_code=404,
        detail=(f"unknown topic '{topic}'; subscribe to the home workspace, "
                f"a room workspace, or an artifact"))


@app.post("/hub/")
async def hub(request: Request) -> Response:
    """WebSub hub: subscribe or unsubscribe a callback to a topic."""
    body = notify.parse_form_or_json(
        request.headers.get("content-type", ""), await request.body())
    mode, topic, callback, lease_seconds, secret = notify.extract_hub_request(body)
    notify.prune_expired()
    # 1. Only topics this server serves may be subscribed, so a caller cannot
    #    register for a resource that will never produce a notification.
    if mode == "subscribe":
        _check_topic(build(request), topic)
    # 2. Store and look up under this server's own origin, never the caller's.
    topic = _canonical_topic(topic)
    if mode == "subscribe":
        await notify.register_subscription(
            topic=topic, callback_url=callback, lease_seconds=lease_seconds,
            subscription_type="websub", secret=secret)
        return Response(status_code=202, content="Subscribed")
    if mode == "unsubscribe":
        await notify.verify_callback_intent(
            callback, mode="unsubscribe", topic=topic, lease_seconds=lease_seconds)
        removed = notify.subscriptions.pop(
            notify.subscription_id(topic, callback), None)
        if removed is None:
            raise HTTPException(
                status_code=404,
                detail="No active subscription found for this topic and callback.")
        return Response(status_code=202, content="Unsubscribed")
    raise HTTPException(status_code=400,
                        detail="Invalid hub.mode. Must be 'subscribe' or 'unsubscribe'.")


@app.get("/_shtd/subscriptions")
def list_subscriptions() -> JSONResponse:
    """The active subscriptions (the worker checks this is empty between episodes)."""
    notify.prune_expired()
    return JSONResponse({
        "count": len(notify.subscriptions),
        "subscriptions": [
            {"topic": s["topic"], "callback": s["callback"], "type": s["type"]}
            for s in notify.subscriptions.values()
        ],
    })


# -- change notification --------------------------------------------------------------------------------


def change_payload(home_topic: str, room_id: str, device_id: str,
                   device_type: str, attribute: str, value: Any, previous: Any,
                   timestamp: Any) -> Optional[Dict[str, Any]]:
    """The WebSub payload of one attribute change, or ``None`` for an attribute the
    tables give no affordance (nothing a subscriber could read back).

    ``propertyUri`` is the property's form URL and ``value`` the reported value: the
    two fields the EnvExplorer's event loop reads to update its state. ``property``,
    ``state``, ``previous`` and ``attributes`` keep SHTD's payload shape.
    """
    mappings = load_mappings()
    row = next((r for r in mappings.affordances(device_type)
                if r["attribute"] == attribute), None)
    if row is None:
        return None
    name = str(row["affordance_name"])
    artifact_uri = f"{home_topic}/{room_id}/artifacts/{device_id}#artifact"
    shown = mappings.read_value(row, value)
    return {
        "deviceId": device_id,
        "artifactUri": artifact_uri,
        "artifactTitle": device_id,
        "property": name,
        "propertyUri": f"{home_topic}/{room_id}/artifacts/{device_id}/properties/{name}",
        "value": shown,
        "state": shown,
        "previous": mappings.read_value(row, previous) if previous is not None else None,
        "attributes": {name: shown},
        "timestamp": timestamp or "",
    }


async def _dispatch_changes(changes: List[Dict[str, Any]]) -> None:
    """Turn diff records into subscriber notifications.

    Each change goes to the subscribers of the home, of its room, and of its artifact.
    """
    # 1. Device types, from the state the change was seen in (the diff has none).
    state = await asyncio.to_thread(lambda: get_client().home_state())
    types = {str(d.get("device_id")): str(d.get("device_type"))
             for room in (state.get("rooms") or {}).values()
             for d in room.get("devices") or []}
    home_topic = f"{_base_url.rstrip('/')}/workspaces/{_home_id}"
    # 2. One payload per change, to every matching subscriber.
    for change in changes:
        if change["kind"] != "device":
            continue
        room_id, device_id = change["room_id"], change["device_id"]
        payload = change_payload(
            home_topic, room_id, device_id, types.get(device_id, ""),
            str(change["path"]), change["value"], change["previous"],
            change.get("timestamp"))
        if payload is None:
            continue
        room_topic = f"{home_topic}/{room_id}"
        topics = [home_topic, room_topic, payload["artifactUri"]]
        targets = notify.targets_for_topics(topics, {"workspaceId": room_topic})
        if targets:
            await notify.notify(targets, payload)


# -- operations -------------------------------------------------------------------------------------------


@app.get("/_shtd/status")
def status(request: Request) -> JSONResponse:
    """What this HBTD is projecting right now, and how many mapping rows still need
    review."""
    td = build(request)
    return JSONResponse({
        "home": td.home,
        "base": td.base,
        "simulator": get_client().base_url,
        "clock": td.state.get("current_time"),
        "tick": td.state.get("current_tick"),
        "rooms": td.room_ids(),
        "devices": td.device_count(),
        "unsettledMappingRows": load_mappings().unsettled(),
    })


@app.get("/_shtd/time")
def virtual_time() -> JSONResponse:
    """The home's fixed clock (``base_time``), so agents configured with
    ``environment.clock.source: http`` read the same endpoint as for SimuHome."""
    try:
        answer = get_client().current_time()
    except SimuHomeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    now = answer.get("now") if isinstance(answer, dict) else answer
    return JSONResponse({"now": now})


def main() -> None:
    """Serve a HomeBench home as Thing Descriptions (SHTD's command line)."""
    parser = argparse.ArgumentParser(description="Serve a HomeBench home as Thing Descriptions.")
    parser.add_argument("--sim", default=os.getenv(SIM_ENV_VAR, "http://127.0.0.1:8000/api"))
    parser.add_argument("--home", default=os.getenv(HOME_ENV_VAR, "home"),
                        help="home id; becomes the home workspace segment")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL,
                        help="seconds between state polls")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    global _client, _home_id, _base_url, _poll_interval
    os.environ[SIM_ENV_VAR] = args.sim
    _client = HomeBenchClient(args.sim)
    _home_id = args.home
    _base_url = f"http://{args.host}:{args.port}"
    _poll_interval = args.poll_interval

    import uvicorn
    LOGGER.info("HBTD on http://%s:%d/  home=%s  engine=%s",
                args.host, args.port, args.home, args.sim)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
