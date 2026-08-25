#!/usr/bin/env python3
"""
SHTD -- serve a loaded SimuHome home as Thing Descriptions.

The HMAS projection of whichever home the paired simulator currently holds:

    GET  /                                              platform
    GET  /workspaces                                     hosted workspaces
    GET  /workspaces/{home}                              home workspace
    GET  /workspaces/{home}/{room}                       room workspace
    GET  /workspaces/{home}/{room}/artifacts             artifact directory
    GET  /workspaces/{home}/{room}/artifacts/{device}    the Thing Description
    GET  .../artifacts/{device}/properties/{name}        one property value
    GET  /workspaces/{home}/{room}/properties/{name}     one environmental value

Change notification is plain WebSub: a TD's `subscribeToWorkspace` /
`subscribeToArtifact` affordance POSTs to `/hub/` naming the resource in
`hub.topic`. Subscribing to a workspace yields notifications from every device
in it and from its environment; subscribing to an artifact yields only that
artifact's own state changes.

SHTD is a **stateless projection**: it holds no world of its own, and re-reads
`GET /api/home/state` per request. That is what makes the worker model work --
`POST /api/simulation/reset` on the simulator changes the episode, and SHTD
follows on the next request with no invalidation step.

    python ami_agents/environment/integration/SimuHome/shtd.py \\
        --sim http://127.0.0.1:8099/api --home qt2_feasible_seed_77 --port 8097
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

try:
    from .classify import classify_device
    from .mappings import load_mappings
    from .sim_client import SimuHomeClient, SimuHomeError
    from .td_builder import SimuHomeTD, report_value, serialize
    from . import notifications as notify
except ImportError:  # pragma: no cover - direct script use
    from classify import classify_device  # type: ignore
    from mappings import load_mappings  # type: ignore
    from sim_client import SimuHomeClient, SimuHomeError  # type: ignore
    from td_builder import SimuHomeTD, report_value, serialize  # type: ignore
    import notifications as notify  # type: ignore

TURTLE = "text/turtle"

app = FastAPI(title="SHTD - SimuHome Thing Descriptions")

_client: Optional[SimuHomeClient] = None
_home_id: str = os.getenv("SHTD_HOME", "home")
_base_url: str = os.getenv("SHTD_BASE_URL", "http://127.0.0.1:8097")


def get_client() -> SimuHomeClient:
    global _client
    if _client is None:
        _client = SimuHomeClient(os.getenv("SIMULATOR_API_BASE_URL"))
    return _client


def build(request: Request) -> SimuHomeTD:
    """A fresh projection of the simulator's current world.

    The base URI is taken from the request so the IRIs SHTD mints are the ones
    an agent can actually dereference, whatever host/port it reached us on.
    """
    try:
        state = get_client().home_state()
    except SimuHomeError as exc:
        raise HTTPException(status_code=502, detail=f"simulator unreachable: {exc}")
    base = str(request.base_url).rstrip("/")
    return SimuHomeTD(base, _home_id, state)


def ttl(graph) -> Response:
    return Response(serialize(graph), media_type=TURTLE)


def _check_home(td: SimuHomeTD, home: str) -> None:
    """One SHTD serves one home -- the one its simulator holds."""
    if home != td.home:
        raise HTTPException(
            status_code=404,
            detail=f"this SHTD serves home '{td.home}', not '{home}'",
        )


# -- discovery -------------------------------------------------------------

@app.get("/", responses={200: {"content": {TURTLE: {}}}})
def platform(request: Request) -> Response:
    return ttl(build(request).platform())


@app.get("/workspaces", responses={200: {"content": {TURTLE: {}}}})
def workspaces(request: Request) -> Response:
    return ttl(build(request).workspaces())


@app.get("/workspaces/{home}", responses={200: {"content": {TURTLE: {}}}})
def home_workspace(home: str, request: Request) -> Response:
    td = build(request)
    _check_home(td, home)
    return ttl(td.home_workspace())


@app.get("/workspaces/{home}/{room}", responses={200: {"content": {TURTLE: {}}}})
def room_workspace(home: str, room: str, request: Request) -> Response:
    td = build(request)
    _check_home(td, home)
    if room not in td.rooms:
        raise HTTPException(status_code=404, detail=f"no room '{room}' in {td.home}")
    return ttl(td.room_workspace(room))


@app.get("/workspaces/{home}/{room}/artifacts",
         responses={200: {"content": {TURTLE: {}}}})
def artifacts(home: str, room: str, request: Request) -> Response:
    td = build(request)
    _check_home(td, home)
    if room not in td.rooms:
        raise HTTPException(status_code=404, detail=f"no room '{room}' in {td.home}")
    return ttl(td.artifacts_directory(room))


@app.get("/workspaces/{home}/{room}/artifacts/{device}",
         responses={200: {"content": {TURTLE: {}}}})
def artifact(home: str, room: str, device: str, request: Request) -> Response:
    td = build(request)
    _check_home(td, home)
    graph = td.artifact(room, device)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"no device '{device}' in {room}")
    return ttl(graph)


# -- property reads --------------------------------------------------------

@app.get("/workspaces/{home}/{room}/artifacts/{device}/properties/{name}")
def read_property(home: str, room: str, device: str, name: str,
                  request: Request) -> JSONResponse:
    """Resolve a TD property name back to its Matter attribute and read it."""
    td = build(request)
    _check_home(td, home)
    dev = td.find_device(room, device)
    if dev is None:
        raise HTTPException(status_code=404, detail=f"no device '{device}' in {room}")

    mappings = load_mappings()
    for path, record in classify_device(dev.get("attributes") or {}).items():
        if record["role"] in ("drop", "thing_metadata"):
            continue
        cluster, attribute = record.get("cluster"), record.get("attribute")
        if mappings.affordance_name(str(cluster), str(attribute)) != name:
            continue
        # THE VALUE, and nothing else. `td:hasOutputSchema` describes the value,
        # so a `js:BooleanSchema` must dereference to `false`, not to an object
        # wrapping it -- otherwise nothing a consumer validates can match.
        # Everything an envelope used to carry (unit, cluster, quantity kind)
        # is already in the Thing Description, which is where it belongs.
        #
        # Reported the way the TD says it is: temperature in degrees, not the
        # centi-units Matter stores.
        return JSONResponse(report_value(record.get("type"), record.get("value")))
    raise HTTPException(status_code=404,
                        detail=f"no property '{name}' on {device}")


@app.get("/workspaces/{home}/{room}/properties/{name}")
def read_room_property(home: str, room: str, name: str,
                       request: Request) -> JSONResponse:
    """Read one of the room's environmental variables, in human units."""
    td = build(request)
    _check_home(td, home)
    if room not in td.rooms:
        raise HTTPException(status_code=404, detail=f"no room '{room}' in {td.home}")

    mappings = load_mappings()
    state = (td.rooms.get(room) or {}).get("state") or {}
    for token, value in state.items():
        row = mappings.room_state_property(token) or {}
        if str(row.get("sosa_property") or token) != name:
            continue
        # As above: the bare value, matching the declared js:NumberSchema.
        return JSONResponse(td.scale_room_state(token, value))
    raise HTTPException(status_code=404,
                        detail=f"no environmental property '{name}' in {room}")


# -- actuation -------------------------------------------------------------

def _resolve_action(td: SimuHomeTD, room: str, device: str,
                    name: str) -> Tuple[Dict[str, Any], str, str, int]:
    """TD action name -> the Matter coordinates SHTD needs to dispatch.

    This is the whole reason cluster ids stay out of the URL: the planner names
    the affordance, SHTD looks up which cluster and attribute that is, and
    whether to drive it by command or by attribute write.
    """
    dev = td.find_device(room, device)
    if dev is None:
        raise HTTPException(status_code=404, detail=f"no device '{device}' in {room}")
    mappings = load_mappings()
    for path, record in classify_device(dev.get("attributes") or {}).items():
        if record["role"] != "affordance" or record["affordance"] != "actuatable":
            continue
        cluster, attribute = str(record.get("cluster")), str(record.get("attribute"))
        if mappings.affordance_name(cluster, attribute) != name:
            continue
        return record, cluster, attribute, int(record.get("endpoint") or 1)
    raise HTTPException(status_code=404,
                        detail=f"no action '{name}' on {device}")


# Which command drives an attribute, for the `command` mechanism. Matter states
# no command->attribute relation, so this mirrors the approved command_targets
# table; the boolean cases pick their command from the requested value.
_COMMAND_FOR = {
    ("OnOff", "OnOff"): lambda v: "On" if _truthy(v) else "Off",
    ("LevelControl", "CurrentLevel"): lambda v: "MoveToLevel",
    ("WindowCovering", "CurrentPositionLiftPercent100ths"):
        lambda v: "GoToLiftPercentage",
    ("WindowCovering", "TargetPositionLiftPercent100ths"):
        lambda v: "GoToLiftPercentage",
    ("Thermostat", "SystemMode"): lambda v: None,
    ("TemperatureControl", "TemperatureSetpoint"): lambda v: "SetTemperature",
    ("DishwasherMode", "CurrentMode"): lambda v: "ChangeToMode",
    ("LaundryWasherMode", "CurrentMode"): lambda v: "ChangeToMode",
    ("RVCRunMode", "CurrentMode"): lambda v: "ChangeToMode",
    ("RVCCleanMode", "CurrentMode"): lambda v: "ChangeToMode",
    ("RTCCMode", "CurrentMode"): lambda v: "ChangeToMode",
}

# Command argument names, taken from SimuHome's own cluster handlers rather than
# from the Matter spec -- the two disagree, and the simulator is the authority
# here. Verified against src/simulator/domain/clusters/:
#
#   level_control._move_to_level(Level, ...)             PascalCase
#   window_covering._go_to_lift_percentage(lift_percent_100ths)
#   temperature_control._set_temperature(target_temperature)
#   {dishwasher,laundry_washer,rvc_run}_mode._change_to_mode(new_mode)
#   {rtcc,rvc_clean}_mode._change_to_mode(mode)          <- differ from their peers
#
# Spec-cased names (NewMode, LiftPercent100thsValue) are rejected with HTTP 400.
_COMMAND_ARGS = {
    "MoveToLevel": lambda v: {"Level": int(v)},
    # Centi-degrees, as the attribute is stored and as the TD reports its
    # min/max. A caller sending 21.5 means 21.5 C.
    "SetTemperature": lambda v: {"target_temperature": int(round(float(v) * 100))},
    "GoToLiftPercentage": lambda v: {"lift_percent_100ths": int(v)},
    "ChangeToMode": lambda v: {"new_mode": int(v)},
}

# Two of the five Mode Base derivatives name the argument `mode` where the other
# three name it `new_mode` -- checked one by one, since the naming is not
# predictable from the cluster.
_COMMAND_ARGS_BY_CLUSTER = {
    ("RTCCMode", "ChangeToMode"): lambda v: {"mode": int(v)},
    ("RVCCleanMode", "ChangeToMode"): lambda v: {"mode": int(v)},
}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "on", "1", "yes"}


@app.post("/workspaces/{home}/{room}/artifacts/{device}/actions/{name}")
async def invoke_action(home: str, room: str, device: str, name: str,
                        request: Request) -> JSONResponse:
    """Invoke an action. Body: {"value": ...} or {"input": ...}.

    Returns an ACK plus the value that was accepted -- never a claim about the
    resulting state. SimuHome models devices responding over ticks, so the
    observed value may not have moved yet; a plan confirms with a verification
    node reading the property back.
    """
    td = build(request)
    _check_home(td, home)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    value = body.get("value", body.get("input", body.get("payload")))

    record, cluster, attribute, endpoint = _resolve_action(td, room, device, name)
    mechanism = str(record.get("mechanism"))
    client = get_client()

    try:
        if mechanism == "attribute_write":
            if value is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"action '{name}' requires a value")
            result = client.write_attribute(device, cluster, attribute, value,
                                            endpoint_id=endpoint)
            dispatched = {"call": "write_attribute", "cluster": cluster,
                          "attribute": attribute, "value": value}
        else:
            chooser = _COMMAND_FOR.get((cluster, attribute))
            command = chooser(value) if chooser else None
            if command is None:
                raise HTTPException(
                    status_code=400,
                    detail=(f"'{name}' on {device} is command-driven but no "
                            f"command is mapped for {cluster}.{attribute}"))
            builder = (_COMMAND_ARGS_BY_CLUSTER.get((cluster, command))
                       or _COMMAND_ARGS.get(command))
            args = builder(value) if builder else {}
            result = client.execute_command(device, cluster, command, args,
                                            endpoint_id=endpoint)
            dispatched = {"call": "execute_command", "cluster": cluster,
                          "command": command, "args": args}
    except SimuHomeError as exc:
        # The simulator enforces device rules (e.g. writing while powered off
        # returns 400). Surface that: a verification-node planner needs it.
        raise HTTPException(status_code=422, detail=str(exc))

    return JSONResponse({
        "ok": True,
        "action": name,
        "device": device,
        "accepted": value,
        "dispatched": dispatched,
        "result": result,
        "note": "acknowledgement only; read the property back to confirm",
    })


# -- subscriptions ---------------------------------------------------------

def _workspace_topic(td: SimuHomeTD, room: Optional[str] = None) -> str:
    return td.room_path(room) if room else td.home_path()


def _artifact_topic(td: SimuHomeTD, room: str, device: str) -> str:
    return f"{td.artifact_path(room, device)}#artifact"


def _check_topic(td: SimuHomeTD, topic: str) -> None:
    """Reject a subscription to something we do not serve.

    Valid topics are exactly the three levels a TD advertises: the home
    workspace, a room workspace, and an artifact.
    """
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
    """WebSub hub, verbatim in behaviour from hasp.py."""
    body = notify.parse_form_or_json(
        request.headers.get("content-type", ""), await request.body())
    mode, topic, callback, lease_seconds, secret = notify.extract_hub_request(body)
    notify.prune_expired()
    # Only topics this server actually serves may be subscribed, so a caller
    # cannot register for a resource that will never produce a notification.
    if mode == "subscribe":
        _check_topic(build(request), topic)
    if mode == "subscribe":
        await notify.register_subscription(
            topic=topic, callback_url=callback, lease_seconds=lease_seconds,
            subscription_type="websub", secret=secret)
        return Response(status_code=202, content="Subscribed")
    if mode == "unsubscribe":
        await notify.verify_callback_intent(
            callback, mode="unsubscribe", topic=topic,
            lease_seconds=lease_seconds)
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
    notify.prune_expired()
    return JSONResponse({
        "count": len(notify.subscriptions),
        "subscriptions": [
            {"topic": s["topic"], "callback": s["callback"], "type": s["type"]}
            for s in notify.subscriptions.values()
        ],
    })


# -- the state poller ------------------------------------------------------

_watcher: Optional["notify.StateWatcher"] = None
_stop = asyncio.Event()
_poll_interval = 1.0


async def _dispatch_changes(changes: List[Dict[str, Any]]) -> None:
    """Turn diff records into subscriber notifications.

    Keyed on `device_id` where HASP keys on `entity_id`; the payload shape is
    otherwise identical, so an agent already speaking to hasp.py needs no
    change.
    """
    base = _base_url.rstrip("/")
    home_topic = f"{base}/workspaces/{_home_id}"
    mappings = load_mappings()

    for change in changes:
        room_id = change.get("room_id")
        room_topic = f"{home_topic}/{room_id}" if room_id else None

        if change["kind"] == "device":
            device_id = change["device_id"]
            artifact_uri = (f"{home_topic}/{room_id}/artifacts/{device_id}#artifact")
            topics = [t for t in (home_topic, room_topic, artifact_uri) if t]
            parsed = str(change["path"]).split(".", 2)
            name = change["path"]
            matter_type = None
            if len(parsed) == 3:
                name = mappings.affordance_name(parsed[1], parsed[2])
                matter_type = (mappings.attribute(parsed[1], parsed[2]) or {}) \
                    .get("matter", {}).get("type")
            # Report the value the way the read route does, or a subscriber
            # would see 2340 where a read of the same property returns 23.4.
            state = report_value(matter_type, change["value"])
            previous = report_value(matter_type, change["previous"])
            payload = {
                "deviceId": device_id,
                "artifactUri": artifact_uri,
                "artifactTitle": device_id,
                "property": name,
                "state": state,
                "previous": previous,
                "attributes": {name: state},
                "matterPath": change["path"],
                "timestamp": change.get("timestamp") or "",
            }
            context = {"workspaceId": room_topic or home_topic}
        else:
            foi = f"{home_topic}/{room_id}#environment"
            topics = [t for t in (home_topic, room_topic) if t]
            row = mappings.room_state_property(str(change["path"])) or {}
            name = str(row.get("sosa_property") or change["path"])
            value = SimuHomeTD.scale_room_state(str(change["path"]), change["value"])
            payload = {
                "featureOfInterest": foi,
                "artifactUri": foi,
                "artifactTitle": f"{room_id} environment",
                "property": name,
                "state": value,
                "previous": SimuHomeTD.scale_room_state(
                    str(change["path"]), change["previous"]),
                "attributes": {name: value},
                "timestamp": change.get("timestamp") or "",
            }
            context = {"workspaceId": room_topic or home_topic}

        targets = notify.targets_for_topics(topics, context)
        if targets:
            await notify.notify(targets, payload)


@app.on_event("startup")
async def _start_poller() -> None:
    global _watcher
    _mappings = load_mappings()

    def _attribute_type(cluster: str, attribute: str) -> Any:
        return (_mappings.attribute(cluster, attribute) or {}) \
            .get("matter", {}).get("type")

    _watcher = notify.StateWatcher(
        lambda: get_client().home_state(),
        report_value=report_value,
        attribute_type=_attribute_type)
    _stop.clear()
    asyncio.create_task(
        notify.poll_loop(_watcher, _dispatch_changes, _poll_interval, _stop))
    print(f"[shtd] state poller started (every {_poll_interval}s)", flush=True)


@app.on_event("shutdown")
async def _stop_poller() -> None:
    _stop.set()


# -- operations ------------------------------------------------------------

@app.get("/_shtd/status")
def status(request: Request) -> JSONResponse:
    """What this SHTD is projecting right now, and whether the mapping tables
    it relies on are fully reviewed."""
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
        "phase": 3,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a SimuHome home as Thing Descriptions.")
    parser.add_argument("--sim", default=os.getenv("SIMULATOR_API_BASE_URL",
                                                   "http://127.0.0.1:8000/api"))
    parser.add_argument("--home", default=os.getenv("SHTD_HOME", "home"),
                        help="episode id; becomes the home workspace segment")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--poll-interval", type=float, default=1.0,
                        help="seconds between state polls; match the simulator tick")
    args = parser.parse_args()

    global _client, _home_id, _base_url, _poll_interval
    os.environ["SIMULATOR_API_BASE_URL"] = args.sim
    _client = SimuHomeClient(args.sim)
    _home_id = args.home
    _base_url = f"http://{args.host}:{args.port}"
    _poll_interval = args.poll_interval

    import uvicorn
    print(f"SHTD on http://{args.host}:{args.port}/  "
          f"home={args.home}  simulator={args.sim}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
