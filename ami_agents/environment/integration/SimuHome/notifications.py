#!/usr/bin/env python3
"""
WebSub subscriptions and state-change notification for SHTD.

**Why this is required, not a nicety.** An agent that actuates a device and then
plans against the state it remembers is planning against a world that no longer
exists. The simulator ticks continuously -- room temperature, humidity and
illuminance drift on their own between any two agent turns -- so a long-running
plan must be told when what it believes stops being true.

**SimuHome cannot push.** Verified: no websocket, SSE, webhook or event bus in
`src/simulator/`. The callbacks that exist (`set_dead_front_callback`,
`set_mode_change_callback`) are internal to the device model. So SHTD polls
`GET /api/home/state` -- one call returns the whole world -- diffs it against the
previous snapshot, and fans out one notification per changed attribute.

Everything downstream of the diff is HASP's contract, unchanged, so an agent
already speaking to `hasp.py` needs no modification: the same `POST /hub/`
mechanics, the same intent verification, the same subscriber payload shape. The
only difference is the key: a SimuHome `device_id` where HA had an `entity_id`.
"""

from __future__ import annotations

import asyncio
import time
import urllib.parse
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from fastapi import HTTPException

# Room states Matter carries in centi-units, which the TD reports divided by
# 100 and rounded. Must match td_builder.CENTI_TOKENS.
ROUNDED_TOKENS = {"temperature", "humidity"}

WEBHOOK_VERIFY_TIMEOUT = 10.0
NOTIFY_TIMEOUT = 10.0

# topic + callback -> subscription record
subscriptions: Dict[str, Dict[str, Any]] = {}


# -- subscription registry -------------------------------------------------

def subscription_id(topic: str, callback_url: str) -> str:
    return f"{topic}-{callback_url}"


def _is_active(subscription: Dict[str, Any]) -> bool:
    lease_seconds = subscription.get("lease_seconds")
    timestamp = subscription.get("timestamp")
    if not lease_seconds:
        return True
    if not isinstance(timestamp, (int, float)):
        return False
    return (time.monotonic() - timestamp) < lease_seconds


def prune_expired() -> None:
    for sub_id in [s for s, sub in subscriptions.items() if not _is_active(sub)]:
        subscriptions.pop(sub_id, None)


async def verify_callback_intent(
    callback_url: str, *, mode: str, topic: str,
    lease_seconds: Optional[int] = None,
) -> None:
    """WebSub intent verification: the callback must echo our challenge.

    This is what stops a third party subscribing someone else's URL to a topic.
    """
    challenge = str(uuid.uuid4())
    params = {
        "hub.mode": mode,
        "hub.topic": topic,
        "hub.callback": callback_url,
        "hub.challenge": challenge,
    }
    if lease_seconds:
        params["hub.lease_seconds"] = str(lease_seconds)
    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_VERIFY_TIMEOUT,
                                     follow_redirects=True) as client:
            response = await client.get(callback_url, params=params)
            response.raise_for_status()
            if response.text != challenge:
                raise HTTPException(
                    status_code=409,
                    detail="Intent verification challenge response mismatch")
    except HTTPException:
        raise
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=412,
            detail=f"Intent verification failed: cannot reach callback: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=412,
            detail=f"Intent verification failed: {exc}") from exc


async def register_subscription(
    *, topic: str, callback_url: str, lease_seconds: Optional[int],
    subscription_type: str, secret: Optional[str] = None,
    verify: bool = True,
) -> str:
    if verify:
        await verify_callback_intent(
            callback_url, mode="subscribe", topic=topic,
            lease_seconds=lease_seconds)
    sub_id = subscription_id(topic, callback_url)
    subscriptions[sub_id] = {
        "topic": topic,
        "callback": callback_url,
        "lease_seconds": lease_seconds,
        "secret": secret,
        "timestamp": time.monotonic(),
        "type": subscription_type,
    }
    return sub_id


def extract_hub_request(
    payload: Dict[str, Any],
) -> Tuple[str, str, str, Optional[int], Optional[str]]:
    """Parse a `POST /hub/` body, accepting both hub.* and bare spellings."""
    hub_mode = payload.get("hub.mode") or payload.get("mode")
    hub_topic = payload.get("hub.topic") or payload.get("topic")
    hub_callback = (payload.get("hub.callback") or payload.get("callback")
                    or payload.get("callbackUrl"))
    raw_lease = payload.get("hub.lease_seconds") or payload.get("lease_seconds")
    hub_secret = payload.get("hub.secret") or payload.get("secret")
    lease_seconds = None
    if raw_lease not in (None, ""):
        try:
            lease_seconds = int(raw_lease)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="hub.lease_seconds must be an integer") from exc
    if not all([hub_mode, hub_topic, hub_callback]):
        raise HTTPException(
            status_code=400,
            detail="Missing required parameters: hub.mode, hub.topic, hub.callback")
    return str(hub_mode), str(hub_topic), str(hub_callback), lease_seconds, hub_secret


def parse_form_or_json(content_type: str, raw_body: bytes) -> Dict[str, Any]:
    if ("application/x-www-form-urlencoded" in content_type
            or "multipart/form-data" in content_type):
        return dict(urllib.parse.parse_qsl(raw_body.decode("utf-8")))
    import json
    return json.loads(raw_body or b"{}")


# -- fan-out ---------------------------------------------------------------

async def notify(targets: List[Dict[str, Any]], payload: Dict[str, Any]) -> int:
    """POST one payload to every matching subscriber. Returns the delivered count.

    A failing callback is logged, never raised: one dead subscriber must not
    stop the others being told, nor bring down the poll loop.
    """
    if not targets:
        return 0
    delivered = 0
    async with httpx.AsyncClient(timeout=NOTIFY_TIMEOUT,
                                 follow_redirects=True) as client:
        for target in targets:
            body = {**payload, **target["context"]}
            try:
                response = await client.post(
                    target["callback"], json=body,
                    headers={"Content-Type": "application/json"})
                response.raise_for_status()
                delivered += 1
            except Exception as exc:  # noqa: BLE001 - deliberately swallowed
                print(f"[shtd] notification failed -> {target['callback']}: {exc}",
                      flush=True)
    return delivered


def targets_for_topics(topics: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Active subscribers of any of `topics`, de-duplicated by callback+topic.

    A subscriber focused on the whole home and on one room would otherwise be
    told twice about the same change.
    """
    prune_expired()
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for topic in topics:
        for sub in subscriptions.values():
            if sub.get("topic") != topic or not _is_active(sub):
                continue
            key = (sub["callback"], topic)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "callback": sub["callback"],
                "context": {**context, "topic": topic},
            })
    return out


# -- the poller ------------------------------------------------------------

class StateWatcher:
    """Polls the simulator and reports what changed since the last look.

    Diffing is per ATTRIBUTE, not per device, so a subscriber focused on one
    artifact is not woken by an unrelated field of it, and the payload says
    exactly which value moved.

    Room environmental state is diffed too: it drifts every tick as the
    simulation runs, and it is what a verification node most often re-reads.
    """

    def __init__(self, read_state: Callable[[], Dict[str, Any]],
                 report_value: Optional[Callable[[Any, Any], Any]] = None,
                 attribute_type: Optional[Callable[[str, str], Any]] = None):
        self._read_state = read_state
        # How a value is reported, so the diff compares what a subscriber will
        # actually see rather than the raw wire value.
        self._report_value = report_value
        self._attribute_type = attribute_type
        self._devices: Dict[str, Dict[str, Any]] = {}
        self._rooms: Dict[str, Dict[str, Any]] = {}
        self._device_room: Dict[str, str] = {}
        self.primed = False

    def _device_change(self, path: str, previous: Any, current: Any) -> bool:
        """Did a device attribute's REPORTED value move?

        Same reasoning as room state: `Thermostat.LocalTemperature` is stored in
        centi-degrees and reported divided, so raw 2337 -> 2338 is invisible to
        a subscriber and must not wake one.
        """
        if previous == current:
            return False
        if self._report_value is None:
            return True
        parts = str(path).split(".", 2)
        if len(parts) != 3 or self._attribute_type is None:
            return True
        matter_type = self._attribute_type(parts[1], parts[2])
        return (self._report_value(matter_type, previous)
                != self._report_value(matter_type, current))

    @staticmethod
    def _reported_change(token: str, previous: Any, current: Any) -> bool:
        """Did the value a subscriber actually SEES move?

        Room temperature and humidity are stored in centi-units and reported
        rounded to two decimals, so the raw value drifts every tick (2330 ->
        2331) while the reported one does not. Diffing the raw value woke every
        subscriber about twice a second for a change they could not observe.
        """
        if previous == current:
            return False
        if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
            if token in ROUNDED_TOKENS:
                return round(previous / 100.0, 2) != round(current / 100.0, 2)
            if isinstance(previous, float) or isinstance(current, float):
                return round(previous, 2) != round(current, 2)
        return True

    def _snapshot(self, state: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
        devices: Dict[str, Dict[str, Any]] = {}
        rooms: Dict[str, Dict[str, Any]] = {}
        device_room: Dict[str, str] = {}
        for room_id, room in (state.get("rooms") or {}).items():
            rooms[room_id] = dict(room.get("state") or {})
            for device in room.get("devices") or []:
                device_id = str(device.get("device_id") or "")
                if not device_id:
                    continue
                devices[device_id] = dict(device.get("attributes") or {})
                device_room[device_id] = room_id
        return devices, rooms, device_room

    def poll(self) -> List[Dict[str, Any]]:
        """Return one change record per attribute that moved.

        The first poll primes the baseline and reports nothing -- otherwise
        every attribute in the home would look like a change on startup.
        """
        state = self._read_state()
        devices, rooms, device_room = self._snapshot(state)
        clock = state.get("current_time")

        if not self.primed:
            self._devices, self._rooms, self._device_room = devices, rooms, device_room
            self.primed = True
            return []

        changes: List[Dict[str, Any]] = []
        for device_id, attributes in devices.items():
            previous = self._devices.get(device_id, {})
            for path, value in attributes.items():
                if path in previous and not self._device_change(
                        path, previous[path], value):
                    continue
                changes.append({
                    "kind": "device",
                    "device_id": device_id,
                    "room_id": device_room.get(device_id),
                    "path": path,
                    "value": value,
                    "previous": previous.get(path),
                    "timestamp": clock,
                })
        for room_id, room_state in rooms.items():
            previous = self._rooms.get(room_id, {})
            for token, value in room_state.items():
                if token in previous and not self._reported_change(
                        token, previous[token], value):
                    continue
                changes.append({
                    "kind": "room",
                    "room_id": room_id,
                    "path": token,
                    "value": value,
                    "previous": previous.get(token),
                    "timestamp": clock,
                })

        self._devices, self._rooms, self._device_room = devices, rooms, device_room
        return changes


# How many consecutive poll failures make an outage worth naming as one, and
# how often to repeat the line thereafter (in polls, so ~1/min at a 1s interval).
_PERSISTENT_FAILURES = 5
_FAILURE_REPEAT_EVERY = 60


async def poll_loop(
    watcher: StateWatcher,
    dispatch: Callable[[List[Dict[str, Any]]], Any],
    interval: float,
    stop: asyncio.Event,
) -> None:
    """Poll until stopped. Never dies on a transient simulator error.

    Surviving every error is what the loop is for, but surviving *quietly* is
    how a dead simulator comes to look like a semantics bug: at a 1s interval an
    unreachable backend prints the same line 3600 times an hour and nothing
    tells you the failure is persistent rather than transient. So count
    consecutive failures, say so when the run becomes one, then throttle the
    repeats -- and announce the recovery, which is the line that says the
    outage is over.
    """
    consecutive = 0
    while not stop.is_set():
        try:
            changes = await asyncio.to_thread(watcher.poll)
            if consecutive >= _PERSISTENT_FAILURES:
                print(f"[shtd] polling recovered after {consecutive} "
                      f"consecutive failures", flush=True)
            consecutive = 0
            if changes:
                await dispatch(changes)
        except Exception as exc:  # noqa: BLE001 - the loop must survive
            consecutive += 1
            if consecutive == _PERSISTENT_FAILURES:
                print(f"[shtd] poll failing persistently ({consecutive} in a "
                      f"row); no notification can be sent while this lasts: "
                      f"{exc}", flush=True)
            elif consecutive < _PERSISTENT_FAILURES or \
                    consecutive % _FAILURE_REPEAT_EVERY == 0:
                print(f"[shtd] poll failed ({consecutive} in a row): {exc}",
                      flush=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
