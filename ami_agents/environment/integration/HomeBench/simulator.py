#!/usr/bin/env python3
"""The HomeBench engine as a small HTTP app, shaped like SimuHome's simulator.

SimuHome runs two processes per worker: a simulator (``../SimuHome``, FastAPI, API
under ``/api``) and SHTD, which projects the simulator's home as Thing Descriptions.
HomeBench gets the same topology, so every harness, run command and debugging recipe
reads the same for both benchmarks. This module is HomeBench's simulator: a FastAPI
app around ``engine.HomeBenchEngine`` with the routes SHTD and the bench use, and the
same ``{status, data, error}`` envelope::

    GET  /api/__health__                       liveness
    POST /api/simulation/reset                 load a home (an initial_home_config)
    POST /api/simulation/fast_forward_to       no time here: answers the state
    GET  /api/home/state                       {current_tick, current_time, rooms}
    GET  /api/time                             {"now": "<base_time>"}
    GET  /api/devices/{id}/attributes          one device's attributes
    GET  /api/devices/{id}/commands            the commands the home offers on it
    POST /api/devices/{id}/commands            apply one command

There is no clock and no physics: ``time`` is the home's fixed ``base_time`` and
state changes only when a command is applied (see ``engine.py`` for the gold-
consistent semantics). Like SimuHome's simulator the process holds ONE home.

Worked example (HTTP, after ``python simulator.py --port 8099``)::

    POST /api/simulation/reset         <homes/home75.json>        -> data: the state
    POST /api/devices/bathroom_light/commands
         {"command_id": "set_brightness", "args": {"value": 20}}
         -> {"status": {"code": 200, ...}, "data": {"status": "applied",
             "attribute": "brightness", "value": 20}, "error": null}
    POST ... {"command_id": "set_brightness", "args": {"value": 101}}
         -> HTTP 400, {"status": {"code": 400, ...}, "data": null,
                       "error": "bathroom_light.brightness: 101 outside [0, 100]"}
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:  # allow `python .../HomeBench/simulator.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ami_agents.environment.integration.HomeBench.engine import (
    BASE_TIME, EngineError, HomeBenchEngine, load_home_file,
)

LOGGER = logging.getLogger(__name__)

# -- constants ------------------------------------------------------------------------

API_PREFIX = "/api"
#: CLI defaults only; a worker allocates free ports and passes them explicitly.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

app = FastAPI(title="HomeBench engine (SimuHome-shaped simulator)")

#: The one home this process holds, and the lock that serialises access to it (the
#: sync routes run in a thread pool).
_engine: Optional[HomeBenchEngine] = None
_lock = threading.Lock()


# -- envelope -------------------------------------------------------------------------


def envelope(data: Any = None, code: int = 200, message: str = "OK",
             error: Optional[str] = None) -> JSONResponse:
    """A response in SimuHome's ``{status: {code, message}, data, error}`` shape."""
    return JSONResponse(
        {"status": {"code": code, "message": message}, "data": data, "error": error},
        status_code=code)


def failure(code: int, message: str, error: str) -> JSONResponse:
    """An error envelope with HTTP status ``code`` (``SimuHomeClient`` raises on it)."""
    return envelope(None, code, message, error)


def _require_engine() -> Optional[JSONResponse]:
    """``None`` when a home is loaded, else the 409 envelope saying none is."""
    if _engine is None:
        return failure(409, "Conflict", "no home loaded: POST /api/simulation/reset")
    return None


# -- lifecycle and observation -------------------------------------------------------------


@app.get(f"{API_PREFIX}/__health__")
def health() -> JSONResponse:
    """Liveness: the worker waits for this before it loads an episode."""
    return envelope({"ok": True})


@app.post(f"{API_PREFIX}/simulation/reset")
async def reset(request: Request) -> JSONResponse:
    """Load a home (an ``initial_home_config``) and return its state.

    The body is taken verbatim, as SimuHome's reset takes an episode's config. A body
    without a ``rooms`` mapping is rejected with 400 and the previous home is kept.
    """
    global _engine
    # 1. The body must look like a home.
    try:
        config = await request.json()
    except ValueError:
        return failure(400, "Bad Request", "body is not JSON")
    if not isinstance(config, dict) or not isinstance(config.get("rooms"), dict):
        return failure(400, "Bad Request", "body needs a 'rooms' mapping")
    # 2. Replace the engine's home, or build the engine on first use.
    with _lock:
        if _engine is None:
            _engine = HomeBenchEngine(config)
            state = _engine.home_state()
        else:
            state = _engine.reset(config)
    LOGGER.info("home loaded: %d rooms", len(state["rooms"]))
    return envelope(state)


@app.post(f"{API_PREFIX}/simulation/fast_forward_to")
def fast_forward_to() -> JSONResponse:
    """SimuHome's evaluator calls this to reach a final tick; HomeBench has no time,
    so the answer is the state as it is (the contract's ``data`` field)."""
    missing = _require_engine()
    if missing is not None:
        return missing
    with _lock:
        return envelope(_engine.home_state())


@app.get(f"{API_PREFIX}/home/state")
def home_state() -> JSONResponse:
    """The whole home: ``{current_tick, current_time, rooms}``."""
    missing = _require_engine()
    if missing is not None:
        return missing
    with _lock:
        return envelope(_engine.home_state())


@app.get(f"{API_PREFIX}/time")
def current_time() -> JSONResponse:
    """The fixed clock: ``{"now": base_time}`` (HomeBench has no time)."""
    missing = _require_engine()
    if missing is not None:
        return missing
    with _lock:
        return envelope({"now": _engine.home_state().get("current_time", BASE_TIME)})


# -- devices ------------------------------------------------------------------------------


@app.get(API_PREFIX + "/devices/{device_id}/attributes")
def device_attributes(device_id: str) -> JSONResponse:
    """One device's attribute map."""
    missing = _require_engine()
    if missing is not None:
        return missing
    try:
        with _lock:
            return envelope(_engine.device(device_id)["attributes"])
    except EngineError as exc:
        return failure(404, "Not Found", str(exc))


@app.get(API_PREFIX + "/devices/{device_id}/commands")
def device_commands(device_id: str) -> JSONResponse:
    """The commands the home offers on a device (``engine.exposed_commands``)."""
    missing = _require_engine()
    if missing is not None:
        return missing
    try:
        with _lock:
            commands: List[str] = _engine.exposed_commands(device_id)
    except EngineError as exc:
        return failure(404, "Not Found", str(exc))
    return envelope({"device_id": device_id, "commands": commands})


@app.post(API_PREFIX + "/devices/{device_id}/commands")
async def apply_command(device_id: str, request: Request) -> JSONResponse:
    """Apply one command: body ``{"command_id": str, "args": {"value": v}?}``.

    404 for an unknown device, 400 for a malformed body or a command or value the
    engine refuses (the engine's message is the ``error``).
    """
    missing = _require_engine()
    if missing is not None:
        return missing
    # 1. The body: a command name, and the single argument in ``args.value``.
    try:
        body: Dict[str, Any] = await request.json()
    except ValueError:
        return failure(400, "Bad Request", "body is not JSON")
    command = body.get("command_id") if isinstance(body, dict) else None
    args = body.get("args") if isinstance(body, dict) else None
    if not isinstance(command, str) or not isinstance(args or {}, dict):
        return failure(400, "Bad Request",
                       "body needs 'command_id' (string) and optional 'args' (object)")
    # 2. Apply it; the engine validates the device, the command and the value.
    try:
        with _lock:
            known = device_id in _engine.device_ids()
            result = _engine.apply(device_id, command, (args or {}).get("value"))
    except EngineError as exc:
        return failure(404 if not known else 400,
                       "Not Found" if not known else "Bad Request", str(exc))
    return envelope({"device_id": device_id, "command": command,
                     "status": result.status, "attribute": result.attribute,
                     "value": result.value})


# -- command line -------------------------------------------------------------------------


def main() -> None:
    """Serve the engine; ``--home`` preloads a ``homes/home<id>.json``."""
    parser = argparse.ArgumentParser(description="Serve the HomeBench engine.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--home", default=None,
                        help="a homes/home<id>.json to load at start (else POST reset)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    global _engine
    if args.home:
        _engine = HomeBenchEngine(load_home_file(args.home))
    import uvicorn
    LOGGER.info("HomeBench engine on http://%s:%d%s  home=%s", args.host, args.port,
                API_PREFIX, args.home or "(none)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
