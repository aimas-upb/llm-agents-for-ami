#!/usr/bin/env python3
"""
Thin client for the SimuHome simulator's HTTP API.

SimuHome (dev/SimuHome) ships a FastAPI server -- `src/simulator/api/app.py`,
started with `python -m src.simulator.api.app` and `SERVER_PORT` -- that speaks
native Matter: clusters, attributes and commands, with no Home Assistant in the
middle. This wraps the handful of endpoints SHTD needs.

Two facts about that API shape the design here:

  * A scenario loads by POSTing a benchmark episode's `initial_home_config`
    VERBATIM to /api/simulation/reset -- `SimulationConfig` takes exactly
    {tick_interval, base_time, rooms}. No conversion, no intermediate format.
  * `GET /api/home/state` returns the whole world in one call: the simulated
    clock, and per room both the environmental aggregates and every device with
    its attributes. That single response drives both the inspector and the
    change-notification differ, so neither needs a chattier path.

Responses are wrapped in an envelope: {"status": {"code", "message"},
"data": ..., "error": ...}. `_unwrap` raises on failure and returns `data`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:8000/api"
BASE_URL_ENV_VAR = "SIMULATOR_API_BASE_URL"


class SimuHomeError(RuntimeError):
    """The simulator rejected a call, or could not be reached."""


def resolve_base_url(base_url: Optional[str] = None) -> str:
    resolved = base_url or os.getenv(BASE_URL_ENV_VAR) or DEFAULT_BASE_URL
    return resolved.rstrip("/")


class SimuHomeClient:
    """Synchronous client. One instance per simulator process.

    The simulator holds ONE home per process (`home` is module-level in
    `src/simulator/api/routes.py`, rebound by reset under a lock), so a client
    instance is bound to whichever home its server currently holds.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = resolve_base_url(base_url)
        self.timeout = timeout

    # -- transport ---------------------------------------------------------

    def _request(self, method: str, path: str, payload: Any = None) -> Any:
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise SimuHomeError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
        except OSError as exc:
            raise SimuHomeError(f"{method} {path} -> {type(exc).__name__}: {exc}") from exc
        return self._unwrap(json.loads(body or "{}"), f"{method} {path}")

    @staticmethod
    def _unwrap(envelope: Dict[str, Any], label: str) -> Any:
        status = envelope.get("status") or {}
        code = status.get("code")
        if code is not None and int(code) >= 400:
            raise SimuHomeError(f"{label} -> {code} {status.get('message')}: {envelope.get('error')}")
        if envelope.get("error"):
            raise SimuHomeError(f"{label} -> {envelope['error']}")
        return envelope.get("data")

    # -- lifecycle ---------------------------------------------------------

    def health(self) -> bool:
        try:
            self._request("GET", "/__health__")
            return True
        except SimuHomeError:
            return False

    def reset(self, initial_home_config: Dict[str, Any]) -> Any:
        """Load a scenario. Takes a benchmark episode's `initial_home_config` as-is."""
        return self._request("POST", "/simulation/reset", initial_home_config)

    def set_tick_interval(self, seconds: float) -> Any:
        return self._request("POST", "/simulation/tick_interval", {"tick_interval": seconds})

    # -- observation -------------------------------------------------------

    def home_state(self) -> Dict[str, Any]:
        """The whole world: clock, rooms, room aggregates, devices + attributes."""
        return self._request("GET", "/home/state")

    def rooms(self) -> Any:
        return self._request("GET", "/rooms")

    def room_states(self, room_id: str) -> Any:
        return self._request("GET", f"/rooms/{room_id}/states")

    def device_attributes(self, device_id: str) -> Any:
        return self._request("GET", f"/devices/{device_id}/attributes")

    def device_structure(self, device_id: str) -> Any:
        return self._request("GET", f"/devices/{device_id}/structure")

    # -- actuation ---------------------------------------------------------

    def execute_command(
        self,
        device_id: str,
        cluster_id: str,
        command_id: str,
        args: Optional[Dict[str, Any]] = None,
        endpoint_id: int = 1,
    ) -> Any:
        """Invoke a Matter command.

        The path for attributes the spec marks read-only: OnOff.OnOff is not
        writable, but On/Off/Toggle drive it.
        """
        return self._request(
            "POST",
            f"/devices/{device_id}/commands",
            {
                "endpoint_id": endpoint_id,
                "cluster_id": cluster_id,
                "command_id": command_id,
                "args": args or {},
            },
        )

    def write_attribute(
        self,
        device_id: str,
        cluster_id: str,
        attribute_id: str,
        value: Any,
        endpoint_id: int = 1,
    ) -> Any:
        """Write a spec-writable attribute, e.g. FanControl.PercentSetting."""
        return self._request(
            "POST",
            f"/devices/{device_id}/attributes/write",
            {
                "endpoint_id": endpoint_id,
                "cluster_id": cluster_id,
                "attribute_id": attribute_id,
                "value": value,
            },
        )

    # -- evaluator contract ------------------------------------------------
    #
    # SimuHome's own episode evaluator (src/pipelines/episode_evaluation/) is
    # handed a client and calls these two by name. Its scoring is
    # COUNTERFACTUAL: it resets the simulator, fast-forwards a baseline to the
    # episode's final tick with no agent acting, and asks whether each goal's
    # variable moved further than it would have on its own. So the evaluator
    # needs the simulator to ITSELF after the episode ends.
    #
    # Both return the raw envelope, because that is what the evaluator expects
    # -- it unwraps with `.get("data", {})` itself.

    def reset_simulation(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Alias of `reset` under the name the evaluator calls."""
        return self._request_envelope("POST", "/simulation/reset", cfg)

    def fast_forward_to(
        self, to_tick: int, room_ids: Optional[list] = None
    ) -> Dict[str, Any]:
        """Advance the simulation to a tick without an agent acting.

        `room_ids` restricts the computation to the rooms a goal mentions, which
        is how the evaluator keeps the baseline cheap.
        """
        payload: Dict[str, Any] = {"to_tick": int(to_tick)}
        if room_ids is not None:
            payload["room_ids"] = list(room_ids)
        return self._request_envelope("POST", "/simulation/fast_forward_to", payload)

    def _request_envelope(self, method: str, path: str, payload: Any = None) -> Any:
        """Like `_request`, but returns the whole envelope rather than `data`."""
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise SimuHomeError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
        except OSError as exc:
            raise SimuHomeError(f"{method} {path} -> {type(exc).__name__}: {exc}") from exc
        return json.loads(body or "{}")

    # -- ground truth ------------------------------------------------------

    def control_rules(self, state: str) -> Any:
        """SimuHome's own causal model for one of
        temperature|humidity|air_quality|illuminance: which device types affect
        it, and the required/optional actions to do so. Used to cross-check the
        hand-curated actuation_effects table.

        The server nests the rules as a JSON *string* under `control_rules`;
        parse it so callers get data rather than text.
        """
        payload = self._request("GET", f"/environment/control_rules/{state}")
        rules = payload.get("control_rules") if isinstance(payload, dict) else payload
        if isinstance(rules, str):
            try:
                rules = json.loads(rules)
            except ValueError as exc:
                raise SimuHomeError(f"control_rules({state}) is not valid JSON") from exc
        return rules
