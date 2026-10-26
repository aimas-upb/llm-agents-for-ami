#!/usr/bin/env python3
"""Thin client for the HomeBench engine's HTTP API (``simulator.py``).

``HomeBenchClient`` has the method names the bench and SHTD-style code call on
``SimuHomeClient`` (``health``, ``reset``, ``home_state``, ``current_time``,
``execute_command``, ``write_attribute``, plus the two evaluator-contract calls
``reset_simulation`` and ``fast_forward_to``), so ``Target(sim, ...)``, the worker
and the evaluator treat the two benchmarks alike. The transport and the envelope
handling are SimuHome's own: ``SimuHomeClient`` is the base class, only the
actuation calls differ, because HomeBench has no Matter clusters::

    client = HomeBenchClient("http://127.0.0.1:8099/api")
    client.reset(initial_home_config)                       # load a home
    client.execute_command("bathroom_light", "set_brightness", {"value": 20})
    client.write_attribute("bathroom_light", "brightness", 20)    # same, by attribute
    client.home_state()["rooms"]["bathroom"]["devices"]     # the state afterwards

Responses are ``{"status": {"code", "message"}, "data", "error"}``; a failure raises
``SimuHomeError`` (the same exception SimuHome's client raises).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ami_agents.environment.integration.SimuHome.sim_client import (
    SimuHomeClient, SimuHomeError,
)

LOGGER = logging.getLogger(__name__)

__all__ = ["HomeBenchClient", "SimuHomeError"]


class HomeBenchClient(SimuHomeClient):
    """Synchronous client for ``simulator.py``. One instance per engine process.

    Inherits ``health``, ``reset``, ``current_time``, ``home_state``, ``device_attributes``,
    ``reset_simulation`` and ``fast_forward_to`` from ``SimuHomeClient``: the routes and
    the envelope are the same. Methods that have no HomeBench meaning (Matter
    structure, scheduled workflows, tick intervals) are inherited too but are not
    served by the engine and raise ``SimuHomeError`` (HTTP 404) if called.
    """

    # -- actuation -------------------------------------------------------------

    def execute_command(  # type: ignore[override]
        self,
        device_id: str,
        command_id: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Apply a HomeBench command (``turn_on``, ``set_brightness``, ...).

        ``args`` is ``{"value": v}`` for a setter and empty for a state command.
        Returns the engine's ``{device_id, command, status, attribute, value}``.
        """
        return self._request(
            "POST", f"/devices/{device_id}/commands",
            {"command_id": command_id, "args": args or {}})

    def write_attribute(  # type: ignore[override]
        self, device_id: str, attribute_id: str, value: Any
    ) -> Any:
        """Write a settable attribute: ``set_<attribute>`` with ``value``.

        The engine models attribute writes as the setter command of the same name,
        so this is ``execute_command(device_id, "set_<attribute>", {"value": v})``.
        """
        return self.execute_command(
            device_id, f"set_{attribute_id}", {"value": value})

    def device_commands(self, device_id: str) -> Any:
        """The commands the home offers on a device (``exposed_commands``)."""
        return self._request("GET", f"/devices/{device_id}/commands")
