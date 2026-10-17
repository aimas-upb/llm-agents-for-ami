"""What our pipeline asked of the home, as the SimuHome tool calls it stands for.

SimuHome's evaluators require that the agent "invoked" certain tools -- mostly
`get_room_devices` / `get_room_states` for the rooms involved. Our agents never
call those tools; they ask the EnvExplorer, which answers from the discovered
home. Only what is asked per request counts here; discovery is offline.

    request-time query                                        SimuHome tool
    env-capability query (goal context, candidate lookup,     get_room_devices {room_id}
      state query with a location): each room it answers
      about
    env-state read of a room property, or of a room's        get_room_states {room_id}
      ambient sensor (it serves the room's state)
    env-state read / plan read of a device property          get_attribute {device_id}
    action POST during execution                             execute_command {device_id}

Entries have SimuHome's `tools_invoked` shape ({tool, params, outcome}) plus
`source` (which of our queries made it); repeated calls are kept once.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# SHTD's URL layout: /workspaces/<home>/<room>[/artifacts/<device>]/(properties|actions)/<name>
_URL = re.compile(r"/workspaces/(?P<home>[^/]+)/(?P<room>[^/]+)"
                  r"(?:/artifacts/(?P<device>[^/]+))?/(?P<kind>properties|actions)/"
                  r"(?P<name>[^/?#]+)")


def parse_url(url: str) -> Optional[Dict[str, str]]:
    """{room, device?, kind, name} of an SHTD property or action URL."""
    match = _URL.search(str(url or ""))
    return match.groupdict() if match else None


class ToolLog:
    """The tool calls one episode amounts to."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self._seen = set()
        # SHTD's virtual ambient sensors, id -> room: a reading of one IS the
        # room's state, so it counts as `get_room_states` (set by the pipeline).
        self.sensors: Dict[str, str] = {}

    def add(self, tool: str, params: Dict[str, Any], source: str, ok: bool = True) -> None:
        key = (tool, tuple(sorted(params.items())), ok)
        if key in self._seen:
            return
        self._seen.add(key)
        self.entries.append({"tool": tool, "params": dict(params),
                             "outcome": {"ok": ok, "status_code": 200 if ok else None,
                                         "error_type": None},
                             "source": source})

    def room_devices(self, room: str, source: str) -> None:
        self.add("get_room_devices", {"room_id": room}, source)

    def read(self, url: str, home: Any, source: str) -> None:
        """A property read, by its URL."""
        parts = parse_url(url)
        if not parts or parts["kind"] != "properties":
            return
        if parts["device"] in self.sensors:
            self.add("get_room_states", {"room_id": self.sensors[parts["device"]]}, source)
        elif parts["device"]:
            self.add("get_attribute", {"device_id": parts["device"]}, source)
        else:
            self.add("get_room_states", {"room_id": parts["room"]}, source)

    def action(self, url: str, ok: bool) -> None:
        parts = parse_url(url)
        if parts and parts["device"]:
            self.add("execute_command", {"device_id": parts["device"]}, "execution", ok)

    def tools_invoked(self) -> List[Dict[str, Any]]:
        """SimuHome's `tools_invoked`, without our `source` field."""
        return [{k: v for k, v in e.items() if k != "source"} for e in self.entries]
