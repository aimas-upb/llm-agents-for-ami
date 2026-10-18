"""Naming affordances in a BT plan: by device, td:title and class -- no ids, no URLs.

The planner model is shown each device with its actions and its properties,
each by its td:title and its class. Its tree names an affordance the same way:

    {"type": "action", "artifact": "utility_room_laundry_washer_1",
     "affordance": "currentMode", "affordance_class": "homeont:SetModeCommand",
     "parameters": {"value": 2}}

and code finds the URL. The node's type decides which side is looked at: an
action node only among the device's actions, a condition or wait_condition
only among its properties. An action and a property may share a title (the
washer's `currentMode` is both); the kind keeps them apart, so a wait can
never read an action's URL.

Example -- the washer, as the planner lists it:

    utility_room_laundry_washer_1
      actions:
      - currentMode | homeont:SetModeCommand | input {"value": {"type": "integer"}} | acts upon ...
      properties:
      - currentMode | homeont:LaundryWasherCurrentMode | reads {"type": "integer"}

    resolve("action", washer, "currentMode", "homeont:SetModeCommand")
        -> the action, its URL
    resolve("wait_condition", washer, "currentMode", "homeont:LaundryWasherCurrentMode")
        -> the property, its URL
    resolve("wait_condition", washer, "currentMode", "homeont:SetModeCommand")
        -> error: no property "currentMode" of that class; its properties are ...

Affordance dicts come in the shapes the planners build (`specific_capability_
context`: `type: action_affordance`, `semantic_types`; the EnvExplorer's:
`affordance_type: action`, `action_name`); `AffordanceCatalog` reads both.

    AffordanceCatalog(affordances)        the devices, their actions and properties
    catalog.resolve(node_type, device, title, cls)  (affordance, None) | (None, error)
    catalog.entry_for_url(url)            the entry with that URL (hints name it)
    catalog.render()                      the listing the model is shown
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# The node types that read a property; every other leaf invokes an action.
READING_NODES = ("condition", "wait_condition")

PLURAL = {"action": "actions", "property": "properties"}

# Field names of the reference in a node.
DEVICE_FIELD, TITLE_FIELD, CLASS_FIELD = "artifact", "affordance", "affordance_class"


# -- one affordance ----------------------------------------------------------

def _kind(aff: Dict[str, Any]) -> str:
    """"action" or "property": from `type` ("action_affordance") or
    `affordance_type` ("action"); an affordance with neither is an action."""
    raw = str(aff.get("type") or aff.get("affordance_type") or "action").lower()
    return "property" if raw.startswith("property") else "action"


def _device(aff: Dict[str, Any]) -> str:
    """The device's td:title, or the last segment of its id when no title came."""
    name = aff.get("artifact_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    artifact = str(aff.get("artifact_id") or aff.get("artifact_uri") or "")
    artifact = artifact.split("#")[0].rstrip("/")
    return artifact.rsplit("/", 1)[-1] if artifact else "environment"


def _target(aff: Dict[str, Any]) -> str:
    """The affordance's URL (hctl:hasTarget), from whichever field carries it."""
    form = aff.get("form") if isinstance(aff.get("form"), dict) else {}
    return str(aff.get("target") or form.get("href") or aff.get("affordance_uri")
               or aff.get("href") or "")


@dataclass
class Entry:
    """One affordance of one device, as the model names it."""

    device: str
    title: str
    kind: str                           # "action" | "property"
    classes: List[str]                  # its most specific classes; may be empty
    url: str
    affordance: Dict[str, Any]          # the dict it came from

    def line(self) -> str:
        """The entry as the model reads it: title | class | input or reading |
        what it acts upon."""
        parts = [self.title, ", ".join(self.classes) or "(no class)"]
        if self.kind == "action":
            schema = self.affordance.get("input_schema")
            if isinstance(schema, dict):
                parts.append("input " + json.dumps(schema.get("properties") or schema))
        else:
            schema = self.affordance.get("output_schema")
            if isinstance(schema, dict):
                parts.append("reads " + json.dumps(schema))
        description = str(self.affordance.get("description") or "").strip()
        if description and not (self.kind == "property" and description.startswith("reads ")
                                and "output_schema" in self.affordance):
            parts.append(description)
        return " | ".join(parts)


# -- the catalog -------------------------------------------------------------

@dataclass
class AffordanceCatalog:
    """Every affordance the planner may use, by device, in the order given."""

    affordances: List[Dict[str, Any]]
    devices: Dict[str, List[Entry]] = field(default_factory=dict)
    by_url: Dict[str, Entry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Group the affordances by device; index them by URL for the hints."""
        for aff in self.affordances or []:
            if not isinstance(aff, dict):
                continue
            title = str(aff.get("action_name") or aff.get("name") or "").strip()
            if not title:
                continue
            classes = [str(c) for c in aff.get("semantic_types") or [] if c]
            entry = Entry(device=_device(aff), title=title, kind=_kind(aff),
                          classes=classes, url=_target(aff), affordance=aff)
            self.devices.setdefault(entry.device, []).append(entry)
            if entry.url:
                self.by_url.setdefault(entry.url.rstrip("/"), entry)

    def entry_for_url(self, url: Any) -> Optional[Entry]:
        """The entry whose URL is ``url``, or None."""
        return self.by_url.get(str(url or "").rstrip("/"))

    # -- resolving a node's reference --------------------------------------

    def resolve(self, node_type: str, device: Any, title: Any, cls: Any
                ) -> Tuple[Optional[Entry], Optional[str]]:
        """(the entry a node names, None) or (None, the error the model reads).

        A reading node (condition, wait_condition) is looked up among the
        device's properties, any other among its actions; the title must match
        exactly, and the class must be one of the entry's classes (an entry
        listed with no class takes any)."""
        kind = "property" if node_type in READING_NODES else "action"
        # 1. The device.
        entries = self.devices.get(str(device or ""))
        if entries is None:
            return None, (f"no device {device!r}; the devices are: "
                          + (", ".join(self.devices) or "none"))
        # 2. Its affordances of the right kind, with that title.
        same_kind = [e for e in entries if e.kind == kind]
        titled = [e for e in same_kind if e.title == str(title or "")]
        listing = "; ".join(f"{e.title} ({', '.join(e.classes) or 'no class'})"
                            for e in same_kind) or "none"
        if not titled:
            return None, (f"{device} has no {kind} {title!r} -- {node_type} nodes name one of "
                          f"its {PLURAL[kind]}: {listing}")
        # 3. The class.
        for entry in titled:
            if not entry.classes or str(cls or "") in entry.classes:
                return entry, None
        return None, (f"{device}'s {kind} {title!r} is not of class {cls!r} -- it is "
                      + " / ".join(", ".join(e.classes) for e in titled))

    # -- the listing -------------------------------------------------------

    def render(self) -> str:
        """The devices with their actions and properties, as the model is
        shown them. No URLs, no ids."""
        lines: List[str] = []
        for device, entries in self.devices.items():
            lines.append(device)
            for kind, heading in PLURAL.items():
                own = [e for e in entries if e.kind == kind]
                if own:
                    lines.append(f"  {heading}:")
                    lines.extend(f"  - {e.line()}" for e in own)
            lines.append("")
        return "\n".join(lines).rstrip()

    def name_of(self, url: Any) -> str:
        """How a hint names the affordance at ``url``: device / title / class,
        or the URL's last segment when it is not in the catalog (then the model
        cannot use it)."""
        entry = self.entry_for_url(url)
        if entry is None:
            return f"{str(url or '').rstrip('/').rsplit('/', 1)[-1]} (not in the device list)"
        return (f"{entry.device} / {entry.kind} {entry.title}"
                + (f" ({entry.classes[0]})" if entry.classes else ""))
