"""Turn an ENV_CAPABILITY_QUERY response into the sentence the user reads.

Most outcomes are answered here, without an LLM: "no", "the purifier cannot do
that", and "yes, these devices can" state a fact the response already carries.
Only a `query` that found something -- a list to be phrased in terms of what
was asked, with option codes to be named by their meaning -- goes to a model;
see `behaviours/capability_answer.py`.

Nothing here emits an identifier. Classes are named through `label_for`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ....shared.utils.class_labels import label_for

# How a property's branch reads to a person.
_NATURE = {
    "actuatable": "changeable",
    "state": "reported",
    "capability": "supported options",
    "measurement": "measured",
    "environment": "room environment",
}


def needs_interpretation(response: Dict[str, Any]) -> bool:
    """Is a model needed to phrase this answer?

    Only for a `query` that found something: listing devices, options or an
    inventory in terms of the user's question. A yes/no, a "cannot" and a "none"
    are built here.
    """
    return (response.get("outcome") == "found"
            and response.get("request_performative") != "query_if")


def phrase(response: Dict[str, Any]) -> str:
    """The deterministic sentence for any outcome."""
    if "error" in response:
        return ("I could not check that: "
                f"{response.get('detail') or response['error']}")

    outcome = response.get("outcome")
    if outcome == "indeterminate":
        return indeterminate(response)
    if outcome == "none":
        return none(response)
    if outcome == "device_lacks_capability":
        return device_lacks(response)
    if response.get("request_performative") == "query_if":
        return yes(response)
    return fallback(response)


def indeterminate(response: Dict[str, Any]) -> str:
    return ("I could not tell which capability you are asking about. "
            "Which device or room, and what would you like it to do or report?")


def none(response: Dict[str, Any]) -> str:
    """Nothing in scope provides it. A named device absent is said as such."""
    query = response.get("query") or {}
    place = _place(query.get("location_class"))
    device = label_for(query.get("device_class"))
    lead = "No. " if response.get("request_performative") == "query_if" else ""

    if device:
        return f"{lead}There is no {device}{place}."
    return f"{lead}Nothing{place} {_provides(query)}."


def device_lacks(response: Dict[str, Any]) -> str:
    """The device is there; what was asked of it is not."""
    query = response.get("query") or {}
    devices = _devices(response.get("entries") or [])
    lead = "No. " if response.get("request_performative") == "query_if" else ""
    verb = "do not" if len(devices) > 1 else "does not"
    return f"{lead}{_capital(_join(devices))} {verb} {_provides(query, bare=True)}."


def yes(response: Dict[str, Any]) -> str:
    """A `query_if` that found something: yes, and by what."""
    devices = _devices(response.get("entries") or [])
    return f"Yes: {_join(devices)}."


def fallback(response: Dict[str, Any]) -> str:
    """Plain phrasing of a found list, for when the model call could not be made.

    Blunter than the model -- capability labels and raw option codes -- but
    correct and never empty.
    """
    entries = response.get("entries") or []
    if not entries:
        return none(response)
    by_device: Dict[str, List[str]] = {}
    for entry in entries:
        device = _device(entry)
        what = label_for(entry.get("affordance_type"))
        values = entry.get("permitted_values") or {}
        if values.get("meaning"):
            what += f" ({values['meaning']})"
        elif "minimum" in values or "maximum" in values:
            what += f" ({values.get('minimum', '?')} to {values.get('maximum', '?')})"
        if what:
            by_device.setdefault(device, []).append(what)
        else:
            by_device.setdefault(device, [])
    parts = [f"{device}: {', '.join(whats)}" if whats else device
             for device, whats in by_device.items()]
    return _capital("; ".join(parts)) + "."


def entries_for_prompt(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The entries as the model should see them: labels, no identifiers or URLs."""
    out = []
    for entry in entries:
        item: Dict[str, Any] = {"device": _kind(entry)}
        if entry.get("workspace_name"):
            item["room"] = entry["workspace_name"]
        for key in ("manufacturer", "model"):
            if entry.get(key):
                item[key] = entry[key]
        if entry.get("affordance_type"):
            item["kind"] = entry.get("affordance_kind")
            item["capability"] = label_for(entry["affordance_type"])
        if entry.get("property_branch"):
            item["nature"] = _NATURE.get(entry["property_branch"],
                                         entry["property_branch"])
        if entry.get("permitted_values"):
            item["allowed_values"] = entry["permitted_values"]
        if entry.get("effect_on"):
            item["affects"] = label_for(entry["effect_on"])
            if entry.get("effect_direction"):
                item["direction"] = entry["effect_direction"]
        out.append(item)
    return out


# -- helpers ------------------------------------------------------------

def _provides(query: Dict[str, Any], bare: bool = False) -> str:
    """What was asked, as a predicate: "can be set to Level Control Brightness"."""
    environment = label_for(query.get("environment_variable"))
    command = query.get("command_class")
    prop = query.get("property_class")
    if environment and command:
        return f"can change the {environment}"
    if prop in ("schema:manufacturer", "schema:model"):
        fact = "manufacturer" if prop == "schema:manufacturer" else "model"
        return f"state{'' if bare else 's'} its {fact}"
    if prop:
        return f"provide{'' if bare else 's'} {label_for(prop)}"
    if environment:
        return f"sense{'' if bare else 's'} the {environment}"
    if command:
        return f"support{'' if bare else 's'} the {label_for(command)}"
    return "do that" if bare else "can do that"


def _place(location_class: Optional[str]) -> str:
    where = label_for(location_class)
    return f" in the {where}" if where else ""


def _kind(entry: Dict[str, Any]) -> str:
    return (label_for(entry.get("artifact_type"))
            or entry.get("artifact_name") or "device")


def _device(entry: Dict[str, Any]) -> str:
    """"the Air Purifier in the Kitchen"."""
    room = entry.get("workspace_name")
    return f"the {_kind(entry)}" + (f" in the {room}" if room else "")


def _devices(entries: List[Dict[str, Any]]) -> List[str]:
    """Each device once, in order."""
    seen: Dict[str, None] = {}
    for entry in entries:
        seen.setdefault(_device(entry), None)
    return list(seen)


def _join(parts: List[str]) -> str:
    """`a`, `a and b`, `a, b and c`."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _capital(text: str) -> str:
    # `.capitalize()` would lowercase the rest ("TV" -> "Tv").
    return text[:1].upper() + text[1:]
