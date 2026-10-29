"""A device called by name, in a state or capability question.

The rule is the goals' (`goal_resolution.named_device_missing`): a name fits
only the device that has it. The parser is given the home's devices and says
how the question picks out its device (`device_reference`) and, for one called
by name, which device has that name (`artifact_name`), or null when none has:

    "is bedroom light 1 on?"   bedroom has only bedroom_dimmable_light_1
        device_reference "name", artifact_name null
        -> not looked up as "a light in the bedroom": the user is told
           "I can't find the light you named in the bedroom. It has:
            bedroom_dimmable_light_1."
    "is the bedroom light on?"
        device_reference "description" -> looked up by kind and room, as before

The EnvExplorer restricts a lookup to `artifact_name` when one is given.

Pure functions: the slots from the parser's output, and the answer's words.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ....shared.models.goal_structure import DEVICE_REFERENCES
from ....shared.utils.class_labels import label_for

# What the parser is shown when the home's devices could not be had: it then
# cannot tell a name from a description, and the slots are cleared.
NO_DEVICES = "(not available)"


# --------------------------------------------------------------------------
# The parser's slots
# --------------------------------------------------------------------------

def device_slots(parsed: Dict[str, Any], devices_known: bool) -> Dict[str, Optional[str]]:
    """`device_reference` and `artifact_name` from the parser's object.

    Without the home's devices the parser could not have found a name, so both
    are cleared rather than read as "named, and missing". A name is kept only
    for a device called by name."""
    # 1. Nothing to have chosen from.
    if not devices_known:
        return {"device_reference": None, "artifact_name": None}
    # 2. How the device is picked out; anything else is no answer.
    reference = parsed.get("device_reference")
    reference = reference if reference in DEVICE_REFERENCES else None
    # 3. The name, for a device called by name only.
    name = parsed.get("artifact_name")
    name = str(name).strip() if isinstance(name, str) and name.strip() else None
    return {"device_reference": reference,
            "artifact_name": name if reference == "name" else None}


def named_device_missing(extraction: Dict[str, Any]) -> bool:
    """True when the question calls its device by name and the home has no
    device of that name: never answered with another device."""
    return (extraction.get("device_reference") == "name"
            and not extraction.get("artifact_name"))


def state_lookup(extraction: Dict[str, Any]) -> Dict[str, Any]:
    """What to ask the EnvExplorer for a state question. For a named device the
    home does not have, only the devices of its kind in its room -- no
    property, so nothing is read: another device's value is no answer, and
    the reply lists them instead."""
    if not named_device_missing(extraction):
        return extraction
    return {**extraction, "device_property": None, "environment_variable": None}


# --------------------------------------------------------------------------
# The answer
# --------------------------------------------------------------------------

def missing_device_answer(extraction: Dict[str, Any], names: Iterable[str]) -> str:
    """The reply for a named device the home does not have: what was looked
    for, where, and the devices of that kind there, when there are any."""
    kind = label_for(extraction.get("device_class")).lower() or "device"
    room = label_for(extraction.get("location_class")).lower()
    where = f" in the {room}" if room else ""
    answer = f"I can't find the {kind} you named{where}."
    listed = sorted({str(n) for n in names if n})
    if listed:
        answer += f" It has: {', '.join(listed)}."
    return answer
