"""Turn an ENV_STATE response into the sentence the user reads.

Three of the four outcomes are answered here, without an LLM: they state a fact
about the environment, and the response already carries everything the sentence
needs. Only the two cases that must interpret the user's *phrasing* -- a
mismatch, and a reading whose value is a dictionary -- go to a model; see
`behaviours/state_answer.py`.

Nothing here emits an identifier. Classes are named through `label_for`, so a
user never sees `homeont:OnOffLightOnOff` or a URL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ....shared.utils.class_labels import label_for


def needs_interpretation(response: Dict[str, Any]) -> bool:
    """Does answering this require reading the user's phrasing?

    True in three cases:

    - a *mismatch*, where several unrelated properties matched and only the
      phrasing says whether that is an answer or an ambiguity;
    - a *resolved artifact*, where the device was found but no affordance
      reports what was asked -- the question decides whether its make and model
      answer it or whether the property is simply unavailable;
    - a value that is a *dictionary*, where one of its fields has to be chosen.
      A list is not a dictionary: it is a value in its own right, stated whole.
    """
    outcome = response.get("outcome")
    if outcome in ("mismatched_affordance", "resolved_artifact"):
        return True
    if outcome != "resolved_affordance":
        return False
    return any(isinstance(a.get("value"), dict)
               for a in response.get("affordances") or [])


# -- the three deterministic outcomes -----------------------------------

def no_affordance(response: Dict[str, Any]) -> str:
    """Nothing in scope matched at all; say what was missing.

    When a device kind was named, nothing of that kind is there -- which is a
    statement about the device, not about a property it fails to report. Saying
    "there is no Refrigerator in the Kitchen that reports Product Name" would
    imply a refrigerator exists.
    """
    query = response.get("query") or {}
    where = label_for(query.get("location_class"))
    prop = label_for(query.get("property_class"))
    device = label_for(query.get("device_class"))
    place = f" in the {where}" if where else ""

    if device:
        sentence = f"There is no {device}{place}."
    elif prop:
        sentence = f"There is no device{place} that reports {prop}."
    else:
        sentence = f"There is no device{place} that can answer that."
    return f"{sentence} Is there something else you would like to check?"


def indeterminate_affordance(response: Dict[str, Any]) -> str:
    """The request named nothing readable; ask for what is missing."""
    query = response.get("query") or {}
    where = label_for(query.get("location_class"))
    device = label_for(query.get("device_class"))

    known = where or device
    if known:
        return (f"I can look that up for the {known}. "
                "Which property would you like to know about?")
    return ("I could not tell what you would like to know. "
            "Which property are you asking about, and in which room?")


def resolved_basic(response: Dict[str, Any]) -> str:
    """A resolution whose values need no interpreting.

    One affordance is stated directly. Several -- the aggregation case, all of
    one semantic type -- are stated individually and then summarised, because
    sensors disagree and picking one silently would be wrong.
    """
    affordances = response.get("affordances") or []
    if not affordances:
        return "I could not read that."

    if len(affordances) == 1:
        return _one(affordances[0])

    prop = label_for(affordances[0].get("affordance_type"))
    where = affordances[0].get("workspace_name")
    place = f" in the {where}" if where else ""

    parts = [f"the {_device(a)} reports {_render(a.get('value'))}"
             for a in affordances]
    listed = _join(parts)
    summary = _summarise([a.get("value") for a in affordances])
    return f"{prop}{place}: {listed}.{summary}"


def fallback(response: Dict[str, Any]) -> str:
    """Plain phrasing for when the LLM call could not be made.

    Correct and never empty, at the cost of being blunter than the model would
    be: a mismatch lists what matched, a reading prints what it read.
    """
    affordances = response.get("affordances") or []
    if not affordances:
        return "I could not read that."

    outcome = response.get("outcome")

    if outcome == "mismatched_affordance":
        where = affordances[0].get("workspace_name")
        place = f" in the {where}" if where else ""
        names = _join([f"the {_describe(a)}" for a in affordances])
        return (f"Several devices{place} match that: {names}. "
                "Which one did you mean?")

    if outcome == "resolved_artifact":
        # The device is there; what was asked about it is not. Report the
        # device itself, which is what the caller has to work with.
        asked = label_for((response.get("query") or {}).get("property_class"))
        found = _join([f"the {_describe(a)}" for a in affordances])
        where = affordances[0].get("workspace_name")
        place = f" in the {where}" if where else ""
        if asked:
            return f"I found {found}{place}, but nothing there reports {asked}."
        return f"I found {found}{place}."

    # `.capitalize()` would lowercase the rest, turning "TV" into "Tv" and
    # "BBC One" into "bbc one". Only the first character is ours to change.
    parts = [f"the {_device(a)} reports {_render(a.get('value'))}"
             for a in affordances]
    sentence = _join(parts)
    return f"{sentence[:1].upper()}{sentence[1:]}."


# -- helpers ------------------------------------------------------------

def _device(affordance: Dict[str, Any]) -> str:
    """How to name one device: its kind, falling back to its instance name."""
    return (label_for(affordance.get("artifact_type"))
            or affordance.get("artifact_name") or "device")


def _describe(affordance: Dict[str, Any]) -> str:
    """The device, with its make where the graph states one.

    "LG Electronics Freezer" rather than "Freezer": when several devices are
    being told apart, or when the question is about the device itself, the make
    is often the distinguishing fact.
    """
    device = _device(affordance)
    manufacturer = affordance.get("manufacturer")
    return f"{manufacturer} {device}" if manufacturer else device


def _one(affordance: Dict[str, Any]) -> str:
    prop = label_for(affordance.get("affordance_type"))
    device = _device(affordance)
    where = affordance.get("workspace_name")
    place = f" in the {where}" if where else ""
    value = affordance.get("value")

    # "The On/Off Light On/Off of the On/Off Light is on" stutters, because a
    # device-specific power property repeats the device it belongs to. Whether
    # a thing is on is said of the thing, not of its property.
    if isinstance(value, bool):
        return f"The {device}{place} is {_render(value)}."
    return f"The {prop} of the {device}{place} is {_render(value)}."


def _render(value: Any) -> str:
    """One value, as a user would read it."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return _join([_render(v) for v in value]) if value else "nothing"
    if value is None:
        return "unavailable"
    return str(value)


def _join(parts: Sequence[str]) -> str:
    """`a`, `a and b`, `a, b and c`."""
    parts = list(parts)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _summarise(values: List[Any]) -> str:
    """The aggregate across several readings of one property.

    Numbers average; anything else takes a majority. A tie is reported as a tie
    rather than resolved arbitrarily -- the disagreement is the information.
    """
    if _all_numeric(values):
        mean = sum(values) / len(values)
        return f" On average, {_trim(mean)}."

    counts: Dict[str, int] = {}
    for value in values:
        counts[_render(value)] = counts.get(_render(value), 0) + 1
    top = max(counts.values())
    winners = sorted(name for name, n in counts.items() if n == top)
    if len(winners) > 1:
        return f" They are split between {_join(winners)}."
    return f" Most are {winners[0]}."


def _all_numeric(values: List[Any]) -> bool:
    return bool(values) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _trim(number: float) -> str:
    """Drop a trailing `.0`, keep two decimals otherwise."""
    rounded = round(number, 2)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)
