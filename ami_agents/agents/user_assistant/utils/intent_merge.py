"""Keep a timed request about one device in one intent.

The segmenter splits "Start dryer 1 at the right time on Normal dryness. Time it
so dishwasher 1 and dryer 1 finish at the same time. Make sure dryer 1 ends in
Stopped state" into three atomic intents. Structured one by one, the first
starts the dryer now and the second has no action to plan: the timing is lost
between them. The dependency structurer handles several goals of one request,
so the pieces about the same device are merged back into one intent before
structuring whenever one of them carries a timing phrase.

    merge_coupled_intents(intents) -> intents, with the GOAL_REQUEST intents of
    one utterance that name the same device (noun + number) joined when one of
    them times the other (finish together, at the same time, time it so, at the
    right time, when / after / before / until ... finishes, wait for ...).

Two more pieces attach to their neighbour even without a shared device: an
intent that is only a wait ("Wait for washer 1 to finish") joins the goal that
follows it, and a goal that continues the previous one -- opening with "after
N minutes", or timing "it" without naming a device ("Pause it when dryer 1
finishes") -- joins the goal before it.

Qualifiers of merged intents are united, with `temporal_dependency` added.
Queries and intents about other devices are left as they are.
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple

from ..models import AtomicIntent

# Device nouns the benchmark's homes use, with their common variants.
_DEVICE_WORDS = (
    r"washing machines?|washers?|dryers?|dish ?washers?|robot(?:ic)? vacuums?(?: cleaners?)?|"
    r"vacuum cleaners?|rvcs?|lights?|dimmer(?: lights?)?|lamps?|fans?|air conditioners?|acs?|"
    r"dehumidifiers?|humidifiers?|air purifiers?|purifiers?|heat pumps?|heaters?|"
    r"refrigerators?|fridges?|freezers?|tvs?|televisions?|blinds?|shades?|curtains?|"
    r"window coverings?|ovens?"
)
_DEVICE = re.compile(rf"\b({_DEVICE_WORDS})\b(?:\s+(\d+))?", re.I)

_TIMING = re.compile(
    r"\b(?:time it so|at the same time|at once|finish(?:es|ed)?\s+(?:up\s+)?together|"
    r"both\s+(?:are\s+)?(?:done|ready|finished)|at the right time|wait for|until|"
    r"(?:when|after|before|once)\b[^.]*?\b(?:finish(?:es|ed)?|done|ends?|complete[sd]?|"
    r"stops?))\b", re.I)

_ALIASES = {"washing machine": "washer", "dish washer": "dishwasher", "robotic vacuum": "robot vacuum",
            "vacuum cleaner": "robot vacuum", "rvc": "robot vacuum", "lamp": "light",
            "dimmer light": "dimmer", "air conditioner": "ac", "television": "tv",
            "fridge": "refrigerator", "air purifier": "purifier", "shade": "blind",
            "curtain": "blind", "window covering": "blind"}


# An intent that only waits for an event: no action of its own.
_WAIT_ONLY = re.compile(
    r"^\s*(?:(?:and|then|also|please)\s+)*(?:wait|hold|hang on)\b[^.]*?\b(?:finish(?:es|ed)?|"
    r"done|ends?|complete[sd]?|stops?)\b\.?\s*$", re.I)
# A goal that continues the previous one: it opens with a relative delay.
_CONTINUES = re.compile(
    r"^\s*(?:(?:and|then|also|please)\s+)*(?:after|in|within)\s+(?:\d+|a|an|another|half an?|"
    r"the next)\s*(?:\S+\s+)?(?:minutes?|mins?|hours?|hrs?|seconds?|moments?)\b", re.I)
# The goal acts on "it" / "them": the device is the previous intent's.
_ACTS_ON_IT = re.compile(r"^\s*(?:(?:and|then|also|please)\s+)*[a-z]+\s+(?:it|them|both)\b", re.I)


def _devices(text: str) -> Set[Tuple[str, str]]:
    """The (device noun, number) pairs the text names; "" when unnumbered."""
    out = set()
    for noun, number in _DEVICE.findall(text):
        noun = re.sub(r"s$", "", noun.lower().replace("robot vacuum cleaner", "robot vacuum"))
        noun = _ALIASES.get(noun, noun)
        out.add((noun, number or ""))
    return out


def _timed(text: str) -> bool:
    return bool(_TIMING.search(text))


def merge_coupled_intents(intents: List[AtomicIntent]) -> List[AtomicIntent]:
    """Join the GOAL_REQUEST intents about one device when one times another
    (module docstring); the result keeps the first intent's position."""
    merged: List[AtomicIntent] = []
    for intent in intents:
        target = None
        if intent.type == "GOAL_REQUEST":
            mine = _devices(intent.text)
            previous = merged[-1] if merged and merged[-1].type == "GOAL_REQUEST" else None
            for earlier in merged:
                if earlier.type != "GOAL_REQUEST":
                    continue
                shared = mine & _devices(earlier.text)
                if shared and (_timed(intent.text) or _timed(earlier.text)):
                    target = earlier
                    break
            if target is None and previous is not None and (
                    _WAIT_ONLY.match(previous.text)
                    or _CONTINUES.match(intent.text)
                    or (_ACTS_ON_IT.match(intent.text) and _timed(intent.text))):
                target = previous
        if target is None:
            merged.append(AtomicIntent(text=intent.text, type=intent.type, reason=intent.reason,
                                       qualifiers=list(intent.qualifiers)))
            continue
        target.text = f"{target.text.rstrip('. ')}. {intent.text.strip()}"
        for qualifier in list(intent.qualifiers) + ["temporal_dependency"]:
            if qualifier not in target.qualifiers:
                target.qualifiers.append(qualifier)
        target.reason = f"{target.reason}; merged with: {intent.reason}"
    return merged
