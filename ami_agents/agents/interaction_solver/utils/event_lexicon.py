"""Complete an appliance-event predicate the structurer left empty.

"When washer 1 in the utility room finishes" names an event every home knows:
the appliance's `operationalState` becoming Stopped (Matter
OperationalStateEnum: 0 stopped, 1 running, 2 paused). The structurer resolves
it most of the time, but in about a third of the recorded cases the predicate
arrives with its text alone -- no device, no property, no value -- and the
dependency planner cannot read it. This lexicon fills those fields
deterministically from the words:

    appliance word   washer / washing machine, dryer, dishwasher, robot vacuum
                     -> the HomeOnt class
    room words       "in the utility room"   -> the HomeOnt room class
    event words      finishes / is finished / done / completes / stops / cycle
                     ends -> operationalState == 0; starts / is running -> == 1
    the instance     "washer 1" -> the device whose title ends in `_1`, among
                     the devices of that class (and room) in the home

A predicate the structurer did resolve is only normalised: an event it worded
as finished / done / stopped but valued as 1 (running) gets the value 0.

`complete_event_predicate(graph, pred)` edits the predicate in place and
returns a note of what it inferred, or None when it changed nothing.
`event_clause(text)` finds the appliance-event clause of a request ("30
minutes after the dryer 1 in the utility room finishes, turn on ...") for a
predicate that arrived without any text.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from rdflib import Graph

from ....shared.models.goal_structure import PredicateSpec
from ....shared.utils.namespaces import TD
from .specific_capability_context import find_artifacts

# Appliances with an Operational State cluster, as the benchmark names them.
APPLIANCES: Tuple[Tuple[str, str], ...] = (
    (r"washing machines?|washers?", "homeont:LaundryWasher"),
    (r"(?:tumble |laundry )?dryers?", "homeont:LaundryDryer"),
    (r"dish ?washers?", "homeont:Dishwasher"),
    (r"robot(?:ic)? vacuums?(?: cleaners?)?|vacuum cleaners?|rvcs?|roombas?", "homeont:Rvc"),
)

ROOMS: Dict[str, str] = {
    "utility room": "homeont:UtilityRoom", "laundry room": "homeont:UtilityRoom",
    "bathroom": "homeont:Bathroom", "kitchen": "homeont:Kitchen",
    "living room": "homeont:LivingRoom", "lounge": "homeont:LivingRoom",
    "bedroom": "homeont:Bedroom", "kids room": "homeont:KidsRoom",
    "kids' room": "homeont:KidsRoom", "office": "homeont:Office",
    "study room": "homeont:StudyRoom", "study": "homeont:StudyRoom",
    "dining room": "homeont:DiningRoom", "hallway": "homeont:Hallway",
    "garage": "homeont:Garage",
}

# Event words -> (operationalState value, the value's label).
STOPPED = ("0", "finished")
RUNNING = ("1", "running")
_STOPPED_WORDS = re.compile(
    r"\b(?:finish(?:es|ed|ing)?|(?:is|are|has|have)\s+(?:been\s+)?(?:finished|done|complete[d]?|"
    r"stopped)|done|complete[sd]?|stop(?:s|ped)?|ends?|(?:cycle|run)\s+(?:is\s+)?(?:over|ends?|"
    r"complete[sd]?))\b", re.I)
_RUNNING_WORDS = re.compile(r"\b(?:starts?|started|begins?|began|(?:is|are)\s+running)\b", re.I)
_INDEX = re.compile(r"\b(\d+)\b")

PROPERTY = "operationalState"


def _appliance(text: str) -> Optional[Tuple[str, str]]:
    """(class, the matched words) of the first appliance the text names."""
    best = None
    for pattern, cls in APPLIANCES:
        m = re.search(rf"\b(?:{pattern})\b", text, re.I)
        if m and (best is None or m.start() < best[2]):
            best = (cls, m.group(0), m.start())
    return (best[0], best[1]) if best else None


def _room(text: str) -> Optional[str]:
    low = text.lower()
    hits = [(low.find(words), cls) for words, cls in ROOMS.items() if words in low]
    if not hits:
        return None
    # The longest phrase at the earliest position ("living room" over "room").
    hits.sort(key=lambda h: (h[0], -len([w for w in ROOMS if ROOMS[w] == h[1]])))
    return hits[0][1]


def _event(text: str) -> Optional[Tuple[str, str]]:
    if _STOPPED_WORDS.search(text):
        return STOPPED
    if _RUNNING_WORDS.search(text):
        return RUNNING
    return None


def _device(graph: Graph, cls: str, room: Optional[str], index: Optional[str]) -> Optional[str]:
    """The title of the one device of `cls` (in `room`) numbered `index`."""
    candidates = find_artifacts(graph, None, cls, room)
    if not candidates and room:
        candidates = find_artifacts(graph, None, cls, None)   # the room may be misnamed
    titles = [str(graph.value(a, TD.title)) for a in candidates if graph.value(a, TD.title)]
    if index:
        titles = [t for t in titles if re.search(rf"(?:^|[^0-9]){re.escape(index)}$", t)]
    return titles[0] if len(titles) == 1 else None


def complete_event_predicate(graph: Graph, pred: PredicateSpec) -> Optional[str]:
    """Fill an appliance-event predicate from its words (module docstring)."""
    text = pred.predicate_text or ""
    if not text or pred.predicate_subject not in (None, "device_property"):
        return None
    event = _event(text)
    if event is None:
        return None
    if pred.artifact_name and pred.property_name and pred.target_value_determined is not None:
        # Resolved by the structurer; only a plainly wrong value is corrected.
        label = (pred.target_value_text or "").strip().lower()
        if (pred.property_name == PROPERTY and event is STOPPED
                and pred.target_value_determined not in ("0", 0)
                and (not label or _STOPPED_WORDS.search(label) or label in ("stopped", "off"))):
            pred.target_value_determined = "0"
            return (f"{pred.predicate_text!r} read as operationalState == 0 (stopped): "
                    f"the structure valued it {pred.target_value_text!r} as "
                    f"{pred.target_value_determined!r}")
        return None

    appliance = _appliance(text)
    if appliance is None:
        return None
    cls, words = appliance
    filled: List[str] = []
    if not pred.artifact_class:
        pred.artifact_class = cls
        filled.append(cls)
    room = pred.location_class or _room(text)
    if room and not pred.location_class:
        pred.location_class = room
    if not pred.artifact_name:
        index = _INDEX.search(text[text.lower().find(words.lower()) + len(words):])
        title = _device(graph, pred.artifact_class, room, index.group(1) if index else None)
        if title:
            pred.artifact_name = title
            filled.append(title)
    if not pred.property_name:
        pred.property_name = PROPERTY
    if not pred.comparison:
        pred.comparison = "=="
    if pred.target_value_determined is None:
        pred.target_value_determined = event[0]
        pred.target_value_text = pred.target_value_text or event[1]
        filled.append(f"{PROPERTY} == {event[0]}")
    pred.predicate_subject = pred.predicate_subject or "device_property"
    pred.predicate_specificity = pred.predicate_specificity or "explicit"
    if not filled:
        return None
    return f"{pred.predicate_text!r} read as {', '.join(filled)} (appliance-event lexicon)"


_APPLIANCE_WORDS = "|".join(pattern for pattern, _ in APPLIANCES)
_ROOM_WORDS = "|".join(sorted((re.escape(w) for w in ROOMS), key=len, reverse=True))
_CLAUSE = re.compile(
    rf"(?:the\s+)?(?:{_APPLIANCE_WORDS})(?:\s+\d+)?(?:\s+in\s+the\s+(?:{_ROOM_WORDS}))?"
    rf"(?:'s)?(?:\s+(?:cycle|run))?\s+"
    rf"(?:finish(?:es|ed|ing)?|(?:is|are|has|have)\s+(?:been\s+)?(?:finished|done|complete[d]?|"
    rf"stopped|running)|(?:is\s+)?done|complete[sd]?|stop(?:s|ped)?|ends?|starts?|started|begins?)"
    rf"(?:\s+(?:drying|washing|cleaning|running|its\s+cycle|the\s+cycle))?", re.I)


def event_clause(text: str) -> Optional[str]:
    """The appliance-event clause in `text`, or None (several clauses: the first)."""
    m = _CLAUSE.search(text or "")
    return m.group(0).strip() if m else None
