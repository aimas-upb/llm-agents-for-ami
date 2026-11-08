"""A device the user named has to be the device that is answered about.

Asked "what brand is refrigerator 1 in the kitchen" in a home with no
refrigerator, the parser -- told to pick from the home's devices -- returns
the freezer, and the agent answers about the freezer. Asked about "dryer 1" it
may return the washer, about "dimmer light 1" the dimmer light 2. The judge
calls that unfaithful, and so would the user. This guard reads the device the
user named straight from their words (kind noun, number, room) and checks the
parser's choice against it:

    device_mentions(text)       -> [Mention(kind, class, number, room class), ...]
    guard_extraction(extraction) a state / capability extraction: a chosen
                                 `artifact_name` of another kind or number is
                                 cleared and the question marked as naming a
                                 device (`device_reference` "name"), which the
                                 answer path turns into "I can't find the ...";
                                 `device_class` is set to the named kind, so a
                                 lookup by description finds that kind or nothing
    guard_goal(goal)             a goal whose `artifact_name` is another device
                                 than the one named: the name is cleared, the
                                 kind set to the one named and the goal left
                                 incomplete with `device_reference` "name", so
                                 the resolver looks that kind up and takes the
                                 device whose title carries the user's number
                                 (`goal_resolution.apply_goal_resolution`), or
                                 asks about it when there is none -- never does
                                 it on another device

The kinds are the benchmark homes' device nouns with their HomeOnt class and
the words their titles carry (`utility_room_laundry_dryer_1`). A text that
names no device kind, or names it without a number ("the bathroom light"),
is left to the parser: it describes, it does not name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ....shared.models.goal_structure import GoalSpec

# (noun pattern, HomeOnt class, the words in the device's title). Longer
# patterns first, so "dimmer light" wins over "light", "dehumidifier" over
# "humidifier", "air quality sensor" over "sensor".
KINDS: Tuple[Tuple[str, str, str], ...] = (
    (r"robot(?:ic)? vacuums?(?: cleaners?)?|vacuum cleaners?|rvcs?", "homeont:Rvc", "rvc"),
    (r"dimm(?:er|able) lights?|dimmers?", "homeont:DimmableLight", "dimmable_light"),
    (r"on[/ -]?off lights?|lights?|lamps?", "homeont:OnOffLight", "on_off_light"),
    (r"washing machines?|washers?", "homeont:LaundryWasher", "laundry_washer"),
    (r"(?:tumble |laundry )?dryers?", "homeont:LaundryDryer", "laundry_dryer"),
    (r"dish ?washers?", "homeont:Dishwasher", "dishwasher"),
    (r"dehumidifiers?", "homeont:Dehumidifier", "dehumidifier"),
    (r"humidifiers?", "homeont:Humidifier", "humidifier"),
    (r"air purifiers?|purifiers?", "homeont:AirPurifier", "air_purifier"),
    (r"air conditioners?|acs?|air cons?", "homeont:AirConditioner", "air_conditioner"),
    (r"heat pumps?", "homeont:HeatPump", "heat_pump"),
    (r"refrigerators?|fridges?", "homeont:Refrigerator", "refrigerator"),
    (r"freezers?", "homeont:Freezer", "freezer"),
    (r"tvs?|televisions?", "homeont:Tv", "tv"),
    (r"fans?", "homeont:Fan", "fan"),
    (r"blinds?|shades?|curtains?|window coverings?(?: controllers?)?",
     "homeont:WindowCoveringController", "window_covering_controller"),
    (r"temperature sensors?|thermometers?", "homeont:TemperatureSensor", "temperature_sensor"),
    (r"humidity sensors?", "homeont:HumiditySensor", "humidity_sensor"),
    (r"air quality sensors?", "homeont:AirQualitySensor", "air_quality_sensor"),
    (r"(?:light|illuminance) sensors?", "homeont:IlluminanceSensor", "illuminance_sensor"),
    (r"occupancy sensors?|motion sensors?", "homeont:OccupancySensor", "occupancy_sensor"),
)
ROOMS: Dict[str, str] = {
    "utility room": "homeont:UtilityRoom", "laundry room": "homeont:UtilityRoom",
    "bathroom": "homeont:Bathroom", "kitchen": "homeont:Kitchen", "living room": "homeont:LivingRoom",
    "bedroom": "homeont:Bedroom", "kids room": "homeont:KidsRoom", "kids' room": "homeont:KidsRoom",
    "office": "homeont:Office", "study room": "homeont:StudyRoom", "study": "homeont:StudyRoom",
    "dining room": "homeont:DiningRoom", "hallway": "homeont:Hallway", "garage": "homeont:Garage",
    "balcony": "homeont:Balcony", "corridor": "homeont:Corridor",
}
_ROOM_WORDS = "|".join(sorted((re.escape(r) for r in ROOMS), key=len, reverse=True))
_MENTION = re.compile(
    r"(?:(?:the|our|my)\s+)?(?:(?P<room_before>" + _ROOM_WORDS + r")\s+)?"
    r"(?P<kind>" + "|".join(p for p, _, _ in KINDS) + r")\s+(?P<number>[1-9])\b"
    # a device number, not a setting or a time: "fan 30%", "dehumidifier 12 minutes after"
    r"(?!\s*(?:%|percent|°|degrees?|minutes?|mins?|hours?|hrs?|seconds?|secs?|[ap]\.?m\b|:|\.\d))"
    r"(?:\s+(?:in|of|from)\s+the\s+(?P<room_after>" + _ROOM_WORDS + r"))?", re.I)


@dataclass
class Mention:
    kind: str                  # the user's words for the kind ("dryer")
    cls: str                   # its HomeOnt class
    title_words: str           # what the device's title carries
    number: str
    room: Optional[str]        # HomeOnt room class, when the words give one
    text: str                  # the words as written ("dryer 1 in the utility room")


def device_mentions(text: str) -> List[Mention]:
    """Every numbered device the text names, in order, each once."""
    out: List[Mention] = []
    for m in _MENTION.finditer(text or ""):
        kind = m.group("kind")
        cls, words = next((c, w) for p, c, w in KINDS if re.fullmatch(p, kind, re.I))
        room = (m.group("room_after") or m.group("room_before") or "").lower()
        text_words = re.sub(r"^(?:the|our|my)\s+", "", m.group(0).strip(), flags=re.I)
        mention = Mention(kind=kind.lower(), cls=cls, title_words=words, number=m.group("number"),
                          room=ROOMS.get(room), text=text_words)
        if not any(o.cls == mention.cls and o.number == mention.number for o in out):
            out.append(mention)
    return out


def device_mention(text: str) -> Optional[Mention]:
    """The first numbered device the text names, or None."""
    mentions = device_mentions(text)
    return mentions[0] if mentions else None


def title_matches(title: str, mention: Mention) -> bool:
    """Whether a device title names the kind and number the user did."""
    t = (title or "").lower()
    return bool(re.search(rf"(?:^|_){mention.title_words}_{mention.number}$", t))


def guard_extraction(extraction: Dict[str, Any]) -> Optional[str]:
    """Check a state / capability extraction against the device the question
    names (module docstring); returns a note when it changed something."""
    mentions = device_mentions(extraction.get("text_intent") or "")
    if not mentions:
        return None
    name = extraction.get("artifact_name")
    if name and any(title_matches(name, m) for m in mentions):
        return None
    mention = mentions[0]
    if len(mentions) > 1 and not name:
        return None                 # two devices asked about; nothing to check against
    note = None
    if name:
        # Another device was picked for the name: not read in its place.
        extraction["artifact_name"] = None
        extraction["device_reference"] = "name"
        note = f"{name!r} is not the {mention.text} the question names"
    if extraction.get("device_class") != mention.cls:
        # The kind the user named, whatever kind the parser picked (a lookup
        # by description then finds that kind, or nothing).
        note = note or (f"the question is about {mention.text}, not a "
                        f"{extraction.get('device_class') or 'device of another kind'}")
        extraction["device_class"] = mention.cls
    if mention.room and not extraction.get("location_class"):
        extraction["location_class"] = mention.room
    return note


def guard_goal(goal: GoalSpec) -> Optional[str]:
    """Check a goal's device against the one its words name; a note when the
    name was cleared and the goal marked as asking for a device not there."""
    # The goal's words may hold the trigger's device too ("when dryer 1
    # finishes, turn on light 1"): the name has to fit one of the devices named.
    mentions = device_mentions(goal.intent_text or "")
    if not mentions or not goal.artifact_name or any(title_matches(goal.artifact_name, m)
                                                     for m in mentions):
        return None
    mention = mentions[0]
    was = goal.artifact_name
    goal.artifact_name = None
    goal.device_reference = "name"
    goal.goal_specificity = "incomplete"
    goal.artifact_class = mention.cls
    if mention.room and not goal.location_class:
        goal.location_class = mention.room
    return f"{was!r} is not the {mention.text} the request names"
