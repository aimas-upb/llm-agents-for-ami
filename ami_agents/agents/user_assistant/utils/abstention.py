"""Abstention for the constrained decodings: a device the home lacks is not replaced
(`decoding: iri_trie_v2`, `decoding: genre_v2`).

A trie of the home's IRIs offers a value for every slot, so a request for a device the
room does not have is answered with the nearest device it does have: the humidifier of
a request becomes the dehumidifier, the air conditioner the heater, and the home runs
that instead of reporting the missing one. In the 400 ECAI episodes of the fep runs
(`homebench.ecai.M3.pv-trie`, `pv-genre`) that was the largest loss on the infeasible
requests: 36 of the 39 `artifact_class` values the trie replaced in the infeasible
episodes of hb1 were a real device kind the room lacks (a `Curtain` in a room with
none), and only 2 of the 21 replaced in the feasible ones. The free reply had named the
device correctly; the trie overruled it.

A reply already has a way to say it: `unmatched_device`, the request's words for the
device, with `artifact_class` and `artifact_name` left null (the structuring prompt asks
for it, and `goal_resolution.unmatched_question` turns it into "I can't find the X in
the Y"). This module decides when the free reply is such a case and rewrites the goal
object into it:

    free reply    "location_class": "homeont:StudyRoom", "artifact_class":
                  "homeont:AirConditioner", "artifact_name": "study_room_ac", ...
    the study room has no air conditioner, and `homeont:AirConditioner` is a device
    family of the ontology (`homeont.ttl`), so the device is missing, not misspelt
    rewritten     "artifact_class": null, "artifact_name": null, ...,
                  "unmatched_device": "air conditioner"

The rule is narrow on purpose. It fires only on `artifact_class`, only when the value is
a device class of the ontology (a garbled value is a spelling error for the trie to
repair), and only when neither it nor another class of its family (`DimmableLight` in a
room with a `ColorLight`: the last word of the class) is in the room. A missing action
(`unmatched_action`) is not treated: where the free `affordance_name` and
`affordance_class` were both outside the device's actions, 3 of the 15 feasible hb1
structures passed after being replaced, against at most 2 infeasible ones
(`unexist_attribute`) that abstaining would turn into a pass, so replacing by the trie
stays the better bet there.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from .goal_context import Vocabulary
from .iri_trie import (
    SLOT_VALUE, Row, _object_end, _object_start,
)

ONTOLOGY = Path(__file__).resolve().parents[3] / "shared" / "ontologies" / "homeont.ttl"

# A device family of the ontology: a class under one of SAREF's five device classes
# (rooms, properties and commands are not devices a request can name).
_CLASS = re.compile(r"^:(\w+) a owl:Class\b", re.M)
_DEVICE_PARENT = re.compile(
    r"rdfs:subClassOf\s+saref:(?:Actuator|Appliance|HVAC|Meter|Sensor)\b")
_CAMEL_WORD = re.compile(r"[A-Z][a-z0-9]*")

# The slots a goal that names a missing device leaves null (the prompt: "leave
# artifact_name and artifact_class null"; the action and its parameter belong to a
# device that is not there).
NULLED_SLOTS = ("artifact_class", "artifact_name", "device_reference", "affordance_class",
                "parent_class", "affordance_name", "parameter_name")
UNMATCHED_DEVICE = "unmatched_device"
UNMATCHED_ACTION = "unmatched_action"


def class_key(value: Optional[str]) -> str:
    """A class name without prefix, case, spacing or a plural `s`: `homeont:AirPurifier`,
    `air purifiers` and `AirPurifier` are one key."""
    local = (value or "").rpartition(":")[2]
    key = re.sub(r"[^a-z0-9]", "", local.lower())
    return key[:-1] if key.endswith("s") else key


def class_family(value: Optional[str]) -> str:
    """The last CamelCase word of a class, lower case: `OnOffLight`, `DimmableLight` and
    `ColorLight` are `light`; `AirPurifier` is `purifier`. Empty for a value with none."""
    words = _CAMEL_WORD.findall((value or "").rpartition(":")[2])
    return words[-1].lower() if words else ""


def class_words(value: Optional[str]) -> str:
    """A class as the words a user says: `homeont:AirConditioner` -> `air conditioner`."""
    local = (value or "").rpartition(":")[2]
    return " ".join(_CAMEL_WORD.findall(local)).lower() or local.lower()


@lru_cache(maxsize=1)
def device_classes() -> FrozenSet[str]:
    """The keys (`class_key`) of the ontology's device families, read once from
    `homeont.ttl`."""
    keys = set()
    for block in re.split(r"\n\s*\n", ONTOLOGY.read_text(encoding="utf-8")):
        named = _CLASS.search(block)
        if named and _DEVICE_PARENT.search(block):
            keys.add(class_key(named.group(1)))
    return frozenset(keys)


def resolve_room(value: Optional[str], vocab: Vocabulary) -> Optional[str]:
    """The home's room class a written `location_class` means (`Store Room`,
    `homeont:StoreRoom`), or None when it names none of them."""
    wanted = class_key(value)
    if not wanted:
        return None
    return next((room for room in vocab.room_classes if class_key(room) == wanted), None)


def room_artifact_classes(rows: List[Row], room: str) -> List[str]:
    """The device classes the home has in ``room`` (a device is in its own room and in
    every room that holds that room)."""
    found: List[str] = []
    for row in rows:
        if room in row.get("location_class", ()):
            found.extend(c for c in row.get("artifact_class", ()) if c not in found)
    return found


def device_absent(free_class: Optional[str], free_room: Optional[str],
                  rows: List[Row], vocab: Vocabulary) -> bool:
    """Whether the free reply names a device kind of the ontology that the room it names
    does not have (module docstring). False whenever it cannot be told: no room the home
    knows, or a class that is not a device family."""
    room = resolve_room(free_room, vocab)
    if room is None or class_key(free_class) not in device_classes():
        return False
    present = room_artifact_classes(rows, room)
    if class_key(free_class) in {class_key(c) for c in present}:
        return False
    family = class_family(free_class)
    return not (family and family in {class_family(c) for c in present})


def _slot_pattern(slot: str) -> "re.Pattern[str]":
    """A slot written with a string or null value, key and colon in group 1."""
    return re.compile(r'("%s"\s*:\s*)(?:null|"[^"]*")' % slot)


def abstain_object(obj: str, words: str) -> str:
    """The goal object ``obj`` (its JSON text, braces included) as a goal about a missing
    device: the IRI slots null, `unmatched_action` null, `unmatched_device` the
    ``words``. A slot the object does not have is left out; `unmatched_device` is added
    when absent."""
    for slot in (*NULLED_SLOTS, UNMATCHED_ACTION):
        obj = _slot_pattern(slot).sub(r"\1null", obj)
    named = json.dumps(words)
    pattern = _slot_pattern(UNMATCHED_DEVICE)
    if pattern.search(obj):
        return pattern.sub(lambda m: m.group(1) + named, obj, count=1)
    close = obj.rindex("}")
    body = obj[:close].rstrip()
    return f'{body},\n  "{UNMATCHED_DEVICE}": {named}\n{obj[close:]}'


def goal_objects(text: str, pos: int = 0) -> List[Tuple[int, int, Dict[str, str]]]:
    """The objects of ``text`` from ``pos`` that hold a constrained slot, in order, as
    (start, index of the closing brace, the slot values the object writes)."""
    objects: Dict[int, Tuple[int, Dict[str, str]]] = {}
    for match in SLOT_VALUE.finditer(text, pos):
        start = _object_start(text, match.start())
        end, slots = objects.setdefault(start, (_object_end(text, start), {}))
        slots.setdefault(match.group(1), match.group(2))
    return [(start, end, slots) for start, (end, slots) in sorted(objects.items())]


def absent_goal(slots: Dict[str, str], rows: List[Row],
                vocab: Vocabulary) -> Optional[str]:
    """The user's words for the missing device when the goal object whose ``slots`` are
    given names one (`device_absent`), else None."""
    free_class = slots.get("artifact_class")
    if device_absent(free_class, slots.get("location_class"), rows, vocab):
        return class_words(free_class)
    return None


def overlay(free: str, decoded: str, rows: List[Row], vocab: Vocabulary,
            pos: int = 0) -> Tuple[str, List[Dict[str, Any]]]:
    """``decoded`` with the abstention the ``free`` reply of the same thinking calls for:
    each goal object of the free reply that names a missing device turns the object at
    the same position of the decoded reply into a goal about it (`abstain_object`).

    Why a free reply to judge by: a decoding under a mask replaces what the model named,
    so the model's own words are the only place the missing device can be read. The two
    replies are paired by position, which holds while they have the same number of goal
    objects; when they do not, nothing is rewritten. Returns the text and, per object
    rewritten, `{"index", "free_class", "room", "words"}`."""
    free_objects = goal_objects(free, pos)
    decoded_objects = goal_objects(decoded)
    if not free_objects or len(free_objects) != len(decoded_objects):
        return decoded, []
    verdicts = [(i, absent_goal(slots, rows, vocab))
                for i, (_, _, slots) in enumerate(free_objects)]
    done: List[Dict[str, Any]] = []
    for index, words in reversed([v for v in verdicts if v[1]]):
        start, end, _ = decoded_objects[index]
        if end >= len(decoded):                # cut off before its closing brace
            continue
        decoded = (decoded[:start] + abstain_object(decoded[start:end + 1], words)
                   + decoded[end + 1:])
        slots = free_objects[index][2]
        done.append({"index": index, "free_class": slots.get("artifact_class"),
                     "room": slots.get("location_class"), "words": words})
    return decoded, list(reversed(done))
