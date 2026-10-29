"""Find the room, device and attribute a HomeBench request clause names.

The gold of an infeasible part is the bare token ``error_input``: it says nothing
about *which* room and device were asked for. The converter needs that to write
``eval.goals`` (``room_id``, ``device_id``, ``reason``), so it recovers it from the
request text with a keyword heuristic, the same kind the HomeBench analysis used
(plan 2.2, point 7). The heuristic is validated against the *feasible* parts, whose
room and device the gold does give: the converter counts how often the two
agree, and the conversion manifest records the result.

Flow, on one request::

    "Set the humidifier intensity to 90 in the guest bedroom, turn off the light
     in the study room."
    split_clauses -> ["Set the humidifier intensity to 90 in the guest bedroom",
                      "turn off the light in the study room"]
    locate(clause 0) -> Target(room="guest_bedroom", device="humidifier",
                               attribute="intensity")

Parts of a multi request line up with clauses by position when the counts agree;
otherwise ``align`` removes the clauses that match the feasible gold parts and
hands the rest to the infeasible parts in order, or gives up (``None``) and the
part is recorded as unresolved. Nothing here guesses silently: an unlocatable part
keeps ``room``/``device`` as ``None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# -- vocabularies -------------------------------------------------------------------

#: Room key -> regex of the phrases the dataset uses (``ding_room`` is "dining room").
ROOM_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("master_bedroom", r"master[ _-]?bedroom"),
    ("guest_bedroom", r"guest[ _-]?bedroom"),
    ("living_room", r"living[ _-]?room"),
    ("ding_room", r"d(?:in)?ing[ _-]?room"),
    ("study_room", r"study[ _-]?room"),
    ("store_room", r"stor(?:e|age)[ _-]?room"),
    ("kitchen", r"kitchen"),
    ("bathroom", r"bathroom"),
    ("foyer", r"foyer"),
    ("corridor", r"corridor"),
    ("balcony", r"balcony"),
    ("garage", r"garage"),
)

#: Device key -> regex, most specific first so ``water heater`` wins over ``heater``
#: and ``garage door`` is masked before the room ``garage`` is looked for.
DEVICE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("water_heater", r"\bwater[ _-]?heaters?\b"),
    ("garage_door", r"\bgarage[ _-]?doors?\b"),
    ("vacuum_robot", r"\b(?:vacuum(?:[ _-]?(?:robot|cleaner))?|robot[ _-]?vacuum)\b"),
    ("media_player", r"\b(?:(?:media|music)[ _-]?players?|media)\b"),
    ("air_conditioner", r"\b(?:air[ _-]?condition(?:er|ing)s?|a/c|ac)\b"),
    ("air_purifiers", r"\bair[ _-]?purifiers?\b"),
    ("dehumidifiers", r"\bdehumidifiers?\b"),
    ("humidifier", r"\bhumidifiers?\b"),
    ("aromatherapy", r"\b(?:aroma[ _-]?therap(?:y|ies)|diffusers?)\b"),
    ("heating", r"\b(?:heating|heaters?)\b"),
    ("blinds", r"\bblinds?\b"),
    ("curtain", r"\b(?:curtains?|drapes?)\b"),
    ("trash", r"\b(?:trash|garbage|bin)\b"),
    # ``fan speed`` / ``fan only mode`` name an attribute, not the fan device.
    ("fan", r"\bfans?\b(?!\s+(?:speed|only|mode))"),
    ("light", r"\b(?:lights?|lamps?|bulbs?)\b"),
)

#: Attribute key -> regex, checked in order (``fan speed`` before ``speed``).
ATTRIBUTE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("brightness", r"\bbrightness\b"),
    ("color", r"\bcolou?rs?\b"),
    ("fan_speed", r"\bfan[ _-]?speed\b"),
    ("swing", r"\bswing\b"),
    ("speed", r"\bspeed\b"),
    ("temperature", r"\btemp(?:erature)?s?\b|\bdegrees\b"),
    ("intensity", r"\bintensity\b"),
    ("interval", r"\binterval\b"),
    ("degree", r"\bdegree\b"),
    ("volume", r"\bvolume\b"),
    ("mode", r"\bmode\b"),
)

_ROOMS = [(key, re.compile(pat, re.I)) for key, pat in ROOM_PATTERNS]
_DEVICES = [(key, re.compile(pat, re.I)) for key, pat in DEVICE_PATTERNS]
_ATTRS = [(key, re.compile(pat, re.I)) for key, pat in ATTRIBUTE_PATTERNS]
_FAN_SPEED = re.compile(r"\bfan[ _-]?speed\b", re.I)
_CONTINUATION = re.compile(r"^(?:(?:by|to|of|for|on|respectively)\b|\d)", re.I)
_PREFIX = re.compile(r"^(?:in|on|at|inside|within)\b", re.I)
_SPLIT = re.compile(r"\s*(?:,\s*and\s+|;\s*|,\s*|\s+and\s+)", re.I)


# -- the located target -------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """What one clause names: any field may be ``None`` when not found."""

    room: Optional[str]
    device: Optional[str]
    attribute: Optional[str]


# -- single clause -----------------------------------------------------------------


def _mask(text: str, span: Tuple[int, int]) -> str:
    """Blank out ``span`` so later patterns cannot match the same words twice."""
    return text[: span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def locate(clause: str) -> Target:
    """Return the room, device and attribute named by one request clause.

    The device is found first and its words are masked, so ``garage door`` is not
    also read as the room ``garage`` and ``fan speed of the air purifiers`` is read
    as an attribute of the purifier, not the fan device. The vacuum robot has no
    room, so its room is always ``None``.
    """
    text = clause
    # 1. Device: first pattern (in specificity order) that matches; mask its words.
    device = None
    for key, pattern in _DEVICES:
        found = pattern.search(text)
        if found:
            device = key
            text = _mask(text, found.span())
            break
    # 2. A bare "fan speed" with no other device names the fan itself.
    if device is None and _FAN_SPEED.search(clause):
        device = "fan"
    # 3. Room: the earliest room phrase in the remaining text (not for the vacuum).
    room = None
    if device != "vacuum_robot":
        hits = [(m.start(), key) for key, p in _ROOMS if (m := p.search(text))]
        room = min(hits)[1] if hits else None
    # 4. Attribute keyword; a curtain with a number is its ``degree``.
    attribute = next((k for k, p in _ATTRS if p.search(clause)), None)
    if attribute is None and device == "curtain" and re.search(r"\d", clause):
        attribute = "degree"
    return Target(room, device, attribute)


# -- clauses of a whole request ---------------------------------------------------------


def split_clauses(query: str) -> List[str]:
    """Split a request into clauses on commas, semicolons and ``and``.

    A chunk is a clause of its own when it names a device or an attribute and does
    not read as a continuation ("by 67 degrees ...", "to 70 in the kitchen"). A chunk
    that names only a room ("In the corridor") is a prefix of the next chunk and is
    merged forward; anything else ("then decrease it by 41%", "foyer") belongs to the
    previous clause and is merged back. Many requests name no device at all ("Set the
    brightness to 100 in the balcony"), which is why an attribute alone starts a
    clause. The result is a best effort: ``align`` checks it against the gold.
    """
    chunks = [c for c in _SPLIT.split(query.strip().rstrip(".")) if c]
    clauses: List[str] = []
    carry = ""
    for chunk in chunks:
        target = locate(chunk)
        named = target.device is not None or target.attribute is not None
        # 1. "In the corridor": a room-only chunk led by a preposition is the prefix
        #    of the next clause; a bare room name ("foyer") continues the previous.
        if not named and target.room is not None and _PREFIX.match(chunk):
            carry = f"{carry} {chunk}".strip()
            continue
        # 2. A continuation or an unlocatable chunk glues onto the previous clause.
        if (not named or _CONTINUATION.match(chunk)) and clauses and not carry:
            clauses[-1] = f"{clauses[-1]}, {chunk}"
            continue
        clauses.append(f"{carry} {chunk}".strip())
        carry = ""
    # 3. A dangling room prefix with nothing after it stays attached to the end.
    if carry:
        if clauses:
            clauses[-1] = f"{clauses[-1]}, {carry}"
        else:
            clauses.append(carry)
    return clauses


def align(
    clauses: Sequence[str],
    feasible: Sequence[Optional[Tuple[Optional[str], str]]],
) -> Tuple[Optional[List[Target]], str]:
    """Pair every gold part with a clause; return ``(targets, method)``.

    ``feasible[i]`` is the gold's ``(room, device)`` for part ``i`` or ``None`` for
    an infeasible part. ``targets[i]`` is the located target of part ``i``.
    Method is ``positional`` (same count, and each feasible part's clause locates
    to the gold's room and device), ``elimination`` (clauses that match a feasible
    part are removed, the rest go to the infeasible parts in order), or
    ``unresolved`` (``targets`` is ``None``).
    """
    located = [locate(c) for c in clauses]

    def matches(target: Target, gold: Tuple[Optional[str], str]) -> bool:
        """True when a located clause names the gold part's room and device."""
        return target.device == gold[1] and (gold[0] is None or target.room == gold[0])

    # 1. Positional: counts agree and every feasible part is confirmed by its clause.
    if len(located) == len(feasible) and all(
        gold is None or matches(target, gold)
        for target, gold in zip(located, feasible)
    ):
        return list(located), "positional"
    # 2. Elimination: drop one clause per feasible part, keep the rest in order.
    remaining = list(range(len(located)))
    for gold in feasible:
        if gold is None:
            continue
        hit = next((i for i in remaining if matches(located[i], gold)), None)
        if hit is None:
            return None, "unresolved"
        remaining.remove(hit)
    bare = iter(remaining)
    if len(remaining) != sum(g is None for g in feasible):
        return None, "unresolved"
    # 3. Rebuild the per-part list: feasible parts echo the gold, others take a clause.
    targets = [
        Target(g[0], g[1], None) if g is not None else located[next(bare)]
        for g in feasible
    ]
    return targets, "elimination"
