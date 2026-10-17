"""Do a goal's stated clock times agree with its relative timing?

A user may name one moment twice: "At 5:48 PM, which is 23 minutes before
washer 1 finishes, …". The structurer keeps the relative description in the
goal's `starts` / `ends` (or a termination predicate) -- what the plan executes
-- and the clock time in `stated_start_time_text` / `stated_end_time_text`.
This module checks, deterministically, that the two name the same moment.

    start_moments(...)   each goal's start, from its relative description:
                           now | after_delay      now (+ offset)
                           at_time               the stated clock time (today or
                                                 tomorrow)
                           on_trigger X          now + remaining(X) -/+ offset
                                                 ("before" / "after" X)
                           after_goal / with_goal G
                                                 G's start (+ offset)
    end_moments(...)     each goal's end: start + duration, or now +
                         remaining(X) for a termination predicate X
    check(...)           every stated start / end against its moment, as times
                         of day (a circular difference, so midnight is fine):
                           over the tolerance  -> a conflict, worded as a
                                                  question to the user
                           within it           -> a note: checked
                           moment unknown      -> a note: could not verify

`remaining` gives, per predicate id, the anchor's remaining time in seconds as
read now (None when nothing reports it). The planner reads it once, from the
property the profile names (`time_to_event`); this module does no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ....shared.models.goal_structure import GoalStructure
from .time_text import TimeTextError, span_seconds, time_of_day

DEFAULT_TOLERANCE_SECONDS = 60.0
DAY = 86400.0


@dataclass
class Moment:
    """A computed moment, and the words that describe how it was reached."""

    at: datetime
    said: str


@dataclass
class Consistency:
    """The check's outcome for one structure."""

    conflicts: List[str] = field(default_factory=list)   # questions for the user
    notes: List[str] = field(default_factory=list)


def _clock(moment: datetime) -> str:
    return moment.strftime("%I:%M %p").lstrip("0")


def _span(text: Optional[str]) -> float:
    if not text:
        return 0.0
    try:
        return span_seconds(text)
    except TimeTextError:
        return 0.0


def _seconds_of_day(text: str) -> Optional[float]:
    try:
        parts = [int(p) for p in time_of_day(text).split(":")]
    except TimeTextError:
        return None
    return parts[0] * 3600.0 + parts[1] * 60.0 + (parts[2] if len(parts) > 2 else 0)


def _resolve(text: str, now: datetime) -> Optional[datetime]:
    """A clock time as the next such moment from now (today, else tomorrow)."""
    seconds = _seconds_of_day(text)
    if seconds is None:
        return None
    moment = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=seconds)
    return moment if moment >= now - timedelta(seconds=1) else moment + timedelta(days=1)


def _apart(stated: str, moment: datetime) -> Optional[float]:
    """Seconds between a stated time of day and a moment's, around the clock."""
    seconds = _seconds_of_day(stated)
    if seconds is None:
        return None
    own = moment.hour * 3600.0 + moment.minute * 60.0 + moment.second
    gap = abs(own - seconds) % DAY
    return min(gap, DAY - gap)


def start_moments(structure: GoalStructure, now: datetime,
                  remaining: Dict[str, Optional[float]]) -> Dict[str, Optional[Moment]]:
    """Each goal's start moment, from its relative description; None where it
    cannot be known (no remaining time for its anchor, a recurrence, ...).

    Each Moment also carries the words for how it was reached ("23 minutes
    before washer 1 finishes (at 5:54 PM)"), so a conflict can be put to the
    user in their own terms. A goal timed from another goal needs that goal's
    start first, so starts are resolved recursively and memoised in `out`.
    """
    goals, predicates = structure.goals, structure.predicates
    out: Dict[str, Optional[Moment]] = {}       # gid -> its start, once resolved

    def start(gid: str, seen: Tuple[str, ...] = ()) -> Optional[Moment]:
        # Already resolved (possibly as an anchor of an earlier goal).
        if gid in out:
            return out[gid]
        # A cycle of goals timed from each other ("A after B, B after A"), or a
        # reference to a goal that is not in the request: no start to give.
        if gid in seen or gid not in goals:
            return None

        timing = goals[gid].timing
        starts = timing.starts if timing else None
        moment: Optional[Moment] = None

        # Untimed or "now": it starts at the moment of the check.
        if starts in (None, "now"):
            moment = Moment(now, "now")

        # "In 13 minutes" / "13 minutes from now": now plus the offset.
        elif starts == "after_delay":
            moment = Moment(now + timedelta(seconds=_span(timing.offset_text)),
                            f"{timing.offset_text} from now")

        # A clock time given alone: its next occurrence (today, else tomorrow).
        # Never itself checked -- only used as an anchor for a goal timed from
        # this one.
        elif starts == "at_time" and timing.time_text:
            at = _resolve(timing.time_text, now)
            moment = Moment(at, timing.time_text) if at else None

        # Due on an event ("23 minutes before washer 1 finishes", "when the
        # dryer stops"): the event happens when its anchor's remaining time
        # runs out, so it is at now + remaining; the trigger's offset then
        # moves the start before or after it. Known only for a single trigger
        # whose remaining time was read -- several triggers, or none readable,
        # leave it unknown.
        elif starts == "on_trigger":
            triggers = [pid for pid, p in predicates.items()
                        if p.role == "trigger" and gid in (p.applies_to or [])]
            if len(triggers) == 1 and remaining.get(triggers[0]) is not None:
                pred = predicates[triggers[0]]
                event = now + timedelta(seconds=remaining[triggers[0]])
                offset = _span(pred.offset_text)
                before = pred.offset_direction == "before"
                at = event - timedelta(seconds=offset) if before else \
                    event + timedelta(seconds=offset)
                what = pred.predicate_text or triggers[0]
                said = (f"{pred.offset_text} {'before' if before else 'after'} {what}"
                        if pred.offset_text else f"when {what}")
                moment = Moment(at, f"{said} (at {_clock(event)})")

        # Timed from another goal: that goal's start first (recursively), then
        # the offset after it ("8 minutes after the previous action"); with_goal
        # starts at the same moment. Unknown if the referenced start is.
        elif starts in ("after_goal", "with_goal") and timing.ref_goal:
            ref = start(timing.ref_goal, seen + (gid,))
            if ref is not None:
                offset = _span(timing.offset_text) if starts == "after_goal" else 0.0
                said = (f"{timing.offset_text} after {ref.said}" if offset
                        else f"with the goal starting {ref.said}")
                moment = Moment(ref.at + timedelta(seconds=offset), said)

        # Anything else (completes_with, a recurrence) has no single start:
        # moment stays None and a stated time on it is reported as unverifiable.
        out[gid] = moment
        return moment

    for gid in goals:
        start(gid)
    return out


def end_moments(structure: GoalStructure, now: datetime,
                remaining: Dict[str, Optional[float]],
                starts: Dict[str, Optional[Moment]]) -> Dict[str, Optional[Moment]]:
    """Each goal's end moment, from its relative description; None where it
    has none or it cannot be known."""
    out: Dict[str, Optional[Moment]] = {}
    for gid, goal in structure.goals.items():
        timing = goal.timing
        moment = None
        if timing and timing.ends_kind == "duration" and starts.get(gid) is not None:
            moment = Moment(starts[gid].at + timedelta(seconds=_span(timing.ends_text)),
                            f"{timing.ends_text} after it starts {starts[gid].said}")
        else:
            terminations = [pid for pid, p in structure.predicates.items()
                            if p.role == "termination" and gid in (p.applies_to or [])]
            if len(terminations) == 1 and remaining.get(terminations[0]) is not None:
                pred = structure.predicates[terminations[0]]
                moment = Moment(now + timedelta(seconds=remaining[terminations[0]]),
                                f"when {pred.predicate_text or terminations[0]}")
        out[gid] = moment
    return out


def check(structure: GoalStructure, now: datetime, remaining: Dict[str, Optional[float]],
          tolerance: float = DEFAULT_TOLERANCE_SECONDS) -> Consistency:
    """Every stated start / end clock time against the moment its relative
    description names."""
    result = Consistency()
    starts = start_moments(structure, now, remaining)
    ends = end_moments(structure, now, remaining, starts)
    for gid, goal in structure.goals.items():
        timing = goal.timing
        if timing is None:
            continue
        what = goal.intent_text or gid
        for kind, stated, moments in (("start", timing.stated_start_time_text, starts),
                                      ("end", timing.stated_end_time_text, ends)):
            if not stated:
                continue
            moment = moments.get(gid)
            gap = _apart(stated, moment.at) if moment is not None else None
            if gap is None:
                result.notes.append(f"{gid}: could not verify the stated {kind} time "
                                    f"{stated!r}")
            elif gap > tolerance:
                result.conflicts.append(
                    f"You asked for \"{what}\" to {kind} at {stated}, but "
                    f"{moment.said} is {_clock(moment.at)}. Which do you mean?")
            else:
                result.notes.append(f"{gid}: stated {kind} time {stated} agrees with "
                                    f"{moment.said}")
    return result
