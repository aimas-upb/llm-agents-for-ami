"""Do a goal's stated clock times agree with its relative timing?

A user may name one moment twice: "At 5:48 PM, which is 23 minutes before
washer 1 finishes, …". The structurer keeps the relative description in the
goal's `starts` / `ends` (or a termination predicate) -- what the plan executes
-- and the clock time in `stated_start_time_text` / `stated_end_time_text`.
This module checks, deterministically, that the two name the same moment.

    start_moments(...)   each goal's start, from its relative description:
                           completes_with X, no start
                                                 now + remaining(X) - its run
                                                 (the start the plan will choose)
                           now | after_delay      now (+ offset)
                           at_time               the stated clock time (today or
                                                 tomorrow)
                           on_trigger X          now + remaining(X) -/+ offset
                                                 ("before" / "after" X)
                           after_goal / with_goal G
                                                 G's start (+ offset)
    end_moments(...)     each goal's end: when the event it completes with
                         happens (now + remaining(X)) or the goal it completes
                         with ends; start + duration; or now + remaining(X) for
                         a termination predicate X
    check(...)           every stated start / end against its moment, as times
                         of day (a circular difference, so midnight is fine):
                           over the tolerance  -> a conflict, worded as a
                                                  question to the user
                           within it           -> a note: checked
                           moment unknown      -> a note: could not verify

    check_completion(...)
                         every goal that completes with a goal or event: can
                         it, given how long it runs and how long the event is
                         away? Anything that cannot be known, or does not fit,
                         is a question for the user -- the plan never guesses a
                         start for a "finish together" request:
                           its run length unknown       -> a question
                           the event's time left unknown -> a question
                           no start, run > time left    -> a question
                           a start, start + run off the
                             event by over the tolerance -> a question
                           otherwise                     -> a note

    triggered_after_timed(...)
                         each goal due on a single trigger ("pause dryer 1 when
                         washer 1 finishes") whose device the request also
                         times by the clock ("start dryer 1 at 18:14")
    check_triggers(...)  each of those: a trigger that happens before the timed
                         goals' start would act on the device before the
                         request has started it -- a question to the user

`remaining` gives, per predicate id, the anchor's remaining time in seconds as
read now (None when nothing reports it). `durations` gives, per goal id, how
long the goal's run takes, in seconds (None when nothing reports it -- a
cycle's countdown is empty while the device is stopped). The planner reads both
once, from the properties the profile names (`time_to_event`,
`completion_timing`); this module does no I/O.

Example -- "start dryer 1 at 18:14, pause it when washer 1 finishes", at 17:00
with the washer's countdown at 30 minutes: G4 (pause) is due at 17:30, before
G1 (start) at 18:14, so `check_triggers` asks whether the pause is still wanted
-- washer 1 will have finished before dryer 1 starts.

Example -- "start dryer 1 now so it and dishwasher 1 finish together by 11:39",
at 10:25 with the dishwasher's countdown at 89 minutes: G3 (start the dryer)
completes with P1 (the dishwasher finishes), so its end is 11:54; the stated
11:39 is 15 minutes off, so `check` returns the question "... to end at 11:39,
but when dishwasher 1 finishes is 11:54 AM. Which do you mean?".
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


def _when(text: str) -> str:
    """ "when <event>", without doubling a "when" the event's words start with
    ("when washer 1 finishes" stays as it is)."""
    return text if text.lower().startswith("when ") else f"when {text}"


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
                  remaining: Dict[str, Optional[float]],
                  durations: Optional[Dict[str, Optional[float]]] = None
                  ) -> Dict[str, Optional[Moment]]:
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

        # Completing with an event, the start left to the plan: it starts when
        # the event's time left equals its own run (right away if the run is
        # already longer). Unknown without both numbers -- never "now".
        if timing is not None and timing.start_chosen:
            left = remaining.get(timing.completes_with)
            run = (durations or {}).get(gid)
            if timing.completes_with in predicates and left is not None and run is not None:
                what = predicates[timing.completes_with].predicate_text or timing.completes_with
                moment = Moment(now + timedelta(seconds=max(0.0, left - run)),
                                f"so that it finishes when {what}")

        # Untimed or "now": it starts at the moment of the check.
        elif starts in (None, "now"):
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
                        if pred.offset_text else _when(what))
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

        # Anything else (a recurrence, a goal completing with a goal) has no single start:
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
        # 1. Completing with an event: it ends when the event happens.
        if timing and timing.completes_with in structure.predicates:
            pid = timing.completes_with
            if remaining.get(pid) is not None:
                moment = Moment(now + timedelta(seconds=remaining[pid]),
                                _when(structure.predicates[pid].predicate_text or pid))
        # 2. Completing with a goal: resolved below, once that goal's end is.
        elif timing and timing.completes_with:
            moment = None
        # 3. A duration from its start, or a termination predicate's event.
        elif timing and timing.ends_kind == "duration" and starts.get(gid) is not None:
            moment = Moment(starts[gid].at + timedelta(seconds=_span(timing.ends_text)),
                            f"{timing.ends_text} after it starts {starts[gid].said}")
        else:
            terminations = [pid for pid, p in structure.predicates.items()
                            if p.role == "termination" and gid in (p.applies_to or [])]
            if len(terminations) == 1 and remaining.get(terminations[0]) is not None:
                pred = structure.predicates[terminations[0]]
                moment = Moment(now + timedelta(seconds=remaining[terminations[0]]),
                                _when(pred.predicate_text or terminations[0]))
        out[gid] = moment
    # 4. Goals completing with another goal end with it (one level: a chain of
    #    "finish together" goals is not followed further).
    for gid, goal in structure.goals.items():
        ref = goal.timing.completes_with if goal.timing else None
        if ref in structure.goals and out.get(ref) is not None:
            out[gid] = Moment(out[ref].at, f"when {structure.goals[ref].intent_text or ref} ends")
    return out


def check(structure: GoalStructure, now: datetime, remaining: Dict[str, Optional[float]],
          tolerance: float = DEFAULT_TOLERANCE_SECONDS,
          durations: Optional[Dict[str, Optional[float]]] = None) -> Consistency:
    """Every stated start / end clock time against the moment its relative
    description names."""
    result = Consistency()
    starts = start_moments(structure, now, remaining, durations)
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


def _minutes(seconds: float) -> str:
    """A duration as the user would say it ("about 90 minutes")."""
    return f"about {max(1, round(seconds / 60))} minutes"


def check_completion(structure: GoalStructure, now: datetime,
                     remaining: Dict[str, Optional[float]],
                     durations: Dict[str, Optional[float]],
                     tolerance: float = DEFAULT_TOLERANCE_SECONDS) -> Consistency:
    """Every goal that completes with a goal or event, checked before planning
    (the module docstring's table): questions for what cannot be known or does
    not fit, notes for what was checked.

    A goal completing with another GOAL is checked only for its own run
    length: the other goal has not started, so its time left is read when the
    tree runs, once it has."""
    result = Consistency()
    starts = start_moments(structure, now, remaining, durations)
    for gid, goal in structure.goals.items():
        timing = goal.timing
        ref = timing.completes_with if timing else None
        if not ref:
            continue
        what = goal.intent_text or gid
        device = goal.artifact_name or "the device"
        event = structure.predicates.get(ref)
        # How the ref reads in a question: the event's words ("dishwasher 1
        # finishes"), or the other goal's run ending.
        if event is not None:
            ref_text = event.predicate_text or ref
        else:
            other = structure.goals.get(ref)
            ref_text = f"\"{(other.intent_text if other else None) or ref}\" ends"

        # 1. How long this goal's run takes: without it, nothing can be timed,
        #    so the request is not carried out on a guess. The event's time
        #    left, when read, is said too: it is what the user would need.
        run = durations.get(gid)
        if run is None:
            left = remaining.get(ref) if event is not None else None
            when = (f" (in {_minutes(left)}, at {_clock(now + timedelta(seconds=left))})"
                    if left is not None else "")
            result.conflicts.append(
                f"I can't do \"{what}\" as asked: {ref_text}{when}, but {device} doesn't "
                f"report how long its run takes until it has started, so I can't make "
                f"sure it finishes then. I won't start it on a guess; give me a start "
                f"time if you want it started anyway.")
            continue
        if event is None:
            result.notes.append(f"{gid}: runs {_minutes(run)}; timed against {ref} as it runs")
            continue

        # 2. How long until the event: without it, the end is unknown.
        left = remaining.get(ref)
        if left is None:
            result.conflicts.append(
                f"I can't make sure \"{what}\" finishes when {ref_text}: nothing tells "
                f"when that will be. Should I start it anyway, or at a time you give?")
            continue
        event_at = now + timedelta(seconds=left)

        # 3a. The start is the plan's to choose: possible only if the run fits
        #     in the time left.
        if timing.start_chosen:
            if run > left + tolerance:
                result.conflicts.append(
                    f"\"{what}\" takes {_minutes(run)}, but {ref_text} in {_minutes(left)} "
                    f"(at {_clock(event_at)}), so they can't finish together. "
                    f"Should I start it anyway?")
            else:
                result.notes.append(f"{gid}: starts at {_clock(event_at - timedelta(seconds=run))}"
                                    f" to finish with {ref} at {_clock(event_at)}")
            continue

        # 3b. The start is the user's: its run must end with the event.
        start = starts.get(gid)
        if start is None:
            result.conflicts.append(
                f"I can't work out when \"{what}\" starts, so I can't make sure it "
                f"finishes when {ref_text}. Should I start it anyway?")
            continue
        end = start.at + timedelta(seconds=run)
        if abs((end - event_at).total_seconds()) > tolerance:
            result.conflicts.append(
                f"\"{what}\" starting {start.said} takes {_minutes(run)} and would finish at "
                f"{_clock(end)}, but {ref_text} at {_clock(event_at)}, so they won't finish "
                f"together. Should I start it anyway?")
        else:
            result.notes.append(f"{gid}: finishes at {_clock(end)}, with {ref}")
    return result


# --------------------------------------------------------------------------
# A trigger against the start of its own device
# --------------------------------------------------------------------------

def _root_start(structure: GoalStructure, gid: str) -> Optional[str]:
    """The `starts` of the goal that ``gid``'s start is timed from, following
    after_goal / with_goal references to their end; None for a cycle of
    references or a goal not in the request."""
    seen = set()
    while gid in structure.goals and gid not in seen:
        seen.add(gid)
        timing = structure.goals[gid].timing
        starts = timing.starts if timing else None
        if starts in ("after_goal", "with_goal") and timing.ref_goal:
            gid = timing.ref_goal
            continue
        return starts
    return None


def triggered_after_timed(structure: GoalStructure) -> Dict[str, Tuple[str, List[str]]]:
    """Goal id -> (its trigger, the clock-timed goals on its device), for each
    goal due on a single trigger whose device the request also starts by the
    clock: "start dryer 1 at 18:14 ... pause dryer 1 when washer 1 finishes".

    Clock-timed means started at a time or after a delay, directly or through
    the goals it follows. A goal timed from the triggered goal itself ("resume
    it 10 minutes later") is not: it comes after the trigger by construction.
    """
    goals, predicates = structure.goals, structure.predicates
    out: Dict[str, Tuple[str, List[str]]] = {}
    for gid, goal in goals.items():
        # 1. A goal due on exactly one trigger, on a named device.
        if not (goal.timing and goal.timing.starts == "on_trigger" and goal.artifact_name):
            continue
        triggers = [pid for pid, p in predicates.items()
                    if p.role == "trigger" and gid in (p.applies_to or [])]
        if len(triggers) != 1:
            continue
        # 2. The other goals on that device whose start the clock decides.
        timed = [other for other, g in goals.items()
                 if other != gid and g.artifact_name == goal.artifact_name
                 and _root_start(structure, other) in ("at_time", "after_delay")]
        if timed:
            out[gid] = (triggers[0], timed)
    return out


def check_triggers(structure: GoalStructure, now: datetime,
                   remaining: Dict[str, Optional[float]],
                   tolerance: float = DEFAULT_TOLERANCE_SECONDS) -> Consistency:
    """Each goal from `triggered_after_timed`: is its trigger due after the
    request has started its device? A trigger due before the last clock-timed
    goal on the device starts would act on the device before the request has
    started it ("pause it when washer 1 finishes" with the washer done before
    the dryer's start): a question for the user. Unknown moments are a note."""
    result = Consistency()
    starts = start_moments(structure, now, remaining)
    for gid, (pid, timed) in triggered_after_timed(structure).items():
        goal = structure.goals[gid]
        # 1. The trigger's moment and the device's start, where both are known.
        due = starts.get(gid)
        known = [(starts[other].at, other) for other in timed if starts.get(other) is not None]
        if due is None or not known:
            result.notes.append(f"{gid}: could not check when {pid} happens against "
                                f"the start of {goal.artifact_name}")
            continue
        # 2. Due before the device is started: ask; otherwise note it.
        started_at, started_by = max(known)
        if (started_at - due.at).total_seconds() > tolerance:
            what = goal.intent_text or gid
            first = structure.goals[started_by].intent_text or started_by
            result.conflicts.append(
                f"\"{what}\" is due {due.said}, but \"{first}\" is set for "
                f"{_clock(started_at)}, after that: {goal.artifact_name} would not have "
                f"started yet. Should I leave \"{what}\" out, or did you mean another time?")
        else:
            result.notes.append(f"{gid}: {pid} is due at {_clock(due.at)}, after "
                                f"{goal.artifact_name} starts at {_clock(started_at)}")
    return result
