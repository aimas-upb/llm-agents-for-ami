"""Deterministic repairs of a dependency structure before it is planned.

The structurer (a small model) sometimes writes a correct piece of the
structure in the wrong place. Each repair here closes one such gap from what
the structure already says, without asking the model again; every repair adds
a note, so the plan's explanation records what was inferred.

    link_triggers   a trigger predicate P whose goals do not wait for it
                    (none starts on_trigger or completes_with P): the goals in
                    P.applies_to that have no start of their own, start now or
                    after a delay start on_trigger instead. A delay the goal
                    carried ("in 9 minutes") moves onto P as its offset when P
                    has none -- "9 minutes after the dryer finishes". A goal
                    placed at a clock time ("at 9:56, which is 10 minutes
                    after the washer finishes") starts on the trigger too, and
                    the clock time moves to `stated_start_time_text`, where
                    the planner checks that the two name the same moment.
                    A goal written as after_goal / with_goal whose ref_goal
                    is the trigger predicate itself starts on it too. Goals
                    placed after or with another GOAL are left as they are:
                    they follow their goal.

    drop_placeholders   a predicate with no text, no role and no goals is
                        the structurer's empty slot, not a condition: dropped
    name_predicates     a predicate with a role and goals but no text takes
                        the appliance-event clause of the request ("30
                        minutes after the dryer 1 finishes, ...") as its
                        text, for the event lexicon to complete; with no such
                        clause it is dropped, and a goal that started on it
                        starts now instead (noted: part of the request is
                        then lost, which the explanation says)
    fill_goal_defaults  a goal naming its device and action but missing
                        goal_effect ("set"), goal_kind ("achievement") or
                        goal_specificity (explicit / incomplete / ambiguous by
                        what it names) gets the default; an "explicit" goal
                        missing its device or action title, with the class
                        given, is "incomplete" (the incomplete planner finds
                        the one candidate by its names)

    complete_together   "time it so A and B finish at the same time": the goal
                        about one appliance completes with the other's run. When
                        the request names an appliance no goal is about (it is
                        already running), a trigger predicate "B finishes" is
                        added and the goal completes with it, its start left to
                        the planner; when both appliances have goals, the later
                        completes with the earlier, and a cycle of two goals
                        completing with each other is broken the same way

`repair_structure(structure)` applies every repair in place and returns the notes.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


from ....shared.models.goal_structure import GoalSpec, GoalStructure, GoalTiming, PredicateSpec
from .event_lexicon import APPLIANCES, ROOMS, event_clause

# Starts the trigger may take over: none, "now", a delay with no anchor, or a
# clock time (kept as the stated time to check against the event).
_UNANCHORED = (None, "now", "after_delay", "at_time")


def _waits_for(goal, pid: str) -> bool:
    timing = goal.timing
    return bool(timing and (timing.starts == "on_trigger" or timing.completes_with == pid))


def link_triggers(structure: GoalStructure) -> List[str]:
    """Make the goals of an unused trigger predicate start on it (module docstring)."""
    notes: List[str] = []
    for pid, pred in structure.predicates.items():
        if pred.role != "trigger":
            continue
        goals = {gid: structure.goals[gid] for gid in pred.applies_to or []
                 if gid in structure.goals}
        if not goals or any(_waits_for(g, pid) for g in goals.values()):
            continue
        linked = []
        for gid, goal in goals.items():
            timing = goal.timing
            if timing is None:
                goal.timing = GoalTiming(starts="on_trigger")
            elif ((timing.starts in _UNANCHORED
                   or (timing.starts in ("after_goal", "with_goal") and timing.ref_goal == pid))
                  and timing.completes_with != pid):
                if (timing.starts in ("after_delay", "after_goal") and timing.offset_text
                        and not pred.offset_text):
                    pred.offset_text = timing.offset_text
                    pred.offset_direction = pred.offset_direction or "after"
                if timing.starts == "at_time" and timing.time_text:
                    timing.stated_start_time_text = timing.stated_start_time_text or timing.time_text
                timing.starts = "on_trigger"
                timing.offset_text = None
                timing.time_text = None
                timing.ref_goal = None
            else:
                continue
            linked.append(gid)
        if linked:
            notes.append(f"{', '.join(linked)} start when {pid} "
                         f"({pred.predicate_text or 'the trigger'}): the structure named "
                         f"the trigger but gave the goal{'s' if len(linked) > 1 else ''} "
                         f"no start on it")
    return notes


def _empty_predicate(pred) -> bool:
    return not (pred.predicate_text or pred.artifact_name or pred.artifact_class
                or pred.property_name or pred.target_value_determined
                or pred.qualitative_condition or pred.environment_var)


def drop_placeholders(structure: GoalStructure) -> List[str]:
    """Drop predicates that carry nothing: no text, role or goals."""
    dropped = [pid for pid, pred in structure.predicates.items()
               if _empty_predicate(pred) and not pred.role and not pred.applies_to]
    for pid in dropped:
        del structure.predicates[pid]
    return [f"{', '.join(dropped)} dropped: empty predicate{'s' if len(dropped) > 1 else ''}"] \
        if dropped else []


def name_predicates(structure: GoalStructure) -> List[str]:
    """Give a text-less predicate the request's event clause, or drop it."""
    notes: List[str] = []
    for pid, pred in list(structure.predicates.items()):
        if not _empty_predicate(pred) or not (pred.role or pred.applies_to):
            continue
        clause = event_clause(structure.intent_text)
        if clause:
            pred.predicate_text = clause
            pred.predicate_subject = pred.predicate_subject or "device_property"
            notes.append(f"{pid} read as {clause!r}: the structure gave it no condition")
            continue
        del structure.predicates[pid]
        released = []
        for gid in pred.applies_to or []:
            goal = structure.goals.get(gid)
            timing = goal.timing if goal else None
            if timing and timing.starts == "on_trigger" and not any(
                    p.role == "trigger" and gid in (p.applies_to or [])
                    for p in structure.predicates.values()):
                timing.starts = None
                released.append(gid)
        notes.append(f"{pid} dropped: a {pred.role or 'condition'} with no condition in it"
                     + (f"; {', '.join(released)} start now instead" if released else ""))
    return notes


def _specificity(goal: GoalSpec) -> Optional[str]:
    if goal.artifact_name and (goal.affordance_name or goal.affordance_class):
        return "explicit"
    if goal.artifact_class or goal.location_class or goal.affordance_class:
        return "incomplete"
    if goal.implied_environment_vars:
        return "ambiguous"
    return None


def fill_goal_defaults(structure: GoalStructure) -> List[str]:
    """Defaults for the slots a well-named goal left empty (module docstring)."""
    notes: List[str] = []
    for gid, goal in structure.goals.items():
        filled = []
        has_action = bool(goal.affordance_name or goal.affordance_class)
        if goal.goal_effect is None and has_action:
            goal.goal_effect = "set"
            filled.append("effect set")
        if goal.goal_kind is None and (has_action or goal.target_value_text):
            goal.goal_kind = "achievement"
            filled.append("an achievement")
        if goal.goal_specificity is None:
            goal.goal_specificity = _specificity(goal)
            if goal.goal_specificity:
                filled.append(goal.goal_specificity)
        elif (goal.goal_specificity == "explicit" and has_action
              and not (goal.artifact_name and goal.affordance_name)
              and (goal.artifact_class or goal.location_class or goal.artifact_name)):
            goal.goal_specificity = "incomplete"
            filled.append("incomplete: " + ("no device title" if not goal.artifact_name
                                            else "no action title"))
        if filled:
            notes.append(f"{gid} taken as {', '.join(filled)}")
    return notes


_TOGETHER = re.compile(
    r"\b(?:time (?:it|them|\w+(?: \d+)?(?: in the [a-z ]+?)?) so|at the same time|at once|"
    r"finish(?:es|ed)?\s+(?:up\s+)?together|(?:both|they|the two)\s+(?:are\s+)?(?:done|ready|"
    r"finished)\s+(?:together|at the same time)?|wrap up together|end together)\b", re.I)
_APPLIANCE_MENTION = re.compile(
    r"(?:the\s+)?(" + "|".join(p for p, _ in APPLIANCES) + r")\s*(\d+)?"
    r"(?:\s+in\s+the\s+(" + "|".join(sorted((re.escape(w) for w in ROOMS), key=len, reverse=True))
    + r"))?", re.I)
_SNAKE = {"homeont:LaundryWasher": "washer", "homeont:LaundryDryer": "dryer",
          "homeont:Dishwasher": "dishwasher", "homeont:Rvc": "rvc"}


def _mentions(text: str) -> List[Tuple[str, str, Optional[str], str]]:
    """(class, number, room class, words) of each appliance the text names."""
    out, seen = [], set()
    for m in _APPLIANCE_MENTION.finditer(text or ""):
        cls = next(c for p, c in APPLIANCES if re.fullmatch(p, m.group(1), re.I))
        key = (cls, m.group(2) or "")
        if key in seen:
            continue
        seen.add(key)
        room = ROOMS.get((m.group(3) or "").lower()) if m.group(3) else None
        out.append((cls, m.group(2) or "", room, m.group(0).strip()))
    return out


def _goal_about(goal: GoalSpec, cls: str, number: str) -> bool:
    """Whether the goal acts on the appliance (by class, or by its title's words)."""
    title = (goal.artifact_name or "").lower()
    word = _SNAKE.get(cls, "")
    by_class = goal.artifact_class == cls
    by_title = bool(word) and word in title and not (
        word == "washer" and "dishwasher" in title)
    if not (by_class or by_title):
        return False
    return not number or bool(re.search(rf"(?:^|[^0-9]){number}$", title)) or not title


def complete_together(structure: GoalStructure) -> List[str]:
    """Make the goals of a finish-together request complete with each other, or
    with the run of the appliance the request only names (module docstring)."""
    text = structure.intent_text or ""
    if not _TOGETHER.search(text) or not structure.goals:
        return []
    if any(g.timing and g.timing.completes_with for g in structure.goals.values()):
        cyclic = [(gid, g) for gid, g in structure.goals.items()
                  if g.timing and g.timing.completes_with in structure.goals
                  and (structure.goals[g.timing.completes_with].timing or GoalTiming()).completes_with == gid]
        if len(cyclic) < 2:
            return []
        first_id = cyclic[0][0]
        cyclic[0][1].timing.completes_with = None
        return [f"{first_id} no longer completes with {cyclic[1][0]}, which completes with it: "
                f"one of the two has to start on its own"]
    mentions = _mentions(text)
    if len(mentions) < 2:
        return []
    with_goals = [(m, [gid for gid, g in structure.goals.items() if _goal_about(g, m[0], m[1])])
                  for m in mentions]
    timed = [gid for gid in structure.goals if any(gid in gids for _, gids in with_goals)]
    without = [m for m, gids in with_goals if not gids]
    if not timed:
        return []
    notes: List[str] = []
    if without:
        cls, number, room, words = without[0]
        pid = next(f"P{n}" for n in range(1, 100) if f"P{n}" not in structure.predicates)
        structure.predicates[pid] = PredicateSpec.from_dict({
            "predicate_specificity": "explicit", "predicate_subject": "device_property",
            "predicate_text": f"{words} finishes", "artifact_class": cls,
            "location_class": room, "role": "trigger", "applies_to": list(timed)})
        for gid in timed:
            goal = structure.goals[gid]
            goal.timing = goal.timing or GoalTiming()
            goal.timing.completes_with = pid
            goal.timing.starts = None
            goal.timing.time_text = goal.timing.offset_text = None
        notes.append(f"{', '.join(timed)} finish with {pid} ({words} finishes): the request times "
                     f"{'them' if len(timed) > 1 else 'it'} with an appliance it only names")
    elif len(timed) >= 2:
        # The anchor runs on its own: the one goal the user starts "now", else
        # the first listed; the others finish with it.
        anchors = [gid for gid in timed
                   if structure.goals[gid].timing and structure.goals[gid].timing.starts == "now"]
        first = anchors[0] if len(anchors) == 1 else timed[0]
        later = [gid for gid in timed if gid != first]
        for gid in later:
            goal = structure.goals[gid]
            goal.timing = goal.timing or GoalTiming()
            goal.timing.completes_with = first
            goal.timing.starts = None
            goal.timing.time_text = goal.timing.offset_text = None
        notes.append(f"{', '.join(later)} finish with {first}: the request wants them done together")
    return notes


REPAIRS = (drop_placeholders, name_predicates, fill_goal_defaults, complete_together, link_triggers)


def repair_structure(structure: GoalStructure) -> List[str]:
    """Every repair, in place; the notes of what changed."""
    notes: List[str] = []
    for repair in REPAIRS:
        notes.extend(repair(structure))
    return notes
