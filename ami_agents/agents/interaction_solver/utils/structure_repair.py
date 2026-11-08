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

`repair_structure(structure)` applies every repair in place and returns the notes.
"""

from __future__ import annotations

from typing import List, Optional

from ....shared.models.goal_structure import GoalSpec, GoalStructure, GoalTiming
from .event_lexicon import event_clause

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


REPAIRS = (drop_placeholders, name_predicates, fill_goal_defaults, link_triggers)


def repair_structure(structure: GoalStructure) -> List[str]:
    """Every repair, in place; the notes of what changed."""
    notes: List[str] = []
    for repair in REPAIRS:
        notes.extend(repair(structure))
    return notes
