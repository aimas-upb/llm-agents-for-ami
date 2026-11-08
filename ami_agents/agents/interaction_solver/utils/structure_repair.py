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

`repair_structure(structure)` applies every repair in place and returns the notes.
"""

from __future__ import annotations

from typing import List

from ....shared.models.goal_structure import GoalStructure, GoalTiming

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


REPAIRS = (link_triggers,)


def repair_structure(structure: GoalStructure) -> List[str]:
    """Every repair, in place; the notes of what changed."""
    notes: List[str] = []
    for repair in REPAIRS:
        notes.extend(repair(structure))
    return notes
