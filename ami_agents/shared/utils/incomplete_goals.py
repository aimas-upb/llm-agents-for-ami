"""Which kind of incomplete goal a goal is, by the slots it leaves empty.

An incomplete goal names some of its room (L, `location_class`), device class
(C, `artifact_class`), device (N, `artifact_name`), action (A,
`affordance_class` / `affordance_name`) and value words (V,
`target_value_text`). What is missing decides who handles it, and how:

    UNNAMED          L, C, N null            UA: ask the user (which device?)
    NO_ACTION        A null, V null          UA: ask the user (do what?)
    VALUE_ONLY       A null, V given         ISA: the devices of C in L, planned
                                             by the small model -- possibly
                                             impossible
    NO_LOCATION      L null, C and A given   UA looks the devices up first (one
                                             room class fills L; several ask the
                                             user, unless the goal means "all");
                                             then the ISA, as NO_NAME
    NO_CLASS         C null, L and A given   ISA: the devices in L with A (of
                                             several kinds: all of them if the
                                             goal means "all", else ask the user)
    NO_NAME          N null, the rest given  ISA: the devices of C in L with A

The cases are tested in that order, so a goal falls in the first that fits.
A goal naming everything but still labelled incomplete is NO_NAME: filtering
on its names finds its one device.

Used by both agents: the UA to decide what to ask and what to send, the ISA to
decide how to plan what it receives.
"""

from __future__ import annotations

from typing import Optional

from ..models.goal_structure import GoalSpec

UNNAMED = "unnamed"
NO_ACTION = "no_action"
VALUE_ONLY = "value_only"
NO_LOCATION = "no_location"
NO_CLASS = "no_class"
NO_NAME = "no_name"

# The cases only the user can resolve: they never reach the InteractionSolver.
UA_CLARIFY = frozenset({UNNAMED, NO_ACTION})
# The cases the InteractionSolver plans (NO_LOCATION after the UA's lookup).
ISA_CASES = frozenset({VALUE_ONLY, NO_LOCATION, NO_CLASS, NO_NAME})


def incomplete_case(goal: GoalSpec) -> Optional[str]:
    """The goal's case, or None when it is not an incomplete goal."""
    if goal.goal_specificity != "incomplete":
        return None

    # Nothing names what the goal is about: only the user can say.
    if not (goal.location_class or goal.artifact_class or goal.artifact_name):
        return UNNAMED

    # The action: known if either its class or its title was found. Without
    # it, the value words are all there is to go on -- and without those too,
    # nothing says what to do.
    has_action = bool(goal.affordance_class or goal.affordance_name)
    if not has_action:
        return VALUE_ONLY if goal.target_value_text else NO_ACTION

    # The action is known; what is missing is where, or on what.
    if not goal.location_class and goal.artifact_class:
        return NO_LOCATION
    if not goal.artifact_class and goal.location_class:
        return NO_CLASS
    return NO_NAME
