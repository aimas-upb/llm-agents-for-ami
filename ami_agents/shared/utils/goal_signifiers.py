"""The exact key a goal is remembered under -- one function for both sides.

The UserAssistant records a successful execution under this key; the
InteractionSolver looks the same goal up under it. Both call `signifier_key`,
so the two can never drift apart. The memory engine stores and matches the key
as it is, by exact equality (`ami_agents/shared/memory/src/storage/memory_store.py`).

    kind              goal                          key fields
    explicit          explicit                      goal_specificity, location_class,
                                                    artifact_class, artifact_name,
                                                    affordance_class, affordance_name,
                                                    parameter_name*, goal_effect,
                                                    target_value_text,
                                                    target_value_is_percentage
    incomplete        incomplete, action named      goal_specificity, location_class,
                      (no name / class / room)      artifact_class, affordance_class,
                                                    affordance_name, parameter_name*,
                                                    target_value_text,
                                                    target_value_is_percentage, quantifier
    incomplete_text   incomplete, no action named   goal_specificity, intent_text
                      (value words only)            (verbatim)
    ambiguous         ambiguous                     the implied variables, each:
                                                    property_name, property_class,
                                                    property_quantity,
                                                    property_measurement_unit,
                                                    space_class, space_name,
                                                    property_change_direction
                                                    (order does not matter)

    * only when not null

`target_value_determined` is never part of a key: a goal whose value was
determined is planned deterministically and never needed remembering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..models.goal_structure import GoalSpec
from .incomplete_goals import ISA_CASES, VALUE_ONLY, incomplete_case


@dataclass(frozen=True)
class SignifierKey:
    """The kind of record and the exact fields it is matched on."""

    kind: str
    key: Dict[str, Any]

    def to_wire_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "key": self.key}


def _fields(goal: GoalSpec, names) -> Dict[str, Any]:
    key = {name: getattr(goal, name) for name in names}
    # The parameter is part of the key only when the goal names one.
    if goal.parameter_name is not None:
        key["parameter_name"] = goal.parameter_name
    return key


_EXPLICIT = ("goal_specificity", "location_class", "artifact_class", "artifact_name",
             "affordance_class", "affordance_name", "goal_effect", "target_value_text",
             "target_value_is_percentage")
_INCOMPLETE = ("goal_specificity", "location_class", "artifact_class", "affordance_class",
               "affordance_name", "target_value_text", "target_value_is_percentage",
               "quantifier")
_VARIABLE = ("property_name", "property_class", "property_quantity",
             "property_measurement_unit", "space_class", "space_name",
             "property_change_direction")


def signifier_key(goal: GoalSpec) -> Optional[SignifierKey]:
    """The key a goal is remembered under, or None for a goal memory does not
    cover (not an achievement, or an incomplete goal only the user can resolve)."""
    if goal.goal_kind != "achievement":
        return None

    if goal.goal_specificity == "explicit":
        return SignifierKey("explicit", _fields(goal, _EXPLICIT))

    if goal.goal_specificity == "incomplete":
        case = incomplete_case(goal)
        if case == VALUE_ONLY:
            # Nothing but the words to go on: the user has to say the same thing.
            return SignifierKey("incomplete_text", {"goal_specificity": "incomplete",
                                                    "intent_text": goal.intent_text})
        if case in ISA_CASES:
            return SignifierKey("incomplete", _fields(goal, _INCOMPLETE))
        return None

    if goal.goal_specificity == "ambiguous" and goal.implied_environment_vars:
        variables = [{name: getattr(var, name) for name in _VARIABLE}
                     for var in goal.implied_environment_vars]
        # A set of variables: sorted, so their order in the goal does not matter.
        variables.sort(key=lambda v: tuple(str(v[name]) for name in _VARIABLE))
        return SignifierKey("ambiguous", {"variables": variables})

    return None

