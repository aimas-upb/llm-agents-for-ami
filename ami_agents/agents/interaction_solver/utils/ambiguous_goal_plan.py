"""Which ambiguous goals the InteractionSolver plans on their own path.

An ambiguous goal ("the bathroom is so damp") names environment variables, not
devices: the planner gets a context built from those variables
(`specific_capability_context._ambiguous_achievement`) and writes the tree
itself (`behaviours/ambiguous_goal_planning.py`).
"""

from __future__ import annotations

from typing import Optional

from ....shared.models.goal_structure import GoalSpec, GoalStructure

# Marks plan entries produced on this path.
SOURCE = "ambiguous_llm"


def ambiguous_goal(structure: GoalStructure) -> Optional[GoalSpec]:
    """The goal, if this structure is one ambiguous achievement goal naming at
    least one environment variable; None leaves it to the planning workflow
    (dependency structures, maintenance goals)."""
    if structure.structure != "simple" or structure.predicates or len(structure.goals) != 1:
        return None
    (goal,) = structure.goals.values()
    if (goal.goal_specificity != "ambiguous" or goal.goal_kind != "achievement"
            or not goal.implied_environment_vars):
        return None
    return goal
