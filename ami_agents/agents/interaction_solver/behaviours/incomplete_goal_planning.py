"""Behaviour: plan incomplete achievement goals, without the general workflow.

Spawned by ``GoalRequestBehaviour`` for the goals ``incomplete_goal`` accepts.
Fetches the TD graph once, then per goal:

    VALUE_ONLY (no action named)    the small model, against every device the
                                    goal's class and room allow, told that the
                                    request may be impossible with them
                                    (``ExplicitGoalLLMPlanningBehaviour`` -- it
                                    dispatches on the goal's specificity)
    otherwise                       candidates(graph, goal) -> choose(...):
        impossible                  a tree-less entry with the reason
        clarify                     a tree-less entry the UA asks the user about
        plan                        each chosen device as an explicit goal,
                                    planned by ``ExplicitGoalPlanningBehaviour``
                                    (set: one action; modify: read -> compute
                                    -> set; deterministic where it can be, the
                                    small model otherwise), then combine(...)
                                    into one entry

The choosing is in ``utils/incomplete_goal_plan.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rdflib import Graph
from spade.behaviour import OneShotBehaviour

from ....shared.models.goal_structure import GoalSpec
from ....shared.utils.demo_log import demo
from ....shared.utils.incomplete_goals import VALUE_ONLY, incomplete_case
from ....shared.utils.logger import LoggerFactory
from ..utils.environment_graph import fetch_environment_graph
from ..utils.incomplete_goal_plan import (
    CLARIFY,
    IMPOSSIBLE,
    SOURCE,
    as_explicit,
    candidates,
    choose,
    clarification_entry,
    combine,
    impossible_entry,
)
from .explicit_goal_llm_planning import ExplicitGoalLLMPlanningBehaviour
from .explicit_goal_planning import ExplicitGoalPlanningBehaviour


class IncompleteGoalPlanningBehaviour(OneShotBehaviour):
    """Plan each given goal; ``results`` holds one plan entry per goal, in order
    (a tree, impossible, a clarification request, or an error)."""

    def __init__(self, goals: List[GoalSpec], logger=None) -> None:
        super().__init__()
        self.goals = goals
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        self.results: List[Dict[str, Any]] = []

    async def run(self) -> None:
        graph = await fetch_environment_graph(self.agent, self.logger)
        for goal in self.goals:
            self.results.append(await self._plan(graph, goal))

    async def _plan(self, graph: Graph, goal: GoalSpec) -> Dict[str, Any]:
        case = incomplete_case(goal)

        # 1. No action named: the small model chooses one among the candidate
        #    devices' affordances -- or says it is impossible.
        if case == VALUE_ONLY:
            self.logger.info(demo(
                f"Incomplete goal {goal.intent_text!r}: no action named -- planning "
                f"against the {goal.artifact_class or 'devices'} with the small model"))
            (entry,) = await self._run(ExplicitGoalLLMPlanningBehaviour(
                [goal], graph, logger=self.logger))
            return {**entry, "source": SOURCE}

        # 2. The action is named: find the devices it fits, and choose.
        found = candidates(graph, goal)
        choice = choose(goal, found)
        self.logger.info(demo(
            f"Incomplete goal {goal.intent_text!r} ({case}, quantifier "
            f"{goal.quantifier}): {len(found)} candidate(s) -> {choice.outcome} "
            f"{[c.title for c in choice.chosen]}"))
        if choice.outcome == IMPOSSIBLE:
            return impossible_entry(choice.reason)
        if choice.outcome == CLARIFY:
            return clarification_entry(choice)

        # 3. Each chosen device as an explicit goal, planned on the explicit
        #    path, and the per-device plans combined.
        goals = [as_explicit(goal, c) for c in choice.chosen]
        behaviour = ExplicitGoalPlanningBehaviour(goals, logger=self.logger, graph=graph)
        return combine(await self._run(behaviour), choice.chosen)

    async def _run(self, behaviour) -> List[Dict[str, Any]]:
        """Run a planning sub-behaviour to completion; its per-goal entries."""
        self.agent.add_behaviour(behaviour)
        await behaviour.join()
        return behaviour.results
