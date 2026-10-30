"""Behaviour: plan explicit achievement goals, without the general workflow.

Spawned by ``GoalRequestBehaviour`` for the goals ``deterministic_goal``
accepts -- explicit, achievement, `set`, naming their device and action.
Fetches the TD graph once, then:

    1. plans each goal deterministically, no LLM:
         set     a one-action tree, or impossible (``utils/explicit_goal_plan.py``)
         modify  read -> compute -> set (``utils/modify_goal_plan.py``); a value
                 stepped along an ordered scale is read first, and a question
                 is the entry when it is not on the scale (an `auto` fan speed)
    2. hands the goals that could not be planned that way (a set whose value
       is not determined; a modify whose recipe cannot be read off the TDs) to
       ``ExplicitGoalLLMPlanningBehaviour``, which plans each against its one
       device with the lightweight model, on the same graph.

So an explicit goal never reaches the general planning workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rdflib import Graph
from spade.behaviour import OneShotBehaviour

from ....shared.models.goal_structure import GoalSpec
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..utils.environment_graph import fetch_environment_graph
from ..utils.explicit_goal_plan import plan_explicit_goal
from ..utils.modify_goal_plan import SOURCE as MODIFY_SOURCE
from ..utils.modify_goal_plan import describe, modify_recipe, modify_tree
from ..utils.property_read import off_scale_check
from .explicit_goal_llm_planning import ExplicitGoalLLMPlanningBehaviour


class ExplicitGoalPlanningBehaviour(OneShotBehaviour):
    """Plan each given goal; ``results`` holds one plan entry per goal, in order
    (a tree, impossible, or an error)."""

    def __init__(self, goals: List[GoalSpec], logger=None,
                 graph: Optional[Graph] = None) -> None:
        super().__init__()
        self.goals = goals
        # Given when the caller has fetched the TD graph already.
        self.graph = graph
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        self.results: List[Dict[str, Any]] = []

    async def run(self) -> None:
        graph = self.graph
        if graph is None:
            graph = await fetch_environment_graph(self.agent, self.logger)

        # 1. Deterministic pass. None marks a goal that cannot be planned from
        #    the TDs alone.
        results: List[Optional[Dict[str, Any]]] = []
        for goal in self.goals:
            where = f"{goal.artifact_name}.{goal.affordance_name}"
            if goal.goal_effect == "modify":
                recipe = modify_recipe(graph, goal)
                result = None if isinstance(recipe, str) else await self._modify_entry(recipe)
                reason = recipe if isinstance(recipe, str) else None
            else:
                result = plan_explicit_goal(graph, goal)
                reason = "the action takes an input and no value was determined"
            results.append(result)
            if result is None:
                self.logger.info(demo(
                    f"Explicit goal {where}: {reason} -- planning against the device "
                    f"with the small model"))
                continue
            outcome = ("IMPOSSIBLE" if result.get("impossible") else
                       "ASK THE USER" if result.get("requires_clarification") else "PLANNED")
            self.logger.info(demo(
                f"Explicit goal {where} = {goal.target_value_determined!r}: "
                f"{outcome} -- {result.get('explanation')}"))

        # 2. The rest, against their own device, with the lightweight LLM.
        pending = [i for i, result in enumerate(results) if result is None]
        if pending:
            llm = ExplicitGoalLLMPlanningBehaviour(
                [self.goals[i] for i in pending], graph, logger=self.logger)
            self.agent.add_behaviour(llm)
            await llm.join()
            for i, result in zip(pending, llm.results):
                results[i] = result

        self.results = results

    @staticmethod
    async def _modify_entry(recipe) -> Dict[str, Any]:
        """The plan entry of a modify recipe: the question to ask when it steps an
        ordered value that is not on its scale (read now, `off_scale_check`), else
        the read -> compute -> set tree."""
        asked = await off_scale_check(recipe, MODIFY_SOURCE)
        if asked is not None:
            return asked
        return {"tree": modify_tree(recipe), "explanation": describe(recipe),
                "source": MODIFY_SOURCE}
