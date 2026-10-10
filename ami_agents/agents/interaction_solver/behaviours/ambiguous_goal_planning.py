"""Behaviour: plan ambiguous achievement goals, without the general workflow.

Spawned by ``GoalRequestBehaviour`` for the goals ``ambiguous_goal`` accepts.
Fetches the TD graph once, then per goal:

    1. build_specific_context(graph, goal)   per implied variable: the room's
                                             variable, the actions acting on it,
                                             the properties they act upon and the
                                             readings of the variable -- grouped
                                             by variable
    2. bt_planner.generate_bt(...)           the general planner model, with only
                                             that context; the brief tells it to
                                             decide the direction itself
    3. a plan entry: a tree, impossible, or an error

No variable with an action acting on it: impossible, with no LLM call.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rdflib import Graph
from spade.behaviour import OneShotBehaviour

from ....shared.models.goal_structure import GoalSpec
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..utils.ambiguous_goal_plan import SOURCE
from ..utils.environment_graph import fetch_environment_graph
from ..utils.specific_capability_context import build_specific_context


class AmbiguousGoalPlanningBehaviour(OneShotBehaviour):
    """Plan each given goal; ``results`` holds one plan entry per goal, in order
    (a tree, impossible, or an error -- never None)."""

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
        variables = ", ".join(f"{v.property_name} in {v.space_name or v.space_class}"
                              for v in goal.implied_environment_vars)

        # 1. The context: what acts on each variable, in its room.
        context = build_specific_context(graph, goal)
        if context is None:
            return {"tree": None, "impossible": True, "source": SOURCE,
                    "explanation": f"Nothing in this home acts on {variables}."}

        # 2. The general planner model, with only that context.
        agent = self.agent
        reasoning = agent.model.startswith("o") or agent.model.startswith("gpt-5")
        self.logger.info(demo(
            f"Ambiguous goal {goal.intent_text!r} ({variables}): planning with "
            f"{agent.model} against {len(context.affordances)} affordances of "
            f"{context.artifacts}"))
        try:
            result = await agent.bt_planner.generate_bt(
                intents=[goal.intent_text or variables],
                affordances=context.affordances,
                observable_property_hints=context.observable_property_hints,
                goal_brief=context.goal_brief,
                client=agent.llm_client,
                model=agent.model,
                temperature=None if reasoning else agent.temperature,
                reasoning_effort=agent.reasoning_effort,
                max_completion_tokens=agent.max_completion_tokens,
            )
        except Exception as exc:
            self.logger.warning(f"Ambiguous goal {goal.intent_text!r}: planning failed: {exc}")
            return {"tree": None, "error": "ambiguous_planning_failed", "source": SOURCE,
                    "explanation": str(exc)}

        # 3. The plan entry, in the multi-plan envelope's shape.
        explanation = result.get("explanation", "")
        if result.get("impossible"):
            return {"tree": None, "impossible": True, "source": SOURCE,
                    "explanation": explanation}
        if not result.get("tree"):
            return {"tree": None, "error": "ambiguous_planning_failed", "source": SOURCE,
                    "explanation": explanation or "The planner returned no tree."}
        return {"tree": result["tree"], "explanation": explanation, "source": SOURCE}
