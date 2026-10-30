"""Behaviour: plan ambiguous achievement goals, without the general workflow.

Spawned by ``GoalRequestBehaviour`` for the goals ``ambiguous_goal`` accepts.
Fetches the TD graph once, then per goal:

    0. recall(agent, goal)                   a past successful plan for exactly
                                             these variables: what it used goes
                                             to the planner as signifier hints,
                                             which take precedence over the
                                             context (the LLM still plans)

    1. build_specific_context(graph, goal)   per implied variable: the room's
                                             variable, the actions acting on it,
                                             the properties they act upon and the
                                             readings of the variable -- grouped
                                             by variable
    2. _read_state(context)                  the current value of every property
                                             that context reads, so the planner
                                             sees what the devices are doing now
                                             (an AC already cooling with its fan
                                             at 0, a dehumidifier running)
    3. bt_planner.generate_bt(...)           the general planner model, with only
                                             that context and state; the brief
                                             tells it to decide the direction
                                             itself
    4. a plan entry: a tree, impossible, or an error, with the reads

No variable with an action acting on it: impossible, with no LLM call.

The entry carries `llm: True` and the `goal_context` (the affordances, per
variable) so the UserAssistant can snapshot those properties before the plan
runs and remember the plan once it has succeeded.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

from rdflib import Graph
from spade.behaviour import OneShotBehaviour

from ....shared.models.goal_structure import GoalSpec
from ....shared.utils.demo_log import demo
from ....shared.utils.llm_models import is_reasoning_model
from ....shared.utils.logger import LoggerFactory
from ..utils.ambiguous_goal_plan import SOURCE
from ..utils.environment_graph import fetch_environment_graph
from ..utils.property_read import read_property_value
from ..utils.signifier_memory import recall
from ..utils.specific_capability_context import build_specific_context


class AmbiguousGoalPlanningBehaviour(OneShotBehaviour):
    """Plan each given goal; ``results`` holds one plan entry per goal, in order
    (a tree, impossible, or an error -- never None)."""

    def __init__(self, goals: List[GoalSpec], logger=None, graph=None) -> None:
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

        # 2. What a past successful plan for exactly these variables used: hints.
        agent = self.agent
        remembered = await recall(agent, goal, self.logger)
        hints = ({"used": remembered.get("used"), "intent_text": remembered.get("intent_text"),
                  "signifier_id": remembered.get("signifier_id")}
                 if remembered and remembered.get("used") else None)

        # 3. What the devices are doing now: without it a plan may set a device
        #    to the state it is already in, and change nothing.
        state, reads = await self._read_state(context)

        # 4. The general planner model, with only that context and state.
        reasoning = is_reasoning_model(agent.model)
        self.logger.info(demo(
            f"Ambiguous goal {goal.intent_text!r} ({variables}): planning with "
            f"{agent.model} against {len(context.affordances)} affordances of "
            f"{context.artifacts}"))
        try:
            result = await agent.bt_planner.generate_bt(
                intents=[goal.intent_text or variables],
                affordances=context.affordances,
                state=state or None,
                observable_property_hints=context.observable_property_hints,
                signifier_hints=hints,
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
                    "explanation": str(exc), "reads": reads}

        # 5. The plan entry, in the multi-plan envelope's shape, with the reads
        #    the plan was made from.
        explanation = result.get("explanation", "")
        if result.get("impossible"):
            return {"tree": None, "impossible": True, "source": SOURCE,
                    "explanation": explanation, "reads": reads}
        if not result.get("tree"):
            return {"tree": None, "error": "ambiguous_planning_failed", "source": SOURCE,
                    "explanation": explanation or "The planner returned no tree.",
                    "reads": reads}
        return {"tree": result["tree"], "explanation": explanation, "source": SOURCE,
                "llm": True, "goal_context": context.goal_context, "reads": reads,
                **({"signifier_id": hints["signifier_id"]} if hints else {})}

    async def _read_state(self, context) -> Tuple[Dict[str, Dict[str, Any]],
                                                  List[Dict[str, Any]]]:
        """The current value of every property the goal's context reads (those
        its actions act upon, and the readings of its variables), as
        ({device: {property: value}}, the reads made). A property whose read
        fails is left out of the state -- unknown, not wrong -- and kept in
        the reads with no value."""
        # 1. Each property once, by URL: several variables may share a device.
        properties: Dict[str, Dict[str, Any]] = {}
        for variable in (context.goal_context or {}).get("variables", []):
            for prop in variable.get("properties", []):
                if prop.get("url"):
                    properties.setdefault(prop["url"], prop)

        # 2. Read them all at once (`read_property_value`: None when a read fails).
        values = await asyncio.gather(*(read_property_value(url) for url in properties))

        # 3. By device and property name, as the planner's Current State takes it.
        state: Dict[str, Dict[str, Any]] = {}
        reads: List[Dict[str, Any]] = []
        for prop, value in zip(properties.values(), values):
            reads.append({"artifact": prop.get("artifact"), "property": prop.get("property"),
                          "url": prop["url"], "value": value})
            if value is not None:
                state.setdefault(prop.get("artifact") or "?", {})[prop.get("property")] = value
        return state, reads
