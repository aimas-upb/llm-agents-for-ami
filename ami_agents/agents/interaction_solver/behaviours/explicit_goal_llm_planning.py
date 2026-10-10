"""Behaviour: plan explicit goals with a lightweight LLM, against one device.

Spawned by ``ExplicitGoalPlanningBehaviour`` for the explicit goals it could
not plan deterministically: their action takes an input and the goal does not
determine the value. Rather than hand them to the general workflow (whole home,
large model, the structure thrown away), each goal is planned against its
specific capability context -- the one device it names -- by the small model
configured under `planning.explicit_planning`.

Per goal:
    1. build_specific_context(graph, goal)    the device's affordances + a brief
    2. set:    explicit_bt_planner.generate_bt(...)   the same planner as the
                                              workflow's, with the small client
                                              and only that context
       modify: one call with MODIFY_RECIPE_PROMPT    the model names the action,
                                              property, mode and range; the
                                              recipe is checked against the TDs
                                              and built into the read -> compute
                                              -> set tree (`modify_goal_plan`)
    3. a plan entry: a tree, impossible, or an error
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from rdflib import Graph
from spade.behaviour import OneShotBehaviour

from ....shared.models.goal_structure import GoalSpec
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..prompts import MODIFY_RECIPE_PROMPT
from ..utils.modify_goal_plan import describe, modify_tree, recipe_from_answer
from ..utils.specific_capability_context import SpecificCapabilityContext, build_specific_context

SOURCE = "explicit_llm"


class ExplicitGoalLLMPlanningBehaviour(OneShotBehaviour):
    """Plan each goal against its own device; ``results`` holds one plan entry
    per goal, in order (a tree, impossible, or an error -- never None)."""

    def __init__(self, goals: List[GoalSpec], graph: Graph, logger=None) -> None:
        super().__init__()
        self.goals = goals
        self.graph = graph
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        self.results: List[Dict[str, Any]] = []

    async def run(self) -> None:
        for goal in self.goals:
            self.results.append(await self._plan(goal))

    async def _plan(self, goal: GoalSpec) -> Dict[str, Any]:
        where = f"{goal.artifact_name}.{goal.affordance_name}"

        # 1. The context: the goal's device(s), or nothing to plan against.
        context = build_specific_context(self.graph, goal)
        if context is None:
            missing = (f"No single device {goal.artifact_name!r} matches the goal's "
                       f"class and room." if goal.artifact_name else
                       f"No {goal.artifact_class or 'device'}"
                       + (f" in a {goal.location_class}" if goal.location_class else "")
                       + ".")
            return {"tree": None, "impossible": True, "source": SOURCE,
                    "explanation": missing}

        if goal.goal_effect == "modify":
            return await self._plan_modify(goal, context)

        # 2. The small model, with only that device's affordances.
        cfg = self.agent.explicit_llm
        reasoning = _is_reasoning(cfg.model)
        self.logger.info(demo(
            f"Explicit goal {where}: planning with {cfg.model} against "
            f"{len(context.affordances)} affordances of {context.artifacts}"))
        try:
            result = await self.agent.explicit_bt_planner.generate_bt(
                intents=[goal.intent_text or where],
                affordances=context.affordances,
                goal_brief=context.goal_brief,
                client=cfg.client,
                model=cfg.model,
                temperature=None if reasoning else cfg.temperature,
                reasoning_effort=cfg.reasoning_effort,
                max_completion_tokens=cfg.max_completion_tokens,
            )
        except Exception as exc:
            self.logger.warning(f"Explicit goal {where}: planning failed: {exc}")
            return {"tree": None, "error": "explicit_planning_failed", "source": SOURCE,
                    "explanation": str(exc)}

        # 3. The plan entry, in the multi-plan envelope's shape.
        explanation = result.get("explanation", "")
        if result.get("impossible"):
            return {"tree": None, "impossible": True, "source": SOURCE,
                    "explanation": explanation}
        if not result.get("tree"):
            return {"tree": None, "error": "explicit_planning_failed", "source": SOURCE,
                    "explanation": explanation or "The planner returned no tree."}
        self.logger.info(demo(f"Explicit goal {where}: planned by {cfg.model}"))
        return {"tree": result["tree"], "explanation": explanation, "source": SOURCE}

    async def _plan_modify(self, goal: GoalSpec,
                           context: SpecificCapabilityContext) -> Dict[str, Any]:
        """A modify goal: the model names the recipe, the TDs check it, and the
        tree is built as the deterministic path builds it."""
        where = f"{goal.artifact_name}.{goal.affordance_name}"
        cfg = self.agent.explicit_llm
        affordances = [{k: a.get(k) for k in ("name", "type", "semantic_types",
                                              "description", "input_schema")
                        if a.get(k) is not None} for a in context.affordances]
        messages = [
            {"role": "system", "content": MODIFY_RECIPE_PROMPT},
            {"role": "user", "content": (
                f"Request: {goal.intent_text or where}\n\nStructured goal:\n"
                f"{context.goal_brief}\n\nAffordances:\n{json.dumps(affordances, indent=1)}")},
        ]
        kwargs: Dict[str, Any] = {}
        if _is_reasoning(cfg.model):
            if cfg.reasoning_effort:
                kwargs["reasoning_effort"] = cfg.reasoning_effort
            if cfg.max_completion_tokens:
                kwargs["max_completion_tokens"] = cfg.max_completion_tokens
        else:
            kwargs["temperature"] = cfg.temperature
        self.logger.info(demo(f"Explicit goal {where}: asking {cfg.model} for a modify recipe"))
        try:
            response = await cfg.client.chat.completions.create(
                model=cfg.model, messages=messages, **kwargs)
            answer = _json_object(response.choices[0].message.content or "")
        except Exception as exc:
            self.logger.warning(f"Explicit goal {where}: recipe request failed: {exc}")
            return {"tree": None, "error": "explicit_planning_failed", "source": SOURCE,
                    "explanation": str(exc)}

        if answer.get("impossible"):
            return {"tree": None, "impossible": True, "source": SOURCE,
                    "explanation": str(answer["impossible"])}
        recipe = recipe_from_answer(self.graph, goal, answer)
        if isinstance(recipe, str):
            self.logger.warning(f"Explicit goal {where}: unusable recipe: {recipe}")
            return {"tree": None, "error": "explicit_planning_failed", "source": SOURCE,
                    "explanation": f"The model's recipe cannot be used: {recipe}."}
        self.logger.info(demo(f"Explicit goal {where}: recipe from {cfg.model}"))
        return {"tree": modify_tree(recipe), "explanation": describe(recipe),
                "source": SOURCE}


def _is_reasoning(model: str) -> bool:
    """Reasoning models (o-series, gpt-5) take no temperature."""
    return model.startswith("o") or model.startswith("gpt-5")


def _json_object(text: str) -> Dict[str, Any]:
    """The model's JSON object, a markdown fence tolerated."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed
