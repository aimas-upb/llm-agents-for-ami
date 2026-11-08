"""Behaviour: structure one GOAL_REQUEST into goals and predicates.

Stage 2 for goals. The segmenter has typed the intent and qualified it; the
qualifiers decide the rest (see `utils/goal_routing.py`):

    1. route       simple or dependency prompt, and which context it needs
    2. context     the goal context from EnvExplorer, scoped to this goal
    3. structure   one LLM call -> a `GoalStructure`; how the call is built
                   (context format, prompt, output schema, repair of the reply)
                   follows `user_assistant.prompt_variants.structuring`, see
                   `utils/goal_structuring_variants.py` (all defaults: as always)
    4. resolve     incomplete goals and device predicates that the
                   environment allows only one reading of are promoted to
                   explicit (see `utils/goal_resolution.py`)

One behaviour per goal, per CLAUDE.md 3.2: goals in one utterance structure
concurrently. No step falls back to a coarser description: without its
context a goal is not structured at all, and the caller says so.
"""

from __future__ import annotations

import asyncio
import json
from typing import List, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.models.goal_structure import GoalStructure
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ....shared.utils.prompt_variants import selection_from_config
from .. import pipeline
from ..utils import loose_json_loads
from ..utils.goal_resolution import (
    apply_goal_resolution,
    apply_predicate_resolution,
    default_quantifiers,
    goal_query,
    predicate_query,
)
from ..utils.goal_context import GoalContext
from ..utils.goal_routing import GoalRoute, route_goal
from ..utils.llm_client import build_behaviour_llm_client, build_llm_call_kwargs


class GoalStructuringBehaviour(OneShotBehaviour):
    """Structure ONE GOAL_REQUEST intent."""

    def __init__(self, intent_text: str, qualifiers: Optional[List[str]] = None,
                 logger=None) -> None:
        super().__init__()
        self.intent_text = intent_text
        self.qualifiers = list(qualifiers or [])
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")
        # Populated by ``run``: a GoalStructure, or None with ``error`` set.
        self.result: Optional[GoalStructure] = None
        self.error: Optional[str] = None
        # True when the goal context could not be had -- a different failure
        # from a parse that went wrong, and reported differently.
        self.context_unavailable = False
        # What the `repair` variant changed in the model's reply, one line each
        # (empty with the default variant).
        self.repairs: List[str] = []

    async def run(self) -> None:
        route = route_goal(self.qualifiers)
        self.logger.info(demo(
            f"[GOAL] Structuring {self.intent_text[:80]!r}: "
            f"qualifiers={self.qualifiers} -> {route.structure} "
            f"(properties={route.with_properties}, environment={route.with_environment})"))

        context = await pipeline.fetch_goal_context_data(
            self.agent, self.logger, self.intent_text,
            with_properties=route.with_properties,
            with_environment=route.with_environment)
        if not context.text:
            self.context_unavailable = True
            self.error = "goal context unavailable"
            return

        try:
            parsed = await self._structure(route, context)
        except Exception as exc:
            self.error = str(exc)
            self.logger.warning("Goal structuring failed for %r: %s",
                                self.intent_text, exc)
            return

        structure = GoalStructure.from_dict(
            parsed, intent_text=self.intent_text, qualifiers=self.qualifiers,
            structure=route.structure)
        if not structure.goals:
            self.error = "no goal could be structured"
            return

        await self._resolve(structure)
        default_quantifiers(structure.predicates)
        self.result = structure
        self.logger.info(demo(
            f"[GOAL structured]\n{json.dumps(structure.to_wire_dict(), indent=2)}"))

    async def _structure(self, route: GoalRoute, context: GoalContext) -> dict:
        """The one LLM call: the parsed JSON object the model answered with.

        Built as the config's `prompt_variants.structuring` says (all defaults: the
        full prompt over the text context, a free-text reply, no repair). Raises when
        the model's reply is not a JSON object, or when a variant cannot work."""
        variants = selection_from_config(self.agent.config, "structuring")

        # 1. The messages: the context rendered, then the system prompt around it.
        rendered = variants["context"](context)
        messages = [
            {"role": "system", "content": variants["prompt"](
                route.structure, rendered, self.qualifiers)},
            {"role": "user", "content": (
                f"Request: {self.intent_text}\n"
                f"Segmenter qualifiers: {', '.join(self.qualifiers) or 'none'}")},
        ]

        # 2. The call; a constrained reply adds its `response_format`.
        llm_cfg = build_behaviour_llm_client(self.agent.config, "intent_parsing")
        kwargs = build_llm_call_kwargs(llm_cfg)
        response_format = variants["constraint"](route.structure, self.qualifiers, context)
        if response_format:
            kwargs["response_format"] = response_format
        response = await llm_cfg.client.chat.completions.create(
            model=llm_cfg.model,
            messages=messages,
            **kwargs,
        )

        # 3. The reply: an object, then the variant's repair of it.
        raw = (response.choices[0].message.content or "").strip()
        parsed = loose_json_loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"expected a JSON object, got: {raw[:200]!r}")
        parsed, self.repairs = variants["repair"](parsed, context)
        for line in self.repairs:
            self.logger.info(demo(f"[GOAL] repaired {line}"))
        return parsed

    async def _resolve(self, structure: GoalStructure) -> None:
        """Look up every incomplete goal and device predicate concurrently, and
        promote those with exactly one fit."""
        jobs = []
        for gid, goal in structure.goals.items():
            payload = goal_query(goal)
            if payload:
                jobs.append((gid, goal, payload, apply_goal_resolution))
        for pid, predicate in structure.predicates.items():
            payload = predicate_query(predicate)
            if payload:
                jobs.append((pid, predicate, payload, apply_predicate_resolution))
        if not jobs:
            return

        responses = await asyncio.gather(*[
            pipeline.resolve_candidates(self.agent, self.logger, payload)
            for _, _, payload, _ in jobs])
        for (key, item, _, apply), response in zip(jobs, responses):
            promoted = apply(item, response)
            self.logger.info(demo(
                f"[GOAL] {key}: {response.get('outcome', 'no answer')} -> "
                f"{'explicit' if promoted else 'unchanged'}"))
