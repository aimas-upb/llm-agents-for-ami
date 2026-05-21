"""Per-request planning workflow: orchestrates the four sub-Behaviours.

Spawned by ``GoalRequestBehaviour`` once per GOAL_REQUEST. It runs the
context queries, decides between the signifier-only fast path and the
LLM path, and writes the final reply onto ``self.reply_envelope``. The
caller (GoalRequestBehaviour) sends the SPADE reply, so the workflow
doesn't need access to the original message envelope itself.
"""

import json
from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ....shared.models.intents import ImplicitGoalIntent, ExplicitGoalIntent
from ...user_assistant.models import Intent
from ..utils.plan_envelope import (
    envelope_error,
    envelope_llm_plan,
    envelope_signifier_reuse,
)
from ..utils.signifier_fast_path import (
    collect_signifier_ids,
    try_build_signifier_only_tree,
)
from ..utils.signifier_matching import merge_signifier_matches
from .bt_plan_generation import BTPlanGenerationBehaviour
from .community_signifier_query import CommunitySignifierQueryBehaviour
from .env_context_query import EnvContextQueryBehaviour
from .implicit_goal_desire import ImplicitGoalDesireInferenceBehaviour
from .signifier_match_query import SignifierMatchQueryBehaviour


class PlanningWorkflowBehaviour(OneShotBehaviour):
    """Coordinate signifier matching, fast path, and LLM planning per request."""

    def __init__(
        self,
        intents: List[Intent],
        workspace_id: Optional[str] = None,
        logger=None,
    ) -> None:
        super().__init__()
        self.intents = intents
        self.workspace_id = workspace_id
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        # Final response envelope, set by ``run``.
        self.reply_envelope: Optional[Dict[str, Any]] = None

    async def run(self) -> None:
        intent_strings = [i.to_query_string() for i in self.intents]

        # Step 0 — Infer affected environment variables for implicit_intent subtypes
        affected_env_vars_by_intent: Dict[str, Optional[List[Dict[str, str]]]] = {}
        for intent_obj in self.intents:
            intent_str = intent_obj.to_query_string()

            # Explicit intents: skip env var inference, skip signifier matching
            if isinstance(intent_obj, ExplicitGoalIntent):
                affected_env_vars_by_intent[intent_str] = None
            # Implicit intents with subtype != "implicit_intent": skip env var inference, skip signifier matching
            elif isinstance(intent_obj, ImplicitGoalIntent) and intent_obj.subtype != "implicit_intent":
                affected_env_vars_by_intent[intent_str] = None
            # Implicit intents with subtype == "implicit_intent": infer env vars
            elif isinstance(intent_obj, ImplicitGoalIntent) and intent_obj.subtype == "implicit_intent":
                inference = ImplicitGoalDesireInferenceBehaviour(
                    intent_text=intent_obj.text_intent,
                    logger=self.logger,
                )
                self.agent.add_behaviour(inference)
                await inference.join()

                if inference.error:
                    self.logger.debug("Env var inference failed: %s", inference.error)
                    affected_env_vars_by_intent[intent_str] = None
                else:
                    env_vars = inference.result.get("affected_env_vars")
                    # Check if result is only unknown pairs
                    if env_vars == [{"variable": "unknown", "direction": "unknown"}]:
                        affected_env_vars_by_intent[intent_str] = None
                    else:
                        affected_env_vars_by_intent[intent_str] = env_vars

        # Step 1 — signifier matching (local + community in parallel).
        # For intents with affected_env_vars, use v3 matching; others skip matching
        merged_matches = await self._gather_signifier_matches(
            intent_strings,
            affected_env_vars_by_intent=affected_env_vars_by_intent,
        )
        self._log_signifier_search(merged_matches)

        # Step 2 — try the signifier-only fast path.
        fast_tree = try_build_signifier_only_tree(merged_matches, self.intents)
        if fast_tree is not None:
            self.logger.info(
                demo(
                    "BT recovered from signifiers (no EnvExplorer context queries, no LLM)"
                )
            )
            self.reply_envelope = envelope_signifier_reuse(
                tree=fast_tree,
                signifier_ids=collect_signifier_ids(merged_matches, self.intents),
                intents=self.intents,
                workspace_id=self.workspace_id,
            )
            return

        # Step 3 — LLM path: gather context, then generate.
        self.logger.info(
            demo("Gathering context from EnvExplorer (workspace_id=%r)"),
            self.workspace_id,
        )
        ctx = EnvContextQueryBehaviour(
            workspace_id=self.workspace_id, logger=self.logger
        )
        self.agent.add_behaviour(ctx)
        await ctx.join()
        if ctx.error:
            self.reply_envelope = envelope_error(
                "context_gathering_failed",
                ctx.error,
                self.intents,
            )
            return

        # Convert affected_env_vars to only populated entries for BT planner
        affected_env_vars_for_planner = {
            intent_str: env_vars
            for intent_str, env_vars in affected_env_vars_by_intent.items()
            if env_vars is not None
        } if affected_env_vars_by_intent else None

        plan_gen = BTPlanGenerationBehaviour(
            intent_strings=intent_strings,
            affordances=ctx.affordances,
            state=(
                ctx.state_payload.get("state")
                if isinstance(ctx.state_payload, dict)
                else ctx.state_payload
            ),
            signifier_hints=merged_matches,
            affected_env_vars=affected_env_vars_for_planner,
            logger=self.logger,
        )
        self.agent.add_behaviour(plan_gen)
        await plan_gen.join()
        if plan_gen.error:
            self.reply_envelope = envelope_error(
                "plan_generation_failed",
                plan_gen.error,
                self.intents,
            )
            return

        self.reply_envelope = envelope_llm_plan(
            result=plan_gen.result,
            intents=self.intents,
            workspace_id=self.workspace_id,
        )

    async def _gather_signifier_matches(
        self,
        intent_strings: List[str],
        affected_env_vars_by_intent: Optional[Dict[str, Optional[List[Dict[str, str]]]]] = None,
    ) -> Dict[str, Any]:
        """Spawn local + community signifier behaviours in parallel and merge results."""
        # Only pass affected_env_vars if they are populated (for v3 matching)
        affected_env_vars_for_matching = {
            intent_str: env_vars
            for intent_str, env_vars in (affected_env_vars_by_intent or {}).items()
            if env_vars is not None
        } if affected_env_vars_by_intent else None

        local_b = SignifierMatchQueryBehaviour(
            intents=self.intents,
            workspace_id=self.workspace_id,
            affected_env_vars=affected_env_vars_for_matching,
            logger=self.logger,
        )
        self.agent.add_behaviour(local_b)

        community_b: Optional[CommunitySignifierQueryBehaviour] = None
        if getattr(self.agent, "community_client", None):
            community_b = CommunitySignifierQueryBehaviour(
                intent_strings=intent_strings, logger=self.logger
            )
            self.agent.add_behaviour(community_b)

        await local_b.join()
        if community_b is not None:
            await community_b.join()

        return merge_signifier_matches(
            local_b.matches,
            community_b.matches if community_b is not None else {},
            intent_strings,
        )

    def _log_signifier_search(self, results: Dict[str, Any]) -> None:
        """Demo-friendly per-intent summary of signifier search outcomes."""
        for intent, payload in results.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("error"):
                self.logger.info(
                    demo("Signifier search failed: intent=%r error=%s"),
                    intent,
                    payload.get("error"),
                )
                continue

            matches = payload.get("matches") if isinstance(payload.get("matches"), list) else []
            finals = (
                payload.get("final_matches")
                if isinstance(payload.get("final_matches"), list)
                else []
            )
            total = payload.get("total_signifiers")
            community_count = sum(
                1 for m in matches if isinstance(m, dict) and m.get("source") == "community"
            )

            if finals:
                self.logger.info(
                    demo(
                        "Signifier search: intent=%r matches=%d (community=%d) "
                        "final=%d top=%s"
                    ),
                    intent,
                    len(matches),
                    community_count,
                    len(finals),
                    finals[0],
                )
            else:
                self.logger.info(
                    demo(
                        "Signifier search: intent=%r matches=%d (community=%d) "
                        "final=0 stored_total=%s"
                    ),
                    intent,
                    len(matches),
                    community_count,
                    total if total is not None else "?",
                )
