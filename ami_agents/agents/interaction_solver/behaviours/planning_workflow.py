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
    envelope_llm_plans,
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
            # Populate affected_env_vars back into intent objects for storage in signifiers
            for intent_obj in self.intents:
                if isinstance(intent_obj, ImplicitGoalIntent):
                    intent_str = intent_obj.to_query_string()
                    if intent_str in affected_env_vars_by_intent:
                        intent_obj.affected_env_vars = affected_env_vars_by_intent[intent_str]

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
            # Populate affected_env_vars even on error for complete intent structure
            for intent_obj in self.intents:
                if isinstance(intent_obj, ImplicitGoalIntent):
                    intent_str = intent_obj.to_query_string()
                    if intent_str in affected_env_vars_by_intent:
                        intent_obj.affected_env_vars = affected_env_vars_by_intent[intent_str]

            self.reply_envelope = envelope_error(
                "context_gathering_failed",
                ctx.error,
                self.intents,
            )
            return

        # Step 4 — Plan each atomic intent separately
        plans_by_intent = {}
        for intent_obj in self.intents:
            intent_str = intent_obj.to_query_string()

            # Convert affected_env_vars to only populated entries for BT planner
            affected_env_vars_for_planner = {}
            if intent_str in affected_env_vars_by_intent and affected_env_vars_by_intent[intent_str] is not None:
                affected_env_vars_for_planner[intent_str] = affected_env_vars_by_intent[intent_str]

            plan_gen = BTPlanGenerationBehaviour(
                intent_strings=[intent_str],  # Plan single intent at a time
                affordances=ctx.affordances,
                state=(
                    ctx.state_payload.get("state")
                    if isinstance(ctx.state_payload, dict)
                    else ctx.state_payload
                ),
                signifier_hints=merged_matches,
                affected_env_vars=affected_env_vars_for_planner if affected_env_vars_for_planner else None,
                logger=self.logger,
            )
            self.agent.add_behaviour(plan_gen)
            await plan_gen.join()

            if plan_gen.error:
                self.reply_envelope = envelope_error(
                    "plan_generation_failed",
                    plan_gen.error,
                    [intent_obj],
                )
                return

            plans_by_intent[intent_str] = plan_gen.result

        # Populate affected_env_vars back into intent objects for storage in signifiers
        for intent_obj in self.intents:
            if isinstance(intent_obj, ImplicitGoalIntent):
                intent_str = intent_obj.to_query_string()
                if intent_str in affected_env_vars_by_intent:
                    intent_obj.affected_env_vars = affected_env_vars_by_intent[intent_str]

        # Combine all plans into a single tree with parallel execution
        self.reply_envelope = envelope_llm_plans(
            plans_by_intent=plans_by_intent,
            intents=self.intents,
            workspace_id=self.workspace_id,
        )

    async def _gather_signifier_matches(
        self,
        intent_strings: List[str],
        affected_env_vars_by_intent: Optional[Dict[str, Optional[List[Dict[str, str]]]]] = None,
    ) -> Dict[str, Any]:
        """Spawn local signifier behaviour and optionally community signifier behaviour.

        Community signifier matching is only queried AFTER local matching if:
        1. There is a community client available
        2. At least one intent is an implicit goal with subtype == "implicit_intent"
        3. Local matching found no (or weak) matches for those implicit intents

        Explicit intents never trigger community queries (they are fully specified).
        """
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

        # Wait for local matching to complete first
        await local_b.join()
        local_matches = local_b.matches

        # Decide whether to query community based on:
        # - Local matching results for implicit_intent subtypes
        # - Availability of community client
        community_b: Optional[CommunitySignifierQueryBehaviour] = None
        implicit_intents_needing_community = []

        if getattr(self.agent, "community_client", None) is not None:
            # Check which implicit_intent subtypes had insufficient local matches
            for intent in self.intents:
                if isinstance(intent, ImplicitGoalIntent) and intent.subtype == "implicit_intent":
                    intent_str = intent.to_query_string()
                    match_data = local_matches.get(intent_str, {})
                    exact_matches = match_data.get("exact_matches", [])
                    # If no exact matches, consult community
                    if not exact_matches:
                        implicit_intents_needing_community.append(intent_str)

            # Only spawn community behaviour if there are implicit intents without local matches
            if implicit_intents_needing_community:
                community_b = CommunitySignifierQueryBehaviour(
                    intent_strings=implicit_intents_needing_community, logger=self.logger
                )
                self.agent.add_behaviour(community_b)
                await community_b.join()

        return merge_signifier_matches(
            local_matches,
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

            exact = payload.get("exact_matches") if isinstance(payload.get("exact_matches"), list) else []
            hints = payload.get("affordance_hints") if isinstance(payload.get("affordance_hints"), list) else []
            total = payload.get("total_signifiers")
            community_count = sum(
                1 for m in exact + hints if isinstance(m, dict) and m.get("source") == "community"
            )

            if exact:
                self.logger.info(
                    demo(
                        "Signifier search: intent=%r exact=%d affordance_hints=%d "
                        "(community=%d) top=%s"
                    ),
                    intent,
                    len(exact),
                    len(hints),
                    community_count,
                    exact[0].get("signifier_id", "?") if exact else "?",
                )
            else:
                self.logger.info(
                    demo(
                        "Signifier search: intent=%r exact=0 affordance_hints=%d "
                        "(community=%d) stored_total=%s"
                    ),
                    intent,
                    len(hints),
                    community_count,
                    total if total is not None else "?",
                )

