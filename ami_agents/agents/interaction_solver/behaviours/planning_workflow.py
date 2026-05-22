"""Per-request planning workflow: orchestrates the four sub-Behaviours.

Spawned by ``GoalRequestBehaviour`` once per GOAL_REQUEST. It runs the
context queries, decides between the signifier-only fast path and the
LLM path, and writes the final reply onto ``self.reply_envelope``. The
caller (GoalRequestBehaviour) sends the SPADE reply, so the workflow
doesn't need access to the original message envelope itself.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ...user_assistant.models import Intent
from ..utils.plan_envelope import (
    envelope_error,
    envelope_llm_plan,
    envelope_signifier_reuse,
)
from ..utils.goal_status import GoalStatus, PlanningPhase
from ..utils.signifier_fast_path import (
    collect_signifier_ids,
    try_build_signifier_only_tree,
)
from ..utils.signifier_matching import merge_signifier_matches
from .bt_plan_generation import BTPlanGenerationBehaviour
from .community_query import CommunityQueryBehaviour
from .env_context_query import EnvContextQueryBehaviour
from .signifier_match_query import SignifierMatchQueryBehaviour


class PlanningWorkflowBehaviour(OneShotBehaviour):
    """Coordinate signifier matching, fast path, and LLM planning per request."""

    def __init__(
        self,
        intents: List[Intent],
        workspace_id: Optional[str] = None,
        intent_type: Optional[str] = None,
        logger=None,
        goal_status: Optional[GoalStatus] = None,
    ) -> None:
        super().__init__()
        self.intents = intents
        self.workspace_id = workspace_id
        self.intent_type = intent_type
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        self.goal_status = goal_status
        # Final response envelope, set by ``run``.
        self.reply_envelope: Optional[Dict[str, Any]] = None

    async def run(self) -> None:
        intent_strings = [i.to_query_string() for i in self.intents]
        if self.goal_status:
            self.goal_status.update_status(
                phase=PlanningPhase.GATHERING_REUSED_PLAN,
                intents=intent_strings,
                workspace_id=self.workspace_id,
                error=None,
                error_detail=None,
            )

        # Step 1 — Query LOCAL signifiers only.
        local_matches = await self._gather_local_signifier_matches(intent_strings)
        self._log_signifier_search(local_matches)

        # Step 2 — Try the signifier-only fast path with local matches.
        fast_tree = try_build_signifier_only_tree(local_matches, self.intents)
        if fast_tree is not None:
            self.logger.info(
                demo(
                    "BT recovered from local signifiers (no community query, no LLM)"
                )
            )
            self.reply_envelope = envelope_signifier_reuse(
                tree=fast_tree,
                signifier_ids=collect_signifier_ids(local_matches, self.intents),
                intents=self.intents,
                workspace_id=self.workspace_id,
                intent_type=self.intent_type,
                goal_id=self.goal_status.goal_id if self.goal_status else None,
            )
            if self.goal_status:
                self.goal_status.update_status(
                    phase=PlanningPhase.COMPLETED_SUCCESS,
                    reused_plan=self.reply_envelope,
                    reused_plan_source="signifier",
                    best_plan=self.reply_envelope,
                    best_plan_source="reused",
                )
            return

        # Step 3 — No local plan; if community is enabled, spawn community query and WAIT
        # for responses so this workflow continues generation and prepares the reply.
        self.logger.info(demo("No local signifier match; checking community..."))
        community_matches = {}
        if getattr(self.agent, "community_enabled", False) and self.goal_status:
            # mark phase and run CommunityQueryBehaviour to collect responses
            self.goal_status.update_status(phase=PlanningPhase.QUERYING_COMMUNITY)
            community_b = CommunityQueryBehaviour(
                goal_status=self.goal_status,
                intent_strings=intent_strings,
                workspace_id=self.workspace_id,
                intent_type=self.intent_type,
                structured_intents=[i.to_dict() for i in self.intents],
                logger=self.logger,
            )
            self.agent.add_behaviour(community_b)
            await community_b.join()
            community_matches = community_b.matches or {}
            self.logger.info(
                demo("Community query finished — received %d responses for goal_id=%s"),
                len(self.goal_status.community_responses),
                self.goal_status.goal_id,
            )

        # Merge local + community matches (community_matches may be empty)
        merged_matches = merge_signifier_matches(local_matches, community_matches, intent_strings)
        self._log_signifier_search(merged_matches)

        # Step 4 — Try the signifier-only fast path with merged (local + community) matches.
        fast_tree = try_build_signifier_only_tree(merged_matches, self.intents)
        if fast_tree is not None:
            self.logger.info(
                demo(
                    "BT recovered from community signifiers (no EnvExplorer context queries, no LLM)"
                )
            )
            self.reply_envelope = envelope_signifier_reuse(
                tree=fast_tree,
                signifier_ids=collect_signifier_ids(merged_matches, self.intents),
                intents=self.intents,
                workspace_id=self.workspace_id,
                intent_type=self.intent_type,
                goal_id=self.goal_status.goal_id if self.goal_status else None,
            )
            if self.goal_status:
                self.goal_status.update_status(
                    phase=PlanningPhase.COMPLETED_SUCCESS,
                    reused_plan=self.reply_envelope,
                    reused_plan_source="signifier",
                    best_plan=self.reply_envelope,
                    best_plan_source="community",
                )
            return

        # Step 5 — LLM path: gather context, then generate.
        self.logger.info(
            demo("Gathering context from EnvExplorer (workspace_id=%r)"),
            self.workspace_id,
        )
        if self.goal_status:
            self.goal_status.update_status(phase=PlanningPhase.GENERATING_LOCAL_PLAN)
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
                self.intent_type,
                goal_id=self.goal_status.goal_id if self.goal_status else None,
            )
            if self.goal_status:
                self.goal_status.update_status(
                    phase=PlanningPhase.COMPLETED_FAILURE,
                    error="context_gathering_failed",
                    error_detail=ctx.error,
                )
            return

        plan_gen = BTPlanGenerationBehaviour(
            intent_strings=intent_strings,
            affordances=ctx.affordances,
            state=(
                ctx.state_payload.get("state")
                if isinstance(ctx.state_payload, dict)
                else ctx.state_payload
            ),
            signifier_hints=merged_matches,
            logger=self.logger,
        )
        self.agent.add_behaviour(plan_gen)
        await plan_gen.join()
        if plan_gen.error:
            self.reply_envelope = envelope_error(
                "plan_generation_failed",
                plan_gen.error,
                self.intents,
                self.intent_type,
                goal_id=self.goal_status.goal_id if self.goal_status else None,
            )
            if self.goal_status:
                self.goal_status.update_status(
                    phase=PlanningPhase.COMPLETED_FAILURE,
                    error="plan_generation_failed",
                    error_detail=plan_gen.error,
                )
            return

        self.reply_envelope = envelope_llm_plan(
            result=plan_gen.result,
            intents=self.intents,
            workspace_id=self.workspace_id,
            intent_type=self.intent_type,
            goal_id=self.goal_status.goal_id if self.goal_status else None,
        )
        if self.goal_status:
            self.goal_status.update_status(
                phase=PlanningPhase.COMPLETED_SUCCESS,
                local_plan=self.reply_envelope,
                local_plan_complete=True,
                best_plan=self.reply_envelope,
                best_plan_source="local",
            )

    async def _gather_local_signifier_matches(
        self, intent_strings: List[str]
    ) -> Dict[str, Any]:
        """Query local signifiers via EnvExplorer only."""
        local_b = SignifierMatchQueryBehaviour(
            intents=self.intents,
            workspace_id=self.workspace_id,
            intent_type=self.intent_type,
            logger=self.logger,
        )
        self.agent.add_behaviour(local_b)
        await local_b.join()
        return local_b.matches

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
