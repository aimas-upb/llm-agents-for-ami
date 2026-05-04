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
from .signifier_match_query import SignifierMatchQueryBehaviour


class PlanningWorkflowBehaviour(OneShotBehaviour):
    """Coordinate signifier matching, fast path, and LLM planning per request."""

    def __init__(
        self,
        intents: List[Intent],
        workspace_id: Optional[str] = None,
        intent_type: Optional[str] = None,
        logger=None,
    ) -> None:
        super().__init__()
        self.intents = intents
        self.workspace_id = workspace_id
        self.intent_type = intent_type
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        # Final response envelope, set by ``run``.
        self.reply_envelope: Optional[Dict[str, Any]] = None

    async def run(self) -> None:
        intent_strings = [i.to_query_string() for i in self.intents]

        # Step 1 — signifier matching (local + community in parallel).
        merged_matches = await self._gather_signifier_matches(intent_strings)
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
                intent_type=self.intent_type,
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
                self.intent_type,
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
            )
            return

        self.reply_envelope = envelope_llm_plan(
            result=plan_gen.result,
            intents=self.intents,
            workspace_id=self.workspace_id,
            intent_type=self.intent_type,
        )

    async def _gather_signifier_matches(
        self, intent_strings: List[str]
    ) -> Dict[str, Any]:
        """Spawn local + community signifier behaviours in parallel and merge results."""
        local_b = SignifierMatchQueryBehaviour(
            intents=self.intents,
            workspace_id=self.workspace_id,
            intent_type=self.intent_type,
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
