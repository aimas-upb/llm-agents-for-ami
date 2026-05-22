"""Behaviour: send COMMUNITY_REQUEST messages and await responses."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour
from spade.message import Message as SpadeMessage

from ....shared.models.messages import (
    META_CONVERSATION_ID,
    META_CORRELATION_ID,
    MessageType,
    serialize_body,
)
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ....shared.utils.spade_rpc import send_via_router
from ..utils.community import Community
from .community_timeout import CommunityTimeoutBehaviour


class CommunityQueryBehaviour(OneShotBehaviour):
    """Query community agents via XMPP and aggregate responses."""

    def __init__(
        self,
        goal_status,
        intent_strings: List[str],
        workspace_id: Optional[str] = None,
        intent_type: Optional[str] = None,
        structured_intents: Optional[List[dict]] = None,
        logger=None,
    ) -> None:
        super().__init__()
        self.goal_status = goal_status
        self.intent_strings = intent_strings
        self.workspace_id = workspace_id
        self.intent_type = intent_type
        self.structured_intents = structured_intents
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        self.matches: Dict[str, Any] = {}

    async def run(self) -> None:
        goal_status = self.goal_status
        goal_id = goal_status.goal_id

        communities: List[Community] = getattr(self.agent, "communities", []) or []
        if not self.agent.community_enabled or not communities:
            self.logger.warning(
                demo("No communities configured for goal_id=%s, skipping community query"),
                goal_id,
            )
            goal_status.continue_triggered = True
            goal_status.community_event.set()
            self.matches = {}
            return

        agents_to_query = set()
        matching_communities: List[str] = []
        for community in communities:
            match_score = community.matches_intents(self.intent_strings)
            if match_score >= self.agent.community_match_threshold:
                matching_communities.append(community.community_id)
                agents_to_query.update(community.member_ids)

        my_jid = str(self.agent.jid).split("/")[0]
        agents_to_query.discard(my_jid)

        goal_status.update_status(relevant_communities=matching_communities)

        if not agents_to_query:
            self.logger.warning(
                demo("No community agents to query for goal_id=%s"),
                goal_id,
            )
            goal_status.continue_triggered = True
            goal_status.community_event.set()
            self.matches = {}
            return

        self.logger.info(
            demo("Sending COMMUNITY_REQUEST to %d agents for goal_id=%s"),
            len(agents_to_query),
            goal_id,
        )

        goal_status.community_expected_responses = len(agents_to_query)
        goal_status.continue_triggered = False
        if goal_status.community_event.is_set():
            goal_status.community_event.clear()

        base_conversation_id = f"community_plan_{goal_id}"
        request_payload = {
            "goal_id": goal_id,
            "intents": self.intent_strings,
            "workspace_id": self.workspace_id,
            "intent_type": self.intent_type,
            "structured_intents": self.structured_intents or [],
        }

        for agent_jid in agents_to_query:
            conversation_id = f"{base_conversation_id}_{agent_jid.split('@')[0]}"
            msg = SpadeMessage(to=agent_jid)
            msg.set_metadata("type", MessageType.COMMUNITY_REQUEST.value)
            msg.set_metadata(META_CONVERSATION_ID, conversation_id)
            msg.set_metadata(META_CORRELATION_ID, str(uuid.uuid4()))
            msg.body = serialize_body(request_payload)
            await send_via_router(self.agent, msg)

        timeout_s = self.agent.community_query_timeout
        timeout_behaviour = CommunityTimeoutBehaviour(goal_status, logger=self.logger)
        loop = asyncio.get_event_loop()

        def add_timeout_behaviour() -> None:
            if not goal_status.continue_triggered:
                self.agent.add_behaviour(timeout_behaviour)

        loop.call_later(timeout_s, add_timeout_behaviour)

        try:
            await asyncio.wait_for(goal_status.community_event.wait(), timeout=timeout_s + 5)
        except asyncio.TimeoutError:
            goal_status.continue_triggered = True

        # Merge collected community responses into a signifier-hints structure
        # and return it to the caller (the PlanningWorkflowBehaviour will
        # continue generation and prepare the final reply).
        self.matches = self._merge_community_responses(goal_status.community_responses)

    def _merge_community_responses(self, responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not responses:
            return {}

        out: Dict[str, Any] = {intent: {"matches": [], "final_matches": []} for intent in self.intent_strings}
        seen_ids: Dict[str, set] = {intent: set() for intent in self.intent_strings}
        seen_finals: Dict[str, set] = {intent: set() for intent in self.intent_strings}

        for entry in responses.values():
            response = entry.get("response") if isinstance(entry, dict) else None
            if not isinstance(response, dict):
                continue
            context = response.get("context") if isinstance(response.get("context"), dict) else {}

            for intent in self.intent_strings:
                payload = context.get(intent)
                if not isinstance(payload, dict):
                    continue

                matches = payload.get("matches") if isinstance(payload.get("matches"), list) else []
                finals = payload.get("final_matches") if isinstance(payload.get("final_matches"), list) else []

                for m in matches:
                    if not isinstance(m, dict):
                        continue
                    sid = m.get("signifier_id")
                    if sid in seen_ids[intent]:
                        continue
                    seen_ids[intent].add(sid)
                    m.setdefault("source", "community")
                    out[intent]["matches"].append(m)

                for f in finals:
                    sf = str(f)
                    if sf in seen_finals[intent]:
                        continue
                    seen_finals[intent].add(sf)
                    out[intent]["final_matches"].append(sf)

        return out
