"""Behaviour: respond to COMMUNITY_REQUEST with local signifier matches."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import META_CORRELATION_ID, MessageType
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..utils.signifier_matching import query_local_signifier_match


class CommunityRequestBehaviour(CyclicBehaviour):
    """Handle COMMUNITY_REQUEST and reply with local signifier matches."""

    def __init__(self, logger=None) -> None:
        super().__init__()
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")

    async def run(self) -> None:
        msg = await self.receive(timeout=1)
        if not msg:
            return
        if msg.get_metadata("type") != MessageType.COMMUNITY_REQUEST.value:
            return

        if not getattr(self.agent, "community_enabled", False):
            self.logger.info(
                demo("Ignoring COMMUNITY_REQUEST because community is disabled on %s"),
                str(self.agent.jid),
            )
            return

        intents: List[str] = []
        workspace_id: Optional[str] = None
        goal_id: Optional[str] = None
        intent_type: Optional[str] = None
        structured_intents: Optional[List[dict]] = None

        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                raw_intents = payload.get("intents")
                if isinstance(raw_intents, list):
                    intents = [str(i).strip() for i in raw_intents if str(i).strip()]
                ws = payload.get("workspace_id")
                if ws:
                    workspace_id = str(ws)
                gid = payload.get("goal_id")
                if gid:
                    goal_id = str(gid)
                it = payload.get("intent_type")
                if it and str(it).upper() in ("EXPLICIT", "IMPLICIT"):
                    intent_type = str(it).upper()
                structured = payload.get("structured_intents")
                if isinstance(structured, list):
                    structured_intents = structured
        except json.JSONDecodeError:
            pass

        self.logger.info(
            demo("Received COMMUNITY_REQUEST: goal_id=%s intents=%s workspace_id=%r from=%s"),
            goal_id,
            intents,
            workspace_id,
            str(msg.sender),
        )

        reply = msg.make_reply()
        reply.set_metadata("type", MessageType.COMMUNITY_RESPONSE.value)
        corr = msg.get_metadata(META_CORRELATION_ID)
        if corr:
            reply.set_metadata(META_CORRELATION_ID, corr)
        if msg.thread:
            reply.thread = msg.thread

        if not intents:
            reply.body = json.dumps(
                {
                    "error": "missing_intents",
                    "detail": "No intents provided in community_request",
                    "goal_id": goal_id,
                }
            )
            await self.send(reply)
            return

        structured_by_intent: Dict[str, dict] = {}
        if structured_intents and len(structured_intents) == len(intents):
            for intent_str, s in zip(intents, structured_intents):
                if isinstance(s, dict):
                    structured_by_intent[intent_str] = s

        results = await asyncio.gather(
            *[
                query_local_signifier_match(
                    self.agent._query_env_explorer,
                    intent_str,
                    workspace_id,
                    intent_type,
                    structured_by_intent.get(intent_str),
                )
                for intent_str in intents
            ],
            return_exceptions=True,
        )

        matches: Dict[str, Any] = {}
        for intent_str, res in zip(intents, results):
            if isinstance(res, Exception):
                matches[intent_str] = {"error": "signifier_query_failed", "detail": str(res)}
            else:
                matches[intent_str] = res if isinstance(res, dict) else {}

        response_data = {
            "responding_agent": str(self.agent.jid),
            "context": matches,
            "goal_id": goal_id,
        }

        self.logger.info(
            demo("COMMUNITY_RESPONSE sent: goal_id=%s intents=%s to=%s"),
            goal_id or "N/A",
            intents,
            str(msg.sender),
        )

        reply.body = json.dumps(response_data, indent=2)
        await self.send(reply)
