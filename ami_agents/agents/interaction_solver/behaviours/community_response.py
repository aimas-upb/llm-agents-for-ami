"""Behaviour: handle COMMUNITY_RESPONSE and update goal status."""

import json
import time
from typing import Any, Dict, Optional

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import META_CORRELATION_ID, MessageType
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory


class CommunityResponseBehaviour(CyclicBehaviour):
    """Collect community responses and trigger planning continuation."""

    def __init__(self, logger=None) -> None:
        super().__init__()
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")

    async def run(self) -> None:
        msg = await self.receive(timeout=1)
        if not msg:
            return
        if msg.get_metadata("type") != MessageType.COMMUNITY_RESPONSE.value:
            return

        goal_id: Optional[str] = None
        response_data: Dict[str, Any] = {}
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                goal_id = payload.get("goal_id")
                response_data = payload
        except json.JSONDecodeError:
            pass

        sender_jid = str(msg.sender)

        self.logger.info(
            demo("Received COMMUNITY_RESPONSE: goal_id=%s from=%s"),
            goal_id or "missing",
            sender_jid,
        )

        if not goal_id:
            self.logger.warning("COMMUNITY_RESPONSE missing goal_id from=%s", sender_jid)
            return

        goal_status = self.agent.goal_statuses.get(str(goal_id))
        if not goal_status:
            self.logger.warning(
                "COMMUNITY_RESPONSE for unknown goal_id=%s from=%s",
                goal_id,
                sender_jid,
            )
            return

        entry = {
            "agent_jid": sender_jid,
            "response": response_data,
            "received_at": time.monotonic(),
        }
        goal_status.community_responses[sender_jid] = entry

        self.logger.info(
            demo("COMMUNITY_RESPONSE appended: goal_id=%s from=%s total_responses=%d"),
            goal_id,
            sender_jid,
            len(goal_status.community_responses),
        )

        if goal_status.community_expected_responses > 0:
            received_count = len(goal_status.community_responses)
            expected_count = goal_status.community_expected_responses
            response_ratio = received_count / expected_count

            if response_ratio >= self.agent.community_min_response_ratio:
                goal_status.continue_triggered = True
                goal_status.community_event.set()
                self.logger.info(
                    demo("Community response ratio reached: goal_id=%s ratio=%.2f"),
                    goal_id,
                    response_ratio,
                )
