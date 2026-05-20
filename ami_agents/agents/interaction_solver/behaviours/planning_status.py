"""Behaviour: handle PLANNING_STATUS_REQUEST messages."""

import json
from typing import Optional

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import META_CORRELATION_ID, MessageType
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory


class PlanningStatusBehaviour(CyclicBehaviour):
    """Respond with planning status for a given goal_id."""

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return
        if msg.get_metadata("type") != MessageType.PLANNING_STATUS_REQUEST.value:
            return

        goal_id: Optional[str] = None
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                raw_goal_id = payload.get("goal_id")
                if raw_goal_id:
                    goal_id = str(raw_goal_id)
        except json.JSONDecodeError:
            pass

        self.logger.info(
            demo("Received PLANNING_STATUS_REQUEST: goal_id=%s from=%s"),
            goal_id or "missing",
            str(msg.sender),
        )

        reply = msg.make_reply()
        reply.set_metadata("type", MessageType.PLANNING_STATUS_RESPONSE.value)
        corr = msg.get_metadata(META_CORRELATION_ID)
        if corr:
            reply.set_metadata(META_CORRELATION_ID, corr)
        if msg.thread:
            reply.thread = msg.thread

        if not goal_id:
            reply.body = json.dumps(
                {
                    "error": "missing_goal_id",
                    "detail": "No goal_id provided in request",
                }
            )
            await self.send(reply)
            return

        goal_status = self.agent.goal_statuses.get(str(goal_id))
        if not goal_status:
            reply.body = json.dumps(
                {
                    "error": "goal_not_found",
                    "detail": f"No goal found with id: {goal_id}",
                    "goal_id": str(goal_id),
                }
            )
            await self.send(reply)
            return

        response = {
            "goal_id": goal_status.goal_id,
            "phase": goal_status.phase.value,
            "intents": goal_status.intents,
            "workspace_id": goal_status.workspace_id,
            "created_at": goal_status.created_at,
            "last_updated": goal_status.last_updated,
            "elapsed_time": goal_status.last_updated - goal_status.created_at,
        }

        if goal_status.reused_plan:
            response["has_reused_plan"] = True
            response["reused_plan_source"] = goal_status.reused_plan_source

        if goal_status.local_plan:
            response["has_local_plan"] = True
            response["local_plan_complete"] = goal_status.local_plan_complete

        if goal_status.relevant_communities:
            response["relevant_communities"] = goal_status.relevant_communities

        if goal_status.community_responses:
            response["community_responses_count"] = len(goal_status.community_responses)

        if goal_status.error:
            response["error"] = goal_status.error
            response["error_detail"] = goal_status.error_detail

        self.logger.info(
            demo("PLANNING_STATUS_RESPONSE: goal_id=%s phase=%s best_plan_source=%s"),
            goal_id,
            goal_status.phase.value,
            goal_status.best_plan_source or "none",
        )

        reply.body = json.dumps(response, indent=2)
        await self.send(reply)
