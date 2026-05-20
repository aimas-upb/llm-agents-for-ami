"""Behaviour: handle GOAL_REQUEST messages from the UserAssistant.

Thin entry point. Parses the request, waits for environment readiness, and
delegates the actual planning to ``PlanningWorkflowBehaviour`` (which in
turn spawns the per-stage sub-behaviours: env context query, signifier
matching, community lookup, BT plan generation).
"""

import json
import uuid
from typing import List, Optional

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import META_CORRELATION_ID, MessageType
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ...user_assistant.models import Intent
from ..utils.plan_envelope import (
    envelope_error,
    envelope_missing_intents,
)
from ..utils.goal_status import GoalStatus, PlanningPhase
from .planning_workflow import PlanningWorkflowBehaviour


class GoalRequestBehaviour(CyclicBehaviour):
    """Receive GOAL_REQUEST, validate, dispatch the planning workflow."""

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return
        if msg.get_metadata("type") != MessageType.GOAL_REQUEST.value:
            return

        intent_objects, workspace_id, intent_type, goal_id = self._parse_request(msg)

        intent_objects = [i for i in intent_objects if i.to_query_string().strip()]
        if not intent_objects:
            await self._reply(
                msg,
                MessageType.GOAL_RESPONSE.value,
                envelope_missing_intents(intent_type, goal_id=goal_id),
            )
            return

        intent_strings = [i.to_query_string() for i in intent_objects]
        self.logger.info(
            demo("Received GOAL_REQUEST: intents=%s workspace_id=%r from=%s"),
            intent_strings,
            workspace_id,
            str(msg.sender),
        )

        ready = await self.agent.await_environment_ready(
            timeout=self.agent.env_ready_timeout,
        )
        if not ready:
            envelope = envelope_error(
                "env_not_ready",
                "Environment discovery not completed (timeout).",
                intent_objects,
                intent_type,
                goal_id=goal_id,
            )
            await self._reply(msg, MessageType.PLAN_CREATED.value, envelope)
            return

        intent_strings = [i.to_query_string() for i in intent_objects]
        if not goal_id:
            goal_id = str(uuid.uuid4())

        goal_status = GoalStatus(
            goal_id=goal_id,
            intents=intent_strings,
            workspace_id=workspace_id,
        )
        goal_status.update_status(phase=PlanningPhase.INITIATED)
        self.agent.goal_statuses[goal_id] = goal_status

        workflow = PlanningWorkflowBehaviour(
            intents=intent_objects,
            workspace_id=workspace_id,
            intent_type=intent_type,
            logger=self.logger,
            goal_status=goal_status,
        )
        self.agent.add_behaviour(workflow)
        await workflow.join()

        envelope = workflow.reply_envelope or envelope_error(
            "plan_generation_failed",
            "Planning workflow returned no envelope.",
            intent_objects,
            intent_type,
            goal_id=goal_id,
        )
        self._log_plan_summary(envelope)
        await self._reply(msg, MessageType.PLAN_CREATED.value, envelope)

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_request(msg):
        """Extract intents, workspace, and intent_type from the GOAL_REQUEST body."""
        intent_objects: List[Intent] = []
        workspace_id: Optional[str] = None
        intent_type: Optional[str] = None
        goal_id: Optional[str] = None
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                ws = payload.get("workspace_id") or payload.get("workspace")
                if ws:
                    workspace_id = str(ws)
                it = payload.get("intent_type")
                if it and str(it).upper() in ("EXPLICIT", "IMPLICIT"):
                    intent_type = str(it).upper()
                gid = payload.get("goal_id")
                if gid:
                    goal_id = str(gid)
            raw_intents = payload.get("intents") if isinstance(payload, dict) else None
            if isinstance(raw_intents, list):
                for item in raw_intents:
                    if isinstance(item, dict):
                        intent_objects.append(Intent.from_dict(item))
                    elif isinstance(item, str) and item.strip():
                        intent_objects.append(
                            Intent(
                                action="unknown",
                                artifact="unknown",
                                intent_text=item.strip(),
                            )
                        )
            else:
                goal_text = (
                    (payload.get("intent") or payload.get("goal"))
                    if isinstance(payload, dict)
                    else None
                )
                if goal_text:
                    intent_objects.append(
                        Intent(
                            action="unknown",
                            artifact="unknown",
                            intent_text=str(goal_text).strip(),
                        )
                    )
        except json.JSONDecodeError:
            if msg.body:
                intent_objects.append(
                    Intent(
                        action="unknown",
                        artifact="unknown",
                        intent_text=str(msg.body).strip(),
                    )
                )
        return intent_objects, workspace_id, intent_type, goal_id

    async def _reply(self, src_msg, reply_type: str, envelope: dict) -> None:
        reply = src_msg.make_reply()
        reply.set_metadata("type", reply_type)
        reply.body = json.dumps(envelope, indent=2)
        corr = src_msg.get_metadata(META_CORRELATION_ID)
        if corr:
            reply.set_metadata(META_CORRELATION_ID, corr)
        if src_msg.thread:
            reply.thread = src_msg.thread
        await self.send(reply)

    def _log_plan_summary(self, envelope: dict) -> None:
        if not isinstance(envelope, dict):
            self.logger.info(demo("Plan created (non-dict body)."))
            return
        tree = envelope.get("tree")
        has_tree = tree is not None and isinstance(tree, dict) and bool(tree)
        self.logger.info(
            demo("Plan created: has_tree=%s signifier_reuse=%s impossible=%s error=%s"),
            has_tree,
            envelope.get("signifier_reuse", False),
            envelope.get("impossible", False),
            envelope.get("error"),
        )
