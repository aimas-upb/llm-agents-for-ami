"""Behaviour: handle GOAL_REQUEST messages from the UserAssistant.

Thin entry point. Parses the request, waits for environment readiness, and
delegates the actual planning to ``PlanningWorkflowBehaviour`` (which in
turn spawns the per-stage sub-behaviours: env context query, signifier
matching, community lookup, BT plan generation).
"""

import json
from typing import List, Optional, Union

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import META_CORRELATION_ID, MessageType
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ....shared.models.intents import ImplicitGoalIntent, ExplicitGoalIntent, goal_intent_from_dict
from ...user_assistant.models import Intent
from ..utils.plan_envelope import envelope_error
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

        intent_objects, workspace_id = self._parse_request(msg)

        intent_objects = [i for i in intent_objects if i.to_query_string().strip()]

        intent_strings = [i.to_query_string() for i in intent_objects]
        self.logger.info(
            demo(f"Received GOAL_REQUEST: intents={intent_strings} workspace_id={workspace_id!r} from={msg.sender}")
        )

        ready = await self.agent.await_environment_ready(
            timeout=self.agent.env_ready_timeout,
        )
        if not ready:
            envelope = envelope_error(
                "env_not_ready",
                "Environment discovery not completed (timeout).",
                intent_objects,
            )
            await self._reply(msg, MessageType.PLAN_CREATED.value, envelope)
            return

        workflow = PlanningWorkflowBehaviour(
            intents=intent_objects,
            workspace_id=workspace_id,
            logger=self.logger,
        )
        self.agent.add_behaviour(workflow)
        await workflow.join()

        envelope = workflow.reply_envelope or envelope_error(
            "plan_generation_failed",
            "Planning workflow returned no envelope.",
            intent_objects,
        )
        self._log_plan_summary(envelope)
        await self._reply(msg, MessageType.PLAN_CREATED.value, envelope)

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_request(msg):
        """Extract intents and workspace from the GOAL_REQUEST body.

        Each intent carries its own type (implicit/explicit) via its dataclass.
        """
        intent_objects: List[Union[ImplicitGoalIntent, ExplicitGoalIntent, Intent]] = []
        workspace_id: Optional[str] = None
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                ws = payload.get("workspace_id") or payload.get("workspace")
                if ws:
                    workspace_id = str(ws)
            raw_intents = payload.get("intents") if isinstance(payload, dict) else None
            if isinstance(raw_intents, list):
                for item in raw_intents:
                    if isinstance(item, dict):
                        # Check if this is a goal intent with category field (new format)
                        if item.get("category") in ("implicit", "explicit"):
                            intent_objects.append(goal_intent_from_dict(item))
                        else:
                            # Fallback for other intent types
                            intent_objects.append(Intent(intent_text=item.get("text_intent", str(item))))
                    elif isinstance(item, str) and item.strip():
                        intent_objects.append(Intent(intent_text=item.strip()))
            else:
                goal_text = (
                    (payload.get("intent") or payload.get("goal"))
                    if isinstance(payload, dict)
                    else None
                )
                if goal_text:
                    intent_objects.append(Intent(intent_text=str(goal_text).strip()))
        except json.JSONDecodeError:
            if msg.body:
                intent_objects.append(Intent(intent_text=str(msg.body).strip()))
        return intent_objects, workspace_id

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

        # Handle both single-tree format (top-level "tree") and multi-plan format (plans list)
        tree = envelope.get("tree")
        has_tree = tree is not None and isinstance(tree, dict) and bool(tree)

        # Check for multi-plan format
        plans = envelope.get("plans")
        if isinstance(plans, list) and not has_tree:
            # Multi-plan format - check if any plan has a tree
            has_tree = any(p.get("tree") for p in plans if isinstance(p, dict))

        self.logger.info(
            demo(f"Plan created: has_tree={has_tree} signifier_reuse={envelope.get('signifier_reuse', False)} impossible={envelope.get('impossible', False)} error={envelope.get('error')}")
        )
