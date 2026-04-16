"""Behaviour for handling GOAL_REQUEST messages from the UserAssistant."""

import json
import logging
from typing import List, Optional

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import (
    META_CORRELATION_ID,
    MessageType,
)
from ....shared.utils.demo_log import demo
from ...user_assistant.models import Intent

logger = logging.getLogger("InteractionSolver")


class GoalRequestBehaviour(CyclicBehaviour):
    """Parse GOAL_REQUEST payload, ensure env is ready, and reply with a plan."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return

        if msg.get_metadata("type") != MessageType.GOAL_REQUEST.value:
            return

        intent_objects: List[Intent] = []
        workspace_id: Optional[str] = None
        intent_type: Optional[str] = None
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                ws = payload.get("workspace_id") or payload.get("workspace")
                if ws:
                    workspace_id = str(ws)
                it = payload.get("intent_type")
                if it and str(it).upper() in ("EXPLICIT", "IMPLICIT"):
                    intent_type = str(it).upper()
            raw_intents = payload.get("intents")
            if isinstance(raw_intents, list):
                for item in raw_intents:
                    if isinstance(item, dict):
                        intent_objects.append(Intent.from_dict(item))
                    elif isinstance(item, str) and item.strip():
                        intent_objects.append(Intent(action="unknown", artifact="unknown", intent_text=item.strip()))
            else:
                goal_text = payload.get("intent") or payload.get("goal")
                if goal_text:
                    intent_objects.append(Intent(action="unknown", artifact="unknown", intent_text=str(goal_text).strip()))
        except json.JSONDecodeError:
            if msg.body:
                intent_objects.append(Intent(action="unknown", artifact="unknown", intent_text=str(msg.body).strip()))

        intent_objects = [i for i in intent_objects if i.to_query_string().strip()]
        if not intent_objects:
            await self._reply_missing_intent(msg)
            return

        intent_strings = [i.to_query_string() for i in intent_objects]
        logger.info(
            demo("Received GOAL_REQUEST: intents=%s workspace_id=%r from=%s"),
            intent_strings,
            workspace_id,
            str(msg.sender),
        )

        ready = await self.agent.await_environment_ready(
            timeout=self.agent.env_ready_timeout,
        )
        if not ready:
            await self._reply_env_not_ready(msg, intent_objects)
            return

        plan_json = await self.agent._generate_plan(
            intent_objects,
            workspace_id=workspace_id,
            intent_type=intent_type,
        )
        self._log_plan_summary(plan_json)
        await self._reply_plan(msg, plan_json)

    async def _reply_missing_intent(self, msg) -> None:
        reply = msg.make_reply()
        reply.set_metadata("type", MessageType.GOAL_RESPONSE.value)
        reply.body = json.dumps(
            {"plan_type": "behavior_tree", "error": "missing_intent", "tree": None, "intents": []}
        )
        self._copy_correlation(msg, reply)
        await self.send(reply)

    async def _reply_env_not_ready(self, msg, intent_objects: List[Intent]) -> None:
        reply = msg.make_reply()
        reply.set_metadata("type", MessageType.PLAN_CREATED.value)
        reply.body = json.dumps(
            {
                "plan_type": "behavior_tree",
                "error": "env_not_ready",
                "detail": "Environment discovery not completed (timeout).",
                "tree": None,
                "intents": [i.to_dict() for i in intent_objects],
            },
            indent=2,
        )
        self._copy_correlation(msg, reply)
        await self.send(reply)

    async def _reply_plan(self, msg, plan_json: str) -> None:
        reply = msg.make_reply()
        reply.set_metadata("type", MessageType.PLAN_CREATED.value)
        reply.body = plan_json
        self._copy_correlation(msg, reply)
        await self.send(reply)

    @staticmethod
    def _copy_correlation(src_msg, reply_msg) -> None:
        corr = src_msg.get_metadata(META_CORRELATION_ID)
        if corr:
            reply_msg.set_metadata(META_CORRELATION_ID, corr)
        if src_msg.thread:
            reply_msg.thread = src_msg.thread

    @staticmethod
    def _log_plan_summary(plan_json: str) -> None:
        try:
            parsed = json.loads(plan_json or "{}")
            if not isinstance(parsed, dict):
                logger.info(demo("Plan created (non-dict body)."))
                return
            tree = parsed.get("tree")
            has_tree = tree is not None and isinstance(tree, dict) and bool(tree)
            logger.info(
                demo("Plan created: has_tree=%s signifier_reuse=%s impossible=%s error=%s"),
                has_tree,
                parsed.get("signifier_reuse", False),
                parsed.get("impossible", False),
                parsed.get("error"),
            )
        except Exception:
            logger.info(demo("Plan created (failed to parse JSON for summary)."))
