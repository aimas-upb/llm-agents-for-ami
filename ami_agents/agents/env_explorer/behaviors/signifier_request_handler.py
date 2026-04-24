"""
Signifier request handler behavior for EnvExplorer agent.
"""

import json
from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo


class SignifierRequestHandler(CyclicBehaviour):
    """Handler for RD4 signifier engine requests (embedded in EnvExplorer)."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return

        # DEBUG: Log ANY message received
        print(f"[SIGNIFIER_HANDLER_DEBUG] Received message type: {msg.get_metadata('type')}")

        msg_type = msg.get_metadata("type")

        if msg_type == MessageType.SIGNIFIER_LIST_REQUEST.value:
            self.agent.logger.info(demo("Received SIGNIFIER_LIST_REQUEST from %s"), str(msg.sender))
            response_payload = await self.agent._rd4_list_signifiers()
            reply_type = MessageType.SIGNIFIER_LIST_RESPONSE.value

        elif msg_type == MessageType.SIGNIFIER_MATCH_REQUEST.value:
            print(f"[SIGNIFIER_MATCH_DEBUG] Received SIGNIFIER_MATCH_REQUEST from {msg.sender}")
            print(f"[BRANCH_DEBUG] Entering SIGNIFIER_MATCH_REQUEST branch!")
            self.agent.logger.info(demo("Received SIGNIFIER_MATCH_REQUEST from %s"), str(msg.sender))
            try:
                payload = json.loads(msg.body or "{}")
                print(f"[PAYLOAD_DEBUG] Parsed payload: {payload}")
            except json.JSONDecodeError:
                payload = {}
                print("[PAYLOAD_DEBUG] JSON decode failed, using empty payload")

            intent = payload.get("intent") or payload.get("query") or payload.get("goal") or ""
            workspace_id = payload.get("workspace_id")
            k = payload.get("k", 10)
            matcher_version = payload.get("matcher_version")
            min_similarity = payload.get("min_similarity")
            intent_type = payload.get("intent_type")  # Extract intent_type for filtering/ranking
            query_structured_intent = payload.get("query_structured_intent")  # Extract structured_intent for v2 matcher

            # DEBUG: Log structured intent info
            print(f"[DEBUG_FINAL] query_structured_intent={query_structured_intent}")
            print(f"[DEBUG_FINAL] matcher_version={matcher_version}")
            self.agent.logger.info(demo("DEBUG: query_structured_intent=%s"), query_structured_intent)
            self.agent.logger.info(demo("DEBUG: matcher_version=%s"), matcher_version)

            response_payload = await self.agent._rd4_match_signifiers(
                intent=str(intent),
                workspace_id=str(workspace_id) if workspace_id else None,
                k=int(k) if str(k).isdigit() else 10,
                matcher_version=str(matcher_version) if matcher_version else None,
                min_similarity=float(min_similarity) if min_similarity is not None else None,
                intent_type=str(intent_type).upper() if intent_type and str(intent_type).upper() in ("EXPLICIT", "IMPLICIT") else None,
                query_structured_intent=query_structured_intent if isinstance(query_structured_intent, dict) else None,
            )
            reply_type = MessageType.SIGNIFIER_MATCH_RESPONSE.value

        elif msg_type == MessageType.SIGNIFIER_RECORD_EXECUTION_REQUEST.value:
            self.agent.logger.info(demo("Received SIGNIFIER_RECORD_EXECUTION_REQUEST from %s"), str(msg.sender))
            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                payload = {}

            try:
                response_payload = await self.agent._rd4_record_execution(
                    plan=payload.get("plan"),
                    execution_report=payload.get("execution_report") or payload.get("execution") or {},
                    sender=str(msg.sender) if getattr(msg, "sender", None) else None,
                    thread=str(msg.thread) if getattr(msg, "thread", None) else None,
                    workspace_id=payload.get("workspace_id"),
                    signifiers=payload.get("signifiers"),  # NEW: accept pre-extracted signifiers from BT
                )
            except Exception as e:
                self.agent.logger.error(
                    demo("!!! EXCEPTION in _rd4_record_execution: %s - %s"),
                    type(e).__name__,
                    str(e),
                    exc_info=True,
                )
                response_payload = {"ok": False, "error": "exception", "detail": str(e)}
            reply_type = MessageType.SIGNIFIER_RECORD_EXECUTION_RESPONSE.value

        else:
            return

        reply = msg.make_reply()
        reply.body = json.dumps(response_payload)
        reply.set_metadata("type", reply_type)

        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, str(correlation_id))
        if msg.thread:
            reply.thread = msg.thread

        await self.send(reply)