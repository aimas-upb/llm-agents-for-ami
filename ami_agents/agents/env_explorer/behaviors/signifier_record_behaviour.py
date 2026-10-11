"""Behaviour: record a successful goal execution in the memory engine.

The UserAssistant sends one goal signifier per successful plan entry (built by
`user_assistant/signifiers.py`, keyed by `shared/utils/goal_signifiers.py`):
`{"signifier": {kind, key, intent_text, goal, tree | context + used, ...}}`.
It is stored as it is; a later record under the same key supersedes it.
"""

import json
from typing import Any, Dict

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import META_CORRELATION_ID, MessageType
from ....shared.utils.demo_log import demo
from ..experience import ensure_experience_engine_ready


class SignifierRecordBehaviour(CyclicBehaviour):
    """Handle SIGNIFIER_RECORD_EXECUTION_REQUEST: one record in, its id out."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg or msg.get_metadata("type") != MessageType.SIGNIFIER_RECORD_EXECUTION_REQUEST.value:
            return
        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}

        response = await self.record(payload)

        reply = msg.make_reply()
        reply.body = json.dumps(response)
        reply.set_metadata("type", MessageType.SIGNIFIER_RECORD_EXECUTION_RESPONSE.value)
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, str(correlation_id))
        if msg.thread:
            reply.thread = msg.thread
        await self.send(reply)

    async def record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Store the payload's signifier; the response says how it went."""
        signifier = payload.get("signifier") if isinstance(payload, dict) else None
        if not isinstance(signifier, dict):
            return {"ok": False, "error": "malformed_payload",
                    "detail": "expected {\"signifier\": {...}}"}
        if not await ensure_experience_engine_ready(self.agent):
            return {"ok": False, "error": "experience_engine_not_ready"}
        try:
            record = self.agent._experience_engine_store.record(signifier)
        except Exception as exc:  # noqa: BLE001 - reported to the sender
            self.agent.logger.warning(f"Signifier not recorded: {exc}")
            return {"ok": False, "error": "invalid_signifier", "detail": str(exc)}
        self.agent.logger.info(demo(
            f"[MEMORY] Recorded {record.kind} signifier {record.signifier_id} "
            f"for {record.intent_text!r}"))
        return {"ok": True, "signifier_id": record.signifier_id,
                "key_digest": record.key_digest}
