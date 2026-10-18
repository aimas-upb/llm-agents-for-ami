"""Behaviour: record a condition-guard decision in the guard store.

The UserAssistant sends one decision per guard the model made for a plan that
ran successfully (`user_assistant/guards.py`):
`{"guard": {key, protection, check?, action?, why?}}`. It is stored as it is
(`GuardStore.record` validates its shape); a later decision under the same key
supersedes it.

Mirrors `SignifierRecordBehaviour`, and shares nothing with it.
"""

import json
from typing import Any, Dict

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import META_CORRELATION_ID, MessageType
from ....shared.utils.demo_log import demo
from ..experience import ensure_experience_engine_ready


class GuardRecordBehaviour(CyclicBehaviour):
    """Handle GUARD_RECORD_REQUEST: one decision in, its id out."""

    async def run(self):
        """One request: parse, store, reply on the request's thread with its
        correlation id."""
        # 1. The request; anything else on this template is ignored.
        msg = await self.receive(timeout=1)
        if not msg or msg.get_metadata("type") != MessageType.GUARD_RECORD_REQUEST.value:
            return
        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}

        # 2. The outcome, then the reply.
        response = await self.record(payload)
        reply = msg.make_reply()
        reply.body = json.dumps(response)
        reply.set_metadata("type", MessageType.GUARD_RECORD_RESPONSE.value)
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, str(correlation_id))
        if msg.thread:
            reply.thread = msg.thread
        await self.send(reply)

    async def record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Store the payload's decision; the response says how it went."""
        guard = payload.get("guard") if isinstance(payload, dict) else None
        if not isinstance(guard, dict):
            return {"ok": False, "error": "malformed_payload",
                    "detail": "expected {\"guard\": {...}}"}
        if not await ensure_experience_engine_ready(self.agent):
            return {"ok": False, "error": "experience_engine_not_ready"}
        try:
            record = self.agent._experience_engine_guard_store.record(guard)
        except Exception as exc:  # noqa: BLE001 - reported to the sender
            self.agent.logger.warning(f"Guard not recorded: {exc}")
            return {"ok": False, "error": "invalid_guard", "detail": str(exc)}
        key = record["key"]
        self.agent.logger.info(demo(
            f"[MEMORY] Recorded guard {record['guard_id']} for "
            f"{key.get('device_name')}.{key.get('affordance_name')} "
            f"(protection={record['protection']})"))
        return {"ok": True, "guard_id": record["guard_id"], "key_digest": record["key_digest"]}
