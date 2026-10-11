"""Behaviour: find a remembered goal execution by its exact key.

The InteractionSolver asks `{"kind": ..., "key": {...}}` (the key from
`shared/utils/goal_signifiers.py`); the answer is the latest record stored
under exactly that key, or none. No similarity, no scoring.
"""

import json
from typing import Any, Dict

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import META_CORRELATION_ID, MessageType
from ....shared.utils.demo_log import demo
from ..experience import ensure_experience_engine_ready


class SignifierMatchBehaviour(CyclicBehaviour):
    """Handle SIGNIFIER_MATCH_REQUEST: a key in, `{"match": record | null}` out."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg or msg.get_metadata("type") != MessageType.SIGNIFIER_MATCH_REQUEST.value:
            return
        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}

        response = await self.match(payload)

        reply = msg.make_reply()
        reply.body = json.dumps(response)
        reply.set_metadata("type", MessageType.SIGNIFIER_MATCH_RESPONSE.value)
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, str(correlation_id))
        if msg.thread:
            reply.thread = msg.thread
        await self.send(reply)

    async def match(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """The record under the payload's kind and key, or `match: None`."""
        kind = payload.get("kind") if isinstance(payload, dict) else None
        key = payload.get("key") if isinstance(payload, dict) else None
        if not isinstance(kind, str) or not isinstance(key, dict):
            return {"ok": False, "error": "malformed_payload", "match": None,
                    "detail": "expected {\"kind\": str, \"key\": {...}}"}
        if not await ensure_experience_engine_ready(self.agent):
            return {"ok": False, "error": "experience_engine_not_ready", "match": None}
        record = self.agent._experience_engine_store.match(kind, key)
        self.agent.logger.info(demo(
            f"[MEMORY] {kind} match: "
            + (f"{record.signifier_id} ({record.intent_text!r})" if record else "none")))
        return {"ok": True, "match": record.to_json_doc() if record else None}
