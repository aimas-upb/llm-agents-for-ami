"""Behaviour: find a remembered condition-guard decision by its exact key.

The InteractionSolver asks `{"key": {...}}` (the key from
`interaction_solver/utils/planning_conditions.guard_key`) just before it would
ask the condition-guard model; the answer is the latest decision stored under
exactly that key in the guard store, or none. The store decides nothing: the
InteractionSolver resolves a hit again on today's TDs and against its plan.

Mirrors `SignifierMatchBehaviour`, and shares nothing with it: its own
message types, its own store.
"""

import json
from typing import Any, Dict

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import META_CORRELATION_ID, MessageType
from ....shared.utils.demo_log import demo
from ..experience import ensure_experience_engine_ready


class GuardMatchBehaviour(CyclicBehaviour):
    """Handle GUARD_MATCH_REQUEST: a key in, `{"match": record | null}` out."""

    async def run(self):
        """One request: parse, look up, reply on the request's thread with its
        correlation id."""
        # 1. The request; anything else on this template is ignored.
        msg = await self.receive(timeout=1)
        if not msg or msg.get_metadata("type") != MessageType.GUARD_MATCH_REQUEST.value:
            return
        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}

        # 2. The answer, then the reply.
        response = await self.match(payload)
        reply = msg.make_reply()
        reply.body = json.dumps(response)
        reply.set_metadata("type", MessageType.GUARD_MATCH_RESPONSE.value)
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, str(correlation_id))
        if msg.thread:
            reply.thread = msg.thread
        await self.send(reply)

    async def match(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """The decision under the payload's key, or `match: None`."""
        key = payload.get("key") if isinstance(payload, dict) else None
        if not isinstance(key, dict):
            return {"ok": False, "error": "malformed_payload", "match": None,
                    "detail": "expected {\"key\": {...}}"}
        if not await ensure_experience_engine_ready(self.agent):
            return {"ok": False, "error": "experience_engine_not_ready", "match": None}
        record = self.agent._experience_engine_guard_store.match(key)
        self.agent.logger.info(demo(
            f"[MEMORY] guard match for {key.get('device_name')}.{key.get('affordance_name')}: "
            + (record["guard_id"] if record else "none")))
        return {"ok": True, "match": record}
