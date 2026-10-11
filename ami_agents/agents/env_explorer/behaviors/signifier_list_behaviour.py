"""Behaviour: list the remembered goal executions (newest first)."""

import json

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import META_CORRELATION_ID, MessageType
from ..experience import list_experience_engine_signifiers


class SignifierListBehaviour(CyclicBehaviour):
    """Handle SIGNIFIER_LIST_REQUEST."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg or msg.get_metadata("type") != MessageType.SIGNIFIER_LIST_REQUEST.value:
            return

        response = await list_experience_engine_signifiers(self.agent)

        reply = msg.make_reply()
        reply.body = json.dumps(response)
        reply.set_metadata("type", MessageType.SIGNIFIER_LIST_RESPONSE.value)
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, str(correlation_id))
        if msg.thread:
            reply.thread = msg.thread
        await self.send(reply)
