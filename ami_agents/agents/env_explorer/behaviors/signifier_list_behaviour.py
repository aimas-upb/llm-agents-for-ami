import json
from typing import Any, Dict
from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo
from ..experience import ensure_experience_engine_ready


class SignifierListBehaviour(CyclicBehaviour):
    """
    Contains all the business logic for listing stored signifiers from the Experience engine registry,
    with proper formatting and error handling.
    """

    async def run(self):
        """Main behavior loop - handles signifier list requests."""
        msg = await self.receive(timeout=1)
        if not msg:
            return

        # Only process SIGNIFIER_LIST_REQUEST messages
        msg_type = msg.get_metadata("type")
        if msg_type != MessageType.SIGNIFIER_LIST_REQUEST.value:
            return

        self.agent.logger.debug(f"SignifierListBehaviour received request from {msg.sender}")

        # Process signifier listing
        response_payload = await self._list_signifiers()

        # Send response
        reply = msg.make_reply()
        reply.body = json.dumps(response_payload)
        reply.set_metadata("type", MessageType.SIGNIFIER_LIST_RESPONSE.value)

        # Preserve correlation ID and thread
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, str(correlation_id))
        if msg.thread:
            reply.thread = msg.thread

        await self.send(reply)

    async def _list_signifiers(self) -> Dict[str, Any]:
        """
        List all stored signifiers - complete business logic.

        This contains all the listing logic that was previously in the agent
        and utility functions, now properly encapsulated in a behavior.
        """
        # Ensure Experience engine is ready
        if not self.agent._experience_engine_ready:
            await ensure_experience_engine_ready(self.agent)

        if not self.agent._experience_engine_registry:
            return {"ok": False, "error": "experience_engine_not_ready"}

        try:
            # Get signifier limit from configuration
            experience_engine_config = self.agent.config.get("experience_engine", {})
            signifier_limit = experience_engine_config.get("signifier_limit", 1000)  # Default from _EXPERIENCE_ENGINE_DEFAULTS

            self.agent.logger.debug(f"Listing signifiers with limit: {signifier_limit}")

            # Get raw signifiers from registry
            signifiers_raw = self.agent._experience_engine_registry.list_signifiers(limit=signifier_limit)

            # Transform to simplified output format
            signifiers_out = []
            for s in signifiers_raw:
                signifier_data = {
                    "signifier_id": s.signifier_id,
                    "version": s.version,
                    "status": getattr(s.status, "value", str(s.status)),
                    "intent": getattr(s.intent, "nl_text", ""),
                    "affordance_uri": s.affordance_uri,
                }

                # Include additional fields if available
                if hasattr(s, 'intent_type') and s.intent_type:
                    signifier_data["intent_type"] = s.intent_type

                if hasattr(s, 'provenance') and s.provenance:
                    signifier_data["created_by"] = getattr(s.provenance, 'created_by', '')
                    signifier_data["source"] = getattr(s.provenance, 'source', '')

                signifiers_out.append(signifier_data)

            self.agent.logger.info(demo("Listed %d signifiers from registry"), len(signifiers_out))

            return {
                "total": len(signifiers_out),
                "signifiers": signifiers_out,
                "limit": signifier_limit,
                "ok": True
            }

        except Exception as e:
            self.agent.logger.error(f"Error listing signifiers: {e}", exc_info=True)
            return {"ok": False, "error": "exception", "detail": str(e)}