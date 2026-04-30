import json
from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo
from ..utils.data_formatting import format_capabilities_payload


class EnvironmentCapabilitiesBehaviour(CyclicBehaviour):
    """
    Handles ENV_CAPABILITIES_REQUEST messages and provides comprehensive
    information about the agent's capabilities, workspaces, and affordances.
    """

    async def run(self):
        """Main behavior loop - handles environment capabilities requests."""
        msg = await self.receive(timeout=1)
        if not msg:
            return

        # Only process ENV_CAPABILITIES_REQUEST messages
        msg_type = msg.get_metadata("type")
        if msg_type != MessageType.ENV_CAPABILITIES_REQUEST.value:
            return

        self.agent.logger.info(demo("Received ENV_CAPABILITIES_REQUEST from %s"), str(msg.sender))

        # Generate capabilities response payload
        response_payload = self._generate_capabilities_payload()

        # Send response
        reply = msg.make_reply()
        reply.body = json.dumps(response_payload)
        reply.set_metadata("type", MessageType.ENV_CAPABILITIES_RESPONSE.value)

        # Preserve correlation ID and thread
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, correlation_id)
        if msg.thread:
            reply.thread = msg.thread

        await self.send(reply)

        self.agent.logger.debug(f"Sent capabilities response with {len(response_payload.get('workspaces', {}))} workspaces")

    def _generate_capabilities_payload(self) -> dict:
        """
        Generate comprehensive capabilities payload.

        Returns a machine-readable JSON payload containing information about
        the agent's workspaces, artifacts, affordances, and capabilities.
        """
        try:
            # Generate capabilities payload using utility function directly
            # This follows SPADE principles by avoiding thin wrapper methods
            return format_capabilities_payload(self.agent)
        except Exception as e:
            self.agent.logger.error(f"Error generating capabilities payload: {e}", exc_info=True)
            return {
                "error": "capabilities_generation_failed",
                "detail": str(e),
                "workspaces": {},
                "summary": "Capabilities generation failed due to internal error"
            }