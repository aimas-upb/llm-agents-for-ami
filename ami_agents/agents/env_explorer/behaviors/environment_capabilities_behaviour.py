import json
import os
from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo
from ..utils.data_formatting import (
    format_capabilities_payload,
    format_capabilities_summary_hierarchical,
    format_capabilities_detailed_rdf,
)


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

        self.agent.logger.info(demo(f"Received ENV_CAPABILITIES_REQUEST from {msg.sender}"))

        # Parse request body to extract detail_level
        detail_level = "summary"  # default
        try:
            if msg.body:
                body = json.loads(msg.body)
                detail_level = body.get("detail_level", "summary")
        except (json.JSONDecodeError, AttributeError):
            pass

        # Generate capabilities response payload
        response_payload = self._generate_capabilities_payload(detail_level)
        ## Log the response payload for debugging (can be removed in production)
        # self.agent.logger.info(demo(f"Generated capabilities payload: {response_payload}"))

        # Send response
        reply = msg.make_reply()
        # For detailed RDF response, payload is a string; otherwise a dict
        if isinstance(response_payload, dict):
            reply.body = json.dumps(response_payload)
        else:
            # detailed RDF - wrap in JSON with detail_level indicator
            reply.body = json.dumps({"detail_level": detail_level, "payload": response_payload})
        reply.set_metadata("type", MessageType.ENV_CAPABILITIES_RESPONSE.value)

        # Preserve correlation ID and thread
        correlation_id = msg.get_metadata(META_CORRELATION_ID)
        if correlation_id:
            reply.set_metadata(META_CORRELATION_ID, correlation_id)
        if msg.thread:
            reply.thread = msg.thread

        await self.send(reply)

        workspace_count = len(response_payload.get('workspaces', {})) if isinstance(response_payload, dict) else 0
        self.agent.logger.info(demo(f"Sent capabilities response with {workspace_count} workspaces"))

    def _generate_capabilities_payload(self, detail_level: str = "summary"):
        """
        Generate capabilities payload at the specified detail level.

        Args:
            detail_level: One of "summary" or "detailed". If "summary", returns
                         hierarchical JSON. If "detailed", returns Turtle RDF string.
                         Unknown values default to "summary".

        Returns:
            For "summary": dict with hierarchical workspace/artifact/affordance tree.
            For "detailed": Turtle RDF string.
            For missing/legacy: dict with full format_capabilities_payload output.
        """
        try:
            if detail_level == "detailed":
                return format_capabilities_detailed_rdf(self.agent)
            elif detail_level == "summary":
                capabilities_payload = format_capabilities_summary_hierarchical(self.agent)
                ## pretty-print the payload if verbose mode is enabled for easier debugging (can be removed in production)
                if os.getenv("VERBOSE_LOGGING", "false").lower() == "true":
                    print(json.dumps(capabilities_payload, indent=2))
                return capabilities_payload
            else:
                # Unknown detail_level - use default legacy payload
                return format_capabilities_payload(self.agent)
        except Exception as e:
            self.agent.logger.error(f"Error generating capabilities payload: {e}", exc_info=True)
            return {
                "error": "capabilities_generation_failed",
                "detail": str(e),
                "discovery_complete": False,
            }