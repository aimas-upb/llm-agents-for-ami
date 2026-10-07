import json
import os
from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo
from ..utils.actuation_context import (
    build_actuation_context,
    render_actuation_context,
    scope_artifacts,
)
from ..utils.data_formatting import (
    format_capabilities_payload,
    format_capabilities_summary_hierarchical,
    format_capabilities_detailed_rdf,
)
from ..utils.state_resolution import discovered_graph


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
        body = {}
        try:
            if msg.body:
                body = json.loads(msg.body)
                detail_level = body.get("detail_level", "summary")
        except (json.JSONDecodeError, AttributeError):
            body = {}

        # Generate capabilities response payload. `goal_context` is the goal
        # structurer's scoped view; `actuation_only` is it with both
        # additions off.
        if detail_level in ("goal_context", "actuation_only"):
            response_payload = self._generate_goal_context_payload(
                body.get("goal_text"),
                with_properties=bool(body.get("with_properties")),
                with_environment=bool(body.get("with_environment")),
                detail_level=detail_level,
            )
        else:
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

    def _generate_goal_context_payload(self, goal_text, with_properties=False,
                                       with_environment=False,
                                       detail_level="goal_context"):
        """The goal context, scoped to `goal_text` when one is given.

        See `utils/actuation_context.py` for the view and the scoping rules.
        """
        try:
            graph = discovered_graph(
                self.agent.artifacts.values(),
                self.agent.environment_map.values(),
                logger=self.agent.logger,
            )
            artifacts, scope = scope_artifacts(
                graph, goal_text,
                max_artifacts=self.agent.actuation_max_artifacts,
                max_actions=self.agent.actuation_max_actions,
                with_properties=with_properties,
            )
            context = build_actuation_context(
                graph, artifacts,
                with_properties=with_properties,
                with_environment=with_environment,
            )
        except Exception as e:
            self.agent.logger.error(f"Error building goal context: {e}", exc_info=True)
            return {
                "error": "goal_context_failed",
                "detail": str(e),
                "detail_level": detail_level,
            }

        self.agent.logger.info(demo(
            f"Goal context for {goal_text!r} "
            f"(properties={with_properties}, environment={with_environment}): "
            f"rule={scope.get('rule')} "
            f"rooms={scope.get('matched_rooms', [])} "
            f"devices={scope.get('matched_devices', [])} "
            f"workspaces={scope.get('workspaces', [])} "
            f"artifacts={'all' if artifacts is None else len(artifacts)}"))
        return {
            "detail_level": detail_level,
            "scope": scope,
            "context": context,
            "text": render_actuation_context(context),
        }

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