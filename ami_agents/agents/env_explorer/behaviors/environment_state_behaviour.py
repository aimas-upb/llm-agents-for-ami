import json
from typing import Dict, Any, Optional
from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo


class EnvironmentStateBehaviour(CyclicBehaviour):
    """
    Handles ENV_STATE_REQUEST messages and provides current state information
    about artifacts, including full snapshots or filtered property queries.
    """

    async def run(self):
        """Main behavior loop - handles environment state requests."""
        msg = await self.receive(timeout=1)
        if not msg:
            return

        # Only process ENV_STATE_REQUEST messages
        msg_type = msg.get_metadata("type")
        if msg_type != MessageType.ENV_STATE_REQUEST.value:
            return

        self.agent.logger.info(demo("Received ENV_STATE_REQUEST from %s"), str(msg.sender))

        # Parse request payload
        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}
            self.agent.logger.warning("Failed to parse state request payload, using empty payload")

        # Extract request parameters
        artifact_id = (
            payload.get("artifact_id")
            or payload.get("artifact")
            or payload.get("artifact_uri")
        )
        property_uri = payload.get("property_uri") or payload.get("property")

        self.agent.logger.debug(f"Processing state request: artifact_id={artifact_id}, property_uri={property_uri}")

        # Generate state response
        response_payload = self._generate_state_response(artifact_id, property_uri)

        # Send response
        reply = msg.make_reply()
        reply.body = json.dumps(response_payload)
        reply.set_metadata("type", MessageType.ENV_STATE_RESPONSE.value)

        # Preserve correlation ID and thread
        correlation_id = msg.get_metadata("correlation_id")
        if correlation_id:
            reply.set_metadata("correlation_id", correlation_id)
        if msg.thread:
            reply.thread = msg.thread

        await self.send(reply)

        # Log response summary
        if "error" in response_payload:
            self.agent.logger.debug(f"Sent error response: {response_payload.get('error')}")
        elif artifact_id:
            self.agent.logger.debug(f"Sent artifact state for: {artifact_id}")
        else:
            artifact_count = len(response_payload.get("artifacts", {}))
            self.agent.logger.debug(f"Sent full environment snapshot with {artifact_count} artifacts")

    def _generate_state_response(self, artifact_id: Optional[str], property_uri: Optional[str]) -> Dict[str, Any]:
        """
        Generate environment state response based on request parameters.

        Args:
            artifact_id: Specific artifact to query (None for full snapshot)
            property_uri: Specific property to query (None for all properties)

        Returns:
            Dictionary containing state information or error details
        """
        try:
            if artifact_id:
                return self._get_artifact_state(artifact_id, property_uri)
            else:
                return self._get_full_environment_snapshot()
        except Exception as e:
            self.agent.logger.error(f"Error generating state response: {e}", exc_info=True)
            return {
                "error": "state_generation_failed",
                "detail": str(e)
            }

    def _get_artifact_state(self, artifact_id: str, property_uri: Optional[str]) -> Dict[str, Any]:
        """
        Get state information for a specific artifact.

        Args:
            artifact_id: The artifact identifier
            property_uri: Optional specific property to query

        Returns:
            Dictionary containing artifact state or error information
        """
        # Find the artifact
        artifact = self.agent.artifacts.get(artifact_id)
        if not artifact:
            return {
                "error": "artifact_not_found",
                "artifact_id": artifact_id,
            }

        # Get the current state
        state = dict(artifact.current_state)

        if property_uri:
            # Return specific property
            if property_uri in state:
                return {
                    "artifact_id": artifact_id,
                    "property_uri": property_uri,
                    "value": state.get(property_uri),
                }
            else:
                return {
                    "error": "property_not_found",
                    "artifact_id": artifact_id,
                    "property_uri": property_uri,
                    "available_properties": list(state.keys())
                }
        else:
            # Return full artifact state
            return {
                "artifact_id": artifact_id,
                "name": artifact.name,
                "workspace_id": getattr(artifact, "workspace_id", None),
                "state": state,
                "property_count": len(state)
            }

    def _get_full_environment_snapshot(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of all artifacts in the environment.

        Returns:
            Dictionary containing all artifacts and their states
        """
        artifacts_snapshot = {}
        total_properties = 0

        for aid, artifact in self.agent.artifacts.items():
            artifact_state = dict(artifact.current_state)
            artifacts_snapshot[aid] = {
                "name": artifact.name,
                "workspace_id": getattr(artifact, "workspace_id", None),
                "state": artifact_state,
            }
            total_properties += len(artifact_state)

        return {
            "artifacts": artifacts_snapshot,
            "summary": {
                "total_artifacts": len(artifacts_snapshot),
                "total_properties": total_properties,
                "workspaces": list(set(
                    artifact.get("workspace_id") for artifact in artifacts_snapshot.values()
                    if artifact.get("workspace_id")
                ))
            }
        }