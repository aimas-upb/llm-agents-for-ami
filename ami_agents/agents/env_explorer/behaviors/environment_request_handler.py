"""
Environment request handler behavior for EnvExplorer agent.
"""

import json
from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo


class EnvironmentRequestHandler(CyclicBehaviour):
    """Generic handler for incoming environment requests."""

    async def run(self):
        """
        Handle affordance match requests.

        TODO: Implementation steps:
        1. Wait for AffordanceMatchRequest message
        2. Process request
        3. Send AffordanceMatchResponse back
        """
        msg = await self.receive(timeout=1)
        if msg:
            msg_type = msg.get_metadata("type")

            # Check for capabilities request
            if msg_type == MessageType.ENV_CAPABILITIES_REQUEST.value:
                self.agent.logger.info(demo("Received ENV_CAPABILITIES_REQUEST from %s"), str(msg.sender))

                # Generate Response (machine-readable JSON payload + summary)
                response_payload = self.agent._generate_capabilities_payload()

                # Send Reply
                reply = msg.make_reply()
                reply.body = json.dumps(response_payload)
                reply.set_metadata("type", MessageType.ENV_CAPABILITIES_RESPONSE.value)

                # Preserve Correlation ID
                correlation_id = msg.get_metadata(META_CORRELATION_ID)
                if correlation_id:
                    reply.set_metadata(META_CORRELATION_ID, correlation_id)
                # Preserve thread as conversation id carrier (if set)
                if msg.thread:
                    reply.thread = msg.thread

                await self.send(reply)

            # Check for state request (full snapshot or filtered)
            elif msg_type == MessageType.ENV_STATE_REQUEST.value:
                self.agent.logger.info(demo("Received ENV_STATE_REQUEST from %s"), str(msg.sender))

                try:
                    payload = json.loads(msg.body or "{}")
                except json.JSONDecodeError:
                    payload = {}

                artifact_id = (
                    payload.get("artifact_id")
                    or payload.get("artifact")
                    or payload.get("artifact_uri")
                )
                property_uri = payload.get("property_uri") or payload.get("property")

                response_payload = {}

                if artifact_id:
                    artifact = self.agent.artifacts.get(artifact_id)
                    if not artifact:
                        response_payload = {
                            "error": "artifact_not_found",
                            "artifact_id": artifact_id,
                        }
                    else:
                        state = dict(artifact.current_state)
                        if property_uri:
                            if property_uri in state:
                                response_payload = {
                                    "artifact_id": artifact_id,
                                    "property_uri": property_uri,
                                    "value": state.get(property_uri),
                                }
                            else:
                                response_payload = {
                                    "error": "property_not_found",
                                    "artifact_id": artifact_id,
                                    "property_uri": property_uri,
                                }
                        else:
                            response_payload = {
                                "artifact_id": artifact_id,
                                "name": artifact.name,
                                "workspace_id": getattr(artifact, "workspace_id", None),
                                "state": state,
                            }
                else:
                    artifacts_snapshot = {}
                    for aid, artifact in self.agent.artifacts.items():
                        artifacts_snapshot[aid] = {
                            "name": artifact.name,
                            "workspace_id": getattr(artifact, "workspace_id", None),
                            "state": dict(artifact.current_state),
                        }
                    response_payload = {"artifacts": artifacts_snapshot}

                reply = msg.make_reply()
                reply.body = json.dumps(response_payload)
                reply.set_metadata("type", MessageType.ENV_STATE_RESPONSE.value)

                correlation_id = msg.get_metadata("correlation_id")
                if correlation_id:
                    reply.set_metadata("correlation_id", correlation_id)
                if msg.thread:
                    reply.thread = msg.thread

                await self.send(reply)