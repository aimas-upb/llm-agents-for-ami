"""
Event processing behavior for EnvExplorer agent.
"""

import asyncio
from spade.behaviour import CyclicBehaviour


class EventProcessingBehaviour(CyclicBehaviour):
    """
    Behavior for processing asynchronous environment events from the mailbox.
    """
    def __init__(self, integration_engine):
        super().__init__()
        self.integration = integration_engine

    async def run(self):
        # 1. Block until an event arrives (efficient)
        try:
            # Check if listener is running
            if not self.integration.notification_listener:
                await asyncio.sleep(1) # Wait for setup
                return

            event_data = await self.integration.event_queue.get()

            # 2. Extract Identity
            # Yggdrasil sends "artifactUri" in the payload
            artifact_uri = event_data.get("artifactUri")
            if not artifact_uri:
                return

            # 3. Find Local Artifact
            # Handle potential suffix mismatch (http://.../light vs http://.../light#artifact)
            artifact = self.integration.artifact_map.get(artifact_uri)
            if not artifact:
                # Try fuzzy match if exact match fails
                artifact = next((a for a in self.integration.artifact_map.values()
                                    if artifact_uri in a.artifact_id or a.artifact_id in artifact_uri), None)

            if not artifact:
                self.agent.logger.warning(f"Received event for unknown artifact: {artifact_uri}")
                return

            # 4. Update State (Digital Twin)
            property_uri = event_data.get("propertyUri")
            value = event_data.get("value")

            if property_uri and value is not None:
                # Update Internal State
                artifact.current_state[property_uri] = value
                self.agent.logger.info(f"STATE UPDATE: {artifact.name} -> {property_uri} = {value}")

        except Exception as e:
            self.agent.logger.error(f"Error processing event: {e}")