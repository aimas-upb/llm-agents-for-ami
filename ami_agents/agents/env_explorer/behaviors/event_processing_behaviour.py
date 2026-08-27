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

            self.agent.logger.debug(f"[EVENT RECEIVED] {event_data}")

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
                # Normalize property URI: prefer /properties/ over /props/ for consistency
                canonical_property_uri = property_uri
                if "/props/" in property_uri and "/properties/" not in property_uri:
                    # Convert /props/ to /properties/ for canonical storage
                    base = artifact.artifact_id.split("#")[0].rstrip("/")
                    suffix = property_uri.split("/props/", 1)[-1]
                    if suffix:
                        canonical_property_uri = f"{base}/properties/{suffix}"

                # Update Internal State with canonical URI
                artifact.current_state[canonical_property_uri] = value

                # Also remove any old /props/ variant if it exists (cleanup duplicates)
                if "/properties/" in canonical_property_uri:
                    base = artifact.artifact_id.split("#")[0].rstrip("/")
                    suffix = canonical_property_uri.split("/properties/", 1)[-1]
                    props_variant = f"{base}/props/{suffix}"
                    if props_variant in artifact.current_state:
                        del artifact.current_state[props_variant]

                self.agent.logger.info(f"STATE UPDATE: {artifact.name} -> {canonical_property_uri} = {value}")

                # Refresh full artifact state to capture all related property changes
                self.agent.logger.debug(f"Refreshing full state for {artifact.name} after property update...")
                try:
                    await self.integration.refresh_artifact_state(artifact_uri)
                except Exception as refresh_err:
                    self.agent.logger.debug(f"Could not refresh artifact state: {refresh_err}")

        except Exception as e:
            self.agent.logger.error(f"Error processing event: {e}")