"""
Initial discovery behavior for EnvExplorer agent.
"""

import asyncio
import json
from rdflib import Graph, Namespace
from spade.behaviour import OneShotBehaviour
from spade.message import Message as SpadeMessage

from ....shared.models.messages import MessageType, META_CORRELATION_ID, ensure_correlation_id
from ....shared.utils.demo_log import demo


TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")


def _detect_tdsosa_support(agent) -> bool:
    for artifact in (agent.artifacts or {}).values():
        td = getattr(artifact, "thing_description", None)
        rdf = getattr(td, "rdf", None)
        if not isinstance(rdf, str) or not rdf.strip():
            continue
        try:
            graph = Graph()
            graph.parse(data=rdf, format="turtle")
        except Exception:
            continue
        if any(str(pred).startswith(str(TDSOSA)) for _, pred, _ in graph):
            return True
        if any(str(obj).startswith(str(TDSOSA)) for _, _, obj in graph):
            return True
    return False


class InitialDiscoveryBehaviour(OneShotBehaviour):
    """Behavior for initial environment discovery."""

    async def run(self):
        """
        Perform initial discovery of the environment.

        TODO: Implementation steps:
        1. Wait for agent to be ready
        2. Crawl entire HMAS environment
        3. Build environment map (workspaces)
        4. Collect all artifacts and Thing Descriptions
        5. Extract affordances from Thing Descriptions
        6. Store in agent's data structures
        7. Subscribe to all workspace change notifications
        8. Mark discovery as complete
        9. Notify UserAssistant and InteractionSolver
        10. Stop this behavior (one-time only)
        """
        self.agent.logger.info(demo("Starting EnvExplorer discovery..."))

        success = await self.agent.integration_engine.initialize({})
        if not success:
            self.agent.logger.error("Failed to initialize Integration Engine.")
            return

        # Start webhook listener for event notifications before subscribing
        callback_url = await self.agent.integration_engine.start_notification_listener()

        try:
            await self.agent.integration_engine.explore_hmas_environment()

            self.agent.environment_map = self.agent.integration_engine.workspace_map
            self.agent.artifacts = self.agent.integration_engine.artifact_map
            self.agent.affordances = self.agent.integration_engine.affordance_map
            self.agent.semantic_capabilities = {
                "td_sosa_supported": _detect_tdsosa_support(self.agent),
            }

            self.agent.logger.info("Subscribing to artifact events...")
            for artifact_id, artifact in self.agent.artifacts.items():
                # The engine uses the internal listener automatically (callback_url=None)
                success = await self.agent.integration_engine.subscribe_to_artifact(
                    artifact_id, callback_url=callback_url
                )
                if success:
                    self.agent.logger.debug(f"Subscribed to {artifact.name}")
                else:
                    self.agent.logger.warning(f"Could not subscribe to {artifact.name} (might be static)")

            self.agent.discovery_complete = True
            self.agent.logger.info(f"Discovery Complete. Found {len(self.agent.artifacts)} artifacts.")
            self.agent.logger.info(
                demo("Discovery summary: workspaces=%d artifacts=%d affordances=%d"),
                len(self.agent.environment_map or {}),
                len(self.agent.artifacts or {}),
                len(self.agent.affordances or {}),
            )

            await self.notify_discovery_complete()

        except Exception as e:
            self.agent.logger.error(f"Error during discovery: {e}", exc_info=True)

    async def notify_discovery_complete(self) -> None:
        """
        Notify other agents that discovery is complete.

        TODO: Implementation steps:
        1. Create ENV_DISCOVERY_COMPLETE message
        2. Send to UserAssistant
        3. Send to InteractionSolver
        4. Log notification sent
        """
        # Use configuration for discovery notification
        discovery_config = self.agent.config.get("discovery", {})
        notify_on_discovery_complete = discovery_config.get("notify_on_discovery_complete", True)
        notify_agents = discovery_config.get("notify_agents", ["user_assistant@localhost", "interaction_solver@localhost"])

        if not notify_on_discovery_complete:
            return

        if not notify_agents:
            self.agent.logger.info("Discovery complete: no notify_agents configured.")
            return

        payload = {
            "discovery_complete": True,
            "artifacts_count": len(self.agent.artifacts or {}),
            "affordances_count": len(self.agent.affordances or {}),
            "yggdrasil_url": getattr(self.agent, "yggdrasil_url", None),
            "semantic_capabilities": dict(getattr(self.agent, "semantic_capabilities", {}) or {}),
        }

        sent = 0
        for jid in notify_agents:
            try:
                msg = SpadeMessage(to=str(jid))
                msg.set_metadata("type", MessageType.ENV_DISCOVERY_COMPLETE.value)
                msg.set_metadata(META_CORRELATION_ID, ensure_correlation_id({}))
                msg.body = json.dumps(payload)
                await self.send(msg)
                sent += 1
            except Exception as e:
                self.agent.logger.warning(f"Failed to notify {jid} of discovery complete: {e}")

        self.agent.logger.info(f"Discovery complete notification sent to {sent}/{len(notify_agents)} agents.")
