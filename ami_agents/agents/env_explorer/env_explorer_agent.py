"""
EnvExplorer Agent - Environment discovery and monitoring.

Classical SPADE agent that crawls, monitors, and manages environment knowledge.
"""

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour, OneShotBehaviour
from spade.message import Message as SpadeMessage
from spade.template import Template

from ...shared.protocols.agent_protocol import IAgent
from ...shared.models.messages import Message, MessageType, AffordanceMatchRequest, AffordanceMatchResponse
from ...shared.models.messages import META_CORRELATION_ID, META_CONVERSATION_ID, ensure_correlation_id
from ...shared.models.environment import (
    Workspace, Artifact, Affordance, Signifier, ChangeEvent, ChangeEventType, AffordanceType
)
from ...shared.models.plan import BehaviorTreePlan
from ...shared.utils.config_resolver import resolve_yggdrasil_url
from ...environment.connection.hmas_client import IHMASClient
from ...environment.integration.integration_engine import YggdrasilIntegration


class EnvExplorerAgent(Agent, IAgent):
    """
    EnvExplorer agent for environment discovery and monitoring.

    Responsibilities:
    - Discover and map the HMAS environment
    - Monitor for changes (artifacts, capabilities, state)
    - Store and retrieve signifiers (usage experiences)
    - Match affordances to goals
    - Notify other agents of changes
    """

    def __init__(self, jid: str, password: str, config: Dict[str, Any],
                 hmas_client: IHMASClient):
        """
        Initialize EnvExplorer agent.

        Args:
            jid: SPADE JID for the agent.
            password: SPADE password.
            config: Agent configuration.
            hmas_client: HMAS client for environment interaction.
        """
        super().__init__(jid, password)
        self.config = config or {}
        self.hmas_client = hmas_client
        self.environment_map = {}
        self.artifacts = {}
        self.affordances = {}
        self.signifiers_store = None
        self.yggdrasil_url = resolve_yggdrasil_url(self.config)
        self.integration_engine = YggdrasilIntegration(self.yggdrasil_url)
        self.discovery_complete = False
        self.logger = logging.getLogger(__name__)

    async def setup(self):
        """
        Setup the agent (SPADE lifecycle method).

        Connects to the Yggdrasil HMAS instance and performs an initial
        environment crawl to populate workspace/artifact/affordance maps.
        """
        self.logger.info(f"EnvExplorerAgent starting...")
        self.add_behaviour(InitialDiscoveryBehaviour())

        # Route environment capability/state requests to the handler using templates
        cap_template = Template()
        cap_template.set_metadata("type", MessageType.ENV_CAPABILITIES_REQUEST.value)
        state_template = Template()
        state_template.set_metadata("type", MessageType.ENV_STATE_REQUEST.value)

        self.add_behaviour(EnvironmentRequestHandler(), template=cap_template)
        self.add_behaviour(EnvironmentRequestHandler(), template=state_template)

        self.add_behaviour(EventProcessingBehaviour(self.integration_engine))

    def _generate_capabilities_summary(self) -> str:
        """
        Formats the internal artifact map into a detailed string for the LLM.
        Includes Forms and Input Schemas so the LLM understands parameters.
        """
        # 1. Check readiness
        if not self.discovery_complete:
            return "Environment discovery is still in progress. Please try again later."

        # 2. Read from Agent Memory (populated by InitialDiscoveryBehaviour)
        artifacts = self.artifacts.values()
        
        if not artifacts:
            return "No artifacts found in the environment."

        summary = "Available Environment Capabilities:\n"
        
        for artifact in artifacts:
            # 3. Retrieve actions
            # We use the engine's helper to filter affordances for this artifact ID
            actions = self.integration_engine.get_affordances_for_artifact(artifact.artifact_id)
            
            # Filter for ACTION types (we only care about what we can DO)
            action_affordances = [a for a in actions if a.affordance_type.value == "action"]
            
            if action_affordances:
                summary += f"Artifact: {artifact.name}\n"
                summary += f"  ID: {artifact.artifact_id}\n"
                summary += f"  Capabilities:\n"
                
                for action in action_affordances:
                    summary += f"    - Action: {action.name}\n"
                    
                    # Include Form Details (Method + URL)
                    # This helps the LLM distinguish between GET (read) and POST (write)
                    if action.form:
                        summary += f"      Target: [{action.form.method}] {action.form.href}\n"
                    
                    # Include Input Schema (Parameters)
                    # This tells the LLM what arguments (e.g. brightness level) are required
                    if action.input_schema:
                        summary += f"      Schema: {json.dumps(action.input_schema)}\n"
                
                summary += "\n"
        
        return summary

    def _generate_capabilities_payload(self) -> Dict[str, Any]:
        """
        Machine-readable capabilities payload for other agents (planning, etc.).
        Includes a human-friendly 'summary' field for convenience.
        """
        if not self.discovery_complete:
            return {
                "discovery_complete": False,
                "error": "discovery_in_progress",
                "summary": "Environment discovery is still in progress. Please try again later.",
                "workspaces": [],
                "artifacts": [],
                "affordances": [],
            }

        artifacts = list(self.artifacts.values())
        if not artifacts:
            return {
                "discovery_complete": True,
                "summary": "No artifacts found in the environment.",
                "workspaces": [],
                "artifacts": [],
                "affordances": [],
            }

        # Workspaces (for multi-workspace UX and scoping)
        workspaces_out: List[Dict[str, Any]] = []
        try:
            for ws in (self.environment_map or {}).values():
                workspaces_out.append(
                    {
                        "workspace_id": ws.workspace_id,
                        "name": ws.name,
                        "workspace_type": getattr(ws.workspace_type, "value", str(ws.workspace_type)),
                        "parent_workspace_id": getattr(ws, "parent_workspace_id", None),
                        "sub_workspaces": list(getattr(ws, "sub_workspaces", []) or []),
                        "artifacts": list(getattr(ws, "artifacts", []) or []),
                    }
                )
        except Exception:
            workspaces_out = []

        affordances_out: List[Dict[str, Any]] = []
        artifacts_out: List[Dict[str, Any]] = []

        for artifact in artifacts:
            affs = self.integration_engine.get_affordances_for_artifact(artifact.artifact_id)
            action_affordances = [a for a in affs if a.affordance_type == AffordanceType.ACTION]

            actions_out: List[Dict[str, Any]] = []
            for action in action_affordances:
                form = action.form
                actions_out.append(
                    {
                        "affordance_id": action.affordance_id,
                        "name": action.name,
                        "description": action.description,
                        "artifact_id": action.artifact_id,
                        "affordance_type": action.affordance_type.value,
                        "semantic_types": list(action.semantic_types or []),
                        "form": {
                            "href": getattr(form, "href", None),
                            "method": getattr(form, "method", None),
                            "content_type": getattr(form, "content_type", None),
                            "operation_type": getattr(form, "operation_type", None),
                            "additional_fields": getattr(form, "additional_fields", None) or {},
                        }
                        if form
                        else None,
                        "input_schema": action.input_schema,
                        "output_schema": action.output_schema,
                    }
                )

                affordances_out.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "artifact_name": artifact.name,
                        "workspace_id": getattr(artifact, "workspace_id", None),
                        "affordance_id": action.affordance_id,
                        "affordance_type": action.affordance_type.value,
                        "action_name": action.name,
                        "method": getattr(action.form, "method", None) if action.form else None,
                        "target": getattr(action.form, "href", None) if action.form else None,
                        "content_type": getattr(action.form, "content_type", None) if action.form else None,
                        "input_schema": action.input_schema,
                    }
                )

            artifacts_out.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "name": artifact.name,
                    "workspace_id": getattr(artifact, "workspace_id", None),
                    "actions": actions_out,
                }
            )

        return {
            "discovery_complete": True,
            "summary": self._generate_capabilities_summary(),
            "workspaces": workspaces_out,
            "artifacts": artifacts_out,
            "affordances": affordances_out,
        }

    async def start(self, *args, **kwargs) -> None:
        """
        Start the EnvExplorer agent.

        Uses SPADE's start to trigger setup/discovery.
        """
        return await super().start(*args, **kwargs)

    async def stop(self) -> None:
        """
        Stop the EnvExplorer agent.

        TODO: Implementation steps:
        1. Stop all behaviors
        2. Unsubscribe from environment events
        3. Disconnect HMAS client
        4. Close signifier storage
        5. Cleanup resources
        """
        if self.integration_engine:
            await self.integration_engine.stop_notification_listener()
        await super().stop()

    async def send_message(self, message: Message) -> bool:
        """
        Send a message to another agent.

        TODO: Implementation steps:
        1. Validate message
        2. Serialize to SPADE message format
        3. Send via SPADE
        4. Log sent message
        5. Return success status
        """
        pass

    async def receive_message(self, message: Message) -> None:
        """
        Receive and process a message.

        TODO: Implementation steps:
        1. Deserialize SPADE message
        2. Route based on message type
        3. Handle appropriately
        """
        pass


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
        self.agent.logger.info("Starting Initial Discovery...")
        
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
            
            await self.notify_discovery_complete()
            
        except Exception as e:
            self.agent.logger.error(f"Error during discovery: {e}", exc_info=True)

        pass

    async def subscribe_to_changes(self) -> None:
        """
        Subscribe to environment change notifications.

        TODO: Implementation steps:
        1. For each workspace in environment_map:
           a. Subscribe to workspace change events
           b. Register callback for change handling
        """
        pass

    async def notify_discovery_complete(self) -> None:
        """
        Notify other agents that discovery is complete.

        TODO: Implementation steps:
        1. Create ENV_DISCOVERY_COMPLETE message
        2. Send to UserAssistant
        3. Send to InteractionSolver
        4. Log notification sent
        """
        discovery_cfg = (self.agent.config or {}).get("discovery", {}) or {}
        if not discovery_cfg.get("notify_on_discovery_complete", True):
            return

        notify_agents = discovery_cfg.get("notify_agents") or []
        if not notify_agents:
            self.agent.logger.info("Discovery complete: no notify_agents configured.")
            return

        payload = {
            "discovery_complete": True,
            "artifacts_count": len(self.agent.artifacts or {}),
            "affordances_count": len(self.agent.affordances or {}),
            "yggdrasil_url": getattr(self.agent, "yggdrasil_url", None),
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

class ChangeMonitoringBehaviour(PeriodicBehaviour):
    """Behavior for monitoring environment changes."""

    async def run(self):
        """
        Periodically check for environment changes.

        TODO: Implementation steps:
        1. Check for new change events
        2. For each change event:
           a. Update internal data structures
           b. Analyze impact on existing plans
           c. Notify UserAssistant if plans affected
        """
        pass

    async def handle_change_event(self, event: ChangeEvent) -> None:
        """
        Handle a specific change event.

        TODO: Implementation steps:
        1. Based on event type:
           - ARTIFACT_ADDED: Add to artifacts map
           - ARTIFACT_REMOVED: Remove from artifacts map
           - CAPABILITY_CHANGED: Update affordances
           - STATE_CHANGED: Update artifact state
        2. Analyze plan impact
        3. Send notifications if needed
        """
        pass

    async def analyze_plan_impact(self, event: ChangeEvent) -> List[str]:
        """
        Analyze which plans are affected by the change.

        TODO: Implementation steps:
        1. Query UserAssistant for maintenance plans
        2. For each plan:
           a. Check if change affects triggering conditions
           b. Check if change affects plan affordances
        3. Collect affected plan_ids
        4. Return list
        """
        pass


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
                self.agent.logger.info(f"Received capabilities request from {msg.sender}")
                
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
                self.agent.logger.info(f"Received state request from {msg.sender}")

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
        pass

    async def match_affordances(self, request: AffordanceMatchRequest) -> AffordanceMatchResponse:
        """
        Match affordances to a goal.

        TODO: Implementation steps:
        1. Extract goal intent
        2. Search signifiers for similar intents
        3. Rank signifiers by:
           a. Intent similarity
           b. Context similarity
        4. If signifiers found, prioritize them
        5. If no signifiers or need more options:
           a. Use hybrid reasoning (rule-based + LLM)
           b. Apply physics-informed modeling
           c. Find suitable affordances
        6. Compile list of matched affordances
        7. Create and return AffordanceMatchResponse
        """
        pass

    async def search_signifiers(self, intent: str, context: Dict[str, Any],
                               threshold: float = 0.8) -> List[Signifier]:
        """
        Search for signifiers matching an intent.

        TODO: Implementation steps:
        1. Query signifier storage
        2. Compute intent similarity (using embeddings)
        3. Filter by threshold
        4. If context matching enabled:
           a. Compute context similarity
           b. Filter by context threshold
        5. Sort by similarity scores
        6. Return matched signifiers
        """
        pass

    async def hybrid_affordance_matching(self, intent: str,
                                        context: Dict[str, Any]) -> List[Affordance]:
        """
        Use hybrid reasoning to match affordances.

        TODO: Implementation steps:
        1. Apply rule-based matching:
           a. Extract keywords from intent
           b. Match against affordance descriptions
        2. Apply LLM reasoning:
           a. Prepare prompt with intent and affordances
           b. Ask LLM to select suitable affordances
           c. Parse LLM response
        3. If physics-informed modeling enabled:
           a. Consider physical constraints
           b. Consider device capabilities
        4. Combine results
        5. Return ranked affordances
        """
        pass


class SignifierManager:
    """Manages storage and retrieval of signifiers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SignifierManager.

        Args:
            config: Signifier configuration.
        """
        self.config = config
        self.storage = None  # Database connection

    async def initialize_storage(self) -> None:
        """
        Initialize signifier storage backend.

        TODO: Implementation steps:
        1. Connect to database (SQLite, etc.)
        2. Create tables if not exist
        3. Load indices for fast retrieval
        """
        pass

    async def store_signifier(self, signifier: Signifier) -> None:
        """
        Store a signifier.

        TODO: Implementation steps:
        1. Check if similar signifier exists
        2. If exists, increment usage_count
        3. Otherwise, insert new signifier
        4. Update indices
        """
        pass

    async def retrieve_signifiers_by_intent(self, intent: str,
                                           threshold: float = 0.8) -> List[Signifier]:
        """
        Retrieve signifiers matching an intent.

        TODO: Implementation steps:
        1. Generate intent embedding
        2. Query vector store or compute similarities
        3. Filter by threshold
        4. Return matched signifiers
        """
        pass

    async def retrieve_signifiers_by_affordance(self, affordance_id: str) -> List[Signifier]:
        """
        Retrieve all signifiers for a specific affordance.

        TODO: Implementation steps:
        1. Query database for affordance_id
        2. Return all matching signifiers
        """
        pass

    async def extract_signifiers_from_plan(self, plan: BehaviorTreePlan) -> List[Signifier]:
        """
        Extract signifiers from a behavior tree plan.

        TODO: Implementation steps:
        1. Get all leaf action nodes from plan
        2. For each action node:
           a. Create signifier with affordance_id
           b. Record intent (from plan goal)
           c. Record context (from plan metadata)
           d. Record parameters used
        3. Store signifiers
        4. Return list of created signifiers
        """
        pass

    async def get_context_for_signifier(self, signifier_id: str) -> Dict[str, Any]:
        """
        Get the context of use for a signifier.

        TODO: Implementation steps:
        1. Retrieve signifier from database
        2. Return context dictionary
        """
        pass
