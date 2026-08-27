"""
EnvExplorer Agent - Environment discovery and monitoring.

Refactored SPADE agent with proper package structure for behaviors and utilities.
"""

import asyncio
import re
from typing import Any, Dict, Optional
from spade.agent import Agent
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
from ...shared.utils.demo_log import demo
from ...shared.utils.logger import LoggerFactory
from ...environment.connection.hmas_client import IHMASClient
from ...environment.integration.integration_engine import YggdrasilIntegration

# Import extracted behaviors
from .behaviors import (
    InitialDiscoveryBehaviour,
    InitializeExperienceEngineBehaviour,
    EventProcessingBehaviour,
    SignifierMatchBehaviour,
    SignifierRecordBehaviour,
    SignifierListBehaviour,
    EnvironmentCapabilitiesBehaviour,
    EnvironmentStateBehaviour,
)


# Configuration constants as fallbacks
_EXPERIENCE_ENGINE_DEFAULTS = {
    "default_matcher_version": "v2",
    "default_min_similarity": 0.5,
    "signifier_limit": 10000,
}
_TIMEOUTS = {
    "message_reception": 5.0,
}


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
            hmas_client: HMAS client instance.
        """
        super().__init__(jid, password)

        self.config = config or {}
        self.hmas_client = hmas_client

        # Environment state
        self.environment_map: Dict[str, Workspace] = {}
        self.artifacts: Dict[str, Artifact] = {}
        self.affordances: Dict[str, Affordance] = {}
        self.signifiers_store = None

        # Integration engine
        self.yggdrasil_url = resolve_yggdrasil_url(config)
        self.integration_engine = YggdrasilIntegration(self.yggdrasil_url)

        # Agent state
        self.discovery_complete = False
        self.semantic_capabilities: Dict[str, Any] = {"td_sosa_supported": False}
        self.semantic_query_enabled = False

        # Initialize logger with configuration
        # The logging config passed from main.py already contains merged global + agent-specific
        logging_config = self.config.get("logging", {})
        self.logger = LoggerFactory.get_logger(f"EnvExplorerAgent[{jid}]", logging_config)

        # Experience engine state
        self._experience_engine_ready = False
        self._experience_engine_lock = asyncio.Lock()
        self._experience_engine_storage_dir: Optional[str] = None
        self._experience_engine_registry = None
        self._experience_engine_matcher_registry = None
        self._experience_engine_context_builder = None
        self._experience_engine_shacl_validator = None
        self._experience_engine_default_matcher_version = self.config.get("experience_engine", {}).get("default_matcher_version", _EXPERIENCE_ENGINE_DEFAULTS["default_matcher_version"])
        min_similarity_raw = self.config.get("experience_engine", {}).get("default_min_similarity", _EXPERIENCE_ENGINE_DEFAULTS["default_min_similarity"])
        self._experience_engine_default_min_similarity = float(min_similarity_raw)
        self._experience_engine_shacl_validation_enabled = self.config.get("experience_engine", {}).get("shacl_validation_enabled", "false").lower() == "true"

    async def setup(self):
        """Set up the agent behaviors and templates."""
        self.logger.info(demo("Setting up EnvExplorerAgent..."))

        # Create message templates
        env_cap_template = Template()
        env_cap_template.set_metadata("type", MessageType.ENV_CAPABILITIES_REQUEST.value)

        env_state_template = Template()
        env_state_template.set_metadata("type", MessageType.ENV_STATE_REQUEST.value)

        # Signifier engine requests (need separate templates for each type)
        sign_match_template = Template()
        sign_match_template.set_metadata("type", MessageType.SIGNIFIER_MATCH_REQUEST.value)

        sign_record_template = Template()
        sign_record_template.set_metadata("type", MessageType.SIGNIFIER_RECORD_EXECUTION_REQUEST.value)

        sign_list_template = Template()
        sign_list_template.set_metadata("type", MessageType.SIGNIFIER_LIST_REQUEST.value)

        # Bootstrap the Experience Engine explicitly during setup so engine
        # readiness is a discrete, observable lifecycle event rather than a
        # side-effect of the first incoming SIGNIFIER_MATCH_REQUEST.
        self.add_behaviour(InitializeExperienceEngineBehaviour())

        self.add_behaviour(InitialDiscoveryBehaviour())
        self.add_behaviour(EnvironmentCapabilitiesBehaviour(), template=env_cap_template)
        self.add_behaviour(EnvironmentStateBehaviour(), template=env_state_template)
        self.add_behaviour(SignifierMatchBehaviour(), template=sign_match_template)
        self.add_behaviour(SignifierRecordBehaviour(), template=sign_record_template)
        self.add_behaviour(SignifierListBehaviour(), template=sign_list_template)
        self.add_behaviour(EventProcessingBehaviour(self.integration_engine))

    # Agent lifecycle methods
    async def start(self, *args, **kwargs):
        """Start the EnvExplorer agent."""
        self.logger.info(demo("Starting EnvExplorer agent"))
        await super().start(*args, **kwargs)

    async def stop(self):
        """Stop the EnvExplorer agent and clean up resources."""
        self.logger.info(demo("Stopping EnvExplorer agent"))
        if self.integration_engine:
            try:
                await self.integration_engine.unsubscribe_all_artifacts()
            except Exception as e:
                self.logger.error(f"Error unsubscribing artifact callbacks: {e}")
            try:
                await self.integration_engine.stop_notification_listener()
            except Exception as e:
                self.logger.error(f"Error stopping integration engine: {e}")
        await super().stop()

    # Implement the previously stubbed message methods
    async def send_message(self, message: Message, recipient_jid: str) -> bool:
        """
        Send message to another agent.

        Args:
            message: Message to send
            recipient_jid: JID of recipient agent

        Returns:
            True if message was sent successfully
        """
        try:
            spade_msg = SpadeMessage(to=recipient_jid)
            spade_msg.set_metadata("type", message.message_type.value)
            spade_msg.set_metadata(META_CORRELATION_ID, message.correlation_id)
            spade_msg.body = message.model_dump_json()
            await self.send(spade_msg)
            self.logger.debug(f"Sent message to {recipient_jid}: {message.message_type}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending message to {recipient_jid}: {e}")
            return False

    async def receive_message(self, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive message with timeout.

        Args:
            timeout: Timeout in seconds (uses default if None)

        Returns:
            Parsed message or None if no message received
        """
        timeout = timeout or self.config.get("timeouts", {}).get("message_reception", _TIMEOUTS["message_reception"])
        try:
            spade_msg = await self.receive(timeout=timeout)
            if spade_msg:
                return Message.model_validate_json(spade_msg.body)
            return None
        except Exception as e:
            self.logger.error(f"Error receiving message: {e}")
            return None
