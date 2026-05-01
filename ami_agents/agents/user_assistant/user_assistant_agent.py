"""
UserAssistant Agent — refactored as a plain SPADE Agent.

The LLM is a *component* (``AsyncOpenAI`` client) used by behaviours for
NLU (intent extraction) and NLG (plan summary / query formatting).
Everything else — plan requesting, confirmation handling, execution,
signifier recording — is handled by deterministic SPADE behaviours.
"""

import asyncio
import os
from typing import Any, Dict

from spade.agent import Agent
from spade.message import Message as SpadeMessage
from spade.template import Template

from ...shared.protocols.agent_protocol import IAgent
from ...shared.models.messages import (
    Message,
    MessageType,
    META_CORRELATION_ID,
    META_CONVERSATION_ID,
    ensure_correlation_id,
    extract_conversation_id,
    serialize_body,
)
from ...shared.utils.spade_rpc import send_via_router
from ...shared.utils.config_resolver import resolve_yggdrasil_url
from ...shared.utils.demo_log import demo
from ...shared.utils.logger import LoggerFactory
from ...shared.state_memory import StateMemoryCache
from ...shared.community.community_client import CommunitySignifierClient

from ...environment.integration.integration_engine import YggdrasilIntegration

from .models import ConversationState
from .behaviours import UserMessageBehaviour, DemoRequestClassifierBehaviour
from .utils import build_llm_client, build_llm_call_kwargs, LLMClientConfig

# Global logger will be replaced by per-instance loggers
# logger = logging.getLogger("UserAssistant")

# Fallback defaults when yaml is missing a key. Every production deployment
# provides these via ``config/agents.yaml``; the constants here just keep the
# agent runnable in isolated unit tests.
_DEFAULT_GOAL_REQUEST_TIMEOUT_S = 60.0
_DEFAULT_RPC_CALL_TIMEOUT_S = 15.0
_DEFAULT_SIGNIFIER_MATCH_TIMEOUT_S = 10.0
_DEFAULT_BT_MAX_TICKS = 50


class UserAssistantAgent(Agent, IAgent):
    """UserAssistant agent with composable behaviours.

    Architecture (following InteractionSolverAgent pattern):
    - Extends plain ``spade.agent.Agent`` (not ``LLMAgent``).
    - ``self.llm_client`` is an ``AsyncOpenAI`` instance used by
      behaviours for NLU and NLG only.
    - All orchestration (plan requesting, confirmation, execution)
      is deterministic.
    """

    def __init__(
        self, jid: str, password: str, config: Dict[str, Any], target_jids: Dict[str, str]
    ):
        super().__init__(jid, password, verify_security=False)
        self.config = config
        self.target_jids = target_jids

        # Initialize logger with configuration
        # The logging config passed from main.py already contains merged global + agent-specific
        logging_config = self.config.get("logging", {})
        self.logger = LoggerFactory.get_logger(f"UserAssistant[{jid}]", logging_config)

        # ── LLM client (component, not base class) ──────────────────
        self._llm_cfg: LLMClientConfig = build_llm_client(config)
        self.llm_client = self._llm_cfg.client
        self.llm_model: str = self._llm_cfg.model
        self.llm_base_url: str = self._llm_cfg.base_url
        self.llm_temperature: float = self._llm_cfg.temperature
        self.llm_reasoning_effort = self._llm_cfg.reasoning_effort
        self.llm_max_completion_tokens = self._llm_cfg.max_completion_tokens

        # ── Timeouts / tick limits (pulled from yaml) ───────────────
        timeouts = config.get("timeouts", {}) or {}
        rpc_cfg = timeouts.get("rpc", {}) or {}
        signifier_cfg = timeouts.get("signifier", {}) or {}
        planning_cfg = config.get("planning", {}) or {}

        self.rpc_call_timeout: float = float(
            rpc_cfg.get("call", _DEFAULT_RPC_CALL_TIMEOUT_S)
        )
        self.signifier_match_timeout: float = float(
            signifier_cfg.get("match", _DEFAULT_SIGNIFIER_MATCH_TIMEOUT_S)
        )
        self.goal_request_timeout: float = float(
            planning_cfg.get("timeout", _DEFAULT_GOAL_REQUEST_TIMEOUT_S)
        )
        self.bt_max_ticks: int = int(
            config.get("bt_execution", {})
            .get("max_ticks", {})
            .get("user_assistant", _DEFAULT_BT_MAX_TICKS)
        )

        # ── Execution engine ────────────────────────────────────────
        self.yggdrasil_url = resolve_yggdrasil_url(config)
        self.execution_engine = YggdrasilIntegration(self.yggdrasil_url)
        self._execution_engine_ready = False
        self._execution_engine_lock = asyncio.Lock()

        # ── Per-conversation state ──────────────────────────────────
        self._conversations: Dict[str, ConversationState] = {}

        # ── State memory cache ──────────────────────────────────────
        self.state_memory = StateMemoryCache()

        # ── Community signifier client (optional) ───────────────────
        community_cfg = (config.get("planning", {}) or {}).get("community", {}) or {}
        community_url = community_cfg.get("api_url") or os.getenv("COMMUNITY_API_URL")
        self.community_client = None
        if community_url:
            self.community_client = CommunitySignifierClient(api_url=community_url)
            self.logger.info("Community signifier client enabled (url=%s)", community_url)

    # ── Conversation state helpers ──────────────────────────────────

    def get_conversation(self, thread: str) -> ConversationState:
        """Get or create per-conversation state."""
        if thread not in self._conversations:
            self._conversations[thread] = ConversationState()
        return self._conversations[thread]

    # ── LLM kwargs builder ──────────────────────────────────────────

    def build_llm_kwargs(self) -> Dict[str, Any]:
        """Build extra kwargs for ``llm_client.chat.completions.create``."""
        return build_llm_call_kwargs(self._llm_cfg)

    # ── Execution engine ────────────────────────────────────────────

    async def ensure_execution_engine_ready(self) -> None:
        """Initialize and hydrate the YggdrasilIntegration engine.

        Cached so it only runs once per agent lifecycle.
        """
        if self._execution_engine_ready:
            return
        async with self._execution_engine_lock:
            if self._execution_engine_ready:
                return
            ok = await self.execution_engine.initialize({})
            if not ok:
                raise RuntimeError(
                    f"Failed to initialize YggdrasilIntegration at {self.yggdrasil_url}"
                )
            await self.execution_engine.explore_hmas_environment()
            self._execution_engine_ready = True

    # ── SPADE lifecycle ─────────────────────────────────────────────

    async def setup(self):
        await super().setup()

        temp_display = "default" if self.llm_model.startswith("o") else self.llm_temperature
        self.logger.info(
            demo(
                "UserAssistant booting (model=%s, base_url=%s, temperature=%s, reasoning_effort=%s)"
            ),
            self.llm_model,
            self.llm_base_url,
            temp_display,
            self.llm_reasoning_effort or "default",
        )

        # Register behaviours with template routing
        t = Template()
        t.set_metadata("message_type", "llm")
        self.add_behaviour(UserMessageBehaviour(self.logger), template=t)
        self.add_behaviour(DemoRequestClassifierBehaviour(self.logger), template=t)

        self.logger.info("UserAssistantAgent initialized (composable behaviours).")

    async def start(self, *args, **kwargs) -> None:
        return await super().start(*args, **kwargs)

    async def stop(self) -> None:
        return await super().stop()

    # ── IAgent protocol ─────────────────────────────────────────────

    async def send_message(self, message: Message | SpadeMessage) -> bool:
        """Send a message to another agent."""
        if isinstance(message, SpadeMessage):
            await send_via_router(self, message)
            return True

        spade_msg = SpadeMessage(to=message.receiver)
        spade_msg.set_metadata("type", message.message_type.value)

        for key, value in (message.metadata or {}).items():
            if value is not None:
                spade_msg.set_metadata(key, str(value))

        corr_id = ensure_correlation_id(message.metadata)
        spade_msg.set_metadata(META_CORRELATION_ID, str(corr_id))

        conv_id = extract_conversation_id(message.conversation_id, message.metadata)
        if conv_id:
            spade_msg.thread = conv_id
            spade_msg.set_metadata(META_CONVERSATION_ID, conv_id)

        spade_msg.body = serialize_body(message.content)
        await send_via_router(self, spade_msg)
        return True

    async def receive_message(self, message: Message) -> None:
        # Incoming messages are routed to behaviours via SPADE templates;
        # this IAgent hook is intentionally a no-op.
        return
