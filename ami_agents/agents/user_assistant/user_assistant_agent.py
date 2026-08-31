"""
UserAssistant Agent — refactored as a plain SPADE Agent.

The LLM is a *component* (``AsyncOpenAI`` client) used by behaviours for
NLU (intent extraction) and NLG (plan summary / query formatting).
Everything else — plan requesting, confirmation handling, execution,
signifier recording — is handled by deterministic SPADE behaviours.
"""

import asyncio
import os
from pathlib import Path
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
from .registry import PlanRegistry, RequestRegistry
from .behaviours import (
    DemoRequestClassifierBehaviour,
    PlanManagementBehaviour,
    UserMessageBehaviour,
)
from .utils import build_llm_client, build_llm_call_kwargs, build_behaviour_llm_client, LLMClientConfig

# Global logger will be replaced by per-instance loggers
# logger = logging.getLogger("UserAssistant")

# Fallback defaults when yaml is missing a key. Every production deployment
# provides these via ``config/agents.yaml``; the constants here just keep the
# agent runnable in isolated unit tests.
_DEFAULT_GOAL_REQUEST_TIMEOUT_S = 60.0
_DEFAULT_RPC_CALL_TIMEOUT_S = 15.0
_DEFAULT_SIGNIFIER_MATCH_TIMEOUT_S = 10.0
_DEFAULT_BT_MAX_TICKS = 50
_DEFAULT_BT_MIN_TICK_YIELD_S = 0.01
_DEFAULT_BT_MAINTENANCE_INTERVAL_S = 180.0
_DEFAULT_BT_MAINTENANCE_TRIGGER_POLL_S = 5.0
# How long an action plan waits for a yes/no before assuming confirmation.
# Read-only plans do not wait at all.
_DEFAULT_CONFIRMATION_TIMEOUT_S = 5.0


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

        # ── Per-behaviour LLM clients ──────────────────────────────
        # Each behaviour (atomic segmentation, intent parsing, plan summarization)
        # gets its own configured client.
        self._behaviour_llm_cfgs: Dict[str, LLMClientConfig] = {}

        # Lazy-initialize per-behaviour configs on first access
        self._behaviour_keys = [
            "atomic_segmentation",
            "intent_parsing",
            "plan_summarization",
        ]

        # Legacy support: expose a default client (uses atomic_segmentation config)
        self._default_llm_cfg = build_behaviour_llm_client(config, "atomic_segmentation")
        self.llm_client = self._default_llm_cfg.client
        self.llm_model: str = self._default_llm_cfg.model
        self.llm_base_url: str = self._default_llm_cfg.base_url
        self.llm_temperature: float = self._default_llm_cfg.temperature
        self.llm_reasoning_effort = self._default_llm_cfg.reasoning_effort
        self.llm_max_completion_tokens = self._default_llm_cfg.max_completion_tokens

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
        self.bt_min_tick_yield: float = float(
            config.get("bt_execution", {})
            .get("min_tick_yield", _DEFAULT_BT_MIN_TICK_YIELD_S)
        )
        self.bt_maintenance_interval: float = float(
            config.get("bt_execution", {})
            .get("maintenance_interval", _DEFAULT_BT_MAINTENANCE_INTERVAL_S)
        )
        self.bt_maintenance_trigger_poll: float = float(
            config.get("bt_execution", {})
            .get("maintenance_trigger_poll",
                 _DEFAULT_BT_MAINTENANCE_TRIGGER_POLL_S)
        )

        # ── Execution engine ────────────────────────────────────────
        self.yggdrasil_url = resolve_yggdrasil_url(config)
        self.execution_engine = YggdrasilIntegration(self.yggdrasil_url)
        self._execution_engine_ready = False
        self._execution_engine_lock = asyncio.Lock()

        # ── Per-conversation state ──────────────────────────────────
        self._conversations: Dict[str, ConversationState] = {}

        # ── Request and plan identity ───────────────────────────────
        # Requests and the plans they produce are tracked by id rather than by
        # thread alone, so a running plan stays addressable after the request
        # that asked for it has finished.
        self.requests = RequestRegistry()
        self.plans = PlanRegistry()
        self.confirmation_timeout: float = float(
            (config.get("planning", {}) or {}).get(
                "confirmation_timeout", _DEFAULT_CONFIRMATION_TIMEOUT_S)
        )

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

    async def notify_conversation(self, thread: str, text: str) -> None:
        """Say something on a thread without a message to reply to.

        A plan finishing is not an answer to anything: the request that asked
        for it ended at confirmation, possibly minutes ago. So the notice is
        addressed to the user directly, on the thread the plan belongs to.
        """
        from spade.message import Message

        user_jid = self.target_jids.get("user")
        if not user_jid:
            self.logger.debug(f"No user JID configured; dropping notice: {text}")
            return
        msg = Message(to=str(user_jid))
        if thread and thread != "__default__":
            msg.thread = thread
        msg.set_metadata("message_type", "llm")
        msg.body = text
        await self.plan_manager.send(msg)

    # ── LLM clients and kwargs builders ─────────────────────────────

    def get_llm_client_for_behaviour(self, behaviour_key: str) -> LLMClientConfig:
        """Get or create a behaviour-specific LLM client config.

        Args:
            behaviour_key: One of "atomic_segmentation", "intent_parsing", "plan_summarization"

        Returns:
            LLMClientConfig with behaviour-specific settings.
        """
        if behaviour_key not in self._behaviour_llm_cfgs:
            self._behaviour_llm_cfgs[behaviour_key] = build_behaviour_llm_client(
                self.config, behaviour_key
            )
        return self._behaviour_llm_cfgs[behaviour_key]

    def build_llm_kwargs(self) -> Dict[str, Any]:
        """Build extra kwargs for ``llm_client.chat.completions.create``.

        Uses the default (atomic_segmentation) config for backward compatibility.
        """
        return build_llm_call_kwargs(self._default_llm_cfg)

    def build_llm_kwargs_for_behaviour(self, behaviour_key: str) -> Dict[str, Any]:
        """Build behaviour-specific LLM kwargs."""
        cfg = self.get_llm_client_for_behaviour(behaviour_key)
        return build_llm_call_kwargs(cfg)

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

        # Load the home ontology for per-span intent parsing. This is the single
        # authoritative copy -- the same file the environment layer serves its
        # semantic types from (see SimuHome/td_builder.py), so the vocabulary the
        # LLM is shown always matches the vocabulary it will be matched against.
        ontology_path = (
            Path(__file__).resolve().parents[2] / "shared" / "ontologies" / "homeont.ttl"
        )
        try:
            self.ontology_ttl = ontology_path.read_text() if ontology_path.exists() else ""
            if not self.ontology_ttl:
                self.logger.warning("Ontology file not found at %s", ontology_path)
        except Exception as e:
            self.logger.warning("Failed to load ontology: %s", e)
            self.ontology_ttl = ""

        temp_display = "default" if self.llm_model.startswith("o") else self.llm_temperature
        self.logger.info(
            demo(
                f"UserAssistant booting (model={self.llm_model}, base_url={self.llm_base_url}, "
                f"temperature={temp_display}, reasoning_effort={self.llm_reasoning_effort or 'default'})"
            )
        )

        # Register behaviours with template routing.
        #
        # `dispatch` gives a copy to EVERY behaviour whose template matches, so
        # these do not compete: the receiver takes fresh utterances, while a
        # reply belonging to a request in progress carries `active_request_id`
        # and is matched by that request's own FSM template instead.
        fresh = Template()
        fresh.set_metadata("message_type", "llm")
        self.add_behaviour(UserMessageBehaviour(self.logger), template=fresh)
        self.add_behaviour(DemoRequestClassifierBehaviour(self.logger), template=fresh)

        # One manager for every confirmed plan. Execution is reached by message,
        # never by call, so a plan outlives the request that asked for it.
        plan_types = (
            Template(metadata={"type": MessageType.PLAN_EXECUTE_REQUEST.value})
            | Template(metadata={"type": MessageType.PLAN_STATUS_REQUEST.value})
            | Template(metadata={"type": MessageType.PLAN_CANCEL_REQUEST.value})
            | Template(metadata={"type": MessageType.PLAN_EXPLAIN_REQUEST.value})
            | Template(metadata={"type": MessageType.PLAN_TRIGGER_REQUEST.value})
        )
        self.plan_manager = PlanManagementBehaviour(self.logger)
        self.add_behaviour(self.plan_manager, template=plan_types)

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
