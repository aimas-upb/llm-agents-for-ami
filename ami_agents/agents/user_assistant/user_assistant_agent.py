"""
UserAssistant Agent — refactored as a plain SPADE Agent.

The LLM is a *component* (``AsyncOpenAI`` client) used by behaviours for
NLU (intent extraction) and NLG (plan summary / query formatting).
Everything else — plan requesting, confirmation handling, execution,
signifier recording — is handled by deterministic SPADE behaviours.
"""

import asyncio
import logging
import os
from typing import Any, Dict

from openai import AsyncOpenAI
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
from ...shared.state_memory import StateMemoryCache

from ...environment.integration.integration_engine import YggdrasilIntegration

from .models import ConversationState
from .behaviours import UserMessageBehaviour, DemoRequestClassifierBehaviour

logger = logging.getLogger("UserAssistant")


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

        # ── LLM client (component, not base class) ──────────────────
        llm_root = config.get("llm", {}) or {}
        provider_name = llm_root.get("default_provider", "openai")
        provider_cfg = (llm_root.get("providers", {}) or {}).get(provider_name, {}) or {}

        api_key = (
            provider_cfg.get("api_key")
            or llm_root.get("api_key")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Missing OpenAI API key. Set OPENAI_API_KEY or "
                "llm.providers.openai.api_key in agents.yaml."
            )

        model = provider_cfg.get("model") or "o3"
        temperature = provider_cfg.get("temperature", 0.7)
        max_completion_tokens = provider_cfg.get("max_completion_tokens")
        base_url = (
            provider_cfg.get("base_url")
            or llm_root.get("base_url")
            or "https://api.openai.com/v1"
        )
        raw_timeout = (llm_root.get("retry", {}) or {}).get("timeout")
        try:
            timeout = float(raw_timeout) if raw_timeout is not None else None
        except Exception:
            timeout = None

        # Reasoning-model adjustments (o-series)
        is_reasoning = str(model).startswith("o")
        if is_reasoning and "openai.com" in str(base_url).lower():
            temperature = 1.0
            if timeout is None or timeout < 120.0:
                timeout = 120.0

        reasoning_effort = (
            provider_cfg.get("reasoning_effort")
            or llm_root.get("reasoning_effort")
            or os.getenv("OPENAI_REASONING_EFFORT")
        )
        if reasoning_effort is None and is_reasoning and "openai.com" in str(base_url):
            reasoning_effort = "high"

        # Persist for demo logs and behaviour access
        self.llm_model: str = str(model)
        self.llm_base_url: str = str(base_url)
        self.llm_temperature: float = float(temperature)
        self.llm_reasoning_effort: str | None = str(reasoning_effort) if reasoning_effort else None
        self.llm_max_completion_tokens: int | None = None
        if is_reasoning and max_completion_tokens is not None:
            try:
                self.llm_max_completion_tokens = int(max_completion_tokens)
            except Exception:
                pass

        # Create AsyncOpenAI client
        client_kwargs: Dict[str, Any] = {"api_key": str(api_key), "base_url": base_url}
        if timeout is not None:
            client_kwargs["timeout"] = float(timeout)
        self.llm_client = AsyncOpenAI(**client_kwargs)

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
            from ...shared.community.community_client import CommunitySignifierClient

            self.community_client = CommunitySignifierClient(api_url=community_url)
            logger.info("Community signifier client enabled (url=%s)", community_url)

    # ── Conversation state helpers ──────────────────────────────────

    def get_conversation(self, thread: str) -> ConversationState:
        """Get or create per-conversation state."""
        if thread not in self._conversations:
            self._conversations[thread] = ConversationState()
        return self._conversations[thread]

    # ── LLM kwargs builder ──────────────────────────────────────────

    def build_llm_kwargs(self) -> Dict[str, Any]:
        """Build extra kwargs for ``llm_client.chat.completions.create``."""
        kwargs: Dict[str, Any] = {}
        if not str(self.llm_model).startswith("o"):
            kwargs["temperature"] = self.llm_temperature
        else:
            # Reasoning models: no temperature, use reasoning_effort
            if self.llm_reasoning_effort:
                kwargs["reasoning_effort"] = self.llm_reasoning_effort
            if self.llm_max_completion_tokens is not None:
                kwargs["max_completion_tokens"] = self.llm_max_completion_tokens
        return kwargs

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

        temp_display = "default" if str(self.llm_model).startswith("o") else self.llm_temperature
        logger.info(
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
        self.add_behaviour(UserMessageBehaviour(), template=t)
        self.add_behaviour(DemoRequestClassifierBehaviour(), template=t)

        logger.info("UserAssistantAgent initialized (composable behaviours).")

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
        pass
