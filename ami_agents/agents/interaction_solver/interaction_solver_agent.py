"""InteractionSolver Agent — goal planning and execution.

Slim SPADE agent. State + lifecycle only. All planning work lives in
behaviours/ and utils/ — the agent itself does no orchestration beyond
registering behaviours and exposing shared resources (LLM client,
EnvExplorer RPC wrapper, BT planner, community client).
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from spade.agent import Agent
from spade.message import Message as SpadeMessage
from spade.template import Template

from ...bt_planning.planning.bt_planner import AsyncBTPlanner
from ...shared.community.community_client import CommunitySignifierClient
from ...shared.models.messages import (
    META_CONVERSATION_ID,
    META_CORRELATION_ID,
    Message,
    MessageType,
    ensure_correlation_id,
    extract_conversation_id,
    serialize_body,
)
from ...shared.protocols.agent_protocol import IAgent
from ...shared.utils.demo_log import demo
from ...shared.utils.logger import LoggerFactory
from ...shared.utils.spade_rpc import RpcTimeoutError, rpc_call, send_via_router
from .behaviours import EnvironmentReadyBehaviour, GoalRequestBehaviour
from .utils import LLMClientConfig, build_llm_client


# Fallback defaults used only when yaml is missing a key.
_DEFAULT_CONTEXT_TIMEOUT_S = 10.0
_DEFAULT_COMMUNITY_TIMEOUT_S = 10.0
_DEFAULT_MAX_PLANNING_ATTEMPTS = 3


class InteractionSolverAgent(Agent, IAgent):
    """InteractionSolver agent: pure state + behaviour registration."""

    def __init__(
        self,
        jid: str,
        password: str,
        config: Dict[str, Any],
        target_jids: Optional[Dict[str, str]] = None,
    ):
        super().__init__(jid, password)
        self.config = config or {}
        self.target_jids = target_jids or {}

        logging_config = self.config.get("logging", {})
        self.logger = LoggerFactory.get_logger(f"InteractionSolver[{jid}]", logging_config)

        # ── Environment readiness state ─────────────────────────────
        self.environment_ready = False
        self._environment_ready_event: asyncio.Event = asyncio.Event()
        self._last_env_ready_payload: Optional[Dict[str, Any]] = None
        self.env_explorer_jid: Optional[str] = None

        # Cached environment snapshot (filled by EnvContextQueryBehaviour).
        self._cached_affordances: List[Dict[str, Any]] = []
        self._cached_state_payload: Dict[str, Any] = {}
        self._cached_affordance_by_id: Dict[str, Dict[str, Any]] = {}

        # ── LLM client ──────────────────────────────────────────────
        self._llm_cfg: LLMClientConfig = build_llm_client(self.config)
        self.llm_client = self._llm_cfg.client
        self.model = self._llm_cfg.model
        self.base_url = self._llm_cfg.base_url
        self.temperature = self._llm_cfg.temperature
        self.reasoning_effort = self._llm_cfg.reasoning_effort
        self.max_tokens = self._llm_cfg.max_tokens
        self.max_completion_tokens = self._llm_cfg.max_completion_tokens

        # ── Timeouts / limits (yaml with fallbacks) ─────────────────
        context_cfg = (self.config.get("planning", {}) or {}).get("context_gathering", {}) or {}
        self.context_timeout: float = float(
            context_cfg.get("timeout", _DEFAULT_CONTEXT_TIMEOUT_S)
        )
        # ``GoalRequestBehaviour`` waits this long for EnvExplorer discovery.
        self.env_ready_timeout: float = self.context_timeout

        planning_llm = (self.config.get("planning", {}) or {}).get("llm_planning", {}) or {}
        max_attempts = int(
            planning_llm.get("max_planning_attempts", _DEFAULT_MAX_PLANNING_ATTEMPTS)
        )
        self.bt_planner = AsyncBTPlanner(max_attempts=max_attempts)

        # ── Community signifier client (optional) ───────────────────
        community_cfg = (self.config.get("planning", {}) or {}).get("community", {}) or {}
        community_url = community_cfg.get("api_url") or os.getenv("COMMUNITY_API_URL")
        self.community_enabled = bool(community_cfg.get("enabled", bool(community_url)))
        self.community_client: Optional[CommunitySignifierClient] = None
        if self.community_enabled and community_url:
            community_timeout = float(
                community_cfg.get("timeout", _DEFAULT_COMMUNITY_TIMEOUT_S)
            )
            self.community_client = CommunitySignifierClient(
                api_url=community_url, timeout=community_timeout
            )

    # ── Environment-ready signalling ────────────────────────────────

    def mark_environment_ready(
        self,
        *,
        sender_jid: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.environment_ready = True
        if sender_jid:
            self.env_explorer_jid = sender_jid
        if payload is not None:
            self._last_env_ready_payload = payload
        if not self._environment_ready_event.is_set():
            self._environment_ready_event.set()

    async def await_environment_ready(self, timeout: float = 10.0) -> bool:
        """Wait until EnvExplorer has completed initial discovery.

        - If already ready: return immediately.
        - Else: try a capabilities RPC (covers missed notifications).
        - Else: wait for the ENV_DISCOVERY_COMPLETE event (up to ``timeout``).
        """
        if self.environment_ready:
            return True

        try:
            raw = await self._query_env_explorer(
                message_type=MessageType.ENV_CAPABILITIES_REQUEST.value,
                body={"query": "all"},
                expect_type=MessageType.ENV_CAPABILITIES_RESPONSE.value,
            )
            payload = json.loads(raw or "{}")
            if payload.get("discovery_complete") is True:
                self.mark_environment_ready(payload=payload)
                return True
        except Exception:
            pass

        try:
            await asyncio.wait_for(self._environment_ready_event.wait(), timeout=timeout)
            return self.environment_ready
        except asyncio.TimeoutError:
            return False

    # ── EnvExplorer RPC (shared by behaviours) ──────────────────────

    async def _query_env_explorer(
        self,
        message_type: str,
        body: Dict[str, Any],
        expect_type: str,
    ) -> str:
        """Send a SPADE message to EnvExplorer and return the raw response body."""
        explorer_jid = self.target_jids.get("explorer")
        if not explorer_jid:
            return json.dumps({"error": "explorer_jid_not_configured"})

        try:
            result = await rpc_call(
                self,
                to_jid=explorer_jid,
                request_type=message_type,
                body=body,
                expect_type=expect_type,
                timeout=self.context_timeout,
            )
            return result.body or ""
        except RpcTimeoutError:
            return json.dumps({"error": "timeout"})
        except Exception as e:
            return json.dumps({"error": "rpc_failed", "detail": str(e)})

    # ── SPADE lifecycle ─────────────────────────────────────────────

    async def setup(self):
        await super().setup()

        env_ready_template = Template()
        env_ready_template.set_metadata("type", MessageType.ENV_DISCOVERY_COMPLETE.value)
        self.add_behaviour(EnvironmentReadyBehaviour(self.logger), template=env_ready_template)

        goal_template = Template()
        goal_template.set_metadata("type", MessageType.GOAL_REQUEST.value)
        self.add_behaviour(GoalRequestBehaviour(self.logger), template=goal_template)

        temp_display = "default" if self.model.startswith("o") else self.temperature
        self.logger.info(
            demo(
                "InteractionSolver booting (model=%s, base_url=%s, temperature=%s, "
                "reasoning_effort=%s)"
            ),
            self.model,
            self.base_url,
            temp_display,
            self.reasoning_effort or "default",
        )
        self.logger.info("InteractionSolverAgent initialized (pure SPADE + behaviour-driven planning).")

    async def start(self, *args, **kwargs) -> None:
        return await super().start(*args, **kwargs)

    async def stop(self) -> None:
        return await super().stop()

    # ── IAgent protocol ─────────────────────────────────────────────

    async def send_message(self, message: Message) -> bool:
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
        # Behaviours route incoming messages via SPADE templates.
        return
