"""
UserAssistant Agent - Main implementation.

Dual implementation with user-facing chat and system-facing plan management.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from spade_llm import LLMAgent
from spade_llm.providers import LLMProvider
from spade.behaviour import CyclicBehaviour
from spade.message import Message as SpadeMessage

from ...shared.protocols.agent_protocol import IAgent, IMessageRouter
from ...shared.models.messages import (
    Message,
    MessageType,
    MessageClassification,
    GoalRequest,
    META_CORRELATION_ID,
    META_CONVERSATION_ID,
    ensure_correlation_id,
    extract_conversation_id,
    serialize_body,
)
from ...shared.utils.spade_rpc import send_via_router
from ...shared.utils.config_resolver import resolve_yggdrasil_url
from ...shared.utils.demo_log import demo
from ...shared.models.plan import Plan, PlanType, PlanStatus

from ...shared.models.messages import MessageType
from .behaviours import ResponseListenerBehaviour

from .prompts import USER_ASSISTANT_SYSTEM_PROMPT
from .tools import (
    QueryCapabilitiesTool,
    QueryEnvironmentStateTool,
    RequestInteractionPlanTool,
    StoreLatestPlanTool,
    RetrieveAndClearLatestPlanTool,
    ExecutePlanTool,
)

from ...environment.integration.integration_engine import YggdrasilIntegration

logger = logging.getLogger("UserAssistant")

class UserAssistantAgent(LLMAgent, IAgent):
    """
    UserAssistant agent with dual functionality:
    1. User-facing: ChatAgent for conversation management
    2. System-facing: Plan management and execution
    """

    def __init__(self, jid: str, password: str, config: Dict[str, Any], target_jids: Dict[str, str]):
        """
        Initialize UserAssistant agent.

        Args:
            jid: SPADE JID for the agent.
            password: SPADE password.
            config: Agent configuration.
        """
        # 1) Setup Provider (OpenAI by default, following agents.yaml structure)
        llm_root = config.get("llm", {}) or {}
        provider_name = llm_root.get("default_provider", "openai")
        provider_cfg = (llm_root.get("providers", {}) or {}).get(provider_name, {}) or {}

        api_key = provider_cfg.get("api_key") or llm_root.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or llm.providers.openai.api_key in agents.yaml.")

        model = provider_cfg.get("model") or "o3"
        temperature = provider_cfg.get("temperature", 0.7)
        max_tokens = provider_cfg.get("max_tokens", None)
        max_completion_tokens = provider_cfg.get("max_completion_tokens", None)
        base_url = provider_cfg.get("base_url") or llm_root.get("base_url") or "https://api.openai.com/v1"
        raw_timeout = (llm_root.get("retry", {}) or {}).get("timeout", None)
        try:
            timeout = float(raw_timeout) if raw_timeout is not None else None
        except Exception:
            timeout = None
        if str(model).startswith("o") and "openai.com" in str(base_url).lower():
            # Reasoning models reject custom temperatures; force the default value.
            temperature = 1.0
            if timeout is None or timeout < 120.0:
                timeout = 120.0

        reasoning_effort = (
            provider_cfg.get("reasoning_effort")
            or llm_root.get("reasoning_effort")
            or os.getenv("OPENAI_REASONING_EFFORT")
        )
        if reasoning_effort is None and str(model).startswith("o") and "openai.com" in str(base_url):
            reasoning_effort = "high"

        # Persist for demo logs (avoid leaking any secrets).
        self.llm_model = str(model)
        self.llm_base_url = str(base_url)
        self.llm_temperature = float(temperature)
        self.llm_reasoning_effort = str(reasoning_effort) if reasoning_effort else None
        self.llm_max_completion_tokens = None
        if str(model).startswith("o"):
            if max_completion_tokens is not None:
                try:
                    self.llm_max_completion_tokens = int(max_completion_tokens)
                except Exception:
                    self.llm_max_completion_tokens = None

        kwargs: Dict[str, Any] = {"base_url": base_url}
        if timeout is not None:
            kwargs["timeout"] = float(timeout)
        if max_tokens is not None and not str(model).startswith("o"):
            kwargs["max_tokens"] = int(max_tokens)

        provider = LLMProvider.create_openai(
            api_key=str(api_key),
            model=str(model),
            temperature=float(temperature),
            **kwargs,
        )

        # Ensure reasoning models use the configured effort level and correct token parameter.
        # spade_llm's provider does not expose `reasoning_effort` or `max_completion_tokens`,
        # so we patch the underlying OpenAI client call.
        if str(model).startswith("o") and (self.llm_reasoning_effort or self.llm_max_completion_tokens is not None):
            try:
                completions = provider.client.chat.completions
                original_create = completions.create

                if not getattr(original_create, "_ami_reasoning_effort_injected", False):
                    def _create_with_reasoning_effort(*args, **kwargs):  # type: ignore[no-redef]
                        if self.llm_reasoning_effort:
                            kwargs.setdefault("reasoning_effort", self.llm_reasoning_effort)
                        # Reasoning models reject custom temperature; rely on default.
                        kwargs.pop("temperature", None)
                        if self.llm_max_completion_tokens is not None:
                            kwargs.setdefault("max_completion_tokens", self.llm_max_completion_tokens)
                        # Reasoning models reject `max_tokens`; ensure we never send it.
                        kwargs.pop("max_tokens", None)
                        return original_create(*args, **kwargs)

                    setattr(_create_with_reasoning_effort, "_ami_reasoning_effort_injected", True)
                    completions.create = _create_with_reasoning_effort  # type: ignore[assignment]
            except Exception:
                # Best-effort; if patching fails, the request will proceed without explicit reasoning_effort.
                pass
 
        # --- Execution engine (YggdrasilIntegration) ---
        # Used by ExecutePlanTool to apply plans to the environment.
        self.yggdrasil_url = resolve_yggdrasil_url(config)
        self.execution_engine = YggdrasilIntegration(self.yggdrasil_url)
        self._execution_engine_ready = False
        self._execution_engine_lock = asyncio.Lock()

        # --- Per-conversation plan state (approval gating) ---
        self._active_conversation_id: str | None = None
        self._plans_by_thread: Dict[str, Dict[str, str]] = {}
        self._approved_plan_hash_by_thread: Dict[str, str] = {}
        self._approved_plan_json_by_thread: Dict[str, str] = {}

        # 2. Setup Tools
        explorer_jid = target_jids.get("explorer")
        self.capabilities_tool = QueryCapabilitiesTool(target_jid=explorer_jid)
        self.state_tool = QueryEnvironmentStateTool(target_jid=explorer_jid)
        self.plan_tool = RequestInteractionPlanTool()
        self.store_plan_tool = StoreLatestPlanTool()
        self.retrieve_plan_tool = RetrieveAndClearLatestPlanTool()
        self.execute_plan_tool = ExecutePlanTool()

        # 3. Initialize Parent (ChatAgent)
        super().__init__(
            jid=jid,
            password=password,
            provider=provider,
            system_prompt=USER_ASSISTANT_SYSTEM_PROMPT,
            tools=[
                self.capabilities_tool,
                self.state_tool,
                self.plan_tool,
                self.store_plan_tool,
                self.retrieve_plan_tool,
                self.execute_plan_tool,
            ],
            verify_security=False
        )

        # 4. Bind Tools
        self.capabilities_tool.set_agent(self)
        self.state_tool.set_agent(self)
        self.plan_tool.set_agent(self)
        self.store_plan_tool.set_agent(self)
        self.retrieve_plan_tool.set_agent(self)
        self.execute_plan_tool.set_agent(self)
        
        # Config & State
        self.config = config
        self.target_jids = target_jids

    # --- Conversation context helpers (used by tools) ---
    @property
    def active_conversation_id(self) -> str | None:
        return self._active_conversation_id

    def store_latest_plan(self, thread: str, plan_json: str, plan_hash: str) -> None:
        self._plans_by_thread[str(thread)] = {"plan_json": plan_json, "plan_hash": plan_hash}

    def retrieve_and_clear_latest_plan(self, thread: str, *, approve: bool) -> Dict[str, str] | None:
        record = self._plans_by_thread.pop(str(thread), None)
        if not record:
            return None
        if approve:
            self._approved_plan_hash_by_thread[str(thread)] = record["plan_hash"]
            self._approved_plan_json_by_thread[str(thread)] = record["plan_json"]
        return record

    def peek_approved_plan_hash(self, thread: str) -> str | None:
        return self._approved_plan_hash_by_thread.get(str(thread))

    def peek_approved_plan_json(self, thread: str) -> str | None:
        return self._approved_plan_json_by_thread.get(str(thread))

    def clear_approved_plan_hash(self, thread: str) -> None:
        self._approved_plan_hash_by_thread.pop(str(thread), None)
        self._approved_plan_json_by_thread.pop(str(thread), None)

    async def ensure_execution_engine_ready(self) -> None:
        """
        Initialize and hydrate the YggdrasilIntegration engine (workspace/artifact/affordance maps).
        Cached so it only runs once per agent lifecycle.
        """
        if self._execution_engine_ready:
            return

        async with self._execution_engine_lock:
            if self._execution_engine_ready:
                return

            ok = await self.execution_engine.initialize({})
            if not ok:
                raise RuntimeError(f"Failed to initialize YggdrasilIntegration at {self.yggdrasil_url}")
            await self.execution_engine.explore_hmas_environment()
            self._execution_engine_ready = True

    class ConversationTrackerBehaviour(CyclicBehaviour):
        """
        Tracks the most recent LLM conversation thread for tool calls.

        SPADE dispatches each incoming message to matching behaviours independently,
        so this does not interfere with SPADE-LLM's internal LLMBehaviour.
        """

        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return
            if msg.get_metadata("message_type") != "llm":
                return
            thread = getattr(msg, "thread", None)
            if thread:
                self.agent._active_conversation_id = str(thread)

    class DemoRequestClassifierBehaviour(CyclicBehaviour):
        """
        Demo-only logging helper: classify user requests as EXPLICIT vs IMPLICIT.

        This does not influence planning; it only emits a readable log line for thesis demos.
        """

        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return
            if msg.get_metadata("message_type") != "llm":
                return

            text = (msg.body or "").strip()
            low = text.lower()
            if low in ("yes", "no", "ok", "okay", "proceed", "continue"):
                return

            # Basic categorization:
            # - QUERY: listing/state questions (no plan expected)
            # - EXPLICIT: direct device actions (e.g., "turn off light308", "open blinds to 50%")
            # - IMPLICIT: comfort/goal statements (e.g., "it's dark", "I can't see on my desk")
            kind = "IMPLICIT"
            if low.startswith(("what", "show", "list", "which", "is", "are")) and (
                "workspace" in low or "workspaces" in low or "device" in low or "devices" in low or "state" in low
            ):
                kind = "QUERY"
            else:
                has_action = any(
                    kw in low
                    for kw in (
                        "turn ",
                        "toggle",
                        "open",
                        "close",
                        "set ",
                        "raise",
                        "lower",
                        "increase",
                        "decrease",
                    )
                )
                mentions_device = (
                    any(tok in low for tok in ("light", "blinds"))
                    or re.search(r"\b\w+\d{3}\b", low) is not None
                    or "%" in low
                )
                if has_action and mentions_device:
                    kind = "EXPLICIT"

            logger.info(demo("Request classified as %s: %r"), kind, text)

    async def setup(self):
        """
        Setup the agent (SPADE lifecycle method).

        TODO: Implementation steps:
        1. Initialize MessageRouter
        2. Initialize ChatAgent component
        3. Initialize PlanManager component
        4. Initialize MemoryManager
        5. Register all behaviors
        6. Subscribe to message topics
        7. Log agent ready
        """
        await super().setup()

        temp_display = "default" if str(self.llm_model).startswith("o") else self.llm_temperature
        logger.info(
            demo("UserAssistant booting (model=%s, base_url=%s, temperature=%s, reasoning_effort=%s)"),
            self.llm_model,
            self.llm_base_url,
            temp_display,
            self.llm_reasoning_effort or "default",
        )

        # Track active conversation thread so tools can route messages and store plans per-conversation.
        from spade.template import Template

        t = Template()
        t.set_metadata("message_type", "llm")
        self.add_behaviour(self.ConversationTrackerBehaviour(), template=t)
        self.add_behaviour(self.DemoRequestClassifierBehaviour(), template=t)

        # --- DEBUG: Log Registered Behaviors and Templates ---
        logger.info("=== DEBUG: Inspecting UserAssistant Behaviors ===")
        if not self.behaviours:
            logger.warning("No behaviors registered! LLMAgent might have failed to init.")

        for behaviour in self.behaviours:
            logger.info(f"Behaviour: {type(behaviour).__name__}")
            t = getattr(behaviour, "template", None)
            if t:
                logger.info("  Template Rules:")
                logger.info(f"    Sender: {t.sender}")
                logger.info(f"    Thread: {t.thread}")
                logger.info(f"    Metadata: {t.metadata}")
            else:
                logger.info("  Template: None (Should match EVERYTHING)")
        logger.info("===============================================")
        # -----------------------------------------------------

        logger.info("UserAssistantAgent initialized (Intent Extraction Mode).")

    async def start(self, *args, **kwargs) -> None:
        """
        Start the UserAssistant agent.

        TODO: Implementation steps:
        1. Call SPADE start method
        2. Wait for connection to SPADE server
        3. Trigger setup
        4. Notify other agents that UserAssistant is ready
        """
        return await super().start(*args, **kwargs)

    async def stop(self) -> None:
        """
        Stop the UserAssistant agent.

        TODO: Implementation steps:
        1. Stop all behaviors
        2. Save active plans and conversations
        3. Disconnect from SPADE server
        4. Cleanup resources
        """
        return await super().stop()

    async def send_message(self, message: Message | SpadeMessage) -> bool:
        """
        Send a message to another agent.

        TODO: Implementation steps:
        1. Validate message
        2. Serialize to SPADE message format
        3. Send via SPADE
        4. Log sent message
        5. Return success status
        """
        # If caller already passed a SPADE message, send it directly.
        if isinstance(message, SpadeMessage):
            await send_via_router(self, message)
            return True

        # Convert internal Message dataclass to SPADE Message.
        spade_msg = SpadeMessage(to=message.receiver)
        spade_msg.set_metadata("type", message.message_type.value)

        # Propagate optional metadata (e.g., correlation_id).
        for key, value in (message.metadata or {}).items():
            if value is not None:
                spade_msg.set_metadata(key, str(value))

        # Ensure correlation_id (request/response pairing).
        corr_id = ensure_correlation_id(message.metadata)
        spade_msg.set_metadata(META_CORRELATION_ID, str(corr_id))

        # Map conversation_id -> XMPP thread (and also keep metadata copy).
        conv_id = extract_conversation_id(message.conversation_id, message.metadata)
        if conv_id:
            spade_msg.thread = conv_id
            spade_msg.set_metadata(META_CONVERSATION_ID, conv_id)

        # Serialize content (prefer JSON for agent-to-agent).
        spade_msg.body = serialize_body(message.content)

        await send_via_router(self, spade_msg)
        return True

    async def receive_message(self, message: Message) -> None:
        """
        Receive and route a message.

        TODO: Implementation steps:
        1. Deserialize SPADE message
        2. Create Message object
        3. Pass to router for handling
        """
        pass


class MessageReceiveBehaviour(CyclicBehaviour):
    """Behavior for receiving messages from other agents."""

    async def run(self):
        """
        Continuously receive and process messages.

        TODO: Implementation steps:
        1. Wait for incoming SPADE message
        2. Deserialize message
        3. Pass to agent's receive_message method
        4. Handle any errors
        """
        pass


class PlanManager:
    """
    System-facing plan management functionality.
    """

    def __init__(self, config: Dict[str, Any], memory_manager):
        """
        Initialize PlanManager.

        Args:
            config: Plan management configuration.
            memory_manager: Memory manager instance.
        """
        self.config = config
        self.memory_manager = memory_manager
        self.storage = None  # Database for plan storage
        self.running_plans = {}
        self.maintenance_plans = {}

    async def initialize_storage(self) -> None:
        """
        Initialize plan storage backend.

        TODO: Implementation steps:
        1. Connect to database (SQLite, PostgreSQL, MongoDB)
        2. Create tables/collections if not exist
        3. Load active plans into memory
        """
        pass

    async def create_plan(self, goal: GoalRequest, behavior_tree: Dict[str, Any]) -> Plan:
        """
        Create a new plan.

        TODO: Implementation steps:
        1. Generate unique plan_id
        2. Determine plan type (immediate or maintenance)
        3. Create Plan object with metadata
        4. Store in database
        5. If immediate, add to running_plans
        6. If maintenance, add to maintenance_plans
        7. Return Plan
        """
        pass

    async def execute_plan(self, plan_id: str) -> bool:
        """
        Execute a plan.

        TODO: Implementation steps:
        1. Retrieve plan
        2. Update status to RUNNING
        3. Execute behavior tree nodes
        4. Monitor execution status
        5. Update timestamps
        6. Handle completion or failure
        7. Return execution result
        """
        pass

    async def cancel_plan(self, plan_id: str) -> bool:
        """
        Cancel a running or maintenance plan.

        TODO: Implementation steps:
        1. Retrieve plan
        2. Stop execution if running
        3. Update status to CANCELLED
        4. Remove from active plans
        5. Persist changes
        6. Return success status
        """
        pass

    async def alter_plan(self, plan_id: str, modifications: Dict[str, Any]) -> bool:
        """
        Alter an existing plan.

        TODO: Implementation steps:
        1. Retrieve plan
        2. Apply modifications to behavior tree
        3. Validate modified plan
        4. Update plan in database
        5. If running, restart execution
        6. Return success status
        """
        pass

    async def repeat_plan(self, plan_id: str) -> str:
        """
        Repeat a previous plan execution.

        TODO: Implementation steps:
        1. Retrieve plan
        2. Create new plan instance with same behavior tree
        3. Execute new plan
        4. Return new plan_id
        """
        pass

    async def get_plan_summary(self, plan_id: str) -> str:
        """
        Get human-readable summary of a plan.

        TODO: Implementation steps:
        1. Retrieve plan
        2. Use LLM to generate summary
        3. Return summary
        """
        pass

    async def check_maintenance_triggers(self, environment_state: Dict[str, Any]) -> List[str]:
        """
        Check if any maintenance plans should be triggered.

        TODO: Implementation steps:
        1. Iterate through maintenance_plans
        2. For each, check if triggering conditions are met
        3. Collect plan_ids that should be executed
        4. Return list of plan_ids
        """
        pass

    async def handle_plan_impact_notification(self, change_event: Any) -> None:
        """
        Handle notification about environment changes affecting plans.

        TODO: Implementation steps:
        1. Receive change event from EnvExplorer
        2. Check which maintenance plans are affected
        3. Trigger affected plans if conditions now met
        4. Notify user if needed
        """
        pass

    async def store_preference(self, preference: Dict[str, Any]) -> None:
        """
        Store a user preference.

        TODO: Implementation steps:
        1. Extract preference details
        2. Associate with relevant goals/plans
        3. Store in database
        4. Update plan matching to consider preferences
        """
        pass
