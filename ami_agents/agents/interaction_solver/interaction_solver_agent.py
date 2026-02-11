"""
InteractionSolver Agent - Goal planning and execution.

Classical SPADE agent that calls an LLM to create plans.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Coroutine
from enum import Enum
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message as SpadeMessage
from spade.template import Template
from openai import AsyncOpenAI

from ...shared.protocols.agent_protocol import IAgent
from ...shared.models.messages import (
    Message, MessageType, GoalRequest, AffordanceMatchRequest, AffordanceMatchResponse
)
from ...shared.models.messages import (
    META_CORRELATION_ID,
    META_CONVERSATION_ID,
    ensure_correlation_id,
    extract_conversation_id,
    serialize_body,
)
from ...shared.models.community import Community
from ...shared.utils.spade_rpc import rpc_call, RpcTimeoutError, send_via_router
from ...shared.utils.demo_log import demo
from ...shared.models.plan import BehaviorTreePlan, NodeTemplate, Plan
from ...shared.protocols.llm_protocol import IPlanGenerator

logger = logging.getLogger("InteractionSolver")

# Threshold for community context matching (0.0 to 1.0)
COMMUNITY_MATCH_THRESHOLD = 0.5
DEFAULT_COMMUNITY_QUERY_TIMEOUT = 30.0
DEFAULT_COMMUNITY_MIN_RESPONSE_RATIO = 0.5

# System prompt for strict multi-intent JSON-Plan 1.2 planning.
INTERACTION_SOLVER_SYSTEM_PROMPT = """
You are Interaction-Solver. You receive a list of intents and must return an executable plan in JSON-Plan 1.2 format.

You will be given environment context (affordances + state) in the user message. You MUST NOT invent actions, URIs, or methods.

MULTI-INTENT RULE (STRICT):
- The plan must satisfy ALL provided intents.
- Each step.intent MUST match exactly one of the provided intents (do not rephrase, change case, or combine intents).        
- If you cannot satisfy all intents using the provided affordances, return a strict failure JSON:
  {"plan_version":"1.2","error":"infeasible","detail":"...","steps":[]}   

PLAN FORMAT  JSON-Plan 1.2
---------------------------
{
  "plan_version": "1.2",
  "steps": [
    {
      "step_id": 1,
      "intent": "<exact intent this step fulfills from the list of intents>",
      "artifact_uri": "<full URI>",
      "affordance_uri": "<full URI>",
      "action_name": "<td:name>",
      "method": "<HTTP verb>",
      "target": "<hctl:hasTarget URI>",
      "content_type": "application/json",
      "payload": { },
      "reasons": [
        {
          "property":  "<property satisfied>",
          "direction": "<increase|decrease|set>",
            "evidence": [
              {
                "artifact":  "<sensor-or-artifact URI>",
                "property":  "<attribute name>",
                "operator":  "<lessThan|lessEqual|greaterThan|greaterEqual|equals>",
                "threshold": "<number|string>",
                "reading":   "<number|string>"
              }
            ],
          "why": "One or two sentences explaining why this step is needed."
        }
      ]
    }
  ]
}

Rules:
  - Always set plan_version to "1.2".
  - Every step must include at least one reasons entry with at least one evidence item.
  - Use complete, non-fabricated URIs. If absent, clearly indicate placeholders.
  - Payload values MUST be valid JSON types: use numbers for numeric values (not quoted strings), booleans for true/false.
  - When setting a parameter, the payload keys MUST match the affordance payload schema exactly (including casing).
  - Use ASCII only in all string fields (no curly quotes, no em/en dashes, no ellipsis character).
  - Output only the JSON plan (no additional prose). If no plan is feasible, return a JSON with plan_version=1.2, error, detail, steps=[].
"""

DEFAULT_TIMEOUT = 10


class PlanningPhase(Enum):
    """Phases of plan construction."""
    INITIATED = "initiated"
    GATHERING_REUSED_PLAN = "gathering_reused_plan"
    GENERATING_LOCAL_PLAN = "generating_local_plan"
    QUERYING_COMMUNITY = "querying_community"
    COMPLETED_SUCCESS = "completed_success"
    COMPLETED_FAILURE = "completed_failure"


class GoalStatus:
    """Structure to track goal planning status and information."""

    def __init__(self, goal_id: str, intents: List[str], workspace_id: Optional[str] = None):
        self.goal_id = goal_id
        self.intents = intents
        self.workspace_id = workspace_id
        self.phase = PlanningPhase.INITIATED
        self.created_at = asyncio.get_event_loop().time()

        # Plan information
        self.reused_plan: Optional[Dict[str, Any]] = None
        self.reused_plan_source: Optional[str] = None  # signifier_id or community agent

        self.local_plan: Optional[Dict[str, Any]] = None
        self.local_plan_complete: bool = False

        # Community interaction
        self.relevant_communities: List[str] = []
        self.community_responses: Dict[str, Dict[str, Any]] = {}  # agent_jid -> response
        self.community_expected_responses: int = 0
        self.continue_triggered: bool = False  # Flag to prevent double execution

        # Best plan selection
        self.best_plan: Optional[Dict[str, Any]] = None
        self.best_plan_source: Optional[str] = None  # "reused", "local", or "community"

        # Context storage
        self.context: Optional[Dict[str, Any]] = None

        # Metadata
        self.last_updated = self.created_at
        self.error: Optional[str] = None
        self.error_detail: Optional[str] = None

    def update_status(self, phase: Optional[PlanningPhase] = None, **kwargs):
        """Update goal status and metadata."""
        if phase is not None:
            self.phase = phase

        self.last_updated = asyncio.get_event_loop().time()

        # Update any provided fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary for serialization."""
        return {
            "goal_id": self.goal_id,
            "intents": self.intents,
            "workspace_id": self.workspace_id,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "reused_plan": self.reused_plan,
            "reused_plan_source": self.reused_plan_source,
            "local_plan": self.local_plan,
            "local_plan_complete": self.local_plan_complete,
            "relevant_communities": self.relevant_communities,
            "community_responses": self.community_responses,
            "best_plan": self.best_plan,
            "best_plan_source": self.best_plan_source,
            "context": self.context,
            "error": self.error,
            "error_detail": self.error_detail,
        }


class InteractionSolverAgent(Agent, IAgent):
    """
    InteractionSolver agent for planning and goal resolution.

    Responsibilities:
    - Receive goal requests from UserAssistant
    - Gather planning context (affordances from EnvExplorer)
    - Generate behavior tree plans using LLM
    - Monitor plan execution
    - Support community-based planning (future)
    """

    def __init__(self, jid: str, password: str, config: Dict[str, Any],
                 plan_generator: IPlanGenerator, target_jids: Optional[Dict[str, str]] = None):
        """
        Initialize InteractionSolver agent.

        Args:
            jid: SPADE JID for the agent.
            password: SPADE password.
            config: Agent configuration.
            plan_generator: LLM-based plan generator.
        """
        super().__init__(jid, password)
        self.config = config or {}
        self.plan_generator = plan_generator  # retained for future behaviour-tree execution
        self.environment_ready = False
        self._environment_ready_event: asyncio.Event = asyncio.Event()
        self._last_env_ready_payload: Optional[Dict[str, Any]] = None
        self.env_explorer_jid = None
        self.target_jids = target_jids or {}

        # Goal tracking dictionary: goal_id -> GoalStatus
        self.goal_list: Dict[str, GoalStatus] = {}

        # Communities this agent belongs to or knows about
        self.communities: List[Community] = []

        # Cache the last known environment context so we can build plans from signifiers
        # without querying EnvExplorer again for capabilities/state.
        self._cached_affordances: List[Dict[str, Any]] = []
        self._cached_state_payload: Dict[str, Any] = {}
        self._cached_affordance_by_id: Dict[str, Dict[str, Any]] = {}

        # --- LLM client config (direct OpenAI call) ---
        llm_root = self.config.get("llm", {}) or {}
        provider_name = llm_root.get("default_provider", "openai")
        provider_cfg = (llm_root.get("providers", {}) or {}).get(provider_name, {}) or {}
        planning_llm = (self.config.get("planning", {}) or {}).get("llm_planning", {}) or {}

        self.model = planning_llm.get("model") or provider_cfg.get("model") or "o3"
        self.temperature = planning_llm.get("temperature", provider_cfg.get("temperature", 0.7))
        max_tokens_cfg = planning_llm.get("max_tokens", provider_cfg.get("max_tokens", None))
        max_completion_tokens_cfg = planning_llm.get("max_completion_tokens", provider_cfg.get("max_completion_tokens", None))

        self.max_tokens = None
        self.max_completion_tokens = None
        if str(self.model).startswith("o"):
            # Reasoning models reject `max_tokens`; only use max_completion_tokens when explicitly provided.
            if max_completion_tokens_cfg is not None:
                try:
                    self.max_completion_tokens = int(max_completion_tokens_cfg)
                except Exception:
                    self.max_completion_tokens = None
        else:
            raw_mt = max_tokens_cfg if max_tokens_cfg is not None else 1500
            try:
                self.max_tokens = int(raw_mt) if raw_mt is not None else None
            except Exception:
                self.max_tokens = 1500
        self.reasoning_effort = (
            planning_llm.get("reasoning_effort")
            or provider_cfg.get("reasoning_effort")
            or llm_root.get("reasoning_effort")
            or os.getenv("OPENAI_REASONING_EFFORT")
        )

        # Prefer explicit YAML config; fall back to environment variable.
        api_key = provider_cfg.get("api_key") or llm_root.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = provider_cfg.get("base_url") or llm_root.get("base_url") or "https://api.openai.com/v1"
        raw_timeout = (llm_root.get("retry", {}) or {}).get("timeout", None)
        try:
            timeout = float(raw_timeout) if raw_timeout is not None else 30.0
        except Exception:
            timeout = 30.0
        if str(self.model).startswith("o") and "openai.com" in str(base_url).lower() and timeout < 120.0:
            timeout = 120.0

        self.base_url = str(base_url)
        if self.reasoning_effort is None and str(self.model).startswith("o") and "openai.com" in self.base_url:
            self.reasoning_effort = "high"
        self.llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def mark_environment_ready(self, *, sender_jid: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        self.environment_ready = True
        if sender_jid:
            self.env_explorer_jid = sender_jid
        if payload is not None:
            self._last_env_ready_payload = payload
        if not self._environment_ready_event.is_set():
            self._environment_ready_event.set()

    async def await_environment_ready(self, timeout: float = 10.0) -> bool:
        """
        Wait until EnvExplorer completed initial discovery.

        Strategy:
        - if already ready: return immediately
        - else: try a capabilities RPC once (covers missed notifications)
        - else: wait for ENV_DISCOVERY_COMPLETE event (up to timeout)
        """
        if self.environment_ready:
            return True

        # Fallback probe (in case notification was sent before we attached the behaviour)
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
            # ignore probe failures; we'll fall back to waiting for the event
            pass

        try:
            await asyncio.wait_for(self._environment_ready_event.wait(), timeout=timeout)
            return self.environment_ready
        except asyncio.TimeoutError:
            return False

    async def setup(self):
        """
        Setup the agent (SPADE lifecycle method).

        TODO: Implementation steps:
        1. Register behaviors:
           - EnvironmentReadyBehaviour (wait for discovery)
           - GoalRequestBehaviour (handle goal requests)
           - PlanningStatusBehaviour (handle status requests)
           - MessageReceiveBehaviour (general messages)
        2. Initialize plan execution monitor
        3. Log agent ready
        """
        await super().setup()

        # Listen for EnvExplorer readiness notification
        env_ready_template = Template()
        env_ready_template.set_metadata("type", MessageType.ENV_DISCOVERY_COMPLETE.value)
        self.add_behaviour(EnvironmentReadyBehaviour(), template=env_ready_template)

        # Register GOAL_REQUEST handler with template to avoid cross-behaviour pickup
        goal_template = Template()
        goal_template.set_metadata("type", MessageType.GOAL_REQUEST.value)
        self.add_behaviour(GoalRequestBehaviour(), template=goal_template)

        # Register PLANNING_STATUS_REQUEST handler
        status_template = Template()
        status_template.set_metadata("type", MessageType.PLANNING_STATUS_REQUEST.value)
        self.add_behaviour(PlanningStatusBehaviour(), template=status_template)

        # Register COMMUNITY_REQUEST handler
        community_request_template = Template()
        community_request_template.set_metadata("type", MessageType.COMMUNITY_REQUEST.value)
        self.add_behaviour(CommunityRequestBehaviour(), template=community_request_template)

        # Register COMMUNITY_RESPONSE handler
        community_response_template = Template()
        community_response_template.set_metadata("type", MessageType.COMMUNITY_RESPONSE.value)
        self.add_behaviour(CommunityResponseBehaviour(), template=community_response_template)

        temp_display = "default" if str(self.model).startswith("o") else self.temperature
        logger.info(
            demo("InteractionSolver booting (model=%s, base_url=%s, temperature=%s, reasoning_effort=%s)"),
            self.model,
            self.base_url,
            temp_display,
            self.reasoning_effort or "default",
        )
        logger.info("InteractionSolverAgent initialized (pure SPADE + direct LLM planning).")

    async def start(self, *args, **kwargs) -> None:
        """
        Start the InteractionSolver agent.

        TODO: Implementation steps:
        1. Call SPADE start method
        2. Wait for connection to SPADE server
        3. Trigger setup
        4. Wait for environment discovery notification
        """
        return await super().start(*args, **kwargs)

    async def stop(self) -> None:
        """
        Stop the InteractionSolver agent.

        TODO: Implementation steps:
        1. Stop all behaviors
        2. Stop any running plan executions
        3. Cleanup resources
        4. Disconnect from SPADE server
        """
        return await super().stop()

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
        # Support both SPADE Message and our dataclass Message (basic passthrough)
        if isinstance(message, SpadeMessage):
            await send_via_router(self, message)
        else:
            spade_msg = SpadeMessage(to=message.receiver)
            spade_msg.set_metadata("type", message.message_type.value)
            # Propagate optional metadata.
            for key, value in (message.metadata or {}).items():
                if value is not None:
                    spade_msg.set_metadata(key, str(value))

            # Ensure correlation_id.
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
        Receive and process a message.

        TODO: Implementation steps:
        1. Deserialize SPADE message
        2. Route based on message type
        3. Handle appropriately
        """
        # Behaviours handle routing; no central router needed here.
        return

    async def _query_env_explorer(self, message_type: str, body: Dict[str, Any], expect_type: str, sender_behaviour=None) -> str:
        """
        Send a SPADE message to EnvExplorer and wait for the expected response type.
        """
        explorer_jid = self.target_jids.get("explorer")
        if not explorer_jid:
            return json.dumps({"error": "explorer_jid_not_configured"})

        timeout = self.config.get("planning", {}).get("context_gathering", {}).get("timeout", DEFAULT_TIMEOUT)
        try:
            result = await rpc_call(
                self,
                to_jid=explorer_jid,
                request_type=message_type,
                body=body,
                expect_type=expect_type,
                timeout=float(timeout),
            )
            return result.body or ""
        except RpcTimeoutError:
            return json.dumps({"error": "timeout"})
        except Exception as e:
            return json.dumps({"error": "rpc_failed", "detail": str(e)})

    async def _query_community_agents(
        self,
        goal_status: "GoalStatus",
    ) -> None:
        # Extract from goal_status
        goal_id = goal_status.goal_id
        intents = goal_status.intents
        workspace_id = goal_status.workspace_id
        context = goal_status.context
        if not self.communities:
            logger.debug("No communities to query for goal_id=%s", goal_id)
            return

        # Collect unique agents from communities that match the context
        agents_to_query = set()
        matching_communities = []

        for community in self.communities:
            match_score = community.matches_context(context)
            logger.debug(
                "Community %s match_score=%.3f for goal_id=%s",
                community.community_id,
                match_score,
                goal_id,
            )
            if match_score >= COMMUNITY_MATCH_THRESHOLD:
                matching_communities.append(community.community_id)
                agents_to_query.update(community.member_ids)
        
        my_jid = str(self.jid).split("/")[0]
        agents_to_query.discard(my_jid)
                
        goal_status.update_status(relevant_communities=matching_communities)

        if not agents_to_query:
            logger.info(
                demo("No community agents to query for goal_id=%s (matching_communities=%s)"),
                goal_id,
                matching_communities,
            )
            return

        logger.info(
            demo("Sending COMMUNITY_REQUEST to %d agents for goal_id=%s from communities=%s"),
            len(agents_to_query),
            goal_id,
            matching_communities,
        )

        base_conversation_id = f"community_plan_{goal_id}"

        request_payload = {
            "goal_id": goal_id,
            "intents": intents,
            "workspace_id": workspace_id,
        }
        goal_status.update_status(community_expected_responses=len(agents_to_query))
        # Send requests to all community agents
        for agent_jid in agents_to_query:
            conversation_id = f"{base_conversation_id}_{agent_jid.split('@')[0]}"
            
            msg = Message(
                sender=str(self.jid),
                receiver=agent_jid,
                message_type=MessageType.COMMUNITY_REQUEST,
                content=request_payload,
                conversation_id=conversation_id,
                metadata={
                    META_CONVERSATION_ID: conversation_id,
                    META_CORRELATION_ID: str(uuid.uuid4()),
                }
            )
            await self.send_message(msg)

        logger.info(
            demo("Community requests sent for goal_id=%s to %d agents"),
            goal_id,
            len(agents_to_query),
        )

    async def _generate_plan(self, intents: List[str], workspace_id: Optional[str] = None, goal_id: Optional[str] = None) -> str:
        """
        Gather context programmatically (affordances + state) and ask the LLM once.
        If the LLM call fails or returns non-JSON, wrap a diagnostic JSON.
        """
        intents = [str(i).strip() for i in (intents or []) if str(i).strip()]
        if not intents:
            return json.dumps(
                {"plan_version": "1.2", "error": "missing_intents", "detail": "No intents provided.", "steps": []},
                indent=2,
            )

        goal_status = self.goal_list[goal_id]
        logger.info(demo("Reusing existing goal: goal_id=%s phase=%s"), goal_id, goal_status.phase.value)
        goal_status.intents = intents
        if workspace_id:
            goal_status.workspace_id = workspace_id

        # Fast-path: if there is a suitable signifier match for every intent, reuse it to build
        # a plan directly, without querying EnvExplorer for capabilities/state and without calling the LLM.
        goal_status.update_status(phase=PlanningPhase.GATHERING_REUSED_PLAN)
        reused_plan = await self._try_build_plan_from_signifiers(intents, workspace_id=workspace_id)
        if reused_plan is not None:
            logger.info(
                demo("Plan recovered from signifiers (no EnvExplorer context queries, no LLM): steps=%d goal_id=%s"),
                len(reused_plan.get("steps") or []),
                goal_id,
            )
            goal_status.update_status(
                phase=PlanningPhase.COMPLETED_SUCCESS,
                reused_plan=reused_plan,
                reused_plan_source="signifiers",
                best_plan=reused_plan,
                best_plan_source="reused"
            )
            return json.dumps(reused_plan, indent=2)

        goal_status.update_status(phase=PlanningPhase.GENERATING_LOCAL_PLAN)

        try:
            context = await self._gather_planning_context(intents, workspace_id=workspace_id)
        except Exception as e:
            logger.warning(f"Context gathering failed: {e}")
            error_plan = {
                "plan_version": "1.2",
                "error": "context_gathering_failed",
                "detail": str(e),
                "steps": [],
            }
            goal_status.update_status(
                phase=PlanningPhase.COMPLETED_FAILURE,
                error="context_gathering_failed",
                error_detail=str(e)
            )
            return json.dumps(error_plan, indent=2)

        # Store context in goal status
        goal_status.update_status(context=context)

        # Query community agents for assistance (if any matching communities)
        goal_status.update_status(phase=PlanningPhase.QUERYING_COMMUNITY)
        await self._query_community_agents(goal_status=goal_status)

        # Start timeout for community responses
        timeout_seconds = DEFAULT_COMMUNITY_QUERY_TIMEOUT
        loop = asyncio.get_event_loop()
        
        def timeout_callback():
            """Called when community query timeout expires."""
            logger.info(
                demo("Community query timeout expired for goal_id=%s, triggering continue_generate_plan"),
                goal_id
            )
            asyncio.create_task(self._continue_generate_plan(goal_status))
        
        timer_handle = loop.call_later(timeout_seconds, timeout_callback)
        
        logger.info(
            demo("Community query timeout started: %.1fs for goal_id=%s"),
            timeout_seconds,
            goal_id
        )
        return 

    async def _continue_generate_plan(self, goal_status: "GoalStatus") -> None:
        """
        Continue plan generation after community queries complete or timeout.
        This method is called either when:
        1. Response ratio >= DEFAULT_COMMUNITY_MIN_RESPONSE_RATIO
        2. Timeout expires
        """
        # Prevent double execution
        if goal_status.continue_triggered:
            logger.debug(
                demo("_continue_generate_plan already triggered for goal_id=%s, skipping"),
                goal_status.goal_id
            )
            return
        
        goal_status.continue_triggered = True
        
        goal_status.update_status(phase=PlanningPhase.GENERATING_LOCAL_PLAN)
        intents = goal_status.intents
        workspace_id = goal_status.workspace_id       
        context = goal_status.context
        
        prompt_messages = self._build_planning_prompt(intents, context, workspace_id=workspace_id)

        def _extract_json_text(raw: str) -> str:
            """
            Best-effort extraction of a JSON object/array from LLM output.
            Handles common cases like markdown code fences.
            """
            if not raw:
                return raw
            s = raw.strip()

            # Strip markdown fences: ```json ... ``` or ``` ... ```
            if s.startswith("```"):
                # remove first fence line
                first_nl = s.find("\n")
                if first_nl != -1:
                    s = s[first_nl + 1 :]
                # remove trailing fence
                if s.rstrip().endswith("```"):
                    s = s.rstrip()
                    s = s[: -3]
                s = s.strip()

            # If still has surrounding prose, try to isolate the first JSON blob.
            # Prefer {...} but allow [...] too.
            obj_start = s.find("{")
            arr_start = s.find("[")
            if obj_start == -1 and arr_start == -1:
                return s

            if obj_start == -1 or (arr_start != -1 and arr_start < obj_start):
                start = arr_start
                end = s.rfind("]")
            else:
                start = obj_start
                end = s.rfind("}")

            if start != -1 and end != -1 and end > start:
                return s[start : end + 1].strip()
            return s

        try:
            completion_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": prompt_messages,
            }
            if not str(self.model).startswith("o"):
                completion_kwargs["temperature"] = self.temperature
            if self.max_completion_tokens is not None:
                completion_kwargs["max_completion_tokens"] = self.max_completion_tokens
            elif self.max_tokens is not None:
                completion_kwargs["max_tokens"] = self.max_tokens
            if self.reasoning_effort and str(self.model).startswith("o"):
                completion_kwargs["reasoning_effort"] = self.reasoning_effort

            completion = await self.llm_client.chat.completions.create(**completion_kwargs)
            content = completion.choices[0].message.content or ""
            content = content.strip()
            json_text = _extract_json_text(content)

            # Try to parse JSON; if invalid, wrap as diagnostic
            try:
                parsed = json.loads(json_text)
                # Enforce strict contract: must be a JSON object with plan_version=1.2 and steps list
                if not isinstance(parsed, dict):
                    raise ValueError("Plan response is not a JSON object")

                if str(parsed.get("plan_version")) != "1.2":
                    return json.dumps(
                        {"plan_version": "1.2", "error": "invalid_plan_version", "detail": "Expected plan_version=1.2", "steps": []},
                        indent=2,
                    )

                steps = parsed.get("steps")
                if not isinstance(steps, list):
                    return json.dumps(
                        {"plan_version": "1.2", "error": "invalid_plan_format", "detail": "Missing steps list", "steps": []},
                        indent=2,
                    )

                if len(steps) == 0:
                    # already strict-failure shape is acceptable; ensure error present if empty
                    if "error" not in parsed:
                        parsed["error"] = "infeasible"
                        parsed["detail"] = parsed.get("detail") or "No feasible plan for all intents."
                    return json.dumps(parsed, indent=2)

                # Coverage check: each input intent must appear at least once in step.intent
                step_intents = {str(s.get("intent")) for s in steps if isinstance(s, dict) and s.get("intent") is not None}
                missing = [i for i in intents if i not in step_intents]
                if missing:
                    return json.dumps(
                        {
                            "plan_version": "1.2",
                            "error": "infeasible",
                            "detail": f"Plan did not cover all intents. Missing: {missing}",
                            "steps": [],
                        },
                        indent=2,
                    )
                local_plan = json.dumps(parsed, indent=2)
                goal_status.update_status(
                    phase=PlanningPhase.COMPLETED_SUCCESS,
                    local_plan=parsed,
                    local_plan_complete=True,
                    best_plan=parsed,
                    best_plan_source="local"
                )
                return local_plan
            except Exception:
                return json.dumps(
                    {
                        "plan_version": "1.2",
                        "error": "non_json_response",
                        "raw_response": content,
                        "steps": [],
                    },
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"LLM plan generation failed, returning error JSON: {e}")
            fallback = {
                "plan_version": "1.2",
                "error": "plan_generation_failed",
                "detail": str(e),
                "steps": [],
            }
            goal_status.update_status(
                phase=PlanningPhase.COMPLETED_FAILURE,
                error="plan_generation_failed",
                error_detail=str(e)
            )
            return json.dumps(fallback, indent=2)

    async def _try_build_plan_from_signifiers(
        self, intents: List[str], workspace_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Try to build a JSON-Plan 1.2 directly from signifier matches.

        Notes:
        - We still query EnvExplorer for SIGNIFIER_MATCH_REQUEST (it hosts the embedded RD4 engine).
        - We MUST NOT query EnvExplorer for ENV_CAPABILITIES_REQUEST / ENV_STATE_REQUEST in this path.
        - If any intent has no usable signifier match, return None.
        """
        try:
            signifier_matches = await self._gather_signifier_matches(intents, workspace_id=workspace_id)
        except Exception:
            return None

        if not isinstance(signifier_matches, dict) or not signifier_matches:
            return None

        def _pick_match(intent: str) -> Optional[Dict[str, Any]]:
            payload = signifier_matches.get(intent)
            if not isinstance(payload, dict):
                return None
            finals = payload.get("final_matches") or []
            matches = payload.get("matches") or []
            if not isinstance(finals, list) or not finals:
                return None
            if not isinstance(matches, list) or not matches:
                return None

            chosen_id = str(finals[0])
            for m in matches:
                if not isinstance(m, dict):
                    continue
                if str(m.get("signifier_id") or "") == chosen_id:
                    return m
            return None

        steps: List[Dict[str, Any]] = []
        for idx, intent in enumerate(intents, start=1):
            match = _pick_match(intent)
            if not match:
                return None

            affordance_uri = str(match.get("affordance_uri") or "").strip()
            if not affordance_uri:
                return None

            payload_hint = match.get("payload_hint")
            payload = payload_hint if isinstance(payload_hint, dict) else {}

            signifier_id = str(match.get("signifier_id") or "").strip()
            similarity = match.get("intent_similarity")

            cached_aff = self._cached_affordance_by_id.get(affordance_uri)
            action_name = ""
            method = None
            target = None
            content_type = "application/json"
            artifact_uri = ""

            if isinstance(cached_aff, dict):
                action_name = str(cached_aff.get("action_name") or "")
                method = cached_aff.get("method")
                target = cached_aff.get("target")
                content_type = str(cached_aff.get("content_type") or "application/json")
                artifact_uri = str(cached_aff.get("artifact_id") or "")

            if not action_name:
                action_name = affordance_uri.rstrip("/").rsplit("/", 1)[-1]

            if not artifact_uri:
                # Best-effort: derive artifact URI from the affordance URI.
                try:
                    prefix, rest = affordance_uri.split("/artifacts/", 1)
                    artifact_name = rest.split("/", 1)[0]
                    artifact_uri = f"{prefix}/artifacts/{artifact_name}#artifact"
                except Exception:
                    artifact_uri = ""

            # If we don't have a cached affordance form, assume the affordance URI is callable.
            if target is None:
                target = affordance_uri
            if method is None:
                method = "POST"

            step_meta: Dict[str, Any] = {"reused_from_signifier": True}
            if signifier_id:
                step_meta["used_signifier_id"] = signifier_id
            if similarity is not None:
                step_meta["used_signifier_similarity"] = similarity

            steps.append(
                {
                    "step_id": idx,
                    "intent": intent,
                    "artifact_uri": artifact_uri,
                    "affordance_uri": affordance_uri,
                    "action_name": action_name,
                    "method": method,
                    "target": target,
                    "content_type": content_type,
                    "payload": payload,
                    "metadata": step_meta,
                    "reasons": [
                        {
                            "property": "reused_signifier",
                            "direction": "set",
                            "evidence": [],
                            "why": "Recovered this step from a previously stored Signifier (intent/context match).",
                        }
                    ],
                }
            )

        return {"plan_version": "1.2", "steps": steps}

    async def _gather_planning_context(self, intents: List[str], workspace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Pull environment context once: affordances and state (from EnvExplorer).
        Returns a dict with capped affordances and optional state snapshot.
        """
        logger.info(demo("Gathering context from EnvExplorer (workspace_id=%r)"), workspace_id)
        raw_aff, raw_state = await asyncio.gather(
            self._query_env_explorer(
                message_type=MessageType.ENV_CAPABILITIES_REQUEST.value,
                body={"query": "all"},
                expect_type=MessageType.ENV_CAPABILITIES_RESPONSE.value,
            ),
            self._query_env_explorer(
                message_type=MessageType.ENV_STATE_REQUEST.value,
                body={},
                expect_type=MessageType.ENV_STATE_RESPONSE.value,
            ),
        )

        def _parse_or_empty(raw: str):
            try:
                return json.loads(raw) if raw else {}
            except Exception:
                return {}

        aff_payload = _parse_or_empty(raw_aff)
        state_payload = _parse_or_empty(raw_state)

        affordances = aff_payload.get("affordances") or aff_payload.get("capabilities") or []
        try:
            raw_artifacts = state_payload.get("artifacts") if isinstance(state_payload, dict) else None
            artifacts_count = len(raw_artifacts) if isinstance(raw_artifacts, dict) else 0
        except Exception:
            artifacts_count = 0

        logger.info(
            demo("Context snapshot: affordances=%s artifacts_in_state=%s"),
            len(affordances) if isinstance(affordances, list) else "?",
            artifacts_count,
        )

        # Optional workspace scoping (helps multi-workspace disambiguation)
        if workspace_id:
            ws = str(workspace_id).strip()
            # Accept either exact workspace URI match OR a human-friendly workspace name like "lab308".
            def _ws_match(value: Any) -> bool:
                if not value:
                    return False
                s = str(value)
                if not ws:
                    return False

                # If caller provided a full URI, prefer direct matching / containment.
                if ws.startswith("http://") or ws.startswith("https://"):
                    return s == ws or ws in s

                # Otherwise treat ws as a short name (e.g., "lab308") and match common URI shapes:
                # - .../workspaces/lab308#workspace
                # - .../workspaces/lab308
                # - .../workspaces/lab308/...
                if s == ws:
                    return True
                if f"/{ws}#" in s:
                    return True
                if f"/{ws}/" in s:
                    return True
                if s.endswith("/" + ws):
                    return True
                # Last resort: substring match (kept permissive for user convenience)
                return ws in s

            if isinstance(affordances, list):
                affordances = [a for a in affordances if isinstance(a, dict) and _ws_match(a.get("workspace_id"))]

            # State payload can be a snapshot with artifacts keyed by id (our EnvExplorer response)
            if isinstance(state_payload, dict) and isinstance(state_payload.get("artifacts"), dict):
                filtered_artifacts: Dict[str, Any] = {}
                for aid, ainfo in state_payload.get("artifacts", {}).items():
                    if isinstance(ainfo, dict) and _ws_match(ainfo.get("workspace_id")):
                        filtered_artifacts[aid] = ainfo
                state_payload = {**state_payload, "artifacts": filtered_artifacts}

            logger.info(
                demo("Workspace scoping applied: affordances=%s artifacts_in_state=%s workspace_id=%r"),
                len(affordances) if isinstance(affordances, list) else "?",
                len(state_payload.get("artifacts") or {}) if isinstance(state_payload, dict) else "?",
                workspace_id,
            )

        # Optional signifier suggestions (embedded RD4 memory hosted by EnvExplorer).
        signifier_matches: Dict[str, Any] = {}
        try:
            signifier_matches = await self._gather_signifier_matches(intents, workspace_id=workspace_id)
        except Exception:
            signifier_matches = {}

        # Cache the most recent context snapshot for future signifier-only planning.
        # (Best-effort; does not affect normal planning if it fails.)
        try:
            self._cached_affordances = affordances if isinstance(affordances, list) else []
            self._cached_state_payload = state_payload if isinstance(state_payload, dict) else {}
            self._cached_affordance_by_id = {}
            if isinstance(self._cached_affordances, list):
                for a in self._cached_affordances:
                    if not isinstance(a, dict):
                        continue
                    aid = str(a.get("affordance_id") or "").strip()
                    if aid:
                        self._cached_affordance_by_id[aid] = a
        except Exception:
            pass

        return {
            "intents": intents,
            "affordances": affordances,
            "state": state_payload.get("state") or state_payload,
            "signifier_matches": signifier_matches,
        }

    async def _gather_signifier_matches(self, intents: List[str], workspace_id: Optional[str] = None) -> Dict[str, Any]:
        """Ask EnvExplorer for signifier matches per intent (best-effort)."""
        explorer_jid = self.target_jids.get("explorer")
        if not explorer_jid:
            return {}

        intents = [str(i).strip() for i in (intents or []) if str(i).strip()]
        if not intents:
            return {}

        async def _one(intent: str) -> Dict[str, Any]:
            raw = await self._query_env_explorer(
                message_type=MessageType.SIGNIFIER_MATCH_REQUEST.value,
                body={
                    "intent": intent,
                    **({"workspace_id": str(workspace_id)} if workspace_id else {}),
                    "k": 5,
                },
                expect_type=MessageType.SIGNIFIER_MATCH_RESPONSE.value,
            )
            try:
                data = json.loads(raw) if raw else {}
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

        results = await asyncio.gather(*[_one(i) for i in intents], return_exceptions=True)

        out: Dict[str, Any] = {}
        for intent, res in zip(intents, results):
            if isinstance(res, Exception):
                out[intent] = {"error": "signifier_query_failed", "detail": str(res)}
            else:
                out[intent] = res

        # Demo-friendly summary logs (kept compact).
        for intent, payload in out.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("error"):
                logger.info(demo("Signifier search failed: intent=%r error=%s"), intent, payload.get("error"))
                continue

            matches = payload.get("matches") if isinstance(payload.get("matches"), list) else []
            finals = payload.get("final_matches") if isinstance(payload.get("final_matches"), list) else []
            total = payload.get("total_signifiers")

            if finals:
                logger.info(
                    demo("Signifier search: intent=%r matches=%d final=%d top=%s"),
                    intent,
                    len(matches),
                    len(finals),
                    finals[0],
                )
            else:
                logger.info(
                    demo("Signifier search: intent=%r matches=%d final=0 stored_total=%s"),
                    intent,
                    len(matches),
                    total if total is not None else "?",
                )
        return out

    def _build_planning_prompt(self, intents: List[str], context: Dict[str, Any], workspace_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Build a strict prompt: system with rules + user with intent and context payload.
        """
        system = (
            "Return ONLY valid JSON. No markdown. No prose.\n"
            "\n"
            "SELF-VALIDATION REQUIRED (do this BEFORE you output):\n"
            "1) Grounding:\n"
            "   - Choose an affordance from the provided context.affordances list.\n"
            "   - step.affordance_uri MUST equal affordance.affordance_id from that list.\n"
            "   - step.artifact_uri MUST match that affordance.artifact_id.\n"
            "   - step.method/step.target/step.content_type MUST match that affordance's method/target/content_type.\n"
            "   - Do NOT invent URIs or actions.\n"
            "2) Payload/schema:\n"
            "   - If the selected affordance has input_schema, step.payload MUST satisfy it (required fields, types, enums).\n"
            "3) Intent coverage:\n"
            "   - Each step.intent MUST exactly match one of the provided intents.\n"
            "   - All provided intents MUST be covered by at least one step, unless already satisfied (see below).\n"
            "4) Redundancy / already satisfied:\n"
            "   - Use the provided state snapshot to avoid redundant steps.\n"
            "   - If the current state already satisfies ALL intents, return a strict no-op JSON:\n"
            "     {\"plan_version\":\"1.2\",\"error\":\"already_satisfied\",\"detail\":\"...\",\"steps\":[]}\n"
            "4b) Signifier-first planning (STRICT):\n"
            "   - context.signifier_matches is a dict keyed by intent string.\n"
            "   - Each entry has:\n"
            "     - matches: list of {signifier_id, affordance_uri, intent_similarity, shacl_conforms, shacl_violations, payload_hint?}\n"
            "     - final_matches: list of signifier_id (best-first) that passed SHACL.\n"
            "   - For each intent:\n"
            "     1) If final_matches is non-empty, pick signifier_id = final_matches[0].\n"
            "     2) Find the corresponding entry in matches to get affordance_uri and payload_hint.\n"
            "     3) Use that affordance_uri ONLY if it exists in context.affordances (affordance_id match) and method/target/content_type match.\n"
            "     4) If payload_hint is present, use it only if it satisfies the selected affordance input_schema; otherwise construct a valid payload.\n"
            "     5) If the signifier suggestion is unusable for any reason, ignore it and plan normally.\n"
            "   - Traceability:\n"
            "     - If you use a signifier for a step, include step.metadata.used_signifier_id and step.metadata.used_signifier_similarity.\n"
            "     - If you ignore an available final_matches suggestion, include step.metadata.signifier_ignored_reason.\n"
            "5) Evidence:\n"
            "   - Every step must include at least one reasons entry with at least one evidence item grounded in the state.\n"
            "   - evidence[].artifact MUST be a key from context.state.artifacts.\n"
            "   - evidence[].property MUST be a full property URI key from that artifact's state dict.\n"
            "   - evidence[].reading MUST equal the reading from the state snapshot.\n"
            "\n"
            "If you cannot satisfy ALL intents using ONLY the provided affordances, return:\n"
            "{\"plan_version\":\"1.2\",\"error\":\"infeasible\",\"detail\":\"...\",\"steps\":[]}\n"
        )

        schema_reminder = (
            "JSON-Plan 1.2 schema:\n"
            "{\n"
            '  "plan_version": "1.2",\n'
            '  "steps": [\n'
            "    {\n"
            '      "step_id": <int>,\n'
            '      "intent": "<intent>",\n'
            '      "artifact_uri": "<uri>",\n'
            '      "affordance_uri": "<uri>",\n'
            '      "action_name": "<name>",\n'
            '      "method": "<HTTP verb>",\n'
            '      "target": "<hctl:hasTarget URI>",\n'
            '      "content_type": "application/json",\n'
            '      "payload": { },\n'
            '      "metadata": { },\n'
            '      "reasons": [\n'
            "        {\n"
            '          "property": "<property satisfied>",\n'
            '          "direction": "<increase|decrease|set>",\n'
            '          "evidence": [\n'
            "            {\n"
            '              "artifact": "<sensor-or-artifact URI>",\n'
            '              "property": "<property URI>",\n'
            '              "operator": "<lessThan|lessEqual|greaterThan|greaterEqual|equals>",\n'
            '              "threshold": "<number|string>",\n'
            '              "reading": "<number|string>"\n'
            "            }\n"
            "          ],\n"
            '          "why": "One or two sentences."\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

        user_payload = {
            "intents": intents,
            **({"workspace_id": workspace_id} if workspace_id else {}),
            "affordances": context.get("affordances", []),
            "state": context.get("state", {}),
            "signifier_matches": context.get("signifier_matches", {}),
            "instructions": [
                "Use only provided affordance URIs and methods; do not invent or alter them.",
                "If workspace_id is provided, only use affordances/artifacts from that workspace.",
                "Use signifiers strictly: if signifier_matches[intent].final_matches is non-empty, prefer final_matches[0] for that intent when it is usable.",
                "If you use a signifier, include step.metadata.used_signifier_id and step.metadata.used_signifier_similarity.",
                "Cover ALL intents. Each step.intent must match exactly one provided intent.",
                "Include at least one reason with evidence per step.",
                "Evidence items must use full artifact URIs and full property URIs from the provided state snapshot.",
                "If missing info, use placeholders but keep JSON valid.",
                "Respond with JSON only, no markdown, no prose.",
            ],
        }

        return [
            {"role": "system", "content": INTERACTION_SOLVER_SYSTEM_PROMPT.strip()},
            {"role": "system", "content": system},
            {"role": "system", "content": schema_reminder},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ]


class PlanningStatusBehaviour(CyclicBehaviour):
    """Behavior for handling planning status requests."""

    async def run(self):
        """
        Handle planning status requests.

        Responds with current planning phase and best plan obtained so far.
        Phases:
        - initiated: Goal just created
        - gathering_reused_plan: Searching for reusable signifier-based plan
        - generating_local_plan: Generating plan with LLM
        - querying_community: Querying community agents (future)
        - completed_success: Plan successfully obtained
        - completed_failure: Planning failed
        """
        msg = await self.receive(timeout=1)
        if not msg:
            return

        if msg.get_metadata("type") != MessageType.PLANNING_STATUS_REQUEST.value:
            return

        goal_id: Optional[str] = None
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                goal_id = payload.get("goal_id")
        except json.JSONDecodeError:
            pass

        logger.info(
            demo("Received PLANNING_STATUS_REQUEST: goal_id=%s from=%s"),
            goal_id or "missing",
            str(msg.sender),
        )

        reply = msg.make_reply()
        reply.set_metadata("type", MessageType.PLANNING_STATUS_RESPONSE.value)
        corr = msg.get_metadata(META_CORRELATION_ID)
        if corr:
            reply.set_metadata(META_CORRELATION_ID, corr)
        if msg.thread:
            reply.thread = msg.thread

        if not goal_id:
            reply.body = json.dumps({
                "error": "missing_goal_id",
                "detail": "No goal_id provided in request"
            })
            await self.send(reply)
            return

        goal_status = self.agent.goal_list.get(str(goal_id))
        if not goal_status:
            reply.body = json.dumps({
                "error": "goal_not_found",
                "detail": f"No goal found with id: {goal_id}",
                "goal_id": str(goal_id)
            })
            await self.send(reply)
            return

        response = {
            "goal_id": goal_status.goal_id,
            "phase": goal_status.phase.value,
            "intents": goal_status.intents,
            "workspace_id": goal_status.workspace_id,
            "created_at": goal_status.created_at,
            "last_updated": goal_status.last_updated,
            "elapsed_time": goal_status.last_updated - goal_status.created_at,
        }

        if goal_status.reused_plan:
            response["has_reused_plan"] = True
            response["reused_plan_source"] = goal_status.reused_plan_source

        if goal_status.local_plan:
            response["has_local_plan"] = True
            response["local_plan_complete"] = goal_status.local_plan_complete

        if goal_status.relevant_communities:
            response["relevant_communities"] = goal_status.relevant_communities

        if goal_status.community_responses:
            response["community_responses_count"] = len(goal_status.community_responses)

        if goal_status.error:
            response["error"] = goal_status.error
            response["error_detail"] = goal_status.error_detail

        logger.info(
            demo("PLANNING_STATUS_RESPONSE: goal_id=%s phase=%s best_plan_source=%s"),
            goal_id,
            goal_status.phase.value,
            goal_status.best_plan_source or "none",
        )

        reply.body = json.dumps(response, indent=2)
        await self.send(reply)


class CommunityRequestBehaviour(CyclicBehaviour):
    """Responds to community_request messages by gathering local planning context and sending it back."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return

        intents: List[str] = []
        workspace_id: Optional[str] = None
        goal_id: Optional[str] = None
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                raw_intents = payload.get("intents")
                if isinstance(raw_intents, list):
                    intents = [str(i).strip() for i in raw_intents if str(i).strip()]
                ws = payload.get("workspace_id")
                if ws:
                    workspace_id = str(ws)
                gid = payload.get("goal_id")
                if gid:
                    goal_id = str(gid)
        except json.JSONDecodeError:
            pass

        logger.info(
            demo("Received COMMUNITY_REQUEST: goal_id=%s intents=%s workspace_id=%r from=%s"),
            goal_id,
            intents,
            workspace_id,
            str(msg.sender),
        )

        reply = msg.make_reply()
        reply.set_metadata("type", MessageType.COMMUNITY_RESPONSE.value)
        corr = msg.get_metadata(META_CORRELATION_ID)
        if corr:
            reply.set_metadata(META_CORRELATION_ID, corr)
        if msg.thread:
            reply.thread = msg.thread

        if not intents:
            reply.body = json.dumps({
                "error": "missing_intents",
                "detail": "No intents provided in community_request",
                "goal_id": goal_id,
            })
            await self.send(reply)
            return

        # Gather local planning context (affordances, state, signifier matches)
        try:
            context = await self.agent._gather_planning_context(intents, workspace_id=workspace_id)
        except Exception as e:
            logger.warning("Failed to gather planning context for community_request: %s", e)
            reply.body = json.dumps({
                "error": "context_gathering_failed",
                "detail": str(e),
                "goal_id": goal_id,
            })
            await self.send(reply)
            return

        response_data = {
            "responding_agent": str(self.agent.jid),
            "context": context,
            "goal_id": goal_id,
        }

        logger.info(
            demo("COMMUNITY_RESPONSE sent: goal_id=%s intents=%s to=%s context=%d"),
            goal_id or "N/A",
            intents,
            str(msg.sender),
            context,
        )

        reply.body = json.dumps(response_data, indent=2)
        await self.send(reply)


class CommunityResponseBehaviour(CyclicBehaviour):
    """Handles community_response messages by appending the response to the goal's community_responses list."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return

        goal_id: Optional[str] = None
        response_data: Dict[str, Any] = {}
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                goal_id = payload.get("goal_id")
                response_data = payload
        except json.JSONDecodeError:
            pass

        sender_jid = str(msg.sender)

        logger.info(
            demo("Received COMMUNITY_RESPONSE: goal_id=%s from=%s"),
            goal_id or "missing",
            sender_jid,
        )

        if not goal_id:
            logger.warning("COMMUNITY_RESPONSE missing goal_id from=%s", sender_jid)
            return

        goal_status = self.agent.goal_list.get(str(goal_id))
        if not goal_status:
            logger.warning(
                "COMMUNITY_RESPONSE for unknown goal_id=%s from=%s",
                goal_id,
                sender_jid,
            )
            return

        entry = {
            "agent_jid": sender_jid,
            "response": response_data,
            "received_at": asyncio.get_event_loop().time(),
        }
        goal_status.community_responses[sender_jid] = entry

        logger.info(
            demo("COMMUNITY_RESPONSE appended: goal_id=%s from=%s total_responses=%d"),
            goal_id,
            sender_jid,
            len(goal_status.community_responses),
        )

        # Check if we have received enough responses based on the minimum response ratio
        if goal_status.community_expected_responses > 0:
            received_count = len(goal_status.community_responses)
            expected_count = goal_status.community_expected_responses
            response_ratio = received_count / expected_count

            if response_ratio >= DEFAULT_COMMUNITY_MIN_RESPONSE_RATIO:
                self.agent._continue_generate_plan(goal_status)


class EnvironmentReadyBehaviour(CyclicBehaviour):
    """Behavior for waiting for environment discovery to complete."""

    async def run(self):
        """
        Wait for environment discovery notification.

        TODO: Implementation steps:
        1. Wait for ENV_DISCOVERY_COMPLETE message from EnvExplorer
        2. Store EnvExplorer JID
        3. Set environment_ready flag
        4. Log that environment is ready
        5. Stop this behavior (one-time only)
        """
        msg = await self.receive(timeout=1)
        if not msg:
            return

        if msg.get_metadata("type") != MessageType.ENV_DISCOVERY_COMPLETE.value:
            return

        payload: Dict[str, Any] = {}
        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {"raw": msg.body or ""}

        self.agent.mark_environment_ready(sender_jid=str(msg.sender), payload=payload)
        logger.info(f"Environment ready (notified by {msg.sender}).")
        # SPADE Behaviour.kill() is not awaitable.
        self.kill()


class GoalRequestBehaviour(CyclicBehaviour):
    """Behavior for handling goal requests and creating plans."""

    async def run(self):
        """
        Handle goal requests from UserAssistant.

        TODO: Implementation steps:
        1. Wait for GOAL_REQUEST message
        2. Extract GoalRequest
        3. Ensure environment is ready
        4. Gather planning context
        5. Generate plan
        6. Send plan back to UserAssistant
        """
        msg = await self.receive(timeout=1)
        if not msg:
            return

        if msg.get_metadata("type") != MessageType.GOAL_REQUEST.value:
            return

        intents: List[str] = []
        workspace_id: Optional[str] = None
        goal_id: Optional[str] = None
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                ws = payload.get("workspace_id") or payload.get("workspace")
                if ws:
                    workspace_id = str(ws)

                gid = payload.get("goal_id")
                if gid:
                    goal_id = str(gid)

            raw_intents = payload.get("intents")
            if isinstance(raw_intents, list):
                intents = [str(i).strip() for i in raw_intents if str(i).strip()]
            else:
                goal_text = payload.get("intent") or payload.get("goal")
                if goal_text:
                    intents = [str(goal_text).strip()]
        except json.JSONDecodeError:
            if msg.body:
                intents = [str(msg.body).strip()]

        intents = [i for i in intents if i]
        if not intents:
            reply = msg.make_reply()
            reply.set_metadata("type", MessageType.GOAL_RESPONSE.value)
            reply.body = json.dumps({"error": "missing_intent"})
            # Propagate correlation_id/thread for request/response pairing.
            corr = msg.get_metadata(META_CORRELATION_ID)
            if corr:
                reply.set_metadata(META_CORRELATION_ID, corr)
            if msg.thread:
                reply.thread = msg.thread
            await self.send(reply)
            return

        # Generate goal_id if not provided
        if not goal_id:
            goal_id = str(uuid.uuid4())

        logger.info(
            demo("Received GOAL_REQUEST: intents=%s workspace_id=%r goal_id=%s from=%s"),
            intents,
            workspace_id,
            goal_id,
            str(msg.sender),
        )

        # Ensure environment is ready before planning.
        env_timeout = float(self.agent.config.get("planning", {}).get("context_gathering", {}).get("timeout", DEFAULT_TIMEOUT))
        ready = await self.agent.await_environment_ready(timeout=env_timeout)
        if not ready:
            reply = msg.make_reply()
            reply.set_metadata("type", MessageType.PLAN_CREATED.value)
            reply.body = json.dumps(
                {
                    "plan_version": "1.2",
                    "error": "env_not_ready",
                    "detail": "Environment discovery not completed (timeout).",
                    "steps": [],
                },
                indent=2,
            )
            corr = msg.get_metadata(META_CORRELATION_ID)
            if corr:
                reply.set_metadata(META_CORRELATION_ID, corr)
            if msg.thread:
                reply.thread = msg.thread
            await self.send(reply)
            return

        plan_json = await self.agent._generate_plan(intents, workspace_id=workspace_id, goal_id=goal_id)

        logger.info(
            demo("GOAL_STATUS: intents=%s workspace_id=%r goal_id=%s from=%s"),
            intents,
            workspace_id,
            goal_id,
            str(msg.sender)
        )
        logger.info(demo(f"Goal status dict: {self.agent.goal_list[goal_id].to_dict()}"))
        logger.info(demo(f'PLAN: {plan_json}'))

        try:
            parsed_plan = json.loads(plan_json or "{}")
            steps = parsed_plan.get("steps") if isinstance(parsed_plan, dict) else None
            step_count = len(steps) if isinstance(steps, list) else 0
            used_signifiers = 0
            if isinstance(steps, list):
                for s in steps:
                    meta = s.get("metadata") if isinstance(s, dict) else None
                    if isinstance(meta, dict) and meta.get("used_signifier_id"):
                        used_signifiers += 1
            logger.info(demo("Plan created: steps=%d used_signifiers=%d"), step_count, used_signifiers)
        except Exception:
            logger.info(demo("Plan created (failed to parse JSON for summary)."))

        reply = msg.make_reply()
        reply.set_metadata("type", MessageType.PLAN_CREATED.value)
        reply.body = plan_json
        # Propagate correlation_id/thread for request/response pairing.
        corr = msg.get_metadata(META_CORRELATION_ID)
        if corr:
            reply.set_metadata(META_CORRELATION_ID, corr)
        if msg.thread:
            reply.thread = msg.thread
        await self.send(reply)

    async def handle_goal_request(self, goal_request: GoalRequest) -> BehaviorTreePlan:
        """
        Handle a goal request and create a plan.

        TODO: Implementation steps:
        1. Validate goal request
        2. Gather planning context
        3. Generate behavior tree plan
        4. Validate plan structure
        5. Return plan
        """
        pass

    async def gather_planning_context(self, goal_request: GoalRequest) -> Dict[str, Any]:
        """
        Gather context needed for planning.

        TODO: Implementation steps:
        1. Create AffordanceMatchRequest
        2. Send to EnvExplorer
        3. Wait for AffordanceMatchResponse
        4. Extract affordances and signifiers
        5. Compile context dictionary with:
           - Available affordances
           - Signifiers (usage experiences)
           - Goal intent
           - Conversation context
        6. Return planning context
        """
        pass

    async def gather_context_from_environment(self, goal_request: GoalRequest) -> List[Dict[str, Any]]:
        """
        Gather context from own environment (via EnvExplorer).

        TODO: Implementation steps:
        1. Create AffordanceMatchRequest with goal
        2. Send to EnvExplorer
        3. Wait for response with timeout
        4. Extract matched affordances
        5. Return affordances
        """
        pass

    async def gather_context_from_community(self, goal_request: GoalRequest) -> List[Dict[str, Any]]:
        """
        Gather context from community (future feature).

        TODO: Implementation steps:
        1. Query community agents for similar goals
        2. Retrieve plans from community
        3. Extract relevant patterns
        4. Return community context
        """
        pass

    async def generate_behavior_tree_plan(self, goal: GoalRequest,
                                         context: Dict[str, Any]) -> BehaviorTreePlan:
        """
        Generate a behavior tree plan using LLM.

        TODO: Implementation steps:
        1. Prepare prompt with:
           - Goal intent
           - Available affordances
           - Signifiers (if any)
           - Context
        2. Call plan_generator.generate_plan()
        3. Parse LLM response into behavior tree structure
        4. Create NodeTemplates with:
           - Status tracking (running, done, failed)
           - Expected duration
           - Retry policy
           - Error handling
        5. Validate plan structure
        6. Create BehaviorTreePlan object
        7. Return plan
        """
        pass

    async def validate_plan(self, plan: BehaviorTreePlan) -> bool:
        """
        Validate a generated plan.

        TODO: Implementation steps:
        1. Check that root node exists
        2. Validate tree structure (no cycles, etc.)
        3. Ensure all leaf nodes have affordance_uri
        4. Check that affordances exist in environment
        5. Return validation result
        """
        pass


class PlanExecutionMonitor:
    """Monitors execution of behavior tree plans."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PlanExecutionMonitor.

        Args:
            config: Execution monitoring configuration.
        """
        self.config = config
        self.monitored_plans = {}

    async def start_monitoring(self, plan: BehaviorTreePlan) -> None:
        """
        Start monitoring a plan execution.

        TODO: Implementation steps:
        1. Add plan to monitored_plans
        2. Set up periodic status checks
        3. Register callbacks for node status updates
        """
        pass

    async def stop_monitoring(self, plan_id: str) -> None:
        """
        Stop monitoring a plan.

        TODO: Implementation steps:
        1. Remove plan from monitored_plans
        2. Stop status checks
        3. Unregister callbacks
        """
        pass

    async def update_node_status(self, plan_id: str, node_id: str,
                                status: str, metadata: Dict[str, Any]) -> None:
        """
        Update status of a node in the plan.

        TODO: Implementation steps:
        1. Find plan and node
        2. Update node status
        3. Update metadata (duration, error, etc.)
        4. If plan completed or failed, notify UserAssistant
        """
        pass

    async def execute_node(self, plan_id: str, node: NodeTemplate) -> bool:
        """
        Execute a single node in the behavior tree.

        TODO: Implementation steps:
        1. Based on node_type:
           - Action: Execute affordance
           - Condition: Evaluate condition
           - Sequence: Execute children in order
           - Selector: Execute children until success
           - Parallel: Execute children concurrently
        2. Update node status
        3. Handle errors and retries
        4. Return execution result
        """
        pass

    async def execute_action_node(self, node: NodeTemplate) -> bool:
        """
        Execute an action node (leaf node with affordance).

        TODO: Implementation steps:
        1. Get affordance URI from node
        2. Prepare action parameters
        3. Invoke action via HMAS client
        4. Wait for result
        5. Update node status based on result
        6. Record actual duration
        7. Return success/failure
        """
        pass

    async def execute_condition_node(self, node: NodeTemplate) -> bool:
        """
        Evaluate a condition node.

        TODO: Implementation steps:
        1. Extract condition from node
        2. Evaluate condition (e.g., check sensor state)
        3. Return True/False
        """
        pass

    async def execute_composite_node(self, node: NodeTemplate) -> bool:
        """
        Execute a composite node (sequence, selector, parallel).

        TODO: Implementation steps:
        1. Based on node_type:
           - Sequence: Execute children, stop on first failure
           - Selector: Execute children, stop on first success
           - Parallel: Execute all children concurrently
        2. Update node status based on children results
        3. Return overall result
        """
        pass
