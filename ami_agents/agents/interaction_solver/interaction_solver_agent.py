"""
InteractionSolver Agent - Goal planning and execution.

Classical SPADE agent that calls an LLM to create plans.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
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
from ...shared.utils.spade_rpc import rpc_call, RpcTimeoutError, send_via_router
from ...shared.utils.demo_log import demo
from ...shared.models.plan import BehaviorTreePlan, NodeTemplate, Plan
from ...shared.protocols.llm_protocol import IPlanGenerator
from ...bt_planning.planning.bt_planner import AsyncBTPlanner
from ...bt_planning.signifier_bridge import build_bt_from_signifiers
from ...shared.community.community_client import CommunitySignifierClient

logger = logging.getLogger("InteractionSolver")

# Planning prompt is now in ami_agents.bt_planning.planning.prompts (BT JSON IR format).

DEFAULT_TIMEOUT = 10


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

        # BT planner: generates JSON IR behavior trees via LLM tool calls
        self.bt_planner = AsyncBTPlanner(max_attempts=3)

        # Community signifier client (cross-environment sharing)
        community_cfg = (self.config.get("planning", {}) or {}).get("community", {}) or {}
        community_url = community_cfg.get("api_url") or os.getenv("COMMUNITY_API_URL")
        self.community_enabled = community_cfg.get("enabled", bool(community_url))
        self.community_client: Optional[CommunitySignifierClient] = None
        if self.community_enabled and community_url:
            community_timeout = float(community_cfg.get("timeout", 5.0))
            self.community_client = CommunitySignifierClient(api_url=community_url, timeout=community_timeout)

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

    async def _generate_plan(self, intents: List[str], workspace_id: Optional[str] = None, intent_type: Optional[str] = None) -> str:
        """
        Gather context programmatically (affordances + state) and generate a BT JSON IR plan.

        Returns a JSON string with the plan in behavior tree format:
        {
            "plan_type": "behavior_tree",
            "tree": { ... },            # JSON IR behavior tree spec (or None)
            "explanation": "...",        # LLM's explanation
            "intents": ["..."],          # Input intents
            "intent_type": "EXPLICIT",   # EXPLICIT or IMPLICIT
            "impossible": false,         # True if the goal is infeasible
            "signifier_reuse": false,    # True if built from signifiers (no LLM)
        }
        """
        intents = [str(i).strip() for i in (intents or []) if str(i).strip()]
        if not intents:
            return json.dumps(
                {"plan_type": "behavior_tree", "error": "missing_intents", "detail": "No intents provided.",
                 "tree": None, "intents": [], "intent_type": intent_type},
                indent=2,
            )

        # Fast-path: if there is a suitable signifier match for every intent, reuse it to build
        # a BT directly, without querying EnvExplorer for capabilities/state and without calling the LLM.
        reused_plan = await self._try_build_plan_from_signifiers(intents, workspace_id=workspace_id, intent_type=intent_type)
        if reused_plan is not None:
            logger.info(demo("BT recovered from signifiers (no EnvExplorer context queries, no LLM)"))
            return json.dumps(reused_plan, indent=2)

        try:
            context = await self._gather_planning_context(intents, workspace_id=workspace_id, intent_type=intent_type)
        except Exception as e:
            logger.warning(f"Context gathering failed: {e}")
            return json.dumps(
                {
                    "plan_type": "behavior_tree",
                    "error": "context_gathering_failed",
                    "detail": str(e),
                    "tree": None,
                    "intents": intents,
                    "intent_type": intent_type,
                },
                indent=2,
            )

        # Generate BT using AsyncBTPlanner (LLM tool call with validation retries)
        try:
            result = await self.bt_planner.generate_bt(
                intents=intents,
                affordances=context.get("affordances", []),
                state=context.get("state"),
                signifier_hints=context.get("signifier_matches"),
                client=self.llm_client,
                model=self.model,
                temperature=self.temperature if not str(self.model).startswith("o") else None,
                reasoning_effort=self.reasoning_effort,
                max_completion_tokens=self.max_completion_tokens,
            )
        except Exception as e:
            logger.warning(f"BT generation failed: {e}")
            return json.dumps(
                {
                    "plan_type": "behavior_tree",
                    "error": "plan_generation_failed",
                    "detail": str(e),
                    "tree": None,
                    "intents": intents,
                    "intent_type": intent_type,
                },
                indent=2,
            )

        # Wrap result in standard format
        output: Dict[str, Any] = {
            "plan_type": "behavior_tree",
            "tree": result.get("tree") or None,
            "explanation": result.get("explanation", ""),
            "intents": intents,
            "workspace_id": workspace_id,  # Include workspace_id for signifier context
            "intent_type": intent_type,  # Include intent_type for signifier recording
            # NOTE: Don't include full state_snapshot/affordances in execution_context - plan JSON
            # becomes too large (causes "Unterminated string" errors in LLM tool calls).
            # UserAssistant will query EnvExplorer for fresh state at execution time instead.
        }

        if result.get("impossible"):
            output["impossible"] = True

        return json.dumps(output, indent=2)

    async def _try_build_plan_from_signifiers(
        self, intents: List[str], workspace_id: Optional[str] = None, intent_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Try to build a BT JSON IR directly from signifier matches (fast path).

        Notes:
        - We still query EnvExplorer for SIGNIFIER_MATCH_REQUEST (it hosts the embedded RD4 engine).
        - We MUST NOT query EnvExplorer for ENV_CAPABILITIES_REQUEST / ENV_STATE_REQUEST in this path.
        - If any intent has no usable signifier match, return None (triggers normal LLM path).
        """
        try:
            signifier_matches = await self._gather_signifier_matches(intents, workspace_id=workspace_id, intent_type=intent_type)
        except Exception:
            return None

        if not isinstance(signifier_matches, dict) or not signifier_matches:
            return None

        tree = build_bt_from_signifiers(signifier_matches, intents)
        if tree is None:
            return None

        # Collect signifier IDs used for traceability
        signifier_ids: List[str] = []
        for intent in intents:
            match_data = signifier_matches.get(intent, {})
            if isinstance(match_data, dict):
                finals = match_data.get("final_matches", [])
                if finals:
                    signifier_ids.append(str(finals[0]))

        return {
            "plan_type": "behavior_tree",
            "tree": tree,
            "explanation": "Plan recovered from signifiers (no LLM call needed).",
            "intents": intents,
            "workspace_id": workspace_id,  # Include workspace_id for consistency
            "intent_type": intent_type,  # Include intent_type for signifier context
            "signifier_reuse": True,
            "signifier_ids": signifier_ids,
        }

    async def _gather_planning_context(self, intents: List[str], workspace_id: Optional[str] = None, intent_type: Optional[str] = None) -> Dict[str, Any]:
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
            signifier_matches = await self._gather_signifier_matches(intents, workspace_id=workspace_id, intent_type=intent_type)
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

    async def _gather_signifier_matches(self, intents: List[str], workspace_id: Optional[str] = None, intent_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask EnvExplorer for signifier matches per intent (best-effort).

        Also queries the community signifier API (if configured) for cross-environment matches.
        Local matches take priority; community matches supplement gaps.
        """
        intents = [str(i).strip() for i in (intents or []) if str(i).strip()]
        if not intents:
            return {}

        # Query local EnvExplorer (via SPADE RPC)
        explorer_jid = self.target_jids.get("explorer")
        local_out: Dict[str, Any] = {}
        if explorer_jid:
            async def _one_local(intent: str) -> Dict[str, Any]:
                logger.info(
                    demo("[INTENT_TYPE] Sending SIGNIFIER_MATCH_REQUEST: intent=%r, workspace_id=%r, intent_type=%r"),
                    intent,
                    workspace_id,
                    intent_type,
                )
                raw = await self._query_env_explorer(
                    message_type=MessageType.SIGNIFIER_MATCH_REQUEST.value,
                    body={
                        "intent": intent,
                        **({"workspace_id": str(workspace_id)} if workspace_id else {}),
                        **({"intent_type": str(intent_type)} if intent_type else {}),
                        "k": 5,
                    },
                    expect_type=MessageType.SIGNIFIER_MATCH_RESPONSE.value,
                )
                try:
                    data = json.loads(raw) if raw else {}
                    return data if isinstance(data, dict) else {}
                except Exception:
                    return {}

            local_results = await asyncio.gather(*[_one_local(i) for i in intents], return_exceptions=True)
            for intent, res in zip(intents, local_results):
                if isinstance(res, Exception):
                    local_out[intent] = {"error": "signifier_query_failed", "detail": str(res)}
                else:
                    local_out[intent] = res

        # Query community signifier API (if configured)
        community_out: Dict[str, Any] = {}
        if self.community_client:
            async def _one_community(intent: str) -> Dict[str, Any]:
                try:
                    data = await self.community_client.match_signifiers(intent)
                    if data:
                        # Tag community matches with source
                        for m in data.get("matches", []):
                            if isinstance(m, dict):
                                m["source"] = "community"
                    return data
                except Exception:
                    return {}

            community_results = await asyncio.gather(*[_one_community(i) for i in intents], return_exceptions=True)
            for intent, res in zip(intents, community_results):
                if isinstance(res, Exception):
                    community_out[intent] = {}
                else:
                    community_out[intent] = res if isinstance(res, dict) else {}

        # Merge: local matches take priority, community supplements
        out = self._merge_signifier_matches(local_out, community_out, intents)

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
            community_count = sum(1 for m in matches if isinstance(m, dict) and m.get("source") == "community")

            if finals:
                logger.info(
                    demo("Signifier search: intent=%r matches=%d (community=%d) final=%d top=%s"),
                    intent,
                    len(matches),
                    community_count,
                    len(finals),
                    finals[0],
                )
            else:
                logger.info(
                    demo("Signifier search: intent=%r matches=%d (community=%d) final=0 stored_total=%s"),
                    intent,
                    len(matches),
                    community_count,
                    total if total is not None else "?",
                )
        return out

    @staticmethod
    def _merge_signifier_matches(
        local: Dict[str, Any],
        community: Dict[str, Any],
        intents: List[str],
    ) -> Dict[str, Any]:
        """
        Merge local and community signifier matches.

        Local matches take priority. Community matches supplement gaps.
        """
        merged: Dict[str, Any] = {}
        for intent in intents:
            local_data = local.get(intent, {})
            community_data = community.get(intent, {})

            if not isinstance(local_data, dict):
                local_data = {}
            if not isinstance(community_data, dict):
                community_data = {}

            local_matches = local_data.get("matches", []) if isinstance(local_data.get("matches"), list) else []
            local_finals = local_data.get("final_matches", []) if isinstance(local_data.get("final_matches"), list) else []
            community_matches = community_data.get("matches", []) if isinstance(community_data.get("matches"), list) else []
            community_finals = community_data.get("final_matches", []) if isinstance(community_data.get("final_matches"), list) else []

            # Combine: local first, then community (dedup by signifier_id)
            seen_ids: set = set()
            combined_matches: List[Dict[str, Any]] = []
            for m in local_matches:
                if isinstance(m, dict):
                    sid = m.get("signifier_id", "")
                    if sid not in seen_ids:
                        seen_ids.add(sid)
                        combined_matches.append(m)
            for m in community_matches:
                if isinstance(m, dict):
                    sid = m.get("signifier_id", "")
                    if sid not in seen_ids:
                        seen_ids.add(sid)
                        combined_matches.append(m)

            # Finals: local finals first, then community finals
            combined_finals: List[str] = []
            seen_final_ids: set = set()
            for f in local_finals:
                sf = str(f)
                if sf not in seen_final_ids:
                    seen_final_ids.add(sf)
                    combined_finals.append(sf)
            for f in community_finals:
                sf = str(f)
                if sf not in seen_final_ids:
                    seen_final_ids.add(sf)
                    combined_finals.append(sf)

            merged[intent] = {
                "matches": combined_matches,
                "final_matches": combined_finals,
            }

            # Preserve error from local if present
            if local_data.get("error"):
                merged[intent]["error"] = local_data["error"]

        return merged

    # NOTE: _build_planning_prompt() removed — BT planning prompt construction is now
    # handled by AsyncBTPlanner and ami_agents.bt_planning.planning.prompts.


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
        intent_type: Optional[str] = None
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                ws = payload.get("workspace_id") or payload.get("workspace")
                if ws:
                    workspace_id = str(ws)
                # Extract intent_type from payload
                it = payload.get("intent_type")
                if it and str(it).upper() in ("EXPLICIT", "IMPLICIT"):
                    intent_type = str(it).upper()
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
            reply.body = json.dumps({"plan_type": "behavior_tree", "error": "missing_intent", "tree": None, "intents": []})
            # Propagate correlation_id/thread for request/response pairing.
            corr = msg.get_metadata(META_CORRELATION_ID)
            if corr:
                reply.set_metadata(META_CORRELATION_ID, corr)
            if msg.thread:
                reply.thread = msg.thread
            await self.send(reply)
            return

        logger.info(
            demo("Received GOAL_REQUEST: intents=%s workspace_id=%r from=%s"),
            intents,
            workspace_id,
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
                    "plan_type": "behavior_tree",
                    "error": "env_not_ready",
                    "detail": "Environment discovery not completed (timeout).",
                    "tree": None,
                    "intents": intents,
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

        plan_json = await self.agent._generate_plan(intents, workspace_id=workspace_id, intent_type=intent_type)

        try:
            parsed_plan = json.loads(plan_json or "{}")
            tree = parsed_plan.get("tree") if isinstance(parsed_plan, dict) else None
            has_tree = tree is not None and isinstance(tree, dict) and bool(tree)
            signifier_reuse = parsed_plan.get("signifier_reuse", False) if isinstance(parsed_plan, dict) else False
            is_impossible = parsed_plan.get("impossible", False) if isinstance(parsed_plan, dict) else False
            error = parsed_plan.get("error") if isinstance(parsed_plan, dict) else None
            logger.info(
                demo("Plan created: has_tree=%s signifier_reuse=%s impossible=%s error=%s"),
                has_tree, signifier_reuse, is_impossible, error,
            )
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
