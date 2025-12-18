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
from ...shared.models.plan import BehaviorTreePlan, NodeTemplate, Plan
from ...shared.protocols.llm_protocol import IPlanGenerator

logger = logging.getLogger("InteractionSolver")

# System prompt for strict multi-intent JSON-Plan 1.2 planning.
INTERACTION_SOLVER_SYSTEM_PROMPT = """
You are Interaction-Solver. You receive a list of intents and must return an executable plan in JSON-Plan 1.2 format.

You will be given environment context (affordances + state) in the user message. You MUST NOT invent actions, URIs, or methods.

MULTI-INTENT RULE (STRICT):
- The plan must satisfy ALL provided intents.
- Each step.intent MUST match exactly one of the provided intents.
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
 - Output only the JSON plan (no additional prose). If no plan is feasible, return a JSON with plan_version=1.2, error, detail, steps=[].
"""

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

        # --- LLM client config (direct OpenAI call) ---
        llm_root = self.config.get("llm", {}) or {}
        provider_name = llm_root.get("default_provider", "openai")
        provider_cfg = (llm_root.get("providers", {}) or {}).get(provider_name, {}) or {}
        planning_llm = (self.config.get("planning", {}) or {}).get("llm_planning", {}) or {}

        self.model = planning_llm.get("model") or provider_cfg.get("model") or "gpt-4"
        self.temperature = planning_llm.get("temperature", provider_cfg.get("temperature", 0.7))
        self.max_tokens = planning_llm.get("max_tokens", provider_cfg.get("max_tokens", 1500))

        # Prefer explicit YAML config; fall back to environment variable.
        api_key = provider_cfg.get("api_key") or llm_root.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = provider_cfg.get("base_url") or llm_root.get("base_url") or "https://api.openai.com/v1"
        timeout = (llm_root.get("retry", {}) or {}).get("timeout", 30)

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

    async def _generate_plan(self, intents: List[str], workspace_id: Optional[str] = None) -> str:
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

        try:
            context = await self._gather_planning_context(intents, workspace_id=workspace_id)
        except Exception as e:
            logger.warning(f"Context gathering failed: {e}")
            return json.dumps(
                {
                    "plan_version": "1.2",
                    "error": "context_gathering_failed",
                    "detail": str(e),
                    "steps": [],
                },
                indent=2,
            )

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
            completion = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=prompt_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
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

                return json.dumps(parsed, indent=2)
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
            return json.dumps(fallback, indent=2)

    async def _gather_planning_context(self, intents: List[str], workspace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Pull environment context once: affordances and state (from EnvExplorer).
        Returns a dict with capped affordances and optional state snapshot.
        """
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

        return {
            "intents": intents,
            "affordances": affordances,
            "state": state_payload.get("state") or state_payload,
        }

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
            "5) Evidence:\n"
            "   - Every step must include at least one reasons entry with at least one evidence item grounded in the state.\n"
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
            '      "reasons": [\n'
            "        {\n"
            '          "property": "<property satisfied>",\n'
            '          "direction": "<increase|decrease|set>",\n'
            '          "evidence": [\n'
            "            {\n"
            '              "artifact": "<sensor-or-artifact URI>",\n'
            '              "property": "<attribute name>",\n'
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
            "instructions": [
                "Use only provided affordance URIs and methods; do not invent or alter them.",
                "If workspace_id is provided, only use affordances/artifacts from that workspace.",
                "Cover ALL intents. Each step.intent must match exactly one provided intent.",
                "Include at least one reason with evidence per step.",
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
        try:
            payload = json.loads(msg.body or "{}")
            if isinstance(payload, dict):
                ws = payload.get("workspace_id") or payload.get("workspace")
                if ws:
                    workspace_id = str(ws)
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

        plan_json = await self.agent._generate_plan(intents, workspace_id=workspace_id)

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
