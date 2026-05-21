"""UserMessageBehaviour: the main conversation state machine.

The LLM is invoked only for:
- NLU  (atomic segmentation)  via ``_segment_into_atomic_intents``
- NLG  (plan summary)         via ``_summarize_plan``
- NLG  (query formatting)     via ``_format_query_response``

Everything else (plan requesting, confirmation handling, execution,
signifier recording) is deterministic.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType
from ....shared.utils.spade_rpc import rpc_call, RpcTimeoutError
from ....shared.utils.demo_log import demo
from ....bt_planning.execution.ir_executor import IRExecutor
from ....bt_planning.execution.base import ExecutionResult
from ....bt_planning.signifier_bridge import extract_signifiers_from_bt
from ....shared.community.community_client import CommunitySignifierClient

from ..models import (
    AtomicIntent,
    ConversationPhase,
    ConversationState,
    Intent,
    CONFIRM_TOKENS,
    REJECT_TOKENS,
)
from ..prompts import (
    ATOMIC_SEGMENTATION_SYSTEM_PROMPT,
    ENV_CAPABILITIES_REQUEST_PARSER_PROMPT,
    ENV_STATE_REQUEST_PARSER_PROMPT,
    GOAL_REQUEST_PARSER_PROMPT,
    PLAN_SUMMARY_SYSTEM_PROMPT,
    QUERY_RESPONSE_SYSTEM_PROMPT,
)
from ....shared.models.intents import (
    ImplicitGoalIntent,
    ExplicitGoalIntent,
    goal_intent_from_dict,
)
from ..utils import (
    coerce_plan_dict,
    canonicalize_plan_for_hash,
    count_bt_nodes,
    bt_preview,
    loose_json_loads,
)
from ....shared.utils.logger import LoggerFactory


# ============================================================================
# Context builder helpers for per-span LLM prompt template variables
# ============================================================================


def _build_workspace_context(caps_data: dict) -> tuple[str, str]:
    """Build workspace context strings for prompt templates.

    Returns:
        (workspace_type_list, workspace_list) — two formatted strings for injection
    """
    if not caps_data or "workspaces" not in caps_data:
        return "", ""

    def _walk_workspaces(workspaces, parent_name=""):
        """Recursively walk workspace hierarchy and yield (type, name, full_name) tuples."""
        for ws in workspaces:
            ws_type = ws.get("id", "unknown")  # workspace_id as simple identifier
            ws_name = ws.get("name", "unknown")
            ws_full = f"{ws_name} ({ws_type})" if parent_name == "" else f"{ws_name} ({ws_type}, parent: {parent_name})"
            yield (ws_type, ws_name, ws_full)

            # Recurse into sub-workspaces
            for sub_node in _walk_workspaces(ws.get("sub_workspaces", []), ws_name):
                yield sub_node

    lines_type_list = []
    lines_list = []
    for ws_type, ws_name, ws_full in _walk_workspaces(caps_data.get("workspaces", [])):
        lines_type_list.append(f"ex:{ws_type.title()} — {ws_name}")
        lines_list.append(f"ex:{ws_type.title()} — {ws_full}")

    return "\n".join(lines_type_list), "\n".join(lines_list)


def _build_artifact_context(caps_data: dict) -> str:
    """Build artifact list context string for prompt templates.

    Returns:
        Formatted string for injection into {artifact_list} placeholder
    """
    if not caps_data or "workspaces" not in caps_data:
        return ""

    lines = []

    def _walk_artifacts(workspaces, workspace_name=""):
        """Recursively walk workspaces and extract artifacts."""
        for ws in workspaces:
            ws_name = ws.get("name", "unknown")

            # Process artifacts in this workspace
            for artifact in ws.get("artifacts", []):
                artifact_id = artifact.get("id", "unknown")
                artifact_name = artifact.get("name", "unknown")

                # Build action and property lists
                actions = []
                properties = []
                for aff in artifact.get("affordances", []):
                    aff_name = aff.get("name", "")
                    aff_type = aff.get("type", "")
                    params = aff.get("parameters", [])
                    param_str = f"params: {', '.join(params)}" if params else ""

                    if aff_type == "action_affordance":
                        if param_str:
                            actions.append(f"{aff_name} ({param_str})")
                        else:
                            actions.append(aff_name)
                    elif aff_type == "property_affordance":
                        if param_str:
                            properties.append(f"{aff_name} ({param_str})")
                        else:
                            properties.append(aff_name)

                action_str = "actions=[" + ", ".join(actions) + "]" if actions else ""
                prop_str = "properties=[" + ", ".join(properties) + "]" if properties else ""
                combined = ", ".join([s for s in [action_str, prop_str] if s])

                lines.append(f"ex:Device — {artifact_name} ({artifact_id}, workspace: {ws_name}): {combined}")

            # Recurse into sub-workspaces
            _walk_artifacts(ws.get("sub_workspaces", []), ws_name)

    _walk_artifacts(caps_data.get("workspaces", []))
    return "\n".join(lines)


# ============================================================================
# UserMessageBehaviour class
# ============================================================================


class UserMessageBehaviour(CyclicBehaviour):
    """UserMessage behavior with configurable self.logger."""

    def __init__(self, logger=None):
        """
        Initialize with optional self.logger.

        Args:
            logger: Logger instance. If None, creates a basic self.logger.
        """
        super().__init__()
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")
    """Main behaviour handling the full user-message conversation flow.

    Registered with ``Template(metadata={"message_type": "llm"})``.
    """

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self):  # noqa: C901
        msg = await self.receive(timeout=1)
        if not msg:
            return

        thread = str(getattr(msg, "thread", None) or "__default__")
        text = (msg.body or "").strip()
        if not text:
            return

        conv = self.agent.get_conversation(thread)

        if conv.phase == ConversationPhase.AWAITING_CONFIRMATION:
            await self._handle_confirmation(msg, thread, text, conv)
            return

        conv.phase = ConversationPhase.SEGMENTING
        conv.user_message = text

        # Fetch capabilities once for use in segmentation and handlers
        capabilities_ctx = await self._fetch_capabilities()

        # Parse capabilities JSON for use in per-span parsers
        try:
            caps_data = json.loads(capabilities_ctx) if capabilities_ctx else {}
        except (json.JSONDecodeError, AttributeError):
            caps_data = {}

        # Segment the user text into atomic intents
        atomic_intents = await self._segment_into_atomic_intents(text, capabilities_ctx)
        if not atomic_intents:
            await self._reply(msg, "I couldn't understand your request. Could you rephrase?")
            conv.phase = ConversationPhase.IDLE
            return

        # Group intents by category
        caps_intents = [a for a in atomic_intents if a.category == "ENV_CAPABILITIES_REQUEST"]
        state_intents = [a for a in atomic_intents if a.category == "ENV_STATE_REQUEST"]
        goal_intents = [a for a in atomic_intents if a.category == "GOAL_REQUEST"]

        # Phase 1: ENV_CAPABILITIES_REQUEST (if any)
        if caps_intents:
            conv.phase = ConversationPhase.EXTRACTING_INTENTS
            for intent in caps_intents:
                extraction = await self._parse_atomic_intent(intent.span, intent.category, caps_data)
                await self._handle_query_capabilities(msg, thread, conv, capabilities_ctx)

        # Phase 2: ENV_STATE_REQUEST (if any)
        if state_intents:
            conv.phase = ConversationPhase.EXTRACTING_INTENTS
            for intent in state_intents:
                extraction = await self._parse_atomic_intent(intent.span, intent.category, caps_data)
                await self._handle_query_state(msg, thread, conv, extraction)

        # Phase 3: GOAL_REQUEST (if any) — batch all goal intents into a single request
        if goal_intents:
            conv.phase = ConversationPhase.EXTRACTING_INTENTS
            # Parse all goal intent spans in parallel
            extractions = await asyncio.gather(
                *[self._parse_atomic_intent(intent.span, intent.category, caps_data) for intent in goal_intents],
                return_exceptions=True,
            )
            # Convert to typed intent objects, filtering errors
            parsed_intents = []
            for extraction in extractions:
                if isinstance(extraction, Exception):
                    self.logger.error("Failed to parse atomic intent: %s", extraction)
                    continue
                if isinstance(extraction, dict):
                    parsed = goal_intent_from_dict(extraction)
                    parsed_intents.append(parsed)

            # Send all parsed intents to ISA in a single GOAL_REQUEST
            if parsed_intents:
                conv.intents = parsed_intents
                await self._handle_goal(msg, thread, conv, parsed_intents)

    # ------------------------------------------------------------------
    # Atomic segmentation and per-span parsing (LLM calls)
    # ------------------------------------------------------------------

    async def _segment_into_atomic_intents(
        self, user_text: str, capabilities_ctx: str = ""
    ) -> list[AtomicIntent]:
        """Call LLM to segment user message into atomic intents with categories."""
        prompt = ATOMIC_SEGMENTATION_SYSTEM_PROMPT.format(capabilities=capabilities_ctx)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ]
        try:
            response = await self.agent.llm_client.chat.completions.create(
                model=self.agent.llm_model,
                messages=messages,
                **self.agent.build_llm_kwargs(),
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = loose_json_loads(raw)
            if not isinstance(parsed, dict):
                self.logger.warning("LLM atomic segmentation returned non-dict: %r", raw[:200])
                return []

            intents_data = parsed.get("intents", [])
            if not isinstance(intents_data, list):
                self.logger.warning("LLM atomic segmentation returned non-list intents")
                return []

            result = []
            for item in intents_data:
                if not isinstance(item, dict):
                    continue
                span = item.get("span", "").strip()
                category = item.get("category", "").strip()
                reason = item.get("reason", "").strip()
                if span and category in ("GOAL_REQUEST", "ENV_STATE_REQUEST", "ENV_CAPABILITIES_REQUEST") and reason:
                    result.append(AtomicIntent(span=span, category=category, reason=reason))
            return result
        except Exception as exc:
            self.logger.error("LLM atomic segmentation failed: %s", exc)
            return []

    async def _parse_atomic_intent(
        self, span: str, category: str, caps_data: dict
    ) -> dict:
        """Parse a single atomic intent span using LLM per-span prompts.

        Args:
            span: The verbatim user text for this atomic intent
            category: One of "GOAL_REQUEST", "ENV_STATE_REQUEST", "ENV_CAPABILITIES_REQUEST"
            caps_data: Parsed capabilities JSON dict from _fetch_capabilities()

        Returns:
            Dict with parsed structured intent fields (LLM output), or fallback dict on error
        """
        # Choose prompt and template variables by category
        if category == "GOAL_REQUEST":
            prompt_template = GOAL_REQUEST_PARSER_PROMPT
            workspace_list, _ = _build_workspace_context(caps_data)
            artifact_list = _build_artifact_context(caps_data)
            ontology = self.agent.ontology_ttl
            prompt = prompt_template.format(
                workspace_list=workspace_list,
                artifact_list=artifact_list,
                ontology=ontology,
            )
        elif category == "ENV_STATE_REQUEST":
            prompt_template = ENV_STATE_REQUEST_PARSER_PROMPT
            workspace_type_list, _ = _build_workspace_context(caps_data)
            artifact_list = _build_artifact_context(caps_data)
            ontology = self.agent.ontology_ttl
            prompt = prompt_template.format(
                workspace_type_list=workspace_type_list,
                artifact_list=artifact_list,
                ontology=ontology,
            )
        elif category == "ENV_CAPABILITIES_REQUEST":
            prompt_template = ENV_CAPABILITIES_REQUEST_PARSER_PROMPT
            ontology = self.agent.ontology_ttl
            prompt = prompt_template.format(ontology=ontology)
        else:
            # Unknown category — return minimal fallback
            return {"text_intent": span}

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": span},
        ]

        try:
            response = await self.agent.llm_client.chat.completions.create(
                model=self.agent.llm_model,
                messages=messages,
                **self.agent.build_llm_kwargs(),
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = loose_json_loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            self.logger.warning("LLM per-span parsing failed for [%s]: %s", category, exc)

        # Fallback: minimal dict with only text_intent
        return {"text_intent": span}

    # ------------------------------------------------------------------
    # NLG: plan summary (LLM call)
    # ------------------------------------------------------------------

    async def _summarize_plan(self, plan_json: str) -> str:
        """Call LLM to produce a user-friendly plan summary."""
        messages = [
            {"role": "system", "content": PLAN_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": plan_json},
        ]
        try:
            response = await self.agent.llm_client.chat.completions.create(
                model=self.agent.llm_model,
                messages=messages,
                **self.agent.build_llm_kwargs(),
            )
            return (response.choices[0].message.content or "").strip() or (
                "Plan ready. Does this plan look good to you?"
            )
        except Exception as exc:
            self.logger.error("LLM plan summary failed: %s", exc)
            return "I have a plan ready. Does this plan look good to you?"

    # ------------------------------------------------------------------
    # NLG: query response formatting (LLM call)
    # ------------------------------------------------------------------

    async def _format_query_response(self, raw_data: str, user_text: str) -> str:
        """Call LLM to format raw query results for the user."""
        messages = [
            {"role": "system", "content": QUERY_RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"User asked: {user_text}\n\nRaw data:\n{raw_data}"},
        ]
        try:
            response = await self.agent.llm_client.chat.completions.create(
                model=self.agent.llm_model,
                messages=messages,
                **self.agent.build_llm_kwargs(),
            )
            return (response.choices[0].message.content or "").strip() or raw_data
        except Exception as exc:
            self.logger.error("LLM query formatting failed: %s", exc)
            return raw_data

    # ------------------------------------------------------------------
    # DETERMINISTIC: handle goal → request plan → summarize
    # ------------------------------------------------------------------

    async def _handle_goal(
        self, msg, thread: str, conv: ConversationState, parsed_intents: list
    ) -> None:
        """Handle a batch of parsed goal intents and send to InteractionSolver.

        Args:
            parsed_intents: List of ExplicitGoalIntent or ImplicitGoalIntent objects
        """
        if not parsed_intents:
            await self._reply(msg, "Could not parse your request. Could you rephrase?")
            conv.phase = ConversationPhase.IDLE
            return

        # Store the parsed intents in conversation state
        conv.intents = parsed_intents

        # Log the intents being sent
        intent_texts = [i.to_query_string() for i in parsed_intents]
        intent_types = [i.intent_type for i in parsed_intents]
        self.logger.info(
            demo("Parsed %d goal intents: %s (types: %s)"),
            len(parsed_intents),
            intent_texts,
            intent_types,
        )

        solver_jid = self.agent.target_jids.get("solver")
        if not solver_jid:
            await self._reply(msg, "Error: InteractionSolver is not configured.")
            conv.phase = ConversationPhase.IDLE
            return

        conv.phase = ConversationPhase.AWAITING_PLAN
        body: Dict[str, Any] = {
            "intents": [i.to_wire_dict() for i in parsed_intents],
        }
        if conv.workspace_id:
            body["workspace_id"] = str(conv.workspace_id)

        timeout = self.agent.goal_request_timeout

        self.logger.info(
            demo("UA -> InteractionSolver GOAL_REQUEST: %d intents, types: %s"),
            len(parsed_intents),
            intent_types,
        )
        try:
            result = await rpc_call(
                self.agent,
                to_jid=str(solver_jid),
                request_type=MessageType.GOAL_REQUEST.value,
                body=body,
                expect_type=MessageType.PLAN_CREATED.value,
                timeout=timeout,
                thread=thread if thread != "__default__" else None,
            )
            await self._process_plan_response(msg, thread, conv, result.body)
        except RpcTimeoutError:
            await self._reply(msg, "Planning timed out. Could you try again?")
            conv.phase = ConversationPhase.IDLE
        except Exception as exc:
            self.logger.error("Goal request failed: %s", exc)
            await self._reply(msg, f"Planning failed: {exc}")
            conv.phase = ConversationPhase.IDLE

    async def _process_plan_response(
        self, msg, thread: str, conv: ConversationState, plan_body: str
    ) -> None:
        """Store plan, summarise via LLM, and ask for confirmation."""
        plan_obj = coerce_plan_dict(plan_body)
        if not isinstance(plan_obj, dict):
            await self._reply(msg, "Received an invalid plan. Could you try rephrasing your request?")
            conv.phase = ConversationPhase.IDLE
            return

        tree = plan_obj.get("tree")
        if not tree or not isinstance(tree, dict):
            error = plan_obj.get("error", "unknown")
            detail = plan_obj.get("detail") or plan_obj.get("explanation", "")
            await self._reply(msg, f"Planning failed: {error}. {detail}\nWhat would you like to change?")
            conv.phase = ConversationPhase.IDLE
            return

        canonical, plan_hash = canonicalize_plan_for_hash(plan_obj)
        conv.plan_json = canonical
        conv.plan_hash = plan_hash

        node_count = count_bt_nodes(tree)
        preview = bt_preview(tree)
        self.logger.info(demo("Plan stored: hash=%s nodes=%d preview=%s"), plan_hash, node_count, preview)

        conv.phase = ConversationPhase.SUMMARIZING_PLAN
        summary = await self._summarize_plan(plan_body if isinstance(plan_body, str) else json.dumps(plan_obj))
        conv.plan_summary = summary
        conv.phase = ConversationPhase.AWAITING_CONFIRMATION

        await self._reply(msg, summary)

    # ------------------------------------------------------------------
    # DETERMINISTIC: confirmation handling
    # ------------------------------------------------------------------

    async def _handle_confirmation(
        self, msg, thread: str, text: str, conv: ConversationState
    ) -> None:
        low = text.lower().strip()

        if low in CONFIRM_TOKENS:
            conv.phase = ConversationPhase.EXECUTING
            self.logger.info(demo("User confirmed plan: thread=%s"), thread)
            exec_result = await self._execute_plan(thread, conv)

            if exec_result.success:
                await self._reply(msg, "Done! The plan was executed successfully.")
            else:
                detail = exec_result.error or exec_result.final_status
                await self._reply(msg, f"Execution failed: {detail}")

            conv.clear_plan()
            conv.phase = ConversationPhase.IDLE

        elif low in REJECT_TOKENS:
            self.logger.info(demo("User rejected plan: thread=%s"), thread)
            conv.clear_plan()
            conv.phase = ConversationPhase.IDLE
            await self._reply(msg, "Okay, I've discarded that plan. What would you like to change?")

        else:
            self.logger.info(demo("New request while awaiting confirmation, discarding plan: thread=%s"), thread)
            conv.clear_plan()
            conv.phase = ConversationPhase.IDLE

            conv.user_message = text
            conv.phase = ConversationPhase.SEGMENTING
            capabilities_ctx = await self._fetch_capabilities()
            atomic_intents = await self._segment_into_atomic_intents(text, capabilities_ctx)
            if not atomic_intents:
                await self._reply(msg, "I couldn't understand your request. Could you rephrase?")
                conv.phase = ConversationPhase.IDLE
                return

            # Group intents by category
            caps_intents = [a for a in atomic_intents if a.category == "ENV_CAPABILITIES_REQUEST"]
            state_intents = [a for a in atomic_intents if a.category == "ENV_STATE_REQUEST"]
            goal_intents = [a for a in atomic_intents if a.category == "GOAL_REQUEST"]

            # Phase 1: ENV_CAPABILITIES_REQUEST (if any)
            if caps_intents:
                conv.phase = ConversationPhase.EXTRACTING_INTENTS
                for intent in caps_intents:
                    stub = self._parse_atomic_intent(intent.span, intent.category, capabilities_ctx)
                    await self._handle_query_capabilities(msg, thread, conv, capabilities_ctx)

            # Phase 2: ENV_STATE_REQUEST (if any)
            if state_intents:
                conv.phase = ConversationPhase.EXTRACTING_INTENTS
                for intent in state_intents:
                    stub = self._parse_atomic_intent(intent.span, intent.category, capabilities_ctx)
                    await self._handle_query_state(msg, thread, conv, stub)

            # Phase 3: GOAL_REQUEST (if any)
            if goal_intents:
                conv.phase = ConversationPhase.EXTRACTING_INTENTS
                for intent in goal_intents:
                    stub = self._parse_atomic_intent(intent.span, intent.category, capabilities_ctx)
                    await self._handle_goal(msg, thread, conv, stub)

    # ------------------------------------------------------------------
    # DETERMINISTIC: plan execution
    # ------------------------------------------------------------------

    async def _execute_plan(self, thread: str, conv: ConversationState) -> ExecutionResult:
        """Execute the stored plan via IRExecutor (deterministic)."""
        plan_obj = coerce_plan_dict(conv.plan_json)
        if not isinstance(plan_obj, dict):
            return ExecutionResult(success=False, error="Invalid plan JSON")

        tree_spec = plan_obj.get("tree", {})
        raw_intents = plan_obj.get("intents", [])

        # Reconstruct typed goal intents from the wire format
        intents = []
        for d in raw_intents:
            if isinstance(d, dict) and d.get("category") in ("implicit", "explicit"):
                intents.append(goal_intent_from_dict(d))
            elif isinstance(d, dict):
                # Fallback for non-goal intents (state/capabilities requests)
                intents.append(Intent(intent_text=d.get("text_intent", str(d))))
            else:
                intents.append(Intent(intent_text=str(d)))

        is_signifier_reuse = plan_obj.get("signifier_reuse", False)
        intent_type = plan_obj.get("intent_type")
        workspace_id = plan_obj.get("workspace_id")

        if not tree_spec or not isinstance(tree_spec, dict):
            return ExecutionResult(success=False, error="Plan has no behavior tree to execute")

        await self.agent.ensure_execution_engine_ready()

        node_count = count_bt_nodes(tree_spec)
        self.logger.info(demo("Executing BT: thread=%s nodes=%d signifier_reuse=%s intent_type=%s"), thread, node_count, is_signifier_reuse, intent_type)

        executor = IRExecutor(max_ticks=self.agent.bt_max_ticks)
        loop = asyncio.get_event_loop()
        try:
            exec_result: ExecutionResult = await loop.run_in_executor(
                None,
                executor.execute_from_spec,
                tree_spec,
            )
        except Exception as exc:
            self.logger.warning(demo("BT execution failed: %s"), exc)
            return ExecutionResult(success=False, error=str(exc))

        self.logger.info(
            demo("BT execution complete: success=%s ticks=%d status=%s"),
            exec_result.success, exec_result.ticks, exec_result.final_status,
        )

        self.agent.state_memory.clear()
        self.logger.info(demo("State memory cache cleared after BT execution"))

        # Only record signifiers for implicit goal intents (explicit goals don't produce experiences)
        if exec_result.success and not is_signifier_reuse and intent_type == "implicit":
            await self._record_signifiers(tree_spec, intents, exec_result, thread, intent_type, workspace_id)

        return exec_result

    # ------------------------------------------------------------------
    # DETERMINISTIC: signifier recording
    # ------------------------------------------------------------------

    async def _record_signifiers(
        self,
        tree_spec: dict,
        intents: list,
        exec_result: ExecutionResult,
        thread: str,
        intent_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        """Extract and record signifiers from an executed BT.

        Only called for implicit goals (where the system inferred the device from context).
        """
        intent_strings = [i.to_query_string() for i in intents]
        structured_intents = [
            i.to_wire_dict() if isinstance(i, (ImplicitGoalIntent, ExplicitGoalIntent, Intent))
            else (i if isinstance(i, dict) else None)
            for i in intents
        ]
        structured_intents = [s for s in structured_intents if s is not None]

        state_snapshot: Optional[dict] = None
        explorer_jid = self.agent.target_jids.get("explorer")
        if explorer_jid and workspace_id:
            try:
                self.logger.info(demo("Querying environment state for signifier context (workspace_id=%r)"), workspace_id)

                state_response = await rpc_call(
                    self.agent,
                    to_jid=str(explorer_jid),
                    request_type=MessageType.ENV_STATE_REQUEST.value,
                    body={},
                    expect_type=MessageType.ENV_STATE_RESPONSE.value,
                    timeout=self.agent.signifier_match_timeout,
                )

                if state_response and state_response.body:
                    raw_state = json.loads(state_response.body) if isinstance(state_response.body, str) else state_response.body

                    if isinstance(raw_state, dict):
                        if "artifacts" in raw_state and isinstance(raw_state["artifacts"], dict):
                            artifacts_dict = raw_state["artifacts"]
                        else:
                            artifacts_dict = raw_state

                        filtered_artifacts = {}
                        for artifact_id, artifact_info in artifacts_dict.items():
                            if isinstance(artifact_info, dict):
                                artifact_ws = artifact_info.get("workspace_id")
                                if artifact_ws and (str(artifact_ws) == str(workspace_id) or workspace_id in str(artifact_ws)):
                                    filtered_artifacts[artifact_id] = artifact_info

                        state_snapshot = {"artifacts": filtered_artifacts}
                        self.logger.info(
                            demo("State snapshot retrieved: %d artifacts for workspace_id=%r"),
                            len(filtered_artifacts), workspace_id
                        )
            except Exception as e:
                self.logger.warning(demo("Failed to query state snapshot: %s (continuing without state)"), e)

        signifiers = extract_signifiers_from_bt(
            tree_spec=tree_spec,
            intents=intent_strings,
            was_successful=exec_result.success,
            workspace_id=workspace_id,
            state_snapshot=state_snapshot,
            intent_type=intent_type,
            structured_intents=structured_intents,
        )
        if not signifiers:
            self.logger.info(demo("No signifiers extracted from BT"))
            return

        self.logger.info(demo("Recording %d signifiers from BT execution (intent_type=%s)"), len(signifiers), intent_type)

        # 1. Record locally via EnvExplorer
        if explorer_jid:
            try:
                rec_res = await rpc_call(
                    self.agent,
                    to_jid=str(explorer_jid),
                    request_type=MessageType.SIGNIFIER_RECORD_EXECUTION_REQUEST.value,
                    body={
                        "plan_type": "behavior_tree",
                        "tree": tree_spec,
                        "signifiers": signifiers,
                        "execution_result": exec_result.to_dict(),
                    },
                    expect_type=MessageType.SIGNIFIER_RECORD_EXECUTION_RESPONSE.value,
                    timeout=self.agent.rpc_call_timeout,
                    thread=(thread if thread != "__default__" else None),
                )
                created_count = None
                try:
                    rec_payload = json.loads(rec_res.body or "{}")
                    if isinstance(rec_payload, dict):
                        created_count = rec_payload.get("created_count")
                except Exception:
                    pass
                self.logger.info(demo("Local signifier recording: created_count=%s"), created_count or "?")
            except Exception as exc:
                self.logger.info(demo("Failed to record signifiers locally: %s"), exc)

        # 2. Publish to community
        community_client = self.agent.community_client
        if isinstance(community_client, CommunitySignifierClient):
            published = 0
            for sig in signifiers:
                try:
                    ok = await community_client.publish_signifier(sig)
                    if ok:
                        published += 1
                except Exception:
                    pass
            self.logger.info(demo("Community signifier publishing: %d/%d published"), published, len(signifiers))

    # ------------------------------------------------------------------
    # DETERMINISTIC: query handlers
    # ------------------------------------------------------------------

    async def _handle_query_capabilities(
        self, msg, thread: str, conv: ConversationState, capabilities_ctx: str
    ) -> None:
        """Handle a capabilities query (deterministic fetch + LLM formatting)."""
        if not capabilities_ctx:
            capabilities_ctx = await self._fetch_capabilities()

        if not capabilities_ctx:
            await self._reply(msg, "Unable to fetch environment capabilities right now.")
            conv.phase = ConversationPhase.IDLE
            return

        formatted = await self._format_query_response(capabilities_ctx, conv.user_message)
        await self._reply(msg, formatted)
        conv.phase = ConversationPhase.IDLE

    async def _handle_query_state(
        self, msg, thread: str, conv: ConversationState, extraction: dict
    ) -> None:
        """Handle a state query (deterministic fetch with cache + LLM formatting)."""
        artifact_id = extraction.get("artifact_id")
        property_uri = extraction.get("property_uri")

        state_memory = self.agent.state_memory
        if property_uri and state_memory.has(property_uri):
            cached = state_memory.get(property_uri)
            self.logger.info(demo("State cache HIT: property_uri=%r value=%r"), property_uri, cached)
            raw_data = json.dumps({"property_uri": property_uri, "value": cached, "source": "cache"})
            formatted = await self._format_query_response(raw_data, conv.user_message)
            await self._reply(msg, formatted)
            conv.phase = ConversationPhase.IDLE
            return

        explorer_jid = self.agent.target_jids.get("explorer")
        if not explorer_jid:
            await self._reply(msg, "Error: EnvExplorer is not configured.")
            conv.phase = ConversationPhase.IDLE
            return

        payload: Dict[str, Any] = {}
        if artifact_id:
            payload["artifact_id"] = artifact_id
        if property_uri:
            payload["property_uri"] = property_uri

        self.logger.info(
            demo("UA -> EnvExplorer ENV_STATE_REQUEST: artifact_id=%r property_uri=%r"),
            artifact_id, property_uri,
        )
        try:
            result = await rpc_call(
                self.agent,
                to_jid=str(explorer_jid),
                request_type=MessageType.ENV_STATE_REQUEST.value,
                body=payload,
                expect_type=MessageType.ENV_STATE_RESPONSE.value,
                timeout=self.agent.rpc_call_timeout,
            )
            try:
                state_data = json.loads(result.body) if isinstance(result.body, str) else result.body
                if isinstance(state_data, dict):
                    if property_uri and "value" in state_data:
                        state_memory.store(property_uri, state_data["value"])
                    elif "artifacts" in state_data and isinstance(state_data["artifacts"], dict):
                        state_memory.store_bulk(state_data["artifacts"])
                    else:
                        state_memory.store_bulk(state_data)
            except Exception:
                pass  # best-effort caching

            formatted = await self._format_query_response(result.body, conv.user_message)
            await self._reply(msg, formatted)
        except RpcTimeoutError:
            await self._reply(msg, "Timeout querying environment state. Please try again.")
        except Exception as exc:
            await self._reply(msg, f"Error querying state: {exc}")

        conv.phase = ConversationPhase.IDLE

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fetch_capabilities(self, detail_level: str = "summary") -> str:
        """Fetch environment capabilities from EnvExplorer via RPC.

        Args:
            detail_level: One of "summary" (default, lightweight hierarchical JSON)
                         or "detailed" (full RDF/Turtle graph).
        """
        explorer_jid = self.agent.target_jids.get("explorer")
        if not explorer_jid:
            return ""
        try:
            result = await rpc_call(
                self.agent,
                to_jid=str(explorer_jid),
                request_type=MessageType.ENV_CAPABILITIES_REQUEST.value,
                body={"query": "all", "detail_level": detail_level},
                expect_type=MessageType.ENV_CAPABILITIES_RESPONSE.value,
                timeout=self.agent.rpc_call_timeout,
            )
            return result.body or ""
        except Exception as exc:
            self.logger.warning("Failed to fetch capabilities: %s", exc)
            return ""

    async def _reply(self, original_msg, text: str) -> None:
        """Send a reply XMPP message to the user."""
        reply = original_msg.make_reply()
        reply.body = text
        await self.send(reply)
