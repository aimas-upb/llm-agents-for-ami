"""
Composable SPADE behaviours for the User Assistant agent.

The LLM is invoked only for:
- NLU  (intent extraction)   via ``_extract_intents``
- NLG  (plan summary)        via ``_summarize_plan``
- NLG  (query formatting)    via ``_format_query_response``

Everything else (plan requesting, confirmation handling, execution,
signifier recording) is deterministic.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional

from spade.behaviour import CyclicBehaviour

from ...shared.models.messages import MessageType
from ...shared.utils.spade_rpc import rpc_call, RpcTimeoutError
from ...shared.utils.demo_log import demo
from ...bt_planning.execution.ir_executor import IRExecutor
from ...bt_planning.execution.base import ExecutionResult
from ...bt_planning.signifier_bridge import extract_signifiers_from_bt
from ...shared.community.community_client import CommunitySignifierClient

from .models import ConversationPhase, ConversationState, Intent, CONFIRM_TOKENS, REJECT_TOKENS
from .prompts import (
    INTENT_EXTRACTION_SYSTEM_PROMPT,
    PLAN_SUMMARY_SYSTEM_PROMPT,
    QUERY_RESPONSE_SYSTEM_PROMPT,
)
from .utils import (
    coerce_plan_dict,
    canonicalize_plan_for_hash,
    count_bt_nodes,
    bt_preview,
    loose_json_loads,
)

logger = logging.getLogger("UserAssistant")

# Re-export for convenience (canonical definitions live in models.py).
_CONFIRM_TOKENS = CONFIRM_TOKENS
_REJECT_TOKENS = REJECT_TOKENS


class UserMessageBehaviour(CyclicBehaviour):
    """Main behaviour handling the full user-message conversation flow.

    Registered with ``Template(metadata={"message_type": "llm"})``.
    """

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self):  # noqa: C901 (complexity acceptable for a state-machine entry)
        msg = await self.receive(timeout=1)
        if not msg:
            return

        thread = str(getattr(msg, "thread", None) or "__default__")
        text = (msg.body or "").strip()
        if not text:
            return

        conv = self.agent.get_conversation(thread)

        # ── CONFIRMATION shortcut (deterministic, no LLM) ──
        if conv.phase == ConversationPhase.AWAITING_CONFIRMATION:
            await self._handle_confirmation(msg, thread, text, conv)
            return

        # ── NEW MESSAGE: LLM intent extraction ──
        conv.phase = ConversationPhase.EXTRACTING_INTENTS
        conv.user_message = text

        # Fetch capabilities (context for intent extraction)
        capabilities_ctx = await self._fetch_capabilities()

        extraction = await self._extract_intents(text, capabilities_ctx)
        classification = extraction.get("classification", "unclear")

        if classification == "goal":
            await self._handle_goal(msg, thread, conv, extraction)
        elif classification == "query_capabilities":
            await self._handle_query_capabilities(msg, thread, conv, capabilities_ctx)
        elif classification == "query_state":
            await self._handle_query_state(msg, thread, conv, extraction)
        elif classification == "confirmation":
            # Confirmation without a pending plan
            await self._reply(msg, "I don't have a pending plan right now. What would you like to do?")
            conv.phase = ConversationPhase.IDLE
        elif classification == "unclear":
            question = extraction.get("question", "Could you clarify what you would like?")
            await self._reply(msg, question)
            conv.phase = ConversationPhase.IDLE
        else:
            await self._reply(msg, "I'm not sure I understood that. Could you rephrase?")
            conv.phase = ConversationPhase.IDLE

    # ------------------------------------------------------------------
    # NLU: intent extraction (LLM call)
    # ------------------------------------------------------------------

    async def _extract_intents(self, user_text: str, capabilities: str) -> dict:
        """Call LLM to classify the user message and extract structured intents."""
        user_content = f"User message: {user_text}"
        if capabilities:
            user_content = f"Capabilities:\n{capabilities}\n\n{user_content}"

        messages = [
            {"role": "system", "content": INTENT_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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
            logger.warning("LLM intent extraction returned non-dict: %r", raw[:200])
            return {"classification": "unclear", "question": "Could you clarify what you would like?"}
        except Exception as exc:
            logger.error("LLM intent extraction failed: %s", exc)
            return {"classification": "unclear", "question": "Something went wrong. Could you try again?"}

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
            logger.error("LLM plan summary failed: %s", exc)
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
            logger.error("LLM query formatting failed: %s", exc)
            return raw_data  # Fallback: raw data is better than nothing

    # ------------------------------------------------------------------
    # DETERMINISTIC: handle goal → request plan → summarize
    # ------------------------------------------------------------------

    async def _handle_goal(
        self, msg, thread: str, conv: ConversationState, extraction: dict
    ) -> None:
        raw_intents = extraction.get("intents", [])
        conv.intents = [Intent.from_dict(i) for i in raw_intents if isinstance(i, dict)]
        conv.workspace_id = extraction.get("workspace_id")
        intent_type = extraction.get("intent_type", "implicit")  # Extract intent_type from LLM response

        if not conv.intents:
            await self._reply(msg, "I couldn't derive any specific intents. Could you be more precise?")
            conv.phase = ConversationPhase.IDLE
            return

        intent_strings = [intent.intent_text or intent.to_canonical_string() for intent in conv.intents]
        logger.info(demo("Derived intents: %s  workspace=%s  intent_type=%s"), intent_strings, conv.workspace_id, intent_type.upper())

        # Send GOAL_REQUEST to InteractionSolver (deterministic RPC)
        solver_jid = self.agent.target_jids.get("solver")
        if not solver_jid:
            await self._reply(msg, "Error: InteractionSolver is not configured.")
            conv.phase = ConversationPhase.IDLE
            return

        conv.phase = ConversationPhase.AWAITING_PLAN
        body: Dict[str, Any] = {
            "intents": intent_strings,
            "structured_intents": [i.to_dict() for i in conv.intents],
            "intent_type": intent_type,  # Include intent_type from LLM classification
        }
        if conv.workspace_id:
            body["workspace_id"] = str(conv.workspace_id)

        planning_cfg = (self.agent.config.get("planning", {}) or {})
        timeout = float(planning_cfg.get("timeout", 60))

        logger.info(demo("UA -> InteractionSolver GOAL_REQUEST: intents=%s intent_type=%s"), intent_strings, intent_type.upper())
        logger.info(demo("Sending %d structured_intents to InteractionSolver"), len(conv.intents))
        for si in conv.intents:
            logger.info(demo("  structured_intent: action=%s artifact=%s parameter=%s value=%s"),
                       si.action, si.artifact, si.parameter, si.value)

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
            logger.error("Goal request failed: %s", exc)
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

        # Store plan (deterministic)
        canonical, plan_hash = canonicalize_plan_for_hash(plan_obj)
        conv.plan_json = canonical
        conv.plan_hash = plan_hash

        node_count = count_bt_nodes(tree)
        preview = bt_preview(tree)
        logger.info(demo("Plan stored: hash=%s nodes=%d preview=%s"), plan_hash, node_count, preview)

        # Summarize (LLM NLG call)
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

        if low in _CONFIRM_TOKENS:
            # Execute plan
            conv.phase = ConversationPhase.EXECUTING
            logger.info(demo("User confirmed plan: thread=%s"), thread)
            exec_result = await self._execute_plan(thread, conv)

            if exec_result.success:
                await self._reply(msg, "Done! The plan was executed successfully.")
            else:
                detail = exec_result.error or exec_result.final_status
                await self._reply(msg, f"Execution failed: {detail}")

            conv.clear_plan()
            conv.phase = ConversationPhase.IDLE

        elif low in _REJECT_TOKENS:
            # Discard plan
            logger.info(demo("User rejected plan: thread=%s"), thread)
            conv.clear_plan()
            conv.phase = ConversationPhase.IDLE
            await self._reply(msg, "Okay, I've discarded that plan. What would you like to change?")

        else:
            # New message while awaiting confirmation → discard and re-process
            logger.info(demo("New request while awaiting confirmation, discarding plan: thread=%s"), thread)
            conv.clear_plan()
            conv.phase = ConversationPhase.IDLE

            # Re-process as a new message (fetch capabilities + extract intents)
            conv.user_message = text
            conv.phase = ConversationPhase.EXTRACTING_INTENTS
            capabilities_ctx = await self._fetch_capabilities()
            extraction = await self._extract_intents(text, capabilities_ctx)
            classification = extraction.get("classification", "unclear")

            if classification == "goal":
                await self._handle_goal(msg, thread, conv, extraction)
            elif classification == "query_capabilities":
                await self._handle_query_capabilities(msg, thread, conv, capabilities_ctx)
            elif classification == "query_state":
                await self._handle_query_state(msg, thread, conv, extraction)
            elif classification == "unclear":
                question = extraction.get("question", "Could you clarify what you would like?")
                await self._reply(msg, question)
                conv.phase = ConversationPhase.IDLE
            else:
                await self._reply(msg, "I'm not sure I understood that. Could you rephrase?")
                conv.phase = ConversationPhase.IDLE

    # ------------------------------------------------------------------
    # DETERMINISTIC: plan execution
    # ------------------------------------------------------------------

    async def _execute_plan(self, thread: str, conv: ConversationState) -> ExecutionResult:
        """Execute the stored plan via IRExecutor (deterministic)."""
        plan_obj = coerce_plan_dict(conv.plan_json)
        if not isinstance(plan_obj, dict):
            return ExecutionResult(success=False, error="Invalid plan JSON")

        tree_spec = plan_obj.get("tree", {})
        intents = plan_obj.get("intents", [])
        is_signifier_reuse = plan_obj.get("signifier_reuse", False)
        intent_type = plan_obj.get("intent_type")  # Extract intent_type for signifier recording
        workspace_id = plan_obj.get("workspace_id")  # Extract workspace_id for signifier recording
        structured_intents = plan_obj.get("structured_intents", [])  # Extract structured_intents for artifact matching

        if not tree_spec or not isinstance(tree_spec, dict):
            return ExecutionResult(success=False, error="Plan has no behavior tree to execute")

        # Ensure execution engine is ready
        await self.agent.ensure_execution_engine_ready()

        node_count = count_bt_nodes(tree_spec)
        logger.info(demo("Executing BT: thread=%s nodes=%d signifier_reuse=%s intent_type=%s"), thread, node_count, is_signifier_reuse, intent_type)

        executor = IRExecutor(max_ticks=50)
        loop = asyncio.get_event_loop()
        try:
            exec_result: ExecutionResult = await loop.run_in_executor(
                None,
                executor.execute_from_spec,
                tree_spec,
            )
        except Exception as exc:
            logger.warning(demo("BT execution failed: %s"), exc)
            return ExecutionResult(success=False, error=str(exc))

        logger.info(
            demo("BT execution complete: success=%s ticks=%d status=%s"),
            exec_result.success, exec_result.ticks, exec_result.final_status,
        )

        # Invalidate state cache (environment may have changed)
        self.agent.state_memory.clear()
        logger.info(demo("State memory cache cleared after BT execution"))

        # Record signifiers from successful execution
        if exec_result.success and not is_signifier_reuse:
            await self._record_signifiers(tree_spec, intents, exec_result, thread, intent_type, workspace_id, structured_intents)

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
        structured_intents: Optional[list] = None,
    ) -> None:
        """Extract and record signifiers from an executed BT."""
        # Query environment state for context-rich signifiers
        state_snapshot: Optional[dict] = None
        explorer_jid = self.agent.target_jids.get("explorer")
        if explorer_jid and workspace_id:
            try:
                logger.info(demo("Querying environment state for signifier context (workspace_id=%r)"), workspace_id)
                state_response = await rpc_call(
                    self.agent,
                    to_jid=str(explorer_jid),
                    request_type=MessageType.ENV_STATE_REQUEST.value,
                    body={},  # Empty body = request all state
                    expect_type=MessageType.ENV_STATE_RESPONSE.value,
                    timeout=10.0,
                )

                if state_response and state_response.body:
                    raw_state = json.loads(state_response.body) if isinstance(state_response.body, str) else state_response.body

                    # Build state_snapshot in expected format: {"artifacts": {...}}
                    if isinstance(raw_state, dict):
                        # Unwrap if EnvExplorer returned {"artifacts": {...}} instead of direct dict
                        if "artifacts" in raw_state and isinstance(raw_state["artifacts"], dict):
                            artifacts_dict = raw_state["artifacts"]
                        else:
                            artifacts_dict = raw_state

                        # Filter by workspace_id
                        filtered_artifacts = {}
                        for artifact_id, artifact_info in artifacts_dict.items():
                            if isinstance(artifact_info, dict):
                                artifact_ws = artifact_info.get("workspace_id")
                                # Match workspace_id (exact or substring)
                                if artifact_ws and (str(artifact_ws) == str(workspace_id) or workspace_id in str(artifact_ws)):
                                    filtered_artifacts[artifact_id] = artifact_info

                        state_snapshot = {"artifacts": filtered_artifacts}
                        logger.info(
                            demo("State snapshot retrieved: %d artifacts for workspace_id=%r"),
                            len(filtered_artifacts), workspace_id
                        )
            except Exception as e:
                logger.warning(demo("Failed to query state snapshot: %s (continuing without state)"), e)

        # Extract signifiers with state context AND structured_intents for artifact-based matching
        signifiers = extract_signifiers_from_bt(
            tree_spec=tree_spec,
            intents=intents,
            was_successful=exec_result.success,
            workspace_id=workspace_id,
            state_snapshot=state_snapshot,
            intent_type=intent_type,
            structured_intents=structured_intents,
        )
        if not signifiers:
            logger.info(demo("No signifiers extracted from BT"))
            return

        logger.info(demo("Recording %d signifiers from BT execution (intent_type=%s)"), len(signifiers), intent_type)

        # 1. Record locally via EnvExplorer
        explorer_jid = self.agent.target_jids.get("explorer")
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
                    timeout=15.0,
                    thread=(thread if thread != "__default__" else None),
                )
                created_count = None
                try:
                    rec_payload = json.loads(rec_res.body or "{}")
                    if isinstance(rec_payload, dict):
                        created_count = rec_payload.get("created_count")
                except Exception:
                    pass
                logger.info(demo("Local signifier recording: created_count=%s"), created_count or "?")
            except Exception as exc:
                logger.info(demo("Failed to record signifiers locally: %s"), exc)

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
            logger.info(demo("Community signifier publishing: %d/%d published"), published, len(signifiers))

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

        # Check cache
        state_memory = self.agent.state_memory
        if property_uri and state_memory.has(property_uri):
            cached = state_memory.get(property_uri)
            logger.info(demo("State cache HIT: property_uri=%r value=%r"), property_uri, cached)
            raw_data = json.dumps({"property_uri": property_uri, "value": cached, "source": "cache"})
            formatted = await self._format_query_response(raw_data, conv.user_message)
            await self._reply(msg, formatted)
            conv.phase = ConversationPhase.IDLE
            return

        # RPC to EnvExplorer
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

        logger.info(
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
                timeout=15.0,
            )
            # Cache results
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
                pass  # Best-effort caching

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

    async def _fetch_capabilities(self) -> str:
        """Fetch environment capabilities from EnvExplorer via RPC."""
        explorer_jid = self.agent.target_jids.get("explorer")
        if not explorer_jid:
            return ""
        try:
            result = await rpc_call(
                self.agent,
                to_jid=str(explorer_jid),
                request_type=MessageType.ENV_CAPABILITIES_REQUEST.value,
                body={"query": "all"},
                expect_type=MessageType.ENV_CAPABILITIES_RESPONSE.value,
                timeout=15.0,
            )
            return result.body or ""
        except Exception as exc:
            logger.warning("Failed to fetch capabilities: %s", exc)
            return ""

    async def _reply(self, original_msg, text: str) -> None:
        """Send a reply XMPP message to the user."""
        reply = original_msg.make_reply()
        reply.body = text
        await self.send(reply)


class DemoRequestClassifierBehaviour(CyclicBehaviour):
    """Demo-only logging helper: classify user requests as EXPLICIT vs IMPLICIT.

    Does not influence planning; only emits a readable log line for demos.
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

        kind = "IMPLICIT"
        if low.startswith(("what", "show", "list", "which", "is", "are")) and (
            "workspace" in low or "workspaces" in low or "device" in low
            or "devices" in low or "state" in low
        ):
            kind = "QUERY"
        else:
            has_action = any(
                kw in low
                for kw in (
                    "turn ", "toggle", "open", "close",
                    "set ", "raise", "lower", "increase", "decrease",
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
