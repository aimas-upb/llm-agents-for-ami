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
import os
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
    ENV_CAPABILITIES_RESPONSE_PROMPT,
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
from ..utils.llm_client import build_behaviour_llm_client, build_llm_call_kwargs
from ....shared.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger("UserAssistant")


# ============================================================================
# Context builder helpers for per-span LLM prompt template variables
# ============================================================================


def _filter_capabilities_json(caps_dict: dict) -> dict:
    """Filter capabilities JSON to reduce size for ENV_CAPABILITIES_RESPONSE_PROMPT.

    Rules:
    1. Omit 'description' if empty
    2. Omit 'form' from action/property affordances
    3. Exclude action affordances without ex: semantic types
    4. Omit output_schema from property affordances
    5. Omit input_schema from action affordances
    6. Omit parameters if empty

    Args:
        caps_dict: Hierarchical capabilities dict from EnvExplorer

    Returns:
        Filtered dict with reduced size
    """
    filtered = {}

    # Copy top-level fields
    for key in ("discovery_complete",):
        if key in caps_dict:
            filtered[key] = caps_dict[key]

    # Filter workspaces
    filtered["workspaces"] = []
    for ws in (caps_dict.get("workspaces") or []):
        filtered_ws = {
            k: v for k, v in ws.items()
            if k not in ("form",) and not (k == "description" and not v)
        }

        # Filter artifacts
        filtered_ws["artifacts"] = []
        for artifact in (ws.get("artifacts") or []):
            filtered_artifact = {
                k: v for k, v in artifact.items()
                if k not in ("form",) and not (k == "description" and not v)
            }

            # Filter affordances
            filtered_artifact["affordances"] = []
            for aff in (artifact.get("affordances") or []):
                aff_type = aff.get("type", "")

                # Rule 3: For action affordances, keep ONLY those with ex: namespace semantic types
                if aff_type == "action_affordance":
                    semantic_types = aff.get("semantic_types", [])
                    # Keep only ex: types (user-facing Home Assistant actions)
                    ex_types = [st for st in semantic_types if st.startswith("ex:")]
                    logger.debug(
                        f"Action affordance {aff.get('name')}: semantic_types={semantic_types}, ex_types={ex_types}"
                    )
                    if not ex_types:
                        logger.debug(
                            f"Filtering out action affordance {aff.get('name')} (no ex: semantic types)"
                        )
                        continue

                # Build filtered affordance
                filtered_aff = {}

                # Copy all fields except form, schemas, parameters, and empty descriptions
                for key, val in aff.items():
                    if key == "form":
                        continue  # Rule 2
                    elif key == "output_schema" and aff_type == "property_affordance":
                        continue  # Rule 4
                    elif key == "input_schema" and aff_type == "action_affordance":
                        continue  # Rule 5
                    elif key == "parameters":
                        # Rule 6: Omit if empty
                        if val and len(val) > 0:
                            filtered_aff[key] = val
                    elif key == "description" and not val:
                        # Rule 1: Omit if empty
                        continue
                    else:
                        filtered_aff[key] = val

                filtered_artifact["affordances"].append(filtered_aff)

            filtered_ws["artifacts"].append(filtered_artifact)

        filtered["workspaces"].append(filtered_ws)

    return filtered


def _filter_capabilities_json_for_state(caps_dict: dict) -> dict:
    """Filter capabilities JSON for ENV_STATE_REQUEST parsing.

    Rules:
    1. Include ONLY property affordances (exclude all actions)
    2. Omit 'form', 'output_schema'
    3. Include 'description' only if non-empty
    4. Omit 'parameters' if empty

    Purpose: Minimal property list for state-query parser to align property names to artifact/property URIs.

    Args:
        caps_dict: Hierarchical capabilities dict from EnvExplorer

    Returns:
        Filtered dict containing only properties (no actions, no schemas/forms)
    """
    filtered = {}

    # Copy top-level fields
    for key in ("discovery_complete",):
        if key in caps_dict:
            filtered[key] = caps_dict[key]

    # Filter workspaces
    filtered["workspaces"] = []
    for ws in (caps_dict.get("workspaces") or []):
        filtered_ws = {
            k: v for k, v in ws.items()
            if k != "form" and not (k == "description" and not v)
        }

        # Filter artifacts
        filtered_ws["artifacts"] = []
        for artifact in (ws.get("artifacts") or []):
            filtered_artifact = {
                k: v for k, v in artifact.items()
                if k != "form" and not (k == "description" and not v)
            }

            # Filter affordances: ONLY property affordances (no actions)
            filtered_artifact["affordances"] = []
            for aff in (artifact.get("affordances") or []):
                aff_type = aff.get("type", "")

                # Skip action affordances entirely
                if aff_type != "property_affordance":
                    continue

                # Build filtered property affordance
                filtered_aff = {}

                # Copy fields except: form, output_schema, and empty description
                for key, val in aff.items():
                    if key == "form":
                        continue  # Omit form
                    elif key == "output_schema":
                        continue  # Omit output schema
                    elif key == "description" and not val:
                        continue  # Omit empty description
                    elif key == "parameters":
                        # Only include if non-empty
                        if val and len(val) > 0:
                            filtered_aff[key] = val
                    else:
                        filtered_aff[key] = val

                filtered_artifact["affordances"].append(filtered_aff)

            filtered_ws["artifacts"].append(filtered_artifact)

        filtered["workspaces"].append(filtered_ws)

    return filtered



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

        self.logger.debug(demo(f"[UserMessageBehaviour.run] Received message: {msg.body[:100] if msg.body else '(empty)'}"))

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

        # Fetch capabilities in lightweight format for atomic segmentation and classification
        capabilities_ctx = await self._fetch_capabilities(detail_level="summary")

        # Parse capabilities JSON for segmentation (lightweight summary format)
        try:
            caps_summary = json.loads(capabilities_ctx) if capabilities_ctx else {}
        except (json.JSONDecodeError, AttributeError):
            caps_summary = {}

        # Filter capabilities for atomic segmentation (same filtering as for ENV_STATE_REQUEST parsing)
        filtered_caps_for_segmentation = _filter_capabilities_json(caps_summary)

        # Segment the user text into atomic intents
        atomic_intents = await self._segment_into_atomic_intents(text, json.dumps(filtered_caps_for_segmentation))
        if not atomic_intents:
            await self._reply(msg, "I couldn't understand your request. Could you rephrase?")
            conv.phase = ConversationPhase.IDLE
            return

        # Log atomic segmentation results
        intent_summary = {}
        for ai in atomic_intents:
            intent_summary[ai.category] = intent_summary.get(ai.category, 0) + 1
        self.logger.info(demo(f"[SEGMENTATION] {len(atomic_intents)} atomic intent(s): {intent_summary}"))
        for ai in atomic_intents:
            self.logger.info(demo(f"[SEGMENTATION] - {ai.category}: span={ai.span!r} reason={ai.reason}"))

        # Group intents by category
        caps_intents = [a for a in atomic_intents if a.category == "ENV_CAPABILITIES_REQUEST"]
        state_intents = [a for a in atomic_intents if a.category == "ENV_STATE_REQUEST"]
        goal_intents = [a for a in atomic_intents if a.category == "GOAL_REQUEST"]

        # Fetch hierarchical capabilities text and JSON for intent parsing phases
        capabilities_hierarchical = ""
        capabilities_hierarchical_state = ""
        if caps_intents or state_intents or goal_intents:
            # Build hierarchical text for ENV_CAPABILITIES_REQUEST and GOAL_REQUEST
            capabilities_hierarchical = self._build_capabilities_hierarchical_text(caps_summary)

            # Build filtered JSON for ENV_STATE_REQUEST (properties only)
            if state_intents:
                filtered_state_caps = _filter_capabilities_json_for_state(caps_summary)
                capabilities_hierarchical_state = json.dumps(filtered_state_caps, indent=2)

            import os
            if os.getenv("VERBOSE_LOGGING"):
                self.logger.info(demo(f"=== HIERARCHICAL CAPABILITIES CONTEXT ===\n{capabilities_hierarchical}"))

        # Phase 1: ENV_CAPABILITIES_REQUEST (if any)
        if caps_intents:
            conv.phase = ConversationPhase.EXTRACTING_INTENTS
            for intent in caps_intents:
                extraction = await self._parse_atomic_intent(intent.span, intent.category, capabilities_hierarchical)
                # Log parsed ENV_CAPABILITIES_REQUEST extraction with yellow label
                no_color = bool(os.getenv("AMI_NO_COLOR")) or bool(os.getenv("NO_COLOR"))
                if no_color:
                    label = "[ENV_CAPABILITIES_REQUEST]"
                else:
                    label = "\x1b[1;33m[ENV_CAPABILITIES_REQUEST]\x1b[0m"
                self.logger.info(demo(f"{label}:\n{json.dumps(extraction, indent=2)}"))
                await self._handle_query_capabilities(msg, thread, conv, capabilities_ctx, extraction)

        # Phase 2: ENV_STATE_REQUEST (if any)
        if state_intents:
            conv.phase = ConversationPhase.EXTRACTING_INTENTS
            for intent in state_intents:
                extraction = await self._parse_atomic_intent(intent.span, intent.category, capabilities_hierarchical_state)
                # Log parsed ENV_STATE_REQUEST extraction with yellow label
                no_color = bool(os.getenv("AMI_NO_COLOR")) or bool(os.getenv("NO_COLOR"))
                if no_color:
                    label = "[ENV_STATE_REQUEST]"
                else:
                    label = "\x1b[1;33m[ENV_STATE_REQUEST]\x1b[0m"
                self.logger.info(demo(f"{label}:\n{json.dumps(extraction, indent=2)}"))
                await self._handle_query_state(msg, thread, conv, extraction)

        # Phase 3: GOAL_REQUEST (if any) — batch all goal intents into a single request
        if goal_intents:
            conv.phase = ConversationPhase.EXTRACTING_INTENTS
            # Parse all goal intent spans in parallel (using the same capabilities_hierarchical fetched above)
            extractions = await asyncio.gather(
                *[self._parse_atomic_intent(intent.span, intent.category, capabilities_hierarchical) for intent in goal_intents],
                return_exceptions=True,
            )
            # Convert to typed intent objects, filtering errors
            parsed_intents = []
            no_color = bool(os.getenv("AMI_NO_COLOR")) or bool(os.getenv("NO_COLOR"))
            if no_color:
                label = "[GOAL_REQUEST]"
            else:
                label = "\x1b[1;33m[GOAL_REQUEST]\x1b[0m"
            for extraction in extractions:
                if isinstance(extraction, Exception):
                    self.logger.error("Failed to parse atomic intent: %s", extraction)
                    continue
                if isinstance(extraction, dict):
                    # Log parsed GOAL_REQUEST extraction with yellow label
                    self.logger.info(demo(f"{label}:\n{json.dumps(extraction, indent=2)}"))
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
        self.logger.info(demo(f"[LLM CALL] Calling ATOMIC_SEGMENTATION_SYSTEM_PROMPT for: {user_text[:100]!r}"))
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

    def _build_capabilities_hierarchical_text(self, caps_summary: dict) -> str:
        """Build human-readable hierarchical text from capabilities summary JSON.

        Filters ACTION affordances to only show those with ex: semantic types,
        but includes ALL PROPERTY affordances. Delegates to format_capabilities_hierarchical_text()
        from env_explorer's data_formatting module, which handles the full formatting.

        Args:
            caps_summary: Hierarchical dict from format_capabilities_summary_hierarchical()

        Returns:
            Human-readable hierarchical string
        """
        if not caps_summary or not caps_summary.get("discovery_complete"):
            return "Environment discovery not yet complete."

        lines = []

        def format_workspace(ws_node: dict, indent: str = "") -> None:
            """Recursively format a workspace and its contents."""
            ws_name = ws_node.get("name", "Unknown")
            lines.append(f"{indent}Workspace: {ws_name}")
            semantic_types = ws_node.get("semantic_types", [])
            if semantic_types:
                lines.append(f"{indent}  semantic_types: {', '.join(semantic_types)}")
            description = ws_node.get("description", "")
            if description:
                lines.append(f"{indent}  description: {description}")

            # Format artifacts in this workspace
            for artifact in (ws_node.get("artifacts") or []):
                artifact_name = artifact.get("name", "Unknown")
                lines.append(f"{indent}  Artifact: {artifact_name}")
                artifact_types = artifact.get("semantic_types", [])
                if artifact_types:
                    lines.append(f"{indent}    semantic_types: {', '.join(artifact_types)}")
                artifact_desc = artifact.get("description", "")
                if artifact_desc:
                    lines.append(f"{indent}    description: {artifact_desc}")

                # Format affordances: ACTION only if has ex: types, PROPERTY always
                for aff in (artifact.get("affordances") or []):
                    aff_type = aff.get("type", "").replace("_affordance", "")

                    # For ACTION affordances, only include if they have ex: semantic types
                    if aff_type == "action":
                        semantic_types = aff.get("semantic_types", [])
                        homeont_types = [st for st in semantic_types if isinstance(st, str) and st.startswith("ex:")]
                        if not homeont_types:
                            continue  # Skip actions without homeont types

                    aff_name = aff.get("name", "Unknown")
                    lines.append(f"{indent}    {aff_type}: {aff_name}")

                    # Include semantic types if present
                    aff_semantic_types = aff.get("semantic_types", [])
                    if aff_semantic_types:
                        lines.append(f"{indent}      semantic_types: {', '.join(aff_semantic_types)}")

                    # Include description if present
                    aff_desc = aff.get("description", "")
                    if aff_desc:
                        lines.append(f"{indent}      description: {aff_desc}")

                    # Include parameters if present
                    parameters = aff.get("parameters", [])
                    if parameters:
                        lines.append(f"{indent}      parameters: {', '.join(parameters)}")

            # Format sub-workspaces
            for sub_ws in (ws_node.get("sub_workspaces") or []):
                format_workspace(sub_ws, indent + "  ")

        # Process all root workspaces
        for ws in (caps_summary.get("workspaces") or []):
            format_workspace(ws)

        return "\n".join(lines) if lines else "No environment capabilities found."

    async def _parse_atomic_intent(
        self, span: str, category: str, capabilities_hierarchical: str
    ) -> dict:
        """Parse a single atomic intent span using LLM per-span prompts.

        Args:
            span: The verbatim user text for this atomic intent
            category: One of "GOAL_REQUEST", "ENV_STATE_REQUEST", "ENV_CAPABILITIES_REQUEST"
            capabilities_hierarchical: Hierarchical text description of environment capabilities

        Returns:
            Dict with parsed structured intent fields (LLM output), or fallback dict on error
        """
        self.logger.info(demo(f"[LLM CALL] Parsing {category}: {span[:80]!r}"))

        # Choose prompt and template variables by category
        ontology = self.agent.ontology_ttl

        if category == "GOAL_REQUEST":
            prompt_template = GOAL_REQUEST_PARSER_PROMPT
            prompt = prompt_template.format(
                capabilities_hierarchical=capabilities_hierarchical,
                ontology=ontology,
            )
        elif category == "ENV_STATE_REQUEST":
            prompt_template = ENV_STATE_REQUEST_PARSER_PROMPT
            prompt = prompt_template.format(
                capabilities_hierarchical=capabilities_hierarchical,
                ontology=ontology,
            )
        elif category == "ENV_CAPABILITIES_REQUEST":
            prompt_template = ENV_CAPABILITIES_REQUEST_PARSER_PROMPT
            prompt = prompt_template.format(
                capabilities_hierarchical=capabilities_hierarchical,
                ontology=ontology,
            )
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

    async def _format_capabilities_response(self, capabilities_ctx: str, extraction: dict) -> str:
        """Call LLM to format capabilities response based on structured extraction."""
        prompt = ENV_CAPABILITIES_RESPONSE_PROMPT.format(capabilities_hierarchical=capabilities_ctx)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(extraction)},
        ]
        try:
            # Build LLM client with capabilities_analysis config
            llm_cfg = build_behaviour_llm_client(self.agent.config, "capabilities_analysis")
            kwargs = build_llm_call_kwargs(llm_cfg)

            response = await llm_cfg.client.chat.completions.create(
                model=llm_cfg.model,
                messages=messages,
                **kwargs,
            )
            return (response.choices[0].message.content or "").strip() or "Unable to process capabilities query."
        except Exception as exc:
            self.logger.error("LLM capabilities formatting failed: %s", exc)
            return "Unable to process capabilities query."

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

        # Log the intents being sent with full structure
        for i, intent in enumerate(parsed_intents):
            import json
            self.logger.info(
                demo(f"Parsed goal intent #{i + 1}: {json.dumps(intent.to_wire_dict(), indent=2)}")
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

        # Extract intent categories from parsed intents
        intent_categories = [
            parsed_intents[i].to_wire_dict().get("category", "unknown")
            for i in range(len(parsed_intents))
        ]

        self.logger.info(
            demo(f"UA -> InteractionSolver GOAL_REQUEST: {len(parsed_intents)} intents, categories: {intent_categories}")
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
        """Store plan(s), summarise via LLM, and ask for confirmation.

        Handles both single-plan (legacy) and multi-plan (new per-intent) responses.
        """
        plan_obj = coerce_plan_dict(plan_body)
        if not isinstance(plan_obj, dict):
            await self._reply(msg, "Received an invalid plan. Could you try rephrasing your request?")
            conv.phase = ConversationPhase.IDLE
            return

        # Check for error in the response
        if plan_obj.get("error"):
            error = plan_obj.get("error", "unknown")
            detail = plan_obj.get("detail") or plan_obj.get("explanation", "")
            await self._reply(msg, f"Planning failed: {error}. {detail}\nWhat would you like to change?")
            conv.phase = ConversationPhase.IDLE
            return

        # Handle multi-plan response (new format with "plans" list)
        if "plans" in plan_obj:
            plans_list = plan_obj.get("plans", [])
            if not plans_list:
                await self._reply(msg, "Planning returned no plans. Could you try rephrasing your request?")
                conv.phase = ConversationPhase.IDLE
                return

            # Store the multi-plan response for async execution
            canonical, plan_hash = canonicalize_plan_for_hash(plan_obj)
            conv.plan_json = canonical
            conv.plan_hash = plan_hash
            conv.plan_count = len(plans_list)

            total_nodes = sum(count_bt_nodes(p.get("tree", {})) for p in plans_list)
            self.logger.info(
                demo(f"Multi-plan stored: hash={plan_hash} plans={len(plans_list)} total_nodes={total_nodes}")
            )

        # Handle single-plan response (legacy format with "tree" at top level)
        else:
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
            self.logger.info(demo(f"Plan stored: hash={plan_hash} nodes={node_count} preview={preview}"))

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
            self.logger.info(demo(f"User confirmed plan: thread={thread}"))
            exec_result = await self._execute_plan(thread, conv)

            if exec_result.success:
                await self._reply(msg, "Done! The plan was executed successfully.")
            else:
                detail = exec_result.error or exec_result.final_status
                await self._reply(msg, f"Execution failed: {detail}")

            conv.clear_plan()
            conv.phase = ConversationPhase.IDLE

        elif low in REJECT_TOKENS:
            self.logger.info(demo(f"User rejected plan: thread={thread}"))
            conv.clear_plan()
            conv.phase = ConversationPhase.IDLE
            await self._reply(msg, "Okay, I've discarded that plan. What would you like to change?")

        else:
            self.logger.info(demo(f"New request while awaiting confirmation, discarding plan: thread={thread}"))
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

            # Fetch hierarchical capabilities text for all intent parsing phases
            capabilities_hierarchical = ""
            if caps_intents or state_intents or goal_intents:
                try:
                    caps_summary = json.loads(capabilities_ctx) if capabilities_ctx else {}
                except (json.JSONDecodeError, AttributeError):
                    caps_summary = {}
                capabilities_hierarchical = self._build_capabilities_hierarchical_text(caps_summary)

            # Phase 1: ENV_CAPABILITIES_REQUEST (if any)
            if caps_intents:
                conv.phase = ConversationPhase.EXTRACTING_INTENTS
                for intent in caps_intents:
                    extraction = await self._parse_atomic_intent(intent.span, intent.category, capabilities_hierarchical)
                    # Log parsed ENV_CAPABILITIES_REQUEST extraction with yellow label
                    no_color = bool(os.getenv("AMI_NO_COLOR")) or bool(os.getenv("NO_COLOR"))
                    if no_color:
                        label = "[ENV_CAPABILITIES_REQUEST]"
                    else:
                        label = "\x1b[1;33m[ENV_CAPABILITIES_REQUEST]\x1b[0m"
                    self.logger.info(demo(f"{label}:\n{json.dumps(extraction, indent=2)}"))
                    await self._handle_query_capabilities(msg, thread, conv, capabilities_ctx, extraction)

            # Phase 2: ENV_STATE_REQUEST (if any)
            if state_intents:
                conv.phase = ConversationPhase.EXTRACTING_INTENTS
                for intent in state_intents:
                    extraction = await self._parse_atomic_intent(intent.span, intent.category, capabilities_hierarchical)
                    # Log parsed ENV_STATE_REQUEST extraction with yellow label
                    no_color = bool(os.getenv("AMI_NO_COLOR")) or bool(os.getenv("NO_COLOR"))
                    if no_color:
                        label = "[ENV_STATE_REQUEST]"
                    else:
                        label = "\x1b[1;33m[ENV_STATE_REQUEST]\x1b[0m"
                    self.logger.info(demo(f"{label}:\n{json.dumps(extraction, indent=2)}"))
                    await self._handle_query_state(msg, thread, conv, extraction)

            # Phase 3: GOAL_REQUEST (if any)
            if goal_intents:
                conv.phase = ConversationPhase.EXTRACTING_INTENTS
                extractions = await asyncio.gather(
                    *[self._parse_atomic_intent(intent.span, intent.category, capabilities_hierarchical) for intent in goal_intents],
                    return_exceptions=True,
                )
                parsed_intents = []
                no_color = bool(os.getenv("AMI_NO_COLOR")) or bool(os.getenv("NO_COLOR"))
                if no_color:
                    label = "[GOAL_REQUEST]"
                else:
                    label = "\x1b[1;33m[GOAL_REQUEST]\x1b[0m"
                for extraction in extractions:
                    if isinstance(extraction, Exception):
                        self.logger.error("Failed to parse atomic intent: %s", extraction)
                        continue
                    if isinstance(extraction, dict):
                        # Log parsed GOAL_REQUEST extraction with yellow label
                        self.logger.info(demo(f"{label}:\n{json.dumps(extraction, indent=2)}"))
                        parsed = goal_intent_from_dict(extraction)
                        parsed_intents.append(parsed)

                if parsed_intents:
                    await self._handle_goal(msg, thread, conv, parsed_intents)

    # ------------------------------------------------------------------
    # DETERMINISTIC: plan execution
    # ------------------------------------------------------------------

    async def _execute_plan(self, thread: str, conv: ConversationState) -> ExecutionResult:
        """Execute the stored plan(s) via IRExecutor.

        Handles both single-plan (legacy) and multi-plan (new per-intent) responses.
        Multi-plan execution happens in parallel (asyncio.gather).
        """
        plan_obj = coerce_plan_dict(conv.plan_json)
        if not isinstance(plan_obj, dict):
            return ExecutionResult(success=False, error="Invalid plan JSON")

        # Handle multi-plan response (new format with "plans" list)
        if "plans" in plan_obj:
            plans_list = plan_obj.get("plans", [])
            if not plans_list:
                return ExecutionResult(success=False, error="No plans to execute")

            await self.agent.ensure_execution_engine_ready()

            # Execute all plans in parallel
            executor = IRExecutor(max_ticks=self.agent.bt_max_ticks)
            loop = asyncio.get_event_loop()

            execution_tasks = []
            for i, plan_entry in enumerate(plans_list):
                tree_spec = plan_entry.get("tree", {})
                intent_data = plan_entry.get("intent", {})
                explanation = plan_entry.get("explanation", "")

                if not tree_spec or not isinstance(tree_spec, dict):
                    self.logger.warning(demo(f"Plan {i} has no tree"))
                    continue

                node_count = count_bt_nodes(tree_spec)
                intent_text = (
                    intent_data.get("text_intent")
                    if isinstance(intent_data, dict)
                    else str(intent_data)
                )
                self.logger.info(
                    demo(f"Executing plan {i + 1}/{len(plans_list)}: intent={intent_text!r} nodes={node_count}")
                )

                # Schedule execution as async task
                task = loop.run_in_executor(None, executor.execute_from_spec, tree_spec)
                execution_tasks.append((i, intent_data, task))

            # Await all executions in parallel
            results = []
            for i, intent_data, task in execution_tasks:
                try:
                    exec_result = await task
                    results.append((i, intent_data, exec_result))
                    self.logger.info(
                        demo(f"Plan {i + 1} execution complete: success={exec_result.success} ticks={exec_result.ticks} status={exec_result.final_status}")
                    )
                except Exception as exc:
                    self.logger.warning(demo(f"Plan {i + 1} execution failed: {exc}"))
                    results.append((i, intent_data, ExecutionResult(success=False, error=str(exc))))

            self.agent.state_memory.clear()
            self.logger.info(demo(f"State memory cache cleared after multi-plan execution"))

            # Check overall success: all plans must succeed
            all_success = all(r[2].success for r in results)

            # Record signifiers for successful plans
            workspace_id = plan_obj.get("workspace_id")
            for i, intent_data, exec_result in results:
                if exec_result.success:
                    tree_spec = plans_list[i].get("tree", {})
                    # Reconstruct intent from wire format
                    if isinstance(intent_data, dict) and intent_data.get("category") in ("implicit", "explicit"):
                        intent_obj = goal_intent_from_dict(intent_data)
                    else:
                        intent_obj = Intent(intent_text=intent_data.get("text_intent", str(intent_data)))

                    # Record signifiers for both implicit and explicit intents
                    if isinstance(intent_obj, ImplicitGoalIntent):
                        await self._record_signifiers(
                            tree_spec, [intent_obj], exec_result, thread,
                            intent_type="IMPLICIT", workspace_id=workspace_id
                        )
                    elif isinstance(intent_obj, ExplicitGoalIntent):
                        # EXPLICIT intents produce signifiers but without context
                        await self._record_signifiers(
                            tree_spec, [intent_obj], exec_result, thread,
                            intent_type="EXPLICIT", workspace_id=workspace_id
                        )

            return ExecutionResult(
                success=all_success,
                final_status=f"Multi-plan execution: {sum(r[2].success for r in results)}/{len(results)} succeeded",
                ticks=sum(r[2].ticks for r in results),
            )

        # Handle single-plan response (legacy format with "tree" at top level)
        else:
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
            self.logger.info(demo(f"Executing BT: thread={thread} nodes={node_count} signifier_reuse={is_signifier_reuse} intent_type={intent_type}"))

            executor = IRExecutor(max_ticks=self.agent.bt_max_ticks)
            loop = asyncio.get_event_loop()
            try:
                exec_result: ExecutionResult = await loop.run_in_executor(
                    None,
                    executor.execute_from_spec,
                    tree_spec,
                )
            except Exception as exc:
                self.logger.warning(demo(f"BT execution failed: {exc}"))
                return ExecutionResult(success=False, error=str(exc))

            self.logger.info(
                demo(f"BT execution complete: success={exec_result.success} ticks={exec_result.ticks} status={exec_result.final_status}")
            )

            self.agent.state_memory.clear()
            self.logger.info(demo(f"State memory cache cleared after BT execution"))

            # Record signifiers for implicit and explicit goal intents (but not for signifier reuse)
            if exec_result.success and not is_signifier_reuse:
                intent_type_upper = str(intent_type).upper() if intent_type else ""
                if intent_type_upper in ("IMPLICIT", "EXPLICIT"):
                    await self._record_signifiers(
                        tree_spec, intents, exec_result, thread, intent_type_upper, workspace_id
                    )

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

        For IMPLICIT intents: includes state context for SHACL validation.
        For EXPLICIT intents: excludes state context (user specified exact target).
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

        # Only fetch state context for IMPLICIT intents
        if intent_type and "IMPLICIT" in str(intent_type).upper():
            if explorer_jid and workspace_id:
                try:
                    self.logger.info(demo(f"Querying environment state for signifier context (workspace_id={workspace_id!r})"))

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
                                demo(f"State snapshot retrieved: {len(filtered_artifacts)} artifacts for workspace_id={workspace_id!r}")
                            )
                except Exception as e:
                    self.logger.warning(demo(f"Failed to query state snapshot: {e} (continuing without state)"))
        else:
            # EXPLICIT intents don't include state context
            if intent_type and "EXPLICIT" in str(intent_type).upper():
                self.logger.info(demo(f"Skipping state context for EXPLICIT intent (user specified exact target)"))

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
            self.logger.info(demo(f"No signifiers extracted from BT"))
            return

        self.logger.info(demo(f"Recording {len(signifiers)} signifiers from BT execution (intent_type={intent_type})"))

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
                self.logger.info(demo(f"Local signifier recording: created_count={created_count or '?'}"))
            except Exception as exc:
                self.logger.info(demo(f"Failed to record signifiers locally: {exc}"))

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
            self.logger.info(demo(f"Community signifier publishing: {published}/{len(signifiers)} published"))

    # ------------------------------------------------------------------
    # DETERMINISTIC: query handlers
    # ------------------------------------------------------------------

    async def _handle_query_capabilities(
        self, msg, thread: str, conv: ConversationState, capabilities_ctx: str, extraction: dict
    ) -> None:
        """Handle a capabilities query (structured + LLM formatting)."""
        if not capabilities_ctx:
            capabilities_ctx = await self._fetch_capabilities()

        if not capabilities_ctx:
            await self._reply(msg, "Unable to fetch environment capabilities right now.")
            conv.phase = ConversationPhase.IDLE
            return

        # Parse and filter capabilities JSON to reduce size
        try:
            caps_dict = json.loads(capabilities_ctx) if isinstance(capabilities_ctx, str) else capabilities_ctx
            ## log caps_dict with indentation for debugging
            # self.logger.info(demo(f"Raw capabilities JSON: {json.dumps(caps_dict, indent=2)}"))
            
            filtered_caps = _filter_capabilities_json(caps_dict)
            filtered_json = json.dumps(filtered_caps, indent=2)

            # self.logger.info(demo(f"Filtered capabilities JSON: {filtered_json}"))
        except Exception as e:
            self.logger.warning(f"Failed to filter capabilities JSON: {e}, using unfiltered")
            filtered_json = capabilities_ctx

        formatted = await self._format_capabilities_response(filtered_json, extraction)
        await self._reply(msg, formatted)
        conv.phase = ConversationPhase.IDLE

    async def _handle_query_state(
        self, msg, thread: str, conv: ConversationState, extraction: dict
    ) -> None:
        """Handle a state query (deterministic fetch with cache + LLM formatting).

        The extraction dict contains LLM-parsed fields: artifact_type, workspace_type,
        artifact_name, property_name, parameter_name. We send these to EnvExplorer,
        which resolves names to artifact_id and property_uri, fetches state, and returns data.
        """
        explorer_jid = self.agent.target_jids.get("explorer")
        if not explorer_jid:
            await self._reply(msg, "Error: EnvExplorer is not configured.")
            conv.phase = ConversationPhase.IDLE
            return

        # Send entire extraction to EnvExplorer for name→ID resolution
        payload = extraction

        self.logger.info(
            demo(f"UA -> EnvExplorer ENV_STATE_REQUEST: extraction={json.dumps(extraction)}")
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

            # Log the raw state response before formatting
            self.logger.info(demo(f"ENV_STATE_RESPONSE received: {result.body}"))

            # Format the raw state response for user-friendly output
            # Use the extracted text_intent from the parsed intent, not the full user message
            text_intent = extraction.get("text_intent", conv.user_message)
            formatted = await self._format_query_response(result.body, text_intent)
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
