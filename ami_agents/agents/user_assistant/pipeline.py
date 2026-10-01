"""
The UA's request pipeline — the LLM and RPC steps, independent of SPADE.

These were methods on `UserMessageBehaviour`, which had grown to own the whole
request lifecycle: receive, segment, parse, plan, summarise, confirm, execute.
Splitting the *steps* from the *orchestration* is what lets the orchestration
become a per-request FSM without dragging 1400 lines along, and it removes the
duplicated pipeline that `run()` and `_handle_confirmation` had each grown a
copy of.

Everything here is a free function taking `agent` and `logger` explicitly, so a
`State`, a `CyclicBehaviour`, or a test can call it without inheriting
anything. Nothing in this module sends, receives, or touches conversation
state — that stays with the behaviours.
"""

from __future__ import annotations

import json

from ...shared.models.messages import MessageType
from ...shared.utils.spade_rpc import rpc_call
from ...shared.utils.demo_log import demo
from ...shared.utils.namespaces import action_types
from .models import AtomicIntent
from .prompts.intent_parsing_prompts import GOAL_REQUEST_PARSER_PROMPT
from .prompts.intent_response_prompts import (
    PLAN_SUMMARY_SYSTEM_PROMPT,
    QUERY_RESPONSE_SYSTEM_PROMPT,
)
from .prompts.intent_segmentation_prompts import ATOMIC_SEGMENTATION_SYSTEM_PROMPT
from .utils import loose_json_loads


async def segment_into_atomic_intents(agent, logger, user_text: str) -> list[AtomicIntent]:
    """Call LLM to segment user message into atomic intents with categories.

    Segmentation is purely linguistic — it takes no environment capabilities.
    Device/action resolution happens in the per-type parser stage.
    """
    logger.info(demo(f"[LLM CALL] Calling ATOMIC_SEGMENTATION_SYSTEM_PROMPT for: {user_text[:100]!r}"))
    messages = [
        {"role": "system", "content": ATOMIC_SEGMENTATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    try:
        response = await agent.llm_client.chat.completions.create(
            model=agent.llm_model,
            messages=messages,
            **agent.build_llm_kwargs(),
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = loose_json_loads(raw)
        if not isinstance(parsed, dict):
            logger.warning("LLM atomic segmentation returned non-dict: %r", raw[:200])
            return []

        intents_data = parsed.get("intents", [])
        if not isinstance(intents_data, list):
            logger.warning("LLM atomic segmentation returned non-list intents")
            return []

        result = []
        for item in intents_data:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "").strip()
            intent_type = item.get("type", "").strip()
            reason = item.get("reason", "").strip()
            # Qualifiers are descriptive only; tolerate a missing or malformed
            # list rather than dropping an otherwise valid intent.
            raw_qualifiers = item.get("qualifiers")
            qualifiers = [
                q.strip() for q in raw_qualifiers
                if isinstance(q, str) and q.strip()
            ] if isinstance(raw_qualifiers, list) else []
            if text and intent_type in ("GOAL_REQUEST", "ENV_STATE_REQUEST", "ENV_CAPABILITIES_REQUEST") and reason:
                result.append(AtomicIntent(
                    text=text, type=intent_type, reason=reason, qualifiers=qualifiers))
        return result
    except Exception as exc:
        logger.error("LLM atomic segmentation failed: %s", exc)
        return []


def build_capabilities_hierarchical_text(caps_summary: dict) -> str:
    """Build human-readable hierarchical text from capabilities summary JSON.

    Filters ACTION affordances to those typed with what they do -- a homeont
    or SAREF command class -- leaving out protocol actions (WebSub, artifact
    CRUD); includes ALL PROPERTY affordances. Delegates to format_capabilities_hierarchical_text()
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

            # Format affordances: ACTION only if typed with what it does
            # (homeont or SAREF), PROPERTY always
            for aff in (artifact.get("affordances") or []):
                aff_type = aff.get("type", "").replace("_affordance", "")

                # Protocol actions (WebSub, artifact CRUD) carry no such type
                if aff_type == "action":
                    if not action_types(aff.get("semantic_types", [])):
                        continue

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


async def parse_atomic_intent(agent, logger, span: str, category: str, capabilities_hierarchical: str) -> dict:
    """Parse a single atomic intent span using LLM per-span prompts.

    Args:
        span: The verbatim user text for this atomic intent
        category: The segmenter's type; only "GOAL_REQUEST" is parsed here
        capabilities_hierarchical: Hierarchical text description of environment capabilities

    Returns:
        Dict with parsed structured intent fields (LLM output), or fallback dict on error
    """
    logger.info(demo(f"[LLM CALL] Parsing {category}: {span[:80]!r}"))

    # Only goals are parsed here. ENV_STATE and ENV_CAPABILITIES questions are
    # structured by their own behaviours, against the ontology context.
    if category != "GOAL_REQUEST":
        return {"text_intent": span}

    prompt = GOAL_REQUEST_PARSER_PROMPT.format(
        capabilities_hierarchical=capabilities_hierarchical,
        ontology=agent.ontology_ttl,
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": span},
    ]

    try:
        response = await agent.llm_client.chat.completions.create(
            model=agent.llm_model,
            messages=messages,
            **agent.build_llm_kwargs(),
        )
        raw = (response.choices[0].message.content or "").strip()
        logger.info(demo("Raw intent extraction output: %s"), raw[:1000])
        parsed = loose_json_loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        logger.warning("LLM per-span parsing failed for [%s]: %s", category, exc)

    # Fallback: minimal dict with only text_intent
    return {"text_intent": span}

# ------------------------------------------------------------------
# NLG: plan summary (LLM call)
# ------------------------------------------------------------------


async def summarize_plan(agent, logger, plan_json: str) -> str:
    """Call LLM to produce a user-friendly plan summary."""
    messages = [
        {"role": "system", "content": PLAN_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": plan_json},
    ]
    try:
        response = await agent.llm_client.chat.completions.create(
            model=agent.llm_model,
            messages=messages,
            **agent.build_llm_kwargs(),
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


async def format_query_response(agent, logger, raw_data: str, user_text: str) -> str:
    """Call LLM to format raw query results for the user."""
    messages = [
        {"role": "system", "content": QUERY_RESPONSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"User asked: {user_text}\n\nRaw data:\n{raw_data}"},
    ]
    try:
        response = await agent.llm_client.chat.completions.create(
            model=agent.llm_model,
            messages=messages,
            **agent.build_llm_kwargs(),
        )
        return (response.choices[0].message.content or "").strip() or raw_data
    except Exception as exc:
        logger.error("LLM query formatting failed: %s", exc)
        return raw_data


# ------------------------------------------------------------------
# DETERMINISTIC: handle goal → request plan → summarize
# ------------------------------------------------------------------


async def fetch_capabilities(agent, logger, detail_level: str = "summary") -> str:
    """Fetch environment capabilities from EnvExplorer via RPC.

    Args:
        detail_level: One of "summary" (default, lightweight hierarchical JSON)
                     or "detailed" (full RDF/Turtle graph).
    """
    explorer_jid = agent.target_jids.get("explorer")
    if not explorer_jid:
        logger.info(demo("Capability fetch skipped: no explorer JID configured"))
        return ""
    try:
        result = await rpc_call(
            agent,
            to_jid=str(explorer_jid),
            request_type=MessageType.ENV_CAPABILITIES_REQUEST.value,
            body={"query": "all", "detail_level": detail_level},
            expect_type=MessageType.ENV_CAPABILITIES_RESPONSE.value,
            timeout=agent.rpc_call_timeout,
        )
        body = result.body or ""
        logger.info(
            demo("Capabilities fetched: body_len=%d"),
            len(body),
        )
        try:
            payload = json.loads(body)
            summary = payload.get("summary") if isinstance(payload, dict) else None
            if isinstance(summary, str) and summary.strip():
                return summary
        except (json.JSONDecodeError, TypeError):
            pass
        return body
    except Exception as exc:
        logger.warning("Failed to fetch capabilities: %s", exc)
        return ""


def collect_action_urls(tree_spec: dict) -> list[str]:
    action_urls: list[str] = []
    seen: set[str] = set()

    def walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "action":
            action_url = str(node.get("action_url") or "").strip()
            if action_url and action_url not in seen:
                seen.add(action_url)
                action_urls.append(action_url)
        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child)

    walk(tree_spec)
    return action_urls

# ------------------------------------------------------------------
# DETERMINISTIC: query handlers
# ------------------------------------------------------------------
