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
from .models import AtomicIntent
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


async def fetch_goal_context(agent, logger, goal_text: str,
                             with_properties: bool = False,
                             with_environment: bool = False) -> str:
    """Fetch the goal context for one goal from EnvExplorer via RPC.

    EnvExplorer scopes it to the rooms or device families `goal_text` names
    once the home is too large to send whole; the flags add what devices
    report and each room's environment variables. Returns "" when the context
    cannot be had -- there is no other environment description to fall back on.
    """
    explorer_jid = agent.target_jids.get("explorer")
    if not explorer_jid:
        logger.info(demo("Goal context fetch skipped: no explorer JID configured"))
        return ""
    try:
        result = await rpc_call(
            agent,
            to_jid=str(explorer_jid),
            request_type=MessageType.ENV_CAPABILITIES_REQUEST.value,
            body={"detail_level": "goal_context", "goal_text": goal_text,
                  "with_properties": with_properties,
                  "with_environment": with_environment},
            expect_type=MessageType.ENV_CAPABILITIES_RESPONSE.value,
            timeout=agent.rpc_call_timeout,
        )
        payload = json.loads(result.body or "{}")
    except Exception as exc:
        logger.warning("Failed to fetch goal context: %s", exc)
        return ""

    if not isinstance(payload, dict) or payload.get("error"):
        logger.warning("Goal context unavailable: %s",
                       payload.get("detail") if isinstance(payload, dict) else payload)
        return ""
    scope = payload.get("scope") or {}
    text = payload.get("text") or ""
    logger.info(demo(
        f"Goal context fetched (properties={with_properties}, "
        f"environment={with_environment}): rule={scope.get('rule')} "
        f"workspaces={scope.get('workspaces', [])} chars={len(text)}"))
    return text


async def resolve_candidates(agent, logger, payload: dict) -> dict:
    """Ask EnvExplorer what in this home fits a partly named goal or condition.

    The same ENV_CAPABILITY_QUERY the capability questions use: classes in,
    matching devices and affordances out, nothing read. Returns the response,
    or {} if it cannot be had -- the caller then leaves the goal as it is.
    """
    explorer_jid = agent.target_jids.get("explorer")
    if not explorer_jid:
        return {}
    try:
        result = await rpc_call(
            agent,
            to_jid=str(explorer_jid),
            request_type=MessageType.ENV_CAPABILITY_QUERY_REQUEST.value,
            body=payload,
            expect_type=MessageType.ENV_CAPABILITY_QUERY_RESPONSE.value,
            timeout=agent.rpc_call_timeout,
        )
        response = json.loads(result.body or "{}")
    except Exception as exc:
        logger.warning("Candidate resolution failed for %r: %s",
                       payload.get("text_intent"), exc)
        return {}
    return response if isinstance(response, dict) else {}


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
