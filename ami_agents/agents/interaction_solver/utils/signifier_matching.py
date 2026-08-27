"""Signifier matching helpers (local EnvExplorer RPC + community API)."""

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ....shared.community.community_client import CommunitySignifierClient
from ....shared.models.messages import MessageType
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger("InteractionSolver")

# A query callable matches the signature of
# ``InteractionSolverAgent._query_env_explorer``: given a message type,
# body dict and expected response type, returns the raw response body.
EnvExplorerQuery = Callable[[str, Dict[str, Any], str], Awaitable[str]]


async def query_local_signifier_match(
    query_env_explorer: EnvExplorerQuery,
    intent: str,
    workspace_id: Optional[str],
    intent_type: Optional[str],
    structured_intent: Optional[Dict[str, Any]],
    affected_env_vars: Optional[List[Dict[str, str]]] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """Send a SIGNIFIER_MATCH_REQUEST to EnvExplorer for one intent.

    For implicit intents, affected_env_vars enables v3 environment-variable
    based matching in the signifier engine.
    """
    logger.info(
        demo(
            "[INTENT_TYPE] Sending SIGNIFIER_MATCH_REQUEST: intent=%r, "
            "workspace_id=%r, intent_type=%r, has_structured_intent=%s, has_env_vars=%s"
        ),
        intent,
        workspace_id,
        intent_type,
        bool(structured_intent),
        bool(affected_env_vars),
    )
    body: Dict[str, Any] = {
        "intent": intent,
        "k": k,
        **({"workspace_id": str(workspace_id)} if workspace_id else {}),
        **({"intent_type": str(intent_type)} if intent_type else {}),
        **({"query_structured_intent": structured_intent} if structured_intent else {}),
        **({"affected_env_vars": affected_env_vars} if affected_env_vars else {}),
    }
    raw = await query_env_explorer(
        MessageType.SIGNIFIER_MATCH_REQUEST.value,
        body,
        MessageType.SIGNIFIER_MATCH_RESPONSE.value,
    )
    try:
        data = json.loads(raw) if raw else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


async def query_community_signifier_match(
    community_client: CommunitySignifierClient,
    intent: str,
) -> Dict[str, Any]:
    """Query the community signifier API for cross-environment matches.

    Tags each match with ``source="community"`` so downstream merging
    can distinguish local vs. community provenance.
    """
    try:
        data = await community_client.match_signifiers(intent)
        if data:
            for m in data.get("matches", []):
                if isinstance(m, dict):
                    m["source"] = "community"
        return data or {}
    except Exception:
        return {}


def merge_signifier_matches(
    local: Dict[str, Any],
    community: Dict[str, Any],
    intents: List[str],
) -> Dict[str, Any]:
    """Merge local and community signifier matches, local-first with dedup.

    Local matches take priority. Community matches supplement gaps.
    Dedup key is ``signifier_id``.

    Handles both new format (exact_matches/affordance_hints) and old format
    (matches/final_matches) for backward compatibility.
    """
    merged: Dict[str, Any] = {}
    for intent in intents:
        local_data = local.get(intent, {}) if isinstance(local.get(intent, {}), dict) else {}
        community_data = community.get(intent, {}) if isinstance(community.get(intent, {}), dict) else {}

        # Support both new (exact_matches/affordance_hints) and old (matches/final_matches) formats
        local_exact = local_data.get("exact_matches") if isinstance(local_data.get("exact_matches"), list) else []
        local_affordance = local_data.get("affordance_hints") if isinstance(local_data.get("affordance_hints"), list) else []
        # Backward compat: old format had matches and final_matches
        if not local_exact and not local_affordance:
            local_exact = local_data.get("final_matches") if isinstance(local_data.get("final_matches"), list) else []
            local_affordance = []

        community_exact = community_data.get("exact_matches") if isinstance(community_data.get("exact_matches"), list) else []
        community_affordance = community_data.get("affordance_hints") if isinstance(community_data.get("affordance_hints"), list) else []
        # Backward compat: old format had matches and final_matches
        if not community_exact and not community_affordance:
            community_exact = community_data.get("final_matches") if isinstance(community_data.get("final_matches"), list) else []
            community_affordance = []

        # Merge exact matches (for fast-path)
        seen_ids: set = set()
        combined_exact: List[Dict[str, Any]] = []
        for m in local_exact:
            if isinstance(m, dict):
                sid = m.get("signifier_id", "")
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    combined_exact.append(m)
        for m in community_exact:
            if isinstance(m, dict):
                sid = m.get("signifier_id", "")
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    combined_exact.append(m)

        # Merge affordance hints (for BT planner)
        combined_affordance: List[Dict[str, Any]] = []
        seen_ids_affordance: set = set()
        for m in local_affordance:
            if isinstance(m, dict):
                sid = m.get("signifier_id", "")
                if sid not in seen_ids_affordance:
                    seen_ids_affordance.add(sid)
                    combined_affordance.append(m)
        for m in community_affordance:
            if isinstance(m, dict):
                sid = m.get("signifier_id", "")
                if sid not in seen_ids_affordance and sid not in seen_ids:
                    seen_ids_affordance.add(sid)
                    combined_affordance.append(m)

        merged[intent] = {
            "exact_matches": combined_exact,
            "affordance_hints": combined_affordance,
        }
        if local_data.get("error"):
            merged[intent]["error"] = local_data["error"]

    return merged
