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
    k: int = 5,
) -> Dict[str, Any]:
    """Send a SIGNIFIER_MATCH_REQUEST to EnvExplorer for one intent."""
    logger.info(
        demo(
            "[INTENT_TYPE] Sending SIGNIFIER_MATCH_REQUEST: intent=%r, "
            "workspace_id=%r, intent_type=%r, has_structured_intent=%s"
        ),
        intent,
        workspace_id,
        intent_type,
        bool(structured_intent),
    )
    body: Dict[str, Any] = {
        "intent": intent,
        "k": k,
        **({"workspace_id": str(workspace_id)} if workspace_id else {}),
        **({"intent_type": str(intent_type)} if intent_type else {}),
        **({"query_structured_intent": structured_intent} if structured_intent else {}),
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
    """
    merged: Dict[str, Any] = {}
    for intent in intents:
        local_data = local.get(intent, {}) if isinstance(local.get(intent, {}), dict) else {}
        community_data = community.get(intent, {}) if isinstance(community.get(intent, {}), dict) else {}

        local_matches = local_data.get("matches") if isinstance(local_data.get("matches"), list) else []
        local_finals = local_data.get("final_matches") if isinstance(local_data.get("final_matches"), list) else []
        community_matches = community_data.get("matches") if isinstance(community_data.get("matches"), list) else []
        community_finals = community_data.get("final_matches") if isinstance(community_data.get("final_matches"), list) else []

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
        if local_data.get("error"):
            merged[intent]["error"] = local_data["error"]

    return merged
