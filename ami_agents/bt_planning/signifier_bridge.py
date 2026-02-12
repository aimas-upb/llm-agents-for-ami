"""
Signifier-BT Bridge

Converts between BehaviorTree JSON IR nodes and the Signifier model
used by the RD4 engine in EnvExplorer.

Functions:
- extract_signifiers_from_bt: Walk BT, collect action nodes as signifiers
- build_bt_from_signifiers: Construct BT JSON IR from signifier matches (fast path)
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_signifiers_from_bt(
    tree_spec: dict,
    intents: list[str],
    was_successful: bool = True,
    workspace_id: Optional[str] = None,
) -> list[dict]:
    """
    Walk a BT JSON IR tree and extract signifier-worthy leaf nodes.

    An ActionAffordanceNode maps to a signifier:
      - intent: the intent this action serves
      - affordance_uri: the action_url
      - payload_hint: the parameters
      - context: workspace_id, success status

    Args:
        tree_spec: JSON IR tree specification
        intents: List of intents this tree was generated for
        was_successful: Whether the BT execution succeeded
        workspace_id: Optional workspace URI for context

    Returns:
        List of signifier dicts ready for recording
    """
    signifiers: list[dict] = []
    _walk_tree(tree_spec, intents, was_successful, signifiers, workspace_id)
    return signifiers


def _walk_tree(
    node: dict,
    intents: list[str],
    was_successful: bool,
    signifiers: list[dict],
    workspace_id: Optional[str],
    depth: int = 0,
) -> None:
    """Recursively walk the tree and collect action nodes as signifiers."""
    if not isinstance(node, dict):
        return

    node_type = node.get("type")

    if node_type == "action":
        action_url = node.get("action_url", "")
        parameters = node.get("parameters", {})
        node_name = node.get("name", "")

        # Try to match this action to an intent
        intent = _match_node_to_intent(node_name, action_url, intents)

        signifiers.append({
            "intent": intent,
            "affordance_uri": action_url,
            "action_name": _extract_action_name(action_url),
            "payload_hint": parameters,
            "workspace_id": workspace_id,
            "was_successful": was_successful,
            "source": "bt_execution",
            "node_name": node_name,
        })

    elif node_type in ("sequence", "selector", "parallel"):
        for child in node.get("children", []):
            _walk_tree(child, intents, was_successful, signifiers, workspace_id, depth + 1)


def _match_node_to_intent(
    node_name: str,
    action_url: str,
    intents: list[str],
) -> str:
    """
    Try to match a BT node to one of the intents.

    Uses simple heuristic: check if intent keywords appear in the node name.
    Falls back to the first intent if no match found.
    """
    node_name_lower = node_name.lower()
    action_name = _extract_action_name(action_url).lower().replace("_", " ")

    for intent in intents:
        intent_lower = intent.lower()
        # Check for keyword overlap
        intent_words = set(intent_lower.split())
        name_words = set(node_name_lower.split()) | set(action_name.split())

        # If there's meaningful overlap (beyond stop words)
        stop_words = {"the", "a", "an", "in", "on", "of", "to", "for", "is", "it"}
        meaningful_overlap = (intent_words - stop_words) & (name_words - stop_words)
        if meaningful_overlap:
            return intent

    # Default: assign to first intent
    return intents[0] if intents else "unknown"


def _extract_action_name(url: str) -> str:
    """Extract the action name from a URL (last path segment)."""
    return url.rstrip("/").rsplit("/", 1)[-1] if url else "unknown"


def build_bt_from_signifiers(
    signifier_matches: dict,
    intents: list[str],
) -> Optional[dict]:
    """
    Construct a BT JSON IR directly from signifier matches (fast path).

    If ALL intents have matching signifiers, we can build a plan
    without calling the LLM.  When an intent has multiple
    ``final_matches``, the corresponding actions are wrapped in a
    ``sequence`` node (ordered execution).  Multiple intents are
    wrapped in a ``parallel`` node (independent goals).

    Args:
        signifier_matches: Dict of intent -> match data
        intents: List of intents to satisfy

    Returns:
        BT JSON IR dict if all intents have matches, None otherwise
    """
    if not signifier_matches or not intents:
        return None

    intent_nodes: list[dict] = []

    for intent in intents:
        match_data = signifier_matches.get(intent)
        if not isinstance(match_data, dict):
            return None  # Not all intents matched

        finals = match_data.get("final_matches", [])
        matches = match_data.get("matches", [])

        if not finals and not matches:
            return None  # This intent has no match

        # Resolve ALL final_matches to their full match dicts
        resolved = _resolve_final_matches(finals, matches)
        if not resolved:
            return None

        # Build action nodes for every resolved match
        intent_actions: list[dict] = []
        for m in resolved:
            affordance_uri = m.get("affordance_uri", "")
            if not affordance_uri:
                return None

            action_node: dict[str, Any] = {
                "name": f"action_{intent.replace(' ', '_')[:30]}",
                "type": "action",
                "action_url": affordance_uri,
            }

            payload = m.get("payload_hint") or m.get("payload")
            if payload and isinstance(payload, dict):
                action_node["parameters"] = payload

            intent_actions.append(action_node)

        # Single action for this intent → bare node; multiple → sequence
        if len(intent_actions) == 1:
            intent_nodes.append(intent_actions[0])
        else:
            intent_nodes.append({
                "name": f"Sequence_{intent.replace(' ', '_')[:25]}",
                "type": "sequence",
                "children": intent_actions,
            })

    if not intent_nodes:
        return None

    # Single intent node: return directly
    if len(intent_nodes) == 1:
        return intent_nodes[0]

    # Multiple intents: wrap in parallel (independent goals)
    return {
        "name": "SignifierReusePlan",
        "type": "parallel",
        "policy": "success_on_all",
        "children": intent_nodes,
    }


def _resolve_final_matches(finals: list, matches: list[dict]) -> list[dict]:
    """Resolve final_match IDs to full match dicts, preserving order.

    Falls back to ``[matches[0]]`` when no *finals* can be resolved.
    """
    matches_by_id = {str(m.get("signifier_id")): m for m in matches}
    resolved = []
    for fid in finals:
        m = matches_by_id.get(str(fid))
        if m:
            resolved.append(m)
    # Fallback: if no finals resolved, use first match
    if not resolved and matches:
        resolved = [matches[0]]
    return resolved
