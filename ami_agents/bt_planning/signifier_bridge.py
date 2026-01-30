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
    without calling the LLM.

    Args:
        signifier_matches: Dict of intent -> match data
        intents: List of intents to satisfy

    Returns:
        BT JSON IR dict if all intents have matches, None otherwise
    """
    if not signifier_matches or not intents:
        return None

    action_nodes: list[dict] = []

    for intent in intents:
        match_data = signifier_matches.get(intent)
        if not isinstance(match_data, dict):
            return None  # Not all intents matched

        finals = match_data.get("final_matches", [])
        matches = match_data.get("matches", [])

        if not finals and not matches:
            return None  # This intent has no match

        # Find best match
        best_match = None
        if matches:
            if finals:
                for m in matches:
                    if str(m.get("signifier_id")) == str(finals[0]):
                        best_match = m
                        break
            if not best_match:
                best_match = matches[0]

        if not best_match:
            return None

        affordance_uri = best_match.get("affordance_uri", "")
        if not affordance_uri:
            return None

        action_node = {
            "name": f"action_{intent.replace(' ', '_')[:30]}",
            "type": "action",
            "action_url": affordance_uri,
        }

        payload = best_match.get("payload_hint") or best_match.get("payload")
        if payload and isinstance(payload, dict):
            action_node["parameters"] = payload

        action_nodes.append(action_node)

    if not action_nodes:
        return None

    # Single action: return directly
    if len(action_nodes) == 1:
        return action_nodes[0]

    # Multiple actions: wrap in parallel (independent commands)
    return {
        "name": "SignifierReusePlan",
        "type": "parallel",
        "policy": "success_on_all",
        "children": action_nodes,
    }
