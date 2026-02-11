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

from ..shared.utils.demo_log import demo

logger = logging.getLogger(__name__)


def extract_signifiers_from_bt(
    tree_spec: dict,
    intents: list[str],
    was_successful: bool = True,
    workspace_id: Optional[str] = None,
    state_snapshot: Optional[dict] = None,
    intent_type: Optional[str] = None,
) -> list[dict]:
    """
    Walk a BT JSON IR tree and extract signifier-worthy leaf nodes.

    An ActionAffordanceNode maps to a signifier:
      - intent: the intent this action serves
      - affordance_uri: the action_url
      - payload_hint: the parameters
      - context: workspace_id, success status, structured_conditions from state
      - intent_type: EXPLICIT or IMPLICIT classification

    Args:
        tree_spec: JSON IR tree specification
        intents: List of intents this tree was generated for
        was_successful: Whether the BT execution succeeded
        workspace_id: Optional workspace URI for context
        state_snapshot: Optional state snapshot from execution context for building structured_conditions
        intent_type: Intent type classification (EXPLICIT or IMPLICIT)

    Returns:
        List of signifier dicts ready for recording
    """
    signifiers: list[dict] = []
    _walk_tree(tree_spec, intents, was_successful, signifiers, workspace_id, state_snapshot, intent_type)
    return signifiers


def _walk_tree(
    node: dict,
    intents: list[str],
    was_successful: bool,
    signifiers: list[dict],
    workspace_id: Optional[str],
    state_snapshot: Optional[dict],
    intent_type: Optional[str],
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

        # Extract structured conditions from state snapshot
        structured_conditions = _extract_conditions_from_state(
            state_snapshot, workspace_id, action_url, parameters
        )

        signifiers.append({
            "intent": intent,
            "affordance_uri": action_url,
            "action_name": _extract_action_name(action_url),
            "payload_hint": parameters,
            "workspace_id": workspace_id,
            "was_successful": was_successful,
            "source": "bt_execution",
            "node_name": node_name,
            "structured_conditions": structured_conditions,
            "intent_type": intent_type,  # Add intent_type for memory engine filtering
        })

    elif node_type in ("sequence", "selector", "parallel"):
        for child in node.get("children", []):
            _walk_tree(child, intents, was_successful, signifiers, workspace_id, state_snapshot, intent_type, depth + 1)


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

    # Best match: track which intent has the most specific overlap
    best_intent = None
    best_overlap_size = 0

    for intent in intents:
        intent_lower = intent.lower()
        # Check for keyword overlap
        intent_words = set(intent_lower.split())
        name_words = set(node_name_lower.split()) | set(action_name.split())

        # If there's meaningful overlap (beyond stop words)
        stop_words = {"the", "a", "an", "in", "on", "of", "to", "for", "is", "it"}
        meaningful_overlap = (intent_words - stop_words) & (name_words - stop_words)

        # Prioritize match that includes action-specific keywords (on/off/set/check)
        # This ensures "turn on" matches "turn on" intent, not "turn off" intent
        action_keywords = {"on", "off", "set", "check", "open", "close", "increase", "decrease"}
        action_specific_overlap = meaningful_overlap & action_keywords

        if action_specific_overlap:
            # If we have action-specific overlap, this is a strong match
            overlap_size = len(meaningful_overlap)
            if overlap_size > best_overlap_size:
                best_overlap_size = overlap_size
                best_intent = intent
        elif meaningful_overlap and best_overlap_size == 0:
            # Generic overlap, only use if no action-specific match found
            overlap_size = len(meaningful_overlap)
            if overlap_size > best_overlap_size:
                best_overlap_size = overlap_size
                best_intent = intent

    # Return best match or default to first intent
    return best_intent if best_intent else (intents[0] if intents else "unknown")


def _extract_action_name(url: str) -> str:
    """Extract the action name from a URL (last path segment)."""
    return url.rstrip("/").rsplit("/", 1)[-1] if url else "unknown"


def _extract_conditions_from_state(
    state_snapshot: Optional[dict],
    workspace_id: Optional[str],
    action_url: str,
    parameters: dict,
) -> list[dict]:
    """
    Extract structured conditions from state snapshot for signifier context.

    Builds conditions array with artifact/propertyAffordance/valueConditions structure
    compatible with CASHMERE IntentContext format.

    Args:
        state_snapshot: State snapshot dict with artifacts and their properties
        workspace_id: Workspace URI for filtering relevant artifacts
        action_url: The action URL being performed (for context)
        parameters: Action parameters (may contain artifact references)

    Returns:
        List of condition dicts in format:
        {
            "artifact": "artifact_uri",
            "propertyAffordance": "property_uri",
            "valueConditions": [{"operator": "equals", "value": <value>}]
        }
    """
    # Debug with both logging and print (logging may not be configured for this module)
    print(f"[DEBUG] _extract_conditions_from_state CALLED: state_snapshot type={type(state_snapshot).__name__ if state_snapshot else 'None'}, workspace_id={workspace_id!r}")
    logger.info(
        demo("_extract_conditions_from_state: state_snapshot type=%s, has_data=%s, workspace_id=%r"),
        type(state_snapshot).__name__ if state_snapshot else "None",
        bool(state_snapshot),
        workspace_id,
    )

    if not state_snapshot or not isinstance(state_snapshot, dict):
        print("[DEBUG] _extract_conditions_from_state: early return (no state_snapshot or not dict)")
        logger.info(demo("_extract_conditions_from_state: early return (no state_snapshot or not dict)"))
        return []

    artifacts = state_snapshot.get("artifacts", {})
    print(f"[DEBUG] _extract_conditions_from_state: artifacts count={len(artifacts) if isinstance(artifacts, dict) else 0}, keys={list(artifacts.keys())[:3] if isinstance(artifacts, dict) else []}")
    logger.info(
        demo("_extract_conditions_from_state: artifacts type=%s, count=%d, keys=%s"),
        type(artifacts).__name__,
        len(artifacts) if isinstance(artifacts, dict) else 0,
        list(artifacts.keys())[:3] if isinstance(artifacts, dict) else [],
    )

    if not isinstance(artifacts, dict):
        logger.info(demo("_extract_conditions_from_state: early return (artifacts not dict)"))
        return []

    conditions: list[dict] = []

    # Extract artifact ID from parameters or action_url
    target_artifact_id = None
    if isinstance(parameters, dict):
        # Common parameter names for artifact references
        for key in ("artifact_id", "artifact", "device_id", "device", "target"):
            if key in parameters:
                target_artifact_id = str(parameters[key])
                break

    # If no artifact ID in parameters, try to extract from action_url
    # e.g., "http://example.org/artifacts/light308/turn_on" -> "light308"
    if not target_artifact_id and action_url:
        url_parts = action_url.rstrip("/").split("/")
        if "artifacts" in url_parts:
            idx = url_parts.index("artifacts")
            if idx + 1 < len(url_parts):
                target_artifact_id = url_parts[idx + 1]

    # Iterate through artifacts and extract relevant properties
    for artifact_id, artifact_info in artifacts.items():
        if not isinstance(artifact_info, dict):
            continue

        # Filter by workspace if specified
        if workspace_id:
            artifact_workspace = artifact_info.get("workspace_id")
            if artifact_workspace and str(artifact_workspace) != str(workspace_id):
                # Also check if workspace_id is substring (e.g., "lab308" in full URI)
                if workspace_id not in str(artifact_workspace):
                    continue

        # Get artifact URI (prefer full URI, fallback to ID)
        artifact_uri = artifact_info.get("artifact_id") or artifact_id

        # Extract properties from artifact state
        # NOTE: EnvExplorer returns "state" field, not "properties"
        properties = artifact_info.get("state") or artifact_info.get("properties", {})
        if not isinstance(properties, dict):
            continue

        for prop_name, prop_value in properties.items():
            # Build property affordance URI (best effort)
            prop_uri = prop_name  # May be a full URI or just a name

            # Determine operator based on value type and context
            operator = "equals"  # Default operator
            if isinstance(prop_value, (int, float)):
                # For numeric values, we don't know if it should be >, <, or =
                # Default to equals, but this could be enhanced with heuristics
                operator = "equals"

            condition = {
                "artifact": artifact_uri,
                "property_affordance": prop_uri,
                "value_conditions": [
                    {
                        "operator": operator,
                        "value": prop_value,
                    }
                ],
            }
            conditions.append(condition)

    # If we have too many conditions (noisy state), filter to most relevant
    # Keep only conditions for the target artifact if identified
    if target_artifact_id and len(conditions) > 5:
        relevant_conditions = [
            c for c in conditions
            if target_artifact_id in str(c.get("artifact", ""))
        ]
        if relevant_conditions:
            logger.info(
                demo("_extract_conditions_from_state: returning %d relevant conditions (filtered from %d)"),
                len(relevant_conditions), len(conditions)
            )
            return relevant_conditions

    # Limit to prevent excessive context (keep max 10 conditions)
    result = conditions[:10]
    print(f"[DEBUG] _extract_conditions_from_state: returning {len(result)} conditions (total found: {len(conditions)})")
    logger.info(demo("_extract_conditions_from_state: returning %d conditions (total found: %d)"), len(result), len(conditions))
    return result


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
