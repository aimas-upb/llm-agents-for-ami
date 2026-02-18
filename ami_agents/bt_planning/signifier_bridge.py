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
    structured_intents: list[dict] | None = None,
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
        structured_intents: Optional list of structured intent dicts for artifact-based matching

    Returns:
        List of signifier dicts ready for recording
    """
    # Build artifact_id -> intent_string mapping for accurate matching
    artifact_to_intent: dict[str, str] = {}
    # Build intent_string -> structured_intent_dict mapping for storage
    intent_to_structured: dict[str, dict] = {}
    if structured_intents and len(structured_intents) == len(intents):
        for intent_str, si in zip(intents, structured_intents):
            if isinstance(si, dict):
                if "artifact" in si:
                    artifact_id = str(si["artifact"])
                    artifact_to_intent[artifact_id] = intent_str
                # Store the full structured_intent for later use
                intent_to_structured[intent_str] = si

    signifiers: list[dict] = []
    _walk_tree(tree_spec, intents, was_successful, signifiers, workspace_id, state_snapshot, intent_type, artifact_to_intent, intent_to_structured)
    return signifiers


def _walk_tree(
    node: dict,
    intents: list[str],
    was_successful: bool,
    signifiers: list[dict],
    workspace_id: Optional[str],
    state_snapshot: Optional[dict],
    intent_type: Optional[str],
    artifact_to_intent: dict[str, str],
    intent_to_structured: dict[str, dict],
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

        # Try to match this action to an intent (using artifact-based matching if available)
        intent = _match_node_to_intent(node_name, action_url, intents, artifact_to_intent)

        # Extract structured conditions from state snapshot
        structured_conditions = _extract_conditions_from_state(
            state_snapshot, workspace_id, action_url, parameters
        )

        # Build signifier dict with structured_intent if available
        sig_dict = {
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
        }

        # Add original structured_intent if available for this intent
        if intent in intent_to_structured:
            sig_dict["structured_intent"] = intent_to_structured[intent]

        signifiers.append(sig_dict)

    elif node_type in ("sequence", "selector", "parallel"):
        for child in node.get("children", []):
            _walk_tree(child, intents, was_successful, signifiers, workspace_id, state_snapshot, intent_type, artifact_to_intent, intent_to_structured, depth + 1)


def _match_node_to_intent(
    node_name: str,
    action_url: str,
    intents: list[str],
    artifact_to_intent: dict[str, str],
) -> str:
    """
    Try to match a BT node to one of the intents.

    Strategy:
    1. If artifact_to_intent mapping is available, extract artifact_id from action_url
       and look up the corresponding intent (most accurate)
    2. Otherwise, use keyword-based heuristic matching
    3. Falls back to first intent if no match found

    Args:
        node_name: Name of the BT action node
        action_url: Affordance URL of the action
        intents: List of intent strings
        artifact_to_intent: Mapping from artifact_id to intent_string (from structured_intents)

    Returns:
        The matched intent string
    """
    # STRATEGY 1: Artifact-based matching (most accurate)
    if artifact_to_intent:
        # Extract artifact_id from action_url
        # e.g., "http://localhost:8080/workspaces/lab308/artifacts/lights_308/ha/light/turn_on"
        #       -> "lights_308"
        if "/artifacts/" in action_url:
            parts = action_url.split("/artifacts/")
            if len(parts) > 1:
                # Get the part after "/artifacts/" and extract first segment
                artifact_part = parts[1].split("/")[0]
                # Try exact match
                if artifact_part in artifact_to_intent:
                    logger.info(
                        demo("[INTENT_MATCH] Exact artifact match: artifact=%s -> intent=%r"),
                        artifact_part,
                        artifact_to_intent[artifact_part]
                    )
                    return artifact_to_intent[artifact_part]
                # Try partial match (e.g., "lights_308" in URL fragment with anchor)
                for artifact_id, intent_str in artifact_to_intent.items():
                    if artifact_id in artifact_part:
                        logger.info(
                            demo("[INTENT_MATCH] Partial artifact match: artifact=%s (from %s) -> intent=%r"),
                            artifact_id,
                            artifact_part,
                            intent_str
                        )
                        return intent_str

    # STRATEGY 2: Keyword-based matching (fallback)
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

    if best_intent:
        logger.info(
            demo("[INTENT_MATCH] Keyword-based match: node=%s action=%s -> intent=%r"),
            node_name,
            action_name,
            best_intent
        )
        return best_intent

    # STRATEGY 3: Fallback to first intent
    fallback = intents[0] if intents else "unknown"
    logger.warning(
        demo("[INTENT_MATCH] No match found, falling back to first intent: %r (node=%s, action_url=%s)"),
        fallback,
        node_name,
        action_url
    )
    return fallback


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


def build_bt_from_signifiers(
    signifier_matches: dict,
    intents: list[str],
    structured_intents: list[dict] | None = None,
) -> Optional[dict]:
    """
    Construct a BT JSON IR directly from signifier matches (fast path).

    If ALL intents have matching signifiers, we can build a plan
    without calling the LLM.  When an intent has multiple
    ``final_matches``, the corresponding actions are wrapped in a
    ``sequence`` node (ordered execution).  Multiple intents are
    wrapped in a ``parallel`` node (independent goals).

    When *structured_intents* are provided, the function applies
    intent-aware payload handling:

    - ``modify`` intents cause the fast path to bail out (returns None)
      because they require a read-compute-set pattern the LLM must handle.
    - ``set`` intents with an explicit value override the signifier's
      ``payload_hint`` so the user's actual target value is used.
    - ``check`` intents skip parameters entirely.
    - When no structured intent is available, falls back to reusing the
      signifier's ``payload_hint`` as-is (backward compatibility).

    Args:
        signifier_matches: Dict of intent -> match data
        intents: List of intents to satisfy
        structured_intents: Optional list of structured intent dicts
            (parallel to *intents*) with keys ``action``, ``parameter``,
            ``value``, etc.

    Returns:
        BT JSON IR dict if all intents have matches, None otherwise
    """
    if not signifier_matches or not intents:
        return None

    # Build lookup: intent_string -> structured_intent_dict
    structured_map: dict[str, dict] = {}
    if structured_intents and len(structured_intents) == len(intents):
        for intent_str, si in zip(intents, structured_intents):
            if isinstance(si, dict):
                structured_map[intent_str] = si
        logger.info(demo("build_bt_from_signifiers: received %d structured_intents for parameter adaptation"), len(structured_intents))
    else:
        logger.info(demo("build_bt_from_signifiers: NO structured_intents, using backward compatibility mode (direct payload_hint copy)"))

    intent_nodes: list[dict] = []

    for intent in intents:
        si = structured_map.get(intent)

        # modify intents require read-compute-set — bail to LLM path
        if si and si.get("action") == "modify":
            logger.info(demo("build_bt_from_signifiers: MODIFY action detected for intent=%r, bailing to LLM path"), intent)
            return None

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

            # Intent-aware payload selection (parameter adaptation logic)
            if si and si.get("action") == "set" and si.get("value") is not None and si.get("parameter"):
                param_name = si["parameter"]
                # Special case: 'on_off' is a semantic parameter, not an API parameter
                # The action (turn_on vs turn_off) is already encoded in the affordance_uri
                if param_name == "on_off":
                    logger.info(demo("build_bt_from_signifiers: SET action with on_off parameter - skipping (encoded in affordance_uri)"))
                    # Don't add on_off to parameters; it's already in the URI
                else:
                    # Use the caller's actual target value, not the stale signifier hint
                    action_node["parameters"] = {param_name: si["value"]}
                    logger.info(demo("build_bt_from_signifiers: SET action - overriding payload_hint with structured_intent value: %s=%s"), param_name, si["value"])
            elif si and si.get("action") == "check":
                logger.info(demo("build_bt_from_signifiers: CHECK action - no parameters needed"))
                pass  # check intents have no parameters
            else:
                # Fallback: reuse signifier's payload_hint as-is (backward compatibility)
                payload = m.get("payload_hint") or m.get("payload")
                if payload and isinstance(payload, dict):
                    action_node["parameters"] = payload
                    logger.info(demo("build_bt_from_signifiers: using payload_hint from signifier (no structured_intent or fallback): %s"), payload)

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
