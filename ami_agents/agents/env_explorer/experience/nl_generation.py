from typing import Any, Dict, List


def generate_nl_description(structured_conditions: List[Dict[str, Any]], ctx_meta: Dict[str, Any]) -> str:
    """
    Generate natural language description from structured conditions.

    Converts structured conditions into human-readable text.
    Falls back to workspace metadata if no conditions available.

    Args:
        structured_conditions: List of condition dicts with artifact, property, value_conditions
        ctx_meta: Context metadata (workspace_id, was_successful, etc.)

    Returns:
        Natural language description string
    """
    if structured_conditions and isinstance(structured_conditions, list) and len(structured_conditions) > 0:
        # Build description from conditions
        parts = []
        for cond in structured_conditions:
            if not isinstance(cond, dict):
                continue

            artifact = cond.get("artifact", "")
            prop = cond.get("property_affordance", "")
            value_conditions = cond.get("value_conditions", [])

            # Extract artifact ID from URI (e.g., "lights_308" from full URI)
            artifact_id = artifact.rstrip("/").rsplit("/", 1)[-1] if artifact else "artifact"
            # Extract property name from URI
            prop_name = prop.rstrip("/").rsplit("/", 1)[-1] if prop else "property"

            # Format value conditions
            for vc in value_conditions:
                if not isinstance(vc, dict):
                    continue
                operator = vc.get("operator", "equals")
                value = vc.get("value")

                if operator == "equals":
                    parts.append(f"{artifact_id} {prop_name} is {value}")
                elif operator == "greater_than":
                    parts.append(f"{artifact_id} {prop_name} > {value}")
                elif operator == "greater_than_or_equal":
                    parts.append(f"{artifact_id} {prop_name} >= {value}")
                elif operator == "less_than":
                    parts.append(f"{artifact_id} {prop_name} < {value}")
                elif operator == "less_than_or_equal":
                    parts.append(f"{artifact_id} {prop_name} <= {value}")
                elif operator == "not_equals":
                    parts.append(f"{artifact_id} {prop_name} != {value}")

        if parts:
            return "; ".join(parts)

    # Fallback: use workspace metadata
    workspace_id = ctx_meta.get("workspace_id", "unknown")
    was_successful = ctx_meta.get("was_successful", True)
    status = "successful" if was_successful else "failed"
    return f"Execution in workspace {workspace_id} ({status})"