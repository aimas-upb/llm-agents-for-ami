"""
Data formatting utilities for EnvExplorer agent.
"""

import json
from typing import Any, Dict, List, Optional

from ....shared.models.environment import AffordanceType


def format_capabilities_summary(agent_instance) -> str:
    """
    Format capabilities for human consumption.

    Formats the internal artifact map into a detailed string for the LLM.
    Includes Forms and Input Schemas so the LLM understands parameters.

    Args:
        agent_instance: The EnvExplorerAgent instance

    Returns:
        Human-readable string summary of environment capabilities
    """
    # 1. Check readiness
    if not agent_instance.discovery_complete:
        return "Environment discovery is still in progress. Please try again later."

    # 2. Read from Agent Memory (populated by InitialDiscoveryBehaviour)
    artifacts = agent_instance.artifacts.values()

    if not artifacts:
        return "No artifacts found in the environment."

    summary = "Available Environment Capabilities:\n"

    for artifact in artifacts:
        # 3. Retrieve actions
        # We use the engine's helper to filter affordances for this artifact ID
        actions = agent_instance.integration_engine.get_affordances_for_artifact(artifact.artifact_id)

        # Filter for ACTION types (we only care about what we can DO)
        action_affordances = [a for a in actions if a.affordance_type.value == "action"]

        if action_affordances:
            summary += f"Artifact: {artifact.name}\n"
            summary += f"  ID: {artifact.artifact_id}\n"
            summary += f"  Capabilities:\n"

            for action in action_affordances:
                summary += f"    - Action: {action.name}\n"

                # Include Form Details (Method + URL)
                # This helps the LLM distinguish between GET (read) and POST (write)
                if action.form:
                    summary += f"      Target: [{action.form.method}] {action.form.href}\n"

                # Include Input Schema (Parameters)
                # This tells the LLM what arguments (e.g. brightness level) are required
                if action.input_schema:
                    summary += f"      Schema: {json.dumps(action.input_schema)}\n"

            summary += "\n"

    return summary


def format_capabilities_payload(agent_instance) -> Dict[str, Any]:
    """
    Format capabilities for machine consumption.

    Machine-readable capabilities payload for other agents (planning, etc.).
    Includes a human-friendly 'summary' field for convenience.

    Args:
        agent_instance: The EnvExplorerAgent instance

    Returns:
        Dictionary containing machine-readable environment capabilities
    """
    if not agent_instance.discovery_complete:
        return {
            "discovery_complete": False,
            "error": "discovery_in_progress",
            "summary": "Environment discovery is still in progress. Please try again later.",
            "workspaces": [],
            "artifacts": [],
            "affordances": [],
        }

    artifacts = list(agent_instance.artifacts.values())
    if not artifacts:
        return {
            "discovery_complete": True,
            "summary": "No artifacts found in the environment.",
            "workspaces": [],
            "artifacts": [],
            "affordances": [],
        }

    # Workspaces (for multi-workspace UX and scoping)
    workspaces_out: List[Dict[str, Any]] = []
    try:
        for ws in (agent_instance.environment_map or {}).values():
            workspaces_out.append(
                {
                    "workspace_id": ws.workspace_id,
                    "name": ws.name,
                    "workspace_type": getattr(ws.workspace_type, "value", str(ws.workspace_type)),
                    "parent_workspace_id": getattr(ws, "parent_workspace_id", None),
                    "sub_workspaces": list(getattr(ws, "sub_workspaces", []) or []),
                    "artifacts": list(getattr(ws, "artifacts", []) or []),
                }
            )
    except Exception:
        workspaces_out = []

    affordances_out: List[Dict[str, Any]] = []
    artifacts_out: List[Dict[str, Any]] = []

    for artifact in artifacts:
        affs = agent_instance.integration_engine.get_affordances_for_artifact(artifact.artifact_id)
        action_affordances = [a for a in affs if a.affordance_type == AffordanceType.ACTION]

        actions_out: List[Dict[str, Any]] = []
        for action in action_affordances:
            form = action.form
            actions_out.append(
                {
                    "affordance_id": action.affordance_id,
                    "name": action.name,
                    "description": action.description,
                    "artifact_id": action.artifact_id,
                    "affordance_type": action.affordance_type.value,
                    "semantic_types": list(action.semantic_types or []),
                    "form": {
                        "href": getattr(form, "href", None),
                        "method": getattr(form, "method", None),
                        "content_type": getattr(form, "content_type", None),
                        "operation_type": getattr(form, "operation_type", None),
                        "additional_fields": getattr(form, "additional_fields", None) or {},
                    }
                    if form
                    else None,
                    "input_schema": action.input_schema,
                    "output_schema": action.output_schema,
                }
            )

            affordances_out.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_name": artifact.name,
                    "workspace_id": getattr(artifact, "workspace_id", None),
                    "affordance_id": action.affordance_id,
                    "affordance_type": action.affordance_type.value,
                    "action_name": action.name,
                    "method": getattr(action.form, "method", None) if action.form else None,
                    "target": getattr(action.form, "href", None) if action.form else None,
                    "content_type": getattr(action.form, "content_type", None) if action.form else None,
                    "input_schema": action.input_schema,
                }
            )

        artifacts_out.append(
            {
                "artifact_id": artifact.artifact_id,
                "name": artifact.name,
                "workspace_id": getattr(artifact, "workspace_id", None),
                "actions": actions_out,
            }
        )

    return {
        "discovery_complete": True,
        "summary": format_capabilities_summary(agent_instance),
        "workspaces": workspaces_out,
        "artifacts": artifacts_out,
        "affordances": affordances_out,
    }


def generate_shacl_shapes_from_conditions(structured_conditions: List[Dict[str, Any]]) -> Optional[str]:
    """
    Generate SHACL shapes (Turtle format) from structured_conditions.

    Maps structured conditions to SHACL constraints for context validation:
    - equals → sh:hasValue (exact match)
    - greater_than → sh:minExclusive
    - greater_than_or_equal → sh:minInclusive
    - less_than → sh:maxExclusive
    - less_than_or_equal → sh:maxInclusive

    Args:
        structured_conditions: List of condition dicts with artifact, property, value_conditions

    Returns:
        SHACL shapes as Turtle string, or None if no valid conditions
    """
    if not structured_conditions or not isinstance(structured_conditions, list):
        return None

    # Group conditions by artifact to create one NodeShape per artifact
    artifact_conditions: Dict[str, List[Dict[str, Any]]] = {}
    for cond in structured_conditions:
        if not isinstance(cond, dict):
            continue
        artifact = cond.get("artifact")
        if not artifact:
            continue
        if artifact not in artifact_conditions:
            artifact_conditions[artifact] = []
        artifact_conditions[artifact].append(cond)

    if not artifact_conditions:
        return None

    # Build SHACL Turtle
    shapes_lines = [
        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
    ]

    for idx, (artifact_uri, conditions) in enumerate(artifact_conditions.items()):
        shape_id = f"<urn:signifier:shape:{idx}>"
        shapes_lines.append(f"{shape_id} a sh:NodeShape ;")
        shapes_lines.append(f"    sh:targetNode <{artifact_uri}> ;")

        # Add property constraints
        for prop_idx, cond in enumerate(conditions):
            prop_uri = cond.get("property_affordance")
            value_conditions = cond.get("value_conditions", [])
            if not prop_uri or not value_conditions:
                continue

            is_last_property = (prop_idx == len(conditions) - 1)
            shapes_lines.append("    sh:property [")
            shapes_lines.append(f"        sh:path <{prop_uri}> ;")

            # Process each value condition
            for vc_idx, vc in enumerate(value_conditions):
                if not isinstance(vc, dict):
                    continue
                operator = vc.get("operator", "equals")
                value = vc.get("value")
                if value is None:
                    continue

                # Determine datatype and format value
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    datatype = "xsd:boolean"
                elif isinstance(value, str):
                    # Escape quotes
                    escaped = value.replace('"', '\\"')
                    value_str = f'"{escaped}"'
                    datatype = "xsd:string"
                elif isinstance(value, int):
                    value_str = str(value)
                    datatype = "xsd:integer"
                elif isinstance(value, float):
                    value_str = str(value)
                    datatype = "xsd:double"
                elif isinstance(value, list):
                    # For lists, we can't easily represent in SHACL - skip
                    continue
                elif isinstance(value, dict):
                    # For dicts, we can't easily represent in SHACL - skip
                    continue
                else:
                    escaped = str(value).replace('"', '\\"')
                    value_str = f'"{escaped}"'
                    datatype = "xsd:string"

                # Add datatype constraint (first, before value constraints)
                shapes_lines.append(f"        sh:datatype {datatype} ;")

                # Map operator to SHACL constraint
                is_last_vc = (vc_idx == len(value_conditions) - 1)
                if operator == "equals":
                    shapes_lines.append(f"        sh:hasValue {value_str}{';' if not is_last_vc else ''}")
                elif operator == "greater_than":
                    shapes_lines.append(f"        sh:minExclusive {value_str}{';' if not is_last_vc else ''}")
                elif operator == "greater_than_or_equal":
                    shapes_lines.append(f"        sh:minInclusive {value_str}{';' if not is_last_vc else ''}")
                elif operator == "less_than":
                    shapes_lines.append(f"        sh:maxExclusive {value_str}{';' if not is_last_vc else ''}")
                elif operator == "less_than_or_equal":
                    shapes_lines.append(f"        sh:maxInclusive {value_str}{';' if not is_last_vc else ''}")
                # For not_equals, we could use sh:not but it's complex - skip for now

            shapes_lines.append(f"    ]{';' if not is_last_property else '.'}")

        shapes_lines.append("")

    return "\n".join(shapes_lines)


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