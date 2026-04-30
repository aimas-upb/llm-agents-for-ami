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


