"""
Data formatting utilities for EnvExplorer agent.
"""

import json
from typing import Any, Dict, List, Optional

from rdflib import Graph
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
            "semantic_capabilities": dict(getattr(agent_instance, "semantic_capabilities", {}) or {}),
        }

    artifacts = list(agent_instance.artifacts.values())
    if not artifacts:
        return {
            "discovery_complete": True,
            "summary": "No artifacts found in the environment.",
            "workspaces": [],
            "artifacts": [],
            "affordances": [],
            "semantic_capabilities": dict(getattr(agent_instance, "semantic_capabilities", {}) or {}),
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
        "semantic_capabilities": dict(getattr(agent_instance, "semantic_capabilities", {}) or {}),
    }


def format_capabilities_summary_hierarchical(agent_instance) -> Dict[str, Any]:
    """
    Format capabilities as a hierarchical JSON structure.

    Organizes workspaces → sub-workspaces → artifacts → affordances with
    name, description, and parameter lists. LLM-friendly for segmentation.

    Args:
        agent_instance: The EnvExplorerAgent instance

    Returns:
        Hierarchical dict following workspace → artifact → affordance structure
    """
    if not agent_instance.discovery_complete:
        return {
            "type": "environment",
            "discovery_complete": False,
            "workspaces": [],
        }

    def build_workspace_node(workspace_id: str) -> Dict[str, Any]:
        """Recursively build a workspace node with sub-workspaces and artifacts."""
        ws = agent_instance.environment_map.get(workspace_id)
        if not ws:
            return None

        artifacts_list = []
        for artifact_id in (ws.artifacts or []):
            artifact = agent_instance.artifacts.get(artifact_id)
            if not artifact:
                continue

            affordances_list = []
            affs = agent_instance.integration_engine.get_affordances_for_artifact(artifact_id)
            for aff in affs:
                # Include ACTION and PROPERTY affordances (skip EVENT)
                if aff.affordance_type not in (AffordanceType.ACTION, AffordanceType.PROPERTY):
                    continue

                # Extract parameters from input_schema (ACTION) or output_schema (PROPERTY)
                parameters = []
                if aff.affordance_type == AffordanceType.ACTION and aff.input_schema:
                    props = aff.input_schema.get("properties", {})
                    if isinstance(props, dict):
                        parameters = list(props.keys())
                elif aff.affordance_type == AffordanceType.PROPERTY and aff.output_schema:
                    props = aff.output_schema.get("properties", {})
                    if isinstance(props, dict):
                        parameters = list(props.keys())

                # Filter out synthetic descriptions (e.g., "Action affordance: ...")
                description = aff.description or ""
                if description.startswith("Action affordance:") or description.startswith("Property affordance:"):
                    description = ""

                affordances_list.append({
                    "type": f"{aff.affordance_type.value}_affordance",
                    "name": aff.name,
                    "description": description,
                    "parameters": parameters,
                })

            if affordances_list:  # Only include artifacts with affordances
                artifacts_list.append({
                    "type": "artifact",
                    "id": artifact_id,
                    "name": artifact.name,
                    "description": "",
                    "affordances": affordances_list,
                })

        # Recursively process sub-workspaces
        sub_workspaces_list = []
        for sub_ws_id in (ws.sub_workspaces or []):
            sub_node = build_workspace_node(sub_ws_id)
            if sub_node:
                sub_workspaces_list.append(sub_node)

        return {
            "type": "workspace",
            "id": workspace_id,
            "name": ws.name,
            "description": "",
            "sub_workspaces": sub_workspaces_list,
            "artifacts": artifacts_list,
        }

    # Find root workspaces (those with no parent)
    root_workspaces = []
    for ws in (agent_instance.environment_map or {}).values():
        if not ws.parent_workspace_id:
            node = build_workspace_node(ws.workspace_id)
            if node:
                root_workspaces.append(node)

    return {
        "type": "environment",
        "discovery_complete": True,
        "workspaces": root_workspaces,
    }


def format_capabilities_detailed_rdf(agent_instance) -> str:
    """
    Format capabilities as a merged RDF/Turtle graph.

    Assembles all Workspace.rdf and Artifact ThingDescription.rdf into a
    single merged Turtle string with canonicalized prefixes.

    Args:
        agent_instance: The EnvExplorerAgent instance

    Returns:
        Turtle string with merged RDF graph and canonicalized prefixes
    """
    if not agent_instance.discovery_complete:
        return ""

    # Canonical prefix block
    prefix_block = """\
@prefix hmas: <https://purl.org/hmas/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix jsonschema: <https://www.w3.org/2019/wot/json-schema#> .
@prefix http: <http://www.w3.org/2011/http#> .
@prefix homeont: <https://example.org/homeont#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

"""

    # Create a merged graph
    merged_graph = Graph()

    # Add all workspace RDF
    for ws in (agent_instance.environment_map or {}).values():
        if ws.rdf:
            try:
                g = Graph()
                g.parse(data=ws.rdf, format="turtle")
                merged_graph += g
            except Exception:
                # Skip malformed RDF
                pass

    # Add all artifact Thing Description RDF
    for artifact in (agent_instance.artifacts or {}).values():
        if artifact.thing_description and artifact.thing_description.rdf:
            try:
                g = Graph()
                g.parse(data=artifact.thing_description.rdf, format="turtle")
                merged_graph += g
            except Exception:
                # Skip malformed RDF
                pass

    # Serialize merged graph to Turtle
    serialized = merged_graph.serialize(format="turtle")

    # Strip any @prefix declarations emitted by rdflib (to avoid duplication)
    # and prepend our canonical block
    lines = serialized.split("\n")
    filtered_lines = [line for line in lines if not line.strip().startswith("@prefix")]
    filtered_content = "\n".join(filtered_lines).strip()

    # Combine prefix block with filtered content
    if filtered_content:
        return prefix_block + filtered_content
    else:
        return prefix_block.rstrip()

