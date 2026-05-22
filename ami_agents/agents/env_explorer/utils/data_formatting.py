"""
Data formatting utilities for EnvExplorer agent.
"""

import json
from typing import Any, Dict, List, Optional

from rdflib import Graph, Namespace, RDF
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


def _ensure_rdf_prefixes(rdf_str: str) -> str:
    """Ensure RDF/Turtle string has necessary namespace declarations before parsing."""
    # Standard prefix block for common namespaces
    prefix_block = """\
@prefix hmas: <https://purl.org/hmas/> .
@prefix ex: <http://example.org/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix jsonschema: <https://www.w3.org/2019/wot/json-schema#> .
@prefix http: <http://www.w3.org/2011/http#> .
@prefix homeont: <https://example.org/homeont#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

"""

    # Check if RDF already has @prefix declarations
    if "@prefix" in rdf_str.lower():
        # Already has prefixes, return as-is
        return rdf_str

    # Prepend standard prefixes
    return prefix_block + rdf_str


def _format_type_with_prefix(type_str: str) -> str:
    """Convert a full URI to ex: prefixed format."""
    if "#" in type_str:
        local = type_str.split("#")[-1]
    else:
        local = type_str.split("/")[-1]
    return f"ex:{local}"


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

    def extract_semantic_types_from_rdf(rdf_str: str) -> List[str]:
        """Extract ALL semantic types (ex: namespace) from RDF/Turtle."""
        if not rdf_str:
            return []
        try:
            from rdflib import Graph, RDF, RDFS, Namespace

            # Ensure namespace declarations are present in the RDF before parsing
            rdf_with_prefixes = _ensure_rdf_prefixes(rdf_str)

            g = Graph()
            g.parse(data=rdf_with_prefixes, format="turtle")

            # Namespaces already bound above

            semantic_types_set = set()  # Use set to avoid duplicates

            # Find the primary artifact subject (has hmas:Artifact type)
            # and get ALL its semantic types (ex: namespace), not affordance types
            for subj in g.subjects(RDF.type, hmas.Artifact):
                # Look for semantic types on this artifact subject (direct types, not nested)
                for type_obj in g.objects(subj, RDF.type):
                    type_str = str(type_obj)
                    # Only collect homeont/example.org types, not hmas or td types
                    if "example.org/" in type_str or "example.org#" in type_str:
                        semantic_types_set.add(_format_type_with_prefix(type_str))
                # Return all collected types for this artifact
                if semantic_types_set:
                    return sorted(list(semantic_types_set))

            # If no Artifact found, try Workspace
            for subj in g.subjects(RDF.type, hmas.Workspace):
                for type_obj in g.objects(subj, RDF.type):
                    type_str = str(type_obj)
                    if "example.org/" in type_str or "example.org#" in type_str:
                        semantic_types_set.add(_format_type_with_prefix(type_str))
                # Return all collected types for this workspace
                if semantic_types_set:
                    return sorted(list(semantic_types_set))

            # Fallback: if no hmas types found, collect all example.org types from the first main subject
            # (handles affordance RDF which may not have hmas wrapper)
            # Only iterate through the first non-blank subject to avoid duplication across graph copies
            first_main_subject = None
            for subj in g.subjects():
                # Skip blank nodes, prefer the first named subject
                if not isinstance(subj, type(None)) and str(subj).startswith("http"):
                    first_main_subject = subj
                    break

            if first_main_subject:
                for type_obj in g.objects(first_main_subject, RDF.type):
                    type_str = str(type_obj)
                    if "example.org/" in type_str or "example.org#" in type_str:
                        semantic_types_set.add(_format_type_with_prefix(type_str))

            if semantic_types_set:
                return sorted(list(semantic_types_set))
        except Exception as e:
            pass
        return []

    def extract_description_from_rdf(rdf_str: str) -> str:
        """Extract rdfs:comment description from RDF/Turtle."""
        if not rdf_str:
            return ""
        try:
            from rdflib import Graph, RDF, RDFS, Namespace

            # Ensure namespace declarations are present in the RDF before parsing
            rdf_with_prefixes = _ensure_rdf_prefixes(rdf_str)

            g = Graph()
            g.parse(data=rdf_with_prefixes, format="turtle")

            hmas = Namespace("https://purl.org/hmas/")

            # Try to find description on artifact subject
            for subj in g.subjects(RDF.type, hmas.Artifact):
                comment = g.value(subj, RDFS.comment)
                if comment:
                    return str(comment)

            # Try to find description on workspace subject
            for subj in g.subjects(RDF.type, hmas.Workspace):
                comment = g.value(subj, RDFS.comment)
                if comment:
                    return str(comment)
        except Exception:
            pass
        return ""

    def build_workspace_node(workspace_id: str) -> Dict[str, Any]:
        """Recursively build a workspace node with sub-workspaces and artifacts."""
        ws = agent_instance.environment_map.get(workspace_id)
        if not ws:
            return None

        # Extract workspace semantic types and description from RDF
        ws_semantic_types = []
        ws_description = ""
        if ws.rdf:
            try:
                from rdflib import Graph, RDF, RDFS, Namespace
                g = Graph()
                g.parse(data=ws.rdf, format="turtle")
                hmas = Namespace("https://purl.org/hmas/")

                # Find workspace subject and ALL its ex: types, plus description
                for subj in g.subjects(RDF.type, hmas.Workspace):
                    for type_obj in g.objects(subj, RDF.type):
                        type_str = str(type_obj)
                        if "example.org/" in type_str or "example.org#" in type_str:
                            if "#" in type_str:
                                ws_semantic_types.append(f"ex:{type_str.split('#')[-1]}")
                            else:
                                ws_semantic_types.append(f"ex:{type_str.split('/')[-1]}")
                    # Extract description
                    comment = g.value(subj, RDFS.comment)
                    if comment:
                        ws_description = str(comment)
            except Exception:
                pass

        artifacts_list = []
        for artifact_id in (ws.artifacts or []):
            artifact = agent_instance.artifacts.get(artifact_id)
            if not artifact:
                continue

            # Extract artifact semantic types and description from Thing Description RDF
            artifact_semantic_types = []
            artifact_description = ""
            if artifact.thing_description and artifact.thing_description.rdf:
                artifact_semantic_types = extract_semantic_types_from_rdf(artifact.thing_description.rdf)
                artifact_description = extract_description_from_rdf(artifact.thing_description.rdf)

            affordances_list = []
            affs = agent_instance.integration_engine.get_affordances_for_artifact(artifact_id)
            for aff in affs:
                # Include ACTION and PROPERTY affordances (skip EVENT)
                if aff.affordance_type not in (AffordanceType.ACTION, AffordanceType.PROPERTY):
                    continue

                # Extract parameters from input_schema (ACTION) or output_schema (PROPERTY)
                parameters = []
                output_properties = []  # For PROPERTY affordances, track output schema property names
                if aff.affordance_type == AffordanceType.ACTION and aff.input_schema:
                    props = aff.input_schema.get("properties", {})
                    if isinstance(props, dict):
                        parameters = list(props.keys())
                elif aff.affordance_type == AffordanceType.PROPERTY and aff.output_schema:
                    props = aff.output_schema.get("properties", {})
                    if isinstance(props, dict):
                        output_properties = list(props.keys())

                # Get semantic types for affordance from RDF
                aff_semantic_types = []
                if aff.rdf:
                    aff_semantic_types = extract_semantic_types_from_rdf(aff.rdf)
                # Fallback to semantic_types if RDF extraction found nothing
                if not aff_semantic_types and aff.semantic_types:
                    # Normalize any full URIs to ex: prefix format
                    aff_semantic_types = [
                        _format_type_with_prefix(st) if ("example.org/" in st or "example.org#" in st) else st
                        for st in aff.semantic_types
                    ]

                # Get description from RDF
                aff_description = ""
                if aff.rdf:
                    aff_description = extract_description_from_rdf(aff.rdf)
                # Fallback to affordance description if RDF extraction found nothing
                if not aff_description:
                    aff_description = aff.description or ""
                    # Filter out synthetic descriptions
                    if aff_description.startswith("Action affordance:") or aff_description.startswith("Property affordance:"):
                        aff_description = ""

                aff_node = {
                    "affordance_id": aff.affordance_id,
                    "artifact_id": artifact_id,
                    "type": f"{aff.affordance_type.value}_affordance",
                    "name": aff.name,
                    "semantic_types": aff_semantic_types,
                    "description": aff_description,
                }

                # Include form information (href, method, etc.)
                if aff.form:
                    aff_node["form"] = {
                        "href": aff.form.href,
                        "method": aff.form.method,
                        "content_type": getattr(aff.form, "content_type", None),
                    }

                # For ACTION affordances, include input parameters
                if aff.affordance_type == AffordanceType.ACTION:
                    aff_node["parameters"] = parameters
                    if aff.input_schema:
                        aff_node["input_schema"] = aff.input_schema
                # For PROPERTY affordances, include output_schema property names as parameters
                elif aff.affordance_type == AffordanceType.PROPERTY:
                    aff_node["parameters"] = output_properties
                    if aff.output_schema:
                        aff_node["output_schema"] = aff.output_schema

                affordances_list.append(aff_node)

            # Include artifacts even if no affordances (show empty list)
            artifacts_list.append({
                "type": "artifact",
                "id": artifact_id,
                "name": artifact.name,
                "semantic_types": artifact_semantic_types,
                "description": artifact_description,
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
            "semantic_types": ws_semantic_types,
            "description": ws_description,
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


def format_capabilities_hierarchical_text(agent_instance) -> str:
    """
    Format capabilities as human-readable hierarchical text for LLM prompts.

    Builds from the hierarchical JSON structure (which has semantic types already extracted).
    Organizes as: Workspace → Artifacts → Affordances, with explicit name: and semantic_types:
    labels, descriptions, and parameters.
    Only includes affordances with semantic_types starting with 'ex:' (homeont namespace).

    Args:
        agent_instance: The EnvExplorerAgent instance

    Returns:
        Human-readable hierarchical string
    """
    # Use the hierarchical JSON structure which has semantic types already properly extracted
    hierarchical_json = format_capabilities_summary_hierarchical(agent_instance)

    if not hierarchical_json.get("discovery_complete"):
        return "Environment discovery not yet complete."

    lines = []

    def format_workspace_node(ws_node: Dict[str, Any], indent: str = "") -> None:
        """Recursively format a workspace node from hierarchical JSON."""
        # Format workspace header with explicit name: and semantic_types:
        lines.append(f"{indent}Workspace:")
        lines.append(f"{indent}  name: {ws_node.get('name', '?')}")
        # Always include hmas:Workspace, plus any ex: types if they exist
        semantic_types = ws_node.get("semantic_types", [])
        all_types = ["hmas:Workspace"] + semantic_types if semantic_types else ["hmas:Workspace"]
        lines.append(f"{indent}  semantic_types: {', '.join(all_types)}")
        description = ws_node.get("description", "")
        if description:
            lines.append(f"{indent}  description: {description}")

        # Format artifacts
        for artifact_node in ws_node.get("artifacts", []):
            lines.append(f"{indent}  Artifact:")
            lines.append(f"{indent}    name: {artifact_node.get('name', '?')}")
            # Always include hmas:Artifact, plus any ex: types if they exist
            artifact_types = artifact_node.get("semantic_types", [])
            artifact_all_types = ["hmas:Artifact"] + artifact_types if artifact_types else ["hmas:Artifact"]
            lines.append(f"{indent}    semantic_types: {', '.join(artifact_all_types)}")
            artifact_desc = artifact_node.get("description", "")
            if artifact_desc:
                lines.append(f"{indent}    description: {artifact_desc}")

            # Format affordances - include ALL affordances (not just those with ex: types)
            has_affordances = False
            for aff_node in artifact_node.get("affordances", []):
                has_affordances = True
                aff_type = aff_node.get("type", "?").replace("_affordance", "")
                lines.append(f"{indent}    {aff_type}:")
                lines.append(f"{indent}      name: {aff_node.get('name', '?')}")

                # Include semantic_types: all of them (ex: types if they exist)
                aff_semantic_types = aff_node.get("semantic_types", [])
                if aff_semantic_types:
                    lines.append(f"{indent}      semantic_types: {', '.join(aff_semantic_types)}")

                aff_desc = aff_node.get("description", "")
                if aff_desc:
                    lines.append(f"{indent}      description: {aff_desc}")

                parameters = aff_node.get("parameters", [])
                if parameters:
                    lines.append(f"{indent}      parameters: {', '.join(parameters)}")

        # Format sub-workspaces
        for sub_ws_node in ws_node.get("sub_workspaces", []):
            format_workspace_node(sub_ws_node, indent + "  ")

    # Process all root workspaces from hierarchical JSON
    for ws_node in hierarchical_json.get("workspaces", []):
        format_workspace_node(ws_node)

    return "\n".join(lines) if lines else "No environment capabilities found."


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

