"""
Data formatting utilities for EnvExplorer agent.
"""

import json
from typing import Any, Dict, List, Optional

from rdflib import Graph, Namespace, RDF, URIRef
from ....shared.models.environment import AffordanceType


def _define_standard_namespaces(graph: Graph) -> dict:
    """Bind all standard RDF namespaces to a graph and return them as a dict."""
    ex = Namespace("http://example.org/")
    hctl = Namespace("https://www.w3.org/2019/wot/hypermedia#")
    hmas = Namespace("https://purl.org/hmas/")
    http = Namespace("http://www.w3.org/2011/http#")
    jacamo = Namespace("https://purl.org/hmas/jacamo/")
    td = Namespace("https://www.w3.org/2019/wot/td#")
    websub = Namespace("https://purl.org/hmas/websub/")
    wotsec = Namespace("https://www.w3.org/2019/wot/security#")

    graph.bind("ex", ex)
    graph.bind("hctl", hctl)
    graph.bind("hmas", hmas)
    graph.bind("http", http)
    graph.bind("jacamo", jacamo)
    graph.bind("td", td)
    graph.bind("websub", websub)
    graph.bind("wotsec", wotsec)

    return {
        "ex": ex, "hctl": hctl, "hmas": hmas, "http": http,
        "jacamo": jacamo, "td": td, "websub": websub, "wotsec": wotsec
    }



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
            from rdflib import Graph, RDF, RDFS

            # Ensure namespace declarations are present in the RDF before parsing
            rdf_with_prefixes = _ensure_rdf_prefixes(rdf_str)

            g = Graph()
            g.parse(data=rdf_with_prefixes, format="turtle")

            # Bind all standard namespaces
            ns = _define_standard_namespaces(g)
            hmas = ns["hmas"]
            ex = ns["ex"]

            semantic_types_set = set()  # Use set to avoid duplicates

            # Find the primary artifact subject (has hmas:Artifact type)
            # and get ALL its semantic types (hmas:Artifact + ex: types)
            for subj in g.subjects(RDF.type, hmas.Artifact):
                # Always include the base hmas:Artifact type
                semantic_types_set.add("hmas:Artifact")

                # Also collect any ex: semantic types on this artifact subject
                for type_obj in g.objects(subj, RDF.type):
                    if type_obj in ex:  # Check if type is in the ex: namespace
                        semantic_types_set.add(f"ex:{str(type_obj).split('/')[-1]}")
                # Return all collected types for this artifact
                if semantic_types_set:
                    return sorted(list(semantic_types_set))

            # If no Artifact found, try Workspace
            for subj in g.subjects(RDF.type, hmas.Workspace):
                for type_obj in g.objects(subj, RDF.type):
                    if type_obj in ex:  # Check if type is in the ex: namespace
                        semantic_types_set.add(f"ex:{str(type_obj).split('/')[-1]}")
                # Return all collected types for this workspace
                if semantic_types_set:
                    return sorted(list(semantic_types_set))

            # Fallback: if no hmas types found, collect all example.org types from any subject
            # (handles affordance RDF which may not have hmas wrapper)
            from rdflib.term import BNode
            for subj in g.subjects():
                for type_obj in g.objects(subj, RDF.type):
                    if type_obj in ex:  # Check if type is in the ex: namespace
                        semantic_types_set.add(f"ex:{str(type_obj).split('/')[-1]}")

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
            from rdflib import Graph, RDF, RDFS

            # Ensure namespace declarations are present in the RDF before parsing
            rdf_with_prefixes = _ensure_rdf_prefixes(rdf_str)

            g = Graph()
            g.parse(data=rdf_with_prefixes, format="turtle")

            # Bind all standard namespaces
            ns = _define_standard_namespaces(g)
            hmas = ns["hmas"]

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
                from rdflib import Graph, RDF, RDFS
                g = Graph()
                g.parse(data=ws.rdf, format="turtle")
                ns = _define_standard_namespaces(g)
                hmas = ns["hmas"]
                ex = ns["ex"]

                # Find workspace subject and ALL its types (hmas:Workspace + ex: types), plus description
                for subj in g.subjects(RDF.type, hmas.Workspace):
                    # Always include the base hmas:Workspace type
                    ws_semantic_types.append("hmas:Workspace")

                    # Also collect any ex: semantic types
                    for type_obj in g.objects(subj, RDF.type):
                        if type_obj in ex:  # Check if type is in the ex: namespace
                            ws_semantic_types.append(f"ex:{str(type_obj).split('/')[-1]}")
                    # Extract description
                    comment = g.value(subj, RDFS.comment)
                    if comment:
                        ws_description = str(comment)
            except Exception:
                pass

        # Remove duplicate types while preserving order.
        seen_ws_types = set()
        ws_semantic_types = [
            t for t in ws_semantic_types
            if not (t in seen_ws_types or seen_ws_types.add(t))
        ]

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

                # Get semantic types for affordance (always include base td: type)
                aff_semantic_types = []

                # Determine the base td: type based on affordance type
                if aff.affordance_type == AffordanceType.ACTION:
                    aff_semantic_types.append("td:ActionAffordance")
                elif aff.affordance_type == AffordanceType.PROPERTY:
                    aff_semantic_types.append("td:PropertyAffordance")

                # Also collect any ex: semantic types from RDF
                if aff.rdf:
                    extracted_types = extract_semantic_types_from_rdf(aff.rdf)
                    aff_semantic_types.extend(extracted_types)

                # Always check semantic_types as fallback (affordance RDF may not include ex: types)
                if aff.semantic_types:
                    ex = Namespace("http://example.org/")
                    for st in aff.semantic_types:
                        # Check if it's an ex: namespace URI
                        try:
                            type_uri = URIRef(st) if not isinstance(st, URIRef) else st
                            if type_uri in ex:
                                formatted = f"ex:{str(type_uri).split('/')[-1]}"
                                # Only add if not already present (avoid duplicates)
                                if formatted not in aff_semantic_types:
                                    aff_semantic_types.append(formatted)
                            else:
                                if st not in aff_semantic_types:
                                    aff_semantic_types.append(st)
                        except:
                            if st not in aff_semantic_types:
                                aff_semantic_types.append(st)

                # Remove duplicates while preserving order
                seen = set()
                aff_semantic_types = [x for x in aff_semantic_types if not (x in seen or seen.add(x))]

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

    def dedupe_preserve_order(values: List[str]) -> List[str]:
        """Remove duplicates while keeping first-seen order."""
        seen = set()
        return [v for v in values if not (v in seen or seen.add(v))]

    def format_workspace_node(ws_node: Dict[str, Any], indent: str = "") -> None:
        """Recursively format a workspace node from hierarchical JSON."""
        # Format workspace header with explicit name: and semantic_types:
        lines.append(f"{indent}Workspace:")
        lines.append(f"{indent}  name: {ws_node.get('name', '?')}")
        # Always include hmas:Workspace, plus any ex: types if they exist
        semantic_types = ws_node.get("semantic_types", [])
        all_types = dedupe_preserve_order(["hmas:Workspace", *semantic_types])
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
            artifact_all_types = dedupe_preserve_order(["hmas:Artifact", *artifact_types])
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
@prefix ex: <http://example.org/> .
@prefix hmas: <https://purl.org/hmas/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix jsonschema: <https://www.w3.org/2019/wot/json-schema#> .
@prefix http: <http://www.w3.org/2011/http#> .
@prefix homeont: <https://example.org/homeont#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix jacamo: <https://purl.org/hmas/jacamo/> .
@prefix wotsec: <https://www.w3.org/2019/wot/security#> .
@prefix websub: <https://purl.org/hmas/websub/> .

"""

    # Create a merged graph
    merged_graph = Graph()

    # Bind canonical namespaces to the merged graph BEFORE adding triples
    # This ensures rdflib uses these prefixes during serialization instead of generating ns1, ns2, etc.
    from rdflib import Namespace
    merged_graph.bind("hmas", Namespace("https://purl.org/hmas/"))
    merged_graph.bind("td", Namespace("https://www.w3.org/2019/wot/td#"))
    merged_graph.bind("hctl", Namespace("https://www.w3.org/2019/wot/hypermedia#"))
    merged_graph.bind("jsonschema", Namespace("https://www.w3.org/2019/wot/json-schema#"))
    merged_graph.bind("http", Namespace("http://www.w3.org/2011/http#"))
    merged_graph.bind("homeont", Namespace("https://example.org/homeont#"))
    merged_graph.bind("rdfs", Namespace("http://www.w3.org/2000/01/rdf-schema#"))
    merged_graph.bind("rdf", Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#"))
    merged_graph.bind("ex", Namespace("http://example.org/"))
    merged_graph.bind("jacamo", Namespace("https://purl.org/hmas/jacamo/"))
    merged_graph.bind("wotsec", Namespace("https://www.w3.org/2019/wot/security#"))
    merged_graph.bind("websub", Namespace("https://purl.org/hmas/websub/"))

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

