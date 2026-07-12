"""
Data formatting utilities for EnvExplorer agent.
"""

import re
from typing import Any, Dict, List

from ....shared.models.environment import AffordanceType


def _short_iri(value: Any) -> str:
    """Return the last path segment of an IRI (without a #fragment)."""
    text = str(value or "").split("#")[0].rstrip("/")
    return text.rsplit("/", 1)[-1] if text else ""


def _normalized(name: str) -> str:
    """Normalize a display name the way artifact ids are derived from it."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _short_type(param_type: Any) -> str:
    """Shorten schema type IRIs like ``...json-schema#NumberSchema`` to ``number``."""
    text = str(param_type or "")
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    if text.endswith("Schema"):
        text = text[: -len("Schema")]
    return text.lower()


def _is_generic_description(description: Any, name: str) -> bool:
    """True for empty or auto-generated descriptions that add no information."""
    d = str(description or "").strip().lower()
    n = name.lower()
    return d in {"", n, f"property affordance: {n}", f"action affordance: {n}"}


def _schema_param_summary(schema: Any) -> str:
    """Summarise a JSON Schema as compact ``name:type*`` parameter labels."""
    if not isinstance(schema, dict):
        return ""
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return ""
    required = set(schema.get("required") or [])
    parts: List[str] = []
    for key, sub in props.items():
        param_type = _short_type(sub.get("type")) if isinstance(sub, dict) else ""
        label = f"{key}:{param_type}" if param_type else str(key)
        if key in required:
            label += "*"
        parts.append(label)
    return ", ".join(parts)


def format_capabilities_summary(agent_instance) -> str:
    """
    Compact, LLM-friendly listing of workspaces, artifacts, and affordances.

    Contains names, descriptions, and parameter names only (no URLs or full
    JSON Schemas) to keep LLM prompts small.

    Args:
        agent_instance: The EnvExplorerAgent instance

    Returns:
        Compact string summary of environment capabilities
    """
    if not agent_instance.discovery_complete:
        return "Environment discovery is still in progress. Please try again later."

    artifacts = list(agent_instance.artifacts.values())
    if not artifacts:
        return "No artifacts found in the environment."

    lines: List[str] = []

    workspaces = list((agent_instance.environment_map or {}).values())
    multiple_workspaces = len(workspaces) > 1
    if workspaces:
        ws_parts = []
        for ws in workspaces:
            label = _short_iri(ws.workspace_id) or str(ws.workspace_id)
            name = getattr(ws, "name", None)
            if name and name != label:
                label = f"{label} ({name})"
            ws_parts.append(label)
        lines.append("Workspaces: " + ", ".join(ws_parts))
        lines.append("")

    # First pass: gather affordances per artifact and factor out property
    # names shared by every artifact (large environments repeat the same
    # generic properties hundreds of times).
    entries = []
    for artifact in artifacts:
        affs = agent_instance.integration_engine.get_affordances_for_artifact(artifact.artifact_id)
        actions = [a for a in affs if a.affordance_type == AffordanceType.ACTION]
        properties = [a for a in affs if a.affordance_type == AffordanceType.PROPERTY]
        if not actions and not properties:
            continue
        entries.append((artifact, actions, [p.name for p in properties]))

    prop_sets = [set(prop_names) for _, _, prop_names in entries if prop_names]
    common_props = set.intersection(*prop_sets) if len(prop_sets) > 1 else set()

    if common_props:
        lines.append(
            "Common properties (every artifact below also has these): "
            + ", ".join(sorted(common_props))
        )
        lines.append("")

    def _artifact_label(artifact) -> str:
        short_id = _short_iri(artifact.artifact_id) or str(artifact.artifact_id)
        name = str(artifact.name or "")
        # Skip the name when the id is just its normalized form.
        label = short_id if _normalized(name) == short_id else f"{name} (id: {short_id})"
        if multiple_workspaces:
            workspace_id = getattr(artifact, "workspace_id", None)
            if workspace_id:
                label += f" [workspace: {_short_iri(workspace_id) or workspace_id}]"
        return label

    for artifact, actions, prop_names in entries:
        if not actions:
            continue
        lines.append(f"Artifact: {_artifact_label(artifact)}")
        lines.append("  Actions:")
        for action in actions:
            entry = f"    - {action.name}"
            params = _schema_param_summary(action.input_schema)
            if params:
                entry += f" (params: {params})"
            if not _is_generic_description(action.description, action.name):
                entry += f" -- {str(action.description).strip()}"
            lines.append(entry)

        extra_props = [p for p in prop_names if p not in common_props]
        if extra_props:
            lines.append("  Properties: " + ", ".join(extra_props))
        lines.append("")

    property_only = [(artifact, prop_names) for artifact, actions, prop_names in entries if not actions]
    if property_only:
        lines.append("Property-only artifacts (id: extra properties):")
        for artifact, prop_names in property_only:
            extra_props = [p for p in prop_names if p not in common_props]
            entry = f"- {_artifact_label(artifact)}"
            if extra_props:
                entry += ": " + ", ".join(extra_props)
            lines.append(entry)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_capabilities_payload(agent_instance) -> Dict[str, Any]:
    """
    Format capabilities for machine consumption.

    Machine-readable capabilities payload for other agents (planning, etc.).
    Includes a compact LLM-friendly 'summary' field for prompt injection.

    The ``affordances`` list contains both action and property affordances;
    property affordances carry the readable target URL for condition nodes.

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
        property_affordances = [a for a in affs if a.affordance_type == AffordanceType.PROPERTY]

        for action in action_affordances:
            affordances_out.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_name": artifact.name,
                    "workspace_id": getattr(artifact, "workspace_id", None),
                    "affordance_id": action.affordance_id,
                    "affordance_type": action.affordance_type.value,
                    "action_name": action.name,
                    "name": action.name,
                    "description": action.description,
                    "method": getattr(action.form, "method", None) if action.form else None,
                    "target": getattr(action.form, "href", None) if action.form else None,
                    "content_type": getattr(action.form, "content_type", None) if action.form else None,
                    "input_schema": action.input_schema,
                }
            )

        for prop in property_affordances:
            affordances_out.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_name": artifact.name,
                    "workspace_id": getattr(artifact, "workspace_id", None),
                    "affordance_id": prop.affordance_id,
                    "affordance_type": prop.affordance_type.value,
                    "name": prop.name,
                    "description": prop.description,
                    "method": (getattr(prop.form, "method", None) if prop.form else None) or "GET",
                    "target": getattr(prop.form, "href", None) if prop.form else None,
                    "content_type": getattr(prop.form, "content_type", None) if prop.form else None,
                    "output_schema": prop.output_schema,
                }
            )

        artifacts_out.append(
            {
                "artifact_id": artifact.artifact_id,
                "name": artifact.name,
                "workspace_id": getattr(artifact, "workspace_id", None),
                "actions": [a.name for a in action_affordances],
                "properties": [p.name for p in property_affordances],
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
