"""
RD4 engine utilities for EnvExplorer agent.
"""

import asyncio
from typing import Any, Dict, Optional

from ....shared.utils.demo_log import demo
from .matching_utils import workspace_match


async def ensure_rd4_ready(agent_instance) -> bool:
    """
    Ensure RD4 engine is ready for the agent.

    Args:
        agent_instance: The EnvExplorerAgent instance

    Returns:
        True if RD4 engine is ready, False if initialization failed
    """
    if agent_instance._rd4_engine_ready:
        return True

    async with agent_instance._rd4_engine_lock:
        if agent_instance._rd4_engine_ready:
            return True

        try:
            from ....shared import memory as rd4_memory

            rd4_memory.ensure_engine_on_path()

            from src.matching.registry import IntentMatcherRegistry  # type: ignore[import-not-found]
            from src.storage.registry import SignifierRegistry  # type: ignore[import-not-found]
            from src.validation.context_builder import ContextGraphBuilder  # type: ignore[import-not-found]
            from src.validation.shacl_validator import SHACLValidator  # type: ignore[import-not-found]

            cfg = get_signifier_config(agent_instance.config)

            storage_dir = cfg.get("storage_dir")
            if not storage_dir:
                storage_dir = str(rd4_memory.get_default_storage_dir())

            enable_authoring_validation = bool(cfg.get("enable_authoring_validation", False))

            agent_instance._rd4_storage_dir = str(storage_dir)
            agent_instance._rd4_registry = SignifierRegistry(
                storage_dir=agent_instance._rd4_storage_dir,
                enable_authoring_validation=enable_authoring_validation,
            )

            # Default to v0 to avoid downloading embedding models unexpectedly.
            matcher_registry = IntentMatcherRegistry(default_version="v0")
            preferred_version = str(cfg.get("matcher_version") or "v0")
            if preferred_version in matcher_registry.list_versions():
                matcher_registry.set_default_version(preferred_version)
            agent_instance._rd4_matcher_registry = matcher_registry
            agent_instance._rd4_default_matcher_version = matcher_registry.get_default_version()

            try:
                agent_instance._rd4_default_min_similarity = float(cfg.get("min_similarity", 0.0))
            except Exception:
                agent_instance._rd4_default_min_similarity = 0.0

            agent_instance._rd4_context_builder = ContextGraphBuilder()
            agent_instance._rd4_shacl_validator = SHACLValidator(enable_caching=bool(cfg.get("enable_shacl_cache", False)))

            agent_instance._rd4_engine_ready = True
            agent_instance.logger.info(
                demo("RD4 signifier engine ready (storage_dir=%s, matcher_default=%s)"),
                agent_instance._rd4_storage_dir,
                agent_instance._rd4_default_matcher_version,
            )
            return True

        except Exception as e:
            agent_instance.logger.error(f"Failed to initialize RD4 engine: {e}", exc_info=True)
            return False


def get_signifier_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get signifier configuration with defaults.

    Args:
        config: Agent configuration dictionary

    Returns:
        Signifier configuration dictionary
    """
    return (config or {}).get("signifiers", {}) or {}


def build_rd4_context_snapshot(agent_instance, workspace_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Build RD4 context snapshot from agent's current state.

    Args:
        agent_instance: The EnvExplorerAgent instance
        workspace_id: Optional workspace ID to filter artifacts

    Returns:
        Context dictionary mapping artifact URIs to their current state
    """
    context: Dict[str, Dict[str, Any]] = {}

    for artifact in (agent_instance.artifacts or {}).values():
        if workspace_id and not workspace_match(getattr(artifact, "workspace_id", None), workspace_id):
            continue

        artifact_uri = str(getattr(artifact, "artifact_id", "") or "")
        if not artifact_uri:
            continue

        state = getattr(artifact, "current_state", {}) or {}
        if not isinstance(state, dict):
            continue

        # ContextGraphBuilder expects: {artifact_uri: {property_uri: value}}
        context[artifact_uri] = {str(k): v for k, v in state.items()}

    return context


async def list_rd4_signifiers(agent_instance, rd4_defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all stored signifiers.

    Args:
        agent_instance: The EnvExplorerAgent instance
        rd4_defaults: Default RD4 configuration values

    Returns:
        Dictionary containing signifiers list or error information
    """
    if not agent_instance._rd4_engine_ready:
        await ensure_rd4_ready(agent_instance)

    if not agent_instance._rd4_registry:
        return {"ok": False, "error": "rd4_engine_not_ready"}

    try:
        signifiers_raw = agent_instance._rd4_registry.list_signifiers(
            limit=agent_instance.config.get("rd4_engine", {}).get("signifier_limit", rd4_defaults["signifier_limit"])
        )
        signifiers_out = []
        for s in signifiers_raw:
            signifiers_out.append({
                "signifier_id": s.signifier_id,
                "version": s.version,
                "status": getattr(s.status, "value", str(s.status)),
                "intent": getattr(s.intent, "nl_text", ""),
                "affordance_uri": s.affordance_uri,
            })
        return {"total": len(signifiers_out), "signifiers": signifiers_out}
    except Exception as e:
        agent_instance.logger.error(f"Error listing signifiers: {e}", exc_info=True)
        return {"ok": False, "error": "exception", "detail": str(e)}