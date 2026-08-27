import asyncio
from typing import Any, Dict, Optional

from ....shared.utils.demo_log import demo
from ..utils.matching_utils import workspace_match


async def ensure_experience_engine_ready(agent_instance) -> bool:
    """
    Ensure Experience engine is ready for the agent.

    Args:
        agent_instance: The EnvExplorerAgent instance

    Returns:
        True if Experience engine is ready, False if initialization failed
    """
    if agent_instance._experience_engine_ready:
        return True

    async with agent_instance._experience_engine_lock:
        if agent_instance._experience_engine_ready:
            return True

        try:
            from ....shared import memory as experience_engine_memory

            experience_engine_memory.ensure_engine_on_path()

            from src.matching.registry import IntentMatcherRegistry  # type: ignore[import-not-found]
            from src.storage.registry import SignifierRegistry  # type: ignore[import-not-found]
            from src.validation.context_builder import ContextGraphBuilder  # type: ignore[import-not-found]
            from src.validation.shacl_validator import SHACLValidator  # type: ignore[import-not-found]

            cfg = get_experience_engine_config(agent_instance.config)

            storage_dir = cfg.get("storage_dir")
            if not storage_dir:
                storage_dir = str(experience_engine_memory.get_default_storage_dir())

            enable_authoring_validation = bool(cfg.get("enable_authoring_validation", False))

            agent_instance._experience_engine_storage_dir = str(storage_dir)
            agent_instance._experience_engine_registry = SignifierRegistry(
                storage_dir=agent_instance._experience_engine_storage_dir,
                enable_authoring_validation=enable_authoring_validation,
            )

            # Use configured default matcher version from agent config
            preferred_version = str(agent_instance._experience_engine_default_matcher_version or "v0")
            matcher_registry = IntentMatcherRegistry(default_version=preferred_version)
            agent_instance._experience_engine_matcher_registry = matcher_registry
            agent_instance._experience_engine_default_matcher_version = matcher_registry.get_default_version()

            try:
                configured_min_similarity = cfg.get(
                    "min_similarity",
                    cfg.get(
                        "default_min_similarity",
                        getattr(agent_instance, "_experience_engine_default_min_similarity", 0.5),
                    ),
                )
                agent_instance._experience_engine_default_min_similarity = float(configured_min_similarity)
            except Exception:
                agent_instance._experience_engine_default_min_similarity = float(
                    getattr(agent_instance, "_experience_engine_default_min_similarity", 0.5)
                )

            agent_instance._experience_engine_context_builder = ContextGraphBuilder()
            agent_instance._experience_engine_shacl_validator = SHACLValidator(enable_caching=bool(cfg.get("enable_shacl_cache", False)))

            agent_instance._experience_engine_ready = True
            agent_instance.logger.info(
                demo(f"Experience engine ready (storage_dir={agent_instance._experience_engine_storage_dir}, matcher_default={agent_instance._experience_engine_default_matcher_version})")
            )
            return True

        except Exception as e:
            agent_instance.logger.error(f"Failed to initialize Experience engine: {e}", exc_info=True)
            return False


def get_experience_engine_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get experience engine configuration with defaults.

    Args:
        config: Agent configuration dictionary

    Returns:
        Experience engine configuration dictionary
    """
    return (config or {}).get("experience_engine", {}) or {}


def build_experience_engine_context_snapshot(agent_instance, workspace_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Build Experience engine context snapshot from agent's current state.

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


async def list_experience_engine_signifiers(agent_instance, experience_engine_defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all stored signifiers.

    Args:
        agent_instance: The EnvExplorerAgent instance
        experience_engine_defaults: Default Experience engine configuration values

    Returns:
        Dictionary containing signifiers list or error information
    """
    if not agent_instance._experience_engine_ready:
        await ensure_experience_engine_ready(agent_instance)

    if not agent_instance._experience_engine_registry:
        return {"ok": False, "error": "experience_engine_not_ready"}

    try:
        signifier_limit_raw = agent_instance.config.get("experience_engine", {}).get("signifier_limit", experience_engine_defaults["signifier_limit"])
        signifier_limit = int(signifier_limit_raw)
        signifiers_raw = agent_instance._experience_engine_registry.list_signifiers(
            limit=signifier_limit
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
