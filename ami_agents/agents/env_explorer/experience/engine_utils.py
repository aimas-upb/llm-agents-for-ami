"""The EnvExplorer's handle on the memory engine.

The engine (`ami_agents/shared/memory`) is a separate component -- also served
on its own as the community memory -- that this agent manages: it opens the
engine's `MemoryStore` once and keeps the SHACL tools ready alongside it. The
engine's `GuardStore` (condition-guard decisions) is opened with it, in the
same storage directory, under its own `guards/` subdirectory.

    ensure_experience_engine_ready(agent)   open the stores (idempotent, locked)
    list_experience_engine_signifiers(...)  the stored records, newest first

The SHACL validator and context builder are set up but not used for matching:
matching is by exact key. They are kept for exact value checks of an
ambiguous goal's stored context later.
"""

from typing import Any, Dict, Optional

from ....shared.utils.demo_log import demo

# Fallback for `experience_engine.signifier_limit` in agents.yaml.
DEFAULT_SIGNIFIER_LIMIT = 10000


async def ensure_experience_engine_ready(agent_instance) -> bool:
    """Open the memory store, the guard store and the SHACL tools, once. True
    when ready."""
    if agent_instance._experience_engine_ready:
        return True

    async with agent_instance._experience_engine_lock:
        if agent_instance._experience_engine_ready:
            return True
        try:
            from ....shared import memory as experience_engine_memory

            experience_engine_memory.ensure_engine_on_path()

            from src.storage.guard_store import GuardStore  # type: ignore[import-not-found]
            from src.storage.memory_store import MemoryStore  # type: ignore[import-not-found]
            from src.validation.context_builder import ContextGraphBuilder  # type: ignore[import-not-found]
            from src.validation.shacl_validator import SHACLValidator  # type: ignore[import-not-found]

            cfg = get_experience_engine_config(agent_instance.config)
            storage_dir = (cfg.get("storage_dir")
                           or str(experience_engine_memory.get_default_storage_dir()))

            agent_instance._experience_engine_storage_dir = str(storage_dir)
            agent_instance._experience_engine_store = MemoryStore(storage_dir)
            agent_instance._experience_engine_guard_store = GuardStore(storage_dir)
            agent_instance._experience_engine_context_builder = ContextGraphBuilder()
            agent_instance._experience_engine_shacl_validator = SHACLValidator(
                enable_caching=str(cfg.get("enable_shacl_cache", "false")).lower() == "true")

            agent_instance._experience_engine_ready = True
            agent_instance.logger.info(demo(
                f"Memory engine ready (storage_dir={storage_dir}, "
                f"records={len(agent_instance._experience_engine_store)}, "
                f"guards={len(agent_instance._experience_engine_guard_store)})"))
            return True
        except Exception as e:
            agent_instance.logger.error(f"Failed to initialize the memory engine: {e}",
                                        exc_info=True)
            return False


def get_experience_engine_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The `experience_engine` section of the agent's config."""
    return (config or {}).get("experience_engine", {}) or {}


async def list_experience_engine_signifiers(agent_instance) -> Dict[str, Any]:
    """The stored records, newest first, up to the configured limit."""
    if not await ensure_experience_engine_ready(agent_instance):
        return {"ok": False, "error": "experience_engine_not_ready"}
    try:
        limit = int(get_experience_engine_config(agent_instance.config).get(
            "signifier_limit", DEFAULT_SIGNIFIER_LIMIT))
        records = agent_instance._experience_engine_store.list(limit=limit)
        return {"total": len(records),
                "signifiers": [{"signifier_id": r.signifier_id, "kind": r.kind,
                                "intent_text": r.intent_text, "key": r.key,
                                "created_at": r.created_at.isoformat()}
                               for r in records]}
    except Exception as e:
        agent_instance.logger.error(f"Error listing signifiers: {e}", exc_info=True)
        return {"ok": False, "error": "exception", "detail": str(e)}
