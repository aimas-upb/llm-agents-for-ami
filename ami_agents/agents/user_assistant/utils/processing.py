"""Request processing limits shared by the live UA and both benchmark backends.

One agent owns a structuring semaphore, so two simultaneous conversations also
share its configured limit. The caller supplies how a behaviour starts/joins:
SPADE uses add_behaviour/join; the in-process harness awaits run directly.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping

DEFAULT_SEGMENTATION_ATTEMPTS = 2
DEFAULT_MAX_INTENTS = 16
DEFAULT_STRUCTURING_CONCURRENCY = 2


def processing_limit(config, key: str, default: int) -> int:
    """Read a positive processing limit from an agent config, with a checked default."""
    settings = config.get("processing", {}) if isinstance(config, Mapping) else {}
    value = settings.get(key, default) if isinstance(settings, Mapping) else default
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"processing.{key} must be a positive integer") from exc
    if isinstance(value, bool) or limit < 1 or (isinstance(value, float) and value != limit):
        raise ValueError(f"processing.{key} must be a positive integer")
    return limit


async def run_structuring(agent, items, run_one):
    """Return results in input order with bounded work shared across this agent's requests."""
    # 1. Reuse the same pool across conversations; harness agents are episode-local.
    pool = vars(agent).get("_structuring_pool")
    if pool is None:
        limit = processing_limit(getattr(agent, "config", {}), "structuring_concurrency",
                                 DEFAULT_STRUCTURING_CONCURRENCY)
        pool = agent._structuring_pool = asyncio.Semaphore(limit)

    # 2. Acquire before spawning a behaviour or starting its HTTP/model request.
    async def bounded(item):
        """Release capacity on success, exception, or cancellation."""
        async with pool:
            return await run_one(item)

    return await asyncio.gather(*(bounded(item) for item in items))
