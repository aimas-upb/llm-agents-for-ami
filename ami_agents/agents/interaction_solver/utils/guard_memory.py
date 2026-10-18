"""Recalling a remembered condition-guard decision, before the model is asked.

The guard memory (`shared/memory/src/storage/guard_store.py`, hosted by the
EnvExplorer) holds one decision per exact key: the action, its device, the
device's room class, and -- for a command its value selects -- that value
(`planning_conditions.guard_key`). It is asked just before the condition-guard
model would be:

    recall_guard(agent, key, logger)   the record under exactly ``key``, or None

The record is not applied as it is: `condition_guard.apply_remembered`
resolves it again on today's TD graph, and `drop_satisfied` decides whether
this plan needs it at all.

Never raises: memory is a shortcut, and an unreachable one just means asking
the model.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ....shared.models.messages import MessageType
from ....shared.utils.demo_log import demo


async def recall_guard(agent, key: Dict[str, Any], logger) -> Optional[Dict[str, Any]]:
    """The decision remembered under ``key``, or None (a miss, or no memory)."""
    # 1. Ask the EnvExplorer's guard store.
    try:
        body = await agent._query_env_explorer(
            message_type=MessageType.GUARD_MATCH_REQUEST.value,
            body={"key": key},
            expect_type=MessageType.GUARD_MATCH_RESPONSE.value,
        )
        response = json.loads(body) if body else {}
        match = response.get("match") if isinstance(response, dict) else None
    except Exception as exc:  # noqa: BLE001 - memory is optional
        logger.warning(f"Guard recall failed for {key.get('device_name')}."
                       f"{key.get('affordance_name')}: {exc}")
        return None

    # 2. A hit is logged; anything that is not a record is a miss.
    if not isinstance(match, dict):
        return None
    logger.info(demo(f"[MEMORY] guard hit for {key.get('device_name')}."
                     f"{key.get('affordance_name')}: {match.get('guard_id')}"))
    return match
