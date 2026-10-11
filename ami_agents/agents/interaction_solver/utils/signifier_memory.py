"""Recalling a remembered goal execution, where an LLM would otherwise be asked.

Memory only holds goals an LLM had to plan (and every ambiguous goal), keyed
exactly by their structure (`shared/utils/goal_signifiers.py`). The planning
paths ask it just before they would call a model:

    explicit / incomplete   a hit is the whole stored plan, replayed as it is
    ambiguous               a hit is hints for the planner (what that plan used)

    recall(agent, goal)   the EnvExplorer's memory first; on a miss the
                          community memory, when configured; else None
    replay_entry(...)     a stored plan as a plan entry

Never raises: memory is a shortcut, and an unreachable one just means planning
as usual.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ....shared.models.goal_structure import GoalSpec
from ....shared.models.messages import MessageType
from ....shared.utils.demo_log import demo
from ....shared.utils.goal_signifiers import signifier_key

# Marks plan entries replayed from memory: not recorded again, nothing was learned.
SOURCE = "memory"


async def recall(agent, goal: GoalSpec, logger) -> Optional[Dict[str, Any]]:
    """The record remembered under the goal's key, or None."""
    sk = signifier_key(goal)
    if sk is None:
        return None

    # 1. The local memory, hosted by the EnvExplorer.
    try:
        body = await agent._query_env_explorer(
            message_type=MessageType.SIGNIFIER_MATCH_REQUEST.value,
            body=sk.to_wire_dict(),
            expect_type=MessageType.SIGNIFIER_MATCH_RESPONSE.value,
        )
        response = json.loads(body) if body else {}
        match = response.get("match") if isinstance(response, dict) else None
    except Exception as exc:  # noqa: BLE001 - memory is optional
        logger.warning(f"Memory recall failed for {goal.intent_text!r}: {exc}")
        match = None
    if isinstance(match, dict):
        logger.info(demo(f"[MEMORY] {sk.kind} hit for {goal.intent_text!r}: "
                         f"{match.get('signifier_id')}"))
        return match

    # 2. The community memory, on a local miss.
    client = getattr(agent, "community_client", None)
    if client is not None:
        try:
            match = await client.match_signifiers(sk.kind, sk.key)
        except Exception as exc:  # noqa: BLE001 - memory is optional
            logger.warning(f"Community recall failed for {goal.intent_text!r}: {exc}")
            match = None
        if isinstance(match, dict):
            logger.info(demo(f"[MEMORY] {sk.kind} community hit for {goal.intent_text!r}: "
                             f"{match.get('signifier_id')}"))
            return match
    return None


def replay_entry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A stored plan as a plan entry, or None when the record holds none."""
    tree = record.get("tree")
    if not isinstance(tree, dict) or not tree:
        return None
    return {"tree": tree, "source": SOURCE, "signifier_id": record.get("signifier_id"),
            "explanation": f"Replaying the plan remembered for "
                           f"{record.get('intent_text')!r} ({record.get('signifier_id')})."}
