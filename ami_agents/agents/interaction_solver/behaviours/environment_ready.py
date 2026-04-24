"""Behaviour that waits for EnvExplorer's ENV_DISCOVERY_COMPLETE signal."""

import json
import logging
from typing import Any, Dict

from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType

logger = logging.getLogger("InteractionSolver")


class EnvironmentReadyBehaviour(CyclicBehaviour):
    """One-shot behaviour: marks the agent ready on ENV_DISCOVERY_COMPLETE."""

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return

        if msg.get_metadata("type") != MessageType.ENV_DISCOVERY_COMPLETE.value:
            return

        payload: Dict[str, Any] = {}
        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {"raw": msg.body or ""}

        self.agent.mark_environment_ready(sender_jid=str(msg.sender), payload=payload)
        logger.info(f"Environment ready (notified by {msg.sender}).")
        self.kill()
