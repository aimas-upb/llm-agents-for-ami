"""Behaviour: ask the community signifier API for cross-environment matches.

Per Alex's chat: "asking for community experience is a behavior". Spawned
per GOAL_REQUEST when ``self.agent.community_client`` is configured.
"""

import asyncio
from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.utils.logger import LoggerFactory
from ..utils.signifier_matching import query_community_signifier_match


class CommunitySignifierQueryBehaviour(OneShotBehaviour):
    """Per-intent community API queries (HTTP), parallel."""

    def __init__(self, intent_strings: List[str], logger=None) -> None:
        super().__init__()
        self.intent_strings = intent_strings
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        # Populated by ``run``: maps intent query-string -> community payload.
        self.matches: Dict[str, Any] = {}

    async def run(self) -> None:
        if not self.intent_strings:
            return
        client = getattr(self.agent, "community_client", None)
        if not client:
            return

        results = await asyncio.gather(
            *[query_community_signifier_match(client, i) for i in self.intent_strings],
            return_exceptions=True,
        )
        for intent_str, res in zip(self.intent_strings, results):
            if isinstance(res, Exception):
                self.matches[intent_str] = {}
            else:
                self.matches[intent_str] = res if isinstance(res, dict) else {}
