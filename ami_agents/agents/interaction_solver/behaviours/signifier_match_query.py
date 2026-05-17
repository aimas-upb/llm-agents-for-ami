"""Behaviour: query EnvExplorer's signifier engine for matches per intent.

Part of the planning workflow spawned per GOAL_REQUEST. Carries the
``intent_type`` (EXPLICIT / IMPLICIT) and the structured intent payload
so EnvExplorer's matcher can apply the right context-validation rules
for the EMAS demo (Alex's "semantic modelling of explicit requests").
"""

import asyncio
from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.utils.logger import LoggerFactory
from ...user_assistant.models import Intent
from ..utils.signifier_matching import query_local_signifier_match


class SignifierMatchQueryBehaviour(OneShotBehaviour):
    """Per-intent SIGNIFIER_MATCH_REQUEST RPCs to EnvExplorer (parallel)."""

    def __init__(
        self,
        intents: List[Intent],
        workspace_id: Optional[str] = None,
        intent_type: Optional[str] = None,
        logger=None,
    ) -> None:
        super().__init__()
        self.intents = intents
        self.workspace_id = workspace_id
        self.intent_type = intent_type
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        # Populated by ``run``: maps intent query-string -> match payload.
        self.matches: Dict[str, Any] = {}

    async def run(self) -> None:
        if not self.intents:
            return
        if not self.agent.target_jids.get("explorer"):
            return

        intent_strings = [i.to_query_string() for i in self.intents]
        structured_by_intent = {i.to_query_string(): i.to_dict() for i in self.intents}

        results = await asyncio.gather(
            *[
                query_local_signifier_match(
                    self.agent._query_env_explorer,
                    intent_str,
                    self.workspace_id,
                    self.intent_type,
                    structured_by_intent.get(intent_str),
                )
                for intent_str in intent_strings
            ],
            return_exceptions=True,
        )

        for intent_str, res in zip(intent_strings, results):
            if isinstance(res, Exception):
                self.matches[intent_str] = {
                    "error": "signifier_query_failed",
                    "detail": str(res),
                }
            else:
                self.matches[intent_str] = res
