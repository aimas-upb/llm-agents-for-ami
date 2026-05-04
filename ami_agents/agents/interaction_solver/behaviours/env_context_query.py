"""Behaviour: query EnvExplorer for environment capabilities + state.

This is one of the three sub-behaviours Alex called out for the InteractionSolver:
"querying and receiving an answer from the EnvExplorer is a behavior".

A new instance is spawned per GOAL_REQUEST. Result is published on
``self.result`` and the spawned task can be awaited via ``behaviour.join()``.
"""

import asyncio
from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.models.messages import MessageType
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..utils.planning_context import match_workspace, parse_or_empty


class EnvContextQueryBehaviour(OneShotBehaviour):
    """Fetch capabilities + state from EnvExplorer in parallel and scope to workspace."""

    def __init__(
        self,
        workspace_id: Optional[str] = None,
        logger=None,
    ) -> None:
        super().__init__()
        self.workspace_id = workspace_id
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        # Populated by ``run``:
        self.affordances: List[Dict[str, Any]] = []
        self.state_payload: Dict[str, Any] = {}
        self.error: Optional[str] = None

    async def run(self) -> None:
        try:
            raw_aff, raw_state = await asyncio.gather(
                self.agent._query_env_explorer(
                    message_type=MessageType.ENV_CAPABILITIES_REQUEST.value,
                    body={"query": "all"},
                    expect_type=MessageType.ENV_CAPABILITIES_RESPONSE.value,
                ),
                self.agent._query_env_explorer(
                    message_type=MessageType.ENV_STATE_REQUEST.value,
                    body={},
                    expect_type=MessageType.ENV_STATE_RESPONSE.value,
                ),
            )
        except Exception as e:
            self.error = str(e)
            self.logger.warning(f"EnvContextQuery failed: {e}")
            return

        aff_payload = parse_or_empty(raw_aff)
        state_payload = parse_or_empty(raw_state)

        affordances = aff_payload.get("affordances") or aff_payload.get("capabilities") or []
        if not isinstance(affordances, list):
            affordances = []

        if isinstance(state_payload, dict):
            artifacts = state_payload.get("artifacts")
            artifacts_count = len(artifacts) if isinstance(artifacts, dict) else 0
        else:
            state_payload = {}
            artifacts_count = 0

        self.logger.info(
            demo("Context snapshot: affordances=%s artifacts_in_state=%s"),
            len(affordances),
            artifacts_count,
        )

        if self.workspace_id:
            affordances = [
                a
                for a in affordances
                if isinstance(a, dict)
                and match_workspace(a.get("workspace_id"), self.workspace_id)
            ]
            if isinstance(state_payload.get("artifacts"), dict):
                filtered: Dict[str, Any] = {}
                for aid, ainfo in state_payload["artifacts"].items():
                    if isinstance(ainfo, dict) and match_workspace(
                        ainfo.get("workspace_id"), self.workspace_id
                    ):
                        filtered[aid] = ainfo
                state_payload = {**state_payload, "artifacts": filtered}

            self.logger.info(
                demo(
                    "Workspace scoping applied: affordances=%s artifacts_in_state=%s "
                    "workspace_id=%r"
                ),
                len(affordances),
                len(state_payload.get("artifacts") or {})
                if isinstance(state_payload, dict)
                else "?",
                self.workspace_id,
            )

        self.affordances = affordances
        self.state_payload = state_payload

        # Cache on the agent so future signifier-only fast-paths can skip this round-trip.
        try:
            self.agent._cached_affordances = affordances
            self.agent._cached_state_payload = state_payload
            self.agent._cached_affordance_by_id = {
                str(a.get("affordance_id") or "").strip(): a
                for a in affordances
                if isinstance(a, dict) and str(a.get("affordance_id") or "").strip()
            }
        except Exception:
            pass
