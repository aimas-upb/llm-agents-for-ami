"""Behaviour: timeout handler for community queries."""

from spade.behaviour import OneShotBehaviour

from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory


class CommunityTimeoutBehaviour(OneShotBehaviour):
    """Trigger planning continuation after community timeout."""

    def __init__(self, goal_status, logger=None) -> None:
        super().__init__()
        self.goal_status = goal_status
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")

    async def run(self) -> None:
        goal_status = self.goal_status
        goal_id = goal_status.goal_id

        if goal_status.continue_triggered:
            self.logger.debug(
                demo("Community timeout skipped; continuation already triggered for goal_id=%s"),
                goal_id,
            )
            return

        self.logger.info(
            demo(
                "Community timeout: goal_id=%s (received %d/%d responses)"
            ),
            goal_id,
            len(goal_status.community_responses),
            goal_status.community_expected_responses,
        )

        goal_status.continue_triggered = True
        goal_status.community_event.set()
