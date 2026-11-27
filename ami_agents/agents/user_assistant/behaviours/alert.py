"""Delivering one alert a running plan raised.

A plan's `alert` node (`bt_planning/nodes/alert_node.py`) only records that an
alert was raised. After every tick, the plan's behaviour drains those records
and spawns one `AlertBehaviour` per alert -- delivering an alert is a discrete
task, independent of the plan's next tick. For example, the plan for "alert me
whenever the grid import goes above 3 kW" ticks past its alert node:

    PlanExecutionBehaviour._tick   AlertNode.drain() -> [Alert("Grid import above 3 kW")]
    AlertBehaviour.run             1. stderr: "ALERT plan=p-12 Grid import above 3 kW"
                                   2. act(): every handler in agent.alert_handlers

Step 1 always happens, so an alert is never lost even when nobody hooked in.
Step 2 is the hook: what else an alert does (a chat notice, a push message, a
test's collector) is a handler the agent is given, or a subclass's `act`.
"""

from __future__ import annotations

import inspect
import logging
import sys
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional, Union

from spade.behaviour import OneShotBehaviour

from ....bt_planning.nodes.alert_node import Alert
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory


# -- the default action: stderr -----------------------------------------------------

ALERT_LOGGER_NAME = "ami_agents.alerts"

# A handler gets the plan, the conversation and the alert; sync or async.
AlertHandler = Callable[["AlertBehaviour", Alert], Union[None, Awaitable[None]]]


def alert_logger() -> logging.Logger:
    """The logger every alert is written to, with its own stderr handler.

    Its own handler, not the agents' configured ones: the agents may log to a
    file only, and an alert is the one line that must reach a person watching
    the console.
    """
    logger = logging.getLogger(ALERT_LOGGER_NAME)
    # 1. One stderr handler, however often this is called.
    if not any(getattr(h, "_ami_alert_stderr", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        handler._ami_alert_stderr = True
        logger.addHandler(handler)
    # 2. Alerts are always shown, and not again through the root logger.
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# -- the behaviour --------------------------------------------------------------------

class AlertBehaviour(OneShotBehaviour):
    """Deliver one alert: write it to stderr, then run the hook (`act`)."""

    def __init__(self, plan_id: str, conversation_id: Optional[str], alert: Alert,
                 logger=None) -> None:
        """One alert of plan `plan_id`, on its conversation (None when unknown)."""
        super().__init__()
        self.plan_id = plan_id
        self.conversation_id = conversation_id
        self.alert = alert
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")

    @property
    def text(self) -> str:
        """What the alert says."""
        return self.alert.text

    @property
    def raised_at(self) -> datetime:
        """When the tree raised it (UTC)."""
        return self.alert.raised_at

    async def run(self) -> None:
        """1. stderr, always; 2. the hook, whose failures never reach the plan."""
        # 1. The default action: the alert on stderr (and flagged for the demo log).
        alert_logger().warning(demo(f"ALERT plan={self.plan_id} {self.alert.text}"))
        # 2. Whatever else an alert should do.
        try:
            await self.act(self.alert)
        except Exception as exc:  # noqa: BLE001 - an alert hook must not stop anything
            self.logger.error(f"Alert hook failed for plan {self.plan_id}: {exc}")

    async def act(self, alert: Alert) -> None:
        """The hook: await every handler in the agent's `alert_handlers`.

        A subclass may override it to do something else. Each handler is run
        on its own, so one failing handler does not keep the others from the
        alert; its failure is logged.
        """
        for handler in list(getattr(self.agent, "alert_handlers", None) or []):
            try:
                result: Any = handler(self, alert)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - see above
                self.logger.error(
                    f"Alert handler {getattr(handler, '__name__', handler)!r} "
                    f"failed for plan {self.plan_id}: {exc}")
