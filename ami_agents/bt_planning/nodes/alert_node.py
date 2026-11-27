"""The `alert` action: a plan tells its user something, and acts on no device.

"Alert me whenever the grid import goes above 3 kW" has a goal -- the user
knows -- and an action that achieves it: an alert. The action is not a device
affordance; the UserAssistant performs it. So the node only records that the
alert was raised, and the UA, which runs the tree, takes it from there:

    IR       {"type": "alert", "name": "G1: alert", "text": "Grid import above 3 kW"}
    tree     AlertNode -- each time the tree enters it, one alert joins `pending`
    UA       after every tick, PlanExecutionBehaviour drains `pending` of every
             AlertNode and spawns one AlertBehaviour per alert
             (`agents/user_assistant/behaviours/alert.py`): stderr, then hooks

Inside a `repeat` ("whenever"), every pass enters the node again and raises a
new alert. The alerts wait on the node, not on the py_trees blackboard: the
blackboard is one for the whole process, and the UA runs several plans at once,
so a shared key would hand one plan's alerts to another.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

import py_trees
from py_trees.common import Status


# -- the alert ----------------------------------------------------------------------

@dataclass(frozen=True)
class Alert:
    """One raised alert: what it says, and when (UTC) the tree raised it."""

    text: str
    raised_at: datetime


# -- the node -----------------------------------------------------------------------

class AlertNode(py_trees.behaviour.Behaviour):
    """Raise one alert each time the tree enters this node; always SUCCESS.

    It exists so a plan can tell its user something as a step like any other:
    after a trigger, inside a loop, between device actions.
    """

    def __init__(self, name: str, text: str) -> None:
        """An alert node saying `text`; nothing raised yet."""
        super().__init__(name=name)
        self.text = text
        # Raised and not yet taken by the UA (`drain`).
        self.pending: List[Alert] = []

    def initialise(self) -> None:
        """Entered afresh (first tick, or a new pass of a loop): raise it.

        `initialise` runs once per entry, not once per tick, so a node the
        tree stays on never raises the same alert twice.
        """
        self.pending.append(Alert(self.text, datetime.now(timezone.utc)))

    def update(self) -> Status:
        """Raising is all the node does; the UA delivers it."""
        return Status.SUCCESS

    def drain(self) -> List[Alert]:
        """The alerts raised since the last drain; empties the queue.

        The UA calls it after every tick, so each alert is delivered once.
        """
        alerts, self.pending = self.pending, []
        return alerts


# -- IR compile / validate (registered in registry.py) -------------------------------

def compile_alert(spec, compile_child) -> AlertNode:
    """`{"type": "alert", "text": ...}` -> AlertNode; the name defaults to the text."""
    return AlertNode(spec.get("name") or f"alert: {spec['text']}", str(spec["text"]))


def validate_alert(spec, path, validate_child) -> List[str]:
    """An alert must say something: a non-empty `text`."""
    text = spec.get("text")
    if not isinstance(text, str) or not text.strip():
        return [f"{path}: alert nodes require a non-empty 'text'"]
    return []
