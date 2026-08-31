"""
Running a plan, one tick per `run()`.

Execution used to happen inside the request handler, in a loop that ticked the
tree to completion. While it ran, the UA answered nothing: no second request, no
"what are you doing?", no "stop". And because the loop owned the tree, there was
no moment between ticks at which anything could intervene.

Here a plan is a `CyclicBehaviour` that ticks exactly once per `run()` and
returns. SPADE's scheduler interleaves it with everything else, so the agent
stays responsive while a plan runs, and control messages land *between* ticks --
never mid-tick, which is what makes cancellation safe.

Achievement and maintenance plans have genuinely different lifecycles, so they
are different behaviours:

- `AchievementPlanBehaviour` — finite. Ticks until SUCCESS or FAILURE, or until
  `max_ticks` runs out, and then it is over.
- `MaintenancePlanBehaviour` — persistent. "Keep the living room at 25°C" has no
  completion. It ticks in bursts and stands down between them, and only an
  explicit cancel ends it.

What they share -- compiling the tree, capturing the recording context,
answering status/cancel/explain, recording signifiers, finishing -- lives on
`PlanExecutionBehaviour`. What differs is `run()`, and nothing else.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import py_trees
from py_trees.common import Status
from spade.behaviour import CyclicBehaviour

from ....bt_planning.nodes.affordance_nodes import (
    ActionAffordanceNode,
    ComparisonPropertyConditionNode,
    PropertyAffordanceNode,
    PropertyConditionNode,
    SettlingTimeWaitNode,
)
from ....bt_planning.nodes.registry import compile_node
from ....shared.models.messages import (
    META_CORRELATION_ID,
    META_PLAN_ID,
    MessageType,
)
from ....shared.models.plan import PlanStatus
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..signifiers import (
    ExecutionOutcome,
    SignifierRecordingBehaviour,
    capture_execution_context,
)


class PlanExecutionBehaviour(CyclicBehaviour):
    """What every running plan does, whatever its lifecycle.

    Subclasses supply `run()`: how often to tick and what a finished tree
    means. Everything else -- setup, control messages, recording, finishing --
    is the same either way, and lives here.
    """

    def __init__(self, plan_id: str, tree_spec: Dict[str, Any],
                 max_ticks: int, tick_period: float, logger=None,
                 recording_options: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.plan_id = plan_id
        self.tree_spec = tree_spec
        self.max_ticks = max_ticks
        self.tick_period = tick_period
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")

        self.tree: Optional[py_trees.behaviour.Behaviour] = None
        self._cancelled = False
        self._cancel_requester = None
        self._setup_done = False

        # Signifier recording. This behaviour's whole involvement is to hold
        # the context it captured before the first tick and hand it back when
        # the plan ends -- it never inspects it.
        self._recording_options = dict(recording_options or {})
        self._context = None

    async def on_start(self) -> None:
        try:
            self.tree = compile_node(self.tree_spec, self._compile_child)
            self.tree.setup_with_descendants()
            self._setup_done = True
            self.agent.plans.mark_started(self.plan_id)

            # Before the first tick: a plan changes the very state that made it
            # appropriate, so the context has to be sampled now, not at the end.
            record = self.agent.plans.get(self.plan_id)
            self._context = await capture_execution_context(
                self.agent,
                plan_id=self.plan_id,
                tree_spec=self.tree_spec,
                request_id=record.request_id if record else None,
                thread=record.plan.conversation_id if record else "__default__",
                logger=self.logger,
                **self._recording_options,
            )
            self.logger.info(demo(
                f"Plan execution started: plan={self.plan_id}"))
        except Exception as exc:  # noqa: BLE001 - reported on the record
            self.logger.error(f"Plan {self.plan_id} failed to compile: {exc}")
            self.agent.plans.finish(self.plan_id, PlanStatus.FAILED,
                                    error=f"compile failed: {exc}")
            self.kill()

    def _compile_child(self, spec: Dict[str, Any]):
        return compile_node(spec, self._compile_child)

    def _finish(self, status: PlanStatus, error: Optional[str] = None) -> None:
        self.agent.plans.finish(self.plan_id, status, error=error)
        self.logger.info(demo(
            f"Plan {status.value}: plan={self.plan_id}"
            + (f" error={error}" if error else "")))
        if self.tree is not None and self._setup_done:
            self.tree.shutdown()
        self._record_signifiers(status)
        asyncio.ensure_future(self._announce(status))
        self.kill()

    def _record_signifiers(self, status: PlanStatus) -> None:
        """Hand the run to a recording behaviour of its own.

        Spawned rather than awaited: recording talks to EnvExplorer and the
        community service, and neither a slow reply nor a failure there should
        touch a plan that has already finished.
        """
        if self._context is None:
            return
        outcome = ExecutionOutcome(
            success=status is PlanStatus.COMPLETED,
            final_status=status.value,
            ticks=self._ticks_taken(),
            executed_actions=self._executed_actions(),
        )
        self.agent.add_behaviour(
            SignifierRecordingBehaviour(self._context, outcome,
                                        logger=self.logger))

    def _ticks_taken(self) -> int:
        record = self.agent.plans.get(self.plan_id)
        return record.ticks if record else 0

    def _executed_actions(self) -> List[tuple]:
        """Actions that actually reached their endpoint.

        A leaf skipped by a sibling condition under a Selector never acted, and
        must not be remembered as though it had.
        """
        if self.tree is None:
            return []
        return [(node.name, node.action_url) for node in self.tree.iterate()
                if isinstance(node, ActionAffordanceNode)
                and getattr(node, "_executed_successfully", False)]

    async def _announce(self, status: PlanStatus) -> None:
        """Tell the user their plan finished, in their own words."""
        record = self.agent.plans.get(self.plan_id)
        if record is None:
            return
        goal = record.goal_description or "your request"
        if status is PlanStatus.COMPLETED:
            text = f"Done: {goal}"
        elif status is PlanStatus.CANCELLED:
            text = f"Stopped: {goal}"
        else:
            text = f"I couldn't finish: {goal}"
        await self.agent.notify_conversation(record.plan.conversation_id, text)

    # -- control messages ------------------------------------------------

    async def _drain_control_messages(self) -> None:
        """Take whatever is waiting, without blocking the tick."""
        while True:
            msg = await self.receive(timeout=0)
            if msg is None:
                return
            await self._handle_control(msg)

    async def _handle_control(self, msg) -> None:
        message_type = msg.get_metadata("type")

        if message_type == MessageType.PLAN_CANCEL_REQUEST.value:
            # Flagged, not acted on here: cancelling lands at the top of the
            # next run(), so it always falls between ticks.
            self._cancelled = True
            self._cancel_requester = msg
            self.logger.info(demo(f"Cancel requested: plan={self.plan_id}"))
            await self._reply(msg, MessageType.PLAN_EXECUTION_STATUS.value,
                              {"plan_id": self.plan_id, "cancelling": True})
            return

        if message_type == MessageType.PLAN_STATUS_REQUEST.value:
            record = self.agent.plans.get(self.plan_id)
            await self._reply(msg, MessageType.PLAN_EXECUTION_STATUS.value,
                              record.snapshot() if record else
                              {"plan_id": self.plan_id, "status": "unknown"})
            return

        if message_type == MessageType.PLAN_EXPLAIN_REQUEST.value:
            await self._reply(msg, MessageType.PLAN_EXPLAIN_RESPONSE.value,
                              self.explain())
            return

        await self._handle_other_control(msg, message_type)

    async def _handle_other_control(self, msg, message_type: str) -> None:
        """Control a particular lifecycle understands. Nothing, by default."""
        self.logger.debug(
            f"Plan {self.plan_id} ignoring {message_type}: "
            f"not meaningful for a {type(self).__name__}")



    async def _reply(self, msg, message_type: str, payload: dict) -> None:
        """Answer the caller.

        `rpc_call` drops a reply whose `type` is not exactly the awaited one
        (spade_rpc.py:82-89) -- silently, so a mismatch looks like a timeout
        rather than an error.
        """
        reply = msg.make_reply()
        reply.set_metadata("type", message_type)
        correlation = msg.get_metadata(META_CORRELATION_ID)
        if correlation:
            reply.set_metadata(META_CORRELATION_ID, correlation)
        reply.set_metadata(META_PLAN_ID, self.plan_id)
        reply.body = json.dumps(payload, default=str)
        await self.send(reply)

    # -- introspection ---------------------------------------------------

    def explain(self) -> Dict[str, Any]:
        """What this plan is doing, and how.

        The *what* comes from the record: the user's own phrasing and the
        planner's explanation. The *how* is read live off the tree, so it
        describes the run in progress rather than the plan as written.
        """
        record = self.agent.plans.get(self.plan_id)
        payload: Dict[str, Any] = {
            "plan_id": self.plan_id,
            "what": {
                "goal_description": record.goal_description if record else None,
                "goal_intent": record.plan.goal_intent if record else None,
                "explanation": (record.plan.behavior_tree.metadata.get("explanation")
                                if record else None),
                "plan_type": record.plan.plan_type.value if record else None,
                "status": record.plan.status.value if record else None,
                "ticks": record.ticks if record else 0,
                "execution_count": (record.plan.execution_count
                                    if record else 0),
            },
            "how": self._leaf_states(),
        }
        return payload

    def _leaf_states(self) -> List[Dict[str, Any]]:
        """Every leaf, what it targets, and where it currently stands."""
        if self.tree is None:
            return []
        leaves: List[Dict[str, Any]] = []
        for node in self.tree.iterate():
            if node.children:
                continue
            entry: Dict[str, Any] = {
                "name": node.name,
                "kind": type(node).__name__,
                "status": node.status.name,
            }
            if isinstance(node, ActionAffordanceNode):
                entry["action_url"] = node.action_url
                entry["invoked"] = getattr(node, "_executed_successfully", False)
            elif isinstance(node, (PropertyConditionNode,
                                   ComparisonPropertyConditionNode)):
                entry["property_url"] = node.property_url
                entry["expected"] = node.expected_value
                entry["actual"] = getattr(node, "_actual_value", None)
                entry["operator"] = getattr(
                    getattr(node, "operator", None), "value", "==")
            elif isinstance(node, PropertyAffordanceNode):
                entry["property_url"] = node.property_url
            elif isinstance(node, SettlingTimeWaitNode):
                entry["settling_seconds"] = node.settling_seconds
                entry["elapsed"] = round(node.elapsed_seconds, 2)
            leaves.append(entry)
        return leaves

    # -- what a subclass fills in ----------------------------------------

    async def run(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class AchievementPlanBehaviour(PlanExecutionBehaviour):
    """A plan that finishes.

    Ticks until the tree reaches a verdict, or until the tick budget runs out
    -- which is a failure, because a finite plan that never concluded did not
    do what it was asked.
    """

    async def run(self) -> None:
        await self._drain_control_messages()

        if self._cancelled:
            self._finish(PlanStatus.CANCELLED)
            return
        if self.agent.plans.get(self.plan_id) is None:
            self.kill()
            return

        # EXACTLY ONE TICK. Everything else this agent has to do happens
        # between this call and the next.
        self.tree.tick_once()
        status = self.tree.status
        ticks = self.agent.plans.bump_tick(self.plan_id, status.name)

        if status is Status.RUNNING:
            if ticks >= self.max_ticks:
                self.logger.warning(
                    f"Plan {self.plan_id} hit max_ticks ({self.max_ticks})")
                self._finish(PlanStatus.FAILED, error="max ticks reached")
                return
            await asyncio.sleep(self.tick_period)
            return

        self._finish(PlanStatus.COMPLETED if status is Status.SUCCESS
                     else PlanStatus.FAILED)


class MaintenancePlanBehaviour(PlanExecutionBehaviour):
    """A plan that keeps watching.

    Ticks in bursts: within a burst it behaves exactly like an achievement plan
    -- same tick budget, same cadence -- and between bursts it stands down for
    `maintenance_interval`, so a standing goal does not spin on the network.

    A burst ends either because the tree reached a verdict or because the
    budget ran out, and those mean different things:

    - **A verdict, SUCCESS or FAILURE.** The round is over and the tree is
      re-armed for the next one. FAILURE re-arms exactly like SUCCESS: a plan
      whose guard condition fails on the first tick has not died, it has found
      the world not yet in need of it.
    - **The budget, with the tree still RUNNING.** The tree is mid-flight, so
      it is left untouched and the next burst carries on from where this one
      stopped. A node that would run forever is the node's problem; resetting
      the tree underneath it would only hide that.

    A burst can also be started early by `PLAN_TRIGGER_REQUEST`. What decides
    to send it -- a WebSub push, a threshold watcher, a person -- is
    deliberately outside this behaviour, which only accepts the nudge.
    """

    def __init__(self, *args, maintenance_interval: float = 180.0,
                 trigger_poll: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.maintenance_interval = maintenance_interval
        # A standing-down plan is not ticking, so it need not look often. This
        # sets how soon a trigger is noticed, and nothing else.
        self.trigger_poll = trigger_poll
        self._burst_ticks = 0
        self._between_bursts = False
        self._next_burst_at = 0.0
        self._triggered = False

    async def run(self) -> None:
        await self._drain_control_messages()

        if self._cancelled:
            self._finish(PlanStatus.CANCELLED)
            return
        if self.agent.plans.get(self.plan_id) is None:
            self.kill()
            return

        if self._between_bursts:
            await self._wait_for_next_burst()
            return

        self.tree.tick_once()
        status = self.tree.status
        self.agent.plans.bump_tick(self.plan_id, status.name)
        self._burst_ticks += 1

        if status is not Status.RUNNING:
            count = self.agent.plans.bump_execution_count(self.plan_id)
            self.logger.info(demo(
                f"Maintenance round finished: plan={self.plan_id} "
                f"round={count} outcome={status.name}"))
            self._rearm()
            self._stand_down()
            return

        if self._burst_ticks >= self.max_ticks:
            self.logger.info(demo(
                f"Maintenance burst spent its {self.max_ticks} ticks with the "
                f"tree still RUNNING: plan={self.plan_id}; the next burst "
                f"resumes where this one stopped"))
            self._stand_down()
            return

        await asyncio.sleep(self.tick_period)

    def _stand_down(self) -> None:
        """Wait for the interval to elapse, or for something to trigger us."""
        self._between_bursts = True
        self._burst_ticks = 0
        self._next_burst_at = self._now() + self.maintenance_interval
        self.agent.plans.mark_waiting(self.plan_id)

    async def _wait_for_next_burst(self) -> None:
        if self._triggered:
            self._triggered = False
            self._start_burst("triggered")
            return
        if self._now() >= self._next_burst_at:
            self._start_burst("interval elapsed")
            return
        # Sleep in slices rather than for the whole remaining interval, so a
        # trigger -- and a cancel, drained at the top of run() -- is acted on
        # without waiting the interval out. Never overshoot the next burst.
        remaining = self._next_burst_at - self._now()
        await asyncio.sleep(max(0.0, min(self.trigger_poll, remaining)))

    def _start_burst(self, reason: str) -> None:
        self._between_bursts = False
        self._burst_ticks = 0
        self.agent.plans.mark_started(self.plan_id)
        self.logger.info(demo(
            f"Maintenance burst starting ({reason}): plan={self.plan_id}"))

    def _rearm(self) -> None:
        """Reset the tree so the next burst starts a fresh round."""
        for node in self.tree.iterate():
            node.stop(Status.INVALID)

    async def _handle_other_control(self, msg, message_type: str) -> None:
        if message_type != MessageType.PLAN_TRIGGER_REQUEST.value:
            await super()._handle_other_control(msg, message_type)
            return
        self.trigger()
        self.logger.info(demo(f"Burst triggered: plan={self.plan_id}"))
        await self._reply(msg, MessageType.PLAN_EXECUTION_STATUS.value,
                          {"plan_id": self.plan_id, "triggered": True})

    def trigger(self) -> None:
        """Ask for a burst now rather than at the end of the interval.

        Public so a future in-process trigger source can call it directly,
        without having to send itself a message.
        """
        self._triggered = True

    @staticmethod
    def _now() -> float:
        return asyncio.get_event_loop().time()
