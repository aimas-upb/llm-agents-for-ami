"""
The UA's plan manager: takes confirmed plans and runs them.

One cyclic behaviour owns the fleet. It accepts `PLAN_EXECUTE_REQUEST` from a
finished `UserRequestBehaviour`, registers each plan, and spawns a
`PlanExecutionBehaviour` per plan, addressed by `ami_plan_id`.

Control messages (status, cancel, explain) arrive here and are routed to the
running plan they name -- or, when they name none, to whatever is running for
that conversation, which is what "stop that" means in practice.

A plan envelope may hold several plans: the ISA plans each intent separately, so
"turn on the light and close the blinds" comes back as two, and each gets its
own id, its own behaviour, and its own lifecycle.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from ....shared.models.messages import (
    META_CORRELATION_ID,
    META_PLAN_ID,
    META_REQUEST_ID,
    MessageType,
)
from ....shared.models.plan import (
    BehaviorTreePlan,
    Plan,
    PlanStatus,
    PlanType,
)
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from .plan_execution import (
    AchievementPlanBehaviour,
    MaintenancePlanBehaviour,
    PlanExecutionBehaviour,
)

_CONTROL_TYPES = {
    MessageType.PLAN_STATUS_REQUEST.value,
    MessageType.PLAN_CANCEL_REQUEST.value,
    MessageType.PLAN_EXPLAIN_REQUEST.value,
    MessageType.PLAN_TRIGGER_REQUEST.value,
}


class PlanManagementBehaviour(CyclicBehaviour):
    """Registers confirmed plans and runs one behaviour per plan."""

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")
        self._executors: Dict[str, PlanExecutionBehaviour] = {}

    async def run(self) -> None:
        msg = await self.receive(timeout=1)
        if msg is None:
            self._reap()
            return

        message_type = msg.get_metadata("type")
        if message_type == MessageType.PLAN_EXECUTE_REQUEST.value:
            await self._accept(msg)
        elif message_type in _CONTROL_TYPES:
            await self._route_control(msg)

    def _reap(self) -> None:
        """Forget behaviours that have ended, so the map does not grow."""
        for plan_id in [p for p, b in self._executors.items() if b.is_killed()]:
            self._executors.pop(plan_id, None)

    # -- accepting a confirmed plan --------------------------------------

    async def _accept(self, msg) -> None:
        try:
            envelope = json.loads(msg.body or "{}")
        except json.JSONDecodeError as exc:
            self.logger.error(f"Unreadable plan envelope: {exc}")
            return

        plan_obj = envelope.get("plan") or {}
        request_id = envelope.get("request_id") or msg.get_metadata(META_REQUEST_ID)
        thread = envelope.get("thread") or "__default__"
        goal_description = envelope.get("goal_description") or ""
        plan_hash = envelope.get("plan_hash")

        for index, (plan_id, tree_spec, intent) in enumerate(
                self._split(envelope.get("plan_id"), plan_obj)):
            if not tree_spec:
                continue
            plan = Plan(
                plan_id=plan_id,
                plan_type=self._classify_type(plan_obj, intent),
                # noqa: the intent dict is the plan's own, one per tree
                status=PlanStatus.CREATED,
                # The user's own words, so an explanation can answer in them.
                goal_description=goal_description,
                goal_intent=self._intent_text(intent) or goal_description,
                conversation_id=thread,
                behavior_tree=BehaviorTreePlan.from_json_ir(
                    plan_id, {"tree": tree_spec, **{
                        k: v for k, v in plan_obj.items() if k != "tree"}}),
                request_id=request_id,
                plan_hash=plan_hash,
            )
            self.agent.plans.register(plan)
            self.agent.requests.note_plan(request_id, plan_id)

            # The lifecycle decides the behaviour: a plan that finishes and a
            # plan that keeps watching are not the same thing wearing a flag.
            options: Dict[str, Any] = {}
            if plan.plan_type is PlanType.MAINTENANCE:
                executor_class = MaintenancePlanBehaviour
                options["maintenance_interval"] = \
                    self.agent.bt_maintenance_interval
                options["trigger_poll"] = \
                    self.agent.bt_maintenance_trigger_poll
            else:
                executor_class = AchievementPlanBehaviour

            executor = executor_class(
                plan_id=plan_id,
                tree_spec=tree_spec,
                max_ticks=self.agent.bt_max_ticks,
                tick_period=self.agent.bt_min_tick_yield,
                logger=self.logger,
                **options,
                # Everything recording needs that only the envelope knows. The
                # executor passes it straight through without reading it.
                recording_options={
                    # This plan's own intent, and therefore its own category.
                    "intents": [intent] if intent else [],
                    "intent_type": self._intent_category(intent),
                    "workspace_id": envelope.get("workspace_id")
                    or plan_obj.get("workspace_id"),
                    "td_sosa_supported": bool(
                        envelope.get("td_sosa_supported")
                        or plan_obj.get("td_sosa_supported")),
                    "is_signifier_reuse": bool(
                        plan_obj.get("signifier_reuse")),
                },
            )
            template = Template()
            template.set_metadata(META_PLAN_ID, plan_id)
            self._executors[plan_id] = executor
            self.agent.add_behaviour(executor, template)

            self.logger.info(demo(
                f"Plan accepted: plan={plan_id} request={request_id} "
                f"type={plan.plan_type.value} goal={goal_description!r}"))

    @staticmethod
    def _split(base_id: Optional[str], plan_obj: Dict[str, Any]):
        """One entry per plan in the envelope, each with its own id and intent.

        The ISA plans one tree per atomic intent, and a multi-plan envelope
        carries that intent on each entry (`envelope_llm_plans`). So a plan has
        exactly one intent, and therefore exactly one category -- which is what
        signifier recording needs, and what a request-level label would blur.
        """
        from ....shared.models.messages import new_plan_id

        plans = plan_obj.get("plans")
        if isinstance(plans, list) and plans:
            out = []
            for index, plan in enumerate(plans):
                if not isinstance(plan, dict):
                    continue
                plan_id = base_id if (index == 0 and base_id) else new_plan_id()
                out.append((plan_id, plan.get("tree"), plan.get("intent")))
            return out

        # Single-plan envelopes list their intents instead.
        intents = plan_obj.get("intents") or []
        intent = intents[0] if intents else None
        return [(base_id or new_plan_id(), plan_obj.get("tree"), intent)]

    @staticmethod
    def _intent_text(intent: Any) -> str:
        """A plan's intent as text, for `goal_intent`."""
        if isinstance(intent, dict):
            return str(intent.get("text_intent") or intent.get("intent_text") or "")
        return str(intent or "")

    @staticmethod
    def _intent_category(intent: Any) -> Optional[str]:
        """This plan's intent category, uppercased.

        One intent, one category -- currently EXPLICIT or IMPLICIT, but the set
        is expected to grow, so this reads whatever the intent declares rather
        than testing against a fixed pair.
        """
        if not isinstance(intent, dict):
            return None
        category = intent.get("category")
        return str(category).upper() if category else None

    @staticmethod
    def _classify_type(plan_obj: Dict[str, Any], intent: Any) -> PlanType:
        """Achievement unless the plan says otherwise.

        Nothing produces a maintenance signal yet -- `plan_type` in the IR is a
        *format* discriminator ("behavior_tree"), not a lifecycle one. The hook
        is `execution_mode`, which segmentation will set once maintenance goals
        are classified upstream.
        """
        mode = str(plan_obj.get("execution_mode") or "").lower()
        if mode in ("maintenance", "persistent", "maintain"):
            return PlanType.MAINTENANCE
        return PlanType.IMMEDIATE

    # -- routing status / cancel / explain -------------------------------

    async def _route_control(self, msg) -> None:
        """Send a control message to the plan(s) it concerns.

        Note this behaviour does not answer these itself: the running plan does,
        because only it knows where its tree currently stands.
        """
        plan_id = msg.get_metadata(META_PLAN_ID)
        targets: List[str]

        if plan_id:
            targets = [plan_id]
        else:
            thread = str(getattr(msg, "thread", None) or "__default__")
            targets = [r.plan_id for r in self.agent.plans.for_conversation(thread)
                       if r.is_active]
            if not targets:
                targets = [r.plan_id for r in self.agent.plans.active()]

        if not targets:
            reply = msg.make_reply()
            reply.set_metadata("type", MessageType.PLAN_EXECUTION_STATUS.value)
            correlation = msg.get_metadata(META_CORRELATION_ID)
            if correlation:
                reply.set_metadata(META_CORRELATION_ID, correlation)
            reply.body = json.dumps({"plans": [], "detail": "nothing running"})
            await self.send(reply)
            return

        for target in targets:
            forwarded = msg.make_reply()
            forwarded.to = self.agent.jid
            forwarded.set_metadata("type", msg.get_metadata("type"))
            forwarded.set_metadata(META_PLAN_ID, target)
            correlation = msg.get_metadata(META_CORRELATION_ID)
            if correlation:
                forwarded.set_metadata(META_CORRELATION_ID, correlation)
            forwarded.body = msg.body or "{}"
            await self.send(forwarded)

    # -- what the agent can ask ------------------------------------------

    def running_plans(self) -> List[Dict[str, Any]]:
        """A snapshot of everything active — the answer to "what's running?"."""
        return [record.snapshot() for record in self.agent.plans.active()]

    def explain_all(self) -> List[Dict[str, Any]]:
        return [executor.explain() for plan_id, executor in self._executors.items()
                if not executor.is_killed()]
