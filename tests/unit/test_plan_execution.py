"""Plan execution: one tick per run, cancel between ticks, maintenance re-arms."""

import json

import pytest
from py_trees.common import Status
from unittest.mock import AsyncMock, MagicMock, patch

from ami_agents.agents.user_assistant.behaviours.plan_execution import (
    AchievementPlanBehaviour,
    MaintenancePlanBehaviour,
)
from ami_agents.agents.user_assistant.behaviours.plan_management import (
    PlanManagementBehaviour,
)
from ami_agents.agents.user_assistant.registry import PlanRegistry, RequestRegistry
from ami_agents.shared.models.messages import (
    META_PLAN_ID,
    MessageType,
)
from ami_agents.shared.models.plan import (
    BehaviorTreePlan,
    Plan,
    PlanStatus,
    PlanType,
)

TREE = {"type": "action", "name": "Act", "action_url": "http://localhost/a"}


def make_agent():
    agent = MagicMock()
    agent.requests = RequestRegistry()
    agent.plans = PlanRegistry()
    agent.bt_max_ticks = 50
    agent.bt_min_tick_yield = 0.0
    agent.jid = "ua@localhost"
    agent.notify_conversation = AsyncMock()
    return agent


def register(agent, plan_id="plan-1", plan_type=PlanType.IMMEDIATE,
             goal="turn on the kitchen light"):
    plan = Plan(
        plan_id=plan_id, plan_type=plan_type, status=PlanStatus.CREATED,
        goal_description=goal, goal_intent=goal, conversation_id="thread-a",
        behavior_tree=BehaviorTreePlan.from_json_ir(plan_id, {"tree": TREE}),
        request_id="req-1")
    agent.plans.register(plan)
    return plan


def make_executor(agent, plan_id="plan-1", statuses=None,
                  behaviour_class=AchievementPlanBehaviour, **kwargs):
    behaviour = behaviour_class(plan_id, TREE, max_ticks=50,
                                tick_period=0.0, **kwargs)
    behaviour.agent = agent
    behaviour.send = AsyncMock()
    behaviour.kill = MagicMock()
    behaviour.receive = AsyncMock(return_value=None)

    tree = MagicMock()
    tree.status = Status.RUNNING
    tree.iterate = MagicMock(return_value=[])
    if statuses:
        seq = list(statuses)

        def tick():
            tree.status = seq.pop(0) if seq else Status.SUCCESS
        tree.tick_once = MagicMock(side_effect=tick)
    else:
        tree.tick_once = MagicMock()
    behaviour.tree = tree
    behaviour._setup_done = True
    return behaviour


class TestOneTickPerRun:
    @pytest.mark.asyncio
    async def test_exactly_one_tick(self):
        agent = make_agent(); register(agent)
        behaviour = make_executor(agent, statuses=[Status.RUNNING])
        await behaviour.run()
        assert behaviour.tree.tick_once.call_count == 1

    @pytest.mark.asyncio
    async def test_running_leaves_the_plan_active(self):
        agent = make_agent(); register(agent)
        behaviour = make_executor(agent, statuses=[Status.RUNNING] * 3)
        for _ in range(3):
            await behaviour.run()
        record = agent.plans.get("plan-1")
        assert record.ticks == 3
        assert record.is_active
        behaviour.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_completes_and_ends(self):
        agent = make_agent(); register(agent)
        behaviour = make_executor(agent, statuses=[Status.SUCCESS])
        await behaviour.run()
        record = agent.plans.get("plan-1")
        assert record.plan.status is PlanStatus.COMPLETED
        assert record.plan.execution_count == 1
        behaviour.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_is_recorded(self):
        agent = make_agent(); register(agent)
        behaviour = make_executor(agent, statuses=[Status.FAILURE])
        await behaviour.run()
        assert agent.plans.get("plan-1").plan.status is PlanStatus.FAILED
        behaviour.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_ticks_stops_an_achievement_plan(self):
        agent = make_agent(); register(agent)
        behaviour = make_executor(agent, statuses=[Status.RUNNING] * 5)
        behaviour.max_ticks = 3
        for _ in range(3):
            await behaviour.run()
        record = agent.plans.get("plan-1")
        assert record.plan.status is PlanStatus.FAILED
        assert record.error == "max ticks reached"


class TestMaintenance:
    @pytest.mark.asyncio
    async def test_success_rearms_instead_of_completing(self):
        agent = make_agent()
        register(agent, plan_type=PlanType.MAINTENANCE,
                 goal="keep the living room at 25 degrees")
        behaviour = make_executor(agent, statuses=[Status.SUCCESS],
                                  behaviour_class=MaintenancePlanBehaviour,
                                  maintenance_interval=3600.0)
        await behaviour.run()

        record = agent.plans.get("plan-1")
        assert record.plan.status is not PlanStatus.COMPLETED
        assert record.is_active
        assert record.plan.execution_count == 1
        behaviour.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_ticks_bounds_a_burst_not_the_plan(self):
        """The budget ends a burst; the plan itself keeps going."""
        agent = make_agent()
        register(agent, plan_type=PlanType.MAINTENANCE)
        behaviour = make_executor(agent, statuses=[Status.RUNNING] * 20,
                                  behaviour_class=MaintenancePlanBehaviour,
                                  maintenance_interval=0.0)
        behaviour.max_ticks = 3
        for _ in range(20):
            await behaviour.run()
        record = agent.plans.get("plan-1")
        assert record.is_active
        behaviour.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_is_the_only_way_out(self):
        agent = make_agent()
        register(agent, plan_type=PlanType.MAINTENANCE)
        behaviour = make_executor(agent, statuses=[Status.SUCCESS] * 5,
                                  behaviour_class=MaintenancePlanBehaviour,
                                  maintenance_interval=0.0)
        await behaviour.run()
        assert agent.plans.get("plan-1").is_active

        behaviour._cancelled = True
        await behaviour.run()
        assert agent.plans.get("plan-1").plan.status is PlanStatus.CANCELLED


class TestMaintenanceBursts:
    """Bursts of ticks, spaced by an interval, resumable, externally triggerable."""

    def _maintenance(self, agent, statuses, max_ticks=3, interval=180.0):
        register(agent, plan_type=PlanType.MAINTENANCE,
                 goal="keep the living room at 25 degrees")
        behaviour = make_executor(
            agent, statuses=statuses,
            behaviour_class=MaintenancePlanBehaviour,
            maintenance_interval=interval)
        behaviour.max_ticks = max_ticks
        return behaviour

    @pytest.mark.asyncio
    async def test_a_burst_uses_the_same_budget_then_stands_down(self):
        agent = make_agent()
        behaviour = self._maintenance(agent, [Status.RUNNING] * 10, max_ticks=3)

        for _ in range(3):
            await behaviour.run()

        assert behaviour.tree.tick_once.call_count == 3
        assert behaviour._between_bursts is True
        assert agent.plans.get("plan-1").is_active
        behaviour.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unfinished_burst_resumes_rather_than_resetting(self):
        """The tree is mid-flight; a reset would discard its progress."""
        agent = make_agent()
        behaviour = self._maintenance(agent, [Status.RUNNING] * 10, max_ticks=2,
                                      interval=0.0)
        for _ in range(2):
            await behaviour.run()
        assert behaviour._between_bursts is True

        # The tree was NOT re-armed, so the next burst continues from here.
        for node in behaviour.tree.iterate():
            node.stop.assert_not_called()

        await behaviour.run()  # interval already elapsed -> new burst
        assert behaviour._between_bursts is False
        assert behaviour._burst_ticks == 0

    @pytest.mark.asyncio
    async def test_success_ends_the_round_and_rearms(self):
        agent = make_agent()
        behaviour = self._maintenance(agent, [Status.SUCCESS])
        await behaviour.run()

        assert agent.plans.get("plan-1").plan.execution_count == 1
        assert behaviour._between_bursts is True
        assert agent.plans.get("plan-1").is_active

    @pytest.mark.asyncio
    async def test_failure_also_rearms(self):
        """A guard condition failing on the first tick is not a death."""
        agent = make_agent()
        behaviour = self._maintenance(agent, [Status.FAILURE])
        await behaviour.run()

        record = agent.plans.get("plan-1")
        assert record.plan.status is not PlanStatus.FAILED
        assert record.is_active
        assert record.plan.execution_count == 1
        behaviour.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_it_waits_out_the_interval_before_the_next_burst(self):
        agent = make_agent()
        behaviour = self._maintenance(agent, [Status.SUCCESS] * 10,
                                      interval=3600.0)
        await behaviour.run()
        ticks_after_first = behaviour.tree.tick_once.call_count

        for _ in range(5):
            await behaviour.run()
        # Still waiting: no further ticks were spent.
        assert behaviour.tree.tick_once.call_count == ticks_after_first
        assert behaviour._between_bursts is True

    @pytest.mark.asyncio
    async def test_a_trigger_starts_a_burst_early(self):
        agent = make_agent()
        behaviour = self._maintenance(agent, [Status.SUCCESS] * 10,
                                      interval=3600.0)
        await behaviour.run()
        assert behaviour._between_bursts is True

        behaviour._triggered = True
        await behaviour.run()

        assert behaviour._between_bursts is False
        assert behaviour._triggered is False

    @pytest.mark.asyncio
    async def test_the_trigger_message_is_accepted(self):
        agent = make_agent()
        behaviour = self._maintenance(agent, [Status.SUCCESS] * 5,
                                      interval=3600.0)
        msg = MagicMock()
        msg.get_metadata = lambda k: {
            "type": MessageType.PLAN_TRIGGER_REQUEST.value,
            META_PLAN_ID: "plan-1"}.get(k)
        msg.make_reply = MagicMock(return_value=MagicMock())
        msg.body = "{}"
        behaviour.receive = AsyncMock(side_effect=[msg, None, None])

        await behaviour.run()
        assert behaviour._triggered is True or \
            behaviour._between_bursts is False

    @pytest.mark.asyncio
    async def test_standing_down_polls_at_the_trigger_cadence(self):
        """Not at the tick cadence: a waiting plan need not look often."""
        agent = make_agent()
        behaviour = self._maintenance(agent, [Status.SUCCESS] * 5,
                                      interval=3600.0)
        behaviour.trigger_poll = 5.0
        await behaviour.run()
        assert behaviour._between_bursts is True

        with patch("ami_agents.agents.user_assistant.behaviours."
                   "plan_execution.asyncio.sleep", AsyncMock()) as sleep:
            await behaviour.run()
        sleep.assert_awaited_once_with(5.0)

    @pytest.mark.asyncio
    async def test_the_last_poll_does_not_overshoot_the_burst(self):
        agent = make_agent()
        behaviour = self._maintenance(agent, [Status.SUCCESS] * 5,
                                      interval=3600.0)
        behaviour.trigger_poll = 5.0
        await behaviour.run()
        # Pretend only 2s remain: sleeping a full 5s would start the burst late.
        behaviour._next_burst_at = behaviour._now() + 2.0

        with patch("ami_agents.agents.user_assistant.behaviours."
                   "plan_execution.asyncio.sleep", AsyncMock()) as sleep:
            await behaviour.run()
        assert sleep.await_args[0][0] <= 2.0

    @pytest.mark.asyncio
    async def test_only_cancel_ends_it(self):
        agent = make_agent()
        behaviour = self._maintenance(agent, [Status.SUCCESS] * 20,
                                      interval=0.0)
        for _ in range(20):
            await behaviour.run()
        assert agent.plans.get("plan-1").is_active

        behaviour._cancelled = True
        await behaviour.run()
        assert agent.plans.get("plan-1").plan.status is PlanStatus.CANCELLED
        behaviour.kill.assert_called()


class TestControl:
    def _control(self, message_type, plan_id="plan-1"):
        msg = MagicMock()
        msg.get_metadata = lambda k: {"type": message_type,
                                      META_PLAN_ID: plan_id}.get(k)
        msg.make_reply = MagicMock(return_value=MagicMock())
        msg.body = "{}"
        return msg

    @pytest.mark.asyncio
    async def test_cancel_lands_between_ticks_never_mid_tick(self):
        agent = make_agent(); register(agent)
        behaviour = make_executor(agent, statuses=[Status.RUNNING] * 3)
        cancel = self._control(MessageType.PLAN_CANCEL_REQUEST.value)
        behaviour.receive = AsyncMock(side_effect=[cancel, None, None])

        await behaviour.run()
        # The cancel arrived, was flagged, and the tree was NOT ticked again.
        assert behaviour._cancelled is True
        assert agent.plans.get("plan-1").plan.status is PlanStatus.CANCELLED
        assert behaviour.tree.tick_once.call_count == 0

    @pytest.mark.asyncio
    async def test_status_request_is_answered_with_the_snapshot(self):
        agent = make_agent(); register(agent)
        behaviour = make_executor(agent, statuses=[Status.RUNNING])
        agent.plans.mark_started("plan-1")
        status = self._control(MessageType.PLAN_STATUS_REQUEST.value)
        behaviour.receive = AsyncMock(side_effect=[status, None])

        await behaviour.run()
        behaviour.send.assert_awaited()
        payload = json.loads(behaviour.send.await_args[0][0].body)
        assert payload["plan_id"] == "plan-1"
        assert payload["goal_description"] == "turn on the kitchen light"
        assert payload["request_id"] == "req-1"

    @pytest.mark.asyncio
    async def test_explain_answers_what_and_how(self):
        agent = make_agent(); register(agent)
        behaviour = make_executor(agent, statuses=[Status.RUNNING])
        explain = self._control(MessageType.PLAN_EXPLAIN_REQUEST.value)
        behaviour.receive = AsyncMock(side_effect=[explain, None])

        await behaviour.run()
        payload = json.loads(behaviour.send.await_args[0][0].body)
        assert payload["what"]["goal_description"] == "turn on the kitchen light"
        assert "how" in payload


class TestPlanManagement:
    @pytest.mark.asyncio
    async def test_accepting_a_plan_registers_it_and_spawns_a_behaviour(self):
        agent = make_agent()
        agent.requests.open("req-1", "thread-a", "turn on the kitchen light")
        manager = PlanManagementBehaviour()
        manager.agent = agent
        manager.send = AsyncMock()

        msg = MagicMock()
        msg.get_metadata = lambda k: (
            MessageType.PLAN_EXECUTE_REQUEST.value if k == "type" else None)
        msg.body = json.dumps({
            "plan_id": "plan-1", "request_id": "req-1", "thread": "thread-a",
            "goal_description": "turn on the kitchen light",
            "plan_hash": "hash-abc", "plan": {"tree": TREE}})

        await manager._accept(msg)

        record = agent.plans.get("plan-1")
        assert record is not None
        assert record.goal_description == "turn on the kitchen light"
        assert record.plan.request_id == "req-1"
        assert record.plan.plan_type is PlanType.IMMEDIATE
        agent.add_behaviour.assert_called_once()
        assert isinstance(agent.add_behaviour.call_args[0][0],
                          AchievementPlanBehaviour)
        # Addressed by its own plan id, so control messages reach just this plan.
        template = agent.add_behaviour.call_args[0][1]
        assert template._metadata[META_PLAN_ID] == "plan-1"

    @pytest.mark.asyncio
    async def test_a_multi_plan_envelope_becomes_several_plans(self):
        agent = make_agent()
        agent.requests.open("req-1", "thread-a", "light on and blinds closed")
        manager = PlanManagementBehaviour()
        manager.agent = agent
        manager.send = AsyncMock()

        msg = MagicMock()
        msg.get_metadata = lambda k: (
            MessageType.PLAN_EXECUTE_REQUEST.value if k == "type" else None)
        msg.body = json.dumps({
            "plan_id": "plan-1", "request_id": "req-1", "thread": "thread-a",
            "goal_description": "light on and blinds closed",
            "plan": {"plans": [{"tree": TREE}, {"tree": TREE}]}})

        await manager._accept(msg)

        assert len(agent.plans) == 2
        assert agent.add_behaviour.call_count == 2
        # Both trace back to the one request that asked for them.
        assert len(agent.requests.get("req-1").plan_ids) == 2

    @pytest.mark.asyncio
    async def test_each_plan_carries_its_own_intent_category(self):
        """One plan, one intent, one category — mixed requests do not blur.

        `envelope_llm_plans` puts each atomic intent on its own plan entry, so
        an explicit goal and an implicit one in the same utterance are recorded
        with the context each actually warrants.
        """
        agent = make_agent()
        agent.requests.open("req-1", "thread-a",
                            "turn on lamp_3 and make the room comfortable")
        manager = PlanManagementBehaviour()
        manager.agent = agent
        manager.send = AsyncMock()

        msg = MagicMock()
        msg.get_metadata = lambda k: (
            MessageType.PLAN_EXECUTE_REQUEST.value if k == "type" else None)
        msg.body = json.dumps({
            "plan_id": "plan-1", "request_id": "req-1", "thread": "thread-a",
            "goal_description": "turn on lamp_3 and make the room comfortable",
            "workspace_id": "ws-1",
            "plan": {"plans": [
                {"tree": TREE, "intent": {"category": "explicit",
                                          "text_intent": "turn on lamp_3"}},
                {"tree": TREE, "intent": {"category": "implicit",
                                          "text_intent": "make it comfortable"}},
            ]}})

        await manager._accept(msg)

        options = [call[0][0]._recording_options
                   for call in agent.add_behaviour.call_args_list]
        assert [o["intent_type"] for o in options] == ["EXPLICIT", "IMPLICIT"]
        # And each carries only its own intent.
        assert options[0]["intents"][0]["text_intent"] == "turn on lamp_3"
        assert options[1]["intents"][0]["text_intent"] == "make it comfortable"
        assert all(o["workspace_id"] == "ws-1" for o in options)

    def test_intent_category_reads_whatever_the_intent_declares(self):
        # The category set is expected to grow beyond explicit/implicit.
        assert PlanManagementBehaviour._intent_category(
            {"category": "explicit"}) == "EXPLICIT"
        assert PlanManagementBehaviour._intent_category(
            {"category": "conditional"}) == "CONDITIONAL"
        assert PlanManagementBehaviour._intent_category({}) is None
        assert PlanManagementBehaviour._intent_category(None) is None

    @pytest.mark.asyncio
    async def test_a_maintenance_plan_gets_the_maintenance_behaviour(self):
        agent = make_agent()
        agent.requests.open("req-1", "thread-a",
                            "keep the living room at 25 degrees")
        agent.bt_maintenance_interval = 180.0
        manager = PlanManagementBehaviour()
        manager.agent = agent
        manager.send = AsyncMock()

        msg = MagicMock()
        msg.get_metadata = lambda k: (
            MessageType.PLAN_EXECUTE_REQUEST.value if k == "type" else None)
        msg.body = json.dumps({
            "plan_id": "plan-1", "request_id": "req-1", "thread": "thread-a",
            "goal_description": "keep the living room at 25 degrees",
            "plan": {"tree": TREE, "execution_mode": "maintenance"}})

        await manager._accept(msg)

        spawned = agent.add_behaviour.call_args[0][0]
        assert isinstance(spawned, MaintenancePlanBehaviour)
        assert spawned.maintenance_interval == 180.0
        assert agent.plans.get("plan-1").plan.plan_type is PlanType.MAINTENANCE

    def test_execution_mode_marks_a_maintenance_plan(self):
        assert PlanManagementBehaviour._classify_type(
            {"execution_mode": "maintenance"}, "") is PlanType.MAINTENANCE
        assert PlanManagementBehaviour._classify_type(
            {"plan_type": "behavior_tree"}, "") is PlanType.IMMEDIATE
        assert PlanManagementBehaviour._classify_type({}, "") is PlanType.IMMEDIATE
