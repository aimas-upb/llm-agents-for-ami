"""Signifier recording: snapshot at launch, and independent of plan execution."""

import json

import pytest
from py_trees.common import Status
from unittest.mock import AsyncMock, MagicMock, patch

from ami_agents.agents.user_assistant.behaviours.plan_execution import (
    AchievementPlanBehaviour,
)
from ami_agents.agents.user_assistant.registry import PlanRegistry, RequestRegistry
from ami_agents.agents.user_assistant.signifiers import (
    ExecutionContext,
    ExecutionOutcome,
    SignifierExtractor,
    SignifierRecordingBehaviour,
    capture_execution_context,
    collect_action_urls,
)
from ami_agents.shared.models.plan import (
    BehaviorTreePlan,
    Plan,
    PlanStatus,
    PlanType,
)

TREE = {"type": "sequence", "children": [
    {"type": "action", "name": "On", "action_url": "http://x/a1"},
    {"type": "action", "name": "Dim", "action_url": "http://x/a2"},
]}


def make_agent():
    agent = MagicMock()
    agent.requests = RequestRegistry()
    agent.plans = PlanRegistry()
    agent.target_jids = {"explorer": "explorer@localhost"}
    agent.signifier_match_timeout = 5.0
    agent.rpc_call_timeout = 5.0
    agent.bt_max_ticks = 50
    agent.bt_min_tick_yield = 0.0
    agent.community_client = None
    return agent


def context(**overrides):
    defaults = dict(plan_id="plan-1", request_id="req-1", thread="thread-a",
                    tree_spec=TREE, intents=[], intent_type="IMPLICIT",
                    workspace_id="ws-1")
    defaults.update(overrides)
    return ExecutionContext(**defaults)


class TestCollectActionUrls:
    def test_walks_the_tree_in_order(self):
        assert collect_action_urls(TREE) == ["http://x/a1", "http://x/a2"]

    def test_deduplicates(self):
        tree = {"type": "sequence", "children": [
            {"type": "action", "action_url": "http://x/a"},
            {"type": "action", "action_url": "http://x/a"}]}
        assert collect_action_urls(tree) == ["http://x/a"]

    def test_ignores_non_actions(self):
        tree = {"type": "condition", "property_url": "http://x/p"}
        assert collect_action_urls(tree) == []


class TestShouldRecord:
    def test_implicit_and_explicit_record(self):
        assert context(intent_type="IMPLICIT").should_record is True
        assert context(intent_type="EXPLICIT").should_record is True

    def test_unknown_intent_type_does_not(self):
        assert context(intent_type=None).should_record is False
        assert context(intent_type="QUERY").should_record is False

    def test_signifier_reuse_does_not(self):
        """Replaying what was learned teaches nothing new."""
        assert context(is_signifier_reuse=True).should_record is False


class TestCaptureContext:
    @pytest.mark.asyncio
    async def test_implicit_intent_snapshots_state(self):
        agent = make_agent()
        response = MagicMock()
        response.body = json.dumps({"artifacts": {
            "light-1": {"workspace_id": "ws-1", "on": False},
            "other-1": {"workspace_id": "ws-2", "on": True}}})

        with patch("ami_agents.agents.user_assistant.signifiers.rpc_call",
                   AsyncMock(return_value=response)):
            ctx = await capture_execution_context(
                agent, "plan-1", TREE, intent_type="IMPLICIT",
                workspace_id="ws-1")

        # Only this workspace's artifacts, and the value as it was BEFORE.
        assert ctx.state_snapshot == {"artifacts": {
            "light-1": {"workspace_id": "ws-1", "on": False}}}

    @pytest.mark.asyncio
    async def test_explicit_intent_skips_state(self):
        """The user named the target, so the surrounding state is not why."""
        agent = make_agent()
        with patch("ami_agents.agents.user_assistant.signifiers.rpc_call",
                   AsyncMock()) as rpc:
            ctx = await capture_execution_context(
                agent, "plan-1", TREE, intent_type="EXPLICIT",
                workspace_id="ws-1")
        assert ctx.state_snapshot is None
        rpc.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reuse_captures_nothing(self):
        agent = make_agent()
        with patch("ami_agents.agents.user_assistant.signifiers.rpc_call",
                   AsyncMock()) as rpc:
            ctx = await capture_execution_context(
                agent, "plan-1", TREE, intent_type="IMPLICIT",
                workspace_id="ws-1", is_signifier_reuse=True)
        assert ctx.state_snapshot is None
        rpc.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_unreachable_explorer_is_survivable(self):
        agent = make_agent()
        with patch("ami_agents.agents.user_assistant.signifiers.rpc_call",
                   AsyncMock(side_effect=RuntimeError("no explorer"))):
            ctx = await capture_execution_context(
                agent, "plan-1", TREE, intent_type="IMPLICIT",
                workspace_id="ws-1")
        assert ctx.state_snapshot is None
        assert ctx.should_record is True  # still recorded, just without state

    @pytest.mark.asyncio
    async def test_td_sosa_effects_are_folded_into_the_snapshot(self):
        agent = make_agent()

        async def fake_rpc(agent_, to_jid, request_type, body, expect_type,
                           timeout, **kwargs):
            response = MagicMock()
            if "semantic" in request_type:
                response.body = json.dumps({"action_effects": [
                    {"action_url": "http://x/a1",
                     "effects": [{"property": "illuminance"}]}]})
            else:
                response.body = json.dumps({"artifacts": {}})
            return response

        with patch("ami_agents.agents.user_assistant.signifiers.rpc_call",
                   AsyncMock(side_effect=fake_rpc)):
            ctx = await capture_execution_context(
                agent, "plan-1", TREE, intent_type="IMPLICIT",
                workspace_id="ws-1", td_sosa_supported=True)

        merged = ctx.merged_snapshot()
        assert merged["td_sosa_action_effects"]["http://x/a1"] == [
            {"property": "illuminance"}]


class TestRecordingBehaviour:
    def _behaviour(self, agent, ctx, outcome, signifiers=None):
        extractor = MagicMock(spec=SignifierExtractor)
        extractor.extract = MagicMock(return_value=signifiers
                                      if signifiers is not None
                                      else [{"id": "sig-1"}])
        behaviour = SignifierRecordingBehaviour(ctx, outcome, extractor)
        behaviour.agent = agent
        return behaviour

    @pytest.mark.asyncio
    async def test_records_a_successful_run(self):
        agent = make_agent()
        outcome = ExecutionOutcome(success=True, final_status="completed")
        behaviour = self._behaviour(agent, context(), outcome)
        response = MagicMock(); response.body = json.dumps({"created_count": 1})

        with patch("ami_agents.agents.user_assistant.signifiers.rpc_call",
                   AsyncMock(return_value=response)):
            await behaviour.run()

        assert behaviour.signifiers == [{"id": "sig-1"}]
        assert behaviour.created_count == 1

    @pytest.mark.asyncio
    async def test_a_failed_run_records_nothing(self):
        agent = make_agent()
        outcome = ExecutionOutcome(success=False, final_status="failed")
        behaviour = self._behaviour(agent, context(), outcome)
        with patch("ami_agents.agents.user_assistant.signifiers.rpc_call",
                   AsyncMock()) as rpc:
            await behaviour.run()
        rpc.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_extraction_failure_does_not_propagate(self):
        """Recording must never turn a good run into a reported failure."""
        agent = make_agent()
        outcome = ExecutionOutcome(success=True, final_status="completed")
        extractor = MagicMock()
        extractor.extract = MagicMock(side_effect=ValueError("bad tree"))
        behaviour = SignifierRecordingBehaviour(context(), outcome, extractor)
        behaviour.agent = agent

        await behaviour.run()  # must not raise
        assert behaviour.signifiers == []

    @pytest.mark.asyncio
    async def test_recording_rpc_failure_is_swallowed(self):
        agent = make_agent()
        outcome = ExecutionOutcome(success=True, final_status="completed")
        behaviour = self._behaviour(agent, context(), outcome)
        with patch("ami_agents.agents.user_assistant.signifiers.rpc_call",
                   AsyncMock(side_effect=RuntimeError("explorer down"))):
            await behaviour.run()  # must not raise
        assert behaviour.created_count is None


class TestDecoupling:
    """The seam: execution holds a context it never inspects."""

    def test_the_extractor_is_replaceable(self):
        outcome = ExecutionOutcome(success=True, final_status="completed")

        class TdSosaExtractor:
            def extract(self, ctx, out):
                return [{"source": "td-sosa", "plan": ctx.plan_id}]

        behaviour = SignifierRecordingBehaviour(context(), outcome,
                                                TdSosaExtractor())
        assert behaviour.extractor.extract(context(), outcome) == [
            {"source": "td-sosa", "plan": "plan-1"}]

    @pytest.mark.asyncio
    async def test_execution_captures_before_the_first_tick(self):
        """The order that matters: a plan changes the state that justified it."""
        agent = make_agent()
        plan = Plan(plan_id="plan-1", plan_type=PlanType.IMMEDIATE,
                    status=PlanStatus.CREATED, goal_description="turn on",
                    goal_intent="turn on", conversation_id="thread-a",
                    behavior_tree=BehaviorTreePlan.from_json_ir(
                        "plan-1", {"tree": TREE}),
                    request_id="req-1")
        agent.plans.register(plan)

        order = []
        behaviour = AchievementPlanBehaviour("plan-1", TREE, 50, 0.0)
        behaviour.agent = agent
        behaviour.kill = MagicMock()

        async def capture(*args, **kwargs):
            order.append("capture")
            return context()

        with patch("ami_agents.agents.user_assistant.behaviours."
                   "plan_execution.capture_execution_context", capture), \
             patch("ami_agents.agents.user_assistant.behaviours."
                   "plan_execution.compile_node") as compile_node:
            tree = MagicMock()
            tree.status = Status.RUNNING
            tree.tick_once = MagicMock(side_effect=lambda: order.append("tick"))
            tree.setup_with_descendants = MagicMock()
            tree.iterate = MagicMock(return_value=[])
            compile_node.return_value = tree

            behaviour.receive = AsyncMock(return_value=None)
            await behaviour.on_start()
            await behaviour.run()

        assert order == ["capture", "tick"], order

    @pytest.mark.asyncio
    async def test_finishing_spawns_a_recorder_rather_than_awaiting_one(self):
        agent = make_agent()
        plan = Plan(plan_id="plan-1", plan_type=PlanType.IMMEDIATE,
                    status=PlanStatus.CREATED, goal_description="turn on",
                    goal_intent="turn on", conversation_id="thread-a",
                    behavior_tree=BehaviorTreePlan.from_json_ir(
                        "plan-1", {"tree": TREE}),
                    request_id="req-1")
        agent.plans.register(plan)

        behaviour = AchievementPlanBehaviour("plan-1", TREE, 50, 0.0)
        behaviour.agent = agent
        behaviour.kill = MagicMock()
        behaviour._context = context()
        behaviour._setup_done = True
        tree = MagicMock()
        tree.status = Status.SUCCESS
        tree.iterate = MagicMock(return_value=[])
        tree.tick_once = MagicMock()
        behaviour.tree = tree
        behaviour.receive = AsyncMock(return_value=None)

        await behaviour.run()

        spawned = [c for c in agent.add_behaviour.call_args_list
                   if isinstance(c[0][0], SignifierRecordingBehaviour)]
        assert len(spawned) == 1
        recorder = spawned[0][0][0]
        # It got the context captured at launch, not a fresh one.
        assert recorder.context is behaviour._context
        assert recorder.outcome.success is True
