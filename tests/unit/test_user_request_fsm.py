"""The per-request FSM: structure, the mandatory self-loop, and confirmation."""

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ami_agents.agents.user_assistant.behaviours.user_request import (
    AWAITING_CONFIRMATION,
    AWAITING_PLAN,
    DONE,
    EXTRACTING,
    SEGMENTING,
    SUBMITTING,
    SUMMARIZING,
    UserRequestBehaviour,
)
from ami_agents.agents.user_assistant.models import ConversationState
from ami_agents.agents.user_assistant.registry import PlanRegistry, RequestRegistry


def make_agent(confirmation_timeout=5.0):
    agent = MagicMock()
    agent.requests = RequestRegistry()
    agent.plans = PlanRegistry()
    agent.confirmation_timeout = confirmation_timeout
    agent.goal_request_timeout = 30.0
    agent._conversations = {}
    conv = ConversationState()
    agent.get_conversation = MagicMock(return_value=conv)
    return agent


def make_fsm(agent=None, text="turn on the kitchen light"):
    agent = agent or make_agent()
    msg = MagicMock()
    msg.body = text
    msg.thread = "thread-a"
    fsm = UserRequestBehaviour("req-1", "thread-a", text, msg)
    fsm.agent = agent
    fsm.send = AsyncMock()
    agent.requests.open("req-1", "thread-a", text)
    return fsm, agent


class TestStructure:
    def test_states_and_initial(self):
        fsm, _ = make_fsm()
        assert set(fsm._states) == {
            SEGMENTING, EXTRACTING, AWAITING_PLAN, SUMMARIZING,
            AWAITING_CONFIRMATION, SUBMITTING, DONE}
        assert fsm.current_state == SEGMENTING

    def test_the_confirmation_self_loop_is_registered(self):
        """Without it, `is_valid_transition` raises and kills the FSM."""
        fsm, _ = make_fsm()
        assert fsm.is_valid_transition(AWAITING_CONFIRMATION, AWAITING_CONFIRMATION)

    def test_every_state_can_reach_done(self):
        fsm, _ = make_fsm()
        for state in (SEGMENTING, EXTRACTING, AWAITING_PLAN, SUMMARIZING,
                      AWAITING_CONFIRMATION, SUBMITTING):
            assert fsm.is_valid_transition(state, DONE), state

    def test_states_hold_a_reference_to_the_machine(self):
        # SPADE hands a State only `agent` and `receive`, never its parent.
        fsm, _ = make_fsm()
        for state in fsm._states.values():
            assert state.request is fsm


class TestConfirmationWaiting:
    """The self-loop is what lets the request wait without dying."""

    @pytest.mark.asyncio
    async def test_survives_many_idle_polls(self):
        fsm, _ = make_fsm(make_agent(confirmation_timeout=3600))
        state = fsm._states[AWAITING_CONFIRMATION]
        state.receive = AsyncMock(return_value=None)
        fsm.start_confirmation_window()

        for _ in range(10):
            await state.run()
            assert state.next_state == AWAITING_CONFIRMATION
            assert fsm.is_valid_transition(AWAITING_CONFIRMATION, state.next_state)

    @pytest.mark.asyncio
    async def test_yes_submits(self):
        fsm, _ = make_fsm(make_agent(confirmation_timeout=3600))
        state = fsm._states[AWAITING_CONFIRMATION]
        reply = MagicMock(); reply.body = "yes"
        state.receive = AsyncMock(return_value=reply)
        fsm.start_confirmation_window()

        await state.run()
        assert state.next_state == SUBMITTING

    @pytest.mark.asyncio
    async def test_no_rejects_and_ends(self):
        fsm, agent = make_fsm(make_agent(confirmation_timeout=3600))
        state = fsm._states[AWAITING_CONFIRMATION]
        reply = MagicMock(); reply.body = "no"
        reply.make_reply = MagicMock(return_value=MagicMock())
        state.receive = AsyncMock(return_value=reply)
        fsm.start_confirmation_window()

        await state.run()
        assert state.next_state == DONE
        assert agent.requests.get("req-1").outcome == "rejected"

    @pytest.mark.asyncio
    async def test_timeout_assumes_yes(self):
        fsm, _ = make_fsm(make_agent(confirmation_timeout=0.0))
        state = fsm._states[AWAITING_CONFIRMATION]
        state.receive = AsyncMock(return_value=None)
        fsm.start_confirmation_window()

        await state.run()
        assert state.next_state == SUBMITTING

    @pytest.mark.asyncio
    async def test_unrelated_text_keeps_waiting(self):
        fsm, _ = make_fsm(make_agent(confirmation_timeout=3600))
        state = fsm._states[AWAITING_CONFIRMATION]
        reply = MagicMock(); reply.body = "what's the weather"
        reply.make_reply = MagicMock(return_value=MagicMock())
        state.receive = AsyncMock(return_value=reply)
        fsm.start_confirmation_window()

        await state.run()
        assert state.next_state == AWAITING_CONFIRMATION


class TestAutoConfirm:
    """A plan that only reads needs no permission."""

    def test_read_only_plan_auto_confirms(self):
        plan = {"tree": {"type": "condition", "property_url": "http://x/p",
                         "expected_value": 1}}
        assert UserRequestBehaviour.auto_confirms(plan) is True

    def test_plan_with_an_action_does_not(self):
        plan = {"tree": {"type": "sequence", "children": [
            {"type": "condition", "property_url": "http://x/p", "expected_value": 1},
            {"type": "action", "action_url": "http://x/a"}]}}
        assert UserRequestBehaviour.auto_confirms(plan) is False

    def test_nested_action_is_found(self):
        plan = {"tree": {"type": "selector", "children": [
            {"type": "sequence", "children": [
                {"type": "parallel", "children": [
                    {"type": "action", "action_url": "http://x/a"}]}]}]}}
        assert UserRequestBehaviour.auto_confirms(plan) is False

    def test_multi_plan_envelope_checks_every_plan(self):
        plan = {"plans": [
            {"tree": {"type": "condition", "property_url": "http://x/p",
                      "expected_value": 1}},
            {"tree": {"type": "action", "action_url": "http://x/a"}}]}
        assert UserRequestBehaviour.auto_confirms(plan) is False


class TestSubmission:
    @pytest.mark.asyncio
    async def test_submitting_hands_the_plan_over_as_a_message(self):
        fsm, agent = make_fsm()
        fsm.plan_obj = {"tree": {"type": "action", "action_url": "http://x/a"}}
        fsm.plan_hash = "hash-abc"
        agent.jid = "ua@localhost"

        state = fsm._states[SUBMITTING]
        await state.run()

        assert state.next_state == DONE
        fsm.send.assert_awaited_once()
        sent = fsm.send.await_args[0][0]
        body = json.loads(sent.body)
        # The user's own words travel with the plan.
        assert body["goal_description"] == "turn on the kitchen light"
        assert body["request_id"] == "req-1"
        assert body["plan_hash"] == "hash-abc"
        assert body["plan_id"].startswith("plan-")
        # And the request records what it started.
        assert agent.requests.get("req-1").plan_ids == [body["plan_id"]]
        assert agent.requests.get("req-1").outcome == "planned"


class TestSegmenting:
    @pytest.mark.asyncio
    async def test_unparseable_input_ends_the_request(self):
        fsm, agent = make_fsm()
        state = fsm._states[SEGMENTING]
        with patch("ami_agents.agents.user_assistant.behaviours.user_request.pipeline") as pl:
            pl.fetch_capabilities = AsyncMock(return_value="{}")
            pl.filter_capabilities_json = MagicMock(return_value={})
            pl.segment_into_atomic_intents = AsyncMock(return_value=[])
            await state.run()

        assert state.next_state == DONE
        assert agent.requests.get("req-1").outcome == "failed"

    @pytest.mark.asyncio
    async def test_segmented_intents_are_recorded_and_advance(self):
        fsm, agent = make_fsm()
        intent = MagicMock(span="turn on the light", category="GOAL_REQUEST",
                           reason="asks for an action")
        state = fsm._states[SEGMENTING]
        with patch("ami_agents.agents.user_assistant.behaviours.user_request.pipeline") as pl:
            pl.fetch_capabilities = AsyncMock(return_value="{}")
            pl.filter_capabilities_json = MagicMock(return_value={})
            pl.segment_into_atomic_intents = AsyncMock(return_value=[intent])
            await state.run()

        assert state.next_state == EXTRACTING
        recorded = agent.requests.get("req-1").atomic_intents
        assert recorded == [{"span": "turn on the light",
                             "category": "GOAL_REQUEST",
                             "reason": "asks for an action"}]
