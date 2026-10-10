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
from ami_agents.shared.models.goal_structure import GoalStructure
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
            pl.segment_into_atomic_intents = AsyncMock(return_value=[])
            await state.run()

        assert state.next_state == DONE
        assert agent.requests.get("req-1").outcome == "failed"

    @pytest.mark.asyncio
    async def test_segmented_intents_are_recorded_and_advance(self):
        fsm, agent = make_fsm()
        intent = MagicMock(text="turn on the light", type="GOAL_REQUEST",
                           reason="asks for an action",
                           qualifiers=["incomplete", "achievement"])
        state = fsm._states[SEGMENTING]
        with patch("ami_agents.agents.user_assistant.behaviours.user_request.pipeline") as pl:
            pl.segment_into_atomic_intents = AsyncMock(return_value=[intent])
            await state.run()

        assert state.next_state == EXTRACTING
        recorded = agent.requests.get("req-1").atomic_intents
        assert recorded == [{"text": "turn on the light",
                             "type": "GOAL_REQUEST",
                             "qualifiers": ["incomplete", "achievement"],
                             "reason": "asks for an action"}]


class _FakeStructuring:
    """Stands in for a structuring OneShotBehaviour: done as soon as joined."""

    def __init__(self, intent_text, logger=None):
        self.intent_text = intent_text
        self.error = None
        self.result = {"text_intent": intent_text,
                       "request_performative": "query_if",
                       "device_property": {"class": "homeont:LevelControlBrightness"}}

    async def join(self):
        return None


class TestExtracting:
    @pytest.mark.asyncio
    async def test_a_capability_question_is_structured_and_answered(self):
        """No goal, so nothing to plan: structured, answered, done."""
        fsm, agent = make_fsm(text="can you dim the kitchen lights")
        fsm.atomic_intents = [MagicMock(text="can you dim the kitchen lights",
                                        type="ENV_CAPABILITIES_REQUEST")]
        fsm.answer_capabilities_query = AsyncMock()
        state = fsm._states[EXTRACTING]
        with patch("ami_agents.agents.user_assistant.behaviours.user_request."
                   "EnvCapabilityStructuringBehaviour", _FakeStructuring):
            await state.run()

        agent.add_behaviour.assert_called_once()
        fsm.answer_capabilities_query.assert_awaited_once()
        sent = fsm.answer_capabilities_query.await_args[0][0]
        assert sent["request_performative"] == "query_if"
        assert sent["device_property"]["class"] == "homeont:LevelControlBrightness"
        assert state.next_state == DONE
        assert agent.requests.get("req-1").outcome == "answered"

    @pytest.mark.asyncio
    async def test_a_goal_without_context_is_not_structured(self):
        """No environment view, nothing to structure against: no fallback, and
        no "rephrase" either -- the wording was not the problem."""
        fsm, agent = make_fsm()
        fsm.atomic_intents = [MagicMock(text="turn on the kitchen light",
                                        type="GOAL_REQUEST", qualifiers=[])]
        state = fsm._states[EXTRACTING]
        _FakeGoalStructuring.outcomes = {"turn on the kitchen light": None}
        with patch(f"{FSM_MODULE}.GoalStructuringBehaviour", _FakeGoalStructuring):
            await state.run()

        assert fsm.send.await_count == 1
        assert "turn on the kitchen light" in fsm.send.await_args[0][0].body
        assert state.next_state == DONE
        assert agent.requests.get("req-1").outcome == "failed"

    @pytest.mark.asyncio
    async def test_each_goal_is_structured_with_its_own_qualifiers(self):
        fsm, _ = make_fsm()
        fsm.atomic_intents = [
            MagicMock(text="turn on the kitchen light", type="GOAL_REQUEST",
                      qualifiers=["explicit", "achievement"]),
            MagicMock(text="when washer 1 finishes start the dryer",
                      type="GOAL_REQUEST",
                      qualifiers=["explicit", "logical_dependency", "achievement"]),
        ]
        _FakeGoalStructuring.outcomes = {
            i.text: GoalStructure.from_dict({"goal_specificity": "explicit"},
                                            intent_text=i.text)
            for i in fsm.atomic_intents}
        _FakeGoalStructuring.seen = []
        state = fsm._states[EXTRACTING]
        with patch(f"{FSM_MODULE}.GoalStructuringBehaviour", _FakeGoalStructuring):
            await state.run()

        assert _FakeGoalStructuring.seen == [
            ("turn on the kitchen light", ["explicit", "achievement"]),
            ("when washer 1 finishes start the dryer",
             ["explicit", "logical_dependency", "achievement"])]
        assert [s.intent_text for s in fsm.parsed_intents] == [
            i.text for i in fsm.atomic_intents]
        assert state.next_state == AWAITING_PLAN


class TestClarification:
    def _structure(self, text, **goal):
        return GoalStructure.from_dict(goal, intent_text=text, structure="simple")

    @pytest.mark.asyncio
    async def test_a_goal_naming_no_device_is_asked_about_not_planned(self):
        fsm, agent = make_fsm(text="turn it off")
        fsm.atomic_intents = [MagicMock(text="turn it off", type="GOAL_REQUEST",
                                        qualifiers=["incomplete", "achievement"])]
        _FakeGoalStructuring.outcomes = {
            "turn it off": self._structure("turn it off", goal_specificity="incomplete")}
        state = fsm._states[EXTRACTING]
        with patch(f"{FSM_MODULE}.GoalStructuringBehaviour", _FakeGoalStructuring):
            await state.run()

        assert fsm.parsed_intents == []
        body = fsm.send.await_args[0][0].body
        assert '"turn it off"' in body and "Which device" in body
        assert state.next_state == DONE
        assert agent.requests.get("req-1").outcome == "clarification_needed"

    @pytest.mark.asyncio
    async def test_the_other_goals_still_go_to_planning(self):
        fsm, _ = make_fsm(text="turn on the kitchen light and turn it off")
        fsm.atomic_intents = [
            MagicMock(text="turn on the kitchen light", type="GOAL_REQUEST",
                      qualifiers=["incomplete", "achievement"]),
            MagicMock(text="turn it off", type="GOAL_REQUEST",
                      qualifiers=["incomplete", "achievement"])]
        _FakeGoalStructuring.outcomes = {
            "turn on the kitchen light": self._structure(
                "turn on the kitchen light", goal_specificity="incomplete",
                location_class="homeont:Kitchen", artifact_class="homeont:Light",
                affordance_class="homeont:SetOnOffCommand"),
            "turn it off": self._structure("turn it off", goal_specificity="incomplete")}
        state = fsm._states[EXTRACTING]
        with patch(f"{FSM_MODULE}.GoalStructuringBehaviour", _FakeGoalStructuring):
            await state.run()

        assert [s.intent_text for s in fsm.parsed_intents] == ["turn on the kitchen light"]
        assert '"turn it off"' in fsm.send.await_args[0][0].body
        assert state.next_state == AWAITING_PLAN


class TestClarificationEntries:
    """The solver's reply may ask about some goals: ask, and confirm the rest."""

    PLAN = {"type": "action", "name": "a", "action_url": "http://h/a"}

    def _fsm_with(self, *entries):
        fsm, agent = make_fsm()
        fsm.plan_body = json.dumps({"plan_type": "bt", "plans": list(entries),
                                    "requires_clarification": True})
        return fsm, agent

    def _ask(self, text):
        return {"tree": None, "requires_clarification": True,
                "explanation": "Which device do you mean: lamp or ceiling_light?",
                "intent": {"text_intent": text}}

    @pytest.mark.asyncio
    async def test_the_question_is_asked_and_the_rest_is_summarised(self):
        fsm, _ = self._fsm_with(self._ask("turn on the bedroom light"),
                                {"tree": self.PLAN, "intent": {"text_intent": "x"}})
        sent = []
        fsm.send = AsyncMock(side_effect=lambda msg: sent.append(msg.body))
        state = fsm._states[SUMMARIZING]
        with patch(f"{FSM_MODULE}.pipeline.summarize_plan",
                   AsyncMock(return_value="I will do x.")) as summarize:
            await state.run()

        assert sent == ['About "turn on the bedroom light": '
                        'Which device do you mean: lamp or ceiling_light?',
                        "I will do x."]
        summarised = json.loads(summarize.await_args[0][2])
        assert [p["intent"]["text_intent"] for p in summarised["plans"]] == ["x"]
        assert "requires_clarification" not in summarised
        assert state.next_state == AWAITING_CONFIRMATION

    @pytest.mark.asyncio
    async def test_only_questions_end_the_request(self):
        fsm, agent = self._fsm_with(self._ask("turn on the bedroom light"))
        state = fsm._states[SUMMARIZING]
        with patch(f"{FSM_MODULE}.pipeline.summarize_plan", AsyncMock()) as summarize:
            await state.run()
        summarize.assert_not_awaited()
        assert state.next_state == DONE
        assert agent.requests.get("req-1").outcome == "clarification_needed"


class TestAwaitingPlan:
    @pytest.mark.asyncio
    async def test_goal_request_carries_the_structures(self):
        fsm, agent = make_fsm()
        agent.target_jids = {"solver": "solver@localhost"}
        fsm.parsed_intents = [GoalStructure.from_dict(
            {"goal_specificity": "explicit", "artifact_name": "kitchen_on_off_light_1"},
            intent_text="turn on the kitchen light", structure="simple")]
        state = fsm._states[AWAITING_PLAN]
        with patch(f"{FSM_MODULE}.rpc_call",
                   AsyncMock(return_value=MagicMock(body="{}"))) as rpc:
            await state.run()

        body = rpc.await_args.kwargs["body"]
        (intent,) = body["intents"]
        assert intent["text_intent"] == "turn on the kitchen light"
        assert intent["goals"]["G1"]["artifact_name"] == "kitchen_on_off_light_1"
        assert state.next_state == SUMMARIZING


FSM_MODULE = "ami_agents.agents.user_assistant.behaviours.user_request"


class _FakeGoalStructuring:
    """Stands in for GoalStructuringBehaviour: done as soon as joined.

    `outcomes` maps a goal's text to its GoalStructure, or None for a goal
    whose context could not be had.
    """

    outcomes: dict = {}
    seen: list = []

    def __init__(self, intent_text, qualifiers=None, logger=None):
        self.intent_text = intent_text
        type(self).seen.append((intent_text, list(qualifiers or [])))
        self.result = type(self).outcomes.get(intent_text)
        self.context_unavailable = self.result is None
        self.error = "goal context unavailable" if self.result is None else None

    async def join(self):
        return None
