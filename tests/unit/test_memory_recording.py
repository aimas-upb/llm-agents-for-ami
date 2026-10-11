"""Memory on the executing side: what the UserAssistant remembers, and when.

    should_record   only LLM-made explicit/incomplete entries and ambiguous ones,
                    never memory replays, never without a goal structure
    snapshot        an ambiguous entry's context properties, read before the
                    plan's first action
    build           explicit: the tree; ambiguous: the context with its values
                    and what the tree used
    recording       only on success; one record to the EnvExplorer
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ami_agents.agents.user_assistant.behaviours.plan_execution import AchievementPlanBehaviour
from ami_agents.agents.user_assistant.registry import PlanRegistry, RequestRegistry
from ami_agents.agents.user_assistant.signifiers import (
    ExecutionContext,
    ExecutionOutcome,
    SignifierRecordingBehaviour,
    build_signifier,
    capture_execution_context,
)
from ami_agents.bt_planning.nodes.http_client import HTTPResponse
from ami_agents.shared.models.goal_structure import GoalStructure
from ami_agents.shared.models.plan import BehaviorTreePlan, Plan, PlanStatus, PlanType

EXPLICIT_INTENT = GoalStructure.from_dict({
    "goal_specificity": "explicit", "goal_kind": "achievement",
    "intent_text": "set the AC to cooling", "artifact_name": "ac",
    "artifact_class": "homeont:AirConditioner", "affordance_class": "homeont:SetModeCommand",
    "affordance_name": "hvacMode", "parameter_name": "NA", "goal_effect": "set",
    "target_value_text": "cooling"},
    intent_text="set the AC to cooling", structure="simple").to_wire_dict()

AMBIGUOUS_INTENT = GoalStructure.from_dict({
    "goal_specificity": "ambiguous", "goal_kind": "achievement",
    "intent_text": "the bathroom is so damp",
    "implied_environment_vars": [{
        "property_name": "relative_humidity", "property_class": "homeont:RelativeHumidity",
        "property_sensed_space": {"space_class": "homeont:Bathroom", "space_name": "Bathroom"},
        "property_change_direction": "decrease"}]},
    intent_text="the bathroom is so damp", structure="simple").to_wire_dict()

GOAL_CONTEXT = {"variables": [{
    "variable": "relative_humidity", "property_class": "homeont:RelativeHumidity",
    "room": "Bathroom",
    "actions": [{"artifact": "dehumidifier", "action": "onOff",
                 "url": "http://h/dehumidifier/actions/onOff", "direction": "affects"}],
    "properties": [
        {"artifact": "dehumidifier", "property": "onOff",
         "url": "http://h/dehumidifier/properties/onOff", "role": "acted_upon"},
        {"artifact": "sensor", "property": "humidity",
         "url": "http://h/sensor/properties/humidity", "role": "reading"}]}]}

TREE = {"type": "action", "name": "ac.hvacMode", "action_url": "http://h/ac/hvacMode",
        "parameters": {"value": 3}}

DAMP_TREE = {"type": "sequence", "name": "dry", "children": [
    {"type": "action", "name": "on", "action_url": "http://h/dehumidifier/actions/onOff",
     "parameters": {"value": True}},
    {"type": "wait_condition", "name": "drier",
     "property_url": "http://h/sensor/properties/humidity",
     "operator": "<", "expected_value": 55}]}


def _context(intent, entry, tree=TREE, **fields):
    return ExecutionContext(plan_id="plan-1", request_id="req-1", thread="t", tree_spec=tree,
                            entry=entry, intent=intent, **fields)


class TestShouldRecord:
    def test_an_explicit_plan_only_when_a_model_made_it(self):
        assert _context(EXPLICIT_INTENT, {"source": "explicit_llm", "llm": True}).should_record
        assert not _context(EXPLICIT_INTENT, {"source": "explicit_deterministic"}).should_record

    def test_never_a_memory_replay(self):
        assert not _context(EXPLICIT_INTENT, {"source": "memory", "llm": True}).should_record

    def test_an_ambiguous_plan_with_its_context(self):
        entry = {"source": "ambiguous_llm", "llm": True, "goal_context": GOAL_CONTEXT}
        assert _context(AMBIGUOUS_INTENT, entry).should_record
        assert not _context(AMBIGUOUS_INTENT, {"llm": True}).should_record
        assert not _context(AMBIGUOUS_INTENT, entry, snapshot_failed=True).should_record

    def test_not_without_a_goal_structure(self):
        assert not _context({"text_intent": "x"}, {"llm": True}).should_record


class TestBuild:
    def test_an_explicit_record_keeps_the_whole_plan(self):
        record = build_signifier(_context(EXPLICIT_INTENT, {"llm": True}))
        assert record["kind"] == "explicit" and record["tree"] == TREE
        assert record["key"]["artifact_name"] == "ac"
        assert "context" not in record

    def test_an_ambiguous_record_keeps_the_context_with_values_and_what_was_used(self):
        snapshot = {"http://h/dehumidifier/properties/onOff": False,
                    "http://h/sensor/properties/humidity": 68}
        record = build_signifier(_context(
            AMBIGUOUS_INTENT, {"llm": True, "goal_context": GOAL_CONTEXT},
            tree=DAMP_TREE, snapshot=snapshot))
        assert record["kind"] == "ambiguous" and "tree" not in record
        values = {p["property"]: p["value"]
                  for p in record["context"]["variables"][0]["properties"]}
        assert values == {"onOff": False, "humidity": 68}
        assert record["used"] == {
            "actions": [{"url": "http://h/dehumidifier/actions/onOff",
                         "parameters": {"value": True},
                         "artifact": "dehumidifier", "action": "onOff"}],
            "properties": [{"url": "http://h/sensor/properties/humidity",
                            "artifact": "sensor", "property": "humidity"}]}
        # The entry's own context is not changed by adding the values.
        assert "value" not in GOAL_CONTEXT["variables"][0]["properties"][0]


class _FakeHTTP:
    """Records every request, in order; GETs answer from `values`."""

    def __init__(self, events, values):
        self.events, self.values = events, values

    def get(self, url, headers=None):
        self.events.append(("GET", url))
        return HTTPResponse(200, self.values.get(url), {}, url, 0.0)

    def post(self, url, payload=None, headers=None):
        self.events.append(("POST", url))
        return HTTPResponse(200, {}, {}, url, 0.0)


class TestSnapshotBeforeThePlan:
    @pytest.mark.asyncio
    async def test_capture_reads_every_context_property(self):
        events = []
        http = _FakeHTTP(events, {"http://h/sensor/properties/humidity": 68})
        context = await capture_execution_context(
            MagicMock(), "plan-1", DAMP_TREE, intent=AMBIGUOUS_INTENT,
            entry={"llm": True, "goal_context": GOAL_CONTEXT}, http_client=http)
        assert context.snapshot == {"http://h/dehumidifier/properties/onOff": None,
                                    "http://h/sensor/properties/humidity": 68}
        assert sorted(url for _, url in events) == [
            "http://h/dehumidifier/properties/onOff", "http://h/sensor/properties/humidity"]

    @pytest.mark.asyncio
    async def test_every_snapshot_read_happens_before_the_first_action(self):
        events = []
        values = {"http://h/sensor/properties/humidity": 68}
        agent = MagicMock()
        agent.plans, agent.requests = PlanRegistry(), RequestRegistry()
        agent.notify_conversation = AsyncMock()
        agent.plans.register(Plan(
            plan_id="plan-1", plan_type=PlanType.IMMEDIATE, status=PlanStatus.CREATED,
            goal_description="the bathroom is so damp", goal_intent="x",
            conversation_id="t", request_id="req-1",
            behavior_tree=BehaviorTreePlan.from_json_ir("plan-1", {"tree": DAMP_TREE})))
        behaviour = AchievementPlanBehaviour(
            "plan-1", DAMP_TREE, max_ticks=20, tick_period=0.0,
            recording_options={"entry": {"llm": True, "goal_context": GOAL_CONTEXT},
                               "intent": AMBIGUOUS_INTENT})
        behaviour.agent = agent
        behaviour.send = AsyncMock()
        behaviour.kill = MagicMock()
        behaviour.receive = AsyncMock(return_value=None)

        module = "ami_agents.agents.user_assistant.signifiers"
        with patch(f"{module}.HTTPClient", lambda config=None: _FakeHTTP(events, values)), \
             patch("ami_agents.bt_planning.nodes.affordance_nodes.HTTPClient",
                   lambda config=None: _FakeHTTP(events, values)):
            await behaviour.on_start()
            for _ in range(20):
                if any(kind == "POST" for kind, _ in events):
                    break
                await behaviour.run()
                await asyncio.sleep(0.01)

        first_post = next(i for i, (kind, _) in enumerate(events) if kind == "POST")
        reads = [i for i, (kind, url) in enumerate(events)
                 if kind == "GET" and url in ("http://h/dehumidifier/properties/onOff",
                                              "http://h/sensor/properties/humidity")]
        assert len(reads) >= 2 and max(reads[:2]) < first_post
        assert behaviour._context.snapshot["http://h/sensor/properties/humidity"] == 68


class TestRecording:
    def _behaviour(self, context, success):
        behaviour = SignifierRecordingBehaviour(
            context, ExecutionOutcome(success=success, final_status="x"))
        behaviour.agent = SimpleNamespace(target_jids={"explorer": "ee@x"},
                                          rpc_call_timeout=1.0, community_client=None)
        return behaviour

    @pytest.mark.asyncio
    async def test_a_successful_llm_plan_is_sent_to_the_explorer(self):
        behaviour = self._behaviour(_context(EXPLICIT_INTENT, {"llm": True}), True)
        reply = SimpleNamespace(body='{"ok": true, "signifier_id": "sig-9"}')
        with patch("ami_agents.agents.user_assistant.signifiers.rpc_call",
                   AsyncMock(return_value=reply)) as rpc:
            await behaviour.run()
        body = rpc.await_args.kwargs["body"]
        assert body["signifier"]["kind"] == "explicit" and body["signifier"]["tree"] == TREE
        assert behaviour.recorded_id == "sig-9"

    @pytest.mark.asyncio
    async def test_a_failed_or_unworthy_plan_is_not(self):
        for context, success in ((_context(EXPLICIT_INTENT, {"llm": True}), False),
                                 (_context(EXPLICIT_INTENT, {"source": "memory"}), True)):
            behaviour = self._behaviour(context, success)
            with patch("ami_agents.agents.user_assistant.signifiers.rpc_call",
                       AsyncMock()) as rpc:
                await behaviour.run()
            rpc.assert_not_awaited()
