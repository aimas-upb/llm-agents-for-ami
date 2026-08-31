"""Request and plan identity for the UserAssistant."""

import json

import pytest

from ami_agents.agents.user_assistant.registry import (
    PlanRegistry,
    RequestRegistry,
)
from ami_agents.shared.models.messages import (
    META_PLAN_ID,
    META_REQUEST_ID,
    MessageType,
    new_plan_id,
    new_request_id,
)
from ami_agents.shared.models.plan import (
    BehaviorTreePlan,
    Plan,
    PlanStatus,
    PlanType,
)


def make_plan(plan_id="plan-1", request_id="req-1",
              plan_type=PlanType.IMMEDIATE, goal="turn on the kitchen light"):
    return Plan(
        plan_id=plan_id,
        plan_type=plan_type,
        status=PlanStatus.CREATED,
        goal_description=goal,
        goal_intent="kitchen light on",
        conversation_id="thread-a",
        behavior_tree=BehaviorTreePlan.from_json_ir(
            plan_id, {"tree": {"type": "action", "action_url": "http://x/1"}}),
        request_id=request_id,
        plan_hash="hash-abc",
    )


class TestIds:
    def test_ids_are_distinguishable_and_unique(self):
        assert new_request_id().startswith("req-")
        assert new_plan_id().startswith("plan-")
        assert len({new_request_id() for _ in range(50)}) == 50
        assert len({new_plan_id() for _ in range(50)}) == 50

    def test_routing_keys_are_plain_strings(self):
        # They travel as SPADE metadata, which is str -> str.
        assert isinstance(META_REQUEST_ID, str)
        assert isinstance(META_PLAN_ID, str)

    def test_plan_management_message_types_exist(self):
        for name in ("PLAN_EXECUTE_REQUEST", "PLAN_STATUS_REQUEST",
                     "PLAN_EXPLAIN_REQUEST", "PLAN_EXPLAIN_RESPONSE",
                     "PLAN_CANCEL_REQUEST", "PLAN_EXECUTION_STATUS"):
            assert isinstance(getattr(MessageType, name).value, str)


class TestRequestRegistry:
    def test_open_get_close(self):
        reg = RequestRegistry()
        record = reg.open("req-1", "thread-a", "turn on the light")
        assert record.is_open
        assert reg.get("req-1") is record
        assert reg.open_requests() == [record]

        closed = reg.close("req-1", "planned")
        assert closed.outcome == "planned"
        assert not closed.is_open
        assert reg.open_requests() == []

    def test_keeps_the_user_phrasing_verbatim(self):
        reg = RequestRegistry()
        text = "keep the living room at 25 degrees, please"
        reg.open("req-1", "thread-a", text)
        assert reg.get("req-1").user_text == text

    def test_links_plans_to_the_request_that_produced_them(self):
        reg = RequestRegistry()
        reg.open("req-1", "thread-a", "two things at once")
        reg.note_plan("req-1", "plan-1")
        reg.note_plan("req-1", "plan-2")
        reg.note_plan("req-1", "plan-1")  # idempotent
        assert reg.get("req-1").plan_ids == ["plan-1", "plan-2"]

    def test_unknown_ids_do_not_raise(self):
        reg = RequestRegistry()
        assert reg.get("nope") is None
        assert reg.close("nope", "planned") is None
        reg.note_plan("nope", "plan-1")
        reg.note_intents("nope", [{"span": "x"}])

    def test_to_jsonl_round_trips(self):
        reg = RequestRegistry()
        reg.open("req-1", "thread-a", "turn on the light")
        reg.note_intents("req-1", [{"span": "turn on the light",
                                    "category": "GOAL_REQUEST"}])
        reg.close("req-1", "planned")
        rows = [json.loads(l) for l in reg.to_jsonl().splitlines()]
        assert len(rows) == 1
        assert rows[0]["request_id"] == "req-1"
        assert rows[0]["user_text"] == "turn on the light"
        assert rows[0]["outcome"] == "planned"


class TestPlanRegistry:
    def test_register_and_snapshot_carries_provenance(self):
        reg = PlanRegistry()
        record = reg.register(make_plan())
        snap = record.snapshot()
        assert snap["plan_id"] == "plan-1"
        assert snap["request_id"] == "req-1"
        # The user's own words survive into the running plan.
        assert snap["goal_description"] == "turn on the kitchen light"
        assert snap["plan_hash"] == "hash-abc"
        assert snap["status"] == "created"

    def test_tick_accounting(self):
        reg = PlanRegistry()
        reg.register(make_plan())
        reg.mark_started("plan-1")
        assert reg.get("plan-1").plan.status is PlanStatus.RUNNING
        assert reg.bump_tick("plan-1", "RUNNING") == 1
        assert reg.bump_tick("plan-1", "RUNNING") == 2
        assert reg.get("plan-1").ticks == 2
        assert reg.get("plan-1").last_status == "RUNNING"

    def test_finishing_an_achievement_plan_counts_one_execution(self):
        reg = PlanRegistry()
        reg.register(make_plan())
        reg.finish("plan-1", PlanStatus.COMPLETED)
        record = reg.get("plan-1")
        assert record.plan.status is PlanStatus.COMPLETED
        assert record.plan.execution_count == 1
        assert record.finished_at is not None
        assert not record.is_active

    def test_maintenance_plan_rearms_without_finishing(self):
        reg = PlanRegistry()
        reg.register(make_plan(plan_type=PlanType.MAINTENANCE,
                               goal="keep the living room at 25 degrees"))
        reg.mark_started("plan-1")
        reg.bump_tick("plan-1")
        assert reg.bump_execution_count("plan-1") == 1
        record = reg.get("plan-1")
        # Re-armed: the round is counted, the tick budget resets, and the plan
        # stays active rather than completing.
        assert record.ticks == 0
        assert record.is_maintenance
        assert record.is_active
        assert reg.active() == [record]

    def test_cancel_leaves_it_inactive_with_no_execution_counted(self):
        reg = PlanRegistry()
        reg.register(make_plan())
        reg.mark_started("plan-1")
        reg.finish("plan-1", PlanStatus.CANCELLED)
        record = reg.get("plan-1")
        assert record.plan.status is PlanStatus.CANCELLED
        assert record.plan.execution_count == 0
        assert not record.is_active

    def test_failure_records_the_error(self):
        reg = PlanRegistry()
        reg.register(make_plan())
        reg.finish("plan-1", PlanStatus.FAILED, error="http 500")
        assert reg.get("plan-1").error == "http 500"
        assert reg.get("plan-1").plan.execution_count == 0

    def test_lookup_by_request_and_conversation(self):
        reg = PlanRegistry()
        reg.register(make_plan("plan-1", "req-1"))
        reg.register(make_plan("plan-2", "req-1"))
        reg.register(make_plan("plan-3", "req-2"))
        assert {r.plan_id for r in reg.for_request("req-1")} == {"plan-1", "plan-2"}
        assert {r.plan_id for r in reg.for_request("req-2")} == {"plan-3"}
        assert len(reg.for_conversation("thread-a")) == 3

    def test_active_excludes_finished(self):
        reg = PlanRegistry()
        reg.register(make_plan("plan-1"))
        reg.register(make_plan("plan-2"))
        reg.mark_started("plan-1")
        reg.finish("plan-2", PlanStatus.COMPLETED)
        assert [r.plan_id for r in reg.active()] == ["plan-1"]

    def test_unknown_ids_do_not_raise(self):
        reg = PlanRegistry()
        assert reg.get("nope") is None
        assert reg.bump_tick("nope") == 0
        assert reg.bump_execution_count("nope") == 0
        assert reg.finish("nope", PlanStatus.COMPLETED) is None
        reg.mark_started("nope")

    def test_to_jsonl_round_trips(self):
        reg = PlanRegistry()
        reg.register(make_plan())
        reg.mark_started("plan-1")
        rows = [json.loads(l) for l in reg.to_jsonl().splitlines()]
        assert len(rows) == 1
        assert rows[0]["plan_id"] == "plan-1"
        assert rows[0]["goal_description"] == "turn on the kitchen light"


class TestIdentityAcrossRegistries:
    def test_a_plan_traces_back_to_the_utterance_that_asked_for_it(self):
        requests, plans = RequestRegistry(), PlanRegistry()
        request_id, plan_id = new_request_id(), new_plan_id()

        requests.open(request_id, "thread-a", "turn on the kitchen light")
        plans.register(make_plan(plan_id, request_id))
        requests.note_plan(request_id, plan_id)

        # From a running plan, back to the user's words.
        record = plans.get(plan_id)
        assert record.request_id == request_id
        assert requests.get(record.request_id).user_text == \
            "turn on the kitchen light"

        # And forward again, from the request to what it started.
        assert requests.get(request_id).plan_ids == [plan_id]

    def test_same_plan_text_run_twice_gives_two_ids_and_one_hash(self):
        plans = PlanRegistry()
        first = plans.register(make_plan("plan-1", "req-1"))
        second = plans.register(make_plan("plan-2", "req-2"))
        assert first.plan_id != second.plan_id
        assert first.plan.plan_hash == second.plan.plan_hash
