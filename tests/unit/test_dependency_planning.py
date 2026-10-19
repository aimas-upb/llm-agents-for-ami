"""The dependency planner end to end on its side of the wire, and where it plugs in.

    DependencyGoalPlanningBehaviour   steps -> predicates -> profile -> compile,
                                      with fake achievement planners and a fake
                                      profiling model
    routing                           dependency structures go to it
    DependencyPlanBehaviour (UA)      no tick budget; the runtime refreshed first
    SHTD /_shtd/time                  the simulator's clock, proxied
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from py_trees.common import Status

from ami_agents.agents.interaction_solver.behaviours.dependency_goal_planning import (
    DependencyGoalPlanningBehaviour,
)
from ami_agents.agents.interaction_solver.utils.dependency_context import dependency_structure
from ami_agents.shared.models.goal_structure import GoalStructure
from tests.unit.test_dependency_compiler import HOME, LIGHT_ON, WASHER_DONE, _goal  # noqa: F401

MODULE = "ami_agents.agents.interaction_solver.behaviours.dependency_goal_planning"


@pytest.fixture(scope="module")
def graph():
    from rdflib import Graph
    from ami_agents.shared.utils.vocabulary import close_types
    g = Graph().parse(data=HOME, format="turtle")
    close_types(g)
    return g


class _FakePlanner:
    """Stands in for the achievement planners: one action per goal, on light_1."""

    seen = []
    result = None

    def __init__(self, goals, logger=None, graph=None):
        type(self).seen.extend(g.intent_text for g in goals)
        self.results = [type(self).result or {
            "tree": {"type": "action", "name": g.intent_text,
                     "action_url": "http://h/light_1/a/onOff", "parameters": {"value": True}},
            "explanation": "x"} for g in goals]

    async def join(self):
        return None


def _agent(model_answer=None, error=None):
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        if error:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(model_answer)))])

    agent = MagicMock()
    agent.config = {"planning": {"dependency": {"poll_seconds": "10"}}}
    agent.explicit_llm = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        model="gpt-5-nano", temperature=0.0, reasoning_effort="low", max_completion_tokens=None)
    return agent, seen


async def _plan(graph, structure, agent, planner_result=None):
    _FakePlanner.seen, _FakePlanner.result = [], planner_result
    behaviour = DependencyGoalPlanningBehaviour([structure])
    behaviour.agent = agent
    with patch(f"{MODULE}.fetch_environment_graph", AsyncMock(return_value=graph)), \
         patch(f"{MODULE}.ExplicitGoalPlanningBehaviour", _FakePlanner), \
         patch(f"{MODULE}.IncompleteGoalPlanningBehaviour", _FakePlanner), \
         patch(f"{MODULE}.AmbiguousGoalPlanningBehaviour", _FakePlanner):
        await behaviour.run()
    (entry,) = behaviour.results
    return entry


def _structure(goals, predicates=None):
    return GoalStructure.from_dict({"goals": goals, "predicates": predicates or {}},
                                   intent_text="the request", structure="dependency")


NAMED = dict(artifact_name="light_1", artifact_class="homeont:OnOffLight",
             affordance_name="onOff", affordance_class="homeont:SetOnOffCommand",
             goal_effect="set", target_value_determined="true")


class TestBehaviour:
    @pytest.mark.asyncio
    async def test_no_questions_no_model_call(self, graph):
        agent, seen = _agent()
        entry = await _plan(graph, _structure(
            {"G1": _goal(intent_text="turn on light 1", timing={"starts": "on_trigger"}, **NAMED)},
            {"P1": {**WASHER_DONE, "role": "trigger", "applies_to": ["G1"]}}), agent)
        assert entry["source"] == "dependency" and entry["execution_mode"] == "dependency"
        assert entry["tree"]["children"][0]["type"] == "wait_for"
        assert entry["runtime"]["reads"][0]["key"] == "/artifacts/washer_1/operationalState"
        assert seen == {} and "llm" not in entry

    @pytest.mark.asyncio
    async def test_questions_go_to_the_small_model_with_each_ids_devices(self, graph):
        """A dependency_anchor question reaches the small model with the
        devices of its goals; an answer naming a device outside them falls
        back; the call is kept on the entry."""
        # 1. The model answers Q1 with a condition on washer_1. The fake
        #    planner's trees act on light_1 only, so washer_1 is in no goal's
        #    context and the answer must fall back.
        cond = {"ref": {"artifact_name": "washer_1", "property_name": "operationalState",
                        "property_field": "NA"}, "comparison": "==", "target_value_text": "0"}
        agent, seen = _agent({"answers": [{"id": "Q1", "question": "dependency_anchor",
                                           "target": "G2",
                                           "answer": {"anchor": "condition", "condition": cond}}],
                              "unresolved": []})
        entry = await _plan(graph, _structure(
            {"G1": _goal(intent_text="start washer 1", timing={"starts": "now"},
                         **{**NAMED, "artifact_name": "washer_1"}),
             "G2": _goal(intent_text="then light 1", timing={"starts": "after_goal",
                                                             "ref_goal": "G1"}, **NAMED)}), agent)
        # 2. The prompt carried the question and G2's device.
        prompt = seen["messages"][0]["content"]
        assert '"dependency_anchor"' in prompt and '"light_1"' in prompt
        assert entry["profile"]["Q1"]["source"] == "fallback"     # washer_1 not in G1's devices
        assert entry["llm"] is True
        # The call is kept on the entry: the questions as sent, the raw reply.
        call = entry["profile_call"]
        assert [q["id"] for q in call["questions"]] == ["Q1"]
        assert '"anchor": "condition"' in call["answer"]

    @pytest.mark.asyncio
    async def test_a_failed_profiling_call_falls_back(self, graph):
        """A failed call does not stop the plan: every question on its
        fallback, noted in the explanation."""
        agent, _ = _agent(error=RuntimeError("rate limited"))
        entry = await _plan(graph, _structure(
            {"G1": _goal(goal_kind="maintenance", timing={"starts": "now"}, **NAMED)}), agent)
        assert entry["profile"]["Q1"]["answer"] == {"mode": "unobserved"}
        assert "fallback applied" in entry["explanation"]
        # Asked, but nothing came back: the call is kept, its answer is None.
        assert entry["profile_call"]["answer"] is None

    @pytest.mark.asyncio
    async def test_a_step_that_cannot_be_planned_stops_the_structure(self, graph):
        agent, seen = _agent()
        entry = await _plan(graph, _structure(
            {"G1": _goal(intent_text="x", timing={"starts": "now"}, **NAMED)}), agent,
            planner_result={"tree": None, "impossible": True, "explanation": "no such device"})
        assert entry["impossible"] and entry["tree"] is None
        assert entry["explanation"] == "G1: no such device"

    @pytest.mark.asyncio
    async def test_an_inconsistent_structure_is_impossible(self, graph):
        agent, seen = _agent()
        entry = await _plan(graph, _structure(
            {"G1": _goal(goal_kind="maintenance", timing={"starts": "now"}, **NAMED),
             "G2": _goal(timing={"starts": "after_goal", "ref_goal": "G1"}, **NAMED)}), agent)
        assert entry["error"] == "inconsistent_structure" and seen == {}

    @pytest.mark.asyncio
    async def test_an_unreadable_predicate_is_an_error(self, graph):
        agent, _ = _agent()
        entry = await _plan(graph, _structure(
            {"G1": _goal(timing={"starts": "on_trigger"}, **NAMED)},
            {"P1": {**WASHER_DONE, "target_value_determined": None, "role": "trigger",
                    "applies_to": ["G1"]}}), agent)
        assert entry["error"] == "unresolvable_predicate"


class TestRouting:
    def test_dependency_structures_and_simple_maintenance(self):
        dep = _structure({"G1": _goal()})
        assert dependency_structure(dep) is dep
        simple = GoalStructure.from_dict(_goal(goal_kind="maintenance"), intent_text="x",
                                         structure="simple")
        assert dependency_structure(simple) is simple
        plain = GoalStructure.from_dict(_goal(), intent_text="x", structure="simple")
        assert dependency_structure(plain) is None


class TestUAExecution:
    @pytest.mark.asyncio
    async def test_the_runtime_is_refreshed_before_every_tick_and_there_is_no_budget(self):
        from ami_agents.agents.user_assistant.behaviours.plan_execution import (
            DependencyPlanBehaviour,
        )
        order = []
        behaviour = DependencyPlanBehaviour("plan-1", {"type": "running", "name": "r"},
                                            max_ticks=2, tick_period=0.0)
        behaviour.agent = MagicMock()
        behaviour.agent.plans.get.return_value = object()
        behaviour.receive = AsyncMock(return_value=None)
        behaviour.kill = MagicMock()
        behaviour.runtime = MagicMock(before_tick=lambda: order.append("refresh"))
        behaviour.tree = MagicMock(status=Status.RUNNING,
                                   tick_once=lambda: order.append("tick"))
        for _ in range(5):
            await behaviour.run()
        assert order == ["refresh", "tick"] * 5
        behaviour.kill.assert_not_called()          # 5 ticks > max_ticks=2: still running

    def test_plan_management_picks_the_dependency_lifecycle(self):
        from ami_agents.agents.user_assistant.behaviours.plan_management import (
            PlanManagementBehaviour,
        )
        from ami_agents.shared.models.plan import PlanType
        assert PlanManagementBehaviour._classify_type(
            {}, None, {"execution_mode": "dependency"}) is PlanType.DEPENDENCY
        assert PlanManagementBehaviour._classify_type({}, None, {}) is PlanType.IMMEDIATE


class TestSHTDClock:
    def test_the_simulators_time_is_proxied(self):
        from fastapi.testclient import TestClient
        from ami_agents.environment.integration.SimuHome import shtd
        fake = MagicMock(current_time=MagicMock(return_value={"now": "2026-10-14 13:00:00"}))
        with patch.object(shtd, "get_client", return_value=fake):
            response = TestClient(shtd.app).get("/_shtd/time")
        assert response.json() == {"now": "2026-10-14 13:00:00"}

    def test_an_unreachable_simulator_is_a_502(self):
        from fastapi.testclient import TestClient
        from ami_agents.environment.integration.SimuHome import shtd
        fake = MagicMock(current_time=MagicMock(side_effect=shtd.SimuHomeError("down")))
        with patch.object(shtd, "get_client", return_value=fake):
            assert TestClient(shtd.app).get("/_shtd/time").status_code == 502


class TestStatedTimes:
    """A start stated both as a clock time and relative to now must agree."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stated, conflict", [("11:35 AM", True), ("11:50 AM", False)])
    async def test_a_disagreement_is_a_question_not_a_plan(self, graph, stated, conflict):
        from datetime import datetime
        agent, seen = _agent()
        clock = MagicMock(now=MagicMock(return_value=datetime(2025, 8, 23, 11, 36, 54)))
        with patch(f"{MODULE}.make_clock", return_value=clock):
            entry = await _plan(graph, _structure({"G1": _goal(
                intent_text="turn on dimmer light 1",
                timing={"starts": "after_delay", "offset_text": "13 minutes",
                        "stated_start_time_text": stated}, **NAMED)}), agent)
        if conflict:
            assert entry["tree"] is None and entry["requires_clarification"]
            assert "11:35 AM" in entry["explanation"] and "11:49 AM" in entry["explanation"]
        else:
            assert entry["tree"] and "agrees" in entry["explanation"]
        assert seen == {}                                  # no model decided it


class TestFinishTogether:
    """A goal that completes with an event, and words with no time in them:
    questions for the user, never a guessed plan."""

    @pytest.mark.asyncio
    async def test_an_unknown_run_length_is_a_question_not_a_plan(self, graph):
        """Seed 45's shape: nothing reports the run length or the event's time
        left (the model names no property), so no start is guessed."""
        from datetime import datetime
        agent, _ = _agent(model_answer={"answers": [], "unresolved": []})
        clock = MagicMock(now=MagicMock(return_value=datetime(2025, 8, 23, 11, 0, 0)))
        with patch(f"{MODULE}.make_clock", return_value=clock):
            entry = await _plan(graph, _structure(
                {"G1": _goal(intent_text="start dimmer light 1", timing={"completes_with": "P1"},
                             **NAMED)},
                {"P1": {**WASHER_DONE, "role": "trigger", "applies_to": ["G1"]}}), agent)
        assert entry["tree"] is None and entry["requires_clarification"]
        assert entry["error"] == "completion_unverifiable"
        assert "start dimmer light 1" in entry["explanation"]

    @pytest.mark.asyncio
    async def test_a_vague_time_is_a_question_not_an_error(self, graph):
        """Seed 45 as it was structured: after_delay "soon"."""
        agent, _ = _agent()
        entry = await _plan(graph, _structure({"G1": _goal(
            intent_text="start dimmer light 1",
            timing={"starts": "after_delay", "offset_text": "soon"}, **NAMED)}), agent)
        assert entry["tree"] is None and entry["requires_clarification"]
        assert entry["error"] == "vague_time" and '"soon"' in entry["explanation"]
