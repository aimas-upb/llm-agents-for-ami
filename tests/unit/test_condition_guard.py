"""Condition guards: resolved, restored, wrapped, dropped, remembered, decided.

On the TD home of test_planning_conditions.py (washer_1 with combined
commands, ha_washer with separate ones, dryer_2 that cannot resume, ac_1):

    resolve_guard       the model's answer found on the step's own device;
                        anything wrong -> no guard, with why
    restore_action      pause -> resume, off -> on, by combined value or by
                        separate command; a setting needs none; a pause with
                        no resume leaves the step unguarded
    wrap_steps          the guard's tree, valid IR, at the step's path
    drop_satisfied      a guard the plan already satisfies, earlier in
                        execution order, is removed again
    apply_remembered    a remembered record resolved again on today's TDs
    decide              memory first, then one model call for the misses
    behaviours          the simple-plan behaviour; the dependency step 3c
    recording           which decisions the UA sends; the EnvExplorer's store
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ami_agents.agents.interaction_solver.behaviours.condition_guard import (
    ConditionGuardBehaviour,
)
from ami_agents.agents.interaction_solver.utils.condition_check import (
    REPLY_SCHEMA,
    RESPONSE_FORMAT,
    decide,
    device_context,
    plan_actions,
    render_condition_guard_prompt,
)
from ami_agents.agents.interaction_solver.utils.condition_guard import (
    CHECK_FAILED,
    ESTABLISHED_BY_PLAN,
    FROM_MEMORY,
    FROM_MODEL,
    Decision,
    apply_remembered,
    drop_satisfied,
    guard_names,
    guards_wire,
    resolve_guard,
    restore_action,
    wrap_steps,
)
from ami_agents.agents.interaction_solver.utils.planning_conditions import conditioned_steps
from ami_agents.agents.user_assistant.guards import GuardRecordingBehaviour, guards_to_record
from ami_agents.bt_planning.serialization import validate_tree
from tests.unit.test_planning_conditions import (  # noqa: F401 - fixtures
    _action_node,
    _artifact,
    act,
    conditions,
    graph,
)

# The model's answer for a washer_1 spin step: proceed unless running, else pause.
PAUSE_FIRST = {"guard": True,
               "check": {"property": "operationalState",
                         "property_class": "homeont:LaundryWasherOperationalState",
                         "operator": "!=", "value": 1},
               "action": {"action": "operationalState",
                          "action_class": "homeont:SetOperationalStateCommand", "value": 2},
               "reason": "spin is refused while running"}


def _step(graph, conditions, node):
    (step,) = conditioned_steps(graph, node, conditions)
    return step


def _guarded(graph, conditions, node, answer=PAUSE_FIRST):
    """(the step, its model decision) for ``node`` and ``answer``."""
    step = _step(graph, conditions, node)
    guard, why = resolve_guard(graph, step, answer)
    assert guard is not None, why
    return step, Decision(step=step, source=FROM_MODEL, guard=guard)


def _seq(*children):
    return {"type": "sequence", "name": "plan", "children": list(children)}


# --------------------------------------------------------------------------
# Resolution and restore
# --------------------------------------------------------------------------

class TestResolve:
    def test_a_good_answer_resolves_on_the_device(self, graph, conditions):
        step = _step(graph, conditions, act("washer_1", "spinSpeedCurrent", 1200))
        guard, why = resolve_guard(graph, step, PAUSE_FIRST)
        assert why is None
        assert (guard.check.url, guard.check.operator, guard.check.value) == \
            ("http://h/washer_1/p/operationalState", "!=", 1)
        assert (guard.action.url, guard.action.value) == ("http://h/washer_1/a/operationalState", 2)

    def test_a_named_state_is_typed_by_matter(self, graph, conditions):
        """"Paused" on an integer schema: Matter's OperationalStateEnum."""
        step = _step(graph, conditions, act("washer_1", "spinSpeedCurrent", 1200))
        answer = {**PAUSE_FIRST, "action": {**PAUSE_FIRST["action"], "value": "Paused"}}
        guard, _ = resolve_guard(graph, step, answer)
        assert guard.action.value == 2

    @pytest.mark.parametrize("change, expected", [
        ({"check": {**PAUSE_FIRST["check"], "property": "nope"}}, "no affordance titled"),
        ({"check": {**PAUSE_FIRST["check"], "property_class": "homeont:OnOff"}}, "is not a"),
        ({"check": {**PAUSE_FIRST["check"], "operator": "~"}}, "not a comparison"),
        ({"action": {"action": "spinSpeedCurrent", "value": 1200}}, "the step itself"),
        ({"action": None}, "no check or no action"),
    ])
    def test_anything_wrong_is_no_guard_with_why(self, graph, conditions, change, expected):
        step = _step(graph, conditions, act("washer_1", "spinSpeedCurrent", 1200))
        guard, why = resolve_guard(graph, step, {**PAUSE_FIRST, **change})
        assert guard is None and expected in why


class TestRestore:
    def test_a_combined_pause_is_resumed_by_the_same_action(self, graph):
        restore, required = restore_action(graph, _artifact("washer_1"),
                                           _action_node("washer_1", "operationalState"), 2)
        assert required and (restore.url, restore.value) == \
            ("http://h/washer_1/a/operationalState", 1)

    def test_a_combined_switch_off_is_switched_back_on(self, graph):
        restore, required = restore_action(graph, _artifact("washer_1"),
                                           _action_node("washer_1", "onOff"), False)
        assert required and restore.value is True

    def test_a_setting_needs_no_restore(self, graph):
        assert restore_action(graph, _artifact("ac_1"), _action_node("ac_1", "hvacMode"), 3) \
            == (None, False)

    def test_a_separate_pause_is_resumed_by_name(self, graph):
        """Two separate StartCommands: the tie is broken by the name "resume"."""
        restore, required = restore_action(graph, _artifact("ha_washer"),
                                           _action_node("ha_washer", "pause"), None)
        assert required and restore.url == "http://h/ha_washer/a/resume"
        assert restore.parameters == {}

    def test_a_pause_with_no_resume_leaves_the_step_unguarded(self, graph, conditions):
        """dryer_2 can pause but not resume: a guard would end its cycle."""
        assert restore_action(graph, _artifact("dryer_2"),
                              _action_node("dryer_2", "pause"), None) == (None, True)
        step = _step(graph, conditions, act("dryer_2", "selectedDrynessLevel", 1))
        guard, why = resolve_guard(graph, step, {
            "check": {"property": "operationalState", "operator": "!=", "value": 1},
            "action": {"action": "pause", "value": None}})
        assert guard is None and "no matching resume" in why


# --------------------------------------------------------------------------
# The tree
# --------------------------------------------------------------------------

class TestWrap:
    def test_the_guard_tree(self, graph, conditions):
        """holds: check, step. establish: not check, pause, step, resume."""
        node = act("washer_1", "spinSpeedCurrent", 1200)
        step, decision = _guarded(graph, conditions, node)
        tree = wrap_steps(node, [step], {step.id: decision})
        assert validate_tree(tree) == [] and guard_names(tree) == [decision.name]
        holds, establish = tree["children"]
        assert [c.get("operator") for c in holds["children"][:1]] == ["!="]
        assert holds["children"][1] == node
        check, pause, planned, resume = establish["children"]
        assert check["operator"] == "==" and check["expected_value"] == 1
        assert pause["parameters"] == {"value": 2} and planned == node
        assert resume["parameters"] == {"value": 1}

    def test_a_nested_step_is_wrapped_where_it_stands(self, graph, conditions):
        tree = _seq(act("light_1", "brightness", 40), act("washer_1", "spinSpeedCurrent", 1200))
        step = _step(graph, conditions, tree)
        guard, _ = resolve_guard(graph, step, PAUSE_FIRST)
        out = wrap_steps(tree, [step], {step.id: Decision(step=step, source=FROM_MODEL,
                                                          guard=guard)})
        assert out["children"][0] == tree["children"][0]
        assert out["children"][1]["type"] == "selector"
        assert tree["children"][1]["type"] == "action"            # the input is kept


class TestDrop:
    def _plan(self, graph, conditions, *before):
        """``before`` then the guarded spin step, wrapped."""
        tree = _seq(*before, act("washer_1", "spinSpeedCurrent", 1200))
        steps = conditioned_steps(graph, tree, conditions)
        spin = steps[-1]
        guard, _ = resolve_guard(graph, spin, PAUSE_FIRST)
        return wrap_steps(tree, [spin], {spin.id: Decision(step=spin, source=FROM_MODEL,
                                                           guard=guard)})

    def test_a_pause_earlier_in_the_plan_satisfies_the_guard(self, graph, conditions):
        tree, notes = drop_satisfied(graph, self._plan(
            graph, conditions, act("washer_1", "operationalState", 2)))
        assert guard_names(tree) == [] and tree["children"][1]["type"] == "action"
        assert notes and "not guarded" in notes[0]

    def test_nothing_earlier_keeps_the_guard(self, graph, conditions):
        tree, notes = drop_satisfied(graph, self._plan(graph, conditions))
        assert len(guard_names(tree)) == 1 and notes == []

    def test_a_resume_after_the_pause_keeps_it(self, graph, conditions):
        tree, _ = drop_satisfied(graph, self._plan(
            graph, conditions, act("washer_1", "operationalState", 2),
            act("washer_1", "operationalState", 1)))
        assert len(guard_names(tree)) == 1

    def test_a_pause_in_a_selector_branch_does_not_count(self, graph, conditions):
        maybe = {"type": "selector", "name": "maybe", "children": [
            {"type": "condition", "name": "c", "property_url": "http://h/x",
             "expected_value": True}, act("washer_1", "operationalState", 2)]}
        tree, _ = drop_satisfied(graph, self._plan(graph, conditions, maybe))
        assert len(guard_names(tree)) == 1

    def test_a_pause_before_a_timer_still_counts(self, graph, conditions):
        """The plan is trusted while it waits."""
        tree, _ = drop_satisfied(graph, self._plan(
            graph, conditions, act("washer_1", "operationalState", 2),
            {"type": "timer", "name": "t", "seconds": 600}))
        assert guard_names(tree) == []


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------

RECORD = {"guard_id": "grd-1", "protection": True,
          "check": {"title": "operationalState", "class": "homeont:LaundryWasherOperationalState",
                    "target": "http://stale/url", "operator": "!=", "value": 1},
          "action": {"title": "operationalState", "class": "homeont:SetOperationalStateCommand",
                     "target": "http://stale/url", "value": 2}}


class TestRemembered:
    def test_a_record_is_resolved_again_on_todays_td(self, graph, conditions):
        """URLs come from the graph, not the record; the restore is derived."""
        step = _step(graph, conditions, act("washer_1", "spinSpeedCurrent", 1200))
        decision = apply_remembered(graph, step, RECORD)
        assert decision.source == FROM_MEMORY
        assert decision.guard.check.url == "http://h/washer_1/p/operationalState"
        assert decision.guard.restore.value == 1

    def test_a_record_that_no_longer_resolves_is_a_miss(self, graph, conditions):
        step = _step(graph, conditions, act("washer_1", "spinSpeedCurrent", 1200))
        stale = {**RECORD, "check": {**RECORD["check"], "title": "gone"}}
        assert apply_remembered(graph, step, stale) is None

    def test_a_remembered_no_guard_is_no_guard(self, graph, conditions):
        step = _step(graph, conditions, act("washer_1", "spinSpeedCurrent", 1200))
        decision = apply_remembered(graph, step, {"protection": False, "why": "not_applicable"})
        assert decision.guard is None and decision.why == "not_applicable"


# --------------------------------------------------------------------------
# The check: memory, then the model
# --------------------------------------------------------------------------

def _model(answers):
    """A profiling-model config whose one call returns ``answers`` (a dict as
    JSON, or raw text)."""
    content = answers if isinstance(answers, str) else json.dumps(answers)
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason="stop")])
    return SimpleNamespace(model="gpt-4o-mini", temperature=0.0, reasoning_effort=None,
                           max_completion_tokens=None,
                           client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                               create=AsyncMock(return_value=response)))))


def _agent(answers=None, record=None):
    """An ISA with the model above and a guard memory answering ``record``."""
    return SimpleNamespace(
        config={"planning": {}}, explicit_llm=_model(answers or {"answers": []}),
        _query_env_explorer=AsyncMock(return_value=json.dumps({"ok": True, "match": record})),
        add_behaviour=lambda behaviour: None)


LOG = MagicMock()


class TestDecide:
    @pytest.mark.asyncio
    async def test_a_memory_hit_asks_no_model(self, graph, conditions):
        node = act("washer_1", "spinSpeedCurrent", 1200)
        steps = conditioned_steps(graph, node, conditions)
        agent = _agent(record=RECORD)
        decisions, _ = await decide(agent, LOG, graph, steps, {}, [])
        assert decisions["S1"].source == FROM_MEMORY and decisions["S1"].guard
        agent.explicit_llm.client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_model_decides_the_misses(self, graph, conditions):
        tree = _seq(act("washer_1", "operationalState", 2),
                    act("washer_1", "spinSpeedCurrent", 1200))
        steps = conditioned_steps(graph, tree, conditions)
        # The pause is no step (cycle commands are checked in code); the spin is S1.
        agent = _agent({"answers": [{"id": "S1", "guard": False, "why": ESTABLISHED_BY_PLAN}]})
        trace = {}
        decisions, notes = await decide(agent, LOG, graph, steps, {}, [], trace)
        assert list(decisions) == ["S1"]
        assert decisions["S1"].why == ESTABLISHED_BY_PLAN and decisions["S1"].guard is None
        assert trace["answer"] and trace["model"] == "gpt-4o-mini" and notes == []

    @pytest.mark.asyncio
    async def test_an_unusable_reply_leaves_every_step_unguarded(self, graph, conditions):
        """A reply with no JSON (a refusal, a truncated one): no guard, and the
        note says the check failed and why -- not that nothing came back."""
        steps = conditioned_steps(graph, act("washer_1", "spinSpeedCurrent", 1200), conditions)
        decisions, notes = await decide(_agent("not json"), LOG, graph, steps, {}, [])
        assert decisions["S1"].guard is None and decisions["S1"].why == CHECK_FAILED
        assert notes[0].startswith("S1: condition check failed: ")

    @pytest.mark.asyncio
    async def test_the_reply_shape_is_enforced_by_the_api(self, graph, conditions):
        """Strict structured output: every answer field required, nothing else."""
        steps = conditioned_steps(graph, act("washer_1", "spinSpeedCurrent", 1200), conditions)
        agent = _agent({"answers": []})
        await decide(agent, LOG, graph, steps, {}, [])
        sent = agent.explicit_llm.client.chat.completions.create.await_args.kwargs
        assert sent["response_format"] == RESPONSE_FORMAT
        item = REPLY_SCHEMA["properties"]["answers"]["items"]
        assert RESPONSE_FORMAT["json_schema"]["strict"] is True
        assert sorted(item["required"]) == sorted(item["properties"])

    def test_the_prompt_carries_the_steps_and_their_conditions(self, graph, conditions):
        """Plan actions are named by their TD titles, not by the tree node's
        name (a planner's "Currentmode")."""
        tree = {**act("washer_1", "spinSpeedCurrent", 1200), "name": "Spinspeedcurrent"}
        steps = conditioned_steps(graph, tree, conditions)
        actions = plan_actions(graph, [(None, tree)], steps)
        assert actions == [{"step": "S1", "device": "washer_1",
                            "action": "spinSpeedCurrent", "value": 1200}]
        prompt = render_condition_guard_prompt("{}", json.dumps(actions), "[S1 steps]", "{}")
        assert "[S1 steps]" in prompt and "{steps}" not in prompt

    def test_the_devices_are_described_without_urls(self, graph, conditions):
        """Kind, name, classes, description and schema -- no target, form or
        artifact id: the model names affordances by title and class."""
        steps = conditioned_steps(graph, act("washer_1", "spinSpeedCurrent", 1200), conditions)
        devices = device_context(graph, steps)
        assert list(devices) == ["washer_1"]
        text = json.dumps(devices)
        assert "http://" not in text and "form" not in text and "artifact_id" not in text
        kinds = {(a["kind"], a["name"]) for a in devices["washer_1"]}
        assert ("action", "operationalState") in kinds and ("property", "operationalState") in kinds


# --------------------------------------------------------------------------
# Where it runs
# --------------------------------------------------------------------------

class TestSimplePlans:
    @pytest.mark.asyncio
    async def test_a_step_gets_its_guard_and_the_entry_its_decisions(self, graph):
        entry = {"tree": act("washer_1", "spinSpeedCurrent", 1200), "explanation": "x"}
        behaviour = ConditionGuardBehaviour([entry], logger=LOG, graph=graph)
        behaviour.agent = _agent({"answers": [{"id": "S1", **PAUSE_FIRST}]})
        await behaviour.run()
        assert len(guard_names(entry["tree"])) == 1 and validate_tree(entry["tree"]) == []
        (guard,) = entry["guards"]
        assert guard["protection"] and guard["source"] == FROM_MODEL and "dropped" not in guard
        assert entry["condition_call"]["answer"]

    @pytest.mark.asyncio
    async def test_a_guard_the_plan_satisfies_is_dropped_and_marked(self, graph):
        """The model guards the spin anyway; the walk sees the pause before it."""
        entry = {"tree": _seq(act("washer_1", "operationalState", 2),
                              act("washer_1", "spinSpeedCurrent", 1200))}
        behaviour = ConditionGuardBehaviour([entry], logger=LOG, graph=graph)
        behaviour.agent = _agent({"answers": [{"id": "S1", **PAUSE_FIRST}]})
        await behaviour.run()
        assert guard_names(entry["tree"]) == []
        assert [g.get("dropped") for g in entry["guards"]] == [True]
        assert "not guarded" in entry["explanation"]

    @pytest.mark.asyncio
    async def test_a_plan_without_conditioned_steps_is_left_alone(self, graph):
        entry = {"tree": act("light_1", "brightness", 40)}
        behaviour = ConditionGuardBehaviour([entry], logger=LOG, graph=graph)
        behaviour.agent = _agent()
        await behaviour.run()
        assert entry == {"tree": act("light_1", "brightness", 40)}


DEPENDENCY = "ami_agents.agents.interaction_solver.behaviours.dependency_goal_planning"
WASHER = dict(goal_specificity="explicit", goal_kind="achievement", goal_effect="set",
              artifact_name="washer_1", artifact_class="homeont:LaundryWasher")


class _WasherPlanner:
    """Stands in for the achievement planners: each goal's action on washer_1."""

    def __init__(self, goals, logger=None, graph=None):
        self.results = [{"tree": act("washer_1", g.affordance_name,
                                     int(g.target_value_determined)),
                         "explanation": "x"} for g in goals]

    async def join(self):
        return None


async def _dependency_entry(graph, goals, agent):
    """One dependency structure through the planner; its entry."""
    from ami_agents.agents.interaction_solver.behaviours.dependency_goal_planning import (
        DependencyGoalPlanningBehaviour,
    )
    from ami_agents.shared.models.goal_structure import GoalStructure
    structure = GoalStructure.from_dict({"goals": goals, "predicates": {}},
                                        intent_text="the request", structure="dependency")
    behaviour = DependencyGoalPlanningBehaviour([structure], logger=LOG)
    behaviour.agent = agent
    with patch(f"{DEPENDENCY}.fetch_environment_graph", AsyncMock(return_value=graph)), \
         patch(f"{DEPENDENCY}.ExplicitGoalPlanningBehaviour", _WasherPlanner):
        await behaviour.run()
    (entry,) = behaviour.results
    return entry


SPIN = {**WASHER, "intent_text": "set the spin to 1200", "affordance_name": "spinSpeedCurrent",
        "affordance_class": "saref:SetAbsoluteLevelCommand", "target_value_determined": "1200"}
PAUSE = {**WASHER, "intent_text": "pause the washer", "affordance_name": "operationalState",
         "affordance_class": "homeont:SetOperationalStateCommand",
         "target_value_determined": "2"}


class TestDependencyPlans:
    @pytest.mark.asyncio
    async def test_a_goals_step_is_guarded_in_the_compiled_tree(self, graph):
        agent = _agent({"answers": [{"id": "S1", **PAUSE_FIRST}]})
        entry = await _dependency_entry(graph, {"G1": {**SPIN, "timing": {"starts": "now"}}},
                                        agent)
        assert len(guard_names(entry["tree"])) == 1
        assert entry["guards"][0]["goal"] == "G1" and entry["condition_call"]

    @pytest.mark.asyncio
    async def test_a_pause_in_an_earlier_goal_drops_the_guard(self, graph):
        """G1 pauses, G2 sets the spin after G1: real order is the compiled tree's."""
        agent = _agent({"answers": [{"id": "S1", **PAUSE_FIRST}]})
        entry = await _dependency_entry(graph, {
            "G1": {**PAUSE, "timing": {"starts": "now"}},
            "G2": {**SPIN, "timing": {"starts": "after_goal", "ref_goal": "G1"}}}, agent)
        assert guard_names(entry["tree"]) == []
        assert entry["guards"][0]["dropped"] is True


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

class TestRecording:
    def test_only_the_models_device_level_decisions_are_recorded(self):
        guard = {"key": {"k": 1}, "protection": True, "source": "model",
                 "check": {}, "action": {}, "restore": {}, "name": "n"}
        entry = {"guards": [
            guard,
            {**guard, "dropped": True},                                    # plan didn't need it
            {**guard, "source": "memory"},                                 # stored already
            {"key": {}, "protection": False, "source": "model", "why": "not_applicable"},
            {"key": {}, "protection": False, "source": "model", "why": ESTABLISHED_BY_PLAN},
        ]}
        recorded = guards_to_record(entry)
        assert recorded == [{"key": {"k": 1}, "protection": True, "check": {}, "action": {}},
                            {"key": {}, "protection": False, "why": "not_applicable"}]

    @pytest.mark.asyncio
    async def test_the_ua_sends_one_request_per_decision_after_success(self):
        context = SimpleNamespace(entry={"guards": [
            {"key": {}, "protection": False, "source": "model", "why": "not_applicable"}]},
            thread="__default__", plan_id="p1")
        agent = SimpleNamespace(target_jids={"explorer": "ee@x"}, rpc_call_timeout=1)
        rpc = AsyncMock(return_value=SimpleNamespace(body=json.dumps({"guard_id": "grd-9"})))
        for success, expected in ((False, []), (True, ["grd-9"])):
            behaviour = GuardRecordingBehaviour(context, SimpleNamespace(success=success),
                                                logger=LOG)
            behaviour.agent = agent
            with patch("ami_agents.agents.user_assistant.guards.rpc_call", rpc):
                await behaviour.run()
            assert behaviour.recorded == expected

    @pytest.mark.asyncio
    async def test_the_explorer_stores_and_finds_a_decision(self, tmp_path):
        from ami_agents.agents.env_explorer.behaviors.guard_match_behaviour import (
            GuardMatchBehaviour,
        )
        from ami_agents.agents.env_explorer.behaviors.guard_record_behaviour import (
            GuardRecordBehaviour,
        )
        from ami_agents.shared.memory import ensure_engine_on_path
        ensure_engine_on_path()
        from src.storage.guard_store import GuardStore
        agent = SimpleNamespace(_experience_engine_guard_store=GuardStore(tmp_path),
                                logger=LOG)
        key = {"affordance_name": "spinSpeedCurrent", "device_name": "washer_1"}
        ready = "ami_agents.agents.env_explorer.behaviors.guard_{}_behaviour." \
                "ensure_experience_engine_ready"
        with patch(ready.format("record"), AsyncMock(return_value=True)), \
             patch(ready.format("match"), AsyncMock(return_value=True)):
            record, match = GuardRecordBehaviour(), GuardMatchBehaviour()
            record.agent = match.agent = agent
            stored = await record.record({"guard": {"key": key, "protection": False,
                                                    "why": "not_applicable"}})
            found = await match.match({"key": key})
            assert stored["ok"] and found["match"]["guard_id"] == stored["guard_id"]
            assert (await match.match({"key": {**key, "device_name": "x"}}))["match"] is None
            assert (await record.record({}))["error"] == "malformed_payload"
