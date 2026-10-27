"""The dependency templates: structure + profile -> one tree and its runtime.

A hand-written home, closed under subclass inference as the ISA closes its graph:

    Living Room   light_1, light_2  OnOffLight  onOff action + onOff property (boolean)
                  sensor            reading of the room's air temperature
                  heater            setpoint action (SetAbsoluteLevelCommand)
    Utility Room  washer_1          operationalState (integer), countdownTime
                                    (integer, unit:SEC), endTime (string)

The goals' own trees (achieve(G)) are stand-in actions: what is tested here is
how the templates place them.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import py_trees
import pytest
from py_trees.common import Status
from rdflib import Graph

from ami_agents.agents.interaction_solver.utils.dependency_compiler import (
    CompileError,
    VagueTime,
    compile_dependency,
)
from ami_agents.agents.interaction_solver.utils.dependency_context import resolve_predicate
from ami_agents.bt_planning.nodes import registry
from ami_agents.bt_planning.nodes.dependency_nodes import CLOCK_KEY
from ami_agents.bt_planning.nodes.http_client import HTTPResponse
from ami_agents.bt_planning.runtime import PlanRuntime
from ami_agents.bt_planning.serialization import validate_tree
from ami_agents.shared.models.goal_structure import GoalStructure
from ami_agents.shared.utils.vocabulary import close_types
from ami_agents.shared.utils.namespaces import (  # noqa: E402
    HCTL,
    HMAS,
    HOMEONT,
    JS,
    QUDT,
    SAREF,
    SSN,
    TD,
    UNIT,
)

PREFIXES = f"""
@prefix hmas: <{HMAS}> .
@prefix homeont: <{HOMEONT}> .
@prefix saref: <{SAREF}> .
@prefix td: <{TD}> .
@prefix hctl: <{HCTL}> .
@prefix js: <{JS}> .
@prefix ssn: <{SSN}> .
@prefix qudt: <{QUDT}> .
@prefix unit: <{UNIT}> .
"""


def _room(key, title, cls, devices, variables=()):
    contains = ", ".join(f"<http://h/{d}#artifact>" for d in devices)
    out = (f"<http://h/{key}#ws> a hmas:Workspace ; td:title \"{title}\" ; "
           f"hmas:contains {contains} ; homeont:hasSpace <http://h/{key}#space> .\n"
           f"<http://h/{key}#space> a homeont:{cls} .\n")
    if variables:
        out += (f"<http://h/{key}#space> homeont:hasEnvironment <http://h/{key}#env> .\n"
                f"<http://h/{key}#env> ssn:hasProperty "
                + ", ".join(f"<http://h/{key}#{v}>" for v, _ in variables) + " .\n"
                + "".join(f"<http://h/{key}#{v}> a homeont:{c} ; td:title \"{v}\" .\n"
                          for v, c in variables))
    return out


def _prop(device, name, cls, schema, extra=""):
    return (f"<http://h/{device}#artifact> td:hasPropertyAffordance <http://h/{device}/p/{name}> .\n"
            f"<http://h/{device}/p/{name}> a td:PropertyAffordance, homeont:{cls} ; td:title \"{name}\" ;"
            f" td:hasForm [ hctl:hasTarget <http://h/{device}/p/{name}> ] ;"
            f" td:hasOutputSchema [ a {schema} ] {extra}.\n")


def _device(name, cls):
    return (f"<http://h/{name}#artifact> a hmas:Artifact, homeont:{cls} ; td:title \"{name}\" ;"
            f" td:hasActionAffordance [ a td:ActionAffordance, homeont:SetOnOffCommand ;"
            f" td:title \"onOff\" ; td:hasForm [ hctl:hasTarget <http://h/{name}/a/onOff> ] ;"
            f" td:hasInputSchema [ a js:BooleanSchema ] ] .\n")


HOME = PREFIXES + "".join([
    _room("living", "Living Room", "LivingRoom", ["light_1", "light_2", "sensor", "heater"],
          [("air_temperature", "AirTemperature")]),
    _room("utility", "Utility Room", "UtilityRoom", ["washer_1"]),
    _device("light_1", "OnOffLight"), _prop("light_1", "onOff", "OnOffLightOnOff", "js:BooleanSchema"),
    _device("light_2", "OnOffLight"), _prop("light_2", "onOff", "OnOffLightOnOff", "js:BooleanSchema"),
    _device("sensor", "TemperatureSensor"),
    _prop("sensor", "temperature", "AirTemperature", "js:NumberSchema"),
    _device("heater", "Heater"),
    _device("washer_1", "LaundryWasher"),
    _prop("washer_1", "operationalState", "OperationalStateOperationalState", "js:IntegerSchema"),
    _prop("washer_1", "countdownTime", "RemainingTime", "js:IntegerSchema", "; qudt:unit unit:SEC "),
    _prop("washer_1", "endTime", "RemainingTime", "js:StringSchema"),
])


@pytest.fixture(scope="module")
def graph():
    g = Graph().parse(data=HOME, format="turtle")
    close_types(g)
    return g


@pytest.fixture(autouse=True)
def clean_blackboard():
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


def _act(gid, device="light_1", value=True):
    return {"type": "action", "name": f"achieve {gid}", "action_url": f"http://h/{device}/a/onOff",
            "parameters": {"value": value}}


LIGHT_ON = {"predicate_specificity": "incomplete", "predicate_subject": "device_property",
            "predicate_text": "a living room light is on", "location_class": "homeont:LivingRoom",
            "artifact_class": "homeont:OnOffLight", "property_name": "onOff",
            "comparison": "==", "target_value_text": "on", "target_value_determined": "true",
            "quantifier": "any"}
WASHER_DONE = {"predicate_specificity": "explicit", "predicate_subject": "device_property",
               "predicate_text": "washer 1 finishes", "artifact_name": "washer_1",
               "property_name": "operationalState", "comparison": "==",
               "target_value_determined": "0"}


def _goal(**fields):
    return {"goal_specificity": "explicit", "goal_kind": "achievement", **fields}


def _plan(graph, goals, predicates=None, profile=None, subtrees=None, **kwargs):
    structure = GoalStructure.from_dict({"goals": goals, "predicates": predicates or {}},
                                        intent_text="x", structure="dependency")
    targets = {pid: resolve_predicate(graph, p) for pid, p in structure.predicates.items()}
    subtrees = subtrees or {gid: _act(gid) for gid in structure.goals}
    compiled = compile_dependency(graph, structure, subtrees, targets, profile or {}, **kwargs)
    assert validate_tree(compiled.tree) == [], validate_tree(compiled.tree)
    return compiled


def _answer(question, target, answer):
    return {"Q": {"question": question, "target": target, "answer": answer, "source": "model"}}


def _types(node):
    return [node["type"]] + [t for c in node.get("children") or [] for t in _types(c)]


# --------------------------------------------------------------------------
# The worked example (bt_templates.md §8)
# --------------------------------------------------------------------------

class TestWorkedExample:
    def test_keep_while_the_light_is_on_then_two_minutes_later(self, graph):
        compiled = _plan(graph,
                         {"G1": _goal(goal_kind="maintenance", timing={"starts": "now"}),
                          "G2": _goal(timing={"starts": "after_goal", "ref_goal": "G1",
                                              "offset_text": "2 minutes"})},
                         {"P1": {**LIGHT_ON, "role": "scope", "applies_to": ["G1"]}},
                         _answer("maintenance_profile", "G1", {"mode": "native"}))
        root = compiled.tree
        assert root["type"] == "sequence"                       # group(G1)
        unless, after = root["children"]
        # Skipped when the scope is already false at the start (every light
        # off): once the lights' state is read, either that check or the goal.
        known, choice = unless["children"]
        assert known["type"] == "wait_for" and {c["op"] for c in known["checks"]} == {"ne"}
        already, ends = choice["children"]
        assert already["name"] == "not P1"
        assert ends["policy"] == "success_on_one"               # U(G1), scope P1
        stop, body = ends["children"]
        # not P1: every living-room light off.
        assert stop["type"] == "wait_for" and stop["combine"] == "and_"
        assert {(c["key"], c["op"], c["value"]) for c in stop["checks"]} == {
            ("/artifacts/light_1/onOff", "ne", True), ("/artifacts/light_2/onOff", "ne", True)}
        assert [c["type"] for c in body["children"]] == ["action", "running"]
        timer, g2 = after["children"]
        assert (timer["type"], timer["seconds"]) == ("timer", 120.0)
        assert g2["name"] == "achieve G2"
        assert {r["key"] for r in compiled.runtime["reads"]} == {
            "/artifacts/light_1/onOff", "/artifacts/light_2/onOff"}
        assert compiled.runtime["clock"] is True


# --------------------------------------------------------------------------
# Start wrappers (§6)
# --------------------------------------------------------------------------

class TestStarts:
    def test_at_time_and_after_delay(self, graph):
        at = _plan(graph, {"G1": _goal(timing={"starts": "at_time", "time_text": "1:39 PM"})})
        assert at.tree["children"][0] == {"type": "wait_until", "name": "G1: wait for 1:39 PM",
                                          "time_of_day": "13:39"}
        delay = _plan(graph, {"G1": _goal(timing={"starts": "after_delay",
                                                  "offset_text": "13 minutes"})})
        assert delay.tree["children"][0]["seconds"] == 780.0

    def test_a_trigger_with_an_offset_after(self, graph):
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "on_trigger"})},
                         {"P1": {**WASHER_DONE, "role": "trigger", "applies_to": ["G1"],
                                 "offset_text": "5 minutes", "offset_direction": "after"}})
        wait, timer, act = compiled.tree["children"]
        assert wait["checks"] == [{"key": "/artifacts/washer_1/operationalState", "op": "eq",
                                   "value": 0}]
        assert timer["seconds"] == 300.0 and act["type"] == "action"

    def test_a_trigger_before_by_a_countdown(self, graph):
        ref = {"artifact_name": "washer_1", "property_name": "countdownTime"}
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "on_trigger"})},
                         {"P1": {**WASHER_DONE, "role": "trigger", "applies_to": ["G1"],
                                 "offset_text": "23 minutes", "offset_direction": "before"}},
                         _answer("time_to_event", "P1",
                                 {"time_to_event": {"ref": ref, "reads_as": "duration"}}))
        wait = compiled.tree["children"][0]
        assert wait["checks"] == [{"key": "/artifacts/washer_1/countdownTime", "op": "le",
                                   "value": 1380.0}]

    def test_a_trigger_before_by_an_end_time(self, graph):
        ref = {"artifact_name": "washer_1", "property_name": "endTime"}
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "on_trigger"})},
                         {"P1": {**WASHER_DONE, "role": "trigger", "applies_to": ["G1"],
                                 "offset_text": "23 minutes", "offset_direction": "before"}},
                         _answer("time_to_event", "P1",
                                 {"time_to_event": {"ref": ref, "reads_as": "timestamp"}}))
        wait = compiled.tree["children"][0]
        assert wait["type"] == "await"
        assert wait["children"][0]["key1"] == CLOCK_KEY
        (derived,) = compiled.runtime["derived"]
        assert (derived["op"], derived["args"]) == ("timestamp_minus", {"seconds": 1380.0})

    def test_trigger_before_fallbacks(self, graph):
        pred = {"P1": {**WASHER_DONE, "role": "trigger", "applies_to": ["G1"],
                       "offset_text": "1 minute", "offset_direction": "before"}}
        at_event = _plan(graph, {"G1": _goal(timing={"starts": "on_trigger"})}, pred,
                         _answer("time_to_event", "P1", {"fallback": "at_event"}))
        assert at_event.tree["children"][0]["type"] == "wait_for"
        assert any("acting when it happens" in n for n in at_event.notes)
        with pytest.raises(CompileError, match="every goal was left out"):
            _plan(graph, {"G1": _goal(timing={"starts": "on_trigger"})}, pred,
                  _answer("time_to_event", "P1", {"fallback": "skip"}))

    def test_recurrence(self, graph):
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "on_trigger"})},
                         {"P1": {**LIGHT_ON, "role": "recurrence", "applies_to": ["G1"]}})
        loop = compiled.tree
        assert loop["type"] == "repeat" and compiled.long_running
        wait, run, rearm = loop["children"][0]["children"]
        assert wait["combine"] == "or_" and rearm["combine"] == "and_"
        assert run["type"] == "ignore_failure"


# --------------------------------------------------------------------------
# Bodies, ends, guards (§3-5)
# --------------------------------------------------------------------------

class TestBodiesEndsGuards:
    def test_a_guard_skips_when_false(self, graph):
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "now"})},
                         {"P1": {**WASHER_DONE, "role": "guard", "applies_to": ["G1"]}})
        guard = compiled.tree
        assert guard["type"] == "selector" and guard["memory"] is True
        assert guard["children"][0]["type"] == "inverter"

    def test_ends_and_termination_stop_the_goal(self, graph):
        compiled = _plan(graph, {"G1": _goal(goal_kind="maintenance",
                                             timing={"starts": "now",
                                                     "ends": {"kind": "duration",
                                                              "text": "for 1 hour"}})},
                         {"P1": {**WASHER_DONE, "role": "termination", "applies_to": ["G1"]}})
        _, choice = compiled.tree["children"]                # [end conditions read, choice]
        already, ends = choice["children"]
        assert already["type"] == "check"                      # P1 already true: skip
        timer, until, body = ends["children"]
        assert (timer["seconds"], until["type"], body["type"]) == (3600.0, "wait_for", "sequence")
        assert not compiled.long_running                       # it has an end

    def test_regulated_maintenance_polls_and_acts_again(self, graph):
        observed = {"artifact_name": "sensor", "property_name": "temperature"}
        compiled = _plan(graph, {"G1": _goal(goal_kind="maintenance",
                                             target_value_determined="22",
                                             timing={"starts": "now"})},
                         profile=_answer("maintenance_profile", "G1",
                                         {"mode": "regulated", "observed_property": observed}),
                         poll_seconds=15)
        loop = compiled.tree
        assert loop["type"] == "repeat" and compiled.long_running
        in_band_or_act, poll = loop["children"][0]["children"]
        in_band, retry = in_band_or_act["children"]
        assert in_band["checks"][0]["op"] == "le" and retry["type"] == "ignore_failure"
        assert poll["seconds"] == 15
        (derived,) = compiled.runtime["derived"]
        assert derived["op"] == "abs_deviation" and derived["args"] == {"target": 22}

    def test_a_filter_acts_on_every_matching_device(self, graph):
        per_device = {"G1": [("light_1", _act("G1", "light_1")), ("light_2", _act("G1", "light_2"))]}
        compiled = _plan(graph, {"G1": _goal(goal_specificity="incomplete", quantifier="all",
                                             timing={"starts": "now"})},
                         {"P1": {**LIGHT_ON, "location_class": None, "artifact_class": None,
                                 "role": "filter", "applies_to": ["G1"]}},
                         subtrees={}, per_device=per_device)
        branches = compiled.tree["children"]
        assert [b["children"][0]["children"][0]["checks"][0]["key"] for b in branches] == [
            "/artifacts/light_1/onOff", "/artifacts/light_2/onOff"]


# --------------------------------------------------------------------------
# Placement (§7)
# --------------------------------------------------------------------------

class TestPlacement:
    def test_after_goal_on_a_condition(self, graph):
        cond = {"ref": {"artifact_name": "washer_1", "property_name": "operationalState"},
                "comparison": "==", "target_value_text": "0"}
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "now"}),
                                 "G2": _goal(timing={"starts": "after_goal", "ref_goal": "G1"})},
                         profile=_answer("dependency_anchor", "G2",
                                         {"anchor": "condition", "condition": cond}))
        first, after = compiled.tree["children"]
        wait, g2 = after["children"]
        assert wait["checks"][0] == {"key": "/artifacts/washer_1/operationalState", "op": "eq",
                                     "value": 0}

    def test_with_goal_runs_alongside(self, graph):
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "now"}),
                                 "G2": _goal(timing={"starts": "with_goal", "ref_goal": "G1"})})
        assert compiled.tree["type"] == "parallel" and compiled.tree["synchronise"] is True

    def test_completes_with_by_remaining_time(self, graph):
        remaining = {"ref": {"artifact_name": "washer_1", "property_name": "countdownTime"},
                     "reads_as": "duration"}
        duration = {"ref": {"artifact_name": "washer_1", "property_name": "countdownTime"},
                    "reads_as": "duration"}
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "now"}),
                                 "G2": _goal(timing={"completes_with": "G1"})},
                         profile=_answer("completion_timing", "G2",
                                         {"ref_remaining": remaining, "target_duration": duration}))
        _, timed = compiled.tree["children"]
        wait, _ = timed["children"]
        assert wait["children"][0]["op"] == "le"
        assert {d["op"] for d in compiled.runtime["derived"]} == {"to_seconds"}

    def test_completes_with_from_a_start_time_records_it(self, graph):
        duration = {"ref": {"artifact_name": "washer_1", "property_name": "countdownTime"},
                    "reads_as": "duration"}
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "now"}),
                                 "G2": _goal(timing={"completes_with": "G1"})},
                         profile=_answer("completion_timing", "G2",
                                         {"ref_remaining": None, "ref_duration": duration,
                                          "target_duration": duration}))
        assert "stamp" in _types(compiled.tree)
        assert "remaining_from_duration" in {d["op"] for d in compiled.runtime["derived"]}


    # "Time it so the dryer finishes when the washer does": completes_with an
    # EVENT (a predicate), with no goal to hang it off -- it heads its own group.
    def test_completes_with_an_event_waits_until_its_remaining_time_fits(self, graph):
        remaining = {"ref": {"artifact_name": "washer_1", "property_name": "countdownTime"},
                     "reads_as": "duration"}
        compiled = _plan(graph, {"G1": _goal(timing={"completes_with": "P1"}),
                                 "G2": _goal(timing={"starts": "after_goal", "ref_goal": "G1"})},
                         {"P1": {**WASHER_DONE, "role": "trigger", "applies_to": ["G1"]}},
                         profile=_answer("completion_timing", "G1",
                                         {"ref_remaining": remaining,
                                          "target_duration": remaining}))
        assert compiled.tree["name"] == "G1: finish with P1"
        wait, body = compiled.tree["children"]
        assert wait["type"] == "await" and wait["children"][0]["op"] == "le"
        assert compiled.notes == []

    def test_completes_with_an_event_with_no_timing_starts_now(self, graph):
        """Reached only without the planner's check, which asks the user
        first (time_consistency.check_completion): noted, never silent."""
        compiled = _plan(graph, {"G1": _goal(timing={"completes_with": "P1"})},
                         {"P1": {**WASHER_DONE, "role": "trigger", "applies_to": ["G1"]}})
        assert compiled.tree["type"] == "action"
        assert compiled.notes == ["G1: no timing to finish with P1 -> starts now"]

    def test_completes_with_and_a_start_the_user_gave_adds_nothing(self, graph):
        """"Start it now so both finish together": checked before planning,
        placed by its own start -- the same tree as without completes_with."""
        pred = {"P1": {**WASHER_DONE, "role": "trigger", "applies_to": ["G1"]}}
        checked = _plan(graph, {"G1": _goal(timing={"starts": "now", "completes_with": "P1"})},
                        pred)
        plain = _plan(graph, {"G1": _goal(timing={"starts": "now"})})
        assert checked.tree == plain.tree and checked.notes == []


class TestErrors:
    def test_unreadable_time_words(self, graph):
        """"Soon" has no time in it: VagueTime, which the planner turns into a
        question for the user."""
        with pytest.raises(VagueTime, match="no span of time") as raised:
            _plan(graph, {"G1": _goal(timing={"starts": "after_delay", "offset_text": "soon"})})
        assert (raised.value.gid, raised.value.text) == ("G1", "soon")

    def test_a_trigger_no_goal_waits_for(self, graph):
        """Seed 46's lost link: P1 conditions G1, but G1 starts now and does not
        complete with it."""
        with pytest.raises(CompileError, match="P1 is a trigger of G1, but none of them waits"):
            _plan(graph, {"G1": _goal(timing={"starts": "now"})},
                  {"P1": {**WASHER_DONE, "role": "trigger", "applies_to": ["G1"]}})

    def test_a_goal_waiting_for_a_goal_not_in_the_request(self, graph):
        with pytest.raises(CompileError, match="not in the request"):
            _plan(graph, {"G1": _goal(timing={"starts": "now"}),
                          "G2": _goal(timing={"starts": "after_goal", "ref_goal": "G9"})})

    def test_a_qualitative_predicate_cannot_be_read(self, graph):
        pred = GoalStructure.from_dict({"goals": {}, "predicates": {"P1": {
            "predicate_specificity": "ambiguous", "predicate_subject": "environment_property",
            "qualitative_condition": "stuffy", "role": "scope"}}}).predicates["P1"]
        assert "qualitative" in resolve_predicate(graph, pred)


# --------------------------------------------------------------------------
# Running a compiled plan, in plan-clock time
# --------------------------------------------------------------------------

class _Clock:
    def __init__(self):
        self.moment = datetime(2026, 10, 14, 13, 0, 0)

    def now(self):
        return self.moment


def _run(compiled, readings, clock, steps, step_seconds):
    """Tick the plan `steps` times, moving the clock; return (clock time, URL)
    of every action POST."""
    posts = []

    def post(url, payload=None, headers=None):
        posts.append((clock.moment, url))
        return HTTPResponse(200, {}, {}, url, 0.0)

    http = MagicMock()
    http.get.side_effect = lambda url, headers=None: HTTPResponse(200, readings.get(url), {},
                                                                  url, 0.0)
    action_client = MagicMock(post=MagicMock(side_effect=post))

    def compile_child(spec):
        return registry.compile_node(spec, compile_child)

    with patch("ami_agents.bt_planning.nodes.affordance_nodes.HTTPClient",
               return_value=action_client):
        tree = compile_child(compiled.tree)
        tree.setup_with_descendants()
        runtime = PlanRuntime(compiled.runtime, clock=clock, http_client=http, synchronous=True)
        for _ in range(steps):
            runtime.before_tick()
            tree.tick_once()
            import time
            time.sleep(0.005)           # let action futures land
            if tree.status != Status.RUNNING:
                break
            clock.moment += timedelta(seconds=step_seconds)
    return posts, tree.status


class TestRunning:
    def test_after_delay_fires_when_due_in_plan_time(self, graph):
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "after_delay",
                                                     "offset_text": "2 minutes"})})
        clock = _Clock()
        posts, status = _run(compiled, {}, clock, steps=400, step_seconds=1)
        assert status == Status.SUCCESS
        (fired, url), = posts
        assert fired >= datetime(2026, 10, 14, 13, 2, 0)
        assert fired <= datetime(2026, 10, 14, 13, 2, 5)

    def test_a_trigger_fires_once_its_property_turns(self, graph):
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "on_trigger"})},
                         {"P1": {**WASHER_DONE, "role": "trigger", "applies_to": ["G1"]}})
        readings = {"http://h/washer_1/p/operationalState": 1}
        clock = _Clock()
        posts, status = _run(compiled, readings, clock, steps=5, step_seconds=1)
        assert posts == [] and status == Status.RUNNING
        readings["http://h/washer_1/p/operationalState"] = 0
        posts, status = _run(compiled, readings, clock, steps=50, step_seconds=1)
        assert len(posts) == 1 and status == Status.SUCCESS


class TestSameDeviceSequence:
    """`sequential_dependency`: several instructions to one device, each
    starting after the one before, with no offset."""

    def test_instructions_run_one_after_another_with_no_wait(self, graph):
        from ami_agents.agents.interaction_solver.utils.goal_profile import (
            build_checklist,
            fallback_profile,
        )
        goals = {"G1": _goal(timing={"starts": "now"}),
                 "G2": _goal(timing={"starts": "after_goal", "ref_goal": "G1"}),
                 "G3": _goal(timing={"starts": "after_goal", "ref_goal": "G2"})}
        structure = GoalStructure.from_dict({"goals": goals, "predicates": {}},
                                            intent_text="x", structure="dependency")
        profile, _ = fallback_profile(build_checklist(structure), "no model")
        compiled = _plan(graph, goals, profile=profile)
        assert "timer" not in _types(compiled.tree)
        actions = [n for n in _types(compiled.tree) if n == "action"]
        assert len(actions) == 3
        names = []

        def walk(node):
            if node.get("type") == "action":
                names.append(node["name"])
            for child in node.get("children") or []:
                walk(child)

        walk(compiled.tree)
        assert names == [_act(g)["name"] for g in ("G1", "G2", "G3")]
        assert not compiled.long_running

    def test_a_same_device_with_goal_runs_after_not_beside(self, graph):
        # "power on and set fan 30% in 20 minutes, then 50% 23 minutes after":
        # the speed sent with the power-on would reach a fan that is still off.
        same = dict(artifact_name="fan_1")
        compiled = _plan(graph, {
            "G1": _goal(timing={"starts": "after_delay", "offset_text": "20 minutes"}, **same),
            "G2": _goal(timing={"starts": "with_goal", "ref_goal": "G1"}, **same),
            "G3": _goal(timing={"starts": "after_goal", "ref_goal": "G2",
                                "offset_text": "23 minutes"}, **same),
            "G4": _goal(timing={"starts": "with_goal", "ref_goal": "G1"},
                        artifact_name="light_1")})
        timer, beside = compiled.tree["children"]
        assert timer["type"] == "timer" and beside["type"] == "parallel"
        fan, light = beside["children"]
        # The fan: G1, then G2, then (23 minutes later) G3 -- one sequence.
        assert fan["type"] == "sequence"
        assert [n for n in _types(fan) if n in ("action", "timer")] == [
            "action", "action", "timer", "action"]
        # Another device keeps starting with G1.
        assert light["type"] == "action" and light["name"] == _act("G4")["name"]


class TestAlreadyEnded:
    """A lasting goal whose end condition already holds at its start need not
    run; a one-time action gets no end conditions at all."""

    def _run(self, graph, washer_state):
        """Tick a maintenance goal ended by "washer done" twice: once before
        the washer's state is known, once after it is set to ``washer_state``.
        Returns (tree status, the goal body's status)."""
        import py_trees
        from py_trees.common import Access, Status
        from ami_agents.bt_planning.nodes.registry import compile_node
        # 1. Compile: a body that stays RUNNING, so only the end can finish it.
        py_trees.blackboard.Blackboard.clear()
        compiled = _plan(graph, {"G1": _goal(goal_kind="maintenance", timing={"starts": "now"})},
                         {"P1": {**WASHER_DONE, "role": "termination", "applies_to": ["G1"]}},
                         subtrees={"G1": {"type": "running", "name": "achieve G1"}})

        def build(spec):
            return compile_node(spec, build)

        # 2. First tick: the state is not read yet, so nothing is decided.
        tree = build(compiled.tree)
        tree.setup_with_descendants()
        tree.tick_once()
        assert tree.status == Status.RUNNING                   # waits for the reading
        # 3. The reading arrives; the second tick decides skip or run.
        writer = py_trees.blackboard.Client(name="t")
        writer.register_key(key="/artifacts/washer_1/operationalState", access=Access.WRITE)
        writer.set("/artifacts/washer_1/operationalState", washer_state)
        tree.tick_once()
        body = next(n for n in tree.iterate() if n.name == "achieve G1")
        return tree.status, body.status

    def test_skipped_when_it_already_holds(self, graph):
        status, body = self._run(graph, 0)                     # washer already done
        assert status.name == "SUCCESS" and body.name == "INVALID"   # never ticked

    def test_runs_when_it_does_not(self, graph):
        status, body = self._run(graph, 1)
        assert status.name == "RUNNING" and body.name == "RUNNING"

    def test_a_one_time_action_has_no_ends(self, graph):
        """An achievement's termination is left out: its command is done before
        any end could stop it. The plan is the bare action, plus a note."""
        compiled = _plan(graph, {"G1": _goal(timing={"starts": "now"})},
                         {"P1": {**WASHER_DONE, "role": "termination", "applies_to": ["G1"]}})
        assert compiled.tree == _act("G1")
        assert any("G1:" in n and "left out" in n for n in compiled.notes)

