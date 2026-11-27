"""The BEIA benchmark's static check, tested on its own reference plans.

No agents and no LLM. Covered:
  * every case file has what the runner reads (query, ground truth, stimuli);
  * every hand-written reference plan validates, compiles to py_trees, and
    scores `static_pass` against its own case -- so a failed row in a real run
    is the plan's fault, not the scorer's;
  * each kind of fact catches what it is for: a plan that acts after a span
    instead of after a HELD span, closes the kitchen window, sets 22 degrees,
    acts on a device instead of alerting, never re-arms, or reads the wrong
    threshold, fails -- on that fact.
"""

import copy
import json
from pathlib import Path

import pytest

from ami_agents.bt_planning.nodes import registry
from ami_agents.bt_planning.serialization import validate_tree
from tests.BEIA import reference_plans as ref
from tests.BEIA.ground_truth import extract_facts, same_value, score, short_ref

CASES = {json.loads(p.read_text())["id"]: json.loads(p.read_text())
         for p in sorted((Path(__file__).resolve().parents[1] / "BEIA" / "cases").glob("*.json"))}


def _compile(tree):
    """An IR tree as py_trees, as the UA compiles it."""
    def child(spec):
        return registry.compile_node(spec, child)
    return child(tree)


# -- the case files -----------------------------------------------------------------------

def test_six_cases_with_what_the_runner_reads():
    assert sorted(CASES) == ["A1", "A2", "A3", "A4", "A5", "A6"]
    for case in CASES.values():
        assert case["query"] and case["ground_truth"] and case["stimuli"]
        for stimulus in case["stimuli"]:
            assert "at" in stimulus


# -- the reference plans ------------------------------------------------------------------

@pytest.mark.parametrize("case_id, build", [
    (case_id, build) for case_id, builds in ref.REFERENCE_PLANS.items() for build in builds])
def test_each_reference_plan_is_valid_and_passes_its_case(case_id, build):
    """Valid IR, compiles, and every fact of its case holds."""
    plan = build()
    for entry in plan["plans"]:
        assert validate_tree(entry["tree"]) == [], validate_tree(entry["tree"])
        _compile(entry["tree"])
    row = score(plan, CASES[case_id]["ground_truth"])
    assert row["static_pass"], row["missing"]


def test_a_plan_as_the_ua_stores_it():
    """The UA keeps the plan as canonical JSON text; that scores the same."""
    assert score(json.dumps(ref.a4()), CASES["A4"]["ground_truth"])["static_pass"]


# -- each fact catches what it is for -----------------------------------------------------

def _actions(plan):
    """Every action node of a plan, to edit in place."""
    out = []

    def walk(node):
        if node.get("type") == "action":
            out.append(node)
        for child in node.get("children") or []:
            walk(child)
    for entry in plan["plans"]:
        walk(entry["tree"])
    return out


def test_after_a_span_is_not_held_for_a_span():
    """A1 with the timer started once and never raced: no `hold`."""
    plan = ref.a1()
    loop = plan["plans"][0]["tree"]["children"][0]
    loop["children"][0] = ref.seq("rise", [
        ref.wait_for("above", [ref.check(ref.PM25, "gt", 25)]),
        {"type": "timer", "name": "5 minutes", "seconds": 300.0}])
    loop["children"][2] = ref.seq("fall", [
        ref.wait_for("below", [ref.check(ref.PM25, "lt", 12)]),
        {"type": "timer", "name": "10 minutes", "seconds": 600.0}])
    row = score(plan, CASES["A1"]["ground_truth"])
    assert not row["hold_ok"] and not row["static_pass"]
    assert row["reads_ok"] and row["acts_ok"]


def test_the_kitchen_window_is_forbidden():
    plan = ref.a4()
    _actions(plan)[0]["action_url"] = ref.act_url(
        "peroni_22/deschidere_inchidere_geam_telecontact/cover_close_cover")
    row = score(plan, CASES["A4"]["ground_truth"])
    assert row["forbidden_hits"] and not row["acts_ok"] and not row["static_pass"]


def test_22_degrees_is_not_21():
    plan = ref.a6()
    _actions(plan)[1]["parameters"] = {"temperature": 22}
    row = score(plan, CASES["A6"]["ground_truth"])
    assert not row["acts_ok"] and row["forbidden_hits"]


def test_the_raw_ir_route_counts_only_with_the_plug_first():
    """A6's other right answer, through the Fibaro plug and the Broadlink."""
    def raw(order):
        steps = {"plug": ref.action("plug", "garaj/priza_inteligenta_fibaro/switch_turn_on"),
                 "cool": ref.action("cool", "garaj/broadlink_rm4_pro_garaj/send_command_set_value",
                                    {"value": "lg_mode_cool"}),
                 "21": ref.action("21", "garaj/broadlink_rm4_pro_garaj/send_command_set_value",
                                  {"value": "lg_temp_21"})}
        plan = ref.a6()
        loop_steps = plan["plans"][0]["tree"]["children"][0]["children"]
        loop_steps[1]["children"][0] = ref.seq("raw", [steps[k] for k in order])
        return plan
    assert score(raw(["plug", "cool", "21"]), CASES["A6"]["ground_truth"])["static_pass"]
    assert not score(raw(["cool", "21", "plug"]), CASES["A6"]["ground_truth"])["acts_ok"]


def test_an_alert_is_not_a_device_action():
    """A3 acting on a device instead of raising an alert."""
    plan = ref.a3()
    steps = plan["plans"][0]["tree"]["children"][0]["children"]
    steps[1] = ref.action("tell", "garaj/priza_inteligenta_fibaro/switch_turn_on")
    row = score(plan, CASES["A3"]["ground_truth"])
    assert not row["alerts_ok"] and row["forbidden_hits"]


def test_a_rule_that_never_rearms():
    """A2 acting once and done: no `whenever`."""
    plan = ref.a2()
    loop_body = plan["plans"][0]["tree"]["children"][0]
    plan["plans"][0]["tree"] = ref.seq("once", loop_body["children"][:2])
    row = score(plan, CASES["A2"]["ground_truth"])
    assert not row["recurring_ok"] and not row["rearm_ok"]


def test_the_wrong_threshold_is_the_wrong_read():
    plan = json.loads(json.dumps(ref.a5()).replace('"value": 65', '"value": 80'))
    row = score(plan, CASES["A5"]["ground_truth"])
    assert not row["static_pass"]


def test_a_device_that_holds_the_band_itself_is_the_other_right_answer():
    row = score(ref.a5_native(), CASES["A5"]["ground_truth"])
    assert row["static_pass"] and row["alternative"] == 1


# -- the pieces ---------------------------------------------------------------------------

def test_refs_and_values():
    assert short_ref("http://h:8080/workspaces/garaj/artifacts/lg_garaj_hvac/actions/"
                     "climate_set_hvac_mode") == "garaj/lg_garaj_hvac/climate_set_hvac_mode"
    assert same_value("21.0", 21) and same_value(True, "true")
    assert same_value(62, {"min": 60, "max": 65}) and not same_value(66, {"min": 60, "max": 65})
    assert same_value("Cool", ["cool", "auto"])


def test_the_facts_of_a_held_rule():
    facts = extract_facts(ref.a1())
    assert facts.hold and facts.recurring
    assert [a.ref for a in facts.actions] == [
        "garaj/blue_air_garage/fan_set_percentage", "garaj/blue_air_garage/fan_set_preset_mode"]
