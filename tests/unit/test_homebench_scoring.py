"""HomeBench scoring (P4): gold_log, the official metrics, the case rules, the judge
prompt, the extra reading, and the statistics and report built on them.

No LLM is called and no server is started. Records are built by hand from a small home
through the real converter (`convert.build_episode`), so an episode has exactly the shape
the bench scores; its final state comes from the real engine. SimuHome's evaluator modules
cannot be imported in this process (they import as the top-level `src`, which the memory
engine also uses), so the in-process tests put thin fakes under the real HomeBench classes
(`HomeBenchJudge`, `MixedEvaluator`, `evaluate.evaluate_record`); one test runs the real
modules in a subprocess with a fake judge model. The groups, and what each pins:

* gold_log: accepted actions become gold tokens, refused ones do not, nothing is dropped
  silently;
* the official metrics on the edge cases of `code/eval.py` (order, duplicates, `fan only`,
  an empty bucket), and gold-as-prediction giving EM 1.0 in all five buckets -- on a hand-built
  set and, when ``../HomeBench`` is present, on all 17,366 rows through the reverse mapping
  (the self-consistency of D3 and D7);
* each case's pass and fail, `mixed` all-or-nothing, the shared-device exemption, unresolved
  parts, `--no-judge`;
* the judge prompt names the room and the attribute; SimuHome's judge gets only the
  missing-device parts;
* the `turn_off` extra reading: only when the request asks for it, per clause, and the strict
  reading stays recoverable (`matched`, `score` against `score_extra`);
* stats / report: official table, the strict and extra rates, the notes, SimuHome's rows
  unchanged.
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from ami_agents.environment.integration.HomeBench.engine import HomeBenchEngine
from ami_agents.environment.integration.HomeBench.mappings import load_mappings
from tests.homebench import convert, judge_prompt, scoring
from tests.simuhome.bench import evaluate, gold_log, homebench_metrics as hbm, report, stats
from tests.simuhome.bench.suites.homebench import HomeBenchEvaluators

ROOT = Path(__file__).resolve().parents[2]
HOMEBENCH_CHECKOUT = ROOT.parent / "HomeBench"
SIMUHOME_CHECKOUT = ROOT.parent / "SimuHome"
needs_homebench = pytest.mark.skipif(not (HOMEBENCH_CHECKOUT / "dataset").is_dir(),
                                     reason="HomeBench checkout not found")
MAPPINGS = load_mappings()


# -- fixtures: a home, episodes, records ---------------------------------------------------------


def _num(value, low=0, high=100):
    """A HomeBench numeric attribute block."""
    return {"value": value, "lowest": str(low), "highest": str(high)}


def raw_home():
    """Living room (light, air conditioner), kitchen (light, blinds -- no curtain), garage
    (light, humidifier), study room (a light with no attributes) and the vacuum robot."""
    return {
        "living_room": {"room_name": "living_room",
                        "light": {"state": "on", "attributes": {"brightness": _num(40)}},
                        "air_conditioner": {"state": "off", "attributes": {
                            "temperature": _num(24, 16, 30)}}},
        "kitchen": {"room_name": "kitchen",
                    "light": {"state": "on", "attributes": {"brightness": _num(10)}},
                    "blinds": {"state": "closed", "attributes": {}}},
        "garage": {"room_name": "garage",
                   "light": {"state": "on", "attributes": {"brightness": _num(40)}},
                   "humidifier": {"state": "on", "attributes": {"intensity": _num(30)}}},
        "study_room": {"room_name": "study_room",
                       "light": {"state": "on", "attributes": {}}},
        "VacuumRobot": {"state": "on", "attributes": {"battery": _num(80)}},
    }


@pytest.fixture(scope="module")
def home():
    """The home as an ``initial_home_config``."""
    return convert.convert_home({"home_id": 7, "home_status": raw_home()})


def make_episode(home, htype, gold, query, name="home7_x"):
    """A real converter episode for a row, with its home merged in."""
    row = {"id": name, "type": htype, "home_id": 7, "input": query, "output": gold}
    episode = convert.build_episode(row, home, Counter()).episode
    episode["initial_home_config"] = home
    return episode


def url(room, device, name):
    """An SHTD action URL."""
    return f"http://hb.test/workspaces/home7/{room}/artifacts/{device}/actions/{name}"


def act(room, device, name, value=None, success=True):
    """One recorded action request (`execution.PostRecorder`'s entry)."""
    body = {} if value is None else {"value": value}
    return {"url": url(room, device, name), "parameters": body, "success": success,
            "status_code": 200 if success else 422}


def final_state(home, calls):
    """The home's state after gold ``calls`` ``(device_id, command, arg)``, by the engine."""
    machine = HomeBenchEngine(home)
    for device_id, command, arg in calls:
        machine.apply(device_id, command, arg)
    return machine.home_state()


def record_for(episode, home, actions=(), calls=(), plans=(), structures=(), outcome=None,
               answer="done"):
    """A record: the actions sent, the final state, the plan entries, the reply. The
    required ``get_room_devices`` reads are all made."""
    tools = [{"tool": r["tool"], "params": r["params"],
              "outcome": {"ok": True, "status_code": 200, "error_type": None}}
             for r in episode["eval"]["required_actions"]]
    return {"suite": "homebench", "episode": "hb_x", "query_type": episode["meta"]["query_type"],
            "case": episode["meta"]["case"], "outcome": outcome or "executed",
            "meter": {"calls": []}, "tools_invoked": tools,
            "steps": [{"action": "finish", "action_input": {"answer": answer}}],
            "execution": {"actions": list(actions)}, "plans": list(plans),
            "structures": list(structures), "final_home_state": final_state(home, calls)}


# -- fakes under the real HomeBench classes -------------------------------------------------------


def _is_subset(expected, actual):
    """SimuHome's `common.is_subset` (the part `evaluate_required_actions` uses)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(k in actual and _is_subset(v, actual[k]) for k, v in expected.items())
    return expected == actual


class FakeCommon:
    """SimuHome's `episode_evaluation.common`, as far as `HomeBenchJudge` uses it. Records
    each panel's prompts so a test can read them."""

    def __init__(self):
        self.panels = []

    @staticmethod
    def evaluate_required_actions(payload):
        required = payload["episode"]["eval"]["required_actions"]
        return [{"tool": r["tool"], "params": r["params"],
                 "invoked": any(i["tool"] == r["tool"] and _is_subset(r["params"], i["params"])
                                for i in payload["tools_invoked"])} for r in required]

    @staticmethod
    def extract_steps(payload):
        return payload["steps"]

    @staticmethod
    def extract_final_answer(steps):
        for step in reversed(steps):
            if step.get("action") == "finish":
                return str(step["action_input"]["answer"])
        return ""

    def run_judge_panel(self, judges, system, user, **_):
        self.panels.append((system, user))
        return panel_verdict(judges)


def panel_verdict(judges):
    """SimuHome's panel rule: any 'Error' is -1, else the majority of A."""
    if "Error" in judges:
        return {"score": -1, "error_type": "Judge Error", "judge": list(judges)}
    ok = judges.count("A") > judges.count("B")
    return {"score": int(ok), "error_type": None if ok else "Context Faithfulness Failure",
            "judge": list(judges)}


class FakeDeviceJudge:
    """Stands for SimuHome's qt3 infeasible module; remembers the goals it was given."""

    def __init__(self):
        self.seen = []

    def evaluate(self, payload, judges):
        self.seen.append(payload["episode"]["eval"]["goals"])
        return panel_verdict(judges)


class FakeQt3Feasible:
    """SimuHome's qt3 feasible evaluator, line for line: required actions, then equality of
    every assert on the final state (it reads ``goal["asserts"]`` of every goal)."""

    @staticmethod
    def evaluate(payload, judges=None):
        required = FakeCommon.evaluate_required_actions(payload)
        tools_ok = all(a["invoked"] for a in required)
        results = []
        for goal in payload["episode"]["eval"]["goals"]:
            actual_devices = {d["device_id"]: d["attributes"]
                              for room in payload["final_home_state"]["rooms"].values()
                              for d in room["devices"]}
            ok = all(actual_devices.get(goal["device_id"], {}).get(a["attribute"]) == a["value"]
                     for a in goal["asserts"])
            results.append(ok)
        return {"score": int(tools_ok and all(results)), "required_actions": required}


class Registry(dict):
    """`HomeBenchEvaluators`, with fakes in place of SimuHome's modules."""

    describe = staticmethod(HomeBenchEvaluators.describe)

    def __init__(self):
        super().__init__()
        self.device_judge = FakeDeviceJudge()
        self.common = FakeCommon()
        self.judge = scoring.HomeBenchJudge(self.device_judge, self.common)

    def __missing__(self, key):
        _, case = key
        self[key] = {"feasible": FakeQt3Feasible(), "infeasible": self.judge,
                     "mixed": scoring.MixedEvaluator(FakeQt3Feasible(), self.judge)}[case]
        return self[key]


def score(record, episode, judges=("A", "A", "A"), registry=None):
    """The verdict `evaluate_record` gives, with the fakes (judges ``()`` is --no-judge)."""
    registry = Registry() if registry is None else registry     # an empty dict is falsy
    return evaluate.evaluate_record(record, episode, registry, list(judges), None)


# -- gold_log ---------------------------------------------------------------------------------------


class TestGoldLog:
    """Accepted actions become the gold's own strings."""

    def test_tokens_come_from_the_reverse_mapping(self):
        """on/off, a level, an enum and the room-less vacuum robot."""
        log = gold_log.gold_log([
            act("kitchen", "kitchen_light", "onOff", True),
            act("kitchen", "kitchen_light", "onOff", False),
            act("kitchen", "kitchen_light", "brightness", 20),
            act("living_room", "living_room_air_conditioner", "mode", "fan_only"),
            act("household", "household_vacuum_robot", "mode", "auto")])
        assert log.tokens == ["kitchen.light.turn_on()", "kitchen.light.turn_off()",
                              "kitchen.light.set_brightness(20)",
                              "living_room.air_conditioner.set_mode(fan_only)",
                              "vacuum_robot.set_mode(auto)"]
        assert log.unmapped == [] and log.refused == 0

    def test_a_refused_request_is_not_a_token_and_is_counted(self):
        """The home answered an error: it changed nothing."""
        log = gold_log.gold_log([act("kitchen", "kitchen_light", "brightness", 999, False),
                                 act("kitchen", "kitchen_light", "brightness", 20)])
        assert log.tokens == ["kitchen.light.set_brightness(20)"] and log.refused == 1

    def test_an_integral_float_is_an_int_and_repeats_are_kept(self):
        """HBTD makes 20.0 an int; the official metric counts a repeat twice."""
        log = gold_log.gold_log([act("kitchen", "kitchen_light", "brightness", 20.0)] * 2)
        assert log.tokens == ["kitchen.light.set_brightness(20)"] * 2
        assert gold_log.action_value({"input": 5}) == 5 and gold_log.action_value(None) is None

    def test_an_accepted_action_without_a_gold_form_is_listed_not_dropped(self):
        """A foreign URL, an unknown action, and a value the table lacks."""
        log = gold_log.gold_log([
            {"url": "http://x/other", "success": True},
            act("kitchen", "kitchen_light", "nonsense", 1),
            act("kitchen", "kitchen_light", "onOff", "maybe"),
            act("kitchen", "light", "onOff", True)])
        assert log.tokens == [] and len(log.unmapped) == 4
        assert all(item["reason"] for item in log.unmapped)

    def test_no_actions_is_an_empty_log(self):
        """Nothing sent, or no list at all."""
        assert gold_log.gold_log(None).tokens == [] == gold_log.gold_log([]).tokens


# -- the official metrics ----------------------------------------------------------------------------


class TestOfficialMetrics:
    """`homebench_metrics` against the logic of code/eval.py."""

    def test_order_is_ignored(self):
        """Multiset equality: the same tokens in another order are an exact match."""
        row = hbm.compare(hbm.predicted_tokens("{b.x.y(),a.x.y()}"),
                          hbm.gold_tokens("'''a.x.y(),b.x.y(),'''"))
        assert row == {"em": True, "tp": 2, "pred": 2, "gold": 2}

    def test_duplicates_count(self):
        """A repeated call counts twice: a prediction with one copy of a doubled gold is
        not an EM, with two true positives of three... and an extra copy costs precision."""
        gold = ["a.x.y()", "a.x.y()"]
        assert hbm.compare(["a.x.y()"], gold) == {"em": False, "tp": 1, "pred": 1, "gold": 2}
        assert hbm.compare(["a.x.y()"] * 3, gold) == {"em": False, "tp": 2, "pred": 3, "gold": 2}

    def test_fan_only_edge_case_is_kept(self):
        """Spaces are removed from a prediction, so `fan only` misses the gold's `fan_only`;
        the token our own gold_log writes (`fan_only`) matches."""
        gold = hbm.gold_tokens("'''living_room.air_conditioner.set_mode(fan_only)'''")
        typed = hbm.predicted_tokens("{living_room.air_conditioner.set_mode(fan only)}")
        assert typed == ["living_room.air_conditioner.set_mode(fanonly)"]
        assert not hbm.compare(typed, gold)["em"]
        ours = gold_log.gold_log([act("living_room", "living_room_air_conditioner", "mode",
                                      "fan_only")]).tokens
        assert hbm.compare(ours, gold)["em"]

    def test_text_is_read_from_braces_like_eval_py(self):
        """Several {} blocks join with a comma, text outside them is ignored, no braces is an
        empty prediction, newlines and empty items are dropped."""
        assert hbm.predicted_tokens("Sure! {a.x.y()}\nand {error_input}, thanks") == \
            ["a.x.y()", "error_input"]
        assert hbm.predicted_tokens("no braces here") == []
        assert hbm.predicted_tokens("{a.x.y(),,\nb.x.y(),}") == ["a.x.y()", "b.x.y()"]
        assert hbm.gold_tokens("''' a.x.y()'''") == ["a.x.y()"]
        assert hbm.gold_tokens("'''error_input,a.x.y(),'''") == ["error_input", "a.x.y()"]
        assert hbm.predicted_tokens(hbm.prediction_text(["a.x.y()", "error_input"])) == \
            ["a.x.y()", "error_input"]

    def test_micro_precision_recall_f1(self):
        """Counts are summed over rows before dividing (micro), not averaged per row."""
        rows = [{"em": True, "tp": 2, "pred": 2, "gold": 2},
                {"em": False, "tp": 1, "pred": 3, "gold": 2}]
        result = hbm.score(rows)
        assert result["em"] == 0.5
        assert result["precision"] == pytest.approx(3 / 5) and result["recall"] == pytest.approx(3 / 4)
        assert result["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))

    def test_an_empty_bucket_does_not_divide_by_zero(self):
        """eval.py raises ZeroDivisionError; here an empty bucket is None everywhere."""
        empty = hbm.score([])
        assert empty["n"] == 0 and empty["em"] is empty["precision"] is empty["f1"] is None
        buckets = hbm.by_bucket([{"bucket": "mix_multi", **hbm.compare(["a"], ["a"])}])
        assert list(buckets) == ["all", *hbm.BUCKETS]
        assert buckets["normal_single"]["em"] is None and buckets["mix_multi"]["em"] == 1.0

    def test_nothing_predicted_is_precision_zero_not_an_error(self):
        """A bucket where no token was predicted: precision 0.0, recall 0.0, F1 0.0."""
        result = hbm.score([hbm.compare([], ["a"])])
        assert (result["em"], result["precision"], result["recall"], result["f1"]) == (0.0,) * 4

    def test_buckets_agree_with_the_converter_for_every_type(self):
        """`bucket_of` is eval.py's rule, and equals `convert.official_bucket` everywhere."""
        types = ["normal", "unexist_device", "unexist_attribute"] + [
            f"multi{n}_{kind}" for n in range(2, 11)
            for kind in ("mix", "normal", "unexist_device", "unexist_attribute")]
        for homebench_type in types:
            assert hbm.bucket_of(homebench_type) == convert.official_bucket(homebench_type)
        with pytest.raises(ValueError):
            hbm.bucket_of("bogus")

    def test_a_goal_asked_about_is_no_error_input(self):
        """Impossible entries are `error_input` tokens; a clarification is no output."""
        record = {"plans": [{"outcome": "impossible"}, {"outcome": "clarify"},
                            {"outcome": "plan"}],
                  "structures": [{"clarification": "which room?"}, {}]}
        assert hbm.refused_goals(record) == {"impossible": 1, "clarify": 2}
        assert hbm.prediction(["a"], record) == ["a", "error_input"]

    def test_generic_part_rates(self):
        """Parts passed over parts; a rate over no parts is None."""
        blocks = [{"parts": {"feasible": 2, "feasible_ok": 1, "infeasible": 2, "infeasible_ok": 2}},
                  {"parts": {"feasible": 2, "feasible_ok": 2, "infeasible": 0, "infeasible_ok": 0}}]
        rates = hbm.parts_rates(blocks)
        assert rates == {"parts_rate": 0.833, "feasible_rate": 0.75, "impossible_rate": 1.0,
                         "parts": 6}
        assert hbm.parts_rates([])["parts_rate"] is None


# -- gold as the prediction: self-consistency of D3 and D7 --------------------------------------------


def gold_actions(gold_text, multi):
    """The action requests that reproduce a gold string (the reverse of `gold_log`), and
    the number of `error_input` parts; calls no action issues are returned apart."""
    actions, refusals, unreachable = [], 0, []
    for call in convert.parse_gold(gold_text, multi):
        if call is None:
            refusals += 1
            continue
        try:
            name, value = MAPPINGS.action_for_command(call.device, call.command, call.arg)
        except Exception:  # noqa: BLE001 - an unreachable vacuum call
            unreachable.append(call)
            continue
        actions.append(act(call.room or "household", call.device_id, name, value))
    return actions, refusals, unreachable


class TestGoldAsPrediction:
    """A record that did exactly what the gold says scores EM 1.0 (`error_input` for each
    part answered impossible)."""

    ROWS = [("normal", "'''kitchen.light.set_brightness(20)'''", "Set the light to 20 in the kitchen."),
            ("unexist_device", "'''error_input'''", "Close the curtains in the kitchen."),
            ("multi2_normal", "'''kitchen.light.turn_on(),garage.humidifier.set_intensity(5),'''",
             "Turn on the light in the kitchen, set the humidifier intensity to 5 in the garage."),
            ("multi2_mix", "'''kitchen.light.set_brightness(20),error_input,'''",
             "Set the light brightness to 20 in the kitchen, close the curtains in the garage."),
            ("multi2_unexist_device", "'''error_input,error_input,'''",
             "Close the curtains in the garage, close the curtains in the study room.")]

    def test_every_bucket_reaches_em_one_on_a_hand_built_set(self, home):
        """One row per official bucket, through gold_log, `prediction` and `by_bucket`."""
        rows = []
        for homebench_type, gold, query in self.ROWS:
            episode = make_episode(home, homebench_type, gold, query)
            actions, refusals, unreachable = gold_actions(
                gold, convert.multi_parts(homebench_type) is not None)
            assert not unreachable
            record = record_for(episode, home, actions=actions,
                                plans=[{"outcome": "impossible"}] * refusals)
            block = scoring.episode_block(record, episode, {})
            assert block["em"]
            assert Counter(block["predicted_tokens"]) == Counter(block["gold_tokens"])
            rows.append({"bucket": block["bucket"], **hbm.compare(block["predicted_tokens"],
                                                                  block["gold_tokens"])})
        buckets = hbm.by_bucket(rows)
        assert [b for b in hbm.BUCKETS if buckets[b]["n"] == 1] == list(hbm.BUCKETS)
        assert all(buckets[b]["em"] == 1.0 and buckets[b]["f1"] == 1.0
                   for b in (hbm.ALL, *hbm.BUCKETS))

    @needs_homebench
    def test_all_17366_gold_rows_round_trip_through_the_mapping(self):
        """Every gold call becomes an action request and back to the same token: EM 1.0 in
        all five buckets, the only misses being the rows with a vacuum call no action can
        issue (plan 3.5), which cannot be reproduced by construction."""
        rows = convert.read_jsonl(HOMEBENCH_CHECKOUT / "dataset" / "test_data.jsonl")
        assert len(rows) == 17366
        scored, misses, unreachable_rows = [], [], set()
        for row in rows:
            multi = convert.multi_parts(row["type"]) is not None
            actions, refusals, unreachable = gold_actions(row["output"], multi)
            if unreachable:
                unreachable_rows.add(row["id"])
            predicted = hbm.prediction(gold_log.gold_log(actions).tokens,
                                       {"plans": [{"outcome": "impossible"}] * refusals})
            result = hbm.compare(predicted, hbm.gold_tokens(row["output"]))
            scored.append({"bucket": hbm.bucket_of(row["type"]), **result})
            if not result["em"]:
                misses.append(row["id"])
        assert set(misses) == unreachable_rows and 0 < len(misses) <= 70
        reproducible = [r for r, row in zip(scored, rows) if row["id"] not in unreachable_rows]
        buckets = hbm.by_bucket(reproducible)
        assert all(buckets[b]["em"] == 1.0 and buckets[b]["precision"] == 1.0
                   and buckets[b]["recall"] == 1.0 for b in (hbm.ALL, *hbm.BUCKETS))
        assert {b: hbm.by_bucket(scored)[b]["n"] for b in hbm.BUCKETS} == {
            "normal_single": 6187, "unexist_single": 6765, "normal_multi": 245,
            "mix_multi": 4072, "error_multi": 97}


# -- each case's rules --------------------------------------------------------------------------------


class TestFeasible:
    """SimuHome's evaluator on the feasible case, with the breakdown."""

    GOLD = "'''kitchen.light.set_brightness(20)'''"
    QUERY = "Set the brightness of the light to 20 in the kitchen."

    def test_pass(self, home):
        """The final state holds the gold value."""
        episode = make_episode(home, "normal", self.GOLD, self.QUERY)
        record = record_for(episode, home, calls=[("kitchen_light", "set_brightness", 20)],
                            actions=[act("kitchen", "kitchen_light", "brightness", 20)])
        verdict = score(record, episode)
        assert verdict["score"] == 1 == verdict["score_extra"]
        check = verdict["breakdown"]["checks"][0]
        assert check["class"] == "ok" and check["matched"] == "simuhome"
        assert verdict["homebench"]["em"] and verdict["homebench"]["bucket"] == "normal_single"

    def test_fail_wrong_value_and_missing(self, home):
        """Another value is wrong_value, an untouched device is missing; both score 0."""
        episode = make_episode(home, "normal", self.GOLD, self.QUERY)
        wrong = score(record_for(episode, home, calls=[("kitchen_light", "set_brightness", 30)]),
                      episode)
        missing = score(record_for(episode, home), episode)
        assert wrong["score"] == 0 == wrong["score_extra"]
        assert wrong["breakdown"]["checks"][0]["class"] == "wrong_value"
        assert missing["breakdown"]["checks"][0]["class"] == "missing"
        assert "matched" not in missing["breakdown"]["checks"][0]

    def test_required_actions_are_part_of_the_pass(self, home):
        """The right state without having looked at the room is still a fail."""
        episode = make_episode(home, "normal", self.GOLD, self.QUERY)
        record = record_for(episode, home, calls=[("kitchen_light", "set_brightness", 20)])
        record["tools_invoked"] = []
        assert score(record, episode)["score"] == 0


class TestUnexistDevice:
    """`infeasible`, the device is not in the room (the kitchen has blinds, no curtain)."""

    GOLD, QUERY = "'''error_input'''", "Close the curtains in the kitchen."

    def episode(self, home):
        return make_episode(home, "unexist_device", self.GOLD, self.QUERY)

    def refused(self):
        return {"plans": [{"outcome": "impossible"}], "structures": [{}]}

    def test_pass_when_refused_and_nothing_done(self, home):
        """Answered impossible, nothing executed, the judge agrees."""
        episode = self.episode(home)
        goal = episode["eval"]["goals"][0]
        assert goal["reason"] == "unexist_device" and goal["device_id"] == "kitchen_curtain"
        verdict = score(record_for(episode, home, **self.refused(), outcome="impossible"),
                        episode)
        assert verdict["score"] == 1 and verdict["execution_check"]["score"] == 1
        assert verdict["homebench"]["predicted_tokens"] == ["error_input"]
        assert verdict["homebench"]["em"] and verdict["homebench"]["parts"]["infeasible_ok"] == 1

    def test_fail_when_a_stand_in_was_operated(self, home):
        """The blinds were closed for the missing curtain: any action in the part's room
        no feasible part asks for is an offence (the plan's stand-in rule would match
        nothing here, the missing device's id never exists)."""
        episode = self.episode(home)
        record = record_for(episode, home, **self.refused(),
                            actions=[act("kitchen", "kitchen_blinds", "openClose", "closed")])
        verdict = score(record, episode)
        assert verdict["score"] == 0 and verdict["error_type"] == "Executed Infeasible Part"
        assert verdict["execution_check"]["offending_actions"][0]["device"] == "kitchen_blinds"
        assert not verdict["homebench"]["em"]

    def test_an_action_in_another_room_is_not_an_offence(self, home):
        """The part's scope is its room."""
        episode = self.episode(home)
        record = record_for(episode, home, **self.refused(),
                            actions=[act("garage", "garage_light", "onOff", True)])
        assert score(record, episode)["execution_check"]["score"] == 1

    def test_fail_when_not_answered_impossible(self, home):
        """Nothing was done and nothing was said."""
        episode = self.episode(home)
        verdict = score(record_for(episode, home), episode)
        assert verdict["score"] == 0 and verdict["error_type"] == "Not Reported Infeasible"

    def test_the_judge_decides_the_rest(self, home):
        """A good execution check and a failing panel is a fail; a judge error is -1;
        --no-judge leaves the execution check alone."""
        episode = self.episode(home)
        record = record_for(episode, home, **self.refused(), outcome="impossible")
        failed = score(record, episode, judges=("B", "B", "A"))
        assert failed["score"] == 0 and failed["execution_check"]["score"] == 1
        assert failed["error_type"] == "Context Faithfulness Failure"
        assert score(record, episode, judges=("A", "Error", "A"))["score"] == -1
        unjudged = score(record, episode, judges=())
        assert unjudged["score"] == 1 and "judge" not in unjudged
        assert unjudged["homebench"]["judged"] is False

    def test_a_clarification_passes_the_check_but_is_no_error_input(self, home):
        """Asked, nothing done: the primary score passes; the official prediction is empty."""
        episode = self.episode(home)
        record = record_for(episode, home, structures=[{"clarification": "which room?"}],
                            outcome="clarify")
        verdict = score(record, episode)
        assert verdict["score"] == 1
        assert verdict["homebench"]["predicted_tokens"] == [] and verdict["homebench"]["clarified"] == 1
        assert not verdict["homebench"]["em"]


class TestUnexistAttribute:
    """`infeasible`, the device exists and lacks the attribute."""

    GOLD = "'''error_input'''"
    QUERY = "Set the brightness of the air conditioner to 50 in the living room."

    def episode(self, home):
        return make_episode(home, "unexist_attribute", self.GOLD, self.QUERY)

    def test_the_part_is_an_attribute_part(self, home):
        """Resolved, the device exists, the attribute is named."""
        goal = self.episode(home)["eval"]["goals"][0]
        assert (goal["reason"], goal["device_id"], goal["attribute"]) == (
            "unexist_attribute", "living_room_air_conditioner", "brightness")

    def test_pass_and_fail_on_the_device(self, home):
        """Nothing done on that device passes; an action on it (a substitute) fails."""
        episode = self.episode(home)
        refused = {"plans": [{"outcome": "impossible"}], "outcome": "impossible"}
        assert score(record_for(episode, home, **refused), episode)["score"] == 1
        record = record_for(episode, home, **refused, calls=[
            ("living_room_air_conditioner", "set_temperature", 20)],
            actions=[act("living_room", "living_room_air_conditioner", "temperature", 20)])
        verdict = score(record, episode)
        assert verdict["score"] == 0 and verdict["error_type"] == "Executed Infeasible Part"
        assert verdict["execution_check"]["infeasible_part"]["parts"][0]["scope"] == "device"

    def test_the_judge_prompt_names_the_room_and_the_attribute(self, home):
        """HomeBench's panel reads a prompt with both, and says the device exists."""
        episode = self.episode(home)
        registry = Registry()
        record = record_for(episode, home, plans=[{"outcome": "impossible"}], outcome="impossible",
                            answer="The air conditioner in the living room has no brightness.")
        assert score(record, episode, registry=registry)["score"] == 1
        (system, user), = registry.common.panels
        assert "living room" in user and "brightness" in user and "air conditioner" in user
        assert "exists but has no attribute" in user and self.QUERY in user
        assert "NO such attribute" in system and "The air conditioner in the living" in user
        assert registry.device_judge.seen == []          # SimuHome's judge is not used for it

    def test_render_goals_for_each_kind_of_part(self):
        """A located attribute part, a located part without an attribute, an unidentified one."""
        located = {"room_id": "living_room", "device_type": "air_conditioner",
                   "reason": "unexist_attribute", "attribute": "fan_speed"}
        assert judge_prompt.render_goal(1, located) == (
            "- Goal 1: room 'living room', device 'air conditioner' exists but has no "
            "attribute 'fan speed'")
        assert "does not offer" in judge_prompt.render_goal(
            2, {"room_id": "garage", "device_type": "fan", "reason": "unknown"})
        assert "were not identified" in judge_prompt.render_goal(
            3, {"room_id": None, "device_id": None, "reason": "unknown"})
        assert judge_prompt.render_goals([located, located]).count("\n") == 1

    def test_a_missing_final_answer_is_a_failure_without_a_judge_call(self, home):
        """SimuHome's early exits: no final answer, no panel."""
        episode = self.episode(home)
        registry = Registry()
        record = record_for(episode, home, plans=[{"outcome": "impossible"}], answer="")
        verdict = score(record, episode, registry=registry)
        assert verdict["score"] == 0 and registry.common.panels == []
        assert verdict["judge"]["panels"]["attribute"]["error_type"] == "Final Response Failure"


class TestMixed:
    """`mixed`: every part must pass."""

    GOLD = "'''kitchen.light.set_brightness(20),error_input,'''"
    QUERY = "Set the brightness of the light to 20 in the kitchen, close the curtains in the garage."

    def episode(self, home):
        return make_episode(home, "multi2_mix", self.GOLD, self.QUERY)

    def good(self, episode, home, **changes):
        """The record that satisfies every part, with ``changes`` applied."""
        args = dict(calls=[("kitchen_light", "set_brightness", 20)],
                    actions=[act("kitchen", "kitchen_light", "brightness", 20)],
                    plans=[{"outcome": "plan"}, {"outcome": "impossible"}])
        args.update(changes)
        return record_for(episode, home, **args)

    def test_all_parts_pass(self, home):
        """Feasible part done, infeasible part refused with nothing done, judge agrees."""
        episode = self.episode(home)
        goals = episode["eval"]["goals"]
        assert [bool(g.get("feasibility") is False) for g in goals] == [False, True]
        verdict = score(self.good(episode, home), episode)
        assert verdict["score"] == 1 == verdict["score_extra"]
        assert verdict["homebench"]["em"] and verdict["homebench"]["bucket"] == "mix_multi"
        assert verdict["homebench"]["parts"] == {"feasible": 1, "feasible_ok": 1,
                                                  "infeasible": 1, "infeasible_ok": 1}

    @pytest.mark.parametrize("change,error", [
        ({"calls": [], "actions": []}, "Goal Failure"),
        ({"plans": [{"outcome": "plan"}]}, "Not Reported Infeasible"),
        ({"actions": [act("kitchen", "kitchen_light", "brightness", 20),
                      act("garage", "garage_light", "onOff", False)]}, "Executed Infeasible Part"),
    ], ids=["feasible-part-fails", "infeasible-part-not-refused", "infeasible-part-executed"])
    def test_one_failing_part_fails_the_episode(self, home, change, error):
        """All-or-nothing, and the error names the part that failed."""
        episode = self.episode(home)
        verdict = score(self.good(episode, home, **change), episode)
        assert verdict["score"] == 0 and verdict["error_type"] == error
        assert not verdict["homebench"]["em"]

    def test_a_failing_judge_fails_it_and_a_judge_error_is_minus_one(self, home):
        """The judge is the third component."""
        episode = self.episode(home)
        record = self.good(episode, home)
        assert score(record, episode, judges=("B", "B", "B"))["score"] == 0
        assert score(record, episode, judges=("Error", "A", "A"))["score"] == -1
        assert score(record, episode, judges=())["score"] == 1          # --no-judge

    def test_the_feasible_evaluator_never_sees_an_infeasible_goal(self, home):
        """SimuHome's qt3 feasible module reads goal['asserts'] of every goal; it is given
        the feasible parts only, and SimuHome's judge the missing-device parts only."""
        episode = self.episode(home)
        registry = Registry()
        assert score(self.good(episode, home), episode, registry=registry)["score"] == 1
        (seen,) = registry.device_judge.seen
        assert [g["reason"] for g in seen] == ["unexist_device"]
        assert [g["device_id"] for g in scoring.feasible_view(episode)["eval"]["goals"]] == [
            "kitchen_light"]

    def test_a_shared_device_does_not_make_the_asked_action_an_offence(self, home):
        """The garage light is asked to change (feasible) and to have an intensity (it has
        none): the brightness action is the feasible part's, an `onOff` is an offence."""
        episode = make_episode(
            home, "multi2_mix", "'''garage.light.set_brightness(20),error_input,'''",
            "Set the brightness of the light to 20 in the garage, set the intensity of the light "
            "in the garage to 5.")
        part = episode["eval"]["goals"][1]
        assert part["reason"] == "unexist_attribute" and part["device_id"] == "garage_light"
        plans = [{"outcome": "plan"}, {"outcome": "impossible"}]
        legit = record_for(episode, home, plans=plans,
                           calls=[("garage_light", "set_brightness", 20)],
                           actions=[act("garage", "garage_light", "brightness", 20)])
        assert score(legit, episode)["score"] == 1
        stray = record_for(episode, home, plans=plans,
                           calls=[("garage_light", "set_brightness", 20)],
                           actions=[act("garage", "garage_light", "brightness", 20),
                                    act("garage", "garage_light", "onOff", False)])
        assert score(stray, episode)["execution_check"]["score"] == 0

    def test_both_judges_run_when_both_kinds_of_part_are_present(self, home):
        """A missing device and a missing attribute: SimuHome's panel and HomeBench's, and the
        verdict passes only if both do."""
        episode = make_episode(
            home, "multi2_unexist_device", "'''error_input,error_input,'''",
            "Close the curtains in the kitchen, set the brightness of the air conditioner to 5 "
            "in the living room.")
        reasons = [g["reason"] for g in episode["eval"]["goals"]]
        assert reasons == ["unexist_device", "unexist_device"]      # the label wins ...
        episode["eval"]["goals"][1]["reason"] = "unexist_attribute"  # ... so set the second by hand
        record = record_for(episode, home, plans=[{"outcome": "impossible"}] * 2,
                            outcome="impossible")
        registry = Registry()
        verdict = score(record, episode, registry=registry)
        assert set(verdict["judge"]["panels"]) == {"device", "attribute"}
        assert len(verdict["judge"]["judge"]) == 6 and verdict["score"] == 1
        assert len(registry.common.panels) == 1 and len(registry.device_judge.seen) == 1

    def test_an_unresolved_part_covers_every_unasked_action(self, home):
        """Nothing says where the part was: any accepted action no feasible part asks for is
        an offence, wherever it happened; the part is counted as unresolved."""
        episode = self.episode(home)
        episode["eval"]["goals"][1].update(room_id=None, device_id=None, device_type=None,
                                           resolved=False, reason="unknown")
        quiet = score(self.good(episode, home), episode)
        assert quiet["score"] == 1 and quiet["homebench"]["unresolved"] == 1
        assert quiet["execution_check"]["unresolved_parts"] == 1
        stray = score(self.good(episode, home, actions=[
            act("kitchen", "kitchen_light", "brightness", 20),
            act("study_room", "study_room_light", "onOff", False)]), episode)
        assert stray["score"] == 0 and stray["error_type"] == "Executed Infeasible Part"

    def test_the_unknown_part_goes_to_homebenchs_judge(self, home):
        """An unidentified part has no device to name: HomeBench's prompt, which reads the
        request, judges it (SimuHome's would demand a device that is not known)."""
        episode = self.episode(home)
        episode["eval"]["goals"][1].update(room_id=None, device_id=None, device_type=None,
                                           resolved=False, reason="unknown")
        registry = Registry()
        assert score(self.good(episode, home), episode, registry=registry)["score"] == 1
        assert registry.device_judge.seen == []
        assert "were not identified" in registry.common.panels[0][1]


# -- the extra reading ---------------------------------------------------------------------------------


class TestTurnOffReading:
    """A gold zero level for a request that asks to turn the device off is also met by
    the device being off (our criterion, not HomeBench's)."""

    def off_record(self, episode, home, device="kitchen_light", commands=("turn_off",)):
        """The agent turned the device off instead of writing the level."""
        return record_for(episode, home, calls=[(device, c, None) for c in commands],
                          actions=[act("kitchen", device, "onOff", False)])

    def test_accepted_when_the_request_says_turn_off(self, home):
        """Strict: brightness 10 != 0 fails; the extra reading holds (state off). `score`
        keeps the benchmark's reading, `score_extra` has the other, and `matched` says which
        reading each check met."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(0)'''",
                               "Turn off the light in the kitchen.")
        verdict = score(self.off_record(episode, home), episode)
        check = verdict["breakdown"]["checks"][0]
        assert verdict["score"] == 0 and verdict["score_extra"] == 1
        assert check["class"] == "ok" and check["matched"] == "turn_off"
        assert check["accepted"] == {"simuhome": 0, "turn_off": "state == off"}
        assert check["expected"] == 0

    def test_the_strict_reading_is_recoverable(self, home):
        """The gold's own value still matches as "simuhome", and `score` is the strict rate."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(0)'''",
                               "Turn off the light in the kitchen.")
        record = record_for(episode, home, calls=[("kitchen_light", "set_brightness", 0)],
                            actions=[act("kitchen", "kitchen_light", "brightness", 0)])
        verdict = score(record, episode)
        assert verdict["score"] == 1 == verdict["score_extra"]
        assert verdict["breakdown"]["checks"][0]["matched"] == "simuhome"

    @pytest.mark.parametrize("query", [
        "Set the brightness of the light to the lowest level in the kitchen.",
        "Dim the light in the kitchen to zero."])
    def test_not_accepted_when_the_request_does_not_ask_for_it(self, home, query):
        """Turning it off does not meet "set the level"; the strict reading stands."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(0)'''", query)
        verdict = score(self.off_record(episode, home), episode)
        assert verdict["score"] == 0 == verdict["score_extra"]
        assert verdict["breakdown"]["checks"][0]["class"] != "ok"
        assert "accepted" not in verdict["breakdown"]["checks"][0]

    def test_switch_off_counts_and_other_levels_never_do(self, home):
        """"switch off" is asked too; a gold of 20 has no extra reading even for a turn-off
        query (only a zero level is relaxed)."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(0)'''",
                               "Switch off the light in the kitchen.")
        assert score(self.off_record(episode, home), episode)["score_extra"] == 1
        other = make_episode(home, "normal", "'''kitchen.light.set_brightness(20)'''",
                             "Turn off the light in the kitchen and set the brightness to 20.")
        assert scoring.extra_readings(other, other["eval"]["goals"][0], "brightness", 20) == {}

    def test_in_a_multi_request_only_the_parts_own_clause_counts(self, home):
        """"Turn off the humidifier" does not turn "set the light brightness to 0" into a
        turn-off: the light part keeps the strict reading, the humidifier part has both."""
        episode = make_episode(
            home, "multi2_normal",
            "'''garage.humidifier.set_intensity(0),kitchen.light.set_brightness(0),'''",
            "Turn off the humidifier in the garage, set the brightness of the light to 0 "
            "in the kitchen.")
        humidifier, light = episode["eval"]["goals"]
        assert scoring.asks_turn_off(episode, humidifier) is True
        assert scoring.asks_turn_off(episode, light) is False
        record = record_for(episode, home,
                            calls=[("garage_humidifier", "turn_off", None)],
                            actions=[act("garage", "garage_humidifier", "onOff", False)])
        checks = score(record, episode)["breakdown"]["checks"]
        assert [c["class"] for c in checks] == ["ok", "missing"]
        assert checks[0]["matched"] == "turn_off"

    def test_an_unattributable_clause_keeps_the_strict_reading(self, home):
        """A multi request whose clauses do not name the part's room and device: no extra."""
        episode = make_episode(
            home, "multi2_normal",
            "'''kitchen.light.set_brightness(0),garage.humidifier.set_intensity(5),'''",
            "Turn off everything, then humidify the garage a little.")
        assert scoring.asks_turn_off(episode, episode["eval"]["goals"][0]) is False

    def test_a_mixed_episode_can_be_rescued_by_the_reading_alone(self, home):
        """`score_extra` is 1 only when nothing but a reading-matched value check differs."""
        episode = make_episode(
            home, "multi2_mix", "'''kitchen.light.set_brightness(0),error_input,'''",
            "Turn off the light in the kitchen, close the curtains in the garage.")
        record = record_for(episode, home, calls=[("kitchen_light", "turn_off", None)],
                            actions=[act("kitchen", "kitchen_light", "onOff", False)],
                            plans=[{"outcome": "plan"}, {"outcome": "impossible"}])
        verdict = score(record, episode)
        assert (verdict["score"], verdict["score_extra"]) == (0, 1)
        refused_nothing = record_for(episode, home, calls=[("kitchen_light", "turn_off", None)],
                                     actions=[act("kitchen", "kitchen_light", "onOff", False)])
        assert score(refused_nothing, episode)["score_extra"] == 0     # the other part fails

    def test_matched_is_stored_for_every_ok_check(self):
        """SimuHome's checks carry `matched` too (a percent reading names itself)."""
        assert evaluate._ok(None) == {"class": "ok", "matched": "simuhome"}
        assert evaluate._ok("simuhome") == {"class": "ok", "matched": "simuhome"}
        assert evaluate._ok("percent_of_range") == {"class": "ok", "matched": "percent_of_range",
                                                    "reading": "percent_of_range"}


# -- infrastructure, wiring ------------------------------------------------------------------------------


class TestWiring:
    """What `evaluate_record` does before and around the evaluators."""

    def test_infrastructure_and_stopped_runs_have_no_block(self, home):
        """-1 before any rule: nothing for the official metrics to read."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(20)'''",
                               "Set the brightness of the light to 20 in the kitchen.")
        record = record_for(episode, home, outcome="stopped")
        verdict = score(record, episode)
        assert verdict["score"] == -1 and "homebench" not in verdict
        record = record_for(episode, home)
        record["error"] = "SimuHomeError: 502"
        assert score(record, episode)["error_type"] == "Infrastructure"

    def test_judged_cases_include_mixed(self):
        """The CLI builds a judge panel for a mixed record as for an infeasible one."""
        assert evaluate.JUDGED_CASES == ("infeasible", "mixed")

    def test_a_plain_registry_adds_no_block(self, home):
        """SimuHome's evaluators registry (a plain dict) has no `describe`; hb or not, nothing
        else changes."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(20)'''",
                               "Set the brightness of the light to 20 in the kitchen.")
        plain = {("hb1", "feasible"): FakeQt3Feasible()}
        record = record_for(episode, home, calls=[("kitchen_light", "set_brightness", 20)])
        assert "homebench" not in evaluate.evaluate_record(record, episode, plain, [], None)

    def test_an_episode_without_gold_is_refused_clearly(self, home):
        """The converter always writes `meta.gold`; without it the execution check cannot know
        which actions were asked for, and says so."""
        episode = make_episode(home, "unexist_device", "'''error_input'''",
                               "Close the curtains in the kitchen.")
        del episode["meta"]["gold"]
        with pytest.raises(ValueError, match="meta.gold"):
            scoring.check_execution({}, episode, True)


# -- the real SimuHome modules, with a fake judge model ----------------------------------------------------


@pytest.mark.skipif(not SIMUHOME_CHECKOUT.is_dir(), reason="SimuHome checkout not found")
class TestWithSimuHomesModules:
    """The registry over the real qt3 evaluators and `run_judge_panel`, in a subprocess (their
    top-level `src` cannot be imported here), with a fake judge that always answers A."""

    SCRIPT = r'''
import json, sys
from collections import Counter
sys.path.insert(0, ".")
from tests.homebench import convert, scoring
from tests.simuhome.bench import evaluate
from tests.simuhome.bench.suites import get_suite

class FakeJudge:
    def __init__(self): self.prompts = []
    def generate(self, messages, response_format=None):
        self.prompts.append(messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].content)
        return "A"

payload = json.load(sys.stdin)
episode, record = payload["episode"], payload["record"]
registry = get_suite("homebench").evaluators()
judges = [FakeJudge() for _ in range(3)]
kinds = {f"hb{n}-{case}": type(registry[(f"hb{n}", case)]).__name__
         for n in (1, 5, 10) for case in ("feasible", "infeasible", "mixed")}
verdict = evaluate.evaluate_record(record, episode, registry, judges, None)
print(json.dumps({"kinds": kinds, "verdict": verdict, "prompts": [j.prompts for j in judges]},
                 default=str))
'''

    def run_script(self, episode, record):
        result = subprocess.run([sys.executable, "-c", self.SCRIPT], cwd=ROOT, text=True,
                                input=json.dumps({"episode": episode, "record": record}),
                                capture_output=True, timeout=180)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_registry_and_a_mixed_episode_end_to_end(self, home):
        """Both kinds of part in one mixed episode: SimuHome's own judge prompt for the missing
        device, ours (room and attribute named) for the missing attribute, six judge calls."""
        episode = make_episode(
            home, "multi3_mix",
            "'''kitchen.light.set_brightness(20),error_input,error_input,'''",
            "Set the brightness of the light to 20 in the kitchen, close the curtains in the "
            "garage, set the brightness of the air conditioner to 50 in the living room.")
        reasons = [g.get("reason") for g in episode["eval"]["goals"]]
        assert reasons == [None, "unexist_device", "unexist_attribute"]
        record = record_for(
            episode, home, calls=[("kitchen_light", "set_brightness", 20)],
            actions=[act("kitchen", "kitchen_light", "brightness", 20)],
            plans=[{"outcome": "plan"}, {"outcome": "impossible"}, {"outcome": "impossible"}],
            answer="Done. There are no curtains in the garage, and the living room air "
                   "conditioner has no brightness.")
        out = self.run_script(episode, record)
        assert out["kinds"]["hb1-feasible"] == "module"       # SimuHome's qt3 module itself
        assert out["kinds"]["hb5-infeasible"] == "HomeBenchJudge"
        assert out["kinds"]["hb10-mixed"] == "MixedEvaluator"
        verdict = out["verdict"]
        assert verdict["score"] == 1 and verdict["score_extra"] == 1
        assert set(verdict["judge"]["panels"]) == {"device", "attribute"}
        assert verdict["judge"]["judge"] == ["A"] * 6
        prompts = [p for judge in out["prompts"] for p in judge]
        assert len(prompts) == 6
        ours = [p for p in prompts if "exists but has no attribute" in p]
        simuhome = [p for p in prompts if "exists but has no attribute" not in p]
        assert len(ours) == len(simuhome) == 3
        assert all("living room" in p and "brightness" in p for p in ours)
        assert all("garage" in p and "curtain" in p for p in simuhome)
        assert verdict["homebench"]["em"] and verdict["homebench"]["judged"]


# -- statistics and report ----------------------------------------------------------------------------------


def evaluated_rows(home):
    """Rows and evaluations for a small run: a strict pass, a rescued turn-off, a failed
    infeasible episode scored without a judge, a stopped one."""
    cases = []
    ep = make_episode(home, "normal", "'''kitchen.light.set_brightness(20)'''",
                      "Set the brightness of the light to 20 in the kitchen.", "a")
    cases.append((ep, record_for(ep, home, calls=[("kitchen_light", "set_brightness", 20)],
                                 actions=[act("kitchen", "kitchen_light", "brightness", 20)]), ("A",) * 3))
    ep = make_episode(home, "normal", "'''kitchen.light.set_brightness(0)'''",
                      "Turn off the light in the kitchen.", "b")
    cases.append((ep, record_for(ep, home, calls=[("kitchen_light", "turn_off", None)],
                                 actions=[act("kitchen", "kitchen_light", "onOff", False)]), ("A",) * 3))
    ep = make_episode(home, "unexist_device", "'''error_input'''",
                      "Close the curtains in the kitchen.", "c")
    cases.append((ep, record_for(ep, home, plans=[{"outcome": "impossible"}]), ()))
    ep = make_episode(home, "multi2_mix", "'''kitchen.light.set_brightness(20),error_input,'''",
                      "Set the brightness of the light to 20 in the kitchen, close the curtains "
                      "in the garage.", "d")
    cases.append((ep, record_for(ep, home, outcome="stopped"), ("A",) * 3))
    return [(record, score(record, ep, judges=judges)) for ep, record, judges in cases]


class TestStatsAndReport:
    """Official table, strict and extra rates, notes; SimuHome's rows unchanged."""

    def test_an_episode_row_carries_the_homebench_columns(self, home):
        """From the evaluation's block; a stopped episode (no block) has none of them."""
        rows = [stats.episode_metrics(r, e) for r, e in evaluated_rows(home)]
        assert rows[0]["bucket"] == "normal_single" and rows[0]["hb_em"] == 1
        assert (rows[1]["score"], rows[1]["score_extra"]) == (0, 1)
        assert rows[3]["score"] == -1 and "bucket" not in rows[3]

    def test_groups_show_both_rates_and_the_generic_ones(self, home):
        """`success` is the benchmark's reading, `success_rate_extra` the other."""
        rows = [stats.episode_metrics(r, e) for r, e in evaluated_rows(home)]
        group = stats.summarise("hb1", "feasible", [r for r in rows if r["type"] == "hb1"
                                                    and r["case"] == "feasible"])
        assert group["success_rate"] == 0.5 and group["success_rate_extra"] == 1.0
        assert group["n_rescued"] == 1 and group["feasible_rate"] == 0.5
        shown = stats.tables([group])["score"][0]
        assert shown["success"] == "50%" and shown["success +readings"] == "100%"

    def test_official_buckets_come_next_to_the_type_table(self, home, tmp_path):
        """The table exists for the five buckets and `all`, an empty bucket is blank, and the
        summary.md has it with the caveats; buckets.csv is written."""
        rows = [stats.episode_metrics(r, e) for r, e in evaluated_rows(home)]
        buckets = {b["bucket"]: b for b in stats.bucket_rows(rows)}
        assert list(buckets) == ["all", *hbm.BUCKETS]
        assert buckets["all"]["n"] == 3 and buckets["mix_multi"]["n"] == 0
        assert buckets["mix_multi"]["em"] is None and buckets["normal_single"]["n"] == 2
        assert buckets["normal_single"]["success_rate"] == 0.5
        assert buckets["normal_single"]["success_rate_extra"] == 1.0
        out = stats.write(tmp_path, rows, stats.aggregate(rows), "t")
        text = (out / "summary.md").read_text(encoding="utf-8")
        assert "Official HomeBench metrics (per bucket)" in text
        assert "re-implemented from the logic of code/eval.py" in text
        assert "success +readings" in text and (out / "buckets.csv").is_file()

    def test_the_no_judge_weakness_is_printed(self, home):
        """An infeasible episode scored without a judge says so in the notes."""
        rows = [stats.episode_metrics(r, e) for r, e in evaluated_rows(home)]
        notes = " ".join(stats.homebench_notes(rows))
        assert "scored without a judge (--no-judge)" in notes and "weaker" in notes
        judged = [stats.episode_metrics(r, e) for r, e in evaluated_rows(home)[:2]]
        assert "without a judge" not in " ".join(stats.homebench_notes(judged))

    def test_percent_requests_are_flagged_and_reported_apart(self, home):
        """"by 50 percent": the gold adds 50 points (10 -> 60), our planner scales (10 -> 15).
        The episode fails by the benchmark's reading, is flagged, and the notes give its
        success rate against the others' -- nothing is relaxed."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(60)'''",
                               "Increase the brightness of the light by 50 percent in the kitchen.",
                               "p")
        record = record_for(episode, home, calls=[("kitchen_light", "set_brightness", 15)],
                            actions=[act("kitchen", "kitchen_light", "brightness", 15)])
        verdict = score(record, episode)
        assert verdict["homebench"]["percent_relative"] is True
        assert verdict["score"] == 0 == verdict["score_extra"]
        rows = [stats.episode_metrics(r, e) for r, e in evaluated_rows(home)]
        rows.append(stats.episode_metrics(record, verdict))
        note = next(n for n in stats.homebench_notes(rows) if "'by N percent'" in n)
        assert "1 of 3 scored feasible / mixed episodes" in note
        assert "success 0% against 50% for the others" in note
        assert not any("'by N percent'" in n for n in stats.homebench_notes(rows[:3]))

    def test_simuhome_rows_and_tables_are_unchanged(self):
        """No HomeBench key, column, table or note appears for SimuHome records."""
        record = {"episode": "qt3_feasible_seed_1", "query_type": "qt3", "case": "feasible",
                  "outcome": "executed", "meter": {}, "plans": [], "structures": []}
        row = stats.episode_metrics(record, {"score": 1, "breakdown": {"checks": []}})
        assert not [k for k in row if k.startswith("hb_") or k in ("bucket", "score_extra")]
        group = stats.summarise("qt3", "feasible", [row])
        assert "success_rate_extra" not in group
        assert "success +readings" not in stats.tables([group])["score"][0]
        assert stats.bucket_rows([row]) == [] and stats.homebench_notes([row]) == []

    def test_report_has_the_official_rows_and_the_extra_rate(self, home):
        """`report.official` and the rate with the extra reading, per variant."""
        data = {"v": [{"record": r, "evaluation": e} for r, e in evaluated_rows(home)]}
        official = {row["bucket"]: row for row in report.official(data)}
        assert official["all"]["n"] == 3 and official["mix_multi"]["em"] is None
        assert official["normal_single"]["rate"] == 0.5 and official["normal_single"]["rate_extra"] == 1.0
        rates = {(r["type"], r["case"]): r for r in report.rates(data)}
        assert rates[("hb1", "feasible")]["rate"] == 0.5 and rates[("hb1", "feasible")]["rate_extra"] == 1.0
        assert report.official({"x": [{"record": {"query_type": "qt1", "case": "feasible"},
                                       "evaluation": {"score": 1}}]}) == []
