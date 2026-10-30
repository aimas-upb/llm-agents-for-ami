"""HomeBench scoring (P4): gold_log, the official metrics, the gold-token rule, the extra
reading, and the statistics and report built on them.

No LLM is called and no server is started. Records are built by hand from a small home
through the real converter (`convert.build_episode`), so an episode has exactly the shape
the bench scores; its final state comes from the real engine, and the verdict from the
real registry (`HomeBenchEvaluators`) through `evaluate.evaluate_record`. HomeBench has no
judge, so nothing of SimuHome's is needed. The groups, and what each pins:

* gold_log: accepted actions become gold tokens, refused ones do not, nothing is dropped
  silently;
* the official metrics on the edge cases of `code/eval.py` (order, duplicates, `fan only`,
  an empty bucket), a question counted as `error_input`, and gold-as-prediction giving EM
  1.0 in all five buckets -- on a hand-built set and, when ``../HomeBench`` is present, on
  all 17,366 rows through the reverse mapping (the self-consistency of D3 and D7);
* the rule: a gold call met by the exact action or by a value that held, extra actions not
  counted, each `error_input` met by a goal answered impossible or with a question,
  `mixed` all-or-nothing, the vacuum's operation calls;
* the `turn_off` extra reading: only when the request asks for it, per clause, and only in
  `score_extra`;
* wiring: no judge for HomeBench, infrastructure failures before any rule;
* stats / report: official table, the strict and extra rates, the notes, SimuHome's rows
  unchanged.
"""

from collections import Counter
from pathlib import Path

import pytest

from ami_agents.environment.integration.HomeBench.engine import HomeBenchEngine
from ami_agents.environment.integration.HomeBench.mappings import load_mappings
from tests.homebench import convert, scoring
from tests.simuhome.bench import evaluate, gold_log, harness, homebench_metrics as hbm, report, stats
from tests.simuhome.bench.suites import get_suite
from tests.simuhome.bench.suites.homebench import HomeBenchEvaluators

ROOT = Path(__file__).resolve().parents[2]
HOMEBENCH_CHECKOUT = ROOT.parent / "HomeBench"
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


# -- the verdict ---------------------------------------------------------------------------------


def score(record, episode):
    """The verdict `evaluate_record` gives with the real HomeBench registry and no judge."""
    return evaluate.evaluate_record(record, episode, HomeBenchEvaluators(), [], None)


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

    def test_a_goal_asked_about_is_an_error_input(self):
        """Impossible entries and questions are both `error_input` tokens: either way the
        request was reported to the user instead of executed."""
        record = {"plans": [{"outcome": "impossible"}, {"outcome": "clarify"},
                            {"outcome": "plan"}],
                  "structures": [{"clarification": "which room?"}, {}]}
        assert hbm.refused_goals(record) == {"impossible": 1, "clarify": 2}
        assert hbm.prediction(["a"], record) == ["a"] + ["error_input"] * 3

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
    the number of `error_input` parts. Every gold call has an action (the vacuum's
    start / stop / pause / charge included), so a call without one raises."""
    actions, refusals = [], 0
    for call in convert.parse_gold(gold_text, multi):
        # 1. An `error_input` part is a refusal, not an action.
        if call is None:
            refusals += 1
            continue
        # 2. Any other part is the action its call stands for.
        name, value = MAPPINGS.action_for_command(call.device, call.command, call.arg)
        actions.append(act(call.room or "household", call.device_id, name, value))
    return actions, refusals


class TestGoldAsPrediction:
    """A record that did exactly what the gold says scores EM 1.0 (`error_input` for each
    part answered impossible) and passes the rule."""

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
            # 1. The record that did exactly the gold.
            episode = make_episode(home, homebench_type, gold, query)
            actions, refusals = gold_actions(
                gold, convert.multi_parts(homebench_type) is not None)
            record = record_for(episode, home, actions=actions,
                                plans=[{"outcome": "impossible"}] * refusals)
            # 2. EM 1.0 on its block, and the rule passes it.
            verdict = score(record, episode)
            block = verdict["homebench"]
            assert block["em"] and verdict["score"] == 1
            assert Counter(block["predicted_tokens"]) == Counter(block["gold_tokens"])
            rows.append({"bucket": block["bucket"], **hbm.compare(block["predicted_tokens"],
                                                                  block["gold_tokens"])})
        # 3. Every bucket has its row, all at 1.0.
        buckets = hbm.by_bucket(rows)
        assert [b for b in hbm.BUCKETS if buckets[b]["n"] == 1] == list(hbm.BUCKETS)
        assert all(buckets[b]["em"] == 1.0 and buckets[b]["f1"] == 1.0
                   for b in (hbm.ALL, *hbm.BUCKETS))

    @needs_homebench
    def test_all_17366_gold_rows_round_trip_through_the_mapping(self):
        """Every gold call becomes an action request and back to the same token: EM 1.0 in
        all five buckets, the vacuum's operation calls included."""
        rows = convert.read_jsonl(HOMEBENCH_CHECKOUT / "dataset" / "test_data.jsonl")
        assert len(rows) == 17366
        scored = []
        for row in rows:
            # 1. The gold as actions and refusals, then back as the predicted tokens.
            multi = convert.multi_parts(row["type"]) is not None
            actions, refusals = gold_actions(row["output"], multi)
            predicted = hbm.prediction(gold_log.gold_log(actions).tokens,
                                       {"plans": [{"outcome": "impossible"}] * refusals})
            scored.append({"bucket": hbm.bucket_of(row["type"]),
                           **hbm.compare(predicted, hbm.gold_tokens(row["output"]))})
        # 2. Every row reproduced, in every bucket.
        buckets = hbm.by_bucket(scored)
        assert all(buckets[b]["em"] == 1.0 and buckets[b]["precision"] == 1.0
                   and buckets[b]["recall"] == 1.0 for b in (hbm.ALL, *hbm.BUCKETS))
        assert {b: buckets[b]["n"] for b in hbm.BUCKETS} == {
            "normal_single": 6187, "unexist_single": 6765, "normal_multi": 245,
            "mix_multi": 4072, "error_multi": 97}


# -- the rule: gold calls -------------------------------------------------------------------------------


class TestGoldCalls:
    """A gold call is met by the exact action, or by a value that held; nothing else counts."""

    GOLD = "'''kitchen.light.set_brightness(20)'''"
    QUERY = "Set the brightness of the light to 20 in the kitchen."

    def episode(self, home):
        """The single command: the kitchen light (brightness 10) to 20."""
        return make_episode(home, "normal", self.GOLD, self.QUERY)

    def test_the_exact_action_passes(self, home):
        """The accepted `brightness 20` is the gold token itself."""
        # 1. The record that sent exactly the gold call.
        episode = self.episode(home)
        record = record_for(episode, home, calls=[("kitchen_light", "set_brightness", 20)],
                            actions=[act("kitchen", "kitchen_light", "brightness", 20)])
        # 2. Passed by both scores, no judge, the check met by the action.
        verdict = score(record, episode)
        assert verdict["score"] == 1 == verdict["score_extra"]
        assert verdict["error_type"] is None and "judge" not in verdict
        check = verdict["breakdown"]["checks"][0]
        assert check["class"] == "ok" and check["matched"] == "action"
        assert check["expected"] == "kitchen.light.set_brightness(20)"
        assert verdict["homebench"]["em"] and verdict["homebench"]["bucket"] == "normal_single"

    def test_another_value_is_wrong_and_no_action_is_missing(self, home):
        """`brightness 30` names the call with another value; nothing sent is missing.
        Both fail as a goal failure."""
        # 1. The same call with 30, and no call at all.
        episode = self.episode(home)
        wrong = score(record_for(episode, home,
                                 actions=[act("kitchen", "kitchen_light", "brightness", 30)]),
                      episode)
        missing = score(record_for(episode, home), episode)
        # 2. Both fail; the check says which way.
        assert wrong["score"] == 0 == missing["score"]
        assert wrong["error_type"] == "Goal Failure"
        assert wrong["breakdown"]["checks"][0]["class"] == "wrong_value"
        assert wrong["breakdown"]["checks"][0]["values"] == ["kitchen.light.set_brightness(30)"]
        assert missing["breakdown"]["checks"][0]["class"] == "missing"

    def test_a_request_the_home_refused_is_no_action(self, home):
        """A 4xx answer changed nothing: it does not meet the gold call."""
        episode = self.episode(home)
        record = record_for(episode, home, actions=[
            act("kitchen", "kitchen_light", "brightness", 20, success=False)])
        assert score(record, episode)["breakdown"]["checks"][0]["class"] == "missing"

    def test_extra_actions_are_not_counted(self, home):
        """Switching the light on first and touching another room's light do not fail the
        episode; the official EM, which counts every token, does."""
        # 1. The gold call among two actions the gold does not have.
        episode = self.episode(home)
        record = record_for(episode, home, actions=[
            act("kitchen", "kitchen_light", "onOff", True),
            act("kitchen", "kitchen_light", "brightness", 20),
            act("garage", "garage_light", "brightness", 5)])
        # 2. The rule passes it; the official EM does not.
        verdict = score(record, episode)
        assert verdict["score"] == 1
        assert not verdict["homebench"]["em"]

    def test_one_action_meets_a_call_the_request_repeats(self, home):
        """The gold repeats the call; the device is at 20 after one action, which is enough."""
        episode = make_episode(
            home, "multi2_normal",
            "'''kitchen.light.set_brightness(20),kitchen.light.set_brightness(20),'''",
            "Set the brightness of the light to 20 in the kitchen, and set the light to 20 in "
            "the kitchen.")
        record = record_for(episode, home, calls=[("kitchen_light", "set_brightness", 20)],
                            actions=[act("kitchen", "kitchen_light", "brightness", 20)])
        assert score(record, episode)["score"] == 1

    def test_the_rooms_need_not_be_looked_up(self, home):
        """SimuHome's required `get_room_devices` reads are not part of the rule."""
        episode = self.episode(home)
        record = record_for(episode, home, actions=[
            act("kitchen", "kitchen_light", "brightness", 20)])
        record["tools_invoked"] = []
        assert score(record, episode)["score"] == 1

    def test_a_value_that_already_held_needs_no_action(self, home):
        """The kitchen light is on before and after: "turn it on" is met with no action."""
        episode = make_episode(home, "normal", "'''kitchen.light.turn_on()'''",
                               "Turn on the light in the kitchen.")
        verdict = score(record_for(episode, home), episode)
        assert verdict["score"] == 1
        assert verdict["breakdown"]["checks"][0]["matched"] == "held"

    def test_held_needs_the_value_before_and_after(self, home):
        """The air conditioner is off: "turn it on" with no action is missing. The kitchen
        light turned off during the run no longer holds "on" at the end."""
        # 1. Not on before: nothing held.
        episode = make_episode(home, "normal", "'''living_room.air_conditioner.turn_on()'''",
                               "Turn on the air conditioner in the living room.")
        assert score(record_for(episode, home), episode)["score"] == 0
        # 2. On before, off after: nothing held either.
        episode = make_episode(home, "normal", "'''kitchen.light.turn_on()'''",
                               "Turn on the light in the kitchen.")
        record = record_for(episode, home, calls=[("kitchen_light", "turn_off", None)],
                            actions=[act("kitchen", "kitchen_light", "onOff", False)])
        assert score(record, episode)["breakdown"]["checks"][0]["class"] == "missing"

    def test_the_vacuums_operation_calls_are_met_by_its_state_action(self, home):
        """`vacuum_robot.pause()` is the vacuum's `state` action with `paused`."""
        episode = make_episode(home, "normal", "'''vacuum_robot.pause()'''",
                               "Pause the vacuum robot.")
        record = record_for(episode, home, calls=[("household_vacuum_robot", "pause", None)],
                            actions=[act("household", "household_vacuum_robot", "state",
                                         "paused")])
        verdict = score(record, episode)
        assert verdict["score"] == 1
        assert verdict["homebench"]["predicted_tokens"] == ["vacuum_robot.pause()"]


# -- the rule: parts that cannot be done --------------------------------------------------------------


class TestRefusals:
    """An `error_input` is met by a goal answered impossible or with a question."""

    GOLD, QUERY = "'''error_input'''", "Close the curtains in the kitchen."

    def episode(self, home):
        """The kitchen has blinds and no curtain."""
        return make_episode(home, "unexist_device", self.GOLD, self.QUERY)

    def test_an_impossible_answer_meets_it(self, home):
        """One goal answered impossible: the part is met, and the refusal check says so."""
        # 1. The record that answered the one part impossible and did nothing.
        episode = self.episode(home)
        verdict = score(record_for(episode, home, plans=[{"outcome": "impossible"}],
                                   outcome="impossible"), episode)
        # 2. Passed; no gold call to check, one error_input met.
        assert verdict["score"] == 1 == verdict["score_extra"]
        assert verdict["execution_check"] == {"score": 1, "check": "refusals",
                                              "error_inputs": 1, "refusals": 1}
        assert verdict["breakdown"]["total"] == 0
        assert verdict["homebench"]["parts"] == {"feasible": 0, "feasible_ok": 0,
                                                 "infeasible": 1, "infeasible_ok": 1}

    def test_a_question_meets_it_too(self, home):
        """A goal stopped at structuring by a question is an `error_input` as well, for the
        rule and for the official metrics."""
        episode = self.episode(home)
        record = record_for(episode, home, structures=[{"clarification": "which room?"}],
                            outcome="clarify")
        verdict = score(record, episode)
        assert verdict["score"] == 1
        assert verdict["homebench"]["clarified"] == 1 and verdict["homebench"]["em"]

    def test_no_refusal_fails(self, home):
        """Nothing done and nothing said: the part is not reported."""
        episode = self.episode(home)
        verdict = score(record_for(episode, home), episode)
        assert verdict["score"] == 0 and verdict["error_type"] == "Not Reported Infeasible"
        assert verdict["execution_check"]["score"] == 0

    def test_a_substitute_is_extra_but_its_refusal_is_missing(self, home):
        """Closing the kitchen blinds instead is not counted as such; the episode fails
        because nobody said the curtain is not there."""
        episode = self.episode(home)
        stand_in = [act("kitchen", "kitchen_blinds", "openClose", "closed")]
        assert score(record_for(episode, home, actions=stand_in), episode)["score"] == 0
        both = record_for(episode, home, actions=stand_in, plans=[{"outcome": "impossible"}])
        assert score(both, episode)["score"] == 1

    def test_each_part_needs_its_own_refusal(self, home):
        """Two parts that cannot be done need two goals answered impossible or asked about."""
        # 1. Two parts that cannot be done.
        episode = make_episode(home, "multi2_unexist_device", "'''error_input,error_input,'''",
                               "Close the curtains in the garage, close the curtains in the "
                               "study room.")
        # 2. One refusal is short; an impossible answer and a question are enough.
        one = record_for(episode, home, plans=[{"outcome": "impossible"}])
        two = record_for(episode, home, plans=[{"outcome": "impossible"}],
                         structures=[{"clarification": "which curtains?"}])
        assert score(one, episode)["score"] == 0
        assert score(two, episode)["score"] == 1


class TestMixed:
    """Both kinds of part in one request: every gold token must be met."""

    GOLD = "'''kitchen.light.set_brightness(20),error_input,'''"
    QUERY = ("Set the brightness of the light to 20 in the kitchen, close the curtains in the "
             "garage.")

    def test_all_parts_met_passes_and_one_missing_part_fails(self, home):
        """The action and the refusal pass; either one alone fails, with its own error."""
        # 1. The action and the refusal the two parts need.
        episode = make_episode(home, "multi2_mix", self.GOLD, self.QUERY)
        action = [act("kitchen", "kitchen_light", "brightness", 20)]
        refusal = [{"outcome": "plan"}, {"outcome": "impossible"}]
        # 2. Both: a pass; either one alone: a fail named after the part that is missing.
        both = score(record_for(episode, home, actions=action, plans=refusal), episode)
        assert both["score"] == 1 and both["homebench"]["em"]
        no_refusal = score(record_for(episode, home, actions=action), episode)
        assert no_refusal["error_type"] == "Not Reported Infeasible"
        no_action = score(record_for(episode, home, plans=refusal), episode)
        assert no_action["error_type"] == "Goal Failure"
        assert no_action["homebench"]["parts"] == {"feasible": 1, "feasible_ok": 0,
                                                   "infeasible": 1, "infeasible_ok": 1}


# -- the extra reading ---------------------------------------------------------------------------------


class TestTurnOffReading:
    """A gold zero level for a request that asks to turn the device off is also met by
    that device's `turn_off()` -- in `score_extra` only (our criterion, not HomeBench's)."""

    def off_record(self, episode, home, device="kitchen_light", room="kitchen"):
        """The agent turned the device off instead of writing the level."""
        return record_for(episode, home, calls=[(device, "turn_off", None)],
                          actions=[act(room, device, "onOff", False)])

    def test_accepted_when_the_request_says_turn_off(self, home):
        """`score` keeps the gold's reading, `score_extra` has the other, and `matched`
        says how the check was met."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(0)'''",
                               "Turn off the light in the kitchen.")
        verdict = score(self.off_record(episode, home), episode)
        check = verdict["breakdown"]["checks"][0]
        assert verdict["score"] == 0 and verdict["score_extra"] == 1
        assert check["class"] == "ok" and check["matched"] == "turn_off"
        assert verdict["homebench"]["parts"]["feasible_ok"] == 0     # the strict count

    def test_the_gold_value_itself_still_matches_as_an_action(self, home):
        """Writing the zero level is the gold call: both scores pass."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(0)'''",
                               "Turn off the light in the kitchen.")
        record = record_for(episode, home, actions=[
            act("kitchen", "kitchen_light", "brightness", 0)])
        verdict = score(record, episode)
        assert verdict["score"] == 1 == verdict["score_extra"]
        assert verdict["breakdown"]["checks"][0]["matched"] == "action"

    @pytest.mark.parametrize("query", [
        "Set the brightness of the light to the lowest level in the kitchen.",
        "Dim the light in the kitchen to zero."])
    def test_not_accepted_when_the_request_does_not_ask_for_it(self, home, query):
        """Turning it off does not meet "set the level"; the strict reading stands."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(0)'''", query)
        verdict = score(self.off_record(episode, home), episode)
        assert verdict["score"] == 0 == verdict["score_extra"]
        assert verdict["breakdown"]["checks"][0]["class"] != "ok"

    def test_switch_off_counts_and_other_levels_never_do(self, home):
        """"switch off" is asked too; a gold of 20 has no turn-off token even for a
        turn-off query (only a zero level is relaxed)."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(0)'''",
                               "Switch off the light in the kitchen.")
        assert score(self.off_record(episode, home), episode)["score_extra"] == 1
        other = make_episode(home, "normal", "'''kitchen.light.set_brightness(20)'''",
                             "Turn off the light in the kitchen and set the brightness to 20.")
        token, call, goal = scoring.gold_parts(other)[0]
        assert scoring.turn_off_token(other, token, call, goal) is None

    def test_in_a_multi_request_only_the_parts_own_clause_counts(self, home):
        """"Turn off the humidifier" does not turn "set the light brightness to 0" into a
        turn-off: the light part keeps the strict reading, the humidifier part has both."""
        episode = make_episode(
            home, "multi2_normal",
            "'''garage.humidifier.set_intensity(0),kitchen.light.set_brightness(0),'''",
            "Turn off the humidifier in the garage, set the brightness of the light to 0 "
            "in the kitchen.")
        # 1. Only the humidifier's clause asks for a turn-off.
        humidifier, light = episode["eval"]["goals"]
        assert scoring.asks_turn_off(episode, humidifier) is True
        assert scoring.asks_turn_off(episode, light) is False
        # 2. Turning the humidifier off meets its part by the reading; the light's is missing.
        record = self.off_record(episode, home, "garage_humidifier", "garage")
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
        """`score_extra` is 1 only when the reading is all that differs: the refusal is
        still needed."""
        episode = make_episode(
            home, "multi2_mix", "'''kitchen.light.set_brightness(0),error_input,'''",
            "Turn off the light in the kitchen, close the curtains in the garage.")
        # 1. Turned off, and the other part refused: rescued by the reading alone.
        record = self.off_record(episode, home)
        record["plans"] = [{"outcome": "plan"}, {"outcome": "impossible"}]
        verdict = score(record, episode)
        assert (verdict["score"], verdict["score_extra"]) == (0, 1)
        # 2. Without the refusal, the reading cannot rescue it.
        assert score(self.off_record(episode, home), episode)["score_extra"] == 0

    def test_matched_is_stored_for_every_ok_check(self):
        """SimuHome's checks carry `matched` too (a percent reading names itself)."""
        assert evaluate._ok(None) == {"class": "ok", "matched": "simuhome"}
        assert evaluate._ok("simuhome") == {"class": "ok", "matched": "simuhome"}
        assert evaluate._ok("percent_of_range") == {"class": "ok", "matched": "percent_of_range",
                                                    "reading": "percent_of_range"}


# -- infrastructure, wiring ------------------------------------------------------------------------------


class TestWiring:
    """What `evaluate_record`, the registry, the suite and the harness do around the rule."""

    def test_infrastructure_and_stopped_runs_have_no_block(self, home):
        """-1 before any rule: nothing for the official metrics to read."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(20)'''",
                               "Set the brightness of the light to 20 in the kitchen.")
        verdict = score(record_for(episode, home, outcome="stopped"), episode)
        assert verdict["score"] == -1 and "homebench" not in verdict
        record = record_for(episode, home)
        record["error"] = "SimuHomeError: 502"
        assert score(record, episode)["error_type"] == "Infrastructure"

    def test_a_plain_registry_adds_no_block(self, home):
        """A registry without `describe` (a plain dict) still scores, without the block."""
        episode = make_episode(home, "normal", "'''kitchen.light.set_brightness(20)'''",
                               "Set the brightness of the light to 20 in the kitchen.")
        plain = {("hb1", "feasible"): scoring}
        record = record_for(episode, home, actions=[
            act("kitchen", "kitchen_light", "brightness", 20)])
        verdict = evaluate.evaluate_record(record, episode, plain, [], None)
        assert verdict["score"] == 1 and "homebench" not in verdict

    def test_an_episode_without_gold_is_refused_clearly(self, home):
        """The converter always writes `meta.gold`; without it there is nothing to meet."""
        episode = make_episode(home, "unexist_device", "'''error_input'''",
                               "Close the curtains in the kitchen.")
        del episode["meta"]["gold"]
        with pytest.raises(ValueError, match="meta.gold"):
            scoring.evaluate_episode({}, episode)

    def test_the_registry_is_the_rule_for_every_type_and_case(self):
        """Every hb type and case is the scoring module; anything else is a KeyError."""
        registry = HomeBenchEvaluators()
        assert all(registry[(f"hb{n}", case)] is scoring for n in (1, 5, 10)
                   for case in ("feasible", "infeasible", "mixed"))
        for key in (("qt3", "feasible"), ("hb3", "bogus")):
            with pytest.raises(KeyError):
                registry[key]

    def test_no_judge_for_homebench(self, tmp_path):
        """The suite has no judge: no judge calls estimated, no `--no-judge` passed on, and
        the CLI's judged cases stay SimuHome's."""
        # 1. The suite says so, and estimates no judge call.
        suite = get_suite("homebench")
        assert suite.judge is False and get_suite("simuhome").judge is True
        assert suite.estimate("hb3_mixed_seed_2") == (4, 4, 0)
        # 2. The harness neither passes --no-judge on nor counts judge calls.
        evaluate_command = harness.commands(tmp_path / "c.yaml", tmp_path, tmp_path, {},
                                            False, tmp_path, suite)[1]
        assert "--no-judge" not in evaluate_command
        assert "judge" not in harness.estimate(["hb3_mixed_seed_2"], tmp_path, False, suite)
        assert evaluate.JUDGED_CASES == ("infeasible", "mixed")


# -- statistics and report ----------------------------------------------------------------------------------


def evaluated_rows(home):
    """Rows and evaluations for a small run: a strict pass, a rescued turn-off, an
    infeasible episode answered impossible, a stopped one."""
    cases = []
    ep = make_episode(home, "normal", "'''kitchen.light.set_brightness(20)'''",
                      "Set the brightness of the light to 20 in the kitchen.", "a")
    cases.append((ep, record_for(ep, home, calls=[("kitchen_light", "set_brightness", 20)],
                                 actions=[act("kitchen", "kitchen_light", "brightness", 20)])))
    ep = make_episode(home, "normal", "'''kitchen.light.set_brightness(0)'''",
                      "Turn off the light in the kitchen.", "b")
    cases.append((ep, record_for(ep, home, calls=[("kitchen_light", "turn_off", None)],
                                 actions=[act("kitchen", "kitchen_light", "onOff", False)])))
    ep = make_episode(home, "unexist_device", "'''error_input'''",
                      "Close the curtains in the kitchen.", "c")
    cases.append((ep, record_for(ep, home, plans=[{"outcome": "impossible"}])))
    ep = make_episode(home, "multi2_mix", "'''kitchen.light.set_brightness(20),error_input,'''",
                      "Set the brightness of the light to 20 in the kitchen, close the curtains "
                      "in the garage.", "d")
    cases.append((ep, record_for(ep, home, outcome="stopped")))
    return [(record, score(record, ep)) for ep, record in cases]


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

    def test_no_judge_column_or_note(self, home):
        """HomeBench has no judge: no judge columns in the rows, none in the notes, and the
        refusal check stands in the "exec check" rate."""
        rows = [stats.episode_metrics(r, e) for r, e in evaluated_rows(home)]
        assert not [k for row in rows for k in row if k in ("hb_judged", "hb_unreachable")]
        assert "judge" not in " ".join(stats.homebench_notes(rows))
        group = stats.summarise("hb1", "infeasible", [r for r in rows if r["case"] == "infeasible"])
        assert group["execution_check_rate"] == 1.0 and group["judge_rate"] is None

    def test_percent_requests_are_flagged_and_reported_apart(self, home):
        """"by 50 percent": the gold adds 50 points (10 -> 60); a plan that scales instead
        (10 -> 15) fails. The episode is flagged, and the notes give its success rate
        against the others' -- nothing is relaxed."""
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
        assert "success 0% against 50% for the others." in note
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
