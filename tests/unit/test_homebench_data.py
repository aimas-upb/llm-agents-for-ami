"""HomeBench data layer (P1): gold grammar, converter, engine. No LLM, no servers.

Two kinds of test, as in ``test_simuhome_bench.py``:

* synthetic tests (engine semantics, grammar errors, goal building, the clause
  locator, the loader) run everywhere;
* the P1 *gate* tests read the real dataset from ``../HomeBench`` and are skipped
  when that checkout is absent: the grammar parses all 17,366 rows, every gold call
  applies to its home through the engine (heating range corrected), the
  ``homebench_type`` counts match the plan (26 types; 6,187 / 6,765 / 245 / 4,072 /
  97 by official bucket), and episode names are unique and stable across two
  conversions.
"""

import copy
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from ami_agents.environment.integration.HomeBench import engine
from ami_agents.environment.integration.HomeBench.engine import (
    EngineError, HomeBenchEngine,
)
from tests.homebench import convert, locate

ROOT = Path(__file__).resolve().parents[2]
HOMEBENCH = ROOT.parent / "HomeBench"
HAVE_DATA = (HOMEBENCH / convert.TEST_DATA).is_file()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="../HomeBench absent")


# -- a small synthetic home --------------------------------------------------------------------


def _home():
    """A two-room home: a light (brightness, color), a heater (16-30 after the
    correction), a curtain, a media player, a trash can and a vacuum robot."""
    raw = {
        "master_bedroom": {
            "light": {"state": "off", "attributes": {
                "brightness": {"value": 50, "lowest": "0", "highest": "100"},
                "color": {"value": [1, 2, 3]}}},
            "heating": {"state": "on", "attributes": {
                "temperature": {"value": 25, "lowest": "30", "highest": "100"},
                "mode": {"value": "heat", "options": ["heat", "fan_only"]}}},
            "curtain": {"state": "closed", "attributes": {
                " degree": {"value": 0, "lowest": 0, "highest": "100"}}},
            "media_player": {"state": "stopped", "attributes": {
                "volume": {"value": 5, "lowest": "0", "highest": "100"}}},
            "trash": {"state": "full", "attributes": {}},
        },
        "kitchen": {"light": {"state": "on", "attributes": {}}},
        "VacuumRobot": {"state": "on", "attributes": {
            "battery": {"value": 100, "lowest": 0, "highest": 100},
            "mode": {"value": "auto", "options": ["auto", "strong", "sleep"]}}},
    }
    for room in ("master_bedroom", "kitchen"):
        raw[room]["room_name"] = room
    return convert.convert_home({"home_id": 1, "home_status": raw})


# -- engine: gold-consistent semantics -----------------------------------------------------------


class TestEngine:
    def test_home_file_shape_and_degree_key(self):
        home = _home()
        assert set(home["rooms"]) == {"master_bedroom", "kitchen", "household"}
        curtain = next(d for d in home["rooms"]["master_bedroom"]["devices"]
                       if d["device_type"] == "curtain")
        assert curtain["device_id"] == "master_bedroom_curtain"
        assert "degree" in curtain["attributes"] and " degree" not in curtain["attributes"]
        assert curtain["constraints"]["degree"] == {"lowest": 0, "highest": 100}
        vacuum = home["rooms"]["household"]["devices"][0]
        assert vacuum["device_id"] == "household_vacuum_robot"
        assert vacuum["attributes"]["cleaning_area"] is None

    def test_no_preconditions(self):
        """The gold sets attributes on devices that are off, closed or stopped."""
        machine = HomeBenchEngine(_home())
        assert machine.apply("master_bedroom_light", "set_brightness", 20).status == "applied"
        machine.apply("master_bedroom_curtain", "set_degree", 40)   # closed curtain
        machine.apply("master_bedroom_media_player", "set_volume", 9)  # stopped player
        machine.apply("master_bedroom_heating", "turn_on")             # already on
        state = machine.device("master_bedroom_light")["attributes"]
        assert state["brightness"] == 20 and state["state"] == "off"

    def test_state_commands(self):
        machine = HomeBenchEngine(_home())
        machine.apply("master_bedroom_light", "turn_on")
        machine.apply("master_bedroom_curtain", "open")
        machine.apply("master_bedroom_media_player", "play")
        machine.apply("master_bedroom_trash", "pack")
        attrs = lambda d: machine.device(d)["attributes"]["state"]  # noqa: E731
        assert [attrs("master_bedroom_light"), attrs("master_bedroom_curtain"),
                attrs("master_bedroom_media_player"), attrs("master_bedroom_trash")] \
            == ["on", "open", "playing", "empty"]

    def test_heating_range_is_corrected_to_16_30(self):
        machine = HomeBenchEngine(_home())
        machine.apply("master_bedroom_heating", "set_temperature", 16)
        machine.apply("master_bedroom_heating", "set_temperature", 30)
        for bad in (15, 31, 60):
            with pytest.raises(EngineError):
                machine.apply("master_bedroom_heating", "set_temperature", bad)

    @pytest.mark.parametrize("device,command,arg", [
        ("master_bedroom_light", "set_brightness", 101),
        ("master_bedroom_light", "set_brightness", -1),
        ("master_bedroom_light", "set_brightness", "20"),
        ("master_bedroom_light", "set_brightness", True),
        ("master_bedroom_heating", "set_mode", "cool"),
        ("master_bedroom_light", "turn_on", 1),
        ("master_bedroom_light", "set_hue", 1),
        ("kitchen_light", "set_brightness", 10),       # no brightness on this light
        ("master_bedroom_trash", "turn_on", None),     # not offered by a trash can
        ("nowhere_light", "turn_on", None),
        ("household_vacuum_robot", "set_cleaning_area", "attic"),
        ("master_bedroom_media_player", "set_song", "x"),   # not modelled
    ])
    def test_refusals(self, device, command, arg):
        with pytest.raises(EngineError):
            HomeBenchEngine(_home()).apply(device, command, arg)

    def test_the_vacuum_operation_commands_set_its_state(self):
        """The gold's start / pause / stop / charge are offered and write ``state``."""
        machine = HomeBenchEngine(_home())
        # 1. Offered, before the two setters (state commands come first).
        assert machine.exposed_commands("household_vacuum_robot") == [
            "charge", "pause", "start", "stop", "set_cleaning_area", "set_mode"]
        # 2. Each one writes the state it names.
        for command, state in (("start", "cleaning"), ("pause", "paused"),
                               ("stop", "stopped"), ("charge", "charging")):
            result = machine.apply("household_vacuum_robot", command)
            assert (result.status, result.attribute, result.value) == (
                "applied", "state", state)
        machine.apply("household_vacuum_robot", "set_cleaning_area", "masterbedroom")

    def test_exposed_commands_follow_the_attributes_present(self):
        machine = HomeBenchEngine(_home())
        assert machine.exposed_commands("kitchen_light") == ["turn_off", "turn_on"]
        assert "set_color" in machine.exposed_commands("master_bedroom_light")
        machine.apply("master_bedroom_light", "set_color", [9, 9, 9])

    def test_reset_restores_the_home_and_isolates_the_input(self):
        home = _home()
        machine = HomeBenchEngine(home)
        before = machine.home_state()
        machine.apply("master_bedroom_light", "set_brightness", 7)
        assert machine.home_state() != before
        assert machine.reset() == before
        assert home["rooms"]["master_bedroom"]["devices"][0]["attributes"][
            "brightness"] == 50                      # the caller's dict is untouched
        assert before["current_tick"] == 0 and before["current_time"] == engine.BASE_TIME


# -- grammar ---------------------------------------------------------------------------------------


class TestGoldGrammar:
    def test_single_and_multi(self):
        single = convert.parse_gold("''' bathroom.light.set_brightness(20)'''", False)
        assert single == [convert.GoldCall("bathroom", "light", "set_brightness", 20)]
        multi = convert.parse_gold(
            "'''error_input,vacuum_robot.set_mode(auto),kitchen.light.turn_on(),'''", True)
        assert multi[0] is None
        assert multi[1] == convert.GoldCall(None, "vacuum_robot", "set_mode", "auto")
        assert multi[1].device_id == "household_vacuum_robot"
        assert multi[2].arg is None

    @pytest.mark.parametrize("output,multi", [
        ("error_input", False),                                    # no frame
        ("'''a.b.c(1),'''", False),                                # comma in a single
        ("'''error_input,kitchen.light.turn_on()'''", True),       # no trailing comma
        ("'''kitchen.light.set_color((1,2,3))'''", False),         # tuple argument
        ("'''light.turn_on()'''", False),                          # roomless non-vacuum
        ("'''kitchen.vacuum_robot.start()'''", False),             # vacuum with a room
        ("'''kitchen.light.set_brightness(a b)'''", False),
    ])
    def test_violations_raise(self, output, multi):
        with pytest.raises(convert.ConversionError):
            convert.parse_gold(output, multi)

    def test_types_cases_and_buckets(self):
        assert convert.episode_type("normal") == "hb1"
        assert convert.episode_type("multi7_mix") == "hb7"
        assert [convert.official_bucket(t) for t in (
            "normal", "unexist_device", "unexist_attribute", "multi3_normal",
            "multi4_mix", "multi2_unexist_device", "multi3_unexist_attribute")] == [
            "normal_single", "unexist_single", "unexist_single", "normal_multi",
            "mix_multi", "error_multi", "error_multi"]
        with pytest.raises(convert.ConversionError):
            convert.official_bucket("multi1_mix")
        call = convert.GoldCall("a", "light", "turn_on", None)
        assert [convert.case_of(p) for p in ([call], [None], [call, None])] == [
            "feasible", "infeasible", "mixed"]


# -- clause locator ----------------------------------------------------------------------------------


class TestLocate:
    @pytest.mark.parametrize("clause,expect", [
        ("Set the brightness of the light in the ding room to 70.",
         ("ding_room", "light", "brightness")),
        ("Close the garage door in the garage", ("garage", "garage_door", None)),
        ("set the fan speed of the air purifiers to high in the study room",
         ("study_room", "air_purifiers", "fan_speed")),
        ("Set the intensity of the dehumidifier in the storage room to 30.",
         ("store_room", "dehumidifiers", "intensity")),
        ("Set the water heater temperature to 55 degrees in the kitchen",
         ("kitchen", "water_heater", "temperature")),
        ("Set the brightness to 100 in the balcony", ("balcony", None, "brightness")),
        ("Set the cleaning mode of the vacuum robot to strong",
         (None, "vacuum_robot", "mode")),
    ])
    def test_locate(self, clause, expect):
        found = locate.locate(clause)
        assert (found.room, found.device, found.attribute) == expect

    def test_split_merges_room_prefix_and_continuations(self):
        query = ("In the corridor, increase the interval of the aromatherapy by 30 "
                 "seconds; turn off the lights in the store room and foyer, and set "
                 "the brightness to 20 in the kitchen.")
        assert locate.split_clauses(query) == [
            "In the corridor increase the interval of the aromatherapy by 30 seconds",
            "turn off the lights in the store room, foyer",
            "set the brightness to 20 in the kitchen"]

    def test_align_positional_elimination_and_unresolved(self):
        clauses = ["turn on the light in the kitchen",
                   "set the brightness of the light in the foyer to 10"]
        kitchen = ("kitchen", "light")
        targets, method = locate.align(clauses, [kitchen, None])
        assert method == "positional" and targets[1].room == "foyer"
        # Clauses in another order than the gold: the feasible clause is removed
        # and the remaining one is handed to the infeasible part.
        targets, method = locate.align(clauses[::-1], [kitchen, None])
        assert method == "elimination" and targets[1].room == "foyer"
        assert targets[0] == locate.Target("kitchen", "light", None)
        # An extra clause is ambiguous: refuse to guess.
        assert locate.align(["noise"] + clauses[::-1], [kitchen, None])[1] == "unresolved"
        # The gold names a feasible part no clause matches.
        assert locate.align(clauses, [("balcony", "fan"), None]) == (None, "unresolved")


# -- episode building --------------------------------------------------------------------------------


def _row(rid, htype, text, output, home_id=1):
    """A HomeBench-shaped test row."""
    return {"id": rid, "type": htype, "input": text, "output": output,
            "home_id": home_id}


class TestEpisodes:
    def test_feasible_goal_asserts_only_the_set_attribute(self):
        stats = Counter()
        built = convert.build_episode(
            _row("h1_one_1", "normal", "Set the brightness of the light to 20 in the "
                 "master bedroom.", "''' master_bedroom.light.set_brightness(20)'''"),
            _home(), stats)
        episode = built.episode
        assert (built.type, built.case) == ("hb1", "feasible")
        goal = episode["eval"]["goals"][0]
        assert goal["device_id"] == "master_bedroom_light"
        assert goal["asserts"] == [{"attribute": "brightness", "value": 20,
                                    "value_type": "int",
                                    "description": "set_brightness(20)"}]
        assert episode["eval"]["required_actions"] == [
            {"tool": "get_room_devices", "params": {"room_id": "master_bedroom"}}]
        assert episode["meta"]["gold"] == "''' master_bedroom.light.set_brightness(20)'''"
        assert episode["user_location"] is None and episode["home_id"] == 1

    def test_state_goal_and_vacuum_part(self):
        """Both parts assert the ``state`` their command writes, the vacuum's too."""
        stats = Counter()
        built = convert.build_episode(
            _row("h1_multi_1", "multi2_normal", "Turn on the light in the master "
                 "bedroom and start the vacuum robot.",
                 "'''master_bedroom.light.turn_on(),vacuum_robot.start(),'''"),
            _home(), stats)
        first, second = built.episode["eval"]["goals"]
        assert first["asserts"][0]["attribute"] == "state"
        assert first["asserts"][0]["value"] == "on"
        assert second["asserts"][0]["attribute"] == "state"
        assert second["asserts"][0]["value"] == "cleaning"
        assert second["asserts"][0]["description"] == "start()"
        assert "unreachable" not in second
        assert "unreachable_parts" not in built.episode["meta"]

    def test_infeasible_parts_carry_reason_and_target(self):
        stats = Counter()
        device = convert.build_episode(
            _row("h1_one_2", "unexist_device", "Turn on the fan in the kitchen.",
                 "'''error_input'''"), _home(), stats).episode
        goal = device["eval"]["goals"][0]
        assert goal == {"room_id": "kitchen", "device_id": "kitchen_fan",
                        "device_type": "fan", "feasibility": False,
                        "reason": "unexist_device", "resolved": True}
        attribute = convert.build_episode(
            _row("h1_one_3", "unexist_attribute", "Set the brightness of the light "
                 "in the kitchen to 30.", "'''error_input'''"), _home(), stats).episode
        goal = attribute["eval"]["goals"][0]
        assert goal["reason"] == "unexist_attribute"
        assert goal["attribute"] == "brightness" and goal["device_id"] == "kitchen_light"
        assert attribute["meta"]["case"] == "infeasible"
        assert stats["reason_label_disagrees"] == 0

    def test_unlocatable_part_is_kept_and_flagged(self):
        stats = Counter()
        goal = convert.build_episode(
            _row("h1_one_4", "unexist_device", "Do something.", "'''error_input'''"),
            _home(), stats).episode["eval"]["goals"][0]
        assert goal["resolved"] is False and goal["room_id"] is None
        assert goal["reason"] == "unexist_device"       # from the row label
        assert stats["parts_unresolved"] == 1

    def test_overwritten_assert_is_superseded_but_identical_repeats_stay(self):
        goals = [
            {"device_id": "d", "asserts": [{"attribute": "x", "value": 1}]},
            {"device_id": "d", "asserts": [{"attribute": "x", "value": 2}]},
            {"device_id": "d", "asserts": [{"attribute": "x", "value": 1}]},
            {"device_id": "e", "asserts": [{"attribute": "x", "value": 1}]},
        ]
        assert convert._mark_superseded(goals) == 1
        assert goals[0]["asserts"] and not goals[1]["asserts"]
        assert goals[1]["superseded_by"] == 2 and "superseded_by" not in goals[0]

    def test_part_count_must_match_the_type(self):
        with pytest.raises(convert.ConversionError):
            convert.build_episode(
                _row("h1_multi_9", "multi3_normal", "x",
                     "'''master_bedroom.light.turn_on(),'''"), _home(), Counter())

    def test_write_and_load_episode_merges_the_home(self, tmp_path):
        home = _home()
        built = convert.build_episode(
            _row("h1_one_1", "normal", "Turn on the light in the master bedroom.",
                 "''' master_bedroom.light.turn_on()'''"), home, Counter())
        built.episode["meta"]["seed"] = 1
        conversion = convert.Conversion(
            1, {1: home}, {"hb1_feasible_seed_1": built.episode}, {"rows": 1})
        convert.write_conversion(conversion, tmp_path)
        (tmp_path / "episodes" / "stale.json").write_text("{}")
        convert.write_conversion(conversion, tmp_path)         # removes the stale file
        assert sorted(p.name for p in (tmp_path / "episodes").iterdir()) == [
            "hb1_feasible_seed_1.json"]
        loaded = convert.load_episode(tmp_path, "hb1_feasible_seed_1")
        assert loaded["initial_home_config"] == home
        assert set(loaded) == {"meta", "query", "user_location", "eval", "home_id",
                               "initial_home_config"}
        assert HomeBenchEngine(loaded["initial_home_config"]).device_ids()


# -- ground-truth corrections -------------------------------------------------------------------------


class TestLabelCorrections:
    """A gold HomeBench's own home contradicts is replaced by the call the request asks
    for; the episode keeps its name and records what was there."""

    ROW = _row("h1_one_9", "unexist_attribute",
               "Set the brightness of the light in the master bedroom to 60.", "'''error_input'''")
    FIX = {"h1_one_9": {"gold": "''' master_bedroom.light.set_brightness(60)'''",
                        "reason": "the light has a brightness"}}

    def _episodes(self):
        """The row as the converter built it, under its converted name."""
        return {"hb1_infeasible_seed_3": convert.build_episode(self.ROW, _home(),
                                                               Counter()).episode}

    def test_the_gold_becomes_the_call(self):
        """Gold, case, label and goals follow the corrected call; the original is kept."""
        episodes = self._episodes()
        # 1. Corrected, and renamed for its new case with its seed and a suffix.
        done = convert.apply_label_corrections(episodes, {1: _home()}, self.FIX)
        assert done == {"hb1_infeasible_seed_3": "hb1_feasible_seed_3_fixed"}
        assert list(episodes) == ["hb1_feasible_seed_3_fixed"]
        episode = episodes["hb1_feasible_seed_3_fixed"]
        meta = episode["meta"]
        # 2. The new gold, as a feasible single call with its assert.
        assert meta["gold"] == "''' master_bedroom.light.set_brightness(60)'''"
        assert (meta["case"], meta["homebench_type"]) == ("feasible", "normal")
        (goal,) = episode["eval"]["goals"]
        assert goal["device_id"] == "master_bedroom_light"
        assert goal["asserts"][0]["attribute"] == "brightness"
        assert goal["asserts"][0]["value"] == 60
        # 3. What was there.
        assert meta["label_correction"] == {"original_gold": "'''error_input'''",
                                            "original_type": "unexist_attribute",
                                            "reason": "the light has a brightness"}

    def test_applying_twice_changes_nothing(self):
        """An episode already corrected and renamed is left alone, so its original
        stays the original; one corrected under its old name is only renamed."""
        # 1. Twice: nothing the second time.
        episodes = self._episodes()
        convert.apply_label_corrections(episodes, {1: _home()}, self.FIX)
        assert convert.apply_label_corrections(episodes, {1: _home()}, self.FIX) == {}
        original = "'''error_input'''"
        assert episodes["hb1_feasible_seed_3_fixed"]["meta"]["label_correction"][
            "original_gold"] == original
        # 2. Corrected but not yet renamed: renamed, not corrected again.
        old = {"hb1_infeasible_seed_3": episodes.pop("hb1_feasible_seed_3_fixed")}
        assert convert.apply_label_corrections(old, {1: _home()}, self.FIX) == {
            "hb1_infeasible_seed_3": "hb1_feasible_seed_3_fixed"}
        assert old["hb1_feasible_seed_3_fixed"]["meta"]["label_correction"][
            "original_gold"] == original

    def test_a_fixed_name_is_a_homebench_name_and_selectable(self, tmp_path):
        """The suite's name pattern accepts the suffix, and selection reads its seed."""
        from tests.simuhome.bench import run
        from tests.simuhome.bench.suites.homebench import NAME, SUITE
        # 1. A HomeBench name, with the cost of a one-part request.
        assert NAME.match("hb1_feasible_seed_2631_fixed")
        assert SUITE.estimate("hb1_feasible_seed_2631_fixed") == (2, 2, 0)
        # 2. Selected by its group and its seed number.
        (tmp_path / "episodes").mkdir()
        for name in ("hb1_feasible_seed_2", "hb1_feasible_seed_2631_fixed"):
            (tmp_path / "episodes" / f"{name}.json").write_text("{}", encoding="utf-8")
        assert run.select_episodes(tmp_path, "hb1_feasible", "2631", None, SUITE) == [
            "hb1_feasible_seed_2631_fixed"]

    def test_the_table_lists_the_ecai_episodes_by_row_id(self):
        """The repo's table: every row's gold parses as its row's kind (single or multi)
        and says why it was corrected; a part made `error_input` names what it lacks."""
        table = convert.load_label_corrections()
        assert len(table) == 35
        for row_id, row in table.items():
            # 1. A well-formed gold for its row, with a reason.
            calls = convert.parse_gold(row["gold"], multi="_multi_" in row_id)
            assert calls and row["reason"], row_id
            # 2. A part corrected to error_input says which attribute is missing.
            if row.get("part_reason") or row.get("attribute"):
                assert None in calls and row["attribute"], row_id
        assert {table[i]["attribute"] for i in ("home45_multi_52", "home49_multi_374")} == {
            "intensity", "temperature"}

    def test_a_second_correction_keeps_the_original(self):
        """An episode corrected before and corrected again keeps the dataset's original
        gold as its original, and keeps its `_fixed` name when its case is unchanged."""
        # 1. Corrected once.
        row = _row("h1_one_9", "unexist_attribute",
                   "Set the brightness of the light in the master bedroom to 60.",
                   "'''error_input'''")
        episodes = {"hb1_infeasible_seed_3": convert.build_episode(row, _home(),
                                                                   Counter()).episode}
        first = {"h1_one_9": {"gold": "''' master_bedroom.light.set_brightness(60)'''",
                              "reason": "has brightness"}}
        convert.apply_label_corrections(episodes, {1: _home()}, first)
        # 2. Corrected again to another value: same name, original kept.
        second = {"h1_one_9": {"gold": "''' master_bedroom.light.set_brightness(70)'''",
                               "reason": "has brightness"}}
        assert convert.apply_label_corrections(episodes, {1: _home()}, second) == {}
        meta = episodes["hb1_feasible_seed_3_fixed"]["meta"]
        assert meta["gold"] == "''' master_bedroom.light.set_brightness(70)'''"
        assert meta["label_correction"]["original_gold"] == "'''error_input'''"

    def test_a_multi_part_is_made_impossible(self):
        """"Set the light to 20 and the curtain's intensity to 10": the second part
        becomes `error_input` on the curtain; the first keeps its goal; the request is
        now mixed, and renamed so."""
        # 1. A converted two-part feasible request, and its correction.
        row = _row("h1_multi_4", "multi2_normal",
                   "Set the light to 20 in the master bedroom and the curtain intensity to "
                   "10 in the master bedroom.",
                   "'''master_bedroom.light.set_brightness(20),"
                   "master_bedroom.curtain.set_degree(10),'''")
        episodes = {"hb2_feasible_seed_4": convert.build_episode(row, _home(),
                                                                 Counter()).episode}
        first_goal = copy.deepcopy(episodes["hb2_feasible_seed_4"]["eval"]["goals"][0])
        fix = {"h1_multi_4": {"gold": "'''master_bedroom.light.set_brightness(20),"
                                      "error_input,'''",
                              "reason": "no intensity", "attribute": "intensity"}}
        # 2. Corrected and renamed for its new case.
        done = convert.apply_label_corrections(episodes, {1: _home()}, fix)
        assert done == {"hb2_feasible_seed_4": "hb2_mixed_seed_4_fixed"}
        episode = episodes["hb2_mixed_seed_4_fixed"]
        assert (episode["meta"]["case"], episode["meta"]["homebench_type"]) == (
            "mixed", "multi2_mix")
        # 3. The first part unchanged, the second an impossible part on the curtain.
        first, second = episode["eval"]["goals"]
        assert first == first_goal
        assert second == {"room_id": "master_bedroom", "device_id": "master_bedroom_curtain",
                          "device_type": "curtain", "feasibility": False,
                          "reason": "unexist_attribute", "resolved": True,
                          "attribute": "intensity"}


# -- the P1 gate on the real dataset ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def rows():
    """The 17,366 test rows."""
    return convert.read_jsonl(HOMEBENCH / convert.TEST_DATA)


@pytest.fixture(scope="module")
def conversion():
    """One conversion of the real dataset, shared by the gate tests."""
    return convert.convert_dataset(HOMEBENCH)


@needs_data
class TestGate:
    def test_grammar_parses_every_row(self, rows):
        assert len(rows) == 17366
        total = 0
        for row in rows:
            expected = convert.multi_parts(row["type"])
            parts = convert.parse_gold(row["output"], multi=expected is not None)
            assert len(parts) == (expected or 1), row["id"]
            total += len(parts)
        assert total > 17366

    def test_every_gold_call_applies_to_its_home(self, conversion):
        report = convert.verify_gold(conversion, HOMEBENCH)
        assert report["rows"] == 17366
        assert report["errors"] == [], report["errors"][:5]
        # Every call applies, the 70 vacuum operation calls included.
        assert report["applied"] == report["calls"]

    def test_heating_correction_is_what_makes_the_gold_valid(self, rows, conversion):
        values = [c.arg for r in rows for c in convert.parse_gold(
            r["output"], convert.multi_parts(r["type"]) is not None)
            if c and c.device == "heating" and c.command == "set_temperature"]
        assert len(values) == 1197 and min(values) >= 16 and max(values) <= 30
        # Against the declared 30-100 many of them would be violations.
        assert sum(v < 30 for v in values) > 800
        checked = 0
        for config in conversion.homes.values():
            for room in config["rooms"].values():
                for device in room["devices"]:
                    # Not every heater has a temperature attribute at all.
                    if device["device_type"] == "heating" and (
                            "temperature" in device["constraints"]):
                        checked += 1
                        assert device["constraints"]["temperature"] == {
                            "lowest": 16, "highest": 30}
        assert checked == 129

    def test_engine_offers_exactly_the_homes_methods(self, conversion):
        """Per device, the engine's commands equal the data's method list, minus the
        three media setters nothing asserts, plus the vacuum's operation commands the
        gold uses."""
        records = convert.read_jsonl(HOMEBENCH / convert.HOME_DATA)
        for record in records:
            machine = HomeBenchEngine(conversion.homes[record["home_id"]])
            offered = {}
            for m in record["method"]:
                room = None if m["room_name"] == "None" else m["room_name"]
                device_id = engine.make_device_id(room, m["device_name"])
                offered.setdefault(device_id, set()).add(m["operation"])
            for device_id in machine.device_ids():
                expected = offered[device_id] - {"set_song", "set_artist", "set_style"}
                if machine.device(device_id)["device_type"] == "vacuum_robot":
                    expected |= {"start", "stop", "pause", "charge"}
                assert set(machine.exposed_commands(device_id)) == expected, (
                    record["home_id"], device_id)
            assert set(machine.device_ids()) == set(offered)

    def test_type_and_bucket_counts_match_the_plan(self, rows, conversion):
        types = Counter(r["type"] for r in rows)
        assert len(types) == 26
        assert types["normal"] == 6187 and types["unexist_device"] == 4194
        assert types["unexist_attribute"] == 2571
        buckets = Counter(convert.official_bucket(r["type"]) for r in rows)
        assert [buckets[b] for b in convert.BUCKETS] == [6187, 6765, 245, 4072, 97]
        manifest = conversion.manifest
        assert [manifest["by_bucket"][b] for b in convert.BUCKETS] == [
            6187, 6765, 245, 4072, 97]
        assert len(manifest["by_homebench_type"]) == 26
        assert manifest["episodes"] == manifest["rows"] == 17366
        # Every episode keeps the official id and label; hb types are hb1..hb10.
        assert {e["meta"]["query_type"] for e in conversion.episodes.values()} == {
            f"hb{n}" for n in range(1, 11)}
        assert Counter(e["meta"]["homebench_type"]
                       for e in conversion.episodes.values()) == types

    def test_names_are_unique_stable_and_seeds_contiguous(self, conversion):
        names = conversion.names_by_id
        assert len(names) == len(set(names.values())) == 17366
        again = convert.convert_dataset(HOMEBENCH)
        assert again.names_by_id == names
        assert again.episodes == conversion.episodes          # byte-for-byte content
        # Another RNG seed reorders the sample but not the set of names.
        other = convert.convert_dataset(HOMEBENCH, rng_seed=1)
        assert set(other.episodes) == set(conversion.episodes)
        assert other.names_by_id != names
        # seed_n is 1..K inside each (type, case), and the name says so.
        groups = {}
        for name, episode in conversion.episodes.items():
            meta = episode["meta"]
            assert name == f"{meta['query_type']}_{meta['case']}_seed_{meta['seed']}"
            groups.setdefault((meta["query_type"], meta["case"]), []).append(meta["seed"])
        for seeds in groups.values():
            assert sorted(seeds) == list(range(1, len(seeds) + 1))
        assert len(groups) == 27 and re.fullmatch(r"[0-9a-f]{64}", conversion.manifest[
            "source"]["test_data"]["sha256"])

    def test_shuffle_spreads_a_small_sample_over_many_homes(self, conversion):
        first = [e for n, e in conversion.episodes.items()
                 if n.startswith("hb1_feasible_seed_") and e["meta"]["seed"] <= 40]
        assert len({e["home_id"] for e in first}) >= 25

    def test_episodes_have_the_simuhome_shape(self, conversion):
        for episode in list(conversion.episodes.values())[:2000]:
            assert set(episode) == {"meta", "query", "user_location", "eval", "home_id"}
            assert len(episode["eval"]["goals"]) == (
                convert.multi_parts(episode["meta"]["homebench_type"]) or 1)
            for goal in episode["eval"]["goals"]:
                assert ("feasibility" in goal) == (goal.get("feasibility") is False)

    def test_infeasible_targets_are_mostly_resolved(self, conversion):
        diag = conversion.manifest["diagnostics"]
        parts = diag["parts_resolved"] + diag["parts_unresolved"]
        assert diag["parts_resolved"] / parts > 0.9
        # Every resolved target names a real room of the home or the household.
        for episode in list(conversion.episodes.values())[:3000]:
            home = conversion.homes[episode["home_id"]]
            for goal in episode["eval"]["goals"]:
                if goal.get("resolved"):
                    assert goal["room_id"] in set(home["rooms"]) | set(engine.ROOMS) | {
                        engine.HOUSEHOLD_ROOM}

    def test_engine_does_not_mutate_the_home_it_was_built_from(self, conversion):
        snapshot = copy.deepcopy(conversion.homes[0])
        machine = HomeBenchEngine(conversion.homes[0])
        lights = [d for d in machine.device_ids() if d.endswith("_light")]
        machine.apply(lights[0], "turn_on")
        assert conversion.homes[0] == snapshot
        assert json.loads(json.dumps(conversion.manifest)) == conversion.manifest
