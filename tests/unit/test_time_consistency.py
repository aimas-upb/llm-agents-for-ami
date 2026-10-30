"""A goal's stated clock times against its relative timing, "finish
together" goals (`completes_with`) against their run lengths, and a trigger
against the clock-timed start of its own device."""

from datetime import datetime

import pytest

from ami_agents.agents.interaction_solver.utils.goal_profile import build_checklist
from ami_agents.agents.interaction_solver.utils.time_consistency import (
    check, check_completion, check_triggers, triggered_after_timed)
from ami_agents.shared.models.goal_structure import GoalStructure, GoalTiming

NOW = datetime(2025, 8, 23, 16, 55, 35)
ACHIEVE = {"goal_specificity": "explicit", "goal_kind": "achievement",
           "intent_text": "set refrigerator 2 to 6 C"}


def _structure(goals, predicates=None):
    return GoalStructure.from_dict({"goals": goals, "predicates": predicates or {}},
                                   intent_text="x", structure="dependency")


def _washer(direction, offset="23 minutes", role="trigger"):
    return {"P1": {"predicate_subject": "device_property", "role": role,
                   "predicate_text": "washer 1 finishes", "applies_to": ["G1"],
                   **({"offset_text": offset, "offset_direction": direction}
                      if role == "trigger" else {})}}


WASHER_LEFT = {"P1": 59 * 60.0}                       # finishes at 17:54:35


def test_the_fields_travel():
    timing = GoalTiming.from_dict({"starts": "on_trigger", "stated_start_time_text": "5:48 PM",
                                   "ends": {"kind": "duration", "text": "1 hour"},
                                   "stated_end_time_text": "6:48 PM"})
    assert GoalTiming.from_dict(timing.to_wire_dict()) == timing


class TestStarts:
    def test_23_minutes_before_the_washer_is_not_5_48(self):
        result = check(_structure({"G1": {**ACHIEVE, "timing": {
            "starts": "on_trigger", "stated_start_time_text": "5:48 PM"}}},
            _washer("before")), NOW, WASHER_LEFT)
        (question,) = result.conflicts
        assert "5:48 PM" in question and "5:31 PM" in question and "5:54 PM" in question

    def test_agreeing_times_plan_with_a_note(self):
        result = check(_structure({"G1": {**ACHIEVE, "timing": {
            "starts": "on_trigger", "stated_start_time_text": "5:31 PM"}}},
            _washer("before")), NOW, WASHER_LEFT)
        assert not result.conflicts and "agrees" in result.notes[0]

    def test_after_the_event(self):
        structure = _structure({"G1": {**ACHIEVE, "timing": {
            "starts": "on_trigger", "stated_start_time_text": "6:00 PM"}}},
            _washer("after", "6 minutes"))
        assert not check(structure, NOW, WASHER_LEFT).conflicts      # 17:54:35 + 6 min
        structure.goals["G1"].timing.stated_start_time_text = "6:30 PM"
        assert check(structure, NOW, WASHER_LEFT).conflicts

    def test_minutes_from_now(self):
        # qt4-1: "At 11:35 AM, that is 13 minutes from now", said at 11:36:54.
        structure = _structure({"G1": {**ACHIEVE, "timing": {
            "starts": "after_delay", "offset_text": "13 minutes",
            "stated_start_time_text": "11:35 AM"}}})
        assert check(structure, datetime(2025, 8, 23, 11, 36, 54), {}).conflicts
        structure.goals["G1"].timing.stated_start_time_text = "11:50 AM"
        assert not check(structure, datetime(2025, 8, 23, 11, 36, 54), {}).conflicts

    def test_after_another_goal(self):
        structure = _structure({
            "G1": {**ACHIEVE, "timing": {"starts": "after_delay", "offset_text": "10 minutes"}},
            "G2": {**ACHIEVE, "timing": {"starts": "after_goal", "ref_goal": "G1",
                                         "offset_text": "5 minutes",
                                         "stated_start_time_text": "5:10 PM"}}})
        assert not check(structure, NOW, {}).conflicts               # 16:55:35 + 15 min
        structure.goals["G2"].timing.stated_start_time_text = "5:30 PM"
        assert check(structure, NOW, {}).conflicts

    @pytest.mark.parametrize("seconds, conflict", [(59, False), (61, True)])
    def test_the_tolerance(self, seconds, conflict):
        now = datetime(2025, 8, 23, 17, 0, 0)
        structure = _structure({"G1": {**ACHIEVE, "timing": {
            "starts": "after_delay", "offset_text": f"{600 + seconds} seconds",
            "stated_start_time_text": "5:10 PM"}}})
        assert bool(check(structure, now, {}, tolerance=60).conflicts) is conflict

    def test_across_midnight(self):
        structure = _structure({"G1": {**ACHIEVE, "timing": {
            "starts": "after_delay", "offset_text": "20 minutes",
            "stated_start_time_text": "12:10 AM"}}})
        assert not check(structure, datetime(2025, 8, 23, 23, 50, 0), {}).conflicts

    def test_no_remaining_time_cannot_be_verified(self):
        result = check(_structure({"G1": {**ACHIEVE, "timing": {
            "starts": "on_trigger", "stated_start_time_text": "5:48 PM"}}},
            _washer("before")), NOW, {"P1": None})
        assert not result.conflicts and "could not verify" in result.notes[0]


class TestEnds:
    def test_a_duration_from_a_known_start(self):
        structure = _structure({"G1": {**ACHIEVE, "goal_kind": "maintenance", "timing": {
            "starts": "after_delay", "offset_text": "25 minutes",
            "ends": {"kind": "duration", "text": "1 hour"}, "stated_end_time_text": "6:20 PM"}}})
        assert not check(structure, NOW, {}).conflicts               # 16:55:35 + 85 min
        structure.goals["G1"].timing.stated_end_time_text = "7:00 PM"
        (question,) = check(structure, NOW, {}).conflicts
        assert "to end at 7:00 PM" in question

    def test_until_the_washer_finishes(self):
        structure = _structure({"G1": {**ACHIEVE, "goal_kind": "maintenance", "timing": {
            "starts": "now", "stated_end_time_text": "5:30 PM"}}}, _washer(None, role="termination"))
        assert check(structure, NOW, WASHER_LEFT).conflicts          # it finishes 17:54
        structure.goals["G1"].timing.stated_end_time_text = "5:54 PM"
        assert not check(structure, NOW, WASHER_LEFT).conflicts


class TestChecklist:
    def test_remaining_time_is_asked_for_a_stated_time_whatever_the_direction(self):
        goals = {"G1": {**ACHIEVE, "timing": {"starts": "on_trigger",
                                              "stated_start_time_text": "6:00 PM"}}}
        checklist = build_checklist(_structure(goals, _washer("after", "6 minutes")))
        assert [(c["question"], c["target"]) for c in checklist] == [("time_to_event", "P1")]
        goals["G1"]["timing"].pop("stated_start_time_text")
        assert build_checklist(_structure(goals, _washer("after", "6 minutes"))) == []

    def test_and_for_a_termination(self):
        goals = {"G1": {**ACHIEVE, "goal_kind": "maintenance", "timing": {
            "starts": "now", "stated_end_time_text": "5:54 PM"}}}
        checklist = build_checklist(_structure(goals, _washer(None, role="termination")))
        assert ("time_to_event", "P1") in [(c["question"], c["target"]) for c in checklist]


# -- finishing together (completes_with) ---------------------------------------

SEED_46_NOW = datetime(2025, 8, 23, 10, 25, 2)
DISHWASHER = {"P1": {"predicate_subject": "device_property", "role": "trigger",
                     "predicate_text": "dishwasher 1 finishes", "applies_to": ["G1"]}}
DRYER = {"goal_specificity": "explicit", "goal_kind": "achievement",
         "intent_text": "start dryer 1", "artifact_name": "bathroom_laundry_dryer_1"}


def _dryer(**timing):
    """Dryer 1's start, completing with the dishwasher (P1)."""
    return _structure({"G1": {**DRYER, "timing": {"completes_with": "P1", **timing}}},
                      DISHWASHER)


class TestCompletesWith:
    def test_the_field_travels_and_no_start_means_chosen(self):
        timing = GoalTiming.from_dict({"completes_with": "P1"})
        assert timing.start_chosen and GoalTiming.from_dict(timing.to_wire_dict()) == timing
        assert not GoalTiming.from_dict({"starts": "now", "completes_with": "P1"}).start_chosen
        assert GoalTiming.from_dict({"starts": "completes_with"}).starts is None

    def test_seed_46_the_stated_end_is_before_the_dishwasher_finishes(self):
        """10:25 with 89 minutes left: the dishwasher finishes at 11:54, so
        "both finish together by 11:39" cannot hold, whatever the dryer takes."""
        result = check(_dryer(starts="now", stated_end_time_text="11:39 AM"),
                       SEED_46_NOW, {"P1": 89 * 60.0})
        (question,) = result.conflicts
        assert "11:39" in question and "11:54" in question

    def test_the_run_length_unknown_is_refused_with_the_events_time(self):
        """A cycle's countdown is empty while the device is stopped: not
        started on a guess, and the event's time left is said."""
        result = check_completion(_dryer(), SEED_46_NOW, {"P1": 3600.0}, {"G1": None})
        (question,) = result.conflicts
        # 1. Why it can't be done, about which device.
        assert "bathroom_laundry_dryer_1 doesn't report how long" in question
        assert "won't start it on a guess" in question
        # 2. When the event is, as read (10:25 + 60 minutes).
        assert "in about 60 minutes, at 11:25 AM" in question

    def test_the_events_time_left_unknown_is_a_question(self):
        result = check_completion(_dryer(), SEED_46_NOW, {"P1": None}, {"G1": 3000.0})
        (question,) = result.conflicts
        assert "nothing tells when that will be" in question

    def test_no_start_and_a_run_longer_than_the_time_left_is_a_question(self):
        result = check_completion(_dryer(), SEED_46_NOW, {"P1": 40 * 60.0}, {"G1": 90 * 60.0})
        (question,) = result.conflicts
        assert "about 90 minutes" in question and "about 40 minutes" in question

    def test_no_start_and_a_run_that_fits_gets_its_start(self):
        result = check_completion(_dryer(), SEED_46_NOW, {"P1": 90 * 60.0}, {"G1": 40 * 60.0})
        assert not result.conflicts and "starts at 11:15 AM" in result.notes[0]

    def test_a_given_start_that_ends_elsewhere_is_a_question(self):
        result = check_completion(_dryer(starts="now"), SEED_46_NOW, {"P1": 89 * 60.0},
                                  {"G1": 60 * 60.0})
        (question,) = result.conflicts
        assert "would finish at 11:25 AM" in question and "11:54 AM" in question

    def test_a_given_start_that_ends_with_the_event_is_noted(self):
        result = check_completion(_dryer(starts="now"), SEED_46_NOW, {"P1": 89 * 60.0},
                                  {"G1": 89 * 60.0})
        assert not result.conflicts and "with P1" in result.notes[0]

    def test_completing_with_a_goal_needs_only_the_run_length(self):
        """The other goal has not started: its time left is read as the tree
        runs."""
        structure = _structure({"G1": {**ACHIEVE, "timing": {"starts": "now"}},
                                "G2": {**DRYER, "timing": {"completes_with": "G1"}}})
        assert check_completion(structure, NOW, {}, {"G2": 600.0}).conflicts == []
        assert check_completion(structure, NOW, {}, {"G2": None}).conflicts


# -- a trigger against its device's start ----------------------------------------

DRYER_1 = "utility_room_laundry_dryer_1"
WASHER_DONE = {"P1": {"predicate_subject": "device_property", "role": "trigger",
                      "predicate_text": "washer 1 finishes", "applies_to": ["G3"]}}


def _start_then_pause(at="6:14 PM", **pause_timing):
    """qt4-3 seed 11's shape: switch dryer 1 on at a time, start its cycle
    after that, pause it when washer 1 finishes (P1)."""
    goal = {"goal_specificity": "explicit", "goal_kind": "achievement",
            "artifact_name": DRYER_1}
    return _structure({
        "G1": {**goal, "intent_text": "switch dryer 1 on",
               "timing": {"starts": "at_time", "time_text": at}},
        "G2": {**goal, "intent_text": "start dryer 1",
               "timing": {"starts": "after_goal", "ref_goal": "G1"}},
        "G3": {**goal, "intent_text": "pause dryer 1",
               "timing": {"starts": "on_trigger", **pause_timing}},
    }, WASHER_DONE)


class TestTriggerBeforeStart:
    def test_the_pause_is_checked_against_the_clock_timed_goals_on_its_device(self):
        """G1 is timed by the clock, G2 through it; P1 is the pause's trigger."""
        assert triggered_after_timed(_start_then_pause()) == {"G3": ("P1", ["G1", "G2"])}

    def test_a_goal_timed_from_the_trigger_is_not_clock_timed(self):
        """"resume it 10 minutes later" follows the pause, so it is no start
        the trigger could come before; with no clock-timed goal, no check."""
        structure = _start_then_pause()
        # 1. Every goal but the pause now follows the pause.
        for gid in ("G1", "G2"):
            structure.goals[gid].timing = GoalTiming.from_dict(
                {"starts": "after_goal", "ref_goal": "G3", "offset_text": "10 minutes"})
        # 2. Nothing left to check.
        assert triggered_after_timed(structure) == {}

    def test_the_washer_finishing_before_the_dryer_starts_is_a_question(self):
        """16:55 with 30 minutes left: the washer finishes at 17:25, before the
        dryer's 18:14 start, so the pause would come first."""
        result = check_triggers(_start_then_pause(), NOW, {"P1": 30 * 60.0})
        (question,) = result.conflicts
        assert "pause dryer 1" in question and "5:25 PM" in question
        assert "6:14 PM" in question and "leave" in question

    def test_the_washer_finishing_after_the_start_is_noted(self):
        result = check_triggers(_start_then_pause(), NOW, {"P1": 120 * 60.0})
        assert not result.conflicts and "after" in result.notes[0]

    def test_the_washers_time_left_unknown_is_only_noted(self):
        result = check_triggers(_start_then_pause(), NOW, {"P1": None})
        assert not result.conflicts and "could not check" in result.notes[0]

    def test_the_profile_asks_for_the_washers_time_left(self):
        """Without a stated time or an offset before, only this check needs it."""
        checklist = build_checklist(_start_then_pause())
        assert ("time_to_event", "P1") in [(c["question"], c["target"]) for c in checklist]

    def test_an_event_said_with_when_is_not_doubled(self):
        """The structurer may keep "when" in the event's words."""
        structure = _start_then_pause()
        structure.predicates["P1"].predicate_text = "when washer 1 finishes"
        (question,) = check_triggers(structure, NOW, {"P1": 30 * 60.0}).conflicts
        assert "due when washer 1 finishes" in question and "when when" not in question
