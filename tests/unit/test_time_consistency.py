"""A goal's stated clock times against its relative timing."""

from datetime import datetime

import pytest

from ami_agents.agents.interaction_solver.utils.goal_profile import build_checklist
from ami_agents.agents.interaction_solver.utils.time_consistency import check
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
