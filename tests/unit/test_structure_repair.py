"""structure_repair: the deterministic repairs applied before dependency planning."""

from ami_agents.agents.interaction_solver.utils.structure_repair import (
    fill_goal_defaults,
    link_triggers,
    repair_structure,
)
from ami_agents.shared.models.goal_structure import GoalStructure

DRYER_DONE = {"predicate_specificity": "explicit", "predicate_subject": "device_property",
              "predicate_text": "dryer 1 finishes", "artifact_name": "dryer_1",
              "property_name": "operationalState", "comparison": "==",
              "target_value_determined": "0", "role": "trigger"}


def _structure(goals, predicates):
    return GoalStructure.from_dict({"goals": goals, "predicates": predicates},
                                   intent_text="x", structure="dependency")


def _goal(**fields):
    return {"goal_specificity": "explicit", "goal_kind": "achievement",
            "intent_text": "turn on light 1", "artifact_name": "light_1", **fields}


class TestLinkTriggers:
    def test_goals_without_timing_start_on_the_trigger(self):
        s = _structure({"G1": _goal(), "G2": _goal(artifact_name="fridge_1")},
                       {"P1": {**DRYER_DONE, "applies_to": ["G1", "G2"],
                               "offset_text": "9 minutes", "offset_direction": "after"}})
        notes = link_triggers(s)
        assert [g.timing.starts for g in s.goals.values()] == ["on_trigger", "on_trigger"]
        assert s.predicates["P1"].offset_text == "9 minutes"
        assert notes == ["G1, G2 start when P1 (dryer 1 finishes): the structure named the "
                         "trigger but gave the goals no start on it"]

    def test_a_goal_delay_becomes_the_trigger_offset(self):
        s = _structure({"G1": _goal(timing={"starts": "after_delay", "offset_text": "9 minutes"})},
                       {"P1": {**DRYER_DONE, "applies_to": ["G1"]}})
        link_triggers(s)
        assert s.goals["G1"].timing.starts == "on_trigger"
        assert s.goals["G1"].timing.offset_text is None
        assert (s.predicates["P1"].offset_text, s.predicates["P1"].offset_direction) == \
            ("9 minutes", "after")

    def test_the_trigger_keeps_its_own_offset(self):
        s = _structure({"G1": _goal(timing={"starts": "after_delay", "offset_text": "9 minutes"})},
                       {"P1": {**DRYER_DONE, "applies_to": ["G1"],
                               "offset_text": "5 minutes", "offset_direction": "before"}})
        link_triggers(s)
        assert (s.predicates["P1"].offset_text, s.predicates["P1"].offset_direction) == \
            ("5 minutes", "before")

    def test_a_structure_that_already_waits_is_untouched(self):
        s = _structure({"G1": _goal(timing={"starts": "on_trigger"}),
                        "G2": _goal(artifact_name="fridge_1")},
                       {"P1": {**DRYER_DONE, "applies_to": ["G1", "G2"]}})
        assert link_triggers(s) == []
        assert s.goals["G2"].timing is None

    def test_completes_with_the_trigger_counts_as_waiting(self):
        s = _structure({"G1": _goal(timing={"completes_with": "P1"})},
                       {"P1": {**DRYER_DONE, "applies_to": ["G1"]}})
        assert link_triggers(s) == []
        assert s.goals["G1"].timing.starts is None

    def test_a_clock_time_becomes_the_stated_time_to_check(self):
        # "At 9:56 AM, which is 10 minutes after washer 1 finishes, ..."
        s = _structure({"G1": _goal(timing={"starts": "at_time", "time_text": "09:56"})},
                       {"P1": {**DRYER_DONE, "applies_to": ["G1"],
                               "offset_text": "10 minutes", "offset_direction": "after"}})
        link_triggers(s)
        timing = s.goals["G1"].timing
        assert (timing.starts, timing.time_text, timing.stated_start_time_text) == \
            ("on_trigger", None, "09:56")

    def test_after_the_predicate_itself_is_on_trigger(self):
        # gpt-5-nano: "17 minutes after the dishwasher finishes" as after_goal of P1
        s = _structure({"G1": _goal(timing={"starts": "after_goal", "ref_goal": "P1",
                                            "offset_text": "17 minutes"}),
                        "G2": _goal(timing={"starts": "after_goal", "ref_goal": "G1"})},
                       {"P1": {**DRYER_DONE, "applies_to": ["G1", "G2"]}})
        link_triggers(s)
        timing = s.goals["G1"].timing
        assert (timing.starts, timing.ref_goal, timing.offset_text) == ("on_trigger", None, None)
        assert (s.predicates["P1"].offset_text, s.predicates["P1"].offset_direction) == \
            ("17 minutes", "after")
        assert s.goals["G2"].timing.starts == "after_goal"

    def test_goal_anchored_starts_are_kept(self):
        s = _structure({"G1": _goal(timing={"starts": "after_goal", "ref_goal": "G0"}),
                        "G2": _goal(timing={"starts": "with_goal", "ref_goal": "G1"})},
                       {"P1": {**DRYER_DONE, "applies_to": ["G1", "G2"]}})
        assert link_triggers(s) == []
        assert s.goals["G1"].timing.starts == "after_goal"
        assert s.goals["G2"].timing.starts == "with_goal"

    def test_only_the_unanchored_goals_move(self):
        s = _structure({"G1": _goal(timing={"starts": "after_goal", "ref_goal": "G2"}),
                        "G2": _goal(artifact_name="fridge_1")},
                       {"P1": {**DRYER_DONE, "applies_to": ["G1", "G2"]}})
        notes = link_triggers(s)
        assert s.goals["G1"].timing.starts == "after_goal"
        assert s.goals["G2"].timing.starts == "on_trigger"
        assert notes[0].startswith("G2 start when P1")

    def test_other_roles_and_unknown_goals_are_ignored(self):
        s = _structure({"G1": _goal()},
                       {"P1": {**DRYER_DONE, "role": "scope", "applies_to": ["G1"]},
                        "P2": {**DRYER_DONE, "applies_to": ["G9"]}})
        assert link_triggers(s) == []
        assert s.goals["G1"].timing is None


def test_repair_structure_runs_every_repair_and_returns_the_notes():
    s = _structure({"G1": _goal()}, {"P1": {**DRYER_DONE, "applies_to": ["G1"]}})
    notes = repair_structure(s)
    assert len(notes) == 1 and s.goals["G1"].timing.starts == "on_trigger"


class TestPredicates:
    def test_an_empty_slot_is_dropped(self):
        s = _structure({"G1": _goal(timing={"starts": "now"})}, {"P1": {"connective": "and"}})
        notes = repair_structure(s)
        assert s.predicates == {} and notes[0].startswith("P1 dropped: empty")

    def test_a_text_less_trigger_takes_the_requests_event_clause(self):
        s = GoalStructure.from_dict(
            {"goals": {"G1": _goal(timing={"starts": "on_trigger"})},
             "predicates": {"P1": {"role": "trigger", "applies_to": ["G1"],
                                   "offset_text": "30 minutes", "offset_direction": "after"}}},
            intent_text="At 3:30 PM, which is 30 minutes after the dryer 1 in the utility room "
                        "finishes, turn on air purifier 1 in the bathroom", structure="dependency")
        repair_structure(s)
        p = s.predicates["P1"]
        assert (p.predicate_text, p.predicate_subject, p.offset_text) == \
            ("the dryer 1 in the utility room finishes", "device_property", "30 minutes")

    def test_a_text_less_trigger_with_no_event_in_the_request_is_dropped(self):
        s = GoalStructure.from_dict(
            {"goals": {"G1": _goal(timing={"starts": "on_trigger"}),
                       "G2": _goal(timing={"starts": "at_time", "time_text": "12:10"})},
             "predicates": {"P1": {"role": "trigger", "applies_to": ["G1", "G2"]}}},
            intent_text="Start dishwasher 1 in the kitchen at 12:10 PM so it is ready for dinner",
            structure="dependency")
        notes = repair_structure(s)
        assert s.predicates == {}
        assert s.goals["G1"].timing.starts is None and s.goals["G2"].timing.starts == "at_time"
        assert notes == ["P1 dropped: a trigger with no condition in it; G1 start now instead"]

    def test_a_released_goal_keeps_another_trigger(self):
        s = GoalStructure.from_dict(
            {"goals": {"G1": _goal(timing={"starts": "on_trigger"})},
             "predicates": {"P1": {"role": "trigger", "applies_to": ["G1"]},
                            "P2": {**DRYER_DONE, "applies_to": ["G1"]}}},
            intent_text="x", structure="dependency")
        repair_structure(s)
        assert list(s.predicates) == ["P2"] and s.goals["G1"].timing.starts == "on_trigger"


class TestGoalDefaults:
    def test_effect_kind_and_specificity_defaults(self):
        s = _structure({"G1": {"intent_text": "start dishwasher 1", "artifact_class": "homeont:Dishwasher",
                               "artifact_name": "kitchen_dishwasher_1",
                               "affordance_class": "homeont:SetOnOffCommand",
                               "affordance_name": "onOff", "target_value_text": "true",
                               "target_value_determined": "true"}}, {})
        notes = fill_goal_defaults(s)
        g = s.goals["G1"]
        assert (g.goal_effect, g.goal_kind, g.goal_specificity) == ("set", "achievement", "explicit")
        assert notes == ["G1 taken as effect set, an achievement, explicit"]

    def test_explicit_without_a_device_title_is_incomplete(self):
        s = _structure({"G1": _goal(artifact_name=None, artifact_class="homeont:WindowCoveringController",
                                    affordance_class="saref:SetAbsoluteLevelCommand",
                                    affordance_name="position", goal_effect="set",
                                    location_class="homeont:LivingRoom")}, {})
        fill_goal_defaults(s)
        assert s.goals["G1"].goal_specificity == "incomplete"

    def test_explicit_without_an_action_title_is_incomplete(self):
        s = _structure({"G1": _goal(artifact_class="homeont:HeatPump",
                                    artifact_name="utility_room_heat_pump_1",
                                    affordance_class="homeont:SetModeCommand", goal_effect="set",
                                    target_value_text="heating mode", target_value_determined="4")}, {})
        fill_goal_defaults(s)
        assert s.goals["G1"].goal_specificity == "incomplete"

    def test_a_class_only_goal_is_incomplete_and_an_empty_goal_is_left(self):
        s = _structure({"G1": {"intent_text": "turn on utility room fan 1",
                               "affordance_class": "homeont:SetOnOffCommand",
                               "affordance_name": "onOff", "target_value_text": "on"},
                        "G2": {"intent_text": "I need the load ready to fold before dinner"}}, {})
        fill_goal_defaults(s)
        assert s.goals["G1"].goal_specificity == "incomplete"
        assert s.goals["G2"].goal_specificity is None and s.goals["G2"].goal_kind is None

    def test_a_well_formed_goal_is_untouched(self):
        s = _structure({"G1": _goal(goal_effect="set", affordance_name="onOff",
                                    affordance_class="homeont:SetOnOffCommand")}, {})
        assert fill_goal_defaults(s) == []
