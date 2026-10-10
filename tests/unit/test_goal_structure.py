"""GoalStructure: what goal structuring hands to the planner."""

from ami_agents.shared.models.goal_structure import (
    NA,
    GoalSpec,
    GoalStructure,
    PredicateSpec,
)


EXPLICIT = {
    "goal_specificity": "explicit", "goal_kind": "achievement",
    "intent_text": "set bathroom AC 1 to cooling",
    "location_class": "homeont:Bathroom", "artifact_class": "homeont:AirConditioner",
    "artifact_name": "bathroom_air_conditioner_1",
    "affordance_class": "homeont:SetModeCommand", "parent_class": "saref:Command",
    "affordance_name": "hvacMode", "parameter_name": "NA",
    "goal_effect": "set", "target_value_text": "cooling mode", "target_value_determined": "3",
}


class TestGoalSpec:
    def test_na_and_null_are_kept_apart(self):
        assert GoalSpec.from_dict({"parameter_name": "NA"}).parameter_name == NA
        assert GoalSpec.from_dict({"parameter_name": None}).parameter_name is None
        assert GoalSpec.from_dict({"parameter_name": "level"}).parameter_name == "level"

    def test_na_means_none_outside_the_two_na_slots(self):
        goal = GoalSpec.from_dict({"artifact_name": "NA", "target_value_text": "null"})
        assert goal.artifact_name is None
        assert goal.target_value_text is None

    def test_values_outside_a_vocabulary_are_dropped(self):
        goal = GoalSpec.from_dict({"goal_specificity": "implicit",
                                   "goal_kind": "Maintenance",
                                   "goal_effect": "toggle"})
        assert goal.goal_specificity is None      # the segmenter says "ambiguous"
        assert goal.goal_kind == "maintenance"    # case folded
        assert goal.goal_effect is None

    def test_explicit_round_trip(self):
        wire = GoalSpec.from_dict(EXPLICIT).to_wire_dict()
        assert wire["goal_dependency_structure"] is None
        for key, value in EXPLICIT.items():
            assert wire[key] == value, key

    def test_a_modify_goal_carries_its_percentage_flag(self):
        goal = GoalSpec.from_dict({**EXPLICIT, "goal_effect": "modify",
                                   "target_value_text": "by 20%",
                                   "target_value_determined": "-20",
                                   "target_value_is_percentage": "true"})
        wire = goal.to_wire_dict()
        assert wire["target_value_is_percentage"] is True
        assert wire["target_value_determined"] == "-20"
        assert GoalSpec.from_dict({"target_value_is_percentage": False}).target_value_is_percentage is False
        # Missing or unreadable: not a percentage.
        assert GoalSpec.from_dict({}).target_value_is_percentage is False
        assert GoalSpec.from_dict({"target_value_is_percentage": "maybe"}).target_value_is_percentage is False

    def test_an_incomplete_goal_carries_its_quantifier(self):
        goal = GoalSpec.from_dict({"goal_specificity": "incomplete",
                                   "intent_text": "turn on the kitchen lights",
                                   "location_class": "homeont:Kitchen",
                                   "artifact_class": "homeont:Light", "quantifier": "All"})
        assert goal.to_wire_dict()["quantifier"] == "all"
        assert GoalSpec.from_dict({"quantifier": "one"}).quantifier == "one"
        # "any" is the predicates' label; for a goal it settles to "one".
        assert GoalSpec.from_dict({"quantifier": "any"}).quantifier == "one"
        assert GoalSpec.from_dict({"quantifier": "each"}).quantifier is None

    def test_ambiguous_carries_environment_vars_not_device_slots(self):
        goal = GoalSpec.from_dict({
            "goal_specificity": "ambiguous", "goal_kind": "achievement",
            "intent_text": "the bathroom is so damp",
            "artifact_name": "should not survive",
            "implied_environment_vars": [{
                "property_name": "relative_humidity",
                "property_quantity": "quantitykind:RelativeHumidity",
                "property_measurement_unit": "unit:PERCENT",
                "property_sensed_space": {"space_class": "homeont:Bathroom",
                                          "space_name": "Bathroom"}}]})
        wire = goal.to_wire_dict()
        assert "artifact_name" not in wire
        (var,) = wire["implied_environment_vars"]
        assert var["property_sensed_space"] == {"space_class": "homeont:Bathroom",
                                                "space_name": "Bathroom"}

    def test_timing(self):
        goal = GoalSpec.from_dict({"timing": {"starts": "after_goal", "ref_goal": "G1",
                                              "offset_text": "8 minutes",
                                              "ends": {"kind": "duration",
                                                       "text": "the next 5 hours"}}})
        assert goal.to_wire_dict()["timing"] == {
            "starts": "after_goal", "offset_text": "8 minutes", "ref_goal": "G1",
            "ends": {"kind": "duration", "text": "the next 5 hours"}}


class TestPredicateSpec:
    def test_device_predicate(self):
        pred = PredicateSpec.from_dict({
            "predicate_specificity": "incomplete", "predicate_subject": "device_property",
            "predicate_text": "the living room blinds are half open",
            "location_class": "homeont:LivingRoom",
            "artifact_class": "homeont:WindowCoveringController",
            "property_class": "homeont:WindowCoveringPosition",
            "property_field": "NA", "comparison": "==", "target_value_text": "half open",
            "quantifier": "all", "role": "scope", "applies_to": ["G1"]})
        wire = pred.to_wire_dict()
        assert wire["property_field"] == NA
        assert wire["quantifier"] == "all"
        assert wire["connective"] == "and"           # the default
        assert "offset_text" not in wire              # only triggers carry offsets

    def test_comparison_must_be_an_operator(self):
        assert PredicateSpec.from_dict({"comparison": ">="}).comparison == ">="
        assert PredicateSpec.from_dict({"comparison": "above"}).comparison is None

    def test_trigger_offsets(self):
        wire = PredicateSpec.from_dict({
            "role": "trigger", "applies_to": "G1",
            "offset_text": "23 minutes", "offset_direction": "before"}).to_wire_dict()
        assert wire["applies_to"] == ["G1"]
        assert (wire["offset_text"], wire["offset_direction"]) == ("23 minutes", "before")

    def test_ambiguous_environment_predicate(self):
        wire = PredicateSpec.from_dict({
            "predicate_specificity": "ambiguous",
            "predicate_subject": "environment_property",
            "qualitative_condition": "very bright",
            "implied_environment_vars": [{"property_name": "illuminance",
                                          "condition_direction": "high"}]}).to_wire_dict()
        assert wire["qualitative_condition"] == "very bright"
        assert wire["implied_environment_vars"][0]["condition_direction"] == "high"


class TestGoalStructure:
    def test_a_bare_goal_is_filed_as_g1(self):
        structure = GoalStructure.from_dict(EXPLICIT, intent_text="set it",
                                            qualifiers=["explicit", "achievement"],
                                            structure="simple")
        assert list(structure.goals) == ["G1"]
        assert structure.predicates == {}

    def test_dependency_structure(self):
        structure = GoalStructure.from_dict(
            {"goals": {"G1": {"goal_specificity": "explicit",
                              "timing": {"starts": "on_trigger"}}},
             "predicates": {"P1": {"role": "trigger", "applies_to": ["G1"]}},
             "unknown_key": 1},
            intent_text="when the washer finishes, turn on the light",
            structure="dependency")
        assert structure.structure == "dependency"
        assert structure.predicates["P1"].applies_to == ["G1"]
        # A goal without its own intent_text inherits the request's.
        assert structure.goals["G1"].intent_text.startswith("when the washer")

    def test_wire_keeps_the_words_for_a_planner_that_cannot_read_the_rest(self):
        wire = GoalStructure.from_dict(EXPLICIT, intent_text="set it").to_wire_dict()
        assert wire["text_intent"] == "set it"
        assert "category" not in wire
