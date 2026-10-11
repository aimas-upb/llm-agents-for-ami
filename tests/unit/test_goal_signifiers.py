"""signifier_key: the exact key a goal is remembered and recalled under."""

from ami_agents.shared.models.goal_structure import GoalSpec
from ami_agents.shared.utils.goal_signifiers import signifier_key

EXPLICIT = {"goal_specificity": "explicit", "goal_kind": "achievement",
            "intent_text": "set the bathroom AC to cooling",
            "location_class": "homeont:Bathroom", "artifact_class": "homeont:AirConditioner",
            "artifact_name": "bathroom_air_conditioner_1",
            "affordance_class": "homeont:SetModeCommand", "affordance_name": "hvacMode",
            "parameter_name": "NA", "goal_effect": "set", "target_value_text": "cooling",
            "target_value_determined": "3"}

INCOMPLETE = {"goal_specificity": "incomplete", "goal_kind": "achievement",
              "intent_text": "turn on the kitchen lights", "location_class": "homeont:Kitchen",
              "artifact_class": "homeont:OnOffLight",
              "affordance_class": "homeont:SetOnOffCommand", "affordance_name": "onOff",
              "goal_effect": "set", "target_value_text": "on", "quantifier": "all"}


def _key(**fields):
    return signifier_key(GoalSpec.from_dict(fields))


def _var(name, cls, room, direction):
    return {"property_name": name, "property_class": cls, "property_quantity": None,
            "property_measurement_unit": None,
            "property_sensed_space": {"space_class": room, "space_name": None},
            "property_change_direction": direction}


class TestExplicit:
    def test_the_whole_goal_description_except_the_determined_value(self):
        sk = _key(**EXPLICIT)
        assert sk.kind == "explicit"
        assert sk.key == {
            "goal_specificity": "explicit", "location_class": "homeont:Bathroom",
            "artifact_class": "homeont:AirConditioner",
            "artifact_name": "bathroom_air_conditioner_1",
            "affordance_class": "homeont:SetModeCommand", "affordance_name": "hvacMode",
            "parameter_name": "NA", "goal_effect": "set", "target_value_text": "cooling",
            "target_value_is_percentage": False}

    def test_the_determined_value_does_not_change_the_key(self):
        assert _key(**EXPLICIT).key == _key(**{**EXPLICIT, "target_value_determined": None}).key

    def test_a_null_parameter_is_left_out(self):
        assert "parameter_name" not in _key(**{**EXPLICIT, "parameter_name": None}).key


class TestIncomplete:
    def test_an_incomplete_goal_naming_its_action(self):
        sk = _key(**INCOMPLETE)
        assert sk.kind == "incomplete"
        assert sk.key["location_class"] == "homeont:Kitchen"
        assert sk.key["quantifier"] == "all"
        assert "artifact_name" not in sk.key and "goal_effect" not in sk.key

    def test_room_and_quantifier_tell_goals_apart(self):
        assert _key(**INCOMPLETE).key != _key(**{**INCOMPLETE, "location_class": "homeont:Bathroom"}).key
        assert _key(**INCOMPLETE).key != _key(**{**INCOMPLETE, "quantifier": "one"}).key

    def test_value_words_only_are_matched_by_the_request_text(self):
        sk = _key(**{**INCOMPLETE, "affordance_class": None, "affordance_name": None,
                     "intent_text": "make the hallway light blue", "target_value_text": "blue"})
        assert sk.kind == "incomplete_text"
        assert sk.key == {"goal_specificity": "incomplete",
                          "intent_text": "make the hallway light blue"}

    def test_goals_only_the_user_can_resolve_have_no_key(self):
        assert _key(**{**INCOMPLETE, "location_class": None, "artifact_class": None}) is None


class TestAmbiguous:
    def test_the_variables_with_their_room_and_direction(self):
        sk = _key(goal_specificity="ambiguous", goal_kind="achievement",
                  implied_environment_vars=[_var("relative_humidity",
                                                 "homeont:RelativeHumidity",
                                                 "homeont:Bathroom", "decrease")])
        assert sk.kind == "ambiguous"
        (var,) = sk.key["variables"]
        assert var["space_class"] == "homeont:Bathroom"
        assert var["property_change_direction"] == "decrease"

    def test_the_order_of_the_variables_does_not_matter(self):
        a = _var("relative_humidity", "homeont:RelativeHumidity", "homeont:LivingRoom", "decrease")
        b = _var("air_temperature", "homeont:AirTemperature", "homeont:LivingRoom", "decrease")
        one = _key(goal_specificity="ambiguous", goal_kind="achievement",
                   implied_environment_vars=[a, b])
        two = _key(goal_specificity="ambiguous", goal_kind="achievement",
                   implied_environment_vars=[b, a])
        assert one.key == two.key

    def test_the_direction_is_part_of_the_key(self):
        up = _var("relative_humidity", "homeont:RelativeHumidity", "homeont:Bathroom", "increase")
        down = _var("relative_humidity", "homeont:RelativeHumidity", "homeont:Bathroom", "decrease")
        assert (_key(goal_specificity="ambiguous", goal_kind="achievement",
                     implied_environment_vars=[up]).key
                != _key(goal_specificity="ambiguous", goal_kind="achievement",
                        implied_environment_vars=[down]).key)


def test_maintenance_goals_are_not_remembered():
    assert _key(**{**EXPLICIT, "goal_kind": "maintenance"}) is None
