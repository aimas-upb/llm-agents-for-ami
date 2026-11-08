"""tree_repair: a planner tree that tests the goal before acting."""

from ami_agents.agents.interaction_solver.utils.tree_repair import goal_condition_first, repair_tree
from ami_agents.bt_planning.serialization import validate_tree
from ami_agents.shared.models.goal_structure import GoalSpec

GOAL = GoalSpec.from_dict({
    "goal_specificity": "ambiguous", "goal_kind": "achievement",
    "intent_text": "the utility room is too warm",
    "implied_environment_vars": [{"property_name": "air_temperature",
                                  "property_class": "homeont:AirTemperature",
                                  "property_change_direction": "decrease"}]})

COND = {"type": "condition", "name": "Check current temperature",
        "property_url": "http://h/utility_room_temperature_sensor_1/properties/air_temperature",
        "affordance_class": "homeont:AirTemperature", "expected_value": 26.38, "operator": "<"}
ACT = {"type": "action", "name": "switch on", "action_url": "http://h/ac_1/a/onOff",
       "parameters": {"value": True}}
DEVICE_COND = {"type": "condition", "name": "ac on?", "property_url": "http://h/ac_1/properties/onOff",
               "expected_value": True, "operator": "=="}


def test_the_goal_test_moves_into_a_selector():
    tree = {"type": "sequence", "name": "Cool the Utility Room", "children": [COND, ACT, ACT]}
    out = goal_condition_first(tree, GOAL)
    assert out["type"] == "selector" and out["children"][0] is COND
    assert out["children"][1]["type"] == "sequence" and out["children"][1]["children"] == [ACT, ACT]
    assert validate_tree(out) == []


def test_a_single_sequence_step_is_kept_as_the_body():
    inner = {"type": "sequence", "name": "steps", "children": [ACT]}
    out = goal_condition_first({"type": "sequence", "children": [COND, inner]}, GOAL)
    assert out["children"][1] is inner


def test_a_device_condition_is_a_step_not_the_goal_test():
    tree = {"type": "sequence", "children": [DEVICE_COND, ACT]}
    assert goal_condition_first(tree, GOAL) is None


def test_a_condition_by_property_name_in_the_url_counts():
    cond = {**COND, "affordance_class": None}
    assert goal_condition_first({"type": "sequence", "children": [cond, ACT]}, GOAL) is not None


def test_other_shapes_are_untouched():
    for tree in ({"type": "selector", "children": [COND, ACT]},
                 {"type": "sequence", "children": [ACT, COND]},
                 {"type": "sequence", "children": [COND]}, ACT):
        assert goal_condition_first(tree, GOAL) is None
    same, notes = repair_tree(ACT, GOAL)
    assert same is ACT and notes == []


def test_repair_tree_notes_what_it_did():
    _, notes = repair_tree({"type": "sequence", "children": [COND, ACT]}, GOAL)
    assert notes == ['the plan tested "Check current temperature" before acting; it now acts '
                     'unless that already holds']
