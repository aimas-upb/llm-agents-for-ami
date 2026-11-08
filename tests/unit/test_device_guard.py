"""device_guard: the device the user named is the one answered about."""

from ami_agents.agents.user_assistant.utils.device_guard import (
    device_mention,
    device_mentions,
    guard_extraction,
    guard_goal,
    title_matches,
)
from ami_agents.shared.models.goal_structure import GoalSpec


def test_mentions():
    m = device_mention("what brand is refrigerator 1 in the kitchen")
    assert (m.cls, m.number, m.room, m.text) == ("homeont:Refrigerator", "1", "homeont:Kitchen",
                                                 "refrigerator 1 in the kitchen")
    assert device_mention("is the bathroom light 1 on").cls == "homeont:OnOffLight"
    assert device_mention("is dimmer light 1 in the living room flickering").cls == "homeont:DimmableLight"
    assert device_mention("start the dehumidifier 2").cls == "homeont:Dehumidifier"
    assert device_mention("is the bathroom light on") is None          # described, not named
    assert device_mention("set fan 30%") is None                        # a setting, not a device
    assert device_mention("power on the dehumidifier 12 minutes after") is None
    assert device_mention("set the fan 2 to 30%").number == "2"
    assert device_mention("how warm is the kitchen") is None


def test_title_matching():
    m = device_mention("dryer 1 in the utility room")
    assert title_matches("utility_room_laundry_dryer_1", m)
    assert not title_matches("utility_room_laundry_washer_1", m)
    assert not title_matches("utility_room_laundry_dryer_2", m)
    light = device_mention("light 1 in the bathroom")
    assert title_matches("bathroom_on_off_light_1", light)
    assert not title_matches("bathroom_dimmable_light_1", light)
    hum = device_mention("humidifier 1")
    assert not title_matches("bathroom_dehumidifier_1", hum)


def test_another_device_chosen_is_cleared_and_the_question_marked_as_naming_one():
    extraction = {"text_intent": "what brand is refrigerator 1 in the kitchen",
                  "device_class": "homeont:Freezer", "location_class": "homeont:Kitchen",
                  "device_reference": "name", "artifact_name": "kitchen_freezer_1"}
    note = guard_extraction(extraction)
    assert extraction["artifact_name"] is None and extraction["device_reference"] == "name"
    assert extraction["device_class"] == "homeont:Refrigerator"
    assert "kitchen_freezer_1" in note


def test_a_described_lookup_takes_the_named_kind():
    extraction = {"text_intent": "is dryer 1 showing any error", "device_class": "homeont:LaundryWasher",
                  "location_class": None, "device_reference": "description", "artifact_name": None}
    guard_extraction(extraction)
    assert (extraction["device_reference"], extraction["device_class"]) == ("description", "homeont:LaundryDryer")


def test_the_right_device_is_left_alone():
    extraction = {"text_intent": "is dryer 1 in the utility room running", "device_class": "homeont:LaundryDryer",
                  "device_reference": "name", "artifact_name": "utility_room_laundry_dryer_1"}
    assert guard_extraction(dict(extraction)) is None
    assert guard_extraction({"text_intent": "how warm is the kitchen", "artifact_name": None}) is None


def test_a_goal_on_another_device_is_asked_about_instead():
    goal = GoalSpec.from_dict({"goal_specificity": "explicit", "goal_kind": "achievement",
                               "intent_text": "turn on dehumidifier 1 in the utility room",
                               "artifact_class": "homeont:LaundryDryer",
                               "artifact_name": "utility_room_laundry_dryer_1",
                               "affordance_name": "onOff"})
    note = guard_goal(goal)
    assert goal.artifact_name is None and goal.unmatched_device is None
    assert (goal.device_reference, goal.goal_specificity) == ("name", "incomplete")
    assert goal.artifact_class == "homeont:Dehumidifier" and note
    right = GoalSpec.from_dict({"intent_text": "turn on light 1 in the kitchen",
                                "artifact_name": "kitchen_on_off_light_1"})
    assert guard_goal(right) is None and right.artifact_name == "kitchen_on_off_light_1"


def test_a_goal_whose_words_hold_the_triggers_device_too():
    goal = GoalSpec.from_dict({"intent_text": "9 minutes after the dryer 1 in the bathroom finishes, "
                                              "turn on light 1 in the living room",
                               "artifact_name": "living_room_on_off_light_1"})
    assert guard_goal(goal) is None
    wrong = GoalSpec.from_dict({"intent_text": goal.intent_text, "artifact_name": "living_room_dimmable_light_1"})
    assert guard_goal(wrong) and wrong.artifact_name is None and wrong.device_reference == "name"
    assert [m.cls for m in device_mentions(goal.intent_text)] == ["homeont:LaundryDryer", "homeont:OnOffLight"]


def test_a_two_part_question_is_checked_against_both_devices():
    extraction = {"text_intent": "what is the product name of blinds 1 in the living room, and is light 1 on",
                  "device_reference": "name", "artifact_name": "living_room_on_off_light_1",
                  "device_class": "homeont:OnOffLight"}
    assert guard_extraction(dict(extraction)) is None
    extraction["artifact_name"] = None
    assert guard_extraction(dict(extraction)) is None
