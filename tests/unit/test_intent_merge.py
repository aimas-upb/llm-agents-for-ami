"""intent_merge: a timed request about one device stays one intent."""

from ami_agents.agents.user_assistant.models import AtomicIntent
from ami_agents.agents.user_assistant.utils.intent_merge import merge_coupled_intents


def _goal(text, *qualifiers):
    return AtomicIntent(text=text, type="GOAL_REQUEST", reason="r", qualifiers=list(qualifiers))


def test_the_finish_together_pieces_are_joined():
    merged = merge_coupled_intents([
        _goal("Start dryer 1 in the utility room at the right time and put it into Running state "
              "at Normal dryness level", "explicit", "achievement"),
        _goal("Time it so dishwasher 1 in the kitchen and dryer 1 in the utility room finish at "
              "the same time", "explicit", "temporal_dependency"),
        _goal("Make sure dryer 1 ends in Stopped state", "explicit"),
    ])
    assert len(merged) == 1
    assert merged[0].text == ("Start dryer 1 in the utility room at the right time and put it into "
                              "Running state at Normal dryness level. Time it so dishwasher 1 in the "
                              "kitchen and dryer 1 in the utility room finish at the same time. Make "
                              "sure dryer 1 ends in Stopped state")
    assert merged[0].qualifiers == ["explicit", "achievement", "temporal_dependency"]


def test_a_pause_when_the_other_finishes_joins_its_start():
    merged = merge_coupled_intents([
        _goal("Start dryer 1 in the utility room at Normal dryness level now"),
        _goal("Pause dryer 1 in the utility room when dryer 1 in the bathroom finishes"),
    ])
    assert len(merged) == 1 and "temporal_dependency" in merged[0].qualifiers


def test_untimed_intents_about_one_device_stay_apart():
    merged = merge_coupled_intents([_goal("turn on light 1 in the kitchen"),
                                    _goal("set light 1 in the kitchen to 50%")])
    assert len(merged) == 2


def test_timed_intents_about_different_devices_stay_apart():
    merged = merge_coupled_intents([
        _goal("Start washer 1 in the utility room now"),
        _goal("When dishwasher 1 in the kitchen finishes, turn on light 1 in the hallway"),
    ])
    assert len(merged) == 2


def test_device_numbers_tell_instances_apart():
    merged = merge_coupled_intents([_goal("Start dryer 1 now"),
                                    _goal("Pause dryer 2 when washer 1 finishes")])
    assert len(merged) == 2


def test_queries_are_never_merged():
    query = AtomicIntent(text="is dryer 1 running?", type="ENV_STATE_REQUEST", reason="q")
    merged = merge_coupled_intents([_goal("Start dryer 1 at the right time"), query])
    assert [i.type for i in merged] == ["GOAL_REQUEST", "ENV_STATE_REQUEST"]


def test_inputs_are_not_mutated():
    first = _goal("Start dryer 1 at the right time", "explicit")
    merge_coupled_intents([first, _goal("Make sure dryer 1 ends in Stopped state")])
    assert first.text == "Start dryer 1 at the right time" and first.qualifiers == ["explicit"]


def test_a_wait_joins_the_goal_that_follows_it():
    merged = merge_coupled_intents([
        _goal("Wait for washer 1 in the utility room to finish"),
        _goal("After 13 minutes start dryer 1 in the utility room in running state"),
    ])
    assert len(merged) == 1
    assert merged[0].text == ("Wait for washer 1 in the utility room to finish. After 13 minutes "
                              "start dryer 1 in the utility room in running state")


def test_a_relative_delay_continues_the_previous_goal():
    merged = merge_coupled_intents([
        _goal("Set refrigerator 1 in the kitchen to 5 degrees"),
        _goal("After 15 minutes bring it back to -18 degrees"),
    ])
    assert len(merged) == 1


def test_it_timed_by_an_event_continues_the_previous_goal():
    merged = merge_coupled_intents([
        _goal("Start washer 1 in the utility room now on Heavy Whites mode"),
        _goal("Pause it when dryer 1 in the utility room finishes"),
    ])
    assert len(merged) == 1


def test_chatter_before_a_goal_is_not_a_wait():
    merged = merge_coupled_intents([
        _goal("Quick heads up I need the wool sweater to rest a bit so it won't shrink"),
        _goal("Start dryer 1 in the utility room now"),
    ])
    assert len(merged) == 2
