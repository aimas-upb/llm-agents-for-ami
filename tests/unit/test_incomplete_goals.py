"""incomplete_case: which kind of incomplete goal, by the slots it leaves empty."""

from ami_agents.shared.models.goal_structure import GoalSpec
from ami_agents.shared.utils.incomplete_goals import (
    NO_ACTION,
    NO_CLASS,
    NO_LOCATION,
    NO_NAME,
    UNNAMED,
    VALUE_ONLY,
    incomplete_case,
)

FULL = {"goal_specificity": "incomplete", "location_class": "homeont:Kitchen",
        "artifact_class": "homeont:OnOffLight", "artifact_name": None,
        "affordance_class": "homeont:SetOnOffCommand", "affordance_name": "onOff",
        "target_value_text": "on"}


def _case(**fields):
    return incomplete_case(GoalSpec.from_dict({**FULL, **fields}))


def test_only_incomplete_goals_have_a_case():
    assert _case(goal_specificity="explicit") is None
    assert _case(goal_specificity="ambiguous") is None


def test_each_case():
    assert _case(location_class=None, artifact_class=None) == UNNAMED
    assert _case(affordance_class=None, affordance_name=None,
                 target_value_text=None) == NO_ACTION
    assert _case(affordance_class=None, affordance_name=None) == VALUE_ONLY
    assert _case(location_class=None) == NO_LOCATION
    assert _case(artifact_class=None) == NO_CLASS
    assert _case() == NO_NAME


def test_precedence():
    # Naming nothing outranks a missing action.
    assert _case(location_class=None, artifact_class=None,
                 affordance_class=None, affordance_name=None) == UNNAMED
    # A missing action outranks a missing room.
    assert _case(location_class=None, affordance_class=None,
                 affordance_name=None) == VALUE_ONLY


def test_either_half_of_the_action_is_enough():
    assert _case(affordance_class=None) == NO_NAME
    assert _case(affordance_name=None) == NO_NAME


def test_a_goal_naming_everything_is_found_by_its_names():
    assert _case(artifact_name="kitchen_light_1") == NO_NAME
    assert _case(location_class=None, artifact_class=None,
                 artifact_name="kitchen_light_1") == NO_NAME
