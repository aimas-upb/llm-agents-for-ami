"""Which structuring prompt a goal gets, and how much context it needs."""

import pytest

from ami_agents.agents.user_assistant.utils.goal_routing import route_goal


@pytest.mark.parametrize("qualifiers, structure, properties, environment", [
    (["explicit", "achievement"], "simple", False, False),
    (["incomplete", "achievement"], "simple", False, False),
    (["ambiguous", "achievement"], "simple", False, True),
    (["explicit", "temporal_dependency", "achievement"], "dependency", True, True),
    (["explicit", "logical_dependency", "achievement"], "dependency", True, True),
    (["explicit", "maintenance"], "dependency", True, True),
    (["ambiguous", "maintenance"], "dependency", True, True),
])
def test_routes(qualifiers, structure, properties, environment):
    route = route_goal(qualifiers)
    assert (route.structure, route.with_properties, route.with_environment) == (
        structure, properties, environment)


@pytest.mark.parametrize("qualifiers", [
    [],                                         # none at all
    ["explicit"],                               # no kind
    ["achievement"],                            # no specificity
    ["explicit", "ambiguous", "achievement"],   # two specificities
    None,
])
def test_missing_or_malformed_qualifiers_get_the_full_superset(qualifiers):
    route = route_goal(qualifiers)
    assert route.structure == "dependency"
    assert route.with_properties and route.with_environment


def test_labels_are_case_and_space_insensitive():
    assert route_goal([" Explicit ", "ACHIEVEMENT"]).structure == "simple"
