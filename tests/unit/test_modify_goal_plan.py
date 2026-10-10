"""Modify goals: the read -> compute -> set recipe, from the TDs or from the model.
And "set" goals stating a percentage, mapped onto the input's range.

A hand-written home, closed under subclass inference as the InteractionSolver
closes its graph. Each action `saref:actsUpon` the property it writes:

    light      brightness   integer 1..254, no unit      -> property brightness
    ac         coolingSetpoint  integer, unit:DEG_C      -> property coolingSetpoint
    fan        fanSpeed     integer 0..100, unit:PERCENT -> property fanSpeed (PERCENT)
    lamp       level        integer, no actsUpon
"""

import pytest
from rdflib import Graph

from ami_agents.agents.interaction_solver.utils.explicit_goal_plan import plan_explicit_goal
from ami_agents.agents.interaction_solver.utils.modify_goal_plan import (
    ModifyRecipe,
    modify_recipe,
    modify_tree,
    plan_modify_goal,
    recipe_from_answer,
)
from ami_agents.shared.models.goal_structure import GoalSpec
from ami_agents.shared.utils.vocabulary import close_types

HOME = """
@prefix hmas: <https://purl.org/hmas/> .
@prefix homeont: <http://example.org/homeont/> .
@prefix saref: <https://saref.etsi.org/core/> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix js: <https://www.w3.org/2019/wot/json-schema#> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix unit: <http://qudt.org/vocab/unit/> .

<http://h/light#artifact> a hmas:Artifact, homeont:DimmableLight ; td:title "light" ;
    td:hasActionAffordance <http://h/light/actions/brightness> ;
    td:hasPropertyAffordance <http://h/light/properties/brightness> .
<http://h/light/actions/brightness> a td:ActionAffordance, saref:SetAbsoluteLevelCommand ;
    td:title "brightness" ; saref:actsUpon <http://h/light/properties/brightness> ;
    td:hasForm [ hctl:hasTarget <http://h/light/actions/brightness> ] ;
    td:hasInputSchema [ a js:IntegerSchema ; js:minimum 1 ; js:maximum 254 ] .
<http://h/light/properties/brightness> a td:PropertyAffordance ; td:title "brightness" ;
    td:hasForm [ hctl:hasTarget <http://h/light/properties/brightness> ] ;
    td:hasOutputSchema [ a js:IntegerSchema ] .

<http://h/ac#artifact> a hmas:Artifact, homeont:AirConditioner ; td:title "ac" ;
    td:hasActionAffordance <http://h/ac/actions/coolingSetpoint> ;
    td:hasPropertyAffordance <http://h/ac/properties/coolingSetpoint> .
<http://h/ac/actions/coolingSetpoint> a td:ActionAffordance, saref:SetAbsoluteLevelCommand ;
    td:title "coolingSetpoint" ; saref:actsUpon <http://h/ac/properties/coolingSetpoint> ;
    td:hasForm [ hctl:hasTarget <http://h/ac/actions/coolingSetpoint> ] ;
    td:hasInputSchema [ a js:IntegerSchema ; qudt:unit unit:DEG_C ] .
<http://h/ac/properties/coolingSetpoint> a td:PropertyAffordance ; td:title "coolingSetpoint" ;
    qudt:unit unit:DEG_C ;
    td:hasForm [ hctl:hasTarget <http://h/ac/properties/coolingSetpoint> ] .

<http://h/fan#artifact> a hmas:Artifact, homeont:Fan ; td:title "fan" ;
    td:hasActionAffordance <http://h/fan/actions/fanSpeed> ;
    td:hasPropertyAffordance <http://h/fan/properties/fanSpeed> .
<http://h/fan/actions/fanSpeed> a td:ActionAffordance, saref:SetAbsoluteLevelCommand ;
    td:title "fanSpeed" ; saref:actsUpon <http://h/fan/properties/fanSpeed> ;
    td:hasForm [ hctl:hasTarget <http://h/fan/actions/fanSpeed> ] ;
    td:hasInputSchema [ a js:IntegerSchema ; qudt:unit unit:PERCENT ] .
<http://h/fan/properties/fanSpeed> a td:PropertyAffordance ; td:title "fanSpeed" ;
    qudt:unit unit:PERCENT ;
    td:hasForm [ hctl:hasTarget <http://h/fan/properties/fanSpeed> ] ;
    td:hasOutputSchema [ a js:IntegerSchema ; js:minimum 0 ; js:maximum 100 ] .

<http://h/lamp#artifact> a hmas:Artifact, homeont:DimmableLight ; td:title "lamp" ;
    td:hasActionAffordance <http://h/lamp/actions/level> .
<http://h/lamp/actions/level> a td:ActionAffordance, saref:SetAbsoluteLevelCommand ;
    td:title "level" ;
    td:hasForm [ hctl:hasTarget <http://h/lamp/actions/level> ] ;
    td:hasInputSchema [ a js:IntegerSchema ] .
"""


@pytest.fixture(scope="module")
def graph():
    g = Graph().parse(data=HOME, format="turtle")
    close_types(g)
    return g


def _goal(device, action, amount, percentage=False, **fields):
    base = {"goal_specificity": "explicit", "goal_kind": "achievement",
            "goal_effect": "modify", "artifact_name": device, "affordance_name": action,
            "parameter_name": "NA", "target_value_determined": amount,
            "target_value_is_percentage": percentage}
    return GoalSpec.from_dict({**base, **fields})


class TestRecipe:
    def test_a_plain_amount_is_added(self, graph):
        recipe = modify_recipe(graph, _goal("ac", "coolingSetpoint", "-2"))
        assert (recipe.mode, recipe.amount) == ("add", -2.0)
        assert recipe.property_url == "http://h/ac/properties/coolingSetpoint"
        assert (recipe.minimum, recipe.maximum) == (None, None)

    def test_a_percentage_of_a_non_percent_property_scales(self, graph):
        recipe = modify_recipe(graph, _goal("light", "brightness", "-20", percentage=True))
        assert (recipe.mode, recipe.amount) == ("scale", -20.0)
        # The property states no range: the action input's applies.
        assert (recipe.minimum, recipe.maximum, recipe.integer) == (1, 254, True)

    def test_a_percentage_of_a_percent_property_is_added(self, graph):
        recipe = modify_recipe(graph, _goal("fan", "fanSpeed", "10", percentage=True))
        assert recipe.mode == "add"
        # The property's own range is preferred.
        assert (recipe.minimum, recipe.maximum) == (0, 100)

    @pytest.mark.parametrize("goal, why", [
        (("lamp", "level", "5"), "acts upon"),
        (("ac", "coolingSetpoint", "a bit"), "not a number"),
        (("ac", "heatingSetpoint", "2"), "actions match"),
    ])
    def test_what_cannot_be_read_off_the_tds_says_why(self, graph, goal, why):
        reason = modify_recipe(graph, _goal(*goal))
        assert isinstance(reason, str) and why in reason
        assert plan_modify_goal(graph, _goal(*goal)) is None


class TestTree:
    def test_read_compute_set_with_device_keys(self, graph):
        entry = plan_modify_goal(graph, _goal("light", "brightness", "-20", percentage=True))
        read, compute, action = entry["tree"]["children"]
        assert read == {"type": "read", "name": "light.brightness: read",
                        "property_url": "http://h/light/properties/brightness",
                        "output": "modify/light/brightness/current"}
        assert compute["op"] == "change_clamped"
        assert compute["inputs"] == ["modify/light/brightness/current"]
        assert compute["args"] == {"amount": -20.0, "mode": "scale", "integer": True,
                                   "min": 1, "max": 254}
        assert action["action_url"] == "http://h/light/actions/brightness"
        assert action["parameter_keys"] == {"value": "modify/light/brightness/new"}
        assert "-20% of the current value" in entry["explanation"]


    def test_a_dotted_device_title_makes_a_valid_key(self):
        recipe = ModifyRecipe(device="light.kitchen", action="brightness",
                              action_url="http://h/a", parameter="value",
                              property_url="http://h/p", amount=1, mode="add")
        read = modify_tree(recipe)["children"][0]
        assert read["output"] == "modify/light_kitchen/brightness/current"


class TestRecipeFromTheModel:
    ANSWER = {"action": "brightness", "parameter": "value", "property": "brightness",
              "mode": "scale", "amount": -20, "min": 1, "max": 254}

    def test_a_valid_answer_is_built_with_urls_from_the_graph(self, graph):
        recipe = recipe_from_answer(graph, _goal("light", "brightness", "-20"), self.ANSWER)
        assert isinstance(recipe, ModifyRecipe)
        assert recipe.action_url == "http://h/light/actions/brightness"
        assert recipe.property_url == "http://h/light/properties/brightness"
        assert modify_tree(recipe)["children"][2]["parameter_keys"] == {
            "value": "modify/light/brightness/new"}

    @pytest.mark.parametrize("change, why", [
        ({"action": "dim"}, "no action"),
        ({"property": "level"}, "no property"),
        ({"mode": "multiply"}, "neither add nor scale"),
        ({"amount": "lots"}, "not a number"),
    ])
    def test_an_answer_naming_what_is_not_there_is_refused(self, graph, change, why):
        reason = recipe_from_answer(graph, _goal("light", "brightness", "-20"),
                                    {**self.ANSWER, **change})
        assert isinstance(reason, str) and why in reason


class TestPercentageSet:
    def _set(self, device, action, value, percentage=True):
        return _goal(device, action, value, percentage, goal_effect="set")

    def test_a_percent_input_takes_the_percentage_as_it_is(self, graph):
        entry = plan_explicit_goal(graph, self._set("fan", "fanSpeed", "40"))
        assert entry["tree"]["parameters"] == {"value": 40}

    def test_a_bounded_input_takes_the_point_that_far_along_its_range(self, graph):
        # 40% of [1, 254]: 1 + 0.4 * 253 = 102.2 -> 102 (an integer input)
        entry = plan_explicit_goal(graph, self._set("light", "brightness", "40"))
        assert entry["tree"]["parameters"] == {"value": 102}
        assert plan_explicit_goal(graph, self._set("light", "brightness", "100")
                                  )["tree"]["parameters"] == {"value": 254}

    def test_an_input_with_neither_unit_nor_range_goes_to_the_small_model(self, graph):
        assert plan_explicit_goal(graph, self._set("lamp", "level", "40")) is None

    def test_a_value_that_is_not_a_percentage_is_set_as_it_is(self, graph):
        entry = plan_explicit_goal(graph, self._set("light", "brightness", "40", False))
        assert entry["tree"]["parameters"] == {"value": 40}
