"""Modify goals: the read -> compute -> set recipe, from the TDs or from the model.
And "set" goals stating a percentage, mapped onto the input's range. And values on
an ordered scale (`homeont:orderedValues`), changed by steps, with a question when
the current value is not on the scale.

A hand-written home, closed under subclass inference as the InteractionSolver
closes its graph. Each action `saref:actsUpon` the property it writes:

    light      brightness   integer 1..254, no unit      -> property brightness
    ac         coolingSetpoint  integer, unit:DEG_C      -> property coolingSetpoint
    fan        fanSpeed     integer 0..100, unit:PERCENT -> property fanSpeed (PERCENT)
    lamp       level        integer, no actsUpon

and, in ORDERED, two values on a scale:

    vent       speed        string enum auto, low, medium, high; order low < medium < high
    mfan       fanMode      integer enum 0..6 (Matter); order 0 < 1 < 2 < 3
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rdflib import Graph

from ami_agents.agents.interaction_solver.utils.explicit_goal_plan import plan_explicit_goal
from ami_agents.agents.interaction_solver.behaviours.explicit_goal_planning import (
    ExplicitGoalPlanningBehaviour,
)
from ami_agents.agents.interaction_solver.utils.modify_goal_plan import (
    ModifyRecipe,
    describe,
    modify_recipe,
    modify_tree,
    off_scale_question,
    plan_modify_goal,
    recipe_from_answer,
)
from ami_agents.agents.interaction_solver.utils.specific_capability_context import json_schema
from ami_agents.bt_planning.nodes.affordance_nodes import PropertyValue
from ami_agents.shared.models.goal_structure import GoalSpec
from ami_agents.shared.utils.vocabulary import close_types
from ami_agents.shared.utils.namespaces import (  # noqa: E402
    HCTL,
    HMAS,
    HOMEONT,
    JS,
    QUDT,
    SAREF,
    TD,
    UNIT,
)

HOME = f"""
@prefix hmas: <{HMAS}> .
@prefix homeont: <{HOMEONT}> .
@prefix saref: <{SAREF}> .
@prefix td: <{TD}> .
@prefix hctl: <{HCTL}> .
@prefix js: <{JS}> .
@prefix qudt: <{QUDT}> .
@prefix unit: <{UNIT}> .

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


ORDERED = f"""
@prefix hmas: <{HMAS}> .
@prefix homeont: <{HOMEONT}> .
@prefix saref: <{SAREF}> .
@prefix td: <{TD}> .
@prefix hctl: <{HCTL}> .
@prefix js: <{JS}> .

<http://h/vent#artifact> a hmas:Artifact, homeont:Fan ; td:title "vent" ;
    td:hasActionAffordance <http://h/vent/actions/speed> ;
    td:hasPropertyAffordance <http://h/vent/properties/speed> .
<http://h/vent/actions/speed> a td:ActionAffordance, homeont:SetModeCommand ;
    td:title "speed" ; saref:actsUpon <http://h/vent/properties/speed> ;
    td:hasForm [ hctl:hasTarget <http://h/vent/actions/speed> ] ;
    td:hasInputSchema [ a js:StringSchema ; js:enum "auto", "low", "medium", "high" ;
                        homeont:orderedValues ( "low" "medium" "high" ) ] .
<http://h/vent/properties/speed> a td:PropertyAffordance ; td:title "speed" ;
    td:hasForm [ hctl:hasTarget <http://h/vent/properties/speed> ] ;
    td:hasOutputSchema [ a js:StringSchema ; js:enum "auto", "low", "medium", "high" ;
                         homeont:orderedValues ( "low" "medium" "high" ) ] .

<http://h/mfan#artifact> a hmas:Artifact, homeont:Fan ; td:title "mfan" ;
    td:hasActionAffordance <http://h/mfan/actions/fanMode> ;
    td:hasPropertyAffordance <http://h/mfan/properties/fanMode> .
<http://h/mfan/actions/fanMode> a td:ActionAffordance, homeont:SetModeCommand ;
    td:title "fanMode" ; saref:actsUpon <http://h/mfan/properties/fanMode> ;
    td:hasForm [ hctl:hasTarget <http://h/mfan/actions/fanMode> ] ;
    td:hasInputSchema [ a js:IntegerSchema ; js:enum 0, 1, 2, 3, 4, 5, 6 ;
                        homeont:orderedValues ( 0 1 2 3 ) ] .
<http://h/mfan/properties/fanMode> a td:PropertyAffordance ; td:title "fanMode" ;
    td:hasForm [ hctl:hasTarget <http://h/mfan/properties/fanMode> ] .
"""


@pytest.fixture(scope="module")
def graph():
    g = Graph().parse(data=HOME, format="turtle")
    close_types(g)
    return g


@pytest.fixture(scope="module")
def ordered():
    """The two devices whose value is on an ordered scale."""
    g = Graph().parse(data=ORDERED, format="turtle")
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
        ({"amount": "lots"}, "not a number"),
    ])
    def test_an_answer_naming_what_is_not_there_is_refused(self, graph, change, why):
        reason = recipe_from_answer(graph, _goal("light", "brightness", "-20"),
                                    {**self.ANSWER, **change})
        assert isinstance(reason, str) and why in reason

    def test_a_percentage_of_a_percent_property_is_added_whatever_the_model_says(self, graph):
        """"Fan speed up by 10%" on a property in unit:PERCENT is 10 points, even when
        the model answers "scale" -- the mode is the unit's, as on the deterministic path."""
        answer = {"action": "fanSpeed", "parameter": "value", "property": "fanSpeed",
                  "mode": "scale", "amount": 10}
        recipe = recipe_from_answer(graph, _goal("fan", "fanSpeed", "10", percentage=True),
                                    answer)
        assert recipe.mode == "add"

    def test_a_percentage_of_another_property_scales_whatever_the_model_says(self, graph):
        """"Brightness down by 20%" on a property with no percent unit scales, even when
        the model answers "add"."""
        recipe = recipe_from_answer(graph, _goal("light", "brightness", "-20", percentage=True),
                                    {**self.ANSWER, "mode": "add"})
        assert recipe.mode == "scale"


class TestOrderedScale:
    """A value on an ordered scale ("fan speed up by 1 level") is changed by steps
    along the TD's `homeont:orderedValues`; off the scale, the user is asked."""

    def _speed(self, amount, percentage=False):
        """The vent's speed changed by ``amount`` levels."""
        return _goal("vent", "speed", amount, percentage)

    def test_the_schema_reader_keeps_the_lists_order(self, ordered):
        """`json_schema` gives `order` as listed (the `enum` is a set, sorted)."""
        schemas = [json_schema(ordered, s) for s in ordered.objects(None, TD.hasInputSchema)]
        orders = {str(s["order"]) for s in schemas}
        assert orders == {str([0, 1, 2, 3]), str(["low", "medium", "high"])}
        vent = next(s for s in schemas if "auto" in s["enum"])
        assert vent["enum"] == ["auto", "high", "low", "medium"]

    def test_a_level_change_steps_along_the_scale(self, ordered):
        """+1 on the vent: an ordered recipe, its tree steps with `step_in_order`."""
        # 1. The recipe: the scale from the TD, one step up.
        recipe = modify_recipe(ordered, self._speed("1"))
        assert recipe.order == ["low", "medium", "high"] and recipe.amount == 1
        # 2. The tree reads, steps, sets.
        read, compute, action = modify_tree(recipe)["children"]
        assert read["property_url"] == "http://h/vent/properties/speed"
        assert (compute["op"], compute["args"]) == (
            "step_in_order", {"order": ["low", "medium", "high"], "steps": 1})
        assert action["action_url"] == "http://h/vent/actions/speed"
        assert describe(recipe) == ("Read vent.speed's value, move it 1 step up along "
                                    "low -> medium -> high (stopping at the ends), set it.")

    @pytest.mark.parametrize("amount, percentage, why", [
        ("1.5", False, "whole number of steps"),
        ("10", True, "percentage has no meaning"),
    ])
    def test_a_change_that_is_not_whole_steps_is_refused(self, ordered, amount, percentage,
                                                         why):
        """Half a level, or a percentage of a list, cannot be planned from the TD."""
        reason = modify_recipe(ordered, self._speed(amount, percentage))
        assert isinstance(reason, str) and why in reason

    def test_an_ordered_integer_enum_is_stepped_not_added(self, ordered):
        """A Matter FanMode is an integer, but "down 2 levels" follows its scale: the
        order wins over arithmetic (5, Auto, is off the scale, not a level)."""
        recipe = modify_recipe(ordered, _goal("mfan", "fanMode", "-2"))
        assert recipe.order == [0, 1, 2, 3] and recipe.amount == -2
        assert modify_tree(recipe)["children"][1]["op"] == "step_in_order"

    def test_the_models_recipe_takes_the_scale_from_the_td(self, ordered):
        """The model names the action and the steps; the order is the TD's."""
        answer = {"action": "speed", "parameter": "value", "property": "speed",
                  "amount": -2, "min": None, "max": None}
        recipe = recipe_from_answer(ordered, self._speed("-2"), answer)
        assert recipe.order == ["low", "medium", "high"] and recipe.amount == -2

    def test_off_the_scale_the_user_is_asked(self, ordered):
        """`auto` has no position: the question states it and the choices. On the
        scale, or unread, there is nothing to ask; a read value is unwrapped."""
        recipe = modify_recipe(ordered, self._speed("1"))
        assert off_scale_question(recipe, "auto") == (
            "The vent's speed is auto, which is not on its scale low -> medium -> high. "
            "Which speed do you want: low, medium or high?")
        assert off_scale_question(recipe, "medium") is None
        assert off_scale_question(recipe, None) is None
        assert off_scale_question(recipe, PropertyValue(success=True, value="auto"))


class TestPlanningAsksOffTheScale:
    """`ExplicitGoalPlanningBehaviour` reads an ordered value before planning: off the
    scale, the entry is a question; on it, or unread, the tree."""

    MODULE = "ami_agents.agents.interaction_solver.behaviours.explicit_goal_planning"
    READ = "ami_agents.agents.interaction_solver.utils.property_read.read_property_value"

    async def _plan(self, graph, current):
        """Plan "vent speed up by 1" with the property read answering ``current``."""
        behaviour = ExplicitGoalPlanningBehaviour([_goal("vent", "speed", "1")])
        behaviour.agent = MagicMock()
        with patch(f"{self.MODULE}.fetch_environment_graph", AsyncMock(return_value=graph)), \
                patch(self.READ, AsyncMock(return_value=current)) as read:
            await behaviour.run()
        (entry,) = behaviour.results
        return entry, read

    @pytest.mark.asyncio
    async def test_off_the_scale_is_a_question(self, ordered):
        """`auto`: a `requires_clarification` entry, read at the property's URL."""
        entry, read = await self._plan(ordered, "auto")
        assert entry["requires_clarification"] and entry["tree"] is None
        assert "is auto, which is not on its scale" in entry["explanation"]
        read.assert_awaited_once_with("http://h/vent/properties/speed")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("current", ["low", None])
    async def test_on_the_scale_or_unread_is_the_tree(self, ordered, current):
        """A value on the scale, or a failed read (unknown): the stepping tree."""
        entry, _ = await self._plan(ordered, current)
        assert not entry.get("requires_clarification")
        assert entry["tree"]["children"][1]["op"] == "step_in_order"


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
