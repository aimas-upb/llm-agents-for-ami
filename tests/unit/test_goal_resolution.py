"""Promoting incomplete goals and predicates when only one reading fits."""

from ami_agents.agents.user_assistant.utils.goal_resolution import (
    apply_goal_resolution,
    apply_predicate_resolution,
    default_quantifiers,
    goal_query,
    is_command_class,
    predicate_query,
)
from ami_agents.shared.models.goal_structure import GoalSpec, PredicateSpec


def _goal(**fields):
    base = {"goal_specificity": "incomplete", "intent_text": "start the dishwasher"}
    return GoalSpec.from_dict({**base, **fields})


def _found(*entries):
    return {"outcome": "found", "entries": list(entries)}


DISHWASHER = {"artifact_name": "kitchen_dishwasher_1", "artifact_type": "homeont:Dishwasher",
              "workspace_class": "homeont:Kitchen", "affordance_name": "operationalState"}


class TestCommandOrProperty:
    def test_commands_and_properties_are_told_apart_by_the_vocabulary(self):
        assert is_command_class("homeont:SetModeCommand")
        assert is_command_class("saref:SetAbsoluteLevelCommand")
        assert not is_command_class("homeont:OnOff")


class TestGoalQuery:
    def test_only_incomplete_goals_are_looked_up(self):
        assert goal_query(_goal(goal_specificity="explicit",
                                artifact_class="homeont:Dishwasher")) is None
        assert goal_query(_goal(goal_specificity="ambiguous")) is None

    def test_naming_neither_room_nor_device_is_forwarded_as_is(self):
        assert goal_query(_goal(affordance_class="homeont:SetOnOffCommand")) is None

    def test_payload_puts_the_affordance_in_the_right_slot(self):
        command = goal_query(_goal(artifact_class="homeont:Dishwasher",
                                   affordance_class="homeont:SetOnOffCommand"))
        assert command["command"] == {"class": "homeont:SetOnOffCommand"}
        prop = goal_query(_goal(location_class="homeont:Kitchen",
                                affordance_class="homeont:OnOff"))
        assert prop["device_property"] == {"class": "homeont:OnOff"}
        assert "command" not in prop


class TestGoalPromotion:
    def test_one_fit_fills_the_gaps_and_makes_it_explicit(self):
        goal = _goal(artifact_class="homeont:Dishwasher",
                     affordance_class="homeont:SetOperationalStateCommand")
        assert apply_goal_resolution(goal, _found(DISHWASHER))
        assert goal.goal_specificity == "explicit"
        assert (goal.location_class, goal.artifact_name, goal.affordance_name) == (
            "homeont:Kitchen", "kitchen_dishwasher_1", "operationalState")

    def test_several_devices_leave_it_incomplete(self):
        goal = _goal(artifact_class="homeont:OnOffLight",
                     affordance_class="homeont:SetOnOffCommand")
        other = dict(DISHWASHER, artifact_name="kitchen_on_off_light_2")
        assert not apply_goal_resolution(goal, _found(DISHWASHER, other))
        assert goal.goal_specificity == "incomplete"
        assert goal.artifact_name is None

    def test_one_device_but_several_affordances_leave_it_incomplete(self):
        goal = _goal(artifact_class="homeont:Dishwasher",
                     affordance_class="homeont:SetModeCommand")
        other = dict(DISHWASHER, affordance_name="mode")
        assert not apply_goal_resolution(goal, _found(DISHWASHER, other))

    def test_no_fit_changes_nothing(self):
        goal = _goal(artifact_class="homeont:Dishwasher")
        assert not apply_goal_resolution(goal, {"outcome": "none", "entries": []})
        assert not apply_goal_resolution(goal, {})
        assert goal.goal_specificity == "incomplete"


class TestPredicates:
    def _pred(self, **fields):
        base = {"predicate_specificity": "incomplete",
                "predicate_subject": "device_property",
                "predicate_text": "while the light is on",
                "property_class": "homeont:OnOff"}
        return PredicateSpec.from_dict({**base, **fields})

    def test_environment_predicates_are_not_looked_up(self):
        assert predicate_query(self._pred(predicate_subject="environment_property",
                                          location_class="homeont:Kitchen")) is None

    def test_no_subject_is_forwarded_as_is(self):
        assert predicate_query(self._pred()) is None

    def test_several_fits_quantify_instead_of_blocking(self):
        pred = self._pred(location_class="homeont:LivingRoom",
                          artifact_class="homeont:OnOffLight")
        lights = [dict(DISHWASHER, artifact_name=f"living_room_light_{i}",
                       affordance_name="onOff") for i in (1, 2)]
        assert not apply_predicate_resolution(pred, _found(*lights))
        assert pred.quantifier == "any"

    def test_an_explicit_all_is_kept(self):
        pred = self._pred(location_class="homeont:LivingRoom", quantifier="all")
        lights = [dict(DISHWASHER, artifact_name=f"l{i}", affordance_name="onOff")
                  for i in (1, 2)]
        apply_predicate_resolution(pred, _found(*lights))
        assert pred.quantifier == "all"

    def test_one_fit_promotes_and_drops_the_quantifier(self):
        pred = self._pred(location_class="homeont:Kitchen", quantifier="any")
        light = dict(DISHWASHER, artifact_name="kitchen_on_off_light_1",
                     artifact_type="homeont:OnOffLight", affordance_name="onOff")
        assert apply_predicate_resolution(pred, _found(light))
        assert pred.predicate_specificity == "explicit"
        assert pred.quantifier is None
        assert pred.property_name == "onOff"

    def test_unresolved_predicates_default_to_any(self):
        preds = {"P1": self._pred(), "P2": self._pred(predicate_specificity="explicit")}
        default_quantifiers(preds)
        assert preds["P1"].quantifier == "any"
        assert preds["P2"].quantifier is None
