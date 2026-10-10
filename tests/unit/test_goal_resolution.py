"""Promoting incomplete goals and predicates when only one reading fits."""

from ami_agents.agents.user_assistant.utils.goal_resolution import (
    apply_goal_resolution,
    apply_predicate_resolution,
    clarification_question,
    default_quantifiers,
    goal_query,
    predicate_query,
    structure_clarification,
)
from ami_agents.shared.models.goal_structure import GoalSpec, GoalStructure, PredicateSpec


def _goal(**fields):
    base = {"goal_specificity": "incomplete", "intent_text": "start the dishwasher"}
    return GoalSpec.from_dict({**base, **fields})


def _found(*entries):
    return {"outcome": "found", "entries": list(entries)}


DISHWASHER = {"artifact_name": "kitchen_dishwasher_1", "artifact_type": "homeont:Dishwasher",
              "workspace_class": "homeont:Kitchen", "affordance_name": "operationalState"}

ONOFF = "homeont:SetOnOffCommand"


def _light(name, room):
    return {"artifact_name": name, "artifact_type": "homeont:OnOffLight",
            "workspace_class": room, "affordance_name": "onOff"}


class TestGoalQuery:
    def test_only_incomplete_goals_with_an_action_are_looked_up(self):
        assert goal_query(_goal(goal_specificity="explicit",
                                artifact_class="homeont:Dishwasher",
                                affordance_class=ONOFF)) is None
        assert goal_query(_goal(goal_specificity="ambiguous")) is None
        # No action: nothing could be promoted.
        assert goal_query(_goal(artifact_class="homeont:Dishwasher")) is None
        assert goal_query(_goal(artifact_class="homeont:Light",
                                target_value_text="blue")) is None

    def test_naming_neither_room_nor_device_is_not_looked_up(self):
        assert goal_query(_goal(affordance_class=ONOFF)) is None

    def test_the_affordance_class_is_looked_up_as_a_command(self):
        payload = goal_query(_goal(artifact_class="homeont:Dishwasher",
                                   affordance_class=ONOFF))
        assert payload["command"] == {"class": ONOFF}
        assert "device_property" not in payload


class TestClarification:
    def test_an_incomplete_goal_naming_nothing_needs_the_user(self):
        question = clarification_question(_goal(intent_text="turn it off",
                                                affordance_class=ONOFF))
        assert question.startswith("Which device")

    def test_a_device_with_no_action_and_no_value_needs_the_user(self):
        question = clarification_question(_goal(artifact_class="homeont:Fan",
                                                location_class="homeont:LivingRoom"))
        assert question == "What should I do with the fan in the living room?"

    def test_value_words_without_an_action_go_on_to_planning(self):
        assert clarification_question(_goal(artifact_class="homeont:OnOffLight",
                                             target_value_text="blue")) is None

    def test_naming_a_device_and_an_action_is_enough(self):
        assert clarification_question(_goal(location_class="homeont:Kitchen",
                                             affordance_class=ONOFF)) is None
        assert clarification_question(_goal(artifact_name="kitchen_dishwasher_1",
                                             affordance_name="onOff")) is None

    def test_explicit_and_ambiguous_goals_never_do(self):
        assert clarification_question(_goal(goal_specificity="explicit")) is None
        assert clarification_question(_goal(goal_specificity="ambiguous")) is None

    def test_one_unclear_goal_holds_back_its_structure(self):
        # "that" points back to an earlier message, not to the washer: the
        # segmenter can only resolve a reference from the utterance itself.
        structure = GoalStructure.from_dict(
            {"goals": {"G1": {"goal_specificity": "explicit",
                              "artifact_name": "washer_1"},
                       "G2": {"goal_specificity": "incomplete"}}},
            intent_text="start washer 1, and when it is done turn that back on",
            structure="dependency")
        assert structure_clarification(structure).startswith("Which device")


class TestGoalPromotion:
    def test_one_fit_fills_the_gaps_and_makes_it_explicit(self):
        goal = _goal(artifact_class="homeont:Dishwasher",
                     affordance_class="homeont:SetOperationalStateCommand")
        assert apply_goal_resolution(goal, _found(DISHWASHER))
        assert goal.goal_specificity == "explicit"
        assert (goal.location_class, goal.artifact_name, goal.affordance_name) == (
            "homeont:Kitchen", "kitchen_dishwasher_1", "operationalState")

    def test_promotion_drops_the_quantifier(self):
        goal = _goal(artifact_class="homeont:Dishwasher",
                     affordance_class="homeont:SetOperationalStateCommand",
                     quantifier="one")
        assert apply_goal_resolution(goal, _found(DISHWASHER))
        assert goal.quantifier is None

    def test_several_devices_leave_it_incomplete_with_its_quantifier(self):
        goal = _goal(location_class="homeont:Kitchen", artifact_class="homeont:OnOffLight",
                     affordance_class=ONOFF, quantifier="all")
        assert not apply_goal_resolution(goal, _found(
            _light("kitchen_light_1", "homeont:Kitchen"),
            _light("kitchen_light_2", "homeont:Kitchen")))
        assert goal.goal_specificity == "incomplete"
        assert goal.artifact_name is None and goal.quantifier == "all"

    def test_several_devices_do_not_invent_a_quantifier(self):
        goal = _goal(location_class="homeont:Kitchen", artifact_class="homeont:OnOffLight",
                     affordance_class=ONOFF)
        apply_goal_resolution(goal, _found(_light("a", "homeont:Kitchen"),
                                           _light("b", "homeont:Kitchen")))
        assert goal.quantifier is None

    def test_one_device_but_several_affordances_leave_it_incomplete(self):
        goal = _goal(artifact_class="homeont:Dishwasher",
                     affordance_class="homeont:SetModeCommand")
        other = dict(DISHWASHER, affordance_name="mode")
        assert not apply_goal_resolution(goal, _found(DISHWASHER, other))

    def test_no_fit_changes_nothing(self):
        goal = _goal(artifact_class="homeont:Dishwasher", affordance_class=ONOFF)
        assert not apply_goal_resolution(goal, {"outcome": "none", "entries": []})
        assert not apply_goal_resolution(goal, {})
        assert goal.goal_specificity == "incomplete"


class TestMissingRoom:
    """A goal naming its device class and action but not its room."""

    def _fans(self, quantifier=None):
        return _goal(intent_text="turn on the fan", artifact_class="homeont:Fan",
                     affordance_class=ONOFF, quantifier=quantifier)

    def _fan(self, name, room):
        return {"artifact_name": name, "artifact_type": "homeont:Fan",
                "workspace_class": room, "affordance_name": "onOff"}

    def test_one_room_class_fills_the_room(self):
        goal = self._fans("one")
        apply_goal_resolution(goal, _found(self._fan("a", "homeont:Bedroom"),
                                           self._fan("b", "homeont:Bedroom")))
        assert goal.location_class == "homeont:Bedroom"
        assert clarification_question(goal) is None

    def test_several_rooms_and_one_device_meant_ask_which_room(self):
        goal = self._fans("one")
        apply_goal_resolution(goal, _found(self._fan("a", "homeont:Bedroom"),
                                           self._fan("b", "homeont:LivingRoom")))
        assert goal.location_class is None
        assert clarification_question(goal) == (
            "Which room do you mean: the bedroom or the living room?")

    def test_several_rooms_and_all_meant_go_on_to_planning(self):
        goal = self._fans("all")
        apply_goal_resolution(goal, _found(self._fan("a", "homeont:Bedroom"),
                                           self._fan("b", "homeont:LivingRoom")))
        assert goal.location_class is None
        assert clarification_question(goal) is None

    def test_one_device_is_promoted(self):
        goal = self._fans("one")
        assert apply_goal_resolution(goal, _found(self._fan("a", "homeont:Bedroom")))
        assert goal.goal_specificity == "explicit"


class TestPredicates:
    def _pred(self, **fields):
        base = {"predicate_specificity": "incomplete",
                "predicate_subject": "device_property",
                "predicate_text": "while the light is on",
                "property_class": "homeont:OnOff"}
        return PredicateSpec.from_dict({**base, **fields})

    def _env_pred(self, space=None, **fields):
        var = {"property_name": "relative_humidity",
               "property_class": "homeont:RelativeHumidity"}
        if space:
            var["property_sensed_space"] = space
        base = {"predicate_specificity": "incomplete",
                "predicate_subject": "environment_property",
                "predicate_text": "while the humidity is above 60%",
                "environment_var": var, "comparison": ">", "target_value_text": "60%", "target_value_determined": "60"}
        return PredicateSpec.from_dict({**base, **fields})

    def test_an_environment_predicate_without_a_room_is_looked_up_by_class(self):
        assert predicate_query(self._env_pred()) == {
            "text_intent": "while the humidity is above 60%",
            "environment_variable": {"class": "homeont:RelativeHumidity"}}

    def test_an_environment_predicate_with_its_room_needs_no_lookup(self):
        pred = self._env_pred(space={"space_class": "homeont:Bathroom",
                                     "space_name": "Bathroom"})
        assert predicate_query(pred) is None

    def test_an_environment_predicate_without_a_class_cannot_be_looked_up(self):
        pred = self._env_pred()
        pred.environment_var.property_class = None
        assert predicate_query(pred) is None

    def test_one_room_reading_the_variable_fills_the_space(self):
        pred = self._env_pred()
        sensors = [{"artifact_name": "bathroom_humidity_sensor_1",
                    "workspace_class": "homeont:Bathroom", "workspace_name": "Bathroom",
                    "affordance_name": "humidity"},
                   {"artifact_name": "bathroom_dehumidifier_1",
                    "workspace_class": "homeont:Bathroom", "workspace_name": "Bathroom",
                    "affordance_name": "humidity"}]
        assert apply_predicate_resolution(pred, _found(*sensors))
        assert (pred.environment_var.space_class, pred.environment_var.space_name) == (
            "homeont:Bathroom", "Bathroom")
        assert pred.predicate_specificity == "explicit"

    def test_a_room_but_no_threshold_stays_incomplete(self):
        pred = self._env_pred(target_value_text=None)
        reading = {"artifact_name": "s", "workspace_class": "homeont:Bathroom",
                   "workspace_name": "Bathroom", "affordance_name": "humidity"}
        assert not apply_predicate_resolution(pred, _found(reading))
        assert pred.environment_var.space_class == "homeont:Bathroom"
        assert pred.predicate_specificity == "incomplete"

    def test_several_rooms_quantify(self):
        pred = self._env_pred()
        readings = [{"artifact_name": f"s{i}", "workspace_class": cls,
                     "workspace_name": name, "affordance_name": "humidity"}
                    for i, (cls, name) in enumerate([("homeont:Bathroom", "Bathroom"),
                                                     ("homeont:Kitchen", "Kitchen")])]
        assert not apply_predicate_resolution(pred, _found(*readings))
        assert pred.quantifier == "any"
        assert pred.environment_var.space_class is None

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
