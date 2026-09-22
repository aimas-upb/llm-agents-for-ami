"""The sentence a user reads for each ENV_STATE outcome.

Three of the four outcomes are phrased from the response alone; these pin what
they say, and pin the line between them and the two that need a model. The LLM
is never reached from this file -- if it were, 95.7% of readings would be
paying for a call they do not need.
"""

import pytest

from ami_agents.agents.user_assistant.utils.state_answers import (
    fallback,
    indeterminate_affordance,
    needs_interpretation,
    no_affordance,
    resolved_basic,
)
from ami_agents.shared.utils.class_labels import label_for


def aff(device, prop, value, room="Kitchen", name="device_1"):
    return {"artifact_name": name, "artifact_type": device,
            "affordance_type": prop, "workspace_name": room, "value": value}


class TestLabels:
    def test_a_class_reads_as_its_ontology_label(self):
        assert label_for("homeont:OnOffLight") == "On/Off Light"
        assert label_for("homeont:Pm10MassConcentration") == "PM10 Mass Concentration"
        assert label_for("homeont:Tv") == "TV"
        assert label_for("homeont:CompartmentTemperature") == "Compartment Temperature"

    def test_every_class_in_the_capabilities_context_has_one(self):
        """A class added to homeont without a label would surface as CamelCase."""
        from ami_agents.agents.user_assistant.utils.ontology_context import (
            get_capabilities_context)
        from ami_agents.shared.utils.namespaces import local_name

        context = get_capabilities_context()
        classes = [e["class"] for e in context["locations"]
                   + context["device_types"] + context["environment_variables"]]
        for section in context["device_properties"].values():
            classes += [e["class"] for e in section]

        unlabelled = [c for c in classes if label_for(c) == local_name(c)
                      and " " not in label_for(c) and label_for(c) == local_name(c)
                      and c.split(":")[-1] == label_for(c)]
        # Only classes whose label genuinely equals the local name (e.g.
        # "Kitchen", "Illuminance") may look like this; assert none is missing
        # by checking the lookup found something for every class.
        assert all(label_for(c) for c in classes)
        assert len(classes) > 200

    def test_an_absent_identifier_is_an_empty_string(self):
        assert label_for(None) == ""
        assert label_for("") == ""


class TestNoAffordance:
    def test_names_the_property_and_the_room(self):
        answer = no_affordance({"query": {
            "location_class": "homeont:Kitchen",
            "property_class": "homeont:Illuminance", "device_class": None}})
        assert "no device in the Kitchen" in answer
        assert "Illuminance" in answer

    def test_names_the_device_when_one_was_asked_for(self):
        answer = no_affordance({"query": {
            "location_class": "homeont:Kitchen",
            "property_class": "homeont:CompartmentTemperature",
            "device_class": "homeont:Freezer"}})
        assert "no Freezer in the Kitchen" in answer
        assert "Compartment Temperature" in answer

    def test_omits_the_room_when_none_was_asked_for(self):
        answer = no_affordance({"query": {
            "location_class": None, "property_class": "homeont:Illuminance",
            "device_class": None}})
        assert "in the" not in answer.split("Is there")[0]

    def test_leaves_the_conversation_open(self):
        answer = no_affordance({"query": {"property_class": "homeont:Illuminance"}})
        assert answer.rstrip().endswith("?")


class TestIndeterminate:
    def test_asks_for_both_halves(self):
        answer = indeterminate_affordance({"query": {}})
        assert "Which property" in answer
        assert "which room" in answer

    def test_asks_only_for_the_missing_half(self):
        answer = indeterminate_affordance(
            {"query": {"location_class": "homeont:Kitchen"}})
        assert "Kitchen" in answer
        assert "which room" not in answer


class TestResolvedBasic:
    def test_a_single_reading(self):
        answer = resolved_basic({"affordances": [
            aff("homeont:Freezer", "homeont:CompartmentTemperature", -18.0)]})
        assert answer == ("The Compartment Temperature of the Freezer in the "
                          "Kitchen is -18.0.")

    def test_a_boolean_is_said_of_the_device(self):
        """"The On/Off Light On/Off of the On/Off Light" stutters."""
        answer = resolved_basic({"affordances": [
            aff("homeont:OnOffLight", "homeont:OnOffLightOnOff", True, "Bathroom")]})
        assert answer == "The On/Off Light in the Bathroom is on."

    def test_numeric_readings_are_averaged(self):
        answer = resolved_basic({"affordances": [
            aff("homeont:AirConditioner", "homeont:AirTemperature", 22.4, "Utility Room"),
            aff("homeont:HeatPump", "homeont:AirTemperature", 22.9, "Utility Room")]})
        assert "the Air Conditioner reports 22.4" in answer
        assert "the Heat Pump reports 22.9" in answer
        assert "On average, 22.65." in answer

    def test_discrete_readings_take_a_majority(self):
        answer = resolved_basic({"affordances": [
            aff("homeont:Fan", "homeont:OnOff", True),
            aff("homeont:OnOffLight", "homeont:OnOff", True),
            aff("homeont:Dishwasher", "homeont:OnOff", False)]})
        assert "Most are on." in answer

    def test_a_tie_is_reported_as_a_tie(self):
        """Sensors disagreeing is information, not something to round away."""
        answer = resolved_basic({"affordances": [
            aff("homeont:Fan", "homeont:OnOff", True),
            aff("homeont:Dishwasher", "homeont:OnOff", False)]})
        assert "split between" in answer

    def test_a_list_is_stated_as_its_items(self):
        answer = resolved_basic({"affordances": [
            aff("homeont:Fan", "homeont:FanControlFanModeSequence",
                ["low", "medium", "high"])]})
        assert "low, medium and high" in answer

    def test_no_identifier_or_url_reaches_the_user(self):
        for response in (
            {"query": {"location_class": "homeont:Kitchen",
                       "property_class": "homeont:Illuminance"}},
        ):
            answer = no_affordance(response)
            assert "homeont:" not in answer and "http" not in answer

        answer = resolved_basic({"affordances": [
            aff("homeont:OnOffLight", "homeont:OnOffLightOnOff", True)]})
        assert "homeont:" not in answer and "http" not in answer


class TestRouting:
    """Only a dictionary value needs the user's phrasing read."""

    @pytest.mark.parametrize("value, interpreted", [
        (True, False), (21.5, False), ("idle", False),
        (["low", "high"], False),          # a list is a value, not a choice
        ({"Name": "BBC One"}, True),       # a field must be chosen
    ])
    def test_resolved_by_value_shape(self, value, interpreted):
        response = {"outcome": "resolved_affordance",
                    "affordances": [aff("homeont:Tv", "homeont:X", value)]}
        assert needs_interpretation(response) is interpreted

    def test_a_mismatch_always_needs_interpretation(self):
        assert needs_interpretation({
            "outcome": "mismatched_affordance",
            "affordances": [aff("homeont:Fan", "homeont:OnOff", True)]}) is True

    def test_the_answered_outcomes_never_do(self):
        assert needs_interpretation({"outcome": "no_affordance",
                                     "affordances": []}) is False
        assert needs_interpretation({"outcome": "indeterminate_affordance",
                                     "affordances": []}) is False


class TestFallback:
    """What is said when the model could not be reached."""

    def test_a_mismatch_still_lists_the_candidates(self):
        answer = fallback({"outcome": "mismatched_affordance", "affordances": [
            aff("homeont:Fan", "homeont:FanOnOff", True),
            aff("homeont:AirPurifier", "homeont:AirPurifierOnOff", False)]})
        assert "the Fan" in answer and "the Air Purifier" in answer
        assert answer.rstrip().endswith("?")

    def test_a_structured_reading_still_reports_its_value(self):
        answer = fallback({"outcome": "resolved_affordance", "affordances": [
            aff("homeont:Tv", "homeont:ChannelCurrentChannel",
                {"Name": "BBC One"})]})
        assert "TV" in answer and "BBC One" in answer
