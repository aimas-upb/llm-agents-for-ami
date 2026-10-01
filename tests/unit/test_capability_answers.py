"""Turning an ENV_CAPABILITY_QUERY response into the user's sentence.

Only a found `query` goes to a model; every other outcome is phrased here, and
none of them may show an identifier.
"""

from ami_agents.agents.user_assistant.utils import capability_answers as ca


def entry(device="homeont:AirPurifier", room="Kitchen", **extra):
    out = {"artifact_name": "kitchen_air_purifier_1", "artifact_type": device,
           "workspace_name": room, "artifact": "http://x/purifier#artifact"}
    out.update(extra)
    return out


def response(outcome, performative="query", entries=(), **query):
    return {"outcome": outcome, "request_performative": performative,
            "entries": list(entries), "query": query, "detail": ""}


class TestRouting:
    def test_only_a_found_query_needs_a_model(self):
        assert ca.needs_interpretation(response("found", "query"))
        assert not ca.needs_interpretation(response("found", "query_if"))
        for outcome in ("none", "device_lacks_capability", "indeterminate"):
            assert not ca.needs_interpretation(response(outcome))


class TestDeterministic:
    def test_yes_names_each_device_once(self):
        text = ca.phrase(response("found", "query_if", [
            entry(affordance_name="onOff"), entry(affordance_name="fanMode"),
            entry(device="homeont:OnOffLight")]))
        assert text == ("Yes: the Air Purifier in the Kitchen and "
                        "the On/Off Light in the Kitchen.")

    def test_none_with_a_named_device_says_it_is_absent(self):
        text = ca.phrase(response("none", "query_if",
                                  device_class="homeont:Refrigerator",
                                  location_class="homeont:Kitchen"))
        assert text == "No. There is no Refrigerator in the Kitchen."

    def test_none_names_what_was_asked(self):
        text = ca.phrase(response(
            "none", location_class="homeont:Kitchen",
            property_class="homeont:LevelControlBrightness"))
        assert text == "Nothing in the Kitchen provides Level Control Brightness."

    def test_a_device_that_lacks_the_capability(self):
        text = ca.phrase(response(
            "device_lacks_capability", "query_if", [entry()],
            device_class="homeont:AirPurifier",
            property_class="homeont:LevelControlBrightness"))
        assert text == ("No. The Air Purifier in the Kitchen does not provide "
                        "Level Control Brightness.")

    def test_an_effect_question_reads_as_changing_the_variable(self):
        text = ca.phrase(response(
            "none", location_class="homeont:Kitchen",
            environment_variable="homeont:AirTemperature",
            command_class="saref:Command"))
        assert text == "Nothing in the Kitchen can change the Air Temperature."

    def test_commands_are_named_by_their_saref_label(self):
        text = ca.phrase(response("none", command_class="saref:OnCommand"))
        assert "On command" in text
        assert "saref:" not in text

    def test_an_error_is_reported(self):
        text = ca.phrase({"error": "malformed_payload", "detail": "bad"})
        assert "bad" in text


class TestFallback:
    def test_lists_capabilities_with_option_meanings(self):
        text = ca.fallback(response("found", "query", [entry(
            affordance_kind="property",
            affordance_type="homeont:FanControlFanMode",
            permitted_values={"enum": [0, 1], "meaning": "0 = Off. 1 = Low."})]))
        assert text.startswith("The Air Purifier in the Kitchen: ")
        assert "0 = Off. 1 = Low." in text
        assert "homeont:" not in text


class TestPromptEntries:
    def test_no_identifiers_or_urls_reach_the_model(self):
        items = ca.entries_for_prompt([entry(
            affordance_kind="property", affordance_name="brightness",
            affordance_type="homeont:LevelControlBrightness",
            property_branch="actuatable",
            permitted_values={"minimum": 1, "maximum": 254})])
        rendered = str(items)
        assert "http" not in rendered
        assert "homeont:" not in rendered
        assert items[0]["nature"] == "changeable"
        assert items[0]["capability"] == "Level Control Brightness"
        assert items[0]["allowed_values"] == {"minimum": 1, "maximum": 254}

    def test_effects_carry_variable_and_direction(self):
        items = ca.entries_for_prompt([entry(
            affordance_kind="action", affordance_type="homeont:SetOnOffCommand",
            effect_on="homeont:Illuminance", effect_direction="increase")])
        assert items[0]["affects"] == "Illuminance"
        assert items[0]["direction"] == "increase"
