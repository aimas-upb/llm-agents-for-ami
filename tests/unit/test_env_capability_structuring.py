"""Normalising the ENV_CAPABILITIES parser's output, without an LLM.

The parser answers with class identifiers and a performative; these pin what
reaches EnvExplorer: no sentinel strings, no branch roots, at most one of
device_property / environment_variable, and `saref:Command` only beside an
environment variable.
"""

from ami_agents.agents.user_assistant.behaviours.env_capability_structuring import (
    SLOTS,
    normalise_capability_slots,
)


def _norm(**parsed):
    return normalise_capability_slots(parsed, "the user's words")


class TestShape:
    def test_every_slot_is_present(self):
        result = _norm()
        for slot in SLOTS:
            assert slot in result
        assert result["text_intent"] == "the user's words"

    def test_sentinels_read_as_absent(self):
        result = _norm(location_class="NA", device_class="",
                       device_property={"class": "null"})
        assert result["location_class"] is None
        assert result["device_class"] is None
        assert result["device_property"] is None

    def test_nested_slots_keep_their_extra_key(self):
        result = _norm(
            device_property={"class": "homeont:FanControlFanMode",
                             "parent_class": "homeont:ActuatableDeviceProperty"},
            command={"class": "homeont:SetOnOffCommand",
                     "parent_class": "saref:OffCommand"})
        assert result["device_property"]["parent_class"] == "homeont:ActuatableDeviceProperty"
        assert result["command"] == {"class": "homeont:SetOnOffCommand",
                                     "parent_class": "saref:OffCommand"}

    def test_metadata_property_has_no_parent(self):
        result = _norm(device_property={"class": "schema:manufacturer",
                                        "parent_class": None})
        assert result["device_property"] == {"class": "schema:manufacturer"}


class TestPerformative:
    def test_query_if_is_kept(self):
        assert _norm(request_performative="query_if")["request_performative"] == "query_if"

    def test_missing_or_unknown_defaults_to_query(self):
        assert _norm()["request_performative"] == "query"
        assert _norm(request_performative="ask")["request_performative"] == "query"


class TestRoots:
    def test_property_roots_are_dropped(self):
        result = _norm(
            device_property={"class": "homeont:ActuatableDeviceProperty"})
        assert result["device_property"] is None

    def test_any_command_alone_is_dropped(self):
        result = _norm(command={"class": "saref:Command"})
        assert result["command"] is None

    def test_any_command_beside_a_variable_is_kept(self):
        """"Can anything cool the kitchen?" -- any action affecting it."""
        result = _norm(
            environment_variable={"class": "homeont:AirTemperature"},
            command={"class": "saref:Command"})
        assert result["command"] == {"class": "saref:Command"}

    def test_dropping_the_variable_also_drops_a_bare_command_root(self):
        """Both property slots filled: the device property wins, and the
        `saref:Command` that only made sense beside the variable goes too."""
        result = _norm(
            device_property={"class": "homeont:FanControlFanMode"},
            environment_variable={"class": "homeont:AirTemperature"},
            command={"class": "saref:Command"})
        assert result["device_property"]["class"] == "homeont:FanControlFanMode"
        assert result["environment_variable"] is None
        assert result["command"] is None
